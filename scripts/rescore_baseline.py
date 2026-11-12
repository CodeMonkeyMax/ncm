"""Re-score a saved baseline shard with the unchanged upstream scorer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from baseline import ROOT, UPSTREAM, digest, git, write_json


def validate_generation_checkpoint(source: Path, manifest: dict) -> None:
    checkpoint = json.loads((source / "generation-checkpoint.json").read_text(encoding="utf-8"))
    expected = {"generation_shard.pkl", "shard_0.pkl", "candidates.jsonl", "contexts.jsonl",
                "references.jsonl", "generation-timing.json", "run-spec.json"}
    hashes = checkpoint["sha256"]
    if set(hashes) != expected:
        raise ValueError("generation checkpoint file list mismatch")
    for name in sorted(expected):
        if digest(source / name) != hashes[name]:
            raise ValueError(f"generation checkpoint hash mismatch: {name}")
    spec = json.loads((source / "run-spec.json").read_text(encoding="utf-8"))
    for key in ("run_id", "task", "upstream_commit", "config_sha256", "reference_model",
                "input_ids", "batch_size", "generation", "adapter_sha256", "profiler_sha256"):
        if spec[key] != manifest[key]:
            raise ValueError(f"generation checkpoint run specification mismatch: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    for value in (args.run_id, args.output_id):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
            parser.error("IDs may contain only letters, digits, hyphens and underscores")
    if args.gpu < 0:
        parser.error("gpu must be nonnegative")
    source = ROOT / "runs" / args.run_id
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] == "succeeded":
        expected_shard_hash = manifest["outputs"]["shard_0.pkl"]
    elif manifest["status"] in ("failed", "running"):
        validate_generation_checkpoint(source, manifest)
        if digest(ROOT / "scripts" / "profile_upstream.py") != manifest["profiler_sha256"]:
            raise ValueError("profiler hash differs from the original generation run")
        expected_shard_hash = json.loads(
            (source / "generation-checkpoint.json").read_text(encoding="utf-8")
        )["sha256"]["shard_0.pkl"]
    else:
        raise ValueError("source run has not finished")
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != manifest["upstream_commit"] or git("status", "--porcelain", cwd=UPSTREAM):
        raise RuntimeError("upstream checkout does not match the recorded clean commit")
    shard = source / "shard_0.pkl"
    if digest(shard) != expected_shard_hash:
        raise ValueError("saved shard hash mismatch")
    config = UPSTREAM / "config" / "eval" / f"{manifest['task']}.yaml"
    if digest(config) != manifest["config_sha256"]:
        raise ValueError("upstream config hash mismatch")
    if manifest["task"] == "xsum":
        from huggingface_hub import snapshot_download

        revision = Path(snapshot_download("gpt2", local_files_only=True)).name
        if revision != manifest["reference_model"]["revision"]:
            raise ValueError("cached GPT-2 main revision differs from the original run")
    out = ROOT / "runs" / args.output_id
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "HF_HUB_OFFLINE": "1"})
    cmd = [sys.executable, str(ROOT / "scripts" / "profile_upstream.py"), "--upstream", str(UPSTREAM),
           "--phase", "scoring", "--timing-out", str(out / "scoring-timing.json"),
           "--config", str(config), "--glob", str(shard)]
    record = {"source_run": args.run_id, "source_status": manifest["status"],
              "shard_sha256": digest(shard), "gpu": args.gpu,
              "status": "running", "exception": None}
    try:
        with (out / "scoring.log").open("x", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=UPSTREAM, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        record["status"] = "succeeded"
        record["scoring_log_sha256"] = digest(out / "scoring.log")
    except Exception as exc:
        record["status"] = "failed"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        print(f"Scoring failed; see {out / 'scoring.log'}", file=sys.stderr)
        raise
    finally:
        write_json(out / "rescore.json", record)
    print((out / "scoring.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
