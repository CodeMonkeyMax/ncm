"""Run a pinned, single-shard official Phase 0 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

from verify_p0a import main as verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".local" / "upstream" / "ncm"
MANIFEST = ROOT / "configs" / "phase0" / "artifacts.json"
LOCK = ROOT / "configs" / "phase0" / "requirements-lock.txt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(shard_path: Path, source_path: Path, count: int, task: str):
    with shard_path.open("rb") as f:
        shard = pickle.load(f)
    with source_path.open("rb") as f:
        sources = pickle.load(f)
    ids, hyps, refs = shard["idxs"], shard["hyps"], shard["refs"]
    if ids != list(range(count)) or len(hyps) != count or len(refs) != count or len(sources) < count:
        raise ValueError("input indices, hypotheses, references or source counts do not match")
    if any(not isinstance(x, str) or not x.strip() for x in [*hyps, *refs]):
        raise ValueError("empty or non-string hypothesis or reference")
    candidates = [
        {"task": task, "input_id": i, "candidate_rank": 0, "seed": None,
         "text": h, "generation_duration_seconds": None, "validity": None, "metrics": None}
        for i, h in zip(ids, hyps)
    ]
    contexts = [
        {"task": task, "input_id": i, "source_token_ids": list(sources[i]),
         "reference_id": f"{task}:test:{i}",
         "constraints": "preserve clues and solve 9x9 grid" if task == "sudoku-easy" else "summarize the source article"}
        for i in ids
    ]
    references = [
        {"task": task, "input_id": i, "reference_id": f"{task}:test:{i}", "text": r}
        for i, r in zip(ids, refs)
    ]
    return candidates, contexts, references


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("xsum", "sudoku-easy"), required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    if a.n < 1 or a.bs < 1 or a.gpu < 0:
        ap.error("n and bs must be positive and gpu must be nonnegative")
    if not a.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in a.run_id):
        ap.error("run-id may contain only letters, digits, hyphens and underscores")
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if git("rev-parse", "HEAD", cwd=UPSTREAM) != spec["upstream"]["commit"]:
        raise RuntimeError("upstream commit does not match artifact manifest")
    if git("status", "--porcelain", cwd=UPSTREAM):
        raise RuntimeError("upstream checkout is modified")
    if verify_artifacts():
        raise RuntimeError("artifact verification failed")

    rel = "xsum" if a.task == "xsum" else "sudoku/easy"
    checkpoint = ROOT / spec["checkpoints"]["local_dir"] / rel
    data = ROOT / spec["datasets"]["local_dir"] / rel
    config = UPSTREAM / "config" / "eval" / f"{a.task}.yaml"
    cfg = OmegaConf.load(config)
    if a.n > int(cfg.eval.n_docs):
        ap.error(f"n exceeds configured test input count {cfg.eval.n_docs}")
    reference_model = None
    if a.task == "xsum":
        snapshot = Path(snapshot_download("gpt2", local_files_only=True))
        reference_model = {"id": "gpt2", "revision": snapshot.name}
    out = ROOT / "runs" / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(a.gpu), "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
    started = datetime.now(timezone.utc).isoformat()
    record = {
        "run_id": a.run_id, "status": "running", "started_utc": started,
        "repository_head": git("rev-parse", "HEAD", cwd=ROOT),
        "repository_dirty": bool(git("status", "--porcelain", cwd=ROOT)),
        "upstream_commit": spec["upstream"]["commit"],
        "artifact_manifest": spec, "lock_sha256": digest(LOCK),
        "environment": {"python": platform.python_version(),
                        "packages": subprocess.check_output(
                            [sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
                        "gpu_driver": subprocess.check_output(
                            ["nvidia-smi", "-i", str(a.gpu), "--query-gpu=driver_version",
                             "--format=csv,noheader"], text=True).strip(),
                        "cuda_allocator_config": env.get("PYTORCH_CUDA_ALLOC_CONF")},
        "adapter_sha256": digest(Path(__file__)),
        "profiler_sha256": digest(ROOT / "scripts" / "profile_upstream.py"),
        "reference_model": reference_model,
        "config_sha256": digest(config), "task": a.task, "split": "test",
        "input_ids": list(range(a.n)), "shards": [{"shard": 0, "gpu": a.gpu, "input_ids": list(range(a.n))}],
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "batch_size": a.bs, "candidates_per_input": 1, "replicate": 0,
        "seed_schedule": None, "seed_note": "upstream does not explicitly seed conditional evaluation",
        "offline": True, "reference_isolation": "upstream shard loads references during generation",
        "warmup_policy": "first output-producing batch excluded from warm generate timing",
        "retries": 0, "exception": None,
    }
    write_json(out / "manifest.json", record)
    worker = ROOT / "scripts" / "profile_upstream.py"
    common = [sys.executable, str(worker), "--upstream", str(UPSTREAM)]
    shard = out / "shard_0.pkl"

    def run(phase: str, args: list[str]) -> None:
        cmd = [*common, "--phase", phase, "--timing-out", str(out / f"{phase}-timing.json"), *args]
        with (out / f"{phase}.log").open("x", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=UPSTREAM, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        run("generation", ["--config", str(config), "--checkpoint-dir", str(checkpoint),
            "--data-dir", str(data), "--n", str(a.n), "--bs", str(a.bs), "--shard", "0",
            "--nshards", "1", "--out", str(shard)])
        candidates, contexts, references = normalize(shard, data / cfg.data.test, a.n, a.task)
        write_jsonl(out / "candidates.jsonl", candidates)
        write_jsonl(out / "contexts.jsonl", contexts)
        write_jsonl(out / "references.jsonl", references)
        run("scoring", ["--config", str(config), "--glob", str(shard)])
        record["status"] = "succeeded"
        record["outputs"] = {p.name: digest(p) for p in (shard, out / "candidates.jsonl", out / "contexts.jsonl",
                                                        out / "references.jsonl", out / "scoring.log")}
        record["metric"] = {"implementation": "upstream scripts/eval_merge.py", "metric": str(cfg.eval.metric)}
        record["timings"] = {
            phase: json.loads((out / f"{phase}-timing.json").read_text(encoding="utf-8"))
            for phase in ("generation", "scoring")
        }
        print((out / "scoring.log").read_text(encoding="utf-8"))
    except Exception as exc:
        record["status"] = "failed"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "manifest.json", record)
        print(f"Run record: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
