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
from huggingface_hub import snapshot_download, try_to_load_from_cache

from verify_p0a import main as verify_artifacts

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".local" / "upstream" / "ncm"
MANIFEST = ROOT / "configs" / "phase0" / "artifacts.json"
LOCK = ROOT / "configs" / "phase0" / "requirements-lock.txt"


def artifact_task_path(task: str) -> str:
    paths = {"xsum": "xsum", "sudoku-easy": "sudoku/easy", "wmt14": "wmt14"}
    try:
        return paths[task]
    except KeyError as exc:
        raise ValueError(f"unsupported baseline task: {task}") from exc


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def repository_snapshot() -> dict:
    status = git("status", "--porcelain", "--untracked-files=all", cwd=ROOT)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    untracked_hashes = {
        path: digest(ROOT / path)
        for path in untracked if path
    }
    return {
        "head": git("rev-parse", "HEAD", cwd=ROOT),
        "dirty": bool(status),
        "diff_head_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files_sha256": untracked_hashes,
    }


def write_json(path: Path, obj: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_generation_data(source_path: Path, data_dir: Path, source_name: str,
                          reference_name: str, count: int) -> None:
    if source_name == reference_name or Path(source_name).name != source_name or Path(reference_name).name != reference_name:
        raise ValueError("generation input and placeholder filenames must be distinct basenames")
    with source_path.open("rb") as f:
        sources = pickle.load(f)
    if len(sources) < count:
        raise ValueError("not enough generation inputs")
    data_dir.mkdir()
    with (data_dir / source_name).open("xb") as f:
        pickle.dump(sources[:count], f)
    with (data_dir / reference_name).open("xb") as f:
        pickle.dump([""] * count, f)


def attach_references(generation_shard: Path, reference_path: Path, output: Path,
                      count: int, task: str) -> None:
    with generation_shard.open("rb") as f:
        shard = pickle.load(f)
    with reference_path.open("rb") as f:
        refs = pickle.load(f)
    if shard["idxs"] != list(range(count)) or len(shard["hyps"]) != count:
        raise ValueError("generation shard indices or hypothesis count do not match")
    if shard["refs"] != [""] * count or len(refs) < count:
        raise ValueError("generation placeholders or reference count do not match")
    if task == "sudoku-easy":
        sys.path.insert(0, str(UPSTREAM))
        try:
            from tokenization.sudoku_tokenizer import SudokuTokenizer
            tokenizer = SudokuTokenizer()
            refs = [tokenizer.decode(r, skip_special_tokens=True) for r in refs[:count]]
        finally:
            sys.path.pop(0)
    else:
        refs = refs[:count]
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError("empty or non-string reference")
    shard["refs"] = refs
    with output.open("xb") as f:
        pickle.dump(shard, f)


def normalize(shard_path: Path, source_path: Path, count: int, task: str,
              batch_times: list[float] | None = None, batch_size: int | None = None):
    if batch_times is not None:
        if batch_size is None or batch_size < 1 or len(batch_times) != (count + batch_size - 1) // batch_size:
            raise ValueError("invalid batch timing schedule")
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
        {"candidate_id": f"{task}:test:{i}:candidate:0",
         "task": task, "input_id": i, "example_id": f"{task}:test:{i}",
         "candidate_rank": 0, "seed": None,
         "text": h, "generation_duration_seconds": None,
         "generation_batch_duration_seconds": batch_times[i // batch_size] if batch_times is not None else None,
         "generation_batch_size": min(batch_size, count - (i // batch_size) * batch_size) if batch_times is not None else None,
         "validity": None, "metrics": None}
        for i, h in zip(ids, hyps)
    ]
    contexts = []
    for i in ids:
        context = {
            "task": task, "input_id": i, "example_id": f"{task}:test:{i}",
            "reference_id": f"{task}:test:{i}",
            "constraints": "preserve clues and solve 9x9 grid"
            if task == "sudoku-easy" else
            "translate the source sentence" if task == "wmt14" else
            "summarize the source article",
        }
        if task == "wmt14":
            if not isinstance(sources[i], str):
                raise ValueError("WMT14 source context must be text")
            context["source_text"] = sources[i]
        else:
            context["source_token_ids"] = list(sources[i])
        contexts.append(context)
    references = [
        {"task": task, "input_id": i, "example_id": f"{task}:test:{i}",
         "reference_id": f"{task}:test:{i}", "text": r}
        for i, r in zip(ids, refs)
    ]
    return candidates, contexts, references


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("xsum", "sudoku-easy", "wmt14"), required=True)
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

    rel = artifact_task_path(a.task)
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
    pre_tokenizer_model = None
    if cfg.tokenizer.pre_tokenizer.name == "hf":
        model_id = str(cfg.tokenizer.pre_tokenizer.model_id)
        pre_tokenizer_spec = spec.get("pre_tokenizers", {})
        if model_id == pre_tokenizer_spec.get("repo"):
            cached_files = [
                try_to_load_from_cache(model_id, name, revision="main")
                for name in ("config.json", "tokenizer.json")
            ]
            if any(path is None for path in cached_files):
                raise RuntimeError(f"pinned tokenizer files are not cached for {model_id}")
            snapshots = {Path(path).parent for path in cached_files if path is not None}
            if len(snapshots) != 1:
                raise RuntimeError(f"cached tokenizer files resolve to different revisions for {model_id}")
            snapshot = snapshots.pop()
        else:
            snapshot = Path(snapshot_download(model_id, local_files_only=True))
        if model_id == pre_tokenizer_spec.get("repo"):
            expected_revision = pre_tokenizer_spec["revision"]
            if snapshot.name != expected_revision:
                raise RuntimeError(
                    f"cached {model_id} revision {snapshot.name} differs from "
                    f"the pinned revision {expected_revision}"
                )
        pre_tokenizer_model = {"id": model_id, "revision": snapshot.name}
    out = ROOT / "runs" / a.run_id
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(a.gpu), "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
    started = datetime.now(timezone.utc).isoformat()
    repo_snapshot = repository_snapshot()
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "-i", str(a.gpu), "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader,nounits"], text=True
    ).strip().split(", ")
    if len(gpu_info) != 3:
        raise RuntimeError(f"unexpected nvidia-smi device response: {gpu_info!r}")
    input_ids = list(range(a.n))
    example_ids = [f"{a.task}:test:{i}" for i in input_ids]
    record = {
        "schema_version": 2,
        "run_id": a.run_id, "status": "running", "started_utc": started,
        "repository": repo_snapshot,
        "upstream_commit": spec["upstream"]["commit"],
        "artifact_manifest": spec, "lock_sha256": digest(LOCK),
        "environment": {"python": platform.python_version(),
                        "packages": subprocess.check_output(
                            [sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
                        "gpu": {"name": gpu_info[0], "memory_total_mib": int(gpu_info[1]),
                                "driver": gpu_info[2]},
                        "cuda_allocator_config": env.get("PYTORCH_CUDA_ALLOC_CONF")},
        "adapter_sha256": digest(Path(__file__)),
        "profiler_sha256": digest(ROOT / "scripts" / "profile_upstream.py"),
        "reference_model": reference_model,
        "pre_tokenizer_model": pre_tokenizer_model,
        "config_sha256": digest(config), "task": a.task, "split": "test",
        "dataset_files": {
            "inputs_sha256": digest(data / cfg.data.test),
            "references_sha256": digest(data / cfg.data.test_refs),
        },
        "input_ids": input_ids, "example_ids": example_ids,
        "input_order_path": "input-order.jsonl",
        "shards": [{"shard": 0, "gpu": a.gpu,
                    "input_ids": input_ids, "example_ids": example_ids}],
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "batch_size": a.bs, "candidates_per_input": 1, "replicate": 0,
        "seed_schedule": None, "seed_note": "upstream does not explicitly seed conditional evaluation",
        "offline": True,
        "reference_isolation": "generation reads staged inputs and empty placeholders; real references are attached after generation, not OS-sandboxed",
        "warmup_policy": "first output-producing batch excluded from warm generation timing",
        "output_paths": {
            "candidates": "candidates.jsonl", "contexts": "contexts.jsonl",
            "references": "references.jsonl", "generation_shard": "generation_shard.pkl",
            "scoring_shard": "shard_0.pkl", "generation_timing": "generation-timing.json",
            "scoring_timing": "scoring-timing.json", "input_order": "input-order.jsonl",
        },
        "retries": 0, "exception": None,
    }
    out.mkdir(parents=True, exist_ok=False)
    run_spec = {
        key: value for key, value in record.items()
        if key not in {"status", "exception", "retries"}
    }
    with (out / "run-spec.json").open("x", encoding="utf-8") as file:
        json.dump(run_spec, file, indent=2, ensure_ascii=False)
        file.write("\n")
    record["run_spec_sha256"] = digest(out / "run-spec.json")
    write_json(out / "manifest.json", record)
    worker = ROOT / "scripts" / "profile_upstream.py"
    common = [sys.executable, str(worker), "--upstream", str(UPSTREAM)]
    shard = out / "shard_0.pkl"
    generation_shard = out / "generation_shard.pkl"
    staged_data = out / "generation-input"

    def run(phase: str, args: list[str]) -> None:
        cmd = [*common, "--phase", phase, "--timing-out", str(out / f"{phase}-timing.json"), *args]
        with (out / f"{phase}.log").open("x", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=UPSTREAM, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        write_jsonl(out / "input-order.jsonl", [
            {"input_id": input_id, "example_id": f"{a.task}:test:{input_id}",
             "split": "test", "upstream_index": input_id, "shard": 0,
             "shard_position": input_id}
            for input_id in input_ids
        ])
        stage_generation_data(data / cfg.data.test, staged_data, str(cfg.data.test),
                              str(cfg.data.test_refs), a.n)
        run("generation", ["--config", str(config), "--checkpoint-dir", str(checkpoint),
            "--data-dir", str(staged_data), "--n", str(a.n), "--bs", str(a.bs), "--shard", "0",
            "--nshards", "1", "--out", str(generation_shard)])
        attach_references(generation_shard, data / cfg.data.test_refs, shard, a.n, a.task)
        timing = json.loads((out / "generation-timing.json").read_text(encoding="utf-8"))
        batch_times = timing["full_batch_seconds"]
        if len(batch_times) != (a.n + a.bs - 1) // a.bs:
            raise ValueError("generation timing batch count does not match input count")
        candidates, contexts, references = normalize(shard, data / cfg.data.test, a.n, a.task,
                                                     batch_times, a.bs)
        write_jsonl(out / "candidates.jsonl", candidates)
        write_jsonl(out / "contexts.jsonl", contexts)
        write_jsonl(out / "references.jsonl", references)
        completed = (generation_shard, shard, out / "candidates.jsonl", out / "contexts.jsonl",
                     out / "references.jsonl", out / "generation-timing.json",
                     out / "input-order.jsonl", out / "run-spec.json")
        with (out / "generation-checkpoint.json").open("x", encoding="utf-8") as file:
            json.dump({"sha256": {p.name: digest(p) for p in completed}}, file, indent=2)
            file.write("\n")
        run("scoring", ["--config", str(config), "--glob", str(shard)])
        record["status"] = "succeeded"
        record["outputs"] = {p.name: digest(p) for p in (
            generation_shard, shard, out / "candidates.jsonl", out / "contexts.jsonl",
            out / "references.jsonl", out / "input-order.jsonl", out / "generation-timing.json",
            out / "scoring-timing.json", out / "scoring.log",
            out / "run-spec.json", out / "generation-checkpoint.json")}
        record["metric"] = {
            "implementation": "scripts/eval_merge.py",
            "version": spec["upstream"]["commit"],
            "metric": str(cfg.eval.metric),
        }
        record["timings"] = {
            phase: json.loads((out / f"{phase}-timing.json").read_text(encoding="utf-8"))
            for phase in ("generation", "scoring")
        }
        print((out / "scoring.log").read_text(encoding="utf-8"))
    except Exception as exc:
        record["status"] = "failed"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        print(f"Run failed; check logs in {out}", file=sys.stderr)
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "manifest.json", record)
        print(f"Run record: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
