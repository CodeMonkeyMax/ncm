"""Prepare a reference-isolated held-out prefix from verified Phase 0 test runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUTPUT = ROOT / ".local" / "artifacts" / "phase1-heldout" / "prepared"
BASELINE_RUNS = {
    "xsum": "p05-xsum-n200-rtx3080-v1",
    "sudoku-easy": "p05-sudoku-n200-rtx3080-v1",
    "wmt14": "p05-wmt14-n200-rtx3080-v1",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number}: blank JSONL record")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no records")
    return rows


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def write_once(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise RuntimeError(f"refusing to overwrite changed held-out data: {path}")
    else:
        with path.open("xb") as file:
            file.write(data)
    return sha256(data)


def heldout_rows(
    task: str,
    contexts: list[dict[str, Any]],
    references: list[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if count < 1 or len(contexts) < count or len(references) < count:
        raise ValueError("the baseline run does not contain the requested held-out prefix")
    inputs, heldout_references = [], []
    for input_id, (context, reference) in enumerate(zip(contexts[:count], references[:count])):
        example_id = f"{task}:test:{input_id}"
        if (context.get("task") != task or context.get("input_id") != input_id
                or context.get("example_id") != example_id):
            raise ValueError(f"invalid held-out context {input_id}")
        source_fields = [key for key in ("source_text", "source_token_ids") if key in context]
        if len(source_fields) != 1:
            raise ValueError(f"held-out context {input_id} must have exactly one source field")
        source = context[source_fields[0]]
        if (not isinstance(source, (str, list)) or not source
                or isinstance(source, str) and not source.strip()):
            raise ValueError(f"held-out context {input_id} has an empty or invalid source")
        if (reference.get("task") != task or reference.get("input_id") != input_id
                or reference.get("example_id") != example_id
                or reference.get("reference_id") != example_id
                or not isinstance(reference.get("text"), str)
                or not reference["text"].strip()):
            raise ValueError(f"invalid held-out reference {input_id}")
        inputs.append({
            "task": task, "split": "test", "input_id": input_id, "source": source,
        })
        heldout_references.append({
            "task": task, "split": "test", "input_id": input_id,
            "example_id": example_id, "reference_id": example_id,
            "text": reference["text"],
        })
    return inputs, heldout_references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(BASELINE_RUNS), required=True)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")

    run_id = BASELINE_RUNS[args.task]
    run_dir = RUNS / run_id
    baseline_manifest_path = run_dir / "manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if (baseline_manifest.get("status") != "succeeded"
            or baseline_manifest.get("task") != args.task
            or baseline_manifest.get("split") != "test"):
        raise ValueError(f"{run_id} is not a successful official test run for {args.task}")

    contexts_path = run_dir / "contexts.jsonl"
    references_path = run_dir / "references.jsonl"
    context_data = contexts_path.read_bytes()
    reference_data = references_path.read_bytes()
    inputs, references = heldout_rows(
        args.task, read_jsonl(contexts_path), read_jsonl(references_path), args.n
    )
    inputs_data = jsonl_bytes(inputs)
    references_data = jsonl_bytes(references)
    output_dir = OUTPUT / args.task
    input_hash = write_once(output_dir / "inputs.jsonl", inputs_data)
    reference_hash = write_once(output_dir / "references.jsonl", references_data)
    dataset_manifest = {
        "task": args.task,
        "split": "test",
        "count": len(inputs),
        "source": {
            "baseline_run_id": run_id,
            "baseline_manifest_sha256": sha256(baseline_manifest_path.read_bytes()),
            "contexts_sha256": sha256(context_data),
            "references_sha256": sha256(reference_data),
        },
        "input_sha256": input_hash,
        "reference_sha256": reference_hash,
    }
    manifest_data = (json.dumps(dataset_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_once(output_dir / "dataset-manifest.json", manifest_data)
    print(f"Prepared {len(inputs)} held-out {args.task} inputs from {run_id}")


if __name__ == "__main__":
    main()
