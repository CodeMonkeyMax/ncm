"""Independently score an archived baseline run using only its saved JSONL records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from metrics import score_run

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (args.run_id, args.output_id)):
        parser.error("IDs may contain only letters, digits, hyphens and underscores")

    source = ROOT / "runs" / args.run_id
    output = ROOT / "runs" / args.output_id
    if not source.is_dir():
        parser.error(f"source run does not exist: {source}")
    output.mkdir(parents=False, exist_ok=False)
    record = {
        "source_run": args.run_id,
        "input_sha256": {
            name: digest(source / name)
            for name in ("candidates.jsonl", "contexts.jsonl", "references.jsonl")
        },
        "status": "running",
        "exception": None,
    }
    try:
        metrics = score_run(source)
        write_json(output / "metrics.json", metrics)
        record["status"] = "succeeded"
        record["metrics_sha256"] = digest(output / "metrics.json")
    except Exception as exc:
        record["status"] = "failed"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        print(f"Independent scoring failed: {record['exception']}", file=sys.stderr)
        raise
    finally:
        write_json(output / "run-record.json", record)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
