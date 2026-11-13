"""Prepare fixed, reference-separated Phase 1 development examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "configs" / "phase0" / "artifacts.json"
OUTPUT = ROOT / ".local" / "artifacts" / "phase1-dev" / "prepared"
SUDOKU_SEED = 20261112
SUDOKU_CLUES = 40
ALL_DIGITS = (1 << 9) - 1


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def write_once(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise RuntimeError(f"refusing to overwrite changed development data: {path}")
    else:
        with path.open("xb") as file:
            file.write(data)
    return sha256(data)


def sudoku_solution(seed: int) -> list[int]:
    rng = random.Random(seed)
    bands = rng.sample(range(3), 3)
    rows = [band * 3 + row for band in bands for row in rng.sample(range(3), 3)]
    stacks = rng.sample(range(3), 3)
    columns = [stack * 3 + column for stack in stacks for column in rng.sample(range(3), 3)]
    symbols = rng.sample(range(1, 10), 9)
    return [
        symbols[(3 * (row % 3) + row // 3 + column) % 9]
        for row in rows
        for column in columns
    ]


def count_solutions(board: list[int], limit: int = 2) -> int:
    rows = [0] * 9
    columns = [0] * 9
    boxes = [0] * 9
    empty = []
    for cell, value in enumerate(board):
        row, column = divmod(cell, 9)
        box = row // 3 * 3 + column // 3
        if value == 0:
            empty.append(cell)
            continue
        bit = 1 << (value - 1)
        if rows[row] & bit or columns[column] & bit or boxes[box] & bit:
            return 0
        rows[row] |= bit
        columns[column] |= bit
        boxes[box] |= bit

    solutions = 0

    def search() -> None:
        nonlocal solutions
        if solutions >= limit:
            return
        best_position = -1
        best_options = 0
        best_count = 10
        for offset, cell in enumerate(empty):
            if board[cell] != 0:
                continue
            row, column = divmod(cell, 9)
            box = row // 3 * 3 + column // 3
            options = ALL_DIGITS & ~(rows[row] | columns[column] | boxes[box])
            option_count = options.bit_count()
            if option_count < best_count:
                best_position, best_options, best_count = offset, options, option_count
                if option_count <= 1:
                    break
        if best_position < 0:
            solutions += 1
            return
        if best_count == 0:
            return
        cell = empty[best_position]
        row, column = divmod(cell, 9)
        box = row // 3 * 3 + column // 3
        while best_options and solutions < limit:
            bit = best_options & -best_options
            best_options -= bit
            board[cell] = bit.bit_length()
            rows[row] |= bit
            columns[column] |= bit
            boxes[box] |= bit
            search()
            board[cell] = 0
            rows[row] ^= bit
            columns[column] ^= bit
            boxes[box] ^= bit

    search()
    return solutions


def sudoku_examples(count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    references = []
    for input_id in range(count):
        rng = random.Random(seed + input_id)
        solution = sudoku_solution(seed + input_id)
        for _ in range(1000):
            blank_cells = set(rng.sample(range(81), 81 - SUDOKU_CLUES))
            puzzle = [0 if cell in blank_cells else value
                      for cell, value in enumerate(solution)]
            if count_solutions(puzzle.copy()) == 1:
                break
        else:
            raise RuntimeError(f"could not generate a unique Sudoku puzzle for input {input_id}")
        encoded = [11]
        for row in range(9):
            encoded.extend(puzzle[row * 9:(row + 1) * 9])
            if row < 8:
                encoded.append(10)
        encoded.append(11)
        target = "|".join(
            "".join(str(value) for value in solution[row * 9:(row + 1) * 9])
            for row in range(9)
        )
        inputs.append({"task": "sudoku-easy", "split": "validation",
                       "input_id": input_id, "source": encoded})
        references.append({"task": "sudoku-easy", "split": "validation",
                           "input_id": input_id,
                           "example_id": f"sudoku-easy:validation:{input_id}",
                           "reference_id": f"sudoku-easy:validation:{input_id}",
                           "text": target})
    return inputs, references


def _xsum_rows(spec: dict[str, Any], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = ROOT / spec["local_dir"] / spec["files"][0]["path"]
    table = parquet.read_table(source, columns=["document", "summary", "id"])
    if table.num_rows < count:
        raise ValueError(f"XSum validation has only {table.num_rows} rows")
    tokenizer_spec = spec["tokenizer"]
    tokenizer_path = snapshot_download(
        tokenizer_spec["repo"], revision=tokenizer_spec["revision"], local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.model_max_length = max(tokenizer.model_max_length, 1_000_000)
    rows = table.slice(0, count).to_pylist()
    inputs, references = [], []
    for input_id, row in enumerate(rows):
        document, summary = row["document"], row["summary"]
        if not isinstance(document, str) or not document.strip():
            raise ValueError(f"empty XSum validation document at input {input_id}")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"empty XSum validation summary at input {input_id}")
        source_ids = tokenizer.encode(document, add_special_tokens=False)
        if not source_ids:
            raise ValueError(f"XSum document tokenized to an empty input at {input_id}")
        inputs.append({"task": "xsum", "split": "validation",
                       "input_id": input_id, "source": source_ids})
        references.append({"task": "xsum", "split": "validation", "input_id": input_id,
                           "example_id": f"xsum:validation:{input_id}",
                           "reference_id": f"xsum:validation:{input_id}", "text": summary})
    return inputs, references


def _wmt_rows(spec: dict[str, Any], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = ROOT / spec["local_dir"] / spec["files"][0]["path"]
    table = parquet.read_table(source, columns=["translation"])
    if table.num_rows < count:
        raise ValueError(f"WMT14 validation has only {table.num_rows} rows")
    rows = table.slice(0, count).to_pylist()
    inputs, references = [], []
    for input_id, row in enumerate(rows):
        pair = row["translation"]
        source_text, target_text = pair["de"], pair["en"]
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"empty WMT14 validation source at input {input_id}")
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError(f"empty WMT14 validation target at input {input_id}")
        inputs.append({"task": "wmt14", "split": "validation",
                       "input_id": input_id, "source": source_text})
        references.append({"task": "wmt14", "split": "validation", "input_id": input_id,
                           "example_id": f"wmt14:validation:{input_id}",
                           "reference_id": f"wmt14:validation:{input_id}",
                           "text": target_text})
    return inputs, references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")

    manifest = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    specs = {spec["task"]: spec for spec in manifest["development_datasets"]}
    builders = {"xsum": _xsum_rows, "wmt14": _wmt_rows}
    for task in ("xsum", "wmt14"):
        inputs, references = builders[task](specs[task], args.n)
        _write_prepared(task, inputs, references, specs[task])

    inputs, references = sudoku_examples(args.n, SUDOKU_SEED)
    _write_prepared(
        "sudoku-easy",
        inputs,
        references,
        {"source": "deterministic synthetic Sudoku generation",
         "seed": SUDOKU_SEED, "clues_per_puzzle": SUDOKU_CLUES},
    )


def _write_prepared(
    task: str,
    inputs: list[dict[str, Any]],
    references: list[dict[str, Any]],
    source_spec: dict[str, Any],
) -> None:
    directory = OUTPUT / task
    inputs_data = jsonl_bytes(inputs)
    references_data = jsonl_bytes(references)
    output_hashes = {
        "inputs.jsonl": write_once(directory / "inputs.jsonl", inputs_data),
        "references.jsonl": write_once(directory / "references.jsonl", references_data),
    }
    manifest = {
        "task": task,
        "split": "validation",
        "count": len(inputs),
        "source": source_spec,
        "input_sha256": output_hashes["inputs.jsonl"],
        "reference_sha256": output_hashes["references.jsonl"],
    }
    manifest_data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_once(directory / "dataset-manifest.json", manifest_data)


if __name__ == "__main__":
    main()
