"""Independent offline metrics for saved baseline candidate pools."""

from __future__ import annotations

from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
import json

import sacrebleu
from rouge_score import rouge_scorer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number}: blank JSONL record")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: record must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no records")
    return rows


def _stable_example_id(row: dict[str, Any], kind: str) -> str | None:
    example_id = row.get("example_id")
    if isinstance(example_id, str) and example_id:
        return example_id
    reference_id = row.get("reference_id")
    if kind != "candidate" and isinstance(reference_id, str) and reference_id:
        return reference_id
    task, input_id = row.get("task"), row.get("input_id")
    if isinstance(task, str) and isinstance(input_id, int) and not isinstance(input_id, bool):
        return f"{task}:test:{input_id}"
    return None


def _record_index(rows: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        row = dict(row)
        example_id = _stable_example_id(row, kind)
        input_id = row.get("input_id")
        task = row.get("task")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"{kind} record has no example_id")
        if not isinstance(input_id, int) or isinstance(input_id, bool) or input_id < 0:
            raise ValueError(f"{kind} record {example_id} has invalid input_id")
        if not isinstance(task, str) or not task:
            raise ValueError(f"{kind} record {example_id} has no task")
        if example_id in indexed:
            raise ValueError(f"duplicate {kind} example_id: {example_id}")
        row["example_id"] = example_id
        indexed[example_id] = row
    return indexed


def validate_records(
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Validate and join saved records, returning a stable input/rank ordering."""
    reference_by_id = _record_index(references, "reference")
    context_by_id = _record_index(contexts, "context")
    candidate_ids: set[str] = set()
    candidates_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task: str | None = None

    for candidate in candidates:
        candidate = dict(candidate)
        candidate_id = candidate.get("candidate_id")
        example_id = candidate.get("example_id")
        input_id = candidate.get("input_id")
        rank = candidate.get("candidate_rank")
        text = candidate.get("text")
        candidate_task = candidate.get("task")
        if not isinstance(input_id, int) or isinstance(input_id, bool) or input_id < 0:
            raise ValueError("candidate has invalid input_id")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError(f"candidate for input {input_id} has invalid candidate_rank")
        if not isinstance(candidate_task, str) or not candidate_task:
            raise ValueError(f"candidate for input {input_id} has no task")
        if not isinstance(candidate_id, str) or not candidate_id:
            candidate_id = f"{candidate_task}:test:{input_id}:candidate:{rank}"
            candidate["candidate_id"] = candidate_id
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        if not isinstance(example_id, str) or not example_id:
            example_id = _stable_example_id(candidate, "candidate")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"candidate {candidate_id} has no example_id")
            candidate["example_id"] = example_id
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"candidate {candidate_id} has an empty generation")
        if task is None:
            task = candidate_task
        elif task != candidate_task:
            raise ValueError("candidate records contain multiple tasks")
        candidates_by_example[example_id].append(candidate)

    if task is None:
        raise ValueError("no candidate records")
    candidate_example_ids = set(candidates_by_example)
    if candidate_example_ids != set(reference_by_id):
        missing = sorted(candidate_example_ids - set(reference_by_id))
        extra = sorted(set(reference_by_id) - candidate_example_ids)
        raise ValueError(f"candidate/reference example mismatch: missing={missing}, extra={extra}")
    if candidate_example_ids != set(context_by_id):
        missing = sorted(candidate_example_ids - set(context_by_id))
        extra = sorted(set(context_by_id) - candidate_example_ids)
        raise ValueError(f"candidate/context example mismatch: missing={missing}, extra={extra}")

    ordered = []
    pool_sizes = set()
    for example_id, pool in candidates_by_example.items():
        reference = reference_by_id[example_id]
        context = context_by_id[example_id]
        first = pool[0]
        for row, kind in ((reference, "reference"), (context, "context")):
            if row["task"] != task or row["input_id"] != first["input_id"]:
                raise ValueError(f"{kind} identity mismatch for {example_id}")
        if any(row["task"] != task or row["input_id"] != first["input_id"] for row in pool):
            raise ValueError(f"candidate identity mismatch for {example_id}")
        if not isinstance(reference.get("text"), str) or not reference["text"].strip():
            raise ValueError(f"missing reference text for {example_id}")
        pool.sort(key=lambda row: row["candidate_rank"])
        ranks = [row["candidate_rank"] for row in pool]
        if ranks != list(range(len(pool))):
            raise ValueError(f"candidate ranks for {example_id} must be contiguous from zero")
        pool_sizes.add(len(pool))
        ordered.extend((row, reference, context) for row in pool)

    if len(pool_sizes) != 1:
        raise ValueError(f"unequal candidate pool sizes: {sorted(pool_sizes)}")
    ordered.sort(key=lambda item: (item[0]["input_id"], item[0]["candidate_rank"]))
    input_ids = [item[0]["input_id"] for item in ordered]
    if len(set(input_ids)) != len(candidates_by_example):
        raise ValueError("multiple example IDs share an input_id")
    return task, ordered


def _sudoku_grid(text: str) -> str | None:
    stripped = text.strip()
    if "|" in stripped:
        rows = stripped.split("|")
        if len(rows) != 9 or any(len(row) != 9 for row in rows):
            return None
        grid = "".join(rows)
    else:
        grid = stripped
    if len(grid) != 81 or any(char not in "123456789" for char in grid):
        return None
    return grid


def _sudoku_puzzle(token_ids: Any) -> str:
    if not isinstance(token_ids, list) or len(token_ids) != 91:
        raise ValueError("Sudoku context must contain 91 tokenizer IDs")
    if token_ids[0] != 11 or token_ids[-1] != 11:
        raise ValueError("Sudoku context must have BOS tokens at both ends")
    cells = []
    for position, token in enumerate(token_ids[1:-1]):
        if not isinstance(token, int) or isinstance(token, bool) or not 0 <= token <= 10:
            raise ValueError("Sudoku context contains an invalid tokenizer ID")
        if position in (9, 19, 29, 39, 49, 59, 69, 79):
            if token != 10:
                raise ValueError("Sudoku context has an invalid row separator")
        else:
            if token == 10:
                raise ValueError("Sudoku context has an unexpected row separator")
            cells.append(str(token))
    if len(cells) != 81:
        raise ValueError("Sudoku context does not contain 81 cells")
    return "".join(cells)


def _valid_sudoku(grid: str | None) -> bool:
    if grid is None:
        return False
    rows = [grid[row * 9:(row + 1) * 9] for row in range(9)]
    columns = ["".join(rows[row][column] for row in range(9)) for column in range(9)]
    boxes = [
        "".join(rows[row][column]
                for row in range(box_row, box_row + 3)
                for column in range(box_column, box_column + 3))
        for box_row in (0, 3, 6)
        for box_column in (0, 3, 6)
    ]
    expected = set("123456789")
    return all(set(unit) == expected for unit in rows + columns + boxes)


def score_sudoku(
    ordered: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    per_candidate = []
    for candidate, reference, context in ordered:
        prediction = _sudoku_grid(candidate["text"])
        target = _sudoku_grid(reference["text"])
        if target is None:
            raise ValueError(f"reference {reference['example_id']} is not a complete Sudoku grid")
        puzzle = _sudoku_puzzle(context.get("source_token_ids"))
        cell_text = candidate["text"].strip().replace("|", "")
        cell_accuracy = sum(
            index < len(cell_text) and cell_text[index] == target[index]
            for index in range(81)
        ) / 81
        valid = _valid_sudoku(prediction)
        preserves_clues = (
            prediction is not None
            and all(clue == "0" or prediction[index] == clue
                    for index, clue in enumerate(puzzle))
        )
        per_candidate.append({
            "candidate_id": candidate["candidate_id"],
            "example_id": candidate["example_id"],
            "candidate_rank": candidate["candidate_rank"],
            "exact_solution": candidate["text"] == reference["text"],
            "well_formed": prediction is not None,
            "valid_grid": valid,
            "preserves_clues": preserves_clues,
            "cell_accuracy": cell_accuracy,
        })
    count = len(per_candidate)
    return {
        "exact_solution_rate": sum(row["exact_solution"] for row in per_candidate) / count,
        "well_formed_rate": sum(row["well_formed"] for row in per_candidate) / count,
        "valid_grid_rate": sum(row["valid_grid"] for row in per_candidate) / count,
        "clue_preservation_rate": sum(row["preserves_clues"] for row in per_candidate) / count,
        "mean_cell_accuracy": sum(row["cell_accuracy"] for row in per_candidate) / count,
        "per_candidate": per_candidate,
    }


def score_xsum(
    ordered: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    per_candidate = []
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for candidate, reference, _ in ordered:
        values = scorer.score(reference["text"], candidate["text"])
        metrics = {name: values[name].fmeasure for name in totals}
        per_candidate.append({
            "candidate_id": candidate["candidate_id"],
            "example_id": candidate["example_id"],
            "candidate_rank": candidate["candidate_rank"],
            **metrics,
        })
        for name, value in metrics.items():
            totals[name] += value
    count = len(per_candidate)
    return {
        "mean_rouge_fmeasure_percent": {
            name: 100 * value / count for name, value in totals.items()
        },
        "per_candidate": per_candidate,
    }


def score_wmt14(
    ordered: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    hypotheses = [candidate["text"] for candidate, _, _ in ordered]
    references = [reference["text"] for _, reference, _ in ordered]
    bleu_metric = sacrebleu.metrics.BLEU()
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    bleu_lowercase = sacrebleu.metrics.BLEU(lowercase=True).corpus_score(
        hypotheses, [references]
    )
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    hypothesis_words = sum(len(text.split()) for text in hypotheses)
    reference_words = sum(len(text.split()) for text in references)
    if reference_words == 0:
        raise ValueError("WMT references have zero whitespace-token length")
    return {
        "corpus_bleu": bleu.score,
        "corpus_bleu_lowercase": bleu_lowercase.score,
        "corpus_chrf": chrf.score,
        "length_ratio": hypothesis_words / reference_words,
        "sacrebleu_signature": str(bleu_metric.get_signature()),
    }


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def score_run(run_dir: Path) -> dict[str, Any]:
    candidates = read_jsonl(run_dir / "candidates.jsonl")
    references = read_jsonl(run_dir / "references.jsonl")
    contexts = read_jsonl(run_dir / "contexts.jsonl")
    task, ordered = validate_records(candidates, references, contexts)
    scorers = {
        "sudoku-easy": score_sudoku,
        "xsum": score_xsum,
        "wmt14": score_wmt14,
    }
    if task not in scorers:
        raise ValueError(f"unsupported metric task: {task}")
    count_per_input = len(ordered) // len({item[0]["input_id"] for item in ordered})
    return {
        "schema_version": 1,
        "task": task,
        "input_count": len({item[0]["input_id"] for item in ordered}),
        "candidate_count": len(ordered),
        "candidates_per_input": count_per_input,
        "metric_implementation": "scripts/metrics.py",
        "metric_dependencies": {
            "rouge_score": _package_version("rouge-score"),
            "sacrebleu": _package_version("sacrebleu"),
        },
        "metrics": scorers[task](ordered),
    }
