"""Analyze a completed held-out candidate pool and its paired greedy baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sacrebleu

from metrics import (
    read_jsonl,
    score_sudoku,
    score_wmt14,
    score_xsum,
    validate_records,
)

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (1, 4, 16, 64, 256)


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


def bootstrap_mean_interval(
    values: list[float], seed: int, samples: int = 5000,
) -> dict[str, float]:
    if not values or samples < 1:
        raise ValueError("bootstrap requires values and at least one sample")
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return {
        "lower_95": means[int(0.025 * samples)],
        "upper_95": means[min(samples - 1, int(0.975 * samples))],
    }


def _per_candidate_metrics(
    task: str,
    ordered: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    if task == "xsum":
        rows = score_xsum(ordered)["per_candidate"]
        return {
            row["candidate_id"]: {
                name: 100 * row[name] for name in ("rouge1", "rouge2", "rougeL")
            }
            for row in rows
        }
    if task == "sudoku-easy":
        rows = score_sudoku(ordered)["per_candidate"]
        return {
            row["candidate_id"]: {
                "exact_solution_rate": float(row["exact_solution"]),
                "valid_grid_rate": float(row["valid_grid"]),
                "clue_preservation_rate": float(row["preserves_clues"]),
            }
            for row in rows
        }
    if task == "wmt14":
        chrf = sacrebleu.metrics.CHRF()
        return {
            candidate["candidate_id"]: {
                "sentence_chrf": chrf.sentence_score(
                    candidate["text"], [reference["text"]]
                ).score,
            }
            for candidate, reference, _ in ordered
        }
    raise ValueError(f"unsupported Phase 1 task: {task}")


def _bootstrap_gain(
    gains: list[float], seed: int, samples: int,
) -> dict[str, float]:
    interval = bootstrap_mean_interval(gains, seed, samples)
    return {
        "mean": sum(gains) / len(gains),
        **interval,
    }


def _source_complexity(context: dict[str, Any], task: str) -> int:
    if task == "wmt14":
        return len(context["source_text"].split())
    source_ids = context["source_token_ids"]
    if task == "sudoku-easy":
        return sum(token in range(1, 10) for token in source_ids)
    return len(source_ids)


def analyze_candidate_pool(
    run_dir: Path,
    greedy_dir: Path,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20261112,
) -> dict[str, Any]:
    analysis_started = time.perf_counter()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    greedy_manifest = json.loads((greedy_dir / "manifest.json").read_text(encoding="utf-8"))
    for label, record in (("candidate", manifest), ("greedy", greedy_manifest)):
        if record.get("status") != "succeeded" or record.get("split") != "test":
            raise ValueError(f"{label} run must be a successful held-out test run")
    is_sampler_greedy = greedy_manifest.get("greedy_baseline") is True
    is_upstream_greedy = (
        greedy_manifest.get("candidates_per_input") == 1
        and greedy_manifest.get("config_sha256") == manifest.get("config_sha256")
        and greedy_manifest.get("upstream_commit") == manifest.get("upstream_commit")
    )
    if not (is_sampler_greedy or is_upstream_greedy):
        raise ValueError("comparison run is not marked as the untouched greedy baseline")
    if manifest.get("task") != greedy_manifest.get("task"):
        raise ValueError("candidate and greedy runs have different tasks")

    candidates = read_jsonl(run_dir / "candidates.jsonl")
    references = read_jsonl(run_dir / "references.jsonl")
    contexts = read_jsonl(run_dir / "contexts.jsonl")
    task, ordered = validate_records(candidates, references, contexts)
    if any(row["split"] != "test" for row in candidates):
        raise ValueError("candidate pool contains non-test records")
    if len(candidates) != len(references) * manifest["k"]:
        raise ValueError("candidate count does not match the run manifest")
    if [row["input_id"] for row in references] != manifest.get("input_ids"):
        raise ValueError("reference input IDs do not match the run manifest")

    greedy_candidates = read_jsonl(greedy_dir / "candidates.jsonl")
    greedy_references = read_jsonl(greedy_dir / "references.jsonl")
    greedy_contexts = read_jsonl(greedy_dir / "contexts.jsonl")
    greedy_task, greedy_ordered = validate_records(
        greedy_candidates, greedy_references, greedy_contexts
    )
    if greedy_task != task:
        raise ValueError("candidate and greedy records have different tasks")
    input_ids = [row["input_id"] for row in references]
    greedy_by_input = {
        row[0]["input_id"]: (row[0], row[1], row[2]) for row in greedy_ordered
    }
    if set(greedy_by_input) != set(input_ids) or len(greedy_candidates) != len(input_ids):
        raise ValueError("candidate and greedy runs do not contain the same input IDs")
    reference_texts = {row["input_id"]: row["text"] for row in references}
    greedy_reference_texts = {row["input_id"]: row["text"] for row in greedy_references}
    if reference_texts != greedy_reference_texts:
        raise ValueError("candidate and greedy runs use different references")
    candidate_contexts = {row["input_id"]: row for row in contexts}
    greedy_contexts_by_input = {row["input_id"]: row for row in greedy_contexts}
    if set(candidate_contexts) != set(greedy_contexts_by_input):
        raise ValueError("candidate and greedy runs use different contexts")
    for input_id, context in candidate_contexts.items():
        sources = [
            (key, context[key]) for key in ("source_text", "source_token_ids") if key in context
        ]
        greedy_sources = [
            (key, greedy_contexts_by_input[input_id][key])
            for key in ("source_text", "source_token_ids")
            if key in greedy_contexts_by_input[input_id]
        ]
        if len(sources) != 1 or sources != greedy_sources:
            raise ValueError(f"candidate and greedy sources differ for input {input_id}")
    if any(greedy_candidates_row.get("candidate_rank") != 0
           for greedy_candidates_row in greedy_candidates):
        raise ValueError("greedy baseline must contain only rank-zero candidates")

    scoring_started = time.perf_counter()
    candidate_metrics = _per_candidate_metrics(task, ordered)
    greedy_metrics = _per_candidate_metrics(task, greedy_ordered)
    pools: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        pools.setdefault(candidate["input_id"], []).append(candidate)
    for pool in pools.values():
        pool.sort(key=lambda row: row["candidate_rank"])
    if set(pools) != set(input_ids):
        raise ValueError("candidate pools do not match reference input IDs")
    if manifest.get("candidate_errors") != sum(
        candidate.get("generation_error") is not None for candidate in candidates
    ):
        raise ValueError("candidate error count does not match the run manifest")

    started = datetime.fromisoformat(manifest["started_utc"])
    ended = datetime.fromisoformat(manifest["ended_utc"])
    wall_seconds = (ended - started).total_seconds()
    generation_seconds = manifest["timings"]["generation_seconds"]
    model_load_seconds = manifest.get("model_load_seconds", 0.0)
    ranks = [k for k in K_VALUES if k <= manifest["k"]]
    metric_names = list(next(iter(candidate_metrics.values())))
    baseline_scorer = {
        "xsum": score_xsum,
        "wmt14": score_wmt14,
        "sudoku-easy": score_sudoku,
    }[task]
    greedy_metrics_summary = baseline_scorer(greedy_ordered)
    greedy_metrics_summary.pop("per_candidate", None)
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "greedy_run_id": greedy_manifest["run_id"],
        "task": task,
        "split": "test",
        "input_count": len(input_ids),
        "candidate_count": len(candidates),
        "candidate_k": manifest["k"],
        "k_values": ranks,
        "generation": manifest["generation"],
        "input_ids": manifest["input_ids"],
        "input_sha256": manifest["inputs_sha256"],
        "candidate_sha256": digest(run_dir / "candidates.jsonl"),
        "references_sha256": digest(run_dir / "references.jsonl"),
        "contexts_sha256": digest(run_dir / "contexts.jsonl"),
        "candidate_errors": sum(
            candidate.get("generation_error") is not None for candidate in candidates
        ),
        "generation_seconds": generation_seconds,
        "generation_seconds_per_candidate": (
            generation_seconds / len(candidates)
        ),
        "model_load_seconds": model_load_seconds,
        "wall_seconds": wall_seconds,
        "non_generation_overhead_seconds": max(
            0.0, wall_seconds - generation_seconds - model_load_seconds
        ),
        "greedy_metrics": greedy_metrics_summary,
        "metric_definitions": {
            "xsum": "mean per-example ROUGE F-measure, percentage; oracle maximizes each metric independently",
            "wmt14": "rank-one corpus BLEU/chrF; oracle ranks each candidate by sentence-level chrF",
            "sudoku-easy": "per-candidate exact, valid-grid, and clue-preservation rates",
        }[task],
        "bootstrap": {
            "method": "paired resampling over input IDs",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "wmt_oracle_metric": (
            "sentence-level chrF; reference-ranked opportunity, not a corpus-level selector"
            if task == "wmt14" else None
        ),
        "k_results": {},
    }

    source_complexities = {
        input_id: _source_complexity(candidate_contexts[input_id], task)
        for input_id in pools
    }
    sorted_inputs = sorted(pools, key=lambda input_id: (source_complexities[input_id], input_id))
    strata = []
    strata_count = min(4, len(sorted_inputs))
    for quartile in range(strata_count):
        start = quartile * len(sorted_inputs) // strata_count
        end = (quartile + 1) * len(sorted_inputs) // strata_count
        group = sorted_inputs[start:end]
        strata.append({
            "quartile": quartile + 1,
            "input_ids": group,
            "complexity_min": min(source_complexities[i] for i in group),
            "complexity_max": max(source_complexities[i] for i in group),
        })

    previous_oracles: dict[str, dict[str, float]] = {}
    previous_k = 0
    for k in ranks:
        prefix_metrics = {
            input_id: [candidate_metrics[row["candidate_id"]] for row in pool[:k]]
            for input_id, pool in pools.items()
        }
        unique_counts = {
            input_id: len({row["text"] for row in pool[:k]})
            for input_id, pool in pools.items()
        }
        first_quality = {
            input_id: candidate_metrics[pool[0]["candidate_id"]]
            for input_id, pool in pools.items()
        }
        greedy_quality = {
            input_id: greedy_metrics[greedy_by_input[input_id][0]["candidate_id"]]
            for input_id in pools
        }
        quality = {}
        for metric_index, name in enumerate(metric_names):
            per_input_sampled = {
                input_id: sum(row[name] for row in values) / len(values)
                for input_id, values in prefix_metrics.items()
            }
            per_input_oracle = {
                input_id: max(row[name] for row in values)
                for input_id, values in prefix_metrics.items()
            }
            improvements = {
                input_id: per_input_oracle[input_id] - first_quality[input_id][name]
                for input_id in pools
            }
            vs_greedy = {
                input_id: per_input_oracle[input_id] - greedy_quality[input_id][name]
                for input_id in pools
            }
            quality[name] = {
                "rank1_mean": sum(first_quality[i][name] for i in pools) / len(pools),
                "mean_sampled": sum(per_input_sampled.values()) / len(pools),
                "oracle_mean": sum(per_input_oracle.values()) / len(pools),
                "marginal_oracle_gain_per_candidate": (
                    (sum(per_input_oracle.values()) / len(pools)
                     - previous_oracles[name]["oracle_mean"]) / (k - previous_k)
                    if previous_k else None
                ),
                "oracle_gain_vs_rank1": _bootstrap_gain(
                    list(improvements.values()), bootstrap_seed + k + metric_index,
                    bootstrap_samples,
                ),
                "improved_over_rank1_fraction": sum(
                    gain > 0 for gain in improvements.values()
                ) / len(pools),
                "oracle_gain_vs_greedy": _bootstrap_gain(
                    list(vs_greedy.values()), bootstrap_seed + 1000 + k + metric_index,
                    bootstrap_samples,
                ),
                "improved_over_greedy_fraction": sum(
                    gain > 0 for gain in vs_greedy.values()
                ) / len(pools),
                "source_complexity_strata": [{
                    "quartile": group["quartile"],
                    "input_count": len(group["input_ids"]),
                    "complexity_min": group["complexity_min"],
                    "complexity_max": group["complexity_max"],
                    "oracle_gain_vs_rank1": sum(
                        improvements[i] for i in group["input_ids"]
                    ) / len(group["input_ids"]),
                    "oracle_gain_vs_greedy": sum(
                        vs_greedy[i] for i in group["input_ids"]
                    ) / len(group["input_ids"]),
                } for group in strata],
            }

        diversity = {
            "prompts_with_alternative": sum(value > 1 for value in unique_counts.values()),
            "mean_unique_candidates": sum(unique_counts.values()) / len(unique_counts),
            "duplicate_candidates": k * len(pools) - sum(unique_counts.values()),
            "duplicate_fraction": (
                k * len(pools) - sum(unique_counts.values())
            ) / (k * len(pools)),
        }
        mean_words = None if task == "sudoku-easy" else sum(
            len(row["text"].split()) for pool in pools.values() for row in pool[:k]
        ) / (len(pools) * k)
        errors = sum(
            row.get("generation_error") is not None
            for pool in pools.values() for row in pool[:k]
        )
        result: dict[str, Any] = {
            "diversity": diversity,
            "mean_output_words": mean_words,
            "candidate_errors": errors,
            "quality": quality,
        }
        if task == "wmt14" and k == 1:
            rank1 = [row for pool in pools.values() for row in pool[:1]]
            rank1_ordered = validate_records(rank1, references, contexts)[1]
            result["rank1_corpus_metrics"] = score_wmt14(rank1_ordered)
        elif task == "xsum" and k == 1:
            rank1 = [row for pool in pools.values() for row in pool[:1]]
            result["rank1_metrics"] = score_xsum(
                validate_records(rank1, references, contexts)[1]
            )
        elif task == "sudoku-easy" and k == 1:
            rank1 = [row for pool in pools.values() for row in pool[:1]]
            result["rank1_metrics"] = score_sudoku(
                validate_records(rank1, references, contexts)[1]
            )
        report["k_results"][str(k)] = result
        previous_oracles = {
            name: {"oracle_mean": values["oracle_mean"]}
            for name, values in quality.items()
        }
        previous_k = k
    report["offline_analysis_seconds"] = time.perf_counter() - scoring_started
    report["total_analysis_wall_seconds"] = time.perf_counter() - analysis_started
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--greedy-run-id", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.bootstrap_seed < 0:
        parser.error("bootstrap samples must be positive and seed nonnegative")
    run_dir = ROOT / "runs" / args.run_id
    greedy_dir = ROOT / "runs" / args.greedy_run_id
    report = analyze_candidate_pool(
        run_dir, greedy_dir, args.bootstrap_samples, args.bootstrap_seed
    )
    report_path = run_dir / "phase1-analysis.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
