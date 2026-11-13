import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_phase1 import analyze_candidate_pool, bootstrap_mean_interval


SOLUTION = "123456789|456789123|789123456|234567891|567891234|891234567|345678912|678912345|912345678"
ALTERNATE = SOLUTION.translate(str.maketrans("12", "21"))


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def sudoku_context(input_id):
    cells = [0] * 81
    cells[0] = 1
    tokens = [11]
    for index, cell in enumerate(cells):
        tokens.append(cell)
        if index % 9 == 8 and index < 80:
            tokens.append(10)
    tokens.append(11)
    example_id = f"sudoku-easy:test:{input_id}"
    return {
        "task": "sudoku-easy",
        "input_id": input_id,
        "example_id": example_id,
        "source_token_ids": tokens,
    }


class Phase1AnalysisTests(unittest.TestCase):
    def test_bootstrap_interval_is_deterministic(self):
        values = [0.0, 1.0, 2.0]
        self.assertEqual(
            bootstrap_mean_interval(values, seed=3, samples=100),
            bootstrap_mean_interval(values, seed=3, samples=100),
        )

    def test_analysis_uses_nested_pools_and_paired_greedy_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, greedy = root / "candidates", root / "greedy"
            run.mkdir()
            greedy.mkdir()
            contexts = [sudoku_context(i) for i in range(2)]
            references = [{
                "task": "sudoku-easy",
                "split": "test",
                "input_id": i,
                "example_id": f"sudoku-easy:test:{i}",
                "reference_id": f"sudoku-easy:test:{i}",
                "text": SOLUTION,
            } for i in range(2)]
            candidates = []
            greedy_candidates = []
            for input_id in range(2):
                example_id = f"sudoku-easy:test:{input_id}"
                for rank in range(4):
                    text = ALTERNATE if input_id == 0 and rank > 0 else SOLUTION
                    candidates.append({
                        "candidate_id": f"{example_id}:candidate:{rank}",
                        "task": "sudoku-easy",
                        "split": "test",
                        "input_id": input_id,
                        "example_id": example_id,
                        "candidate_rank": rank,
                        "text": text,
                        "generation_duration_seconds": 0.1,
                        "generation_error": None,
                    })
                greedy_candidates.append({
                    "candidate_id": f"{example_id}:candidate:0",
                    "task": "sudoku-easy",
                    "input_id": input_id,
                    "example_id": example_id,
                    "candidate_rank": 0,
                    "text": SOLUTION,
                })

            for directory, rows in (
                (run, candidates),
                (greedy, greedy_candidates),
            ):
                write_jsonl(directory / "candidates.jsonl", rows)
                write_jsonl(directory / "contexts.jsonl", contexts)
                write_jsonl(directory / "references.jsonl", references)
            (run / "manifest.json").write_text(json.dumps({
                "run_id": "candidate-run",
                "status": "succeeded",
                "started_utc": "2026-11-13T00:00:00+00:00",
                "ended_utc": "2026-11-13T00:00:02+00:00",
                "task": "sudoku-easy",
                "split": "test",
                "k": 4,
                "input_ids": [0, 1],
                "inputs_sha256": "inputs",
                "config_sha256": "config",
                "upstream_commit": "commit",
                "candidate_errors": 0,
                "generation": {},
                "timings": {"generation_seconds": 0.8},
            }), encoding="utf-8")
            (greedy / "manifest.json").write_text(json.dumps({
                "run_id": "greedy-run",
                "status": "succeeded",
                "task": "sudoku-easy",
                "split": "test",
                "greedy_baseline": True,
            }), encoding="utf-8")

            result = analyze_candidate_pool(run, greedy, bootstrap_samples=100)

        self.assertEqual(result["input_count"], 2)
        self.assertEqual(result["candidate_errors"], 0)
        self.assertEqual(result["k_results"]["1"]["quality"]["exact_solution_rate"]["rank1_mean"], 1)
        k4 = result["k_results"]["4"]
        self.assertEqual(k4["diversity"]["prompts_with_alternative"], 1)
        self.assertEqual(k4["quality"]["exact_solution_rate"]["oracle_mean"], 1)
        self.assertEqual(k4["quality"]["exact_solution_rate"]["oracle_gain_vs_greedy"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
