import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics import score_sudoku, score_wmt14, score_xsum, validate_records
import sacrebleu
from rouge_score import rouge_scorer


SOLUTION = "123456789|456789123|789123456|234567891|567891234|891234567|345678912|678912345|912345678"
ALTERNATE = SOLUTION.translate(str.maketrans("12", "21"))


def sudoku_context(clue=1):
    cells = [0] * 81
    cells[0] = clue
    tokens = [11]
    for index, cell in enumerate(cells):
        tokens.append(cell)
        if index % 9 == 8 and index < 80:
            tokens.append(10)
    tokens.append(11)
    return tokens


def records(task="sudoku-easy", generations=None, references=None, contexts=None):
    generations = generations or [SOLUTION]
    references = references or [SOLUTION] * len(generations)
    contexts = contexts or [sudoku_context()] * len(generations)
    candidates = []
    refs = []
    ctxs = []
    for index, (text, reference, context) in enumerate(zip(generations, references, contexts)):
        example_id = f"{task}:test:{index}"
        candidates.append({
            "candidate_id": f"{example_id}:candidate:0",
            "task": task,
            "input_id": index,
            "example_id": example_id,
            "candidate_rank": 0,
            "text": text,
        })
        refs.append({
            "task": task,
            "input_id": index,
            "example_id": example_id,
            "reference_id": example_id,
            "text": reference,
        })
        ctxs.append({
            "task": task,
            "input_id": index,
            "example_id": example_id,
            "source_token_ids": context,
        })
    return candidates, refs, ctxs


class SudokuMetricTests(unittest.TestCase):
    def test_perfect_grid_is_valid_and_preserves_clue(self):
        joined = validate_records(*records())
        result = score_sudoku(joined[1])
        self.assertEqual(result["exact_solution_rate"], 1)
        self.assertEqual(result["valid_grid_rate"], 1)
        self.assertEqual(result["clue_preservation_rate"], 1)
        self.assertEqual(result["mean_cell_accuracy"], 1)

    def test_valid_grid_that_violates_clue_is_not_accepted(self):
        joined = validate_records(*records(generations=[ALTERNATE]))
        result = score_sudoku(joined[1])
        self.assertTrue(result["per_candidate"][0]["valid_grid"])
        self.assertFalse(result["per_candidate"][0]["preserves_clues"])

    def test_malformed_and_partially_correct_grids_are_scored_without_crashing(self):
        partial = SOLUTION.replace("123456789", "023456789", 1)
        malformed = SOLUTION[:-1]
        joined = validate_records(*records(generations=[partial, malformed]))
        result = score_sudoku(joined[1])
        self.assertEqual(result["valid_grid_rate"], 0)
        self.assertEqual(result["mean_cell_accuracy"], (80 / 81 + 80 / 81) / 2)
        self.assertFalse(result["per_candidate"][0]["preserves_clues"])
        self.assertFalse(result["per_candidate"][1]["well_formed"])

    def test_failed_generation_is_retained_and_scored_as_invalid(self):
        candidates, refs, contexts = records()
        candidates[0]["text"] = ""
        candidates[0]["generation_error"] = "RuntimeError: simulated failure"
        result = score_sudoku(validate_records(candidates, refs, contexts)[1])
        self.assertFalse(result["per_candidate"][0]["well_formed"])
        self.assertEqual(result["exact_solution_rate"], 0)


class TextMetricTests(unittest.TestCase):
    def test_xsum_matches_upstream_rouge_fmeasure_convention(self):
        hypotheses = ["A cat runs.", "The quick brown fox."]
        references = ["A cat ran.", "A quick fox."]
        candidates, refs, contexts = records(
            "xsum", hypotheses, references, [{} for _ in hypotheses]
        )
        result = score_xsum(validate_records(candidates, refs, contexts)[1])
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        expected = {
            name: 100 * sum(
                scorer.score(reference, hypothesis)[name].fmeasure
                for hypothesis, reference in zip(hypotheses, references)
            ) / len(hypotheses)
            for name in ("rouge1", "rouge2", "rougeL")
        }
        self.assertEqual(result["mean_rouge_fmeasure_percent"], expected)

    def test_failed_empty_generation_is_retained_and_scores_zero(self):
        candidates, refs, contexts = records(
            "xsum", [""], ["a reference summary"], [{}]
        )
        candidates[0]["generation_error"] = "EmptyGeneration: model produced no text"
        result = score_xsum(validate_records(candidates, refs, contexts)[1])
        self.assertEqual(result["mean_rouge_fmeasure_percent"]["rouge1"], 0)
        self.assertEqual(result["mean_rouge_fmeasure_percent"]["rougeL"], 0)

    def test_wmt_matches_upstream_corpus_metrics_and_signature(self):
        hypotheses = ["Das ist ein Test .", "Ein weiterer Satz ."]
        references = ["Das ist ein Test.", "Ein weiterer Satz."]
        candidates, refs, contexts = records(
            "wmt14", hypotheses, references, [{} for _ in hypotheses]
        )
        result = score_wmt14(validate_records(candidates, refs, contexts)[1])
        bleu_metric = sacrebleu.metrics.BLEU()
        bleu_metric.corpus_score(hypotheses, [references])
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        lowercase_bleu = sacrebleu.corpus_bleu(
            hypotheses, [references], lowercase=True
        )
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        self.assertEqual(result["corpus_bleu"], bleu.score)
        self.assertEqual(result["corpus_bleu_lowercase"], lowercase_bleu.score)
        self.assertEqual(result["corpus_chrf"], chrf.score)
        self.assertEqual(result["sacrebleu_signature"], str(bleu_metric.get_signature()))
        self.assertEqual(result["length_ratio"], 9 / 7)


class RecordValidationTests(unittest.TestCase):
    def test_out_of_order_records_are_joined_and_sorted_by_input(self):
        candidates, references, contexts = records(
            generations=[SOLUTION, SOLUTION], references=[SOLUTION, SOLUTION]
        )
        for rows in (candidates, references, contexts):
            rows.reverse()
        _, ordered = validate_records(candidates, references, contexts)
        self.assertEqual([item[0]["input_id"] for item in ordered], [0, 1])

    def test_legacy_records_get_stable_ids_from_task_and_input(self):
        candidates, references, contexts = records()
        for row in candidates:
            row.pop("candidate_id")
            row.pop("example_id")
        for row in references:
            row.pop("example_id")
        for row in contexts:
            row.pop("example_id")
        _, ordered = validate_records(candidates, references, contexts)
        self.assertEqual(ordered[0][0]["example_id"], "sudoku-easy:test:0")
        self.assertEqual(
            ordered[0][0]["candidate_id"], "sudoku-easy:test:0:candidate:0"
        )

    def test_rejects_duplicate_candidate_ids(self):
        candidates, references, contexts = records()
        candidates.append(candidates[0].copy())
        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            validate_records(candidates, references, contexts)

    def test_rejects_missing_references(self):
        candidates, references, contexts = records()
        references.clear()
        with self.assertRaisesRegex(ValueError, "candidate/reference example mismatch"):
            validate_records(candidates, references, contexts)

    def test_rejects_empty_generations(self):
        candidates, references, contexts = records(generations=["  "])
        with self.assertRaisesRegex(ValueError, "empty generation"):
            validate_records(candidates, references, contexts)

    def test_rejects_unequal_pool_sizes(self):
        candidates, references, contexts = records(
            generations=[SOLUTION, SOLUTION], references=[SOLUTION, SOLUTION]
        )
        candidates.append({
            **candidates[0],
            "candidate_id": "extra",
            "candidate_rank": 1,
        })
        with self.assertRaisesRegex(ValueError, "unequal candidate pool sizes"):
            validate_records(candidates, references, contexts)


if __name__ == "__main__":
    unittest.main()
