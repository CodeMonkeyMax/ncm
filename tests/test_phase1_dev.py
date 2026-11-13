import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_phase1_dev import count_solutions, sudoku_examples
from sample_candidates import candidate_seed, context_for, load_input_rows, load_reference_rows


class SudokuDevelopmentDataTests(unittest.TestCase):
    def test_examples_are_deterministic_unique_and_match_easy_clue_count(self):
        inputs, references = sudoku_examples(3, 20261112)
        repeated_inputs, repeated_references = sudoku_examples(3, 20261112)
        self.assertEqual(inputs, repeated_inputs)
        self.assertEqual(references, repeated_references)
        self.assertEqual(len(inputs), len(references))
        self.assertEqual(len({tuple(row["source"]) for row in inputs}), len(inputs))

        for source, reference in zip(inputs, references):
            encoded = source["source"]
            self.assertEqual(len(encoded), 91)
            self.assertEqual(encoded[0], 11)
            self.assertEqual(encoded[-1], 11)
            puzzle = [value for value in encoded[1:-1] if value != 10]
            solution = [int(value) for value in reference["text"].replace("|", "")]
            self.assertEqual(sum(value != 0 for value in puzzle), 40)
            self.assertEqual(count_solutions(puzzle.copy()), 1)
            self.assertEqual(len(solution), 81)
            self.assertEqual(count_solutions(solution.copy()), 1)
            self.assertTrue(all(not clue or clue == solution[index]
                                for index, clue in enumerate(puzzle)))


class CandidateSeedTests(unittest.TestCase):
    def test_seed_is_stable_and_distinct_across_candidate_identity(self):
        first = candidate_seed(17, "xsum", 3, 0, 4)
        self.assertEqual(first, candidate_seed(17, "xsum", 3, 0, 4))
        self.assertNotEqual(first, candidate_seed(17, "xsum", 3, 0, 5))
        self.assertNotEqual(first, candidate_seed(17, "xsum", 4, 0, 4))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1 << 63)


if __name__ == "__main__":
    unittest.main()
