import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline import normalize


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.shard = root / "shard.pkl"
        self.source = root / "sources.pkl"
        self.sources = [[1, 2], [3, 4]]
        self.result = {"idxs": [0, 1], "hyps": ["a", "b"], "refs": ["r1", "r2"]}

    def normalize(self):
        self.shard.write_bytes(pickle.dumps(self.result))
        self.source.write_bytes(pickle.dumps(self.sources))
        return normalize(self.shard, self.source, 2, "xsum")

    def test_records_are_separated_and_ordered(self):
        candidates, contexts, references = self.normalize()
        self.assertEqual([x["input_id"] for x in candidates], [0, 1])
        self.assertEqual(contexts[1]["source_token_ids"], [3, 4])
        self.assertEqual(references[0]["text"], "r1")
        self.assertNotIn("text", contexts[0])
        self.assertNotIn("reference", candidates[0])

    def test_rejects_mismatched_counts(self):
        self.result["refs"].pop()
        with self.assertRaises(ValueError):
            self.normalize()

    def test_rejects_duplicate_or_out_of_order_ids(self):
        for ids in ([0, 0], [1, 0]):
            self.result["idxs"] = ids
            with self.assertRaises(ValueError):
                self.normalize()

    def test_rejects_empty_generation(self):
        self.result["hyps"][0] = " "
        with self.assertRaises(ValueError):
            self.normalize()

    def test_rejects_missing_sources(self):
        self.sources.pop()
        with self.assertRaises(ValueError):
            self.normalize()


if __name__ == "__main__":
    unittest.main()
