import pickle
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline import attach_references, digest, normalize, repository_snapshot, stage_generation_data
from rescore_baseline import validate_generation_checkpoint


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
        self.assertEqual([x["example_id"] for x in candidates], ["xsum:test:0", "xsum:test:1"])
        self.assertEqual([x["candidate_id"] for x in candidates],
                         ["xsum:test:0:candidate:0", "xsum:test:1:candidate:0"])
        self.assertEqual(contexts[0]["example_id"], candidates[0]["example_id"])
        self.assertEqual(references[0]["example_id"], candidates[0]["example_id"])
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

    def test_staged_inputs_have_only_placeholder_references(self):
        self.source.write_bytes(pickle.dumps(self.sources))
        stage = self.source.parent / "stage"
        stage_generation_data(self.source, stage, "inputs.pkl", "refs.pkl", 2)
        with (stage / "inputs.pkl").open("rb") as f:
            self.assertEqual(pickle.load(f), self.sources)
        with (stage / "refs.pkl").open("rb") as f:
            self.assertEqual(pickle.load(f), ["", ""])

    def test_attach_references_preserves_generation_shard(self):
        self.result["refs"] = ["", ""]
        self.shard.write_bytes(pickle.dumps(self.result))
        refs = self.source.parent / "refs.pkl"
        refs.write_bytes(pickle.dumps(["r1", "r2"]))
        output = self.source.parent / "scoring.pkl"
        attach_references(self.shard, refs, output, 2, "xsum")
        with output.open("rb") as f:
            self.assertEqual(pickle.load(f)["refs"], ["r1", "r2"])
        with self.shard.open("rb") as f:
            self.assertEqual(pickle.load(f)["refs"], ["", ""])

    def test_rejects_real_references_in_generation_shard(self):
        self.shard.write_bytes(pickle.dumps(self.result))
        refs = self.source.parent / "refs.pkl"
        refs.write_bytes(pickle.dumps(["r1", "r2"]))
        with self.assertRaises(ValueError):
            attach_references(self.shard, refs, self.source.parent / "output.pkl", 2, "xsum")

    def test_sudoku_references_use_upstream_decoding(self):
        self.result["refs"] = ["", ""]
        self.shard.write_bytes(pickle.dumps(self.result))
        refs = self.source.parent / "refs.pkl"
        refs.write_bytes(pickle.dumps([[11, 1, 10, 2], [11, 3, 10, 4]]))
        output = self.source.parent / "scoring.pkl"
        attach_references(self.shard, refs, output, 2, "sudoku-easy")
        with output.open("rb") as f:
            self.assertEqual(pickle.load(f)["refs"], ["1|2", "3|4"])

    def test_rejects_same_input_and_reference_filename(self):
        self.source.write_bytes(pickle.dumps(self.sources))
        with self.assertRaises(ValueError):
            stage_generation_data(self.source, self.source.parent / "stage", "same", "same", 2)

    def test_batch_times_are_shared_not_per_candidate_latency(self):
        self.shard.write_bytes(pickle.dumps(self.result))
        self.source.write_bytes(pickle.dumps(self.sources))
        candidates, _, _ = normalize(self.shard, self.source, 2, "xsum", [1.5], 2)
        self.assertEqual([x["generation_batch_duration_seconds"] for x in candidates], [1.5, 1.5])
        self.assertEqual([x["generation_batch_size"] for x in candidates], [2, 2])
        self.assertTrue(all(x["generation_duration_seconds"] is None for x in candidates))
        with self.assertRaises(ValueError):
            normalize(self.shard, self.source, 2, "xsum", [], 2)

    def test_checkpoint_detects_changed_outputs_and_spec(self):
        root = self.source.parent
        spec = {"schema_version": 2, "run_id": "test", "task": "xsum", "upstream_commit": "sha",
                "config_sha256": "sha", "reference_model": None, "input_ids": [0],
                "batch_size": 1, "generation": {}, "adapter_sha256": "sha", "profiler_sha256": "sha"}
        (root / "run-spec.json").write_text(json.dumps(spec), encoding="utf-8")
        names = ("generation_shard.pkl", "shard_0.pkl", "candidates.jsonl", "contexts.jsonl",
                 "references.jsonl", "generation-timing.json", "input-order.jsonl")
        for name in names:
            (root / name).write_text("data", encoding="utf-8")
        hashes = {name: digest(root / name) for name in (*names, "run-spec.json")}
        (root / "generation-checkpoint.json").write_text(json.dumps({"sha256": hashes}), encoding="utf-8")
        manifest = {**spec, "run_spec_sha256": digest(root / "run-spec.json")}
        validate_generation_checkpoint(root, manifest)
        with self.assertRaisesRegex(ValueError, "specification mismatch"):
            validate_generation_checkpoint(root, {**manifest, "batch_size": 2})
        (root / "candidates.jsonl").write_text("corrupt", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_generation_checkpoint(root, manifest)

    def test_legacy_generation_checkpoint_remains_rescorable(self):
        root = self.source.parent / "legacy"
        root.mkdir()
        spec = {"run_id": "legacy", "task": "xsum", "upstream_commit": "sha",
                "config_sha256": "sha", "reference_model": None, "input_ids": [0],
                "batch_size": 1, "generation": {}, "adapter_sha256": "sha", "profiler_sha256": "sha"}
        (root / "run-spec.json").write_text(json.dumps(spec), encoding="utf-8")
        names = ("generation_shard.pkl", "shard_0.pkl", "candidates.jsonl", "contexts.jsonl",
                 "references.jsonl", "generation-timing.json")
        for name in names:
            (root / name).write_text("data", encoding="utf-8")
        hashes = {name: digest(root / name) for name in (*names, "run-spec.json")}
        (root / "generation-checkpoint.json").write_text(json.dumps({"sha256": hashes}), encoding="utf-8")
        validate_generation_checkpoint(root, spec)

    def test_repository_snapshot_records_git_and_worktree_hashes(self):
        snapshot = repository_snapshot()
        self.assertTrue(snapshot["head"])
        self.assertEqual(len(snapshot["diff_head_sha256"]), 64)
        self.assertIsInstance(snapshot["untracked_files_sha256"], dict)


if __name__ == "__main__":
    unittest.main()
