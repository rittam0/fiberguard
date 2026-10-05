from pathlib import Path
import tempfile
import unittest

from fiberguard.data import discover_raw_files, iter_chunks, iter_rows
from fiberguard.entities import scan_and_collect, validate_paired_scans
from fiberguard.reliability import abstention_metrics, multiclass_brier_score, select_review_threshold
from fiberguard.split import assert_disjoint, split_lightpaths

PREAMBLE = """failure_description = \"'0' : No failure, '1' : ECL failure, '2' : EDFA failure, '3' : NLI failure\"\n\"Time stamp\" \"LP length (km)\" \"Laser current (mA)\" \"LP power (dBm)\" \"OSNR (dB)\" \"BER (dB)\" \"Failure type\"\n"""


class DataLoaderTests(unittest.TestCase):
    def test_discovery_parsing_and_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = PREAMBLE + "1 514 40.1 -2.0 24.7 -267.3 0\n2 514 40.2 -1.9 24.8 -267.2 1\n"
            (root / "sample_train_900.txt").write_text(content, encoding="ascii")
            (root / "sample_test_300.txt").write_text(content, encoding="ascii")
            files = discover_raw_files(root)
            self.assertEqual(list(iter_rows(files["train"]))[0], (1, 514.0, 40.1, -2.0, 24.7, -267.3, 0))
            self.assertEqual([len(chunk) for chunk in iter_chunks(files["test"], 1)], [1, 1])

    def test_schema_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("failure_description = x\n\"wrong\"\n1\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                list(iter_rows(path))

    def test_entity_validation_and_disjoint_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, block_rows in (("train", 2), ("test", 1)):
                rows = []
                for entity, length in enumerate((514, 799)):
                    for label in range(4):
                        for timestamp in range(1, block_rows + 1):
                            rows.append(f"{timestamp} {length} 40 -2 24 -267 {label}\n")
                path = root / f"sample_{name}.txt"
                path.write_text(PREAMBLE + "".join(rows), encoding="ascii")
                paths.append(path)
            train = scan_and_collect(paths[0], 2, expected_lightpaths=2)
            test = scan_and_collect(paths[1], 1, expected_lightpaths=2)
            evidence = validate_paired_scans(train, test, 2, 1, 2)
            self.assertTrue(evidence["valid"])
        split = split_lightpaths(count=20, seed=42)
        assert_disjoint(split, expected_count=20)
        self.assertEqual({key: len(value) for key, value in split.items()}, {"train": 14, "validation": 3, "final_test": 3})

    def test_reliability_and_abstention_metrics(self):
        import numpy as np

        y = np.array([0, 1, 2, 3])
        probabilities = np.array([
            [0.9, 0.05, 0.03, 0.02],
            [0.4, 0.5, 0.05, 0.05],
            [0.4, 0.1, 0.45, 0.05],
            [0.1, 0.1, 0.1, 0.7],
        ])
        self.assertGreater(multiclass_brier_score(y, probabilities), 0)
        threshold, _ = select_review_threshold(y, probabilities, max_review_rate=0.75)
        result = abstention_metrics(y, probabilities, threshold)
        self.assertGreaterEqual(result["confident_accuracy"], 0.0)

    def test_split_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint({"train": [0], "validation": [0], "final_test": [1]}, 2)

    def test_raw_review_boundary_and_error_accounting(self):
        import numpy as np
        y = np.array([0, 1, 2])
        p = np.array([[0.99, 0.01, 0, 0], [0.6, 0.4, 0, 0], [0, 0, 1, 0]])
        result = abstention_metrics(y, p, 0.99)
        self.assertEqual(result["review_count"], 1)
        self.assertEqual(result["wrong_predictions_flagged"], 1)
        self.assertEqual(result["remaining_wrong_confident_predictions"], 0)
        self.assertEqual(result["confident_accuracy"], 1)

    def test_production_artifacts_replay_drift_and_operator_view(self):
        import json
        ROOT = Path(__file__).resolve().parents[1]

        manifest = json.loads((ROOT / "artifacts" / "production_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_name"], "fiberguard-fault-classifier")
        self.assertEqual(manifest["model_version"], "1")
        self.assertEqual(manifest["raw_review_threshold"], 0.99)
        replay = json.loads((ROOT / "artifacts" / "historical_replay.json").read_text(encoding="utf-8"))
        self.assertEqual(replay["summary"]["rows"], 4560)
        self.assertEqual(replay["summary"]["lightpaths"], 114)
        self.assertEqual(set(replay["summary"]["class_counts"].values()), {1140})

        validation = json.loads((ROOT / "artifacts" / "drift_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(validation["representative_sample"]["rows"], 4560)
        self.assertTrue(validation["controlled_drift_injection"]["materially_higher"])
        self.assertFalse(validation["controlled_drift_injection"]["labels_modified"])
        operator_view = (ROOT / "src" / "fiberguard" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Network Failure Intelligence", operator_view)
        self.assertIn("Historical telemetry replay", operator_view)


if __name__ == "__main__":
    unittest.main()
