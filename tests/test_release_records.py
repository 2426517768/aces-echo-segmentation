"""Checks on the archived experiment cohort and source identity."""
from pathlib import Path
import hashlib
import json
import unittest
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class ArchivedExperimentTests(unittest.TestCase):
    def test_cohort_is_disjoint_and_label_budget_is_fixed(self):
        splits = read(ROOT/"data/splits/camus_patients.json")
        train, val, test = [set(splits[k]) for k in ("train", "val", "test")]
        self.assertEqual(tuple(map(len, (train, val, test))), (400, 50, 50))
        self.assertFalse(train & val or train & test or val & test)
        budgets = read(ROOT/"data/splits/annotation_budgets.json")
        self.assertEqual(len(budgets["25"]), 100)
        self.assertTrue(set(budgets["25"]) <= train)
        for path in (ROOT/"results/archived_runs").glob("*/annotation_budget.json"):
            ids = read(path)["labeled_training_patients"]
            fraction = read(path.parent/"contract.json")["label_fraction"]
            self.assertEqual(ids, budgets["25" if fraction == 0.25 else "100"])

    def test_all_configurations_have_the_same_three_test_patient_sets(self):
        grouped = defaultdict(set)
        for path in (ROOT/"results/archived_runs").glob("*/test/summary.json"):
            summary = read(path)
            grouped[(summary["method"], summary["label_fraction"])].add(summary["seed"])
            self.assertEqual((summary["patients"], summary["frames"]), (50, 200))
            self.assertFalse(summary["synthetic"])
        self.assertEqual(len(grouped), 11)
        self.assertTrue(all(seeds == {17, 29, 43} for seeds in grouped.values()))

    def test_training_core_matches_recorded_v14_source_hashes(self):
        manifest = read(ROOT/"docs/recorded_code_manifest.json")
        selected = [entry for entry in manifest["source_files"]
                    if entry["path"].startswith("aces/") and entry["path"].endswith(".py")]
        self.assertEqual(len(selected), 12)
        for entry in selected:
            self.assertEqual(hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest(),
                             entry["sha256"], entry["path"])


if __name__ == "__main__":
    unittest.main()
