"""Check immutable release files, fixed patient partitions and run completeness."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify(root=ROOT):
    root = Path(root).resolve()
    inventory = read(root/"FILE_MANIFEST.json")
    for entry in inventory["files"]:
        path = root/entry["path"]
        if not path.is_file():
            raise ValueError(f"Missing release file: {entry['path']}")
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Release file changed: {entry['path']}")
    split_lists = read(root/"data/splits/camus_patients.json")
    splits = {k: set(v) for k, v in split_lists.items()}
    if {k: len(v) for k, v in splits.items()} != {"train": 400, "val": 50, "test": 50}:
        raise ValueError("Expected recorded 400/50/50 partition")
    if any(len(v) != len(splits[k]) for k, v in split_lists.items()):
        raise ValueError("Duplicate patient ID in a partition")
    if splits["train"] & splits["val"] or splits["train"] & splits["test"] or splits["val"] & splits["test"]:
        raise ValueError("Patient split leakage")
    budgets = read(root/"data/splits/annotation_budgets.json")
    if len(budgets["25"]) != 100 or set(budgets["100"]) != splits["train"]:
        raise ValueError("Unexpected patient annotation budgets")
    if not set(budgets["25"]) <= splits["train"]:
        raise ValueError("Visible labels include a held-out patient")
    plan = read(root/"results/published/planned_jobs.json")["jobs"]
    names = {f"{j['method']}_labels{round(j['label_fraction']*100):03d}_seed{j['seed']}" for j in plan}
    archived = root/"results/archived_runs"
    actual = {p.name for p in archived.iterdir() if p.is_dir()}
    if len(names) != 33 or names != actual:
        raise ValueError("Recorded run set differs from the 33-job plan")
    for name in sorted(names):
        contract = read(archived/name/"contract.json")
        expected_labels = budgets["25" if contract["label_fraction"] == 0.25 else "100"]
        if contract["labeled_patients"] != expected_labels:
            raise ValueError(f"Annotation subset changed: {name}")
        if contract["manifest_hash"] != inventory["recorded_manifest_digest"]:
            raise ValueError(f"Historical manifest identity changed: {name}")
    result = {"status": "passed", "files_checked": len(inventory["files"]),
              "patients": {k: len(v) for k, v in splits.items()}, "runs": len(names)}
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
