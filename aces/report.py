from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from .metrics import paired_bootstrap
from .utils import read_json, write_json, write_csv


def export_report(runs_dir, output_dir, expected_seeds=(17, 29, 43)):
    runs_dir, output_dir = Path(runs_dir), Path(output_dir)
    groups = defaultdict(list)
    for path in runs_dir.rglob("summary.json"):
        if path.parent.name not in {"test", "external"}:
            continue
        s = read_json(path)
        if s.get("synthetic"):
            continue
        s["_path"] = str(path)
        groups[(s["method"], s["label_fraction"], s["split"])].append(s)
    rows = []
    comparisons = []
    for (method, fraction, split), entries in sorted(groups.items()):
        seen = [e["seed"] for e in entries]
        if len(seen) != len(set(seen)):
            raise ValueError(f"Duplicate seed result for {method}/{fraction}/{split}; use one frozen run set")
        row = {"method": method, "label_fraction": fraction, "split": split,
               "completed_seeds": ",".join(map(str, sorted(seen))),
               "expected_seed_set_complete": set(seen) == set(expected_seeds),
               "distance_unit": entries[0]["distance_unit"], "patients": entries[0]["patients"]}
        if len({e["distance_unit"] for e in entries}) != 1 or len({e["patients"] for e in entries}) != 1:
            raise ValueError("Runs disagree on task units or patient counts")
        if len({e["dataset_signature"] for e in entries}) != 1:
            raise ValueError("Different dataset manifests cannot be pooled across seeds")
        for metric in ["dice", "iou", "hd95", "assd", "empty_prediction"]:
            a = [e[metric] for e in entries if e[metric] is not None]
            row[metric+"_mean"] = float(np.mean(a)) if a else None
            row[metric+"_sd"] = float(np.std(a, ddof=1)) if len(a) > 1 else None
        rows.append(row)
    for (method, fraction, split), entries in sorted(groups.items()):
        key = ("aces", fraction, split)
        if method == "aces" or key not in groups:
            continue
        if {e["seed"] for e in entries} != set(expected_seeds) or {e["seed"] for e in groups[key]} != set(expected_seeds):
            continue
        if {e["dataset_signature"] for e in entries} != {e["dataset_signature"] for e in groups[key]}:
            raise ValueError("Paired methods were evaluated on different data manifests")
        def patient_means(items):
            per_seed = []
            for item in items:
                f = Path(item["_path"]).parent/"per_patient.csv"
                with f.open(encoding="utf-8-sig", newline="") as stream:
                    data = {r["patient_id"]: float(r["dice"]) for r in csv.DictReader(stream) if r["dice"]}
                per_seed.append(data)
            if any(set(x) != set(per_seed[0]) for x in per_seed):
                raise ValueError("Patient sets differ across seeds")
            return {p: float(np.mean([d[p] for d in per_seed])) for p in per_seed[0]}
        stats = paired_bootstrap(patient_means(groups[key]), patient_means(entries))
        comparisons.append({"contrast": "aces - "+method, "label_fraction": fraction, "split": split,
                            "mean_difference": stats["mean_difference"], "ci95_low": stats["ci95"][0],
                            "ci95_high": stats["ci95"][1], "patients": stats["patients"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/"results_by_method.csv", rows)
    write_csv(output_dir/"paired_patient_bootstrap.csv", comparisons)
    status = {"empirical_run_groups": len(rows), "expected_seeds": list(expected_seeds),
              "apnet": "not_implemented_as_a_validated_reproduction",
              "synthetic_smoke_results_excluded": True, "rows": rows, "paired_comparisons": comparisons}
    write_json(output_dir/"results.json", status)
    text = "# ACES 实际运行结果\n\n仅从真实数据运行目录汇总；不包含合成自检。单种子不能报告为三种子标准差。\n\n"
    if rows:
        text += "| 方法 | 标签比例 | 集合 | 已完成种子 | Dice 均值 | 种子 SD |\n|---|---|---|---|---:|---:|\n"
        for r in rows:
            sd = "未满两个种子" if r["dice_sd"] is None else f"{r['dice_sd']:.5f}"
            text += f"| {r['method']} | {r['label_fraction']:.0%} | {r['split']} | {r['completed_seeds']} | {r['dice_mean']:.5f} | {sd} |\n"
    else:
        text += "尚无真实数据结果。\n"
    text += "\nAP-Net 尚未完成可核验复现，不填其性能。未准备或未确认 HMC-QU 时，外部验证为未运行，不以零分或模拟结果代替。完整的原研究方案仍需补齐该文献基线。\n"
    (output_dir/"RESULTS.md").write_text(text, encoding="utf-8")
    return status
