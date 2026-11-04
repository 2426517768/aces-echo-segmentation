"""Recompute tables and paired intervals from archived metrics; no images or GPU."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
METRICS = ("dice", "iou", "hd95", "assd", "empty_prediction")
METHODS25 = ("unet", "mean_teacher", "segformer_b0", "graph_supervised",
             "graph_confidence", "graph_gaussian", "aces_view_agnostic", "aces")
METHODS100 = ("unet", "mean_teacher", "aces")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equal(actual, expected, where, tolerance=1e-10):
    if not math.isfinite(float(actual)) or abs(float(actual)-float(expected)) > tolerance:
        raise ValueError(f"Record mismatch at {where}: {actual!r} != {expected!r}")


def mean(values):
    return float(np.mean(list(values)))


def sd(values):
    return float(np.std(list(values), ddof=1))


def aggregate_frames(rows):
    """Match source aggregation: frames/recordings, then classes, then patients."""
    recordings = defaultdict(list)
    for row in rows:
        recordings[(row["patient_id"], row["class_id"], row["recording_id"])].append(row)
    classes = defaultdict(list)
    for (patient, cls, _), frames in recordings.items():
        classes[(patient, cls)].append({m: mean(float(r[m]) for r in frames) for m in METRICS})
    class_values = {key: {m: mean(r[m] for r in recs) for m in METRICS}
                    for key, recs in classes.items()}
    patients = defaultdict(list)
    for (patient, _), values in class_values.items():
        patients[patient].append(values)
    patient_values = {key: {m: mean(r[m] for r in cls) for m in METRICS}
                      for key, cls in patients.items()}
    macro = {m: mean(p[m] for p in patient_values.values()) for m in METRICS}
    return macro, patient_values, class_values


def bootstrap(left, right, repetitions=2000):
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparisons require identical, nonempty patient sets")
    values = np.array([left[p]-right[p] for p in sorted(left)])
    rng = np.random.default_rng(20261029)
    draws = np.array([rng.choice(values, len(values), replace=True).mean()
                      for _ in range(repetitions)])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean_difference": float(values.mean()), "ci95_low": float(low),
            "ci95_high": float(high), "patients": len(values)}


def reproduce(records, output):
    records, output = Path(records).resolve(), Path(output).resolve()
    if output.is_relative_to(records) or output == ROOT/"results/published":
        raise ValueError("Write regenerated outputs outside archived records/published results")
    runs, grouped, patient_maps = {}, defaultdict(list), {}
    diagnostics = []
    for fraction, methods in [(0.25, METHODS25), (1.0, METHODS100)]:
        for method in methods:
            for seed in SEEDS:
                name = f"{method}_labels{round(fraction*100):03d}_seed{seed}"
                folder = records/name
                summary = read_json(folder/"test/summary.json")
                contract = read_json(folder/"contract.json")
                if summary.get("synthetic") is not False or contract["method"] != method:
                    raise ValueError(f"Unexpected data/task identity: {name}")
                if (summary["seed"], summary["label_fraction"], summary["split"]) != (seed, fraction, "test"):
                    raise ValueError(f"Unexpected run identity: {name}")
                frames = read_csv(folder/"test/per_frame_class.csv")
                if len(frames) != 600 or len({(r["case_id"], r["class_id"]) for r in frames}) != 600:
                    raise ValueError(f"Expected 600 unique frame/class rows: {name}")
                macro, patients, classes = aggregate_frames(frames)
                if len(patients) != 50 or len(classes) != 150:
                    raise ValueError(f"Unexpected patient/class counts: {name}")
                for metric in METRICS:
                    equal(macro[metric], summary[metric], f"{name}/{metric}")
                saved_patients = read_csv(folder/"test/per_patient.csv")
                if len(saved_patients) != 50:
                    raise ValueError(f"Expected 50 patient rows: {name}")
                for row in saved_patients:
                    for metric in METRICS:
                        equal(patients[row["patient_id"]][metric], row[metric],
                              f"{name}/{row['patient_id']}/{metric}")
                for row in read_csv(folder/"test/per_patient_class.csv"):
                    for metric in METRICS:
                        equal(classes[(row["patient_id"], row["class_id"])][metric], row[metric],
                              f"{name}/{row['patient_id']}/{row['class_id']}/{metric}")
                strata = {}
                for row in read_csv(folder/"test/stratified.csv"):
                    subset = [r for r in frames if r[row["stratum"]] == row["value"]]
                    sub_macro, sub_patients, _ = aggregate_frames(subset)
                    for metric in METRICS:
                        equal(sub_macro[metric], row[metric], f"{name}/{row['stratum']}/{row['value']}/{metric}")
                    equal(len(sub_patients), row["patients"], f"{name}/stratum patients")
                    strata[(row["stratum"], row["value"])] = row
                for mode in ("raw", "source_temperature"):
                    calibration = summary["calibration"][mode]
                    ece = sum(abs(c-a) for c, a in zip(calibration["bin_confidence_mass"],
                                                      calibration["bin_correct_mass"]))
                    equal(ece, calibration["ece"], f"{name}/{mode}/ECE")
                history = read_csv(folder/"history.csv")
                if [int(row["epoch"]) for row in history] != list(range(1, 201)):
                    raise ValueError(f"Expected 200 consecutive epochs: {name}")
                accepted = mean(float(row["accepted_pixel_fraction"])
                                for row in history if int(row["epoch"]) >= 21)
                class_dice = {cls: mean(v["dice"] for (p, c), v in classes.items() if c == str(cls))
                              for cls in (1, 2, 3)}
                entry = {"name": name, "summary": summary, "patients": patients,
                         "class_dice": class_dice, "strata": strata, "accepted": accepted,
                         "history": history}
                runs[name] = entry
                grouped[(method, fraction)].append(entry)
                patient_maps[(method, fraction, seed)] = {p: v["dice"] for p, v in patients.items()}
                diagnostics.append({"run": name, "method": method, "label_fraction": fraction,
                                    "seed": seed, **macro, "accepted_epoch21_200": accepted})

    def patient_mean(method, fraction):
        sets = [patient_maps[(method, fraction, s)] for s in SEEDS]
        if any(set(s) != set(sets[0]) for s in sets[1:]):
            raise ValueError("Patient sets differ across seeds")
        return {p: mean(s[p] for s in sets) for p in sets[0]}

    main_table = []
    for method in ("unet", "mean_teacher", "segformer_b0", "aces"):
        group = grouped[(method, 0.25)]
        row = {"method": method, "label_fraction": 0.25}
        for metric in METRICS:
            values = [e["summary"][metric] for e in group]
            row[metric+"_mean"], row[metric+"_sd"] = mean(values), sd(values)
        for cls, label in [(1, "lv_cavity"), (2, "myocardium"), (3, "left_atrium")]:
            row[label+"_dice"] = mean(e["class_dice"][cls] for e in group)
        main_table.append(row)
    ablations = []
    filters = {"aces": ("yes", "yes", "confidence + Student-t"),
               "aces_view_agnostic": ("yes", "no", "confidence + Student-t"),
               "mean_teacher": ("no", "no", "confidence"),
               "graph_confidence": ("yes", "yes", "confidence"),
               "graph_gaussian": ("yes", "yes", "confidence + Gaussian"),
               "graph_supervised": ("yes", "yes", "none")}
    for method, (graph, view, filt) in filters.items():
        group = grouped[(method, 0.25)]
        supervised = method == "graph_supervised"
        ablations.append({"method": method, "graph": graph, "view_conditioning": view,
                          "filter": filt, "dice_mean": mean(e["summary"]["dice"] for e in group),
                          "dice_sd": sd(e["summary"]["dice"] for e in group),
                          "hd95_mean": mean(e["summary"]["hd95"] for e in group),
                          "accepted_mean": "" if supervised else mean(e["accepted"] for e in group),
                          "accepted_sd": "" if supervised else sd(e["accepted"] for e in group)})
    budgets = []
    for fraction in (0.25, 1.0):
        for method in METHODS100:
            group = grouped[(method, fraction)]
            row = {"label_fraction": fraction, "method": method,
                   "macro_dice": mean(e["summary"]["dice"] for e in group)}
            for column, key in [("A2C", ("view", "A2C")), ("A4C", ("view", "A4C")),
                                ("ED", ("phase", "ED")), ("ES", ("phase", "ES")),
                                ("Good", ("quality", "Good")), ("Medium", ("quality", "Medium")),
                                ("Poor", ("quality", "Poor"))]:
                row[column] = mean(float(e["strata"][key]["dice"]) for e in group)
            budgets.append(row)
    calibrated = []
    for method in ("unet", "mean_teacher", "segformer_b0", "aces"):
        for mode in ("raw", "source_temperature"):
            group = grouped[(method, 0.25)]
            temperatures = [1.0 if mode == "raw" else e["summary"]["temperature"] for e in group]
            row = {"method": method, "probability_setting": mode,
                   "temperature_mean": mean(temperatures), "temperature_sd": sd(temperatures)}
            for metric in ("nll", "brier", "ece"):
                values = [e["summary"]["calibration"][mode][metric] for e in group]
                row[metric+"_mean"], row[metric+"_sd"] = mean(values), sd(values)
            calibrated.append(row)
    published = read_json(ROOT/"results/published/results.json")
    for row in published["rows"]:
        group = grouped[(row["method"], row["label_fraction"])]
        for metric in METRICS:
            values = [e["summary"][metric] for e in group]
            equal(mean(values), row[metric+"_mean"], f"published {row['method']}/{metric}/mean")
            equal(sd(values), row[metric+"_sd"], f"published {row['method']}/{metric}/SD")
    paired = []
    for row in published["paired_comparisons"]:
        left, right = row["contrast"].split(" - ")
        stats = bootstrap(patient_mean(left, row["label_fraction"]),
                          patient_mean(right, row["label_fraction"]))
        for key in ("mean_difference", "ci95_low", "ci95_high"):
            equal(stats[key], row[key], f"published bootstrap/{row['contrast']}/{key}")
        paired.append({**row, **stats})
    additional = []
    for fraction, left, right in [
        (0.25, "mean_teacher", "unet"),
        (0.25, "graph_confidence", "mean_teacher"),
        (0.25, "graph_confidence", "graph_supervised")]:
        additional.append({"contrast": left+" - "+right, "label_fraction": fraction,
                           **bootstrap(patient_mean(left, fraction), patient_mean(right, fraction))})
    for method in METHODS100:
        additional.append({"contrast": method+" 100% - 25%", "label_fraction": "1.0 - 0.25",
                           **bootstrap(patient_mean(method, 1.0), patient_mean(method, 0.25))})
    acceptance_rows = []
    validation_rows = []
    for (method, fraction), group in grouped.items():
        for epoch in range(1, 201):
            accepted = [float(e["history"][epoch-1]["accepted_pixel_fraction"]) for e in group]
            dice = [float(e["history"][epoch-1]["validation_macro_dice"]) for e in group]
            common = {"method": method, "label_fraction": fraction, "epoch": epoch}
            acceptance_rows.append({**common, "mean": mean(accepted), "min": min(accepted), "max": max(accepted)})
            validation_rows.append({**common, "mean": mean(dice), "min": min(dice), "max": max(dice)})
    aces, mt = patient_mean("aces", 0.25), patient_mean("mean_teacher", 0.25)
    snapshots = read_json(ROOT/"data/metadata/manifest_snapshot.json")["records"]
    poor = {r["patient_id"] for r in snapshots if r["split"] == "test" and r["quality"] == "Poor"}
    scatter = [{"patient_id": p, "mean_teacher_dice": mt[p], "aces_dice": aces[p],
                "aces_minus_mean_teacher": aces[p]-mt[p], "has_poor_sequence": p in poor} for p in sorted(aces)]
    output.mkdir(parents=True, exist_ok=True)
    files = {"table1_main.csv": main_table, "table2_ablation.csv": ablations,
             "table3_budget_strata.csv": budgets, "table4_calibration.csv": calibrated,
             "paired_patient_bootstrap.csv": paired, "additional_paired_comparisons.csv": additional,
             "per_run_recomputed.csv": diagnostics, "acceptance_curves.csv": acceptance_rows,
             "validation_curves.csv": validation_rows, "patient_scatter.csv": scatter}
    for name, rows in files.items():
        write_csv(output/name, rows)
    report = {"status": "passed", "runs": len(runs), "configurations": len(grouped),
              "seeds": list(SEEDS), "test_patients": 50, "test_frame_class_rows": len(runs)*600,
              "published_bootstrap_comparisons_recomputed": len(paired),
              "numeric_tolerance": 1e-10, "bootstrap_seed": 20261029, "bootstrap_repetitions": 2000,
              "scatter_correlation": float(np.corrcoef(list(aces.values()), list(mt.values()))[0, 1]),
              "aces_higher_patients": sum(aces[p] > mt[p] for p in aces),
              "scope": "Reaggregation of supplied metric records and saved-bin ECE; no model training or pixel-level metric rerun.",
              "calibration_scope": "NLL/Brier retained from saved summary records; probabilities are not included.",
              "files": sorted(files)}
    (output/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT/"results/archived_runs")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs/reproduced")
    args = parser.parse_args(argv)
    reproduce(args.records, args.out)


if __name__ == "__main__":
    main()
