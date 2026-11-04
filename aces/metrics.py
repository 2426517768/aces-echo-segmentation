from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def binary_metrics(pred, reference, spacing_xy=None):
    p, g = np.asarray(pred, bool), np.asarray(reference, bool)
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError("Metric masks must be on the same 2D native annotation grid")
    unit = "mm" if spacing_xy is not None else "pixels"
    spacing = tuple(reversed(spacing_xy)) if spacing_xy is not None else (1.0, 1.0)
    if not g.any():
        return {"dice": None, "iou": None, "hd95": None, "assd": None, "empty_prediction": float(not p.any()),
                "missing_reference": 1, "distance_unit": unit}
    inter = np.logical_and(p, g).sum()
    dice = 2*inter/(p.sum()+g.sum())
    iou = inter/np.logical_or(p, g).sum()
    if not p.any():
        diagonal = math.hypot(g.shape[0]*spacing[0], g.shape[1]*spacing[1])
        return {"dice": 0.0, "iou": 0.0, "hd95": diagonal, "assd": diagonal,
                "empty_prediction": 1.0, "missing_reference": 0, "distance_unit": unit}
    ps = p ^ binary_erosion(p, structure=np.ones((3, 3)), border_value=0)
    gs = g ^ binary_erosion(g, structure=np.ones((3, 3)), border_value=0)
    distances = np.concatenate([distance_transform_edt(~gs, sampling=spacing)[ps],
                                distance_transform_edt(~ps, sampling=spacing)[gs]])
    return {"dice": float(dice), "iou": float(iou), "hd95": float(np.percentile(distances, 95)),
            "assd": float(distances.mean()), "empty_prediction": 0.0, "missing_reference": 0, "distance_unit": unit}


def probability_metrics(prob, target, bins=15):
    """Exact per-frame pixels; aggregation below gives every patient equal mass."""
    prob = np.asarray(prob, np.float64)
    target = np.asarray(target, np.int64)
    p = prob.reshape(prob.shape[0], -1)
    y = target.reshape(-1)
    if not np.allclose(p.sum(0), 1, atol=1e-4):
        raise ValueError("Expected normalized class probabilities")
    chosen = p[y, np.arange(len(y))]
    nll = -np.log(np.clip(chosen, 1e-12, 1)).mean()
    one_hot = np.eye(p.shape[0])[y].T
    brier = ((p-one_hot)**2).sum(0).mean()
    conf = p.max(0)
    correct = p.argmax(0) == y
    ids = np.minimum((conf*bins).astype(int), bins-1)
    mass = np.bincount(ids, minlength=bins)/len(y)
    conf_sum = np.bincount(ids, weights=conf, minlength=bins)/len(y)
    correct_sum = np.bincount(ids, weights=correct.astype(float), minlength=bins)/len(y)
    return {"nll": float(nll), "brier": float(brier), "mass": mass.tolist(),
            "confidence_mass": conf_sum.tolist(), "correct_mass": correct_sum.tolist()}


def _mean_dict(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def aggregate_case_metrics(rows):
    """Equal class/view/phase contribution within patient; recording-first for external videos."""
    patient_class = defaultdict(list)
    keys = ["dice", "iou", "hd95", "assd", "empty_prediction"]
    units = {r["distance_unit"] for r in rows}
    if len(units) != 1:
        raise ValueError("Cannot pool physical millimeters and pixel distances")
    for r in rows:
        patient_class[(r["patient_id"], r["class_id"])].append(r)
    classes = []
    for (p, c), group in sorted(patient_class.items()):
        recordings = defaultdict(list)
        for r in group:
            recordings[r.get("recording_id", r["patient_id"])].append(r)
        rec_means = [_mean_dict(rr, keys) for rr in recordings.values()]
        means = _mean_dict(rec_means, keys)
        means.update(patient_id=p, class_id=c, missing_reference_frames=sum(r["missing_reference"] for r in group))
        classes.append(means)
    patients = []
    for p in sorted({r["patient_id"] for r in classes}):
        group = [r for r in classes if r["patient_id"] == p]
        row = _mean_dict(group, keys)
        row.update(patient_id=p, evaluated_classes=sum(r["dice"] is not None for r in group))
        patients.append(row)
    summary = _mean_dict(patients, keys)
    summary.update(patients=sum(r["dice"] is not None for r in patients), patients_total=len(patients), distance_unit=next(iter(units)),
                   missing_reference_frames=sum(r["missing_reference"] for r in rows))
    return summary, patients, classes


def aggregate_calibration(rows, bootstrap_repetitions=2000):
    keys = ["nll", "brier", "mass", "confidence_mass", "correct_mass"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["patient_id"]].append(r)
    averages = []
    for patient, rr in groups.items():
        recordings = defaultdict(list)
        for r in rr:
            recordings[r.get("recording_id", patient)].append(r)
        recs = [{k: np.mean([x[k] for x in rs], axis=0) for k in keys} for rs in recordings.values()]
        averages.append({k: np.mean([x[k] for x in recs], axis=0) for k in keys})
    if not averages:
        raise ValueError("No calibration observations")
    vals = {k: np.mean([r[k] for r in averages], axis=0) for k in keys}
    ece = np.abs(vals["confidence_mass"]-vals["correct_mass"]).sum()
    result = {"nll": float(vals["nll"]), "brier": float(vals["brier"]), "ece": float(ece),
            "bin_mass": vals["mass"].tolist(), "bin_confidence_mass": vals["confidence_mass"].tolist(),
            "bin_correct_mass": vals["correct_mass"].tolist(), "patients": len(averages)}
    if bootstrap_repetitions:
        rng = np.random.default_rng(20261029)
        nll = np.array([a["nll"] for a in averages])
        brier = np.array([a["brier"] for a in averages])
        differences = np.stack([a["confidence_mass"]-a["correct_mass"] for a in averages])
        draws = []
        for _ in range(bootstrap_repetitions):
            ix = rng.integers(0, len(averages), len(averages))
            draws.append([nll[ix].mean(), brier[ix].mean(), np.abs(differences[ix].mean(0)).sum()])
        bounds = np.quantile(draws, [0.025, 0.975], axis=0)
        result["patient_bootstrap_ci95"] = {key: bounds[:, i].tolist() for i, key in enumerate(["nll", "brier", "ece"])}
    return result


def paired_bootstrap(left, right, repetitions=2000, seed=20261029):
    if set(left) != set(right) or not left:
        raise ValueError("Paired bootstrap requires identical nonempty patient sets")
    ids = sorted(left)
    values = np.array([left[p]-right[p] for p in ids], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite paired endpoint")
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)])
    return {"mean_difference": float(values.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(), "patients": len(ids)}
