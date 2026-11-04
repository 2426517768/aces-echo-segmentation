from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from .utils import digest


@torch.no_grad()
def mask_features(mask):
    """BHW masks -> dimensionless Bx6 features and nonempty-class flags."""
    b, height, width = mask.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=mask.device, dtype=torch.float32) / height,
                            torch.arange(width, device=mask.device, dtype=torch.float32) / width, indexing="ij")
    areas, centroids = [], []
    for c in [1, 2, 3]:
        m = (mask == c).float()
        a = m.sum((1, 2))
        areas.append(a)
        centroids.append(torch.stack([(m * xx).sum((1, 2)), (m * yy).sum((1, 2))], -1) / a.clamp_min(1)[:, None])
    valid = torch.stack(areas, -1).gt(0).all(-1)
    f = torch.cat([torch.stack([(areas[1].clamp_min(1) / areas[0].clamp_min(1)).log(),
                                (areas[2].clamp_min(1) / areas[0].clamp_min(1)).log()], -1),
                   centroids[1] - centroids[0], centroids[2] - centroids[0]], -1)
    return f, valid


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    idx = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
    return float(values[order[min(idx, len(order)-1)]])


class AnatomicalPrior:
    def __init__(self, state):
        self.state = state
        self.location = torch.tensor(state["location"], dtype=torch.float32)
        self.scale = torch.tensor(state["scale"], dtype=torch.float32)

    @classmethod
    def fit(cls, labeled_records, load_canonical):
        if not labeled_records or any(r["split"] != "train" for r in labeled_records):
            raise ValueError("Prior fitting requires labeled training patients only")
        by_view = {0: [], 1: []}
        for r in labeled_records:
            image, mask, meta = load_canonical(r)
            if mask is None:
                continue
            f, valid = mask_features(mask[None])
            if valid.item():
                by_view[meta["view_id"]].append((r["patient_id"], f[0].numpy()))
        locations, scales = [], []
        counts = {}
        for v in [0, 1]:
            samples = by_view[v]
            if not samples:
                raise ValueError(f"No valid labeled training masks for view {v}; cannot fit prior from held-out masks")
            count = Counter(p for p, _ in samples)
            weights = np.array([1 / count[p] for p, _ in samples], dtype=float)
            values = np.stack([f for _, f in samples])
            center = np.array([weighted_median(values[:, j], weights) for j in range(6)])
            mad = np.array([weighted_median(abs(values[:, j]-center[j]), weights) for j in range(6)])
            locations.append(center.tolist())
            scales.append(np.maximum(1.4826 * mad, 1e-3).tolist())
            counts[str(v)] = {"patients": len(count), "frames": len(samples)}
        patients = sorted({r["patient_id"] for r in labeled_records})
        return cls({"location": locations, "scale": scales, "nu": 5.0,
                    "training_patients": patients, "patient_hash": digest(patients), "valid_fit_counts": counts})

    @torch.no_grad()
    def score(self, mask, view, family="student_t"):
        f, valid = mask_features(mask)
        center = self.location.to(mask.device)[view]
        scale = self.scale.to(mask.device)[view]
        z2 = ((f - center) / scale).square()
        if family == "gaussian":
            score = torch.exp(-z2 / 2).mean(-1)
        elif family == "student_t":
            nu = self.state["nu"]
            score = torch.pow(1 + z2 / nu, -(nu + 1) / 2).mean(-1)
        else:
            raise ValueError(family)
        return torch.where(valid, score, torch.zeros_like(score))
