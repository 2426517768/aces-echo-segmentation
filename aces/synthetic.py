"""Generated geometry for software tests only; never a clinical benchmark."""
from pathlib import Path
import numpy as np
from .data import save_case, validate_manifest
from .utils import write_json


def make_synthetic(out, size=64):
    out = Path(out)
    rng = np.random.default_rng(123)
    yy, xx = np.mgrid[:size, :size]
    records = []
    for patient in range(8):
        split = "train" if patient < 4 else "val" if patient < 6 else "test"
        for view in ["A2C", "A4C"]:
            for phase in ["ED", "ES"]:
                mask = np.zeros((size, size), np.uint8)
                cx = size*(0.47 if view == "A2C" else 0.53)
                cy = size*0.55
                ry = size*(0.24 if phase == "ED" else 0.19)
                outer = ((xx-cx)/(size*0.19))**2+((yy-cy)/(ry*1.17))**2 < 1
                inner = ((xx-cx)/(size*0.14))**2+((yy-cy)/ry)**2 < 1
                mask[outer] = 2
                mask[inner] = 1
                atrium = ((xx-cx-size*0.02)/(size*0.13))**2+((yy-size*0.20)/(size*0.10))**2 < 1
                mask[atrium] = 3
                image = np.clip(mask*45+40+rng.normal(0, 10, mask.shape), 0, 255).astype(np.float32)
                rec = {"case_id": f"SYNTHETIC_p{patient}_{view}_{phase}", "patient_id": f"SYNTHETIC_p{patient}",
                       "view": view, "phase": phase, "cohort": "CAMUS", "split": split,
                       "spacing_xy": [0.4, 0.6], "quality": "Synthetic"}
                records.append(save_case(out, rec, image, mask))
    manifest = {"schema": 1, "cohort": "CAMUS", "synthetic": True,
                "warning": "SOFTWARE TEST DATA ONLY. NOT CLINICAL RESULTS.", "records": records}
    manifest["counts"] = validate_manifest(manifest)
    write_json(out/"manifest.json", manifest)
    return out/"manifest.json"
