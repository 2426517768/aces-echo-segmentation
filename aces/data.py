from __future__ import annotations
from collections import defaultdict, OrderedDict
import csv
import json
import math
from pathlib import Path
import re
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .utils import read_json, write_json, sha256, digest

VIEWS = {"A2C": 0, "A4C": 1, "2CH": 0, "4CH": 1}


def safe_extract(archive, destination):
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (dest/info.filename).resolve()
            if not target.is_relative_to(dest) or "\\" in info.filename or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe zip member: {info.filename}")
        z.extractall(dest)


def read_array(path, frame=None, mat_key=None, frame_axis=0):
    path = Path(path)
    name = path.name.lower()
    spacing = None
    if name.endswith((".nii.gz", ".nii", ".mhd", ".mha")):
        try:
            import SimpleITK as sitk
        except ImportError as e:
            raise RuntimeError("Medical image reading requires SimpleITK") from e
        im = sitk.ReadImage(str(path))
        a = sitk.GetArrayFromImage(im)
        spacing = list(im.GetSpacing()[:2])
    elif name.endswith(".npy"):
        a = np.load(path, allow_pickle=False)
    elif name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            a = z[mat_key or "arr_0"].copy()
    elif name.endswith(".mat"):
        from scipy.io import loadmat
        if not mat_key:
            raise ValueError("MAT mask/image requires an explicit mat_key in external CSV")
        data = loadmat(path)
        if mat_key not in data:
            raise ValueError(f"Missing MAT variable {mat_key} in {path}")
        a = data[mat_key]
    elif name.endswith((".avi", ".mp4", ".mov")):
        if frame is None:
            raise ValueError("Video frame index must be explicit and zero-based")
        import cv2
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, a = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot decode video frame {frame}: {path}")
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        return a, None
    else:
        a = np.asarray(Image.open(path))
        if a.ndim == 3:
            if np.array_equal(a[:, :, 0], a[:, :, 1]) and np.array_equal(a[:, :, 0], a[:, :, 2]):
                a = a[:, :, 0]
            else:
                raise ValueError(f"RGB image/mask requires an explicit grayscale conversion: {path}")
    if frame is not None:
        if a.ndim < 3:
            raise ValueError(f"Frame index supplied for a 2D array: {path}")
        a = np.take(a, int(frame), axis=int(frame_axis))
    while a.ndim > 2 and 1 in a.shape:
        a = np.squeeze(a, axis=next(i for i, d in enumerate(a.shape) if d == 1))
    return a, spacing


def normalize_image(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite 2D grayscale pixels")
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a-lo)/(hi-lo), 0, 1).astype(np.float32)


def letterbox(image, mask, size):
    h, w = image.shape
    scale = min(size/h, size/w)
    nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
    top, left = (size-nh)//2, (size-nw)//2
    x = torch.from_numpy(normalize_image(image))[None, None]
    x = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False)[0]
    x = F.pad(x, (left, size-nw-left, top, size-nh-top))
    y = None
    if mask is not None:
        y = F.interpolate(torch.as_tensor(mask, dtype=torch.float32)[None, None], (nh, nw), mode="nearest")[0, 0].long()
        y = F.pad(y, (left, size-nw-left, top, size-nh-top))
    meta = {"native_shape": [h, w], "resized_shape": [nh, nw], "top": top, "left": left}
    return x, y, meta


def native_logits(logits, meta):
    nh, nw = meta["resized_shape"]
    top, left = meta["top"], meta["left"]
    cut = logits[..., top:top+nh, left:left+nw]
    return F.interpolate(cut, size=meta["native_shape"], mode="bilinear", align_corners=False)


def validate_manifest(manifest):
    records = manifest["records"]
    if not records:
        raise ValueError("Empty manifest")
    split_for_patient, ids = {}, set()
    for r in records:
        patient = (r.get("cohort", "CAMUS"), r["patient_id"])
        if patient in split_for_patient and split_for_patient[patient] != r["split"]:
            raise ValueError(f"Patient leakage across splits: {patient}")
        split_for_patient[patient] = r["split"]
        if r["case_id"] in ids:
            raise ValueError(f"Duplicate case id: {r['case_id']}")
        ids.add(r["case_id"])
        if r["view"] not in {"A2C", "A4C"}:
            raise ValueError("Only known A2C/A4C metadata is supported")
        if r["split"] in {"val", "test", "external"} and not r.get("has_mask"):
            raise ValueError(f"Held-out frame lacks a reference: {r['case_id']}")
        if r.get("spacing_xy") is not None:
            sp = np.asarray(r["spacing_xy"], dtype=float)
            if sp.shape != (2,) or not np.isfinite(sp).all() or np.any(sp <= 0):
                raise ValueError("Physical spacing must be two positive values")
    for split in ["train", "val", "test"]:
        if manifest.get("cohort") == "CAMUS" and not any(r["split"] == split and r["has_mask"] for r in records):
            raise ValueError(f"No labeled {split} cases; provide actual released references and split lists")
    return {s: {"patients": len({r["patient_id"] for r in records if r["split"] == s}),
                "frames": sum(r["split"] == s for r in records),
                "labeled_frames": sum(r["split"] == s and r["has_mask"] for r in records)}
            for s in sorted({r["split"] for r in records})}


def annotation_subset(records, fraction, seed=20261029):
    if not 0 < fraction <= 1:
        raise ValueError("label_fraction must be in (0,1]")
    groups = defaultdict(list)
    for r in records:
        if r["split"] == "train" and r["has_mask"]:
            groups[r["patient_id"]].append(r)
    if not groups:
        raise ValueError("No training labels")
    # Stable patient-level random subset. All methods/seeds use precisely this subset.
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n = max(1, round(len(ids)*fraction))
    patients = sorted(ids[i] for i in order[:n])
    return patients, [r for r in records if r["split"] == "train" and r["has_mask"] and r["patient_id"] in set(patients)]


class RecordStore:
    def __init__(self, manifest_path, size=256, cache_items=32):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = read_json(manifest_path)
        validate_manifest(self.manifest)
        self.root, self.size, self.limit = self.manifest_path.parent, size, cache_items
        for r in self.manifest["records"]:
            path = self.root/r["path"]
            if not path.is_file() or sha256(path) != r.get("cache_sha256"):
                raise ValueError(f"Prepared cache missing or changed: {r['case_id']}; restore or rebuild in a new prepared-data directory")
        self.cache = OrderedDict()

    def native(self, record, with_mask=True):
        with np.load(self.root/record["path"], allow_pickle=False) as z:
            image = z["image"].copy()
            # Hidden training labels are not fetched from the file by an unlabeled loader.
            mask = z["mask"].astype(np.int64) if with_mask and record["has_mask"] else None
        return image, mask

    def canonical(self, record, with_mask=True):
        key = (record["case_id"], with_mask)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        image, mask = self.native(record, with_mask)
        x, y, meta = letterbox(image, mask, self.size)
        meta["view_id"] = VIEWS[record["view"]]
        val = (x, y, meta)
        self.cache[key] = val
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return val

    def batch(self, records, n, rng, device, with_mask=True):
        groups = defaultdict(list)
        for r in records:
            groups[r["patient_id"]].append(r)
        patients = sorted(groups)
        if not patients:
            raise ValueError("Empty sampling pool")
        selected = [rng.choice(groups[rng.choice(patients)]) for _ in range(n)]
        self.last_batch_case_ids = [r["case_id"] for r in selected]
        vals = [self.canonical(r, with_mask) for r in selected]
        x = torch.stack([a[0] for a in vals]).to(device)
        v = torch.tensor([a[2]["view_id"] for a in vals], dtype=torch.long, device=device)
        y = torch.stack([a[1] for a in vals]).to(device) if with_mask else None
        return x, y, v


def parse_info(path):
    out = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"\s*([^:=]+)\s*[:=]\s*(.*?)\s*$", line)
            if m:
                out[m[1].strip().lower().replace(" ", "")] = m[2]
    return out


def patient_list(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    ids = re.findall(r"patient\d+", text, flags=re.I)
    if not ids:
        raise ValueError(f"No patient IDs in official split file {path}")
    return sorted(set(x.lower() for x in ids))


def resolve_splits(root, ids, config, infos):
    explicit = config.get("split_file")
    provenance = []
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = Path.cwd()/p
        raw = read_json(p)
        splits = {k: [str(x).lower() for x in raw.get(k, [])] for k in ["train", "val", "test"]}
        provenance.append({"path": str(p.resolve()), "sha256": sha256(p)})
    else:
        splits = {}
        aliases = {"train": {"train", "training", "train_patients", "training_patients", "trainingset", "subgroup_training"},
                   "val": {"val", "valid", "validation", "val_patients", "validation_patients", "validationset", "subgroup_validation"},
                   "test": {"test", "testing", "test_patients", "testing_patients", "testingset", "subgroup_testing"}}
        for k, names in aliases.items():
            candidates = [p for p in root.rglob("*.txt") if p.stem.lower() in names]
            if len(candidates) > 1:
                unique = {tuple(patient_list(p)) for p in candidates}
                if len(unique) != 1:
                    raise ValueError(f"Ambiguous {k} split files; specify split_file explicitly")
            if candidates:
                splits[k] = patient_list(candidates[0])
                provenance.append({"path": str(candidates[0].resolve()), "sha256": sha256(candidates[0])})
        if not splits.get("train") or not splits.get("test"):
            raise ValueError("未找到官方训练/测试患者清单。把清单整理成 data/splits.json 并填写 settings.json 的 split_file；不会按患者编号猜分组。")
    train, val, test = [set(splits.get(k, [])) for k in ["train", "val", "test"]]
    if train & val or train & test or val & test:
        raise ValueError("Official split lists overlap; do not flatten legacy train/test IDs")
    if not train or not test:
        raise ValueError("Train and test patient lists must be nonempty")
    if not val:
        count = config.get("validation_patients", 50)
        if len(train) <= count:
            raise ValueError("Not enough development patients to create validation partition")
        # Stratified random round-robin allocation, no use of test cases.
        strata = defaultdict(list)
        for p in sorted(train):
            info = infos.get(p, {})
            quality = info.get("imagequality", "Unknown")
            try:
                ef = float(info.get("ef", "nan"))
                category = "low" if ef < 45 else "high" if ef > 55 else "middle" if math.isfinite(ef) else "unknown"
            except ValueError:
                category = "unknown"
            strata[(quality, category)].append(p)
        rng = np.random.default_rng(config.get("split_seed", 20261029))
        # Proportional systematic sampling of the sorted strata after within-stratum shuffling.
        ordered = []
        for key in sorted(strata):
            a = strata[key]
            rng.shuffle(a)
            ordered.extend(a)
        picks = np.floor((np.arange(count)+rng.random())*len(ordered)/count).astype(int)
        val = {ordered[i] for i in picks}
        train -= val
    known = train | val | test
    if not known <= set(ids):
        raise ValueError(f"Split patients missing from loaded images: {sorted(known-set(ids))[:10]}")
    if set(ids)-known:
        raise ValueError("Unassigned patients in data; specify their true split or use one unambiguous release")
    mapping = {p: k for k, group in [("train", train), ("val", val), ("test", test)] for p in group}
    return mapping, provenance


def save_case(out, record, image, mask):
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"Invalid image {record['case_id']}")
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError(f"Image/mask grids differ for {record['case_id']}; declare and resolve alignment explicitly")
        if not np.isfinite(mask).all() or not np.equal(mask, np.round(mask)).all():
            raise ValueError("Mask labels must be finite integers")
    path = out/"cases"/(record["case_id"]+".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"image": image.astype(np.float32)}
    if mask is not None:
        arrays["mask"] = mask.astype(np.uint8)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)
    record.update(path=str(path.relative_to(out)).replace("\\", "/"), has_mask=mask is not None, cache_sha256=sha256(path))
    return record


def prepare_camus(root, out, config):
    root, out = Path(root).resolve(), Path(out).resolve()
    # Adding an independent external evaluation must not invalidate source-training data.
    keys = {"label_map_confirmed", "label_map", "split_file", "validation_patients", "split_seed", "intermediate_frames_per_sequence"}
    config = {k:v for k,v in config.items() if k in keys}
    if not config.get("label_map_confirmed", False):
        raise ValueError("请先核对数据提供者标签说明，在 settings.json 设置 label_map_confirmed=true；默认映射不能自动视为已核对。")
    label_map = {int(k): int(v) for k, v in config.get("label_map", {0:0, 1:1, 2:2, 3:3}).items()}
    if set(label_map.values()) != {0, 1, 2, 3}:
        raise ValueError("label_map must cover exactly background/cavity/myocardium/atrium labels 0/1/2/3")
    pattern = re.compile(r"^(patient\d+)_(2CH|4CH)_(ED|ES)\.(nii\.gz|nii|mhd|mha)$", re.I)
    entries = {}
    infos = {}
    for p in root.rglob("*"):
        m = pattern.match(p.name) if p.is_file() else None
        if not m:
            continue
        patient, view, phase, ext = m.groups()
        key = (patient.lower(), view.upper(), phase.upper())
        if key in entries:
            raise ValueError(f"Duplicate patient/view/phase in raw data: {key}. Keep only one release.")
        gt = p.with_name(p.name[:-(len(ext)+1)]+"_gt."+ext)
        cfg = p.parent/("Info_"+view.upper()+".cfg")
        info = parse_info(cfg)
        infos.setdefault(patient.lower(), info)
        entries[key] = (p, gt if gt.exists() else None, info)
    if not entries:
        raise ValueError(f"No CAMUS ED/ES NIfTI or MHD pairs found under {root}")
    mapping, split_provenance = resolve_splits(root, {k[0] for k in entries}, config, infos)
    source_inventory = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(root.rglob("*")) if p.is_file() and
                        p.name.lower().endswith((".nii.gz", ".nii", ".mhd", ".mha", ".raw", ".zraw", ".cfg", ".txt", ".json"))
                        and not p.name.startswith(".extracted_")
                        and not any(part.startswith(".camus_download_") for part in p.relative_to(root).parts)]
    signature = digest({"source_inventory": source_inventory, "sources": [(str(p), p.stat().st_size, p.stat().st_mtime_ns,
                                     str(gt), gt.stat().st_size if gt else None, gt.stat().st_mtime_ns if gt else None)
                                    for p, gt, _ in entries.values()], "mapping": mapping,
                        "config": config, "split_provenance": split_provenance})
    target = out/"manifest.json"
    if target.exists() and read_json(target).get("source_signature") == signature:
        print("CAMUS prepared manifest unchanged; reusing cache.", flush=True)
        return target
    records = []
    for idx, ((patient, view, phase), (p, gt, info)) in enumerate(sorted(entries.items())):
        image, spacing = read_array(p)
        mask = None
        if gt:
            raw, mask_spacing = read_array(gt)
            if spacing and mask_spacing and not np.allclose(spacing, mask_spacing):
                raise ValueError("Image/reference spacing mismatch")
            import SimpleITK as sitk
            pi, gi = sitk.ImageFileReader(), sitk.ImageFileReader()
            pi.SetFileName(str(p)); gi.SetFileName(str(gt))
            pi.ReadImageInformation(); gi.ReadImageInformation()
            if pi.GetSize() != gi.GetSize() or not np.allclose(pi.GetOrigin(), gi.GetOrigin()) or not np.allclose(pi.GetDirection(), gi.GetDirection()):
                raise ValueError("Image/reference physical grids differ (size/origin/direction)")
            if not set(np.unique(raw).tolist()) <= set(label_map):
                raise ValueError("Unknown label values; correct the documented label_map")
            mask = np.zeros(raw.shape, dtype=np.uint8)
            for a, b in label_map.items():
                mask[raw == a] = b
        rec = {"case_id": f"{patient}_{view}_{phase}", "patient_id": patient, "view": "A2C" if view == "2CH" else "A4C",
               "phase": phase, "split": mapping[patient], "cohort": "CAMUS", "spacing_xy": spacing,
               "quality": info.get("imagequality", "Unknown"), "source_image": str(p), "source_mask": str(gt) if gt else None}
        records.append(save_case(out, rec, image, mask))
        if idx % 50 == 0:
            print(f"Preparing endpoint {idx+1}/{len(entries)}", flush=True)
    # Optional intermediate frames, ONLY for training patients; labels are never fabricated.
    if config.get("intermediate_frames_per_sequence", 8) > 0:
        processed = set()
        for key, (p, gt, info) in sorted(entries.items()):
            patient, view, phase = key
            if mapping[patient] != "train" or (patient, view) in processed:
                continue
            processed.add((patient, view))
            prefix = f"{patient}_{view}_"
            sequences = sorted(f for f in p.parent.iterdir() if f.name.lower().startswith(prefix.lower()) and "sequence" in f.name.lower())
            sequences = [f for f in sequences if f.name.lower().endswith((".nii.gz", ".nii", ".mhd", ".mha")) and "_gt" not in f.name.lower()]
            if not sequences:
                continue
            sp = next((f for f in sequences if "half_sequence" in f.name.lower()), sequences[0])
            volume, spacing = read_array(sp)
            if volume.ndim != 3:
                raise ValueError(f"Expected [frame,height,width] sequence: {sp}")
            n = min(len(volume), config.get("intermediate_frames_per_sequence", 8))
            selected = sorted(set(np.linspace(0, len(volume)-1, n+2).round().astype(int)[1:-1].tolist()))
            for frame in selected:
                rec = {"case_id": f"{patient}_{view}_MID_{frame:04d}", "patient_id": patient, "view": "A2C" if view == "2CH" else "A4C",
                       "phase": "MID", "split": "train", "cohort": "CAMUS", "spacing_xy": spacing,
                       "quality": info.get("imagequality", "Unknown"), "source_image": str(sp), "source_frame": frame}
                records.append(save_case(out, rec, volume[frame], None))
    manifest = {"schema": 1, "cohort": "CAMUS", "synthetic": False, "source_signature": signature,
                "source_root": str(root), "split_provenance": split_provenance, "config": config,
                "actual_split_patients": {s: sorted(p for p, k in mapping.items() if k == s) for s in ["train", "val", "test"]}, "records": records}
    manifest["counts"] = validate_manifest(manifest)
    write_json(target, manifest)
    write_json(out/"dataset_audit.json", {k:v for k,v in manifest.items() if k != "records"})
    print("CAMUS counts:", manifest["counts"], flush=True)
    return target


def prepare_external(csv_path, out, confirmed_target):
    if confirmed_target not in {"myocardium", "epicardial_area"}:
        raise ValueError("External target must be explicitly confirmed as myocardium or epicardial_area")
    csv_path, out = Path(csv_path).resolve(), Path(out).resolve()
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("External CSV is empty")
    source_paths = set()
    for row in rows:
        for key in ["image", "mask"]:
            if row.get(key):
                p = Path(row[key])
                source_paths.add((p if p.is_absolute() else csv_path.parent/p).resolve())
    signature = digest({"csv": sha256(csv_path), "target": confirmed_target,
                        "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(source_paths)]})
    target = out/"manifest.json"
    if target.exists() and read_json(target).get("source_signature") == signature:
        return target
    records = []
    for i, row in enumerate(rows):
        if row.get("confirmed", "").lower() != "true":
            raise ValueError(f"External row {i+2} not confirmed; do not guess patient IDs or annotation alignment")
        if not row.get("patient_id") or not row.get("image") or not row.get("mask"):
            raise ValueError("External CSV requires patient_id/image/mask")
        def get(name):
            p = Path(row[name])
            return p if p.is_absolute() else csv_path.parent/p
        def index(name):
            return int(row[name]) if row.get(name, "").strip() else None
        image, _ = read_array(get("image"), index("image_frame"), row.get("image_key") or None, int(row.get("image_frame_axis") or 0))
        mask, _ = read_array(get("mask"), index("mask_frame"), row.get("mask_key") or None, int(row.get("mask_frame_axis") or 0))
        if row.get("image_grid_transform", "none") != "none":
            if row["image_grid_transform"] != "resize_to_mask":
                raise ValueError("Supported explicit transform: resize_to_mask")
            if row.get("alignment_basis", "").strip() == "":
                raise ValueError("resize_to_mask requires alignment_basis from dataset documentation")
            image = np.asarray(Image.fromarray(image.astype(np.float32)).resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR))
        foreground = int(row.get("foreground_value") or 1)
        if not set(np.unique(mask).tolist()) <= {0, foreground}:
            raise ValueError("External reference must have exactly documented binary values")
        mask = (mask == foreground).astype(np.uint8)
        spacing = [float(row["spacing_x"]), float(row["spacing_y"])] if row.get("spacing_x") and row.get("spacing_y") else None
        rec = {"case_id": "HMC_"+digest(row)[:20], "patient_id": row["patient_id"],
               "recording_id": row.get("recording_id") or row["patient_id"], "view": "A4C", "phase": row.get("phase") or "CYCLE",
               "split": "external", "cohort": "HMC-QU", "spacing_xy": spacing,
               "external_target": confirmed_target, "quality": "Unknown", "mapping_row": row}
        records.append(save_case(out, rec, image, mask))
    manifest = {"schema": 1, "cohort": "HMC-QU", "synthetic": False,
                "source_signature": signature, "external_target": confirmed_target, "records": records}
    manifest["counts"] = validate_manifest(manifest)
    write_json(target, manifest)
    return target
