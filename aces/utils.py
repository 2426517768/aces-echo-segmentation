from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import platform
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def system_info():
    out = {"python": sys.version, "os": platform.platform(), "torch": torch.__version__,
           "numpy": np.__version__, "cuda_build": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        out.update(gpu=torch.cuda.get_device_name(0),
                   gpu_memory_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                   capability=list(torch.cuda.get_device_capability(0)),
                   compiled_arches=torch.cuda.get_arch_list())
        try:
            out["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                timeout=10, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            out["nvidia_smi"] = "unavailable"
    return out


def device_check(require_cuda=True):
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("没有可用 CUDA。请更新 NVIDIA 驱动并安装本包指定的 CUDA 12.8 PyTorch；正式训练不会静默改用 CPU。")
        return torch.device("cpu")
    d = torch.device("cuda")
    # The kernel launch checks compatibility, not just the GPU name.
    try:
        # Evaluation/latency callers disable autograd. The compatibility probe
        # intentionally tests backward as well as forward, so give only this
        # small probe its own context and restore the caller's mode afterward.
        with torch.inference_mode(False), torch.enable_grad():
            x = torch.randn(64, 64, device=d, requires_grad=True)
            (x @ x).square().mean().backward()
            torch.cuda.synchronize()
    except RuntimeError as e:
        raise RuntimeError("CUDA 前反向自检失败，底层错误：" + str(e)) from e
    return d


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)
