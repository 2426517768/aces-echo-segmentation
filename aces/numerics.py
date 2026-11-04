"""Explicit numerical policy and fail-closed diagnostics, without changing samples."""
from __future__ import annotations
import copy
import torch
from .utils import write_json

POLICY_VERSION = "1.4"


class NumericalError(RuntimeError):
    def __init__(self, stage, details):
        self.stage, self.details = stage, details
        super().__init__(f"Nonfinite {stage}; 最近有效断点保留，未跳过样本或生成替代结果。")


def effective_amp(method, cfg, device):
    # A numerical safeguard, not a change to the persisted experiment contract.
    return bool(cfg.get("amp", True) and device.type == "cuda" and method != "segformer_b0")


def precision_record(method, cfg, device, previous=None):
    mode = "float16_amp" if effective_amp(method, cfg, device) else "float32"
    history = copy.deepcopy((previous or {}).get("precision_history", []))
    if previous is not None and not history:
        history.append({"epoch": 1, "update": 1,
                        "precision": "float16_amp" if previous.get("scaler") else "float32",
                        "source": "legacy_checkpoint_scaler_state"})
    epoch, step = ((previous["epoch"], previous["next_step"]) if previous is not None else (0, 0))
    if epoch < cfg["epochs"] and (not history or history[-1]["precision"] != mode):
        history.append({"epoch": epoch+1, "update": step+1, "precision": mode,
                        "source": "runtime_policy_"+POLICY_VERSION})
    return {"policy_version": POLICY_VERSION, "method": method,
            "requested_amp": bool(cfg.get("amp", True)), "active_precision": mode,
            "evaluation_precision": mode, "precision_history": history,
            "reason": "SegFormer FP32 numerical safeguard" if method == "segformer_b0" else "configured precision"}


def require_finite(value, stage):
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return
    usable = value.detach()[finite]
    raise NumericalError(stage, {"shape": list(value.shape), "dtype": str(value.dtype),
        "nonfinite_values": int((~finite).sum().item()),
        "finite_min": float(usable.min().item()) if usable.numel() else None,
        "finite_max": float(usable.max().item()) if usable.numel() else None})


def require_finite_state(value, stage, prefix=""):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            try:
                require_finite(value, stage)
            except NumericalError as error:
                error.details["state_path"] = prefix
                raise
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_state(item, stage, prefix+"/"+str(key))
    elif isinstance(value, (list, tuple)):
        for key, item in enumerate(value):
            require_finite_state(item, stage, prefix+"/"+str(key))


def require_finite_gradients(model):
    # Public norm helper checks all gradients efficiently. Infinite max_norm
    # leaves finite gradients unchanged; there is no gradient clipping policy.
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                      norm_type=float("inf"), error_if_nonfinite=True)
    except RuntimeError as error:
        raise NumericalError("gradients", {"error": str(error)}) from error


def record_failure(run, error, **context):
    write_json(run/"numerical_error.json", {"status": "stopped", "stage": error.stage,
        "details": error.details, "policy_version": POLICY_VERSION, **context})
