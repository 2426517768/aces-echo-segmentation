from __future__ import annotations
from collections import defaultdict
from contextlib import nullcontext
import copy
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .data import RecordStore, annotation_subset, native_logits
from .losses import supervised_loss, augment_supervised, pseudo_targets, unsupervised_loss, update_teacher
from .metrics import binary_metrics, probability_metrics, aggregate_case_metrics, aggregate_calibration
from .models import build_model, METHODS
from .numerics import (NumericalError, effective_amp, precision_record, require_finite,
                       require_finite_state, require_finite_gradients, record_failure)
from .prior import AnatomicalPrior
from .utils import (atomic_torch_save, load_checkpoint, write_json, read_json, write_csv,
                    digest, seed_everything, system_info, device_check, sha256)


def autocast(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(enabled)) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def validation_score(model, store, device, amp=False):
    model.eval()
    patient_classes = defaultdict(list)
    for r in store.manifest["records"]:
        if r["split"] != "val" or not r["has_mask"]:
            continue
        x, _, meta = store.canonical(r)
        view = torch.tensor([meta["view_id"]], device=device)
        with autocast(device, amp):
            logits = model(x[None].to(device), view)["logits"]
        require_finite(logits, "validation logits: "+r["case_id"])
        pred = native_logits(logits.float(), meta)[0].argmax(0).cpu().numpy()
        native_image, gt = store.native(r)
        for c in [1, 2, 3]:
            g, p = gt == c, pred == c
            if g.any():
                patient_classes[(r["patient_id"], c)].append(float(2*(p & g).sum()/(p.sum()+g.sum())))
    patients = defaultdict(list)
    for (p, c), vals in patient_classes.items():
        patients[p].append(float(np.mean(vals)))
    if not patients:
        raise ValueError("No scored validation patients")
    return float(np.mean([np.mean(v) for v in patients.values()]))


def train(manifest_path, output_dir, method, cfg, seed=17, label_fraction=0.25, require_cuda=True):
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    opts = METHODS[method]
    if opts["ssl"] and cfg["epochs"] <= cfg["warmup_epochs"]:
        raise ValueError("SSL training epochs must exceed supervised warm-up")
    if cfg["effective_batch"] % cfg["micro_batch"]:
        raise ValueError("effective_batch must be divisible by micro_batch")
    seed_everything(seed, cfg.get("cpu_threads", 4))
    device = device_check(require_cuda)
    store = RecordStore(manifest_path, cfg["image_size"], cfg.get("cache_items", 32))
    if store.manifest.get("synthetic") and require_cuda:
        raise ValueError("Synthetic data belongs to the separate smoke-test mode")
    records = store.manifest["records"]
    labeled_patients, labeled = annotation_subset(records, label_fraction, cfg.get("split_seed", 20261029))
    unlabeled = [r for r in records if r["split"] == "train" and
                 (not r["has_mask"] or r["patient_id"] not in set(labeled_patients))]
    run = Path(output_dir)
    run.mkdir(parents=True, exist_ok=True)
    contract = {"config": cfg, "method": method, "seed": seed, "label_fraction": label_fraction,
                "manifest_hash": digest(store.manifest), "labeled_patients": labeled_patients}
    contract_hash = digest(contract)
    if (run/"contract.json").exists() and read_json(run/"contract.json") != contract:
        raise ValueError("Existing run has another configuration/data signature; use a new output folder")
    write_json(run/"contract.json", contract)
    write_json(run/"hardware.json", system_info())
    write_json(run/"annotation_budget.json", {"labeled_training_patients": labeled_patients,
               "labeled_frames": len(labeled), "unlabeled_frames": len(unlabeled), "test_never_in_unlabeled_pool": True})
    model = build_model(method, cfg).to(device)
    teacher = None
    prior = None
    if opts["filter"] in {"gaussian", "student_t"}:
        prior = AnatomicalPrior.fit(labeled, store.canonical)
        write_json(run/"anatomical_prior.json", prior.state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    amp = effective_amp(method, cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = random.Random(seed)
    epoch, step, best = 0, 0, -math.inf
    sums = {"loss": 0.0, "accepted": 0.0, "updates": 0}
    elapsed = 0.0
    history = []
    state = None
    if (run/"last.pt").exists():
        state = load_checkpoint(run/"last.pt")
        if state["contract_hash"] != contract_hash:
            raise ValueError("Checkpoint/data/config signature mismatch")
        if method == "segformer_b0":
            try:
                require_finite_state({"student": state["student"], "optimizer": state["optimizer"]}, "checkpoint")
            except NumericalError as error:
                record_failure(run, error, method=method, seed=seed, epoch=state["epoch"]+1,
                               next_update=state["next_step"]+1, contract_hash=contract_hash)
                raise
        model.load_state_dict(state["student"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch, step, best, sums = state["epoch"], state["next_step"], state["best_score"], state["sums"]
        elapsed, history = state["elapsed_seconds"], state["history"]
        rng.setstate(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state["teacher"] is not None:
            teacher = copy.deepcopy(model).eval()
            teacher.load_state_dict(state["teacher"])
        print(f"Resuming {method}: epoch={epoch+1}, next_update={step+1}", flush=True)
    numerical_precision = precision_record(method, cfg, device, state)
    if method == "segformer_b0":
        write_json(run/"numerical_precision.json", numerical_precision)
        print("SegFormer 数值策略 v1.4：FP32 训练及评估；保留检查点、优化器、学习率计划和采样状态。", flush=True)
        # Legacy best scores were computed under FP16 on CUDA. Re-score those
        # same weights on validation only, under the new FP32 policy, before
        # comparing future epochs. Model construction must not consume the
        # resumed training RNG stream.
        if (run/"best.pt").exists():
            legacy_best = load_checkpoint(run/"best.pt")
            if legacy_best["contract"] != contract:
                raise ValueError("Best checkpoint/data/config signature mismatch")
            try:
                require_finite_state(legacy_best["model"], "best checkpoint")
                if (legacy_best.get("numerical_precision") or {}).get("evaluation_precision") != "float32":
                    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        best_model, _ = load_model_for_evaluation(run/"best.pt", device)
                        score = validation_score(best_model, store, device, amp=False)
                        del best_model
                    original_score = legacy_best.get("validation_dice")
                    legacy_best.update(validation_dice=score, numerical_precision=numerical_precision)
                    atomic_torch_save(legacy_best, run/"best.pt")
                    write_json(run/"best_fp32_revalidation.json", {
                        "source_split": "validation_only", "checkpoint_epoch": legacy_best["epoch"],
                        "previous_validation_dice": original_score, "fp32_validation_dice": score,
                        "model_weights_changed": False})
                    print(f"旧最佳权重已在验证集按 FP32 复核：Dice={score:.5f}。", flush=True)
                best = float(legacy_best["validation_dice"])
            except NumericalError as error:
                record_failure(run, error, method=method, seed=seed, epoch=epoch+1,
                               next_update=step+1, contract_hash=contract_hash)
                raise
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    since = time.perf_counter()
    accum = cfg["effective_batch"]//cfg["micro_batch"]

    def snapshot(ep, next_step, running):
        if method == "segformer_b0":
            require_finite_state({"student": model.state_dict(), "optimizer": optimizer.state_dict()},
                                 "checkpoint before save")
        return {"format_version": 1, "contract_hash": contract_hash, "contract": contract,
                "student": model.state_dict(), "teacher": teacher.state_dict() if teacher is not None else None,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": ep, "next_step": next_step, "best_score": best, "sums": running,
                "history": history, "elapsed_seconds": elapsed+time.perf_counter()-since,
                "sampling_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "prior": prior.state if prior else None,
                "numerical_precision": numerical_precision,
                "precision_history": numerical_precision["precision_history"]}
    step_index = step
    try:
        while epoch < cfg["epochs"]:
            if opts["ssl"] and teacher is None and epoch >= cfg["warmup_epochs"]:
                teacher = copy.deepcopy(model).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)
            model.train()
            for step_index in range(step, cfg["steps_per_epoch"]):
                optimizer.zero_grad(set_to_none=True)
                loss_sum, accepted_sum = 0.0, 0.0
                for _ in range(accum):
                    image, mask, view = store.batch(labeled, cfg["micro_batch"], rng, device)
                    if method == "segformer_b0":
                        require_finite(image, "training input")
                    image, mask = augment_supervised(image, mask)
                    with autocast(device, amp):
                        output = model(image, view)
                        loss = supervised_loss(output, mask, cfg.get("auxiliary_weight", 0.1))
                        if teacher is not None and unlabeled:
                            u, _, uv = store.batch(unlabeled, cfg["micro_batch"], rng, device, with_mask=False)
                            strong, pseudo, weights, scores = pseudo_targets(
                                teacher, u, uv, prior, opts["filter"],
                                cfg.get("confidence_threshold", 0.9), cfg.get("anatomy_threshold", 0.5))
                            student_u = model(strong, uv)["logits"]
                            ramp = min(1.0, (epoch+1-cfg["warmup_epochs"])/max(1, cfg["ramp_epochs"]))
                            loss = loss + ramp*unsupervised_loss(student_u, pseudo, weights)
                            accepted_sum += float((weights > 0).float().mean())/accum
                    require_finite(loss, "training loss")
                    scaler.scale(loss/accum).backward()
                    loss_sum += float(loss.detach())/accum
                previous_scale = scaler.get_scale()
                if method == "segformer_b0":
                    require_finite_gradients(model)
                scaler.step(optimizer)
                scaler.update()
                did_step = not amp or scaler.get_scale() >= previous_scale
                if teacher is not None and did_step:
                    update_teacher(teacher, model, cfg["ema_decay"])
                sums["loss"] += loss_sum
                sums["accepted"] += accepted_sum
                sums["updates"] += 1
                next_step = step_index+1
                if next_step % cfg.get("checkpoint_every_updates", 25) == 0:
                    atomic_torch_save(snapshot(epoch, next_step, sums), run/"last.pt")
                if next_step % cfg.get("log_every_updates", 10) == 0 or next_step == cfg["steps_per_epoch"]:
                    print(f"{method} seed={seed} label={label_fraction:.0%} epoch {epoch+1}/{cfg['epochs']} update {next_step}/{cfg['steps_per_epoch']} loss={loss_sum:.4f} accepted={accepted_sum:.3f}", flush=True)
            eval_model = teacher if teacher is not None else model
            score = validation_score(eval_model, store, device, amp)
            eligible = not opts["ssl"] or teacher is not None
            if eligible and score > best:
                best = score
                atomic_torch_save({"format_version": 1, "model": eval_model.state_dict(), "contract": contract,
                                   "epoch": epoch+1, "validation_dice": best,
                                   "evaluation_weights": "ema_teacher" if teacher is not None else "student",
                                   "prior": prior.state if prior else None,
                                   "numerical_precision": numerical_precision}, run/"best.pt")
            history.append({"epoch": epoch+1, "mean_loss": sums["loss"]/max(1, sums["updates"]),
                            "accepted_pixel_fraction": sums["accepted"]/max(1, sums["updates"]),
                            "validation_macro_dice": score, "eligible_for_checkpoint_selection": eligible,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            print(f"Validation patient macro Dice={score:.5f}; best eligible={best:.5f}", flush=True)
            scheduler.step()
            epoch, step = epoch+1, 0
            sums = {"loss": 0.0, "accepted": 0.0, "updates": 0}
            atomic_torch_save(snapshot(epoch, step, sums), run/"last.pt")
            write_csv(run/"history.csv", history)
        if not (run/"best.pt").exists():
            raise RuntimeError("No eligible checkpoint; SSL runs need epochs beyond warm-up")
        write_json(run/"training_complete.json", {"status": "completed", "contract_hash": contract_hash,
            "elapsed_seconds": elapsed+time.perf_counter()-since, "validation_best": best,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else None,
            "parameter_count": sum(p.numel() for p in model.parameters()), "synthetic": store.manifest.get("synthetic", False),
            "micro_batch": cfg["micro_batch"], "effective_batch": cfg["effective_batch"]})
        if method == "segformer_b0":
            complete = read_json(run/"training_complete.json")
            complete["numerical_precision"] = numerical_precision
            write_json(run/"training_complete.json", complete)
    except NumericalError as error:
        record_failure(run, error, method=method, seed=seed, epoch=epoch+1,
                       update=step_index+1, contract_hash=contract_hash,
                       case_ids=getattr(store, "last_batch_case_ids", []),
                       numerical_precision=numerical_precision)
        raise
    except KeyboardInterrupt:
        print("已停止。重新双击将从最近的完整更新检查点继续；不保存中断中的半个更新。", flush=True)
        raise
    except torch.OutOfMemoryError as e:
        raise RuntimeError("显存不足。关闭其他 GPU 程序；将 micro_batch 改成 1、effective_batch 保持 4，使用新的运行目录重新开始。不会静默缩小图像或改变模型。") from e
    return run/"best.pt"


def load_model_for_evaluation(checkpoint, device):
    state = load_checkpoint(checkpoint)
    require_finite_state(state["model"], "evaluation checkpoint")
    contract = state["contract"]
    model = build_model(contract["method"], contract["config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def collect_temperature_samples(model, store, device, pixels_per_patient=8192):
    records = [r for r in store.manifest["records"] if r["split"] == "val"]
    grouped = defaultdict(list)
    for r in records:
        grouped[r["patient_id"]].append(r)
    logits_list, labels_list, weights_list = [], [], []
    rng = np.random.default_rng(20261029)
    for p in sorted(grouped):
        rr = grouped[p]
        for r in rr:
            x, _, meta = store.canonical(r)
            v = torch.tensor([meta["view_id"]], device=device)
            raw = model(x[None].to(device), v)["logits"]
            require_finite(raw, "calibration logits: "+r["case_id"])
            logits = native_logits(raw.float(), meta)[0].cpu().flatten(1).T
            _, gt = store.native(r)
            n = min(max(1, pixels_per_patient//len(rr)), len(logits))
            indices = rng.choice(len(logits), n, replace=False)
            logits_list.append(logits[indices])
            labels_list.append(torch.from_numpy(gt.reshape(-1)[indices]).long())
            weights_list.append(torch.full((n,), 1/(len(grouped)*len(rr)*n)))
    return torch.cat(logits_list), torch.cat(labels_list), torch.cat(weights_list)


def fit_temperature(checkpoint, manifest_path, output, require_cuda=True, pixels_per_patient=8192):
    device = device_check(require_cuda)
    model, state = load_model_for_evaluation(checkpoint, device)
    store = RecordStore(manifest_path, state["contract"]["config"]["image_size"])
    if digest(store.manifest) != state["contract"]["manifest_hash"]:
        raise ValueError("Temperature fitting must use this checkpoint's original CAMUS manifest")
    logits, labels, weights = collect_temperature_samples(model, store, device, pixels_per_patient)
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=60, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        temp = log_t.clamp(-4, 4).exp()
        loss = (F.cross_entropy(logits/temp, labels, reduction="none")*weights).sum()
        loss.backward()
        return loss
    opt.step(closure)
    temperature = float(log_t.detach().clamp(-4, 4).exp())
    write_json(output, {"temperature": temperature, "source_split": "CAMUS_validation_only",
                        "source_manifest_hash": digest(store.manifest), "sampled_pixels": len(labels),
                        "sampling": "fixed seed; equal patients, equal frames; capped pixels per patient",
                        "pixels_per_patient_cap": pixels_per_patient,
                        "checkpoint_method": state["contract"]["method"], "checkpoint_epoch": state["epoch"],
                        "checkpoint_sha256": sha256(checkpoint)})
    return temperature


@torch.no_grad()
def evaluate(checkpoint, manifest_path, output_dir, split="test", temperature=1.0, require_cuda=True, save_predictions=True):
    device = device_check(require_cuda)
    model, state = load_model_for_evaluation(checkpoint, device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = RecordStore(manifest_path, state["contract"]["config"]["image_size"])
    external = split == "external"
    if not external and digest(store.manifest) != state["contract"]["manifest_hash"]:
        raise ValueError("Internal evaluation data must match the training manifest")
    if external and not store.manifest.get("external_target"):
        raise ValueError("External label schema was not confirmed")
    records = [r for r in store.manifest["records"] if r["split"] == split and r["has_mask"]]
    if not records:
        raise ValueError(f"No annotated {split} records")
    case_rows, calibration = [], {"raw": [], "source_temperature": []}
    amp = effective_amp(state["contract"]["method"], state["contract"]["config"], device)
    for r in records:
        x, _, meta = store.canonical(r)
        v = torch.tensor([meta["view_id"]], device=device)
        with autocast(device, amp):
            raw = model(x[None].to(device), v)["logits"]
        require_finite(raw, "evaluation logits: "+r["case_id"])
        logits = native_logits(raw.float(), meta)[0].cpu()
        _, gt = store.native(r)
        if external:
            fg = [2] if store.manifest["external_target"] == "myocardium" else [1, 2]
            # Fixed semantic projection: sum probabilities, then a binary 0.5 decision.
            foreground = logits.softmax(0)[fg].sum(0)
            prediction = (foreground >= 0.5).numpy().astype(np.uint8)
            classes = [1]
        else:
            prediction = logits.argmax(0).numpy().astype(np.uint8)
            classes = [1, 2, 3]
        for c in classes:
            scores = binary_metrics(prediction == c, gt == c, None if external else r.get("spacing_xy"))
            scores.update(case_id=r["case_id"], patient_id=r["patient_id"], class_id=c, view=r["view"], phase=r["phase"],
                          quality=r.get("quality", "Unknown"), recording_id=r.get("recording_id", r["patient_id"]))
            case_rows.append(scores)
        for mode, temp in [("raw", 1.0), ("source_temperature", temperature)]:
            prob = (logits/temp).softmax(0)
            if external:
                foreground = prob[fg].sum(0)
                prob = torch.stack([1-foreground, foreground])
            cal = probability_metrics(prob.numpy(), gt)
            cal.update(patient_id=r["patient_id"], recording_id=r.get("recording_id", r["patient_id"]))
            calibration[mode].append(cal)
        if save_predictions:
            dest = out/"predictions"
            dest.mkdir(exist_ok=True)
            Image.fromarray(prediction).save(dest/(r["case_id"]+".png"))
            from .data import normalize_image
            gray = (normalize_image(store.native(r, with_mask=False)[0])*255).astype(np.uint8)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            palette = {1: np.array([45,127,249]), 2: np.array([0,166,166]), 3: np.array([230,159,0])}
            panels = []
            for segmentation in [gt, prediction]:
                panel = rgb.copy()
                for c, color in palette.items():
                    region = segmentation == c
                    panel[region] = np.round(0.65*panel[region]+0.35*color).astype(np.uint8)
                panels.append(panel)
            overlays = out/"overlays"
            overlays.mkdir(exist_ok=True)
            # Exploratory native-grid panel: reference left, prediction right; no fabricated clinical image.
            Image.fromarray(np.concatenate(panels, axis=1)).save(overlays/(r["case_id"]+".png"))
    summary, patients, patient_classes = aggregate_case_metrics(case_rows)
    if summary["dice"] is None:
        write_csv(out/"unscorable_frames.csv", case_rows)
        raise ValueError("No nonempty reference target can be scored; see unscorable_frames.csv")
    summary.update(split=split, method=state["contract"]["method"], seed=state["contract"]["seed"],
                   label_fraction=state["contract"]["label_fraction"], frames=len(records),
                   external_target=store.manifest.get("external_target"), synthetic=store.manifest.get("synthetic", False),
                   checkpoint_epoch=state["epoch"], temperature=temperature, dataset_signature=digest(store.manifest),
                   inference_precision="float16_amp" if amp else "float32",
                   numerical_precision=state.get("numerical_precision"))
    summary["calibration"] = {k: aggregate_calibration(v) for k, v in calibration.items()}
    write_csv(out/"per_frame_class.csv", case_rows)
    write_csv(out/"per_patient.csv", patients)
    write_csv(out/"per_patient_class.csv", patient_classes)
    # Each stratum is recomputed with equal patient weighting rather than a pooled frame mean.
    strata = []
    for field in ["view", "phase", "quality", "class_id"]:
        for value in sorted({r[field] for r in case_rows}, key=str):
            sub, _, _ = aggregate_case_metrics([r for r in case_rows if r[field] == value])
            strata.append({"stratum": field, "value": value, **sub})
    write_csv(out/"stratified.csv", strata)
    write_json(out/"summary.json", summary)
    print(f"{split} patient macro Dice={summary['dice']:.5f}; results saved to {out}", flush=True)
    return summary


@torch.no_grad()
def benchmark(checkpoint, manifest_path, output, require_cuda=True, warmup=50, repetitions=200):
    device = device_check(require_cuda)
    model, state = load_model_for_evaluation(checkpoint, device)
    store = RecordStore(manifest_path, state["contract"]["config"]["image_size"])
    if digest(store.manifest) != state["contract"]["manifest_hash"]:
        raise ValueError("Benchmark must use the checkpoint's source data manifest")
    r = next(r for r in store.manifest["records"] if r["split"] == "val")
    x, _, meta = store.canonical(r, with_mask=False)
    x = x[None].to(device)
    v = torch.tensor([meta["view_id"]], device=device)
    amp = effective_amp(state["contract"]["method"], state["contract"]["config"], device)
    for _ in range(warmup):
        with autocast(device, amp):
            prediction = model(x,v)
    if warmup:
        require_finite(prediction["logits"], "benchmark logits")
    times = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast(device, amp):
            prediction = model(x,v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    if repetitions:
        require_finite(prediction["logits"], "benchmark logits")
    write_json(output, {"batch_size": 1, "warmup": warmup, "repetitions": repetitions,
                        "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times,95)),
                        "scope": "model forward only; no preprocessing or IO or robot control",
                        "hardware": system_info(), "synthetic": store.manifest.get("synthetic", False),
                        "inference_precision": "float16_amp" if amp else "float32"})
