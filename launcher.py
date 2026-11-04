from __future__ import annotations
import argparse
import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, s):
        self.terminal.write(s)
        self.log.write(s)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return False


@contextmanager
def single_instance(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise RuntimeError("检测到另一个启动记录。确认没有训练正在运行后，删除 pipeline.lock 再双击。不要同时启动两个训练包。") from e
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        Path(path).unlink(missing_ok=True)


def unpack_if_present(settings):
    from aces.data import safe_extract
    from aces.utils import sha256, write_json, read_json
    raw = ROOT/settings["paths"]["camus_root"]
    candidates = sorted(set(list((ROOT/"data").glob("*CAMUS*.zip"))+list((ROOT/"data").glob("*camus*.zip"))+list(raw.glob("*.zip"))))
    for archive in candidates:
        marker = raw/(".extracted_"+archive.name+".json")
        identity = sha256(archive)
        if marker.exists() and read_json(marker).get("sha256") == identity:
            continue
        print("Extracting:", archive, flush=True)
        safe_extract(archive, raw)
        write_json(marker, {"archive": str(archive), "sha256": identity})


def ensure_data(settings, settings_path):
    from aces.data import prepare_camus, prepare_external
    from aces.utils import write_json
    unpack_if_present(settings)
    raw = ROOT/settings["paths"]["camus_root"]
    def has_images():
        return any(raw.rglob("*_ED.nii.gz")) or any(raw.rglob("*_ED.mhd")) or any(raw.rglob("*_ED.nii")) or any(raw.rglob("*_ED.mha"))
    import download_public
    if not has_images() or download_public.incomplete_dataset(raw):
        print("CAMUS 数据尚未就绪，正在读取下载清单并续传。", flush=True)
        if download_public.main() != 0:
            raise RuntimeError("未下载数据。准备好官方数据后再次双击 START.bat。")
        unpack_if_present(settings)
    if not has_images():
        raise RuntimeError("未找到可识别的 CAMUS ED 图像，请检查官方下载包结构与版本；不生成替代数据。")
    if not settings["data"].get("label_map_confirmed"):
        print("首次数据核对：请按下载版本的官方说明核对标签。默认 0=背景、1=左室腔、2=心肌、3=左房。")
        answer = input("若确认与你的数据一致，输入 YES；否则回车停止并更正 settings.json: ").strip()
        if answer != "YES":
            raise RuntimeError("标签尚未确认，未开始训练。")
        settings["data"]["label_map_confirmed"] = True
        write_json(settings_path, settings)
    manifest = prepare_camus(raw, ROOT/settings["paths"]["prepared_camus"], settings["data"])
    external = None
    external_csv = ROOT/settings["paths"]["external_csv"]
    if external_csv.exists():
        external = prepare_external(external_csv, ROOT/settings["paths"]["prepared_external"], settings["data"].get("external_target"))
    else:
        write_json(ROOT/settings["paths"]["results"]/"external_status.json", {
            "status": "not_run", "reason": "HMC-QU requires actual images, references, verified patient/frame mapping and target definition"})
        print("HMC-QU 未准备：本次仅运行 CAMUS，外部结果记为未运行。", flush=True)
    return manifest, external


def gpu_preflight(cfg):
    import torch
    from aces.models import build_model
    from aces.utils import device_check, system_info, write_json, seed_everything
    device = device_check(True)
    seed_everything(17, cfg.get("cpu_threads", 4))
    model = build_model("aces", cfg).to(device)
    teacher = build_model("aces", cfg).to(device).eval()
    x = torch.zeros(cfg["micro_batch"], 1, cfg["image_size"], cfg["image_size"], device=device)
    v = torch.zeros(cfg["micro_batch"], dtype=torch.long, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            pseudo = teacher(x, v)["logits"]
        out = model(x, v)
        loss = out["logits"].float().square().mean()+out["aux"].float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    info = system_info()
    info["preflight_peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
    info["preflight_note"] = "model/teacher and forward/backward only; optimizer and real data not benchmarked"
    write_json(ROOT/"results"/"gpu_preflight.json", info)
    print(f"GPU check passed: {info['gpu']}; probe allocated {info['preflight_peak_allocated_gib']:.2f} GiB.", flush=True)
    del model, teacher, x, v, out, pseudo, loss
    torch.cuda.empty_cache()


def run_pipeline(mode, settings_path):
    from aces.utils import read_json, write_json, digest, load_checkpoint, sha256
    from aces.engine import train, fit_temperature, evaluate, benchmark
    from aces.report import export_report
    from aces.cli import main as cli
    settings = read_json(settings_path)
    if mode == "smoke":
        return cli(["smoke", "--out", "smoke_only"])
    if mode == "tests":
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/"tests")))
        return 0 if result.wasSuccessful() else 1
    gpu_preflight(settings["training"])
    if mode == "doctor":
        return 0
    if shutil.disk_usage(ROOT).free < 15*2**30:
        raise RuntimeError("可用磁盘空间低于 15GiB。请先清理空间；完整实验建议预留至少 80GB。")
    manifest, external = ensure_data(settings, settings_path)
    if mode == "prepare":
        return 0
    runs = ROOT/settings["paths"]["runs"]
    if mode in {"main", "evaluate"}:
        jobs = [("aces", 0.25, 17)]
    else:
        seeds = settings["suite"]["seeds"]
        jobs = [(m, 0.25, s) for m in settings["suite"]["methods_25_percent"] for s in seeds]
        jobs += [(m, 1.0, s) for m in settings["suite"]["methods_100_percent"] for s in seeds]
    write_json(ROOT/settings["paths"]["results"]/"planned_jobs.json", {
        "jobs": [{"method": m, "label_fraction": f, "seed": s} for m,f,s in jobs],
        "AP_Net": "not_included_as_validated_reproduction", "evaluations_are_empirical_only": True})
    for index, (method, fraction, seed) in enumerate(jobs):
        run = runs/f"{method}_labels{round(fraction*100):03d}_seed{seed}"
        best = run/"best.pt"
        print(f"\n===== Job {index+1}/{len(jobs)}: {method}, labels={fraction:.0%}, seed={seed} =====", flush=True)
        if mode != "evaluate":
            # Calling train even for completed jobs verifies the frozen contract before reusing a checkpoint.
            best = train(manifest, run, method, settings["training"], seed, fraction)
        elif not best.exists():
            raise RuntimeError("还没有 best.pt；请先训练，不能直接生成结果。")
        current_checkpoint_hash = sha256(best)
        if (not (run/"latency.json").exists() or
            (method == "segformer_b0" and read_json(run/"latency.json").get("inference_precision") != "float32")):
            benchmark(best, manifest, run/"latency.json")
        if not (run/"temperature.json").exists() or read_json(run/"temperature.json").get("checkpoint_sha256") != current_checkpoint_hash:
            temperature = fit_temperature(best, manifest, run/"temperature.json")
        else:
            temperature = read_json(run/"temperature.json")["temperature"]
        for split, data in [("test", manifest), ("external", external)]:
            if data is None:
                continue
            evaluation_identity = {"checkpoint_sha256": current_checkpoint_hash,
                "manifest_hash": digest(read_json(data)), "temperature": temperature}
            marker = run/split/"evaluation_identity.json"
            if marker.exists() and read_json(marker) == evaluation_identity and (run/split/"summary.json").exists():
                print(f"Reusing completed {split} evaluation.", flush=True)
                continue
            evaluate(best, data, run/split, split, temperature, save_predictions=settings["suite"]["save_predictions"])
            write_json(marker, evaluation_identity)
        export_report(runs, ROOT/settings["paths"]["results"], settings["suite"]["seeds"])
    print("运行结束。查看 results/RESULTS.md 与各 run 的 CSV/JSON。", flush=True)
    return 0


def main():
    os.chdir(ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["main", "suite", "smoke", "doctor", "prepare", "evaluate", "tests"], default="main")
    parser.add_argument("--settings", default="settings.json")
    args = parser.parse_args()
    logdir = ROOT/"logs"
    logdir.mkdir(exist_ok=True)
    logpath = logdir/(time.strftime("%Y%m%d_%H%M%S")+"_"+args.mode+".log")
    with logpath.open("a", encoding="utf-8") as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            with single_instance(ROOT/"pipeline.lock"):
                return run_pipeline(args.mode, Path(args.settings))
        except KeyboardInterrupt:
            print("已停止。重新双击可继续最近检查点。")
            return 130
        except Exception as error:
            import download_public
            if isinstance(error, (download_public.DownloadError, download_public.DatasetDownloadIncomplete)):
                traceback.print_exc(file=log)
                print(str(error))
            else:
                traceback.print_exc()
            print("运行未完成；没有生成替代结果。错误已写入", logpath)
            return 1
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    raise SystemExit(main())
