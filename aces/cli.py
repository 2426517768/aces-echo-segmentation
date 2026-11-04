from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .utils import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACES reproducible public-data segmentation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settings", default="settings.json")
    p = sub.add_parser("prepare-external")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", required=True, choices=["myocardium", "epicardial_area"])
    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="aces")
    p.add_argument("--settings", default="settings.json")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--labels", type=float, default=0.25)
    p.add_argument("--allow-cpu", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["val", "test", "external"])
    p.add_argument("--temperature-json")
    p.add_argument("--allow-cpu", action="store_true")
    p = sub.add_parser("calibrate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    sub.add_parser("doctor")
    p = sub.add_parser("smoke")
    p.add_argument("--out", default="smoke_only")
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        from .data import prepare_camus
        prepare_camus(args.root, args.out, read_json(args.settings)["data"])
    elif args.cmd == "prepare-external":
        from .data import prepare_external
        prepare_external(args.csv, args.out, args.target)
    elif args.cmd == "train":
        from .engine import train
        train(args.manifest, args.out, args.method, read_json(args.settings)["training"], args.seed, args.labels, not args.allow_cpu)
    elif args.cmd == "evaluate":
        from .engine import evaluate
        temp = read_json(args.temperature_json)["temperature"] if args.temperature_json else 1.0
        evaluate(args.checkpoint, args.manifest, args.out, args.split, temp, not args.allow_cpu)
    elif args.cmd == "calibrate":
        from .engine import fit_temperature
        fit_temperature(args.checkpoint, args.manifest, args.out, not args.allow_cpu)
    elif args.cmd == "report":
        from .report import export_report
        export_report(args.runs, args.out)
    elif args.cmd == "doctor":
        from .utils import device_check, system_info
        import json
        device_check(True)
        print(json.dumps(system_info(), ensure_ascii=False, indent=2))
    elif args.cmd == "smoke":
        from .synthetic import make_synthetic
        from .engine import train, evaluate, fit_temperature
        out = Path(args.out)
        manifest = make_synthetic(out/"SYNTHETIC_DATA")
        cfg = read_json("settings.json")["training"]
        cfg.update(image_size=64, base_channels=8, graph_grid=4, graph_dim=32, graph_heads=4,
                   epochs=2, steps_per_epoch=2, warmup_epochs=0, ramp_epochs=1,
                   micro_batch=1, effective_batch=2, log_every_updates=1, checkpoint_every_updates=1, amp=False)
        best = train(manifest, out/"SYNTHETIC_RUN", "aces", cfg, 17, 0.5, require_cuda=False)
        temp = fit_temperature(best, manifest, out/"SYNTHETIC_RUN"/"temperature.json", require_cuda=False, pixels_per_patient=256)
        evaluate(best, manifest, out/"SYNTHETIC_RUN"/"test", temperature=temp, require_cuda=False)
        write_json(out/"SMOKE_STATUS.json", {"software_smoke_test": "passed", "clinical_results": False})
        print("合成数据流程自检结束；这些分数不是临床实验结果。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
