"""
训练「状态头」：三态分类 normal / inverted / fallen（YOLO-cls）。

流水线: 检测框 ROI → 本分类头 → state

用法:
  python export_dataset_learn_state_cls.py
  python train_state_head.py --train
  python train_state_head.py --train --model yolo26n-cls.pt --epochs 60
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_state_cls"
DEFAULT_MODEL = "yolo26n-cls.pt"


def parse_args():
    p = argparse.ArgumentParser(description="训练零件状态分类头")
    p.add_argument("--train", action="store_true")
    p.add_argument("--export", action="store_true", help="先导出裁剪数据")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--project", default="runs/classify")
    p.add_argument("--name", default="state-head-yolo26-v3")
    p.add_argument("--device", default="")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.export or not (DATA / "train").is_dir():
        print("[INFO] 导出 dataset_state_cls ...")
        subprocess.run(
            [sys.executable, str(ROOT / "export_dataset_learn_state_cls.py")],
            check=True,
            cwd=str(ROOT),
        )

    if not args.train:
        print("状态头训练指南")
        print("=" * 50)
        print("1) python export_dataset_learn_state_cls.py")
        print("2) python train_state_head.py --train --export --model yolo26n-cls.pt")
        print("3) python eval_state_head.py  # 检查三类召回，避免全猜 normal")
        print(f"数据: {DATA}")
        return

    if not (DATA / "train").is_dir():
        raise FileNotFoundError(f"缺少 {DATA}/train，请先 --export")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请在 conda wrs 环境安装 ultralytics") from exc

    run_dir = ROOT / args.project / args.name
    last = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last.exists():
            raise SystemExit(f"未找到 {last}")
        model = YOLO(str(last))
        kw = dict(resume=True)
        if args.device:
            kw["device"] = args.device
        model.train(**kw)
    else:
        print(
            f"[INFO] 状态头 train data={DATA} model={args.model} "
            f"epochs={args.epochs} batch={args.batch}"
        )
        model = YOLO(args.model)
        kw = dict(
            data=str(DATA.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(ROOT / args.project),
            name=args.name,
            exist_ok=True,
            patience=args.patience,
            # 分类头：少做会混淆姿态的增强
            flipud=0.0,
            fliplr=0.3,
            erasing=0.15,
            cos_lr=True,
        )
        if args.device:
            kw["device"] = args.device
        model.train(**kw)

    best = run_dir / "weights" / "best.pt"
    print(f"\n[DONE] 状态头权重: {best}")
    print(f"评估: python eval_state_head.py --weights {best}")
    print(
        "联合推理: python infer_detect_with_state.py "
        f"--state-weights {best}"
    )


if __name__ == "__main__":
    main()
