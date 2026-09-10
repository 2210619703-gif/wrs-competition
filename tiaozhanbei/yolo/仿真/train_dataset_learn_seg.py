"""
使用 dataset_learn 转换后的 YOLO-seg 数据集训练实例分割模型。

流程:
  1. python convert_dataset_learn_to_yolo_seg.py
  2. python train_dataset_learn_seg.py --train --model yolov8s-seg.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAIN = "yolo26s-seg.pt"


def data_yaml(fine: bool) -> Path:
    d = "dataset_yolo_learn_seg_fine" if fine else "dataset_yolo_learn_seg"
    return ROOT / d / "data.yaml"


def count_images(fine: bool) -> tuple[int, int]:
    d = "dataset_yolo_learn_seg_fine" if fine else "dataset_yolo_learn_seg"
    train_n = len(list((ROOT / d / "images" / "train").glob("*.*")))
    val_n = len(list((ROOT / d / "images" / "val").glob("*.*")))
    return train_n, val_n


def parse_args():
    p = argparse.ArgumentParser(description="用 dataset_learn 训练 YOLO-seg")
    p.add_argument("--train", action="store_true")
    p.add_argument("--convert", action="store_true")
    p.add_argument("--fine-grained", action="store_true", help="细分类模式（124类，而非8大类）")
    p.add_argument("--model", default=DEFAULT_PRETRAIN, help="分割预训练权重")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=0, help="0=自动")
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name", default=None, help="默认按模式自动命名")
    p.add_argument("--device", default="")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    fine = args.fine_grained
    yaml_path = data_yaml(fine)

    if args.name is None:
        args.name = "train-dataset-learn-yolo26-seg-fine" if fine else "train-dataset-learn-yolo26-seg"

    if args.convert or not yaml_path.exists():
        print(f"[INFO] 转换 dataset_learn -> YOLO-seg ({'细分类' if fine else '8大类'}) ...")
        cmd = [sys.executable, str(ROOT / "convert_dataset_learn_to_yolo_seg.py")]
        if fine:
            cmd.append("--fine-grained")
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    train_n, val_n = count_images(fine) if yaml_path.exists() else (0, 0)

    if not args.train:
        print("dataset_learn YOLO-seg 训练指南")
        print("=" * 50)
        print(f"模式: {'细分类' if fine else '8大类'}")
        print(f"1) python convert_dataset_learn_to_yolo_seg.py{' --fine-grained' if fine else ''}")
        print(f"2) python train_dataset_learn_seg.py --train --model yolo26s-seg.pt --epochs 100{' --fine-grained' if fine else ''}")
        print(f"\nyaml: {yaml_path}")
        print(f"当前样本: train={train_n} val={val_n}")
        return

    if not yaml_path.exists():
        raise FileNotFoundError(f"缺少 {yaml_path}，请先 --convert")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请在 (wrs) 环境安装 ultralytics") from exc

    if args.batch <= 0:
        args.batch = 4 if train_n >= 200 else 2

    run_dir = ROOT / args.project / args.name
    last_ckpt = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last_ckpt.exists():
            raise SystemExit(f"未找到 checkpoint: {last_ckpt}")
        print(f"[INFO] 续训: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        kw = dict(resume=True)
        if args.device:
            kw["device"] = args.device
        model.train(**kw)
    else:
        print(
            f"[INFO] 分割训练 data={yaml_path} model={args.model} "
            f"epochs={args.epochs} batch={args.batch} train={train_n} val={val_n} "
            f"fine={fine}"
        )
        model = YOLO(args.model)
        kw = dict(
            data=str(yaml_path.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(ROOT / args.project),
            name=args.name,
            exist_ok=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0 if train_n >= 100 else 0.8,
            mixup=0.05,
            close_mosaic=10,
            patience=40,
            cos_lr=True,
            warmup_epochs=3.0,
        )
        if args.device:
            kw["device"] = args.device
        model.train(**kw)

    best = run_dir / "weights" / "best.pt"
    print(f"\n[DONE] 分割权重: {best}")
    d = "dataset_yolo_learn_seg_fine" if fine else "dataset_yolo_learn_seg"
    print(
        f"测试: python -c \"from ultralytics import YOLO; "
        f"YOLO(r'{best}').predict(r'{d}/images/val/0005.jpg', save=True)\""
    )


if __name__ == "__main__":
    main()
