"""在任意电脑上训练 零件_yolo（113 类 YOLO-seg）。

用法:
  python train_parts_yolo.py              # 只检查数据和 GPU
  python train_parts_yolo.py --train      # 有 GPU 会自动用 0 号卡
  python train_parts_yolo.py --train --device 0 --epochs 100 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data.yaml"
DEFAULT_MODEL = ROOT / "yolo26s-seg.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 零件 YOLO-seg（换机版）")
    p.add_argument("--data", type=Path, default=None, help="YOLO data.yaml；默认本目录 dataset/data.yaml")
    p.add_argument("--train", action="store_true")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=0, help="0=自动：GPU 用 8，CPU 用 2")
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=-1, help="-1=自动：GPU 用 4，CPU 用 0")
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name", default="train-parts-yolo-seg")
    p.add_argument("--patience", type=int, default=30)
    return p.parse_args()


def rewrite_yaml_path() -> None:
    """把 data.yaml 的 path 写成当前机器上的绝对路径。"""
    text = DATA.read_text(encoding="utf-8")
    local = DATA.parent.resolve().as_posix()
    lines = []
    done = False
    for line in text.splitlines():
        if line.startswith("path:") and not done:
            lines.append(f"path: {local}")
            done = True
        else:
            lines.append(line)
    if not done:
        lines.insert(0, f"path: {local}")
    DATA.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_split() -> tuple[int, int]:
    img = DATA.parent / "images"
    tr = len(list((img / "train").glob("*.*"))) if (img / "train").is_dir() else 0
    va = len(list((img / "val").glob("*.*"))) if (img / "val").is_dir() else 0
    return tr, va


def main() -> None:
    args = parse_args()
    global DATA
    if args.data is not None:
        DATA = args.data if args.data.is_absolute() else (ROOT / args.data)
    if not DATA.is_file():
        raise SystemExit(f"找不到 {DATA}")

    rewrite_yaml_path()
    yaml_text = DATA.read_text(encoding="utf-8")
    train_n, val_n = count_split()
    nc_line = next((ln for ln in yaml_text.splitlines() if ln.startswith("nc:")), "nc: ?")

    print("零件 YOLO-seg 训练（实机 113 类）")
    print("=" * 50)
    print(f"data: {DATA}")
    print(f"{nc_line}   train={train_n} val={val_n}")
    print(f"model: {args.model}")
    if train_n == 0 or val_n == 0:
        raise SystemExit(
            "dataset/images 或 labels 为空。\n"
            "请把真实 YOLO 数据放到本目录 dataset/ 下：\n"
            "  dataset/images/train、dataset/images/val\n"
            "  dataset/labels/train、dataset/labels/val\n"
            "或指定已有配置：python train_parts_yolo.py --data 某路径/data.yaml --train"
        )
    if train_n != 3819 or val_n != 959:
        print(f"[WARN] 预期 train=3819 val=959，当前 train={train_n} val={val_n}")

    if not args.train:
        print()
        print("检查通过。开始训练请执行:")
        print("  python train_parts_yolo.py --train --device 0")
        return

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请先安装 ultralytics 和 torch，见 使用说明.md") from exc

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "none"
    if args.batch <= 0:
        args.batch = 8 if has_cuda else 2
    if args.workers < 0:
        args.workers = 4 if has_cuda else 0
    if not args.device and has_cuda:
        args.device = "0"
    if not has_cuda:
        print("[WARN] 未检测到 CUDA，将走 CPU，会非常慢。")

    print(f"[INFO] cuda={has_cuda} gpu={gpu_name} device={args.device or 'cpu'} "
          f"batch={args.batch} epochs={args.epochs} workers={args.workers}")

    model = YOLO(str(model_path))
    kw = dict(
        data=str(DATA.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / args.project),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.1,
        close_mosaic=10,
        patience=args.patience,
        cos_lr=True,
        warmup_epochs=3.0,
    )
    if args.device:
        kw["device"] = args.device
    model.train(**kw)
    best = ROOT / args.project / args.name / "weights" / "best.pt"
    print(f"\n[DONE] {best}")


if __name__ == "__main__":
    main()
