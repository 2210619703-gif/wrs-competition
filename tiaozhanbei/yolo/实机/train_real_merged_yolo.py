"""单件 + 多件实拍合并的 113 类 YOLO-seg。

交件权重 `best.pt` 来自实验 `train-real-merged-seg-2`。
本脚本默认与那次训练一致：从官方 `yolo26s-seg.pt` 训 100 epoch，
结果写到 `runs/segment/train-real-merged-seg-2/`，不覆盖本目录 `best.pt`。

用法:
  python train_real_merged_yolo.py
  python train_real_merged_yolo.py --train --device 0
  python train_real_merged_yolo.py --train --model best.pt --device 0 --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data.yaml"
START_MODEL = ROOT / "yolo26s-seg.pt"
SUBMITTED = ROOT / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单件+多件合并 YOLO-seg（与 train-real-merged-seg-2 一致）")
    p.add_argument("--train", action="store_true")
    p.add_argument("--model", default="", help="起始权重；默认本目录 yolo26s-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=0, help="0=自动：GPU 用 8，CPU 用 2")
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=-1, help="-1=自动：GPU 用 4，CPU 用 0")
    p.add_argument("--freeze", type=int, default=0, help="冻结前 N 层；交件那次为 0")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name", default="train-real-merged-seg-2")
    p.add_argument("--patience", type=int, default=30)
    return p.parse_args()


def rewrite_yaml_path() -> None:
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


def count_split() -> tuple[int, int, int, int]:
    img = DATA.parent / "images"
    tr = list((img / "train").glob("*.*")) if (img / "train").is_dir() else []
    va = list((img / "val").glob("*.*")) if (img / "val").is_dir() else []
    scene_tr = sum(1 for p in tr if p.name.startswith("scene_"))
    scene_va = sum(1 for p in va if p.name.startswith("scene_"))
    return len(tr), len(va), scene_tr, scene_va


def pick_model(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for p in (START_MODEL, SUBMITTED):
        if p.is_file():
            return p
    raise FileNotFoundError("找不到 yolo26s-seg.pt 或 best.pt")


def main() -> None:
    args = parse_args()
    if not DATA.is_file():
        raise SystemExit(f"找不到 {DATA}")

    rewrite_yaml_path()
    yaml_text = DATA.read_text(encoding="utf-8")
    train_n, val_n, scene_tr, scene_va = count_split()
    nc_line = next((ln for ln in yaml_text.splitlines() if ln.startswith("nc:")), "nc: ?")

    print("单件 + 多件 实拍 YOLO-seg（train-real-merged-seg-2）")
    print("=" * 50)
    print(f"data: {DATA}")
    print(f"{nc_line}   train={train_n} val={val_n}")
    print(f"  其中多件 scene_*  train={scene_tr} val={scene_va}")
    if train_n == 0 or val_n == 0:
        raise SystemExit("dataset/images 为空。")
    if train_n != 3884 or val_n != 975:
        print("[WARN] 预期 train=3884 val=975（单件 3819+959，多件 65+16）")
    if scene_tr + scene_va == 0:
        print("[WARN] 没有 scene_ 前缀的多件图，可能只拷了单件集")

    model_path = pick_model(args.model)
    print(f"model: {model_path}")

    if not args.train:
        print()
        print("检查通过。开始训练请执行:")
        print("  python train_real_merged_yolo.py --train --device 0")
        return

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请先安装 ultralytics 和带 CUDA 的 torch，见 使用说明.md") from exc

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

    print(
        f"[INFO] cuda={has_cuda} gpu={gpu_name} device={args.device or 'cpu'} "
        f"batch={args.batch} epochs={args.epochs} freeze={args.freeze} workers={args.workers}"
    )
    print("[INFO] 输出目录不会覆盖本目录 best.pt")

    model = YOLO(str(model_path))
    kw = dict(
        data=str(DATA.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / args.project),
        name=args.name,
        exist_ok=False,
        workers=args.workers,
        lr0=args.lr0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.15,
        close_mosaic=10,
        patience=args.patience,
        cos_lr=True,
        warmup_epochs=3.0,
        cls_remap=True,
    )
    if args.freeze > 0:
        kw["freeze"] = args.freeze
    if args.device:
        kw["device"] = args.device
    model.train(**kw)
    best = ROOT / args.project / args.name / "weights" / "best.pt"
    print(f"\n[DONE] {best}")
    print("[INFO] 需要更新交件权重时，把该文件拷回本目录 best.pt。")


if __name__ == "__main__":
    main()
