"""
用仿真 dataset_learn/annotations.json 训练 YOLO-seg，类别名与 JSON 的 class 字段一致。

best.pt 是 124 类中文细分类分割权重（YOLO26s-seg）。本脚本默认从它继续微调，
推理 JSON 由 infer_seg_annotations.py 导出，字段对齐仿真 annotations.json。

流程:
  1. 把 annotations.json 的 polygon/mask 转成 YOLO-seg 标签（细分类）
  2. 从 best.pt 微调，类别顺序锁定为权重里的 names
  3. 可选：导出一张图，验证 JSON 格式

用法（conda activate wrs）:
  # 转换 + 从 best.pt 微调
  python train_annotations_yolo.py --train

  # 只转换（锁定 best.pt 的 124 类顺序）
  python train_annotations_yolo.py --convert

  # 续训
  python train_annotations_yolo.py --train --resume

  # 训练后导出一张 val 图为 annotations.json
  python train_annotations_yolo.py --train --export

  # 不训练，只用当前 best.pt 导出
  python infer_seg_annotations.py --image dataset_learn/0001/0001.jpg --pretty --save-masks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from sim_dirs import DIR_DATA

SRC = DIR_DATA
DST = ROOT / "dataset_yolo_learn_seg_fine"
DATA_YAML = DST / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "best.pt"
CONVERT = ROOT / "convert_dataset_learn_to_yolo_seg.py"
INFER = ROOT / "infer_seg_annotations.py"


def parse_args():
    p = argparse.ArgumentParser(description="annotations.json 对齐的 YOLO-seg 训练")
    p.add_argument("--train", action="store_true", help="启动训练")
    p.add_argument("--convert", action="store_true", help="先转换 dataset_learn")
    p.add_argument(
        "--model",
        default=str(DEFAULT_WEIGHTS),
        help="起始权重（默认项目根目录 best.pt）",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=0, help="0=按样本量自动")
    p.add_argument("--freeze", type=int, default=0, help="冻结前 N 层（微调可设 10）")
    p.add_argument("--lr0", type=float, default=None, help="初始学习率；微调建议 0.001")
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name", default="train-annotations-yolo26-seg-fine")
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--append", action="store_true", help="转换时只追加新样本，不重建整个 YOLO 集")
    p.add_argument("--min-id", type=int, default=0, help="转换时只处理编号 >= min-id 的样本")
    p.add_argument("--src", type=Path, default=None, help="仿真样本根目录，默认 dataset_learn")
    p.add_argument("--yolo-dst", type=Path, default=None, help="YOLO 数据集目录，默认 dataset_yolo_learn_seg_fine")
    p.add_argument(
        "--export",
        action="store_true",
        help="训练结束后导出一张图为 annotations.json 格式",
    )
    p.add_argument(
        "--export-image",
        type=Path,
        default=None,
        help="导出用图片；默认 dataset_learn/0001/0001.jpg 或 val 第一张",
    )
    return p.parse_args()


def count_images() -> tuple[int, int]:
    train_n = len(list((DST / "images" / "train").glob("*.*")))
    val_n = len(list((DST / "images" / "val").glob("*.*")))
    return train_n, val_n


def names_from_model(model_path: Path) -> list[str] | None:
    if not model_path.is_file():
        return None
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = getattr(model, "names", None)
    if not names:
        return None
    if isinstance(names, dict):
        return [str(names[i]) for i in range(len(names))]
    return [str(n) for n in names]


def run_convert(
    model_path: Path,
    val_ratio: float,
    append: bool = False,
    min_id: int = 0,
    src: Path | None = None,
    dst: Path | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(CONVERT),
        "--fine-grained",
        "--val-ratio",
        str(val_ratio),
    ]
    if append:
        cmd.append("--append")
    if min_id > 0:
        cmd += ["--min-id", str(min_id)]
    if src is not None:
        cmd += ["--src", str(src)]
    if dst is not None:
        cmd += ["--dst", str(dst)]
    names = names_from_model(model_path)
    names_json = (dst or DST) / "classes_zh.lock.json"
    if names:
        names_json.parent.mkdir(parents=True, exist_ok=True)
        names_json.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd += ["--class-names", str(names_json)]
        print(f"[INFO] 锁定 {len(names)} 类，与 {model_path.name} 一致")
    else:
        print("[WARN] 无法从权重读取 names，将扫描 annotations.json 重新编号")

    print("[INFO] 转换 dataset_learn -> YOLO-seg 细分类 ...")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def pick_export_image(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    for cand in (
        SRC / "0001" / "0001.jpg",
        SRC / "8156" / "8156.jpg",
    ):
        if cand.exists():
            return cand
    val_imgs = sorted((DST / "images" / "val").glob("*.*"))
    if val_imgs:
        return val_imgs[0]
    raise FileNotFoundError("没有可导出的图片，请用 --export-image 指定")


def run_export(weights: Path, image: Path, out_dir: Path) -> None:
    out_json = out_dir / "annotations.json"
    cmd = [
        sys.executable,
        str(INFER),
        "--image",
        str(image),
        "--weights",
        str(weights),
        "--out",
        str(out_json),
        "--pretty",
        "--save-masks",
    ]
    print(f"[INFO] 导出 annotations.json: {image} -> {out_json}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def print_guide(train_n: int, val_n: int) -> None:
    print("annotations.json 对齐训练指南")
    print("=" * 50)
    print("类别: annotations.json 的 class 中文细分类（与 best.pt 的 124 类一致）")
    print("输出: infer_seg_annotations.py → 与仿真 annotations.json 同字段")
    print()
    print("1) 转换:")
    print("   python train_annotations_yolo.py --convert")
    print("2) 从 best.pt 微调:")
    print("   python train_annotations_yolo.py --train --model best.pt --epochs 100")
    print("3) 推理导出（格式对齐仿真）:")
    print(
        "   python infer_seg_annotations.py "
        "--image dataset_learn/0001/0001.jpg --pretty --save-masks"
    )
    print()
    print(f"yaml: {DATA_YAML}")
    print(f"当前样本: train={train_n} val={val_n}")


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()

    src = Path(args.src) if args.src else SRC
    yolo_dst = Path(args.yolo_dst) if args.yolo_dst else DST
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    if not yolo_dst.is_absolute():
        yolo_dst = (ROOT / yolo_dst).resolve()
    data_yaml = yolo_dst / "data.yaml"

    need_convert = args.convert or not data_yaml.exists()
    if need_convert:
        if not src.is_dir():
            raise FileNotFoundError(f"未找到仿真数据集: {src}")
        run_convert(
            model_path,
            args.val_ratio,
            append=args.append,
            min_id=args.min_id,
            src=src,
            dst=yolo_dst,
        )

    train_n = len(list((yolo_dst / "images" / "train").glob("*.*"))) if data_yaml.exists() else 0
    val_n = len(list((yolo_dst / "images" / "val").glob("*.*"))) if data_yaml.exists() else 0

    if not args.train:
        print_guide(train_n, val_n)
        return

    if not data_yaml.exists():
        raise FileNotFoundError(f"缺少 {data_yaml}，请先 --convert")

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
        kw = dict(resume=True, workers=args.workers)
        if args.device:
            kw["device"] = args.device
        model.train(**kw)
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"起始权重不存在: {model_path}")
        print(
            f"[INFO] 训练 data={data_yaml} model={model_path} "
            f"epochs={args.epochs} batch={args.batch} "
            f"train={train_n} val={val_n}"
        )
        model = YOLO(str(model_path))
        kw = dict(
            data=str(data_yaml.resolve()),
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
            mosaic=1.0 if train_n >= 100 else 0.8,
            mixup=0.05,
            close_mosaic=10,
            patience=40,
            cos_lr=True,
            warmup_epochs=3.0,
        )
        if args.freeze > 0:
            kw["freeze"] = args.freeze
        if args.lr0 is not None:
            kw["lr0"] = args.lr0
        if args.device:
            kw["device"] = args.device
        model.train(**kw)

    best = run_dir / "weights" / "best.pt"
    print(f"\n[DONE] 权重: {best}")
    print(
        "导出 JSON:\n"
        f"  python infer_seg_annotations.py --image dataset_learn/0001/0001.jpg "
        f"--weights {best} --pretty --save-masks"
    )

    if args.export:
        image = pick_export_image(args.export_image)
        run_export(best if best.exists() else model_path, image, run_dir / "export_sample")


if __name__ == "__main__":
    main()
