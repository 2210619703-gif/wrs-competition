#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LabelMe JSON / detect 标签 → YOLO Segment 多边形标签。

Segment 格式（每行）：
    class_id x1 y1 x2 y2 x3 y3 ...   （归一化 0~1，至少 3 个点）

用法（仓库根目录）：
    python wrs/drivers/devices/realsense/block_images/labelme_json_to_seg.py --inplace
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# 顺序=class_id；orange 追加为 4，旧红绿蓝黄标签 id 不变
CLASS_NAMES = ["red", "green", "blue", "yellow", "orange"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def _clip01(v: float) -> float:
    return min(1.0, max(0.0, v))


def polygon_line(class_id: int, points_px: list, w: int, h: int) -> str:
    coords = []
    for x, y in points_px:
        coords.append(f"{_clip01(x / w):.6f}")
        coords.append(f"{_clip01(y / h):.6f}")
    return f"{class_id} " + " ".join(coords)


def detect_line_to_polygon(parts: list[str]) -> list[tuple[float, float]] | None:
    if len(parts) < 5:
        return None
    cx, cy, bw, bh = map(float, parts[1:5])
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy - bh / 2
    x3, y3 = cx + bw / 2, cy + bh / 2
    x4, y4 = cx - bw / 2, cy + bh / 2
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def convert_json(json_path: Path, images_dir: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_name = data.get("imagePath") or (json_path.stem + ".jpg")
    # LabelMe 常写入相对路径；只用文件名在 images_dir 查找
    img_path = images_dir / Path(image_name).name
    if not img_path.is_file():
        for ext in (".jpg", ".jpeg", ".png"):
            alt = images_dir / (json_path.stem + ext)
            if alt.is_file():
                img_path = alt
                break
    if not img_path.is_file():
        print(
            f"[seg] 跳过 {json_path.name}: 在 {images_dir} 找不到 "
            f"{Path(image_name).name}（请把对应 jpg 拷到该目录）"
        )
        return []

    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip().lower()
        if label not in CLASS_TO_ID:
            print(f"[seg] 警告 {json_path.name}: 未知标签 {label!r}")
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        lines.append(polygon_line(CLASS_TO_ID[label], pts, w, h))
    return lines


def convert_detect_txt(txt_path: Path, images_dir: Path) -> list[str]:
    import cv2
    stem = txt_path.stem
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / (stem + ext)
        if p.is_file():
            img_path = p
            break
    if img_path is None:
        return []

    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    lines = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 9:
            lines.append(line.strip())
            continue
        if len(parts) != 5:
            continue
        poly_norm = detect_line_to_polygon(parts)
        if not poly_norm:
            continue
        pts_px = [(x * w, y * h) for x, y in poly_norm]
        lines.append(polygon_line(int(parts[0]), pts_px, w, h))
    return lines


def convert_split(root: Path, split: str, labels_dir: Path, detect_backup: Path) -> tuple[int, int]:
    images_dir = root / "images" / split
    src_labels = root / "labels" / split
    if not src_labels.is_dir():
        return 0, 0

    out_dir = labels_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_inst = 0
    json_stems = {p.stem for p in src_labels.glob("*.json")}

    for json_path in sorted(src_labels.glob("*.json")):
        lines = convert_json(json_path, images_dir)
        if not lines:
            continue
        (out_dir / f"{json_path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_files += 1
        n_inst += len(lines)

    for txt_path in sorted(src_labels.glob("*.txt")):
        if txt_path.stem in json_stems:
            continue
        lines = convert_detect_txt(txt_path, images_dir)
        if not lines:
            continue
        (out_dir / txt_path.name).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_files += 1
        n_inst += len(lines)

    return n_files, n_inst


def main():
    parser = argparse.ArgumentParser(description="LabelMe / detect → YOLO Segment")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="block_images 根目录",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="备份 labels → labels_detect，写入 Segment 到 labels/",
    )
    args = parser.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    labels_dir = root / "labels"
    detect_backup = root / "labels_detect"
    if args.inplace and labels_dir.is_dir() and not detect_backup.is_dir():
        shutil.copytree(labels_dir, detect_backup)
        print(f"[seg] 已备份 detect 标签 → {detect_backup}")

    if not args.inplace:
        labels_dir = root / "labels_seg"
        labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seg] 输出目录: {labels_dir}")
    total_f = total_i = 0
    for split in splits:
        nf, ni = convert_split(root, split, labels_dir, detect_backup)
        print(f"  {split}: {nf} 文件, {ni} 实例")
        total_f += nf
        total_i += ni

    print(f"[seg] 合计 {total_f} 文件, {total_i} 实例")
    print("[seg] 训练: python train_blocks_seg.py")


if __name__ == "__main__":
    main()
