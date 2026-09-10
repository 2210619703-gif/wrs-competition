#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 YOLO detect 标签 (class cx cy w h) 转为 OBB 标签 (class x1 y1 x2 y2 x3 y3 x4 y4)。

Ultralytics OBB 格式：四个角点归一化坐标，顺时针（左上→右上→右下→左下）。

用法（仓库根目录）：
    python wrs/drivers/devices/realsense/block_images/convert_detect_to_obb_labels.py --inplace
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def detect_line_to_obb(parts: list[str]) -> str | None:
    if len(parts) < 5:
        return None
    cls_id = parts[0]
    cx, cy, w, h = map(float, parts[1:5])
    corners = (
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    )
    coords = []
    for x, y in corners:
        coords.append(f"{min(1.0, max(0.0, x)):.6f}")
        coords.append(f"{min(1.0, max(0.0, y)):.6f}")
    return cls_id + " " + " ".join(coords)


def convert_label_file(src: Path, dst: Path) -> int:
    lines_out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 9:
            lines_out.append(line)
            continue
        obb = detect_line_to_obb(parts)
        if obb:
            lines_out.append(obb)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return len(lines_out)


def convert_split(dataset_root: Path, split: str, dst_root: Path) -> tuple[int, int]:
    src_dir = dataset_root / "labels" / split
    if not src_dir.is_dir():
        return 0, 0
    dst_dir = dst_root / split
    n_files = n_boxes = 0
    for src in sorted(src_dir.glob("*.txt")):
        n_boxes += convert_label_file(src, dst_dir / src.name)
        n_files += 1
    return n_files, n_boxes


def main():
    parser = argparse.ArgumentParser(description="detect 标签 → OBB 四角点标签")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="block_images 数据集根目录",
    )
    parser.add_argument("--splits", default="train,val,test", help="逗号分隔")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="备份 labels → labels_detect，再写入 OBB 到 labels/（Ultralytics 训练用）",
    )
    args = parser.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.inplace:
        backup = root / "labels_detect"
        labels = root / "labels"
        if labels.is_dir() and not backup.is_dir():
            shutil.copytree(labels, backup)
            print(f"[obb] 已备份 detect 标签 → {backup}")
        dst_root = labels
    else:
        dst_root = root / "labels_obb"

    print(f"[obb] 数据集: {root}")
    for split in splits:
        nf, nb = convert_split(root, split, dst_root)
        print(f"  {split}: {nf} 文件, {nb} 框 → {dst_root.name}/{split}/")
    print("[obb] 完成。训练: python train_blocks_obb.py")


if __name__ == "__main__":
    main()
