#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把「重新标注」的 LabelMe OBB JSON 转成 labels_obb/（不碰 labels_seg）。

【标注规范】每块方块只点顶面 4 个角（Create Polygons，点 4 下）
类别：red / green / blue / yellow / orange

【LabelMe 操作】
  1. Open Dir → images/train（或 val）
  2. Change Save Dir → labelme_obb/train（或 val）  ← 重要：不要存到 labels/
  3. 每块：顶面四角 polygon，标签写颜色（小写英文）
  4. 保存 JSON（与图片同名）

【转换】
  python wrs/drivers/devices/realsense/block_images/labelme_json_to_obb.py

默认只读 labelme_obb/，不会用旧的 Segment 外轮廓 JSON。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# 顺序=class_id；orange 追加为 4，旧红绿蓝黄标签 id 不变
CLASS_NAMES = ["red", "green", "blue", "yellow", "orange"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _imread_unicode(path: Path):
    """cv2.imread 在 Windows 上读不了含中文路径；用 imdecode 兜底。"""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def polygon_to_obb_line(class_id: int, points_px: list, w: int, h: int) -> str | None:
    """4 点直接用；多于 4 点则 minAreaRect（兼容误标）。"""
    pts = np.asarray(points_px, dtype=np.float32)
    if len(pts) < 3:
        return None
    if len(pts) == 4:
        box = pts
    else:
        rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
        box = cv2.boxPoints(rect)
    coords = []
    for x, y in box:
        coords.append(f"{_clip01(x / w):.6f}")
        coords.append(f"{_clip01(y / h):.6f}")
    return f"{class_id} " + " ".join(coords)


def convert_json(json_path: Path, images_dir: Path) -> tuple[list[str], list[str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_name = data.get("imagePath") or (json_path.stem + ".jpg")
    img_path = images_dir / Path(image_name).name
    if not img_path.is_file():
        for ext in (".jpg", ".jpeg", ".png"):
            alt = images_dir / (json_path.stem + ext)
            if alt.is_file():
                img_path = alt
                break
    if not img_path.is_file():
        return [], [f"缺图片: {json_path.name}"]

    img = _imread_unicode(img_path)
    if img is None:
        return [], [f"无法读图: {img_path}"]
    h, w = img.shape[:2]

    lines, warns = [], []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if label not in CLASS_TO_ID:
            warns.append(f"{json_path.name}: 未知标签 {label!r}")
            continue
        st = str(shape.get("shape_type", "polygon")).lower()
        pts = shape.get("points", [])
        # 推荐: oriented_rectangle；兼容: 4点 polygon；水平 rectangle 可转但无旋转
        if st in ("oriented_rectangle", "rotation", "rotaterectangle"):
            if len(pts) < 4:
                warns.append(f"{json_path.name}/{label}: 有向矩形点数不足 ({len(pts)})")
                continue
        elif st == "rectangle":
            warns.append(f"{json_path.name}/{label}: 用了水平矩形，无旋转角（建议改用有向矩形）")
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif st == "polygon":
            if len(pts) != 4:
                warns.append(
                    f"{json_path.name}/{label}: 多边形 {len(pts)} 点（OBB 请用「有向矩形」或正好 4 点）"
                )
        else:
            warns.append(f"{json_path.name}/{label}: 跳过类型 {st}")
            continue
        line = polygon_to_obb_line(CLASS_TO_ID[label], pts, w, h)
        if line:
            lines.append(line)
    return lines, warns


def convert_split(
    root: Path,
    split: str,
    json_root: Path,
    out_root: Path,
) -> tuple[int, int, list[str]]:
    images_dir = root / "images" / split
    src_dir = json_root / split
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清空该 split 旧的自动 OBB，避免混入未重标样本
    for old in out_dir.glob("*.txt"):
        old.unlink()

    if not src_dir.is_dir():
        return 0, 0, [f"无目录 {src_dir}"]

    n_files = n_inst = 0
    warns: list[str] = []
    jsons = sorted(src_dir.glob("*.json"))
    if not jsons:
        return 0, 0, [f"{src_dir} 下还没有 JSON，请先用 LabelMe 标注"]

    for json_path in jsons:
        lines, w = convert_json(json_path, images_dir)
        warns.extend(w)
        if not lines:
            continue
        (out_dir / f"{json_path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_files += 1
        n_inst += len(lines)
    return n_files, n_inst, warns


def main():
    parser = argparse.ArgumentParser(description="仅从 labelme_obb/ 生成 labels_obb/")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--json-root",
        default=None,
        help="LabelMe JSON 根目录，默认 <root>/labelme_obb",
    )
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    root = Path(args.root)
    json_root = Path(args.json_root) if args.json_root else root / "labelme_obb"
    out_root = root / "labels_obb"

    print(f"[obb-relabel] JSON 来源: {json_root}")
    print(f"[obb-relabel] 输出:     {out_root}")
    print("[obb-relabel] 不会修改 labels_seg / Segment 权重")

    total_f = total_i = 0
    all_warns: list[str] = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        nf, ni, warns = convert_split(root, split, json_root, out_root)
        print(f"  {split}: {nf} 张图, {ni} 个框")
        total_f += nf
        total_i += ni
        all_warns.extend(warns)

    if all_warns:
        print(f"[obb-relabel] 警告 {len(all_warns)} 条（显示前 15）:")
        for w in all_warns[:15]:
            print(f"  - {w}")

    if total_f == 0:
        print(
            "\n[obb-relabel] 还没有可用标注。请：\n"
            "  1. LabelMe Open Dir = images/train\n"
            f"  2. Change Save Dir = {json_root / 'train'}\n"
            "  3. 每块顶面 4 角点完后保存\n"
            "  4. 再运行本脚本"
        )
        return

    print(f"[obb-relabel] 完成: {total_f} 图 / {total_i} 框")
    print("训练: python train_blocks_obb.py --device cpu --restore-seg")


if __name__ == "__main__":
    main()
