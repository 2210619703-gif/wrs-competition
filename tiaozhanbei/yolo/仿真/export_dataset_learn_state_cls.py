"""
从 dataset_learn 导出「状态头」分类数据集（YOLO-cls / ImageFolder）。

每个物体：按 bbox 裁剪 RGB → 按 state 分到 normal / inverted / fallen。
少数类会过采样，缓解 normal 占比过高。

用法:
  python export_dataset_learn_state_cls.py
  python export_dataset_learn_state_cls.py --val-ratio 0.2 --pad 0.12
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from industrial_state_config import STATE_NAMES, STATE_TO_ID
from sim_dirs import DIR_DATA

ROOT = Path(__file__).resolve().parent
SRC = DIR_DATA
DST = ROOT / "dataset_state_cls"


def parse_args():
    p = argparse.ArgumentParser(description="导出状态分类裁剪图")
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--pad", type=float, default=0.12, help="bbox 外扩比例")
    p.add_argument("--min-side", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--balance",
        action="store_true",
        default=True,
        help="对少数类过采样（默认拉到与最多类同量，近似 1:1:1）",
    )
    p.add_argument("--no-balance", action="store_false", dest="balance")
    p.add_argument(
        "--balance-ratio",
        type=float,
        default=1.0,
        help="少数类目标数量 = ratio * max(各类数量)；1.0=完全均衡",
    )
    p.add_argument(
        "--max-normal-ratio",
        type=float,
        default=1.5,
        help="train 中 normal 最多保留为少数类均值的该倍数；0=不截断",
    )
    return p.parse_args()


def find_rgb(scene_dir: Path, image_name: str | None) -> Path | None:
    if image_name:
        p = scene_dir / image_name
        if p.is_file():
            return p
    for pat in ("*.jpg", "*.png", "*.jpeg"):
        cands = [x for x in scene_dir.glob(pat) if "_labeled" not in x.name]
        if cands:
            return sorted(cands)[0]
    return None


def crop_bbox(img: np.ndarray, bbox, pad: float, min_side: int) -> np.ndarray | None:
    if not bbox or len(bbox) != 4:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w - 1, x2 + pad * bw))
    y2 = int(min(h - 1, y2 + pad * bh))
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return img[y1:y2, x1:x2].copy()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.dst.exists():
        shutil.rmtree(args.dst)
    for split in ("train", "val"):
        for name in STATE_NAMES:
            (args.dst / split / name).mkdir(parents=True, exist_ok=True)

    scenes = sorted(
        d for d in args.src.iterdir() if d.is_dir() and (d / "annotations.json").is_file()
    )
    rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * args.val_ratio))
    val_set = set(scenes[:n_val])

    # (split, state, path_written) 收集后做平衡
    by_key: dict[tuple[str, str], list[Path]] = defaultdict(list)
    skipped = 0

    for scene in scenes:
        split = "val" if scene in val_set else "train"
        data = json.loads((scene / "annotations.json").read_text(encoding="utf-8"))
        rgb_path = find_rgb(scene, data.get("image"))
        if rgb_path is None:
            skipped += 1
            continue
        img = cv2.imdecode(np.fromfile(str(rgb_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        for obj in data.get("objects") or []:
            state = (obj.get("state") or "").strip().lower()
            if state not in STATE_TO_ID:
                skipped += 1
                continue
            crop = crop_bbox(img, obj.get("bbox"), args.pad, args.min_side)
            if crop is None:
                skipped += 1
                continue
            oid = int(obj.get("id") or 0)
            out = args.dst / split / state / f"{scene.name}_{oid:02d}.jpg"
            ok, buf = cv2.imencode(".jpg", crop)
            if not ok:
                skipped += 1
                continue
            buf.tofile(str(out))
            by_key[(split, state)].append(out)

    # 可选：截断 train 中过多的 normal，避免再被 balance 拉成「假均衡」
    if args.max_normal_ratio and args.max_normal_ratio > 0:
        for split in ("train",):
            minority = [
                len(by_key[(split, s)])
                for s in STATE_NAMES
                if s != "normal" and by_key[(split, s)]
            ]
            if not minority:
                continue
            avg_m = sum(minority) / len(minority)
            cap = max(1, int(avg_m * float(args.max_normal_ratio)))
            normals = by_key[(split, "normal")]
            if len(normals) > cap:
                rng.shuffle(normals)
                keep = set(normals[:cap])
                for p in normals[cap:]:
                    try:
                        p.unlink()
                    except OSError:
                        pass
                by_key[(split, "normal")] = list(keep)
                print(f"[INFO] train normal 截断 {len(normals)} -> {cap}")

    if args.balance:
        for split in ("train", "val"):
            counts = {s: len(by_key[(split, s)]) for s in STATE_NAMES}
            if not any(counts.values()):
                continue
            max_n = max(counts.values())
            # train: 与最多类对齐（默认 1:1:1）；val: 不翻倍，保持真实分布便于评估
            if split != "train":
                continue
            target = max(1, int(max_n * float(args.balance_ratio)))
            for state in STATE_NAMES:
                paths = by_key[(split, state)]
                if not paths:
                    continue
                base_paths = list(paths)
                i = 0
                while len(paths) < target:
                    src = base_paths[i % len(base_paths)]
                    i += 1
                    dst = src.with_name(f"{src.stem}_bal{i}{src.suffix}")
                    img = cv2.imdecode(
                        np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is None:
                        shutil.copy2(src, dst)
                    else:
                        # 轻量增强；避免过多上下翻转混淆 inverted
                        if i % 2 == 0:
                            img = cv2.flip(img, 1)
                        ok, buf = cv2.imencode(".jpg", img)
                        if ok:
                            buf.tofile(str(dst))
                        else:
                            shutil.copy2(src, dst)
                    paths.append(dst)
    # 统计
    print(f"[DONE] 输出: {args.dst.resolve()}")
    for split in ("train", "val"):
        parts = []
        for s in STATE_NAMES:
            n = len(list((args.dst / split / s).glob("*.jpg")))
            parts.append(f"{s}={n}")
        print(f"  {split}: " + ", ".join(parts))
    print(f"  skipped≈{skipped}  scenes={len(scenes)} val_scenes={n_val}")
    print("下一步: python train_state_head.py --train")


if __name__ == "__main__":
    main()
