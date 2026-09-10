#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切换 / 备份 YOLO 标签集，避免 Segment 与 OBB 互相覆盖。

永久目录（不要手改删）：
  labels_seg/     Segment 多边形标签
  labels_obb/     OBB 四角点标签
  labels_detect/  原始 detect 水平框
  labels/         Ultralytics 训练时实际读取的目录（可切换）

用法：
  python switch_labelset.py backup-seg   # 把当前 labels/ 存成 labels_seg/
  python switch_labelset.py use-seg      # labels_seg → labels（训 Seg）
  python switch_labelset.py use-obb      # labels_obb → labels（训 OBB）
  python switch_labelset.py status
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACTIVE = ROOT / "labels"
SEG = ROOT / "labels_seg"
OBB = ROOT / "labels_obb"
DETECT = ROOT / "labels_detect"


def _count_txt(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("*/**/*.txt")) + sum(1 for _ in d.glob("*.txt"))


def _count_split(d: Path, split: str) -> int:
    p = d / split
    if not p.is_dir():
        return 0
    return len(list(p.glob("*.txt")))


def _sample_fmt(d: Path) -> str:
    split_dir = d / "train"
    if not split_dir.is_dir():
        return "empty"
    files = list(split_dir.glob("*.txt"))
    if not files:
        return "empty"
    # Use max field count among a few samples (multi polygons have >9)
    best_n = 0
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        best_n = max(best_n, len(lines[0].split()))
    n = best_n
    if n == 5:
        return "detect(5)"
    if n == 9:
        return "obb(9)"
    if n > 9 and n % 2 == 1:
        return f"segment(max {n} fields)"
    if n >= 7 and n % 2 == 1:
        return f"polygon({n})"
    return f"unknown({n})"


def status() -> None:
    print(f"[labels] root={ROOT}")
    for name, path in (
        ("labels (active)", ACTIVE),
        ("labels_seg", SEG),
        ("labels_obb", OBB),
        ("labels_detect", DETECT),
    ):
        if not path.is_dir():
            print(f"  {name}: missing")
            continue
        tr = _count_split(path, "train")
        va = _count_split(path, "val")
        print(f"  {name}: train={tr} val={va} fmt={_sample_fmt(path)}")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def backup_seg() -> None:
    if not ACTIVE.is_dir():
        raise FileNotFoundError(f"没有 {ACTIVE}")
    fmt = _sample_fmt(ACTIVE)
    if "segment" not in fmt and "fields" not in fmt:
        # still allow if many coords
        sample = next((ACTIVE / "train").glob("*.txt"), None)
        if sample:
            n = len(sample.read_text(encoding="utf-8").splitlines()[0].split())
            if n < 7:
                raise ValueError(f"当前 labels/ 不像 Segment（{fmt}），拒绝覆盖 labels_seg/")
    _copy_tree(ACTIVE, SEG)
    print(f"[labels] 已备份 Segment → {SEG}")
    status()


def use_seg() -> None:
    if not SEG.is_dir():
        raise FileNotFoundError(f"缺少 {SEG}，先运行: python switch_labelset.py backup-seg")
    _copy_tree(SEG, ACTIVE)
    print(f"[labels] 已激活 Segment → {ACTIVE}")
    status()


def use_obb() -> None:
    if not OBB.is_dir() or _count_split(OBB, "train") == 0:
        raise FileNotFoundError(
            f"缺少 {OBB}，先运行:\n"
            "  python wrs/drivers/devices/realsense/block_images/labelme_json_to_obb.py"
        )
    if not SEG.is_dir():
        # safety: auto backup current if it looks like segment
        sample = next((ACTIVE / "train").glob("*.txt"), None)
        if sample:
            n = len(sample.read_text(encoding="utf-8").splitlines()[0].split())
            if n >= 7:
                print("[labels] 激活 OBB 前自动备份当前 labels/ → labels_seg/")
                _copy_tree(ACTIVE, SEG)
    _copy_tree(OBB, ACTIVE)
    print(f"[labels] 已激活 OBB → {ACTIVE}")
    status()


def main():
    parser = argparse.ArgumentParser(description="切换 Segment / OBB 标签集")
    parser.add_argument(
        "action",
        choices=["status", "backup-seg", "use-seg", "use-obb"],
    )
    args = parser.parse_args()
    if args.action == "status":
        status()
    elif args.action == "backup-seg":
        backup_seg()
    elif args.action == "use-seg":
        use_seg()
    elif args.action == "use-obb":
        use_obb()


if __name__ == "__main__":
    main()
