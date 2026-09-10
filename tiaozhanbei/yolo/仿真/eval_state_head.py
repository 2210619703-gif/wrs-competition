"""
按类评估状态头，检查是否塌成全猜 normal。

用法:
  python eval_state_head.py
  python eval_state_head.py --weights runs/classify/state-head-yolo26/weights/best.pt
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from industrial_state_config import STATE_NAMES

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "权重" / "state-head-yolo26.pt"
if not DEFAULT_WEIGHTS.is_file():
    DEFAULT_WEIGHTS = ROOT / "runs" / "classify" / "state-head-yolo26" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "dataset_state_cls" / "val"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--max-per-class", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    cm = defaultdict(Counter)  # gt -> pred
    total = 0
    correct = 0

    for gt in STATE_NAMES:
        folder = args.data / gt
        files = sorted(folder.glob("*.jpg"))[: args.max_per_class]
        for f in files:
            img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            r = model.predict(img, verbose=False)[0]
            pred = r.names[int(r.probs.top1)]
            cm[gt][pred] += 1
            total += 1
            if pred == gt:
                correct += 1

    print(f"[EVAL] weights={args.weights}")
    print(f"[EVAL] overall acc={correct / max(1, total):.3f}  n={total}")
    print("gt\\pred".ljust(12), *(s.ljust(10) for s in STATE_NAMES), "recall")
    for gt in STATE_NAMES:
        row = [cm[gt][p] for p in STATE_NAMES]
        n = sum(row) or 1
        rec = cm[gt][gt] / n
        print(gt.ljust(12), *(str(v).ljust(10) for v in row), f"{rec:.3f}")
        # 若几乎全预测成某一类，提示塌缩
        pred_mode, pred_n = max(cm[gt].items(), key=lambda x: x[1]) if cm[gt] else ("?", 0)
        if pred_mode != gt and pred_n / n > 0.8:
            print(f"  !! {gt} 几乎全被判成 {pred_mode} ({pred_n}/{n})")


if __name__ == "__main__":
    main()
