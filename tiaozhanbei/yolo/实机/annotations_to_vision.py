"""
把 annotations.json 转成决策侧 vision_objects 数组。

用于大模型 / 抓取规划微调：name + conf + coord + z + yaw。

用法:
  python annotations_to_vision.py --ann dataset_learn/0001/yolo_annotations.json
  python annotations_to_vision.py --ann dataset_learn/0001/annotations.json --out vision.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def objects_to_vision(data: dict) -> list[dict]:
    items = []
    for obj in data.get("objects") or []:
        pose = obj.get("pose_6d") or {}
        x = float(pose.get("x") or 0.0)
        y = float(pose.get("y") or 0.0)
        z = float(pose.get("z") or 0.0)
        yaw_rad = float(pose.get("yaw") or 0.0)
        yaw_deg = yaw_rad * 180.0 / math.pi
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        item = {
            "name": obj.get("class"),
            "conf": float(obj.get("score") or 1.0),
            "coord": [round(x, 6), round(y, 6)],
            "z": round(z, 6),
            "yaw": round(yaw_deg, 3),
            "rotate_matrix": [
                [round(c, 6), round(-s, 6), 0.0],
                [round(s, 6), round(c, 6), 0.0],
            ],
            "state": obj.get("state", "normal"),
        }
        items.append(item)
    return items


def main():
    p = argparse.ArgumentParser(description="annotations.json → vision_objects")
    p.add_argument("--ann", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    data = json.loads(args.ann.read_text(encoding="utf-8"))
    items = objects_to_vision(data)
    payload = json.dumps(items, ensure_ascii=False, indent=2)
    out = args.out or args.ann.with_name("vision_objects.json")
    out.write_text(payload, encoding="utf-8")
    print(f"[DONE] {len(items)} objects → {out}")


if __name__ == "__main__":
    main()
