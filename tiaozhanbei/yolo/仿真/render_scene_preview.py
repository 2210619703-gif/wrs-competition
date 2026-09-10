#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render a simulation-style scene preview from YOLO annotations.

This is intentionally lightweight: it only reconstructs the desk, storage box,
parts, and robot home pose, then saves one screenshot for the UI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrs import wd, mgm  # noqa: E402
from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import (  # noqa: E402
    PantheraHTSglArm,
)

from tiaozhanbei.sim.environment import attach_desk_stripe, gen_desk  # noqa: E402
from tiaozhanbei.sim.run_agent_pick_place_side_sim import (  # noqa: E402
    configure_gripper_for_demo,
    load_parts_from_annotations,
    make_storage_box,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO annotations -> sim preview screenshot")
    p.add_argument("--annotations", required=True, help="YOLO 输出的 000x.json")
    p.add_argument("--out", required=True, help="保存截图路径")
    p.add_argument("--highlight-class", default="", help="需要高亮的零件 class")
    p.add_argument("--stripe", default="vert_gray")
    return p.parse_args()


def find_highlight_id(ann: dict, class_name: str) -> int | None:
    if not class_name:
        return None
    for obj in ann.get("objects") or []:
        if obj.get("class") == class_name:
            return obj.get("id")
    return None


def storage_xy(ann: dict) -> np.ndarray:
    storage = (ann.get("scene_layout") or {}).get("storage_box") or {}
    xyz = storage.get("pos_m") or [0.3, -0.2, 0.0]
    return np.asarray(xyz[:2], dtype=float)


def flatten_on_white(path: Path) -> None:
    img = Image.open(path)
    if img.mode != "RGBA":
        return
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.alpha_composite(img)
    white.convert("RGB").save(path)


def main() -> None:
    args = parse_args()
    ann_path = Path(args.annotations)
    out_path = Path(args.out)
    ann = json.loads(ann_path.read_text(encoding="utf-8"))

    highlight_id = find_highlight_id(ann, args.highlight_class)
    base = wd.World(
        cam_pos=[1.65, -1.25, 1.05],
        lookat_pos=[0.03, -0.03, 0.08],
        w=1920,
        h=1080,
    )
    base.setBackgroundColor(1, 1, 1)
    mgm.gen_frame(ax_length=0.12).attach_to(base)

    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style=args.stripe, seed=0)

    for info in load_parts_from_annotations(ann, highlight_id=highlight_id):
        info["model"].attach_to(base)

    for box_part in make_storage_box(storage_xy(ann)):
        box_part.attach_to(base)

    robot = PantheraHTSglArm(enable_cc=False)
    configure_gripper_for_demo(robot, jaw_max=0.055)
    # Keep Panthera's initial home posture exactly as WRS shows it when the
    # simulation scene first opens; do not move it to the planned first frame.
    try:
        robot.end_effector.change_jaw_width(0.055)
    except Exception:
        pass
    robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(6):
        base.graphicsEngine.renderFrame()
    ok = base.win.saveScreenshot(str(out_path))
    if ok:
        flatten_on_white(out_path)
    print(f"[preview] saved={out_path} ok={ok}", flush=True)
    try:
        base.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
