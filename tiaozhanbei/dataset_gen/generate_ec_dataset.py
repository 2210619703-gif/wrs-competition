#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 EC 单样本数据集（结构对齐 ``dataset_learn/0001``）。

全部零件为 ``Bugle screws1.stl``，姿态含 normal / inverted / fallen，
位置落在 Panthera-HT 可达区且避开料盘。

用法（仓库根目录）::
    python tiaozhanbei/dataset_gen/generate_ec_dataset.py
    python tiaozhanbei/dataset_gen/generate_ec_dataset.py --n-parts 12 --sample-id 0502
    python tiaozhanbei/dataset_gen/generate_ec_dataset.py --n-parts 48   # 一格一钉
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs import wd, rm, mgm, mcm
import wrs.basis.data_adapter as da

from tiaozhanbei.sim.environment import (
    MODEL_SCALE,
    gen_desk,
    attach_desk_stripe,
)
from tiaozhanbei.dataset_gen.sim_data_camera import SimDataCamera
from tiaozhanbei.sim.ec.ec_layout import (
    BUGLE_STL,
    PART_CLASS,
    SLOT_COLS,
    SLOT_ROWS,
    choose_state,
    layout_dict,
    sample_part_positions,
    sample_pose_for_state,
    save_layout_json,
    state_from_rpy,
)


def _load_reach_pts():
    cache = os.path.join(_TB_DIR, "dataset_learn", "_reach_xy_cache.npy")
    cache = os.path.abspath(cache)
    if os.path.isfile(cache):
        pts = np.load(cache)
        if pts.ndim == 2 and pts.shape[1] == 2:
            print(f"[reach] loaded {len(pts)} pts from {cache}")
            return pts
    return None


def _load_screw_on_table(roll: float, pitch: float, yaw: float):
    mesh = da.trm.load(BUGLE_STL)
    mesh.apply_scale(np.array([MODEL_SCALE, MODEL_SCALE, MODEL_SCALE]))
    bounds = mesh.bounds
    xy_c = (bounds[0, :2] + bounds[1, :2]) / 2.0
    mesh.apply_translation(np.array([-xy_c[0], -xy_c[1], 0.0]))
    H = np.eye(4)
    H[:3, :3] = rm.rotmat_from_euler(float(roll), float(pitch), float(yaw))
    mesh.apply_transform(H)
    mesh.apply_translation(np.array([0.0, 0.0, -float(mesh.bounds[0, 2])]))
    height = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])
    return mesh, height


def build_ec_scene(
    n_parts: int,
    seed: int,
    desk_stripe: str = "vert_gray",
    show_frame: bool = False,
):
    rng = np.random.default_rng(seed)
    reach = _load_reach_pts()
    positions = sample_part_positions(n_parts, seed=seed + 1, reach_pts=reach)

    base = wd.World(cam_pos=[1.2, -0.9, 0.85], lookat_pos=[0.05, 0.0, 0.05])
    if show_frame:
        mgm.gen_frame(ax_length=0.12).attach_to(base)
    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    desk_stripe_info = attach_desk_stripe(base, style=desk_stripe, seed=seed)

    colors = [
        rm.vec(0.80, 0.42, 0.36),
        rm.vec(0.42, 0.62, 0.86),
        rm.vec(0.48, 0.75, 0.46),
        rm.vec(0.82, 0.70, 0.38),
        rm.vec(0.65, 0.50, 0.75),
    ]
    part_infos = []
    for i in range(n_parts):
        st = choose_state(rng)
        roll, pitch, yaw = sample_pose_for_state(rng, st)
        st = state_from_rpy(roll, pitch, yaw)
        mesh, height = _load_screw_on_table(roll, pitch, yaw)
        rgb = colors[i % len(colors)]
        model = mcm.CollisionModel(mesh, name=f"{PART_CLASS}_{i+1}", rgb=rgb)
        pos = rm.vec(float(positions[i][0]), float(positions[i][1]), 0.0)
        rotmat = rm.rotmat_from_euler(float(roll), float(pitch), float(yaw))
        model.pos = pos
        model.rotmat = np.eye(3)  # 旋转已烘焙进 mesh
        model.attach_to(base)
        part_infos.append(
            {
                "id": i + 1,
                "name": PART_CLASS,
                "class": PART_CLASS,
                "state": st,
                "model": model,
                "pos": np.asarray(pos, dtype=float),
                "rotmat": rotmat,  # 供 capture pose_6d 使用
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "height": height,
                "stl_path": os.path.abspath(BUGLE_STL),
                "rel_path": "Bugle screws1.stl",
            }
        )
    return {
        "base": base,
        "part_infos": part_infos,
        "desk_stripe": desk_stripe_info,
    }


def main():
    p = argparse.ArgumentParser(description="生成 EC Bugle screw 装箱数据集样本")
    p.add_argument("--sample-id", default="0502")
    p.add_argument(
        "--n-parts",
        type=int,
        default=12,
        help=f"螺丝数量（料盘共 {SLOT_COLS*SLOT_ROWS} 格；可设为该值一格一钉）",
    )
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--stripe", default="vert_gray")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--show-frame", action="store_true", help="调试用：渲染三维坐标轴")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    n_slots = SLOT_COLS * SLOT_ROWS
    n_parts = int(args.n_parts)
    if n_parts < 1:
        raise ValueError("--n-parts 至少为 1")
    if n_parts > n_slots:
        print(f"[warn] n_parts={n_parts} > n_slots={n_slots}，仍生成，但无法一格一钉")

    out_dir = os.path.join(_TB_DIR, "dataset_learn", args.sample_id)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    save_layout_json()

    scene = build_ec_scene(
        n_parts,
        seed=args.seed,
        desk_stripe=args.stripe,
        show_frame=args.show_frame,
    )
    base = scene["base"]
    part_infos = scene["part_infos"]

    camera = SimDataCamera(
        base=base,
        cam_pos=np.array([0.65, -0.95, 0.75]),
        lookat_pos=np.array([0.0, 0.0, 0.03]),
        resolution=np.array([args.width, args.height]),
        fov=45.0,
        near=0.01,
        far=5.0,
    )
    meta = camera.capture_learn_format(
        part_infos=part_infos,
        output_dir=out_dir,
        sample_id=args.sample_id,
        pointcloud_stride=2,
    )

    # 补写 state / 统一 stl / scene_layout
    by_id = {int(info["id"]): info for info in part_infos}
    for obj in meta.get("objects") or []:
        info = by_id.get(int(obj.get("id")))
        if info is None:
            continue
        obj["state"] = info["state"]
        obj["class"] = PART_CLASS
        obj["stl_path"] = os.path.abspath(BUGLE_STL)
        pose = obj.get("pose_6d") or {}
        pose["roll"] = float(info["roll"])
        pose["pitch"] = float(info["pitch"])
        pose["yaw"] = float(info["yaw"])
        pose["x"] = float(info["pos"][0])
        pose["y"] = float(info["pos"][1])
        pose["z"] = 0.0
        obj["pose_6d"] = pose

    meta["seed"] = args.seed
    meta["desk_stripe"] = {
        "style": scene["desk_stripe"]["style"],
        "texture_path": scene["desk_stripe"]["texture_path"],
    }
    meta["part_paths"] = ["Bugle screws1.stl"] * n_parts
    meta["scene_layout"] = layout_dict()
    meta["n_parts"] = n_parts
    meta["n_slots"] = n_slots
    meta["manifest_round"] = 1
    meta["ec_note"] = (
        "全零件为 Bugle screws1.stl；料盘 box.stl 采集时不渲染；"
        "仿真用 run_ec_bin_sim.py + 智能体 task JSON"
    )

    ann_path = os.path.join(out_dir, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    states = {}
    for o in meta["objects"]:
        states[o.get("state", "?")] = states.get(o.get("state", "?"), 0) + 1
    print(f"[ec] saved -> {os.path.abspath(out_dir)}")
    print(f"[ec] n_parts={n_parts} n_slots={n_slots} states={states}")
    print(f"[ec] annotations -> {ann_path}")

    if args.show:
        base.run()
    else:
        try:
            base.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
