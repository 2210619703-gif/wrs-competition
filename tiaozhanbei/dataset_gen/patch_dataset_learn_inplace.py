#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""原地修补 ``dataset_learn``（不生成 v2）：

1. 仅修正顶抓 IK 不可达 / 与盒或邻件冲突的零件位置；
2. 随机给一部分样本增加 roll/pitch 放置；
3. 全部 objects 写入 ``state`` ∈ {normal, inverted, fallen}；
4. 保证零件互不重叠且避开收纳盒 / 基座禁区。

用法（仓库根目录）::
    python tiaozhanbei/dataset_gen/patch_dataset_learn_inplace.py
    python tiaozhanbei/dataset_gen/patch_dataset_learn_inplace.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wrs.basis.data_adapter as da
import wrs.basis.robot_math as rm
from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

from tiaozhanbei.sim.environment import MODEL_SCALE
from tiaozhanbei.sim.scene_layout import (
    PART_MIN_CENTER_DIST,
    PART_SPAWN_XY_MAX,
    PART_SPAWN_XY_MIN,
    is_valid_part_xy,
)

HOME_CONF = np.array([0.0, 0.30, 0.30, 0.0, 0.20, 0.0], dtype=float)
DEFAULT_DATASET = os.path.join(_TB_DIR, "dataset_learn")
DEFAULT_GRASP = os.path.join(
    _TB_DIR, "sim", "grasp", "panthera_ht_block_grasps_topdown_yaw.pickle"
)
REACH_CACHE = os.path.join(_TB_DIR, "dataset_learn", "_reach_xy_cache.npy")

# 随机旋转：约 45% 样本做姿态增强；样本内对象再按权重抽样状态
ROTATE_SAMPLE_FRAC = 0.45
STATE_WEIGHTS = {"normal": 0.50, "inverted": 0.22, "fallen": 0.28}


def _resolve_stl(stl_path: str) -> str:
    if os.path.isfile(stl_path):
        return stl_path
    marker = "Part_Model"
    norm = stl_path.replace("\\", "/")
    if marker in norm:
        rel = norm.split(marker, 1)[1].lstrip("/")
        alt = os.path.join(_TB_DIR, "Part_Model", *rel.split("/"))
        if os.path.isfile(alt):
            return alt
    raise FileNotFoundError(stl_path)


_RADIUS_MEMO = {}


def mesh_xy_radius(stl_path: str, roll: float, pitch: float, yaw: float) -> float:
    """旋转后水平 AABB 外接圆半径（米），用于防重叠。"""
    key = (
        stl_path,
        round(float(roll), 2),
        round(float(pitch), 2),
        round(float(yaw), 2),
    )
    if key in _RADIUS_MEMO:
        return _RADIUS_MEMO[key]
    mesh = da.trm.load(_resolve_stl(stl_path))
    mesh.apply_scale(np.array([MODEL_SCALE, MODEL_SCALE, MODEL_SCALE]))
    bounds = mesh.bounds
    xy_c = (bounds[0, :2] + bounds[1, :2]) / 2.0
    mesh.apply_translation(np.array([-xy_c[0], -xy_c[1], 0.0]))
    H = np.eye(4)
    H[:3, :3] = rm.rotmat_from_euler(float(roll), float(pitch), float(yaw))
    mesh.apply_transform(H)
    b = mesh.bounds
    hx = 0.5 * float(b[1, 0] - b[0, 0])
    hy = 0.5 * float(b[1, 1] - b[0, 1])
    rad = float(np.hypot(hx, hy))
    _RADIUS_MEMO[key] = rad
    return rad


def state_from_rpy(roll: float, pitch: float, yaw: float) -> str:
    """按局部 +z 与世界 +z 对齐度判定三态。"""
    R = rm.rotmat_from_euler(float(roll), float(pitch), float(yaw))
    align = float(R[2, 2])  # local z · world z
    if align >= 0.70:
        return "normal"
    if align <= -0.70:
        return "inverted"
    return "fallen"


def sample_pose_for_state(rng: np.random.Generator, state: str):
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    if state == "normal":
        # 小扰动，仍视为正放
        roll = float(rng.uniform(-0.08, 0.08))
        pitch = float(rng.uniform(-0.08, 0.08))
        return roll, pitch, yaw
    if state == "inverted":
        if rng.random() < 0.5:
            return float(np.pi + rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.1, 0.1)), yaw
        return float(rng.uniform(-0.1, 0.1)), float(np.pi + rng.uniform(-0.1, 0.1)), yaw
    # fallen：侧躺
    if rng.random() < 0.5:
        roll = float(np.pi / 2.0 * rng.choice([-1.0, 1.0]) + rng.uniform(-0.15, 0.15))
        pitch = float(rng.uniform(-0.2, 0.2))
    else:
        roll = float(rng.uniform(-0.2, 0.2))
        pitch = float(np.pi / 2.0 * rng.choice([-1.0, 1.0]) + rng.uniform(-0.15, 0.15))
    return roll, pitch, yaw


def choose_state(rng: np.random.Generator) -> str:
    keys = list(STATE_WEIGHTS.keys())
    w = np.array([STATE_WEIGHTS[k] for k in keys], dtype=float)
    w /= w.sum()
    return str(rng.choice(keys, p=w))


def build_reach_cache(robot, grasp_rots, cache_path: str, force: bool = False) -> np.ndarray:
    if (not force) and os.path.isfile(cache_path):
        pts = np.load(cache_path)
        if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) > 50:
            print(f"[cache] load {len(pts)} reachable xy <- {cache_path}")
            return pts

    print("[cache] probing topdown IK grid ...")
    xs = np.linspace(float(PART_SPAWN_XY_MIN[0]), float(PART_SPAWN_XY_MAX[0]), 29)
    ys = np.linspace(float(PART_SPAWN_XY_MIN[1]), float(PART_SPAWN_XY_MAX[1]), 33)
    good = []
    t0 = time.time()
    for x in xs:
        for y in ys:
            if not is_valid_part_xy((x, y)):
                continue
            if _ik_ok(robot, grasp_rots, x, y):
                good.append((float(x), float(y)))
    pts = np.asarray(good, dtype=float)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, pts)
    print(f"[cache] {len(pts)} reachable cells in {time.time() - t0:.1f}s -> {cache_path}")
    return pts


def _ik_ok(robot, grasp_rots, x: float, y: float) -> bool:
    for R in grasp_rots:
        for z in (0.025, 0.045, 0.065, 0.085):
            q = robot.ik(
                tgt_pos=np.array([x, y, z], dtype=float),
                tgt_rotmat=R,
                seed_jnt_values=HOME_CONF,
            )
            if q is not None:
                return True
    return False


def _ik_ok_cached(xy_key, cache_dict, robot, grasp_rots, x, y):
    if xy_key in cache_dict:
        return cache_dict[xy_key]
    ok = _ik_ok(robot, grasp_rots, x, y)
    cache_dict[xy_key] = ok
    return ok


def pack_radius(mesh_r: float) -> float:
    """长条零件外接圆过大，打包时截断，避免无解；仍保证中心距下限。"""
    return float(min(max(float(mesh_r), 0.015), 0.045))


def overlaps(xy, radius, others, min_center=PART_MIN_CENTER_DIST) -> bool:
    x, y = float(xy[0]), float(xy[1])
    for ox, oy, orad in others:
        need = max(float(min_center), float(radius) + float(orad) + 0.004)
        if (x - ox) ** 2 + (y - oy) ** 2 < need ** 2:
            return True
    return False


def relocate(
    rng: np.random.Generator,
    reach_pts: np.ndarray,
    radius: float,
    occupied,
    max_tries: int = 800,
):
    """在可达点中找一个不重叠、合法的新位置。"""
    order = rng.permutation(len(reach_pts))
    for soft_min in (PART_MIN_CENTER_DIST, 0.08, 0.07):
        for idx in order[:max_tries]:
            x, y = float(reach_pts[idx, 0]), float(reach_pts[idx, 1])
            x += float(rng.uniform(-0.01, 0.01))
            y += float(rng.uniform(-0.01, 0.01))
            if not is_valid_part_xy((x, y)):
                continue
            if overlaps((x, y), radius, occupied, min_center=soft_min):
                continue
            return np.array([x, y], dtype=float)
        for idx in order:
            x, y = float(reach_pts[idx, 0]), float(reach_pts[idx, 1])
            if not is_valid_part_xy((x, y)):
                continue
            if overlaps((x, y), radius, occupied, min_center=soft_min):
                continue
            return np.array([x, y], dtype=float)
    raise RuntimeError("无法为零件找到可达且不重叠的位置")


def nearest_reach_dist(xy, reach_pts: np.ndarray) -> float:
    d = reach_pts - np.asarray(xy, dtype=float)[:2]
    return float(np.sqrt(np.min(np.sum(d * d, axis=1))))


def patch_sample(
    ann: dict,
    rng: np.random.Generator,
    robot,
    grasp_rots,
    reach_pts: np.ndarray,
    ik_memo: dict,
    apply_random_rot: bool,
):
    objs = ann.get("objects") or []
    # 先决定每个对象的目标姿态
    planned = []
    for obj in objs:
        pose = dict(obj.get("pose_6d") or {})
        yaw0 = float(pose.get("yaw", 0.0))
        if apply_random_rot:
            st = choose_state(rng)
            roll, pitch, yaw = sample_pose_for_state(rng, st)
        else:
            roll = float(pose.get("roll", 0.0) or 0.0)
            pitch = float(pose.get("pitch", 0.0) or 0.0)
            yaw = yaw0
            # 旧数据多为纯 yaw → normal；若已有姿态则按规则重算
            st = state_from_rpy(roll, pitch, yaw)
        # 统一用规则对齐 state（防止数值扰动导致标签不一致）
        st = state_from_rpy(roll, pitch, yaw)
        mesh_r = mesh_xy_radius(obj.get("stl_path") or "", roll, pitch, yaw)
        planned.append(
            {
                "obj": obj,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "state": st,
                "radius": pack_radius(mesh_r),
                "xy": np.array(
                    [float(pose.get("x", 0.0)), float(pose.get("y", 0.0))], dtype=float
                ),
            }
        )

    # 按原顺序尽量保留；冲突/不可达才挪
    occupied = []
    n_moved = 0
    n_rot = 0
    for item in planned:
        obj = item["obj"]
        old = obj.get("pose_6d") or {}
        old_rp = abs(float(old.get("roll", 0.0) or 0.0)) + abs(
            float(old.get("pitch", 0.0) or 0.0)
        )
        if abs(item["roll"]) + abs(item["pitch"]) > 0.15 and old_rp < 0.15:
            n_rot += 1

        xy = item["xy"]
        rad = item["radius"]
        key = (round(float(xy[0]), 2), round(float(xy[1]), 2))
        need_move = False
        if not is_valid_part_xy(xy):
            need_move = True
        elif overlaps(xy, rad, occupied):
            need_move = True
        elif not _ik_ok_cached(key, ik_memo, robot, grasp_rots, float(xy[0]), float(xy[1])):
            need_move = True

        if need_move:
            xy = relocate(rng, reach_pts, rad, occupied)
            n_moved += 1

        occupied.append((float(xy[0]), float(xy[1]), float(rad)))
        obj["pose_6d"] = {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "z": 0.0,
            "roll": float(item["roll"]),
            "pitch": float(item["pitch"]),
            "yaw": float(item["yaw"]),
        }
        obj["state"] = item["state"]

    return n_moved, n_rot


def main():
    p = argparse.ArgumentParser(description="原地修补 dataset_learn 位姿/状态")
    p.add_argument("--dataset-root", default=DEFAULT_DATASET)
    p.add_argument("--grasp-pickle", default=DEFAULT_GRASP)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本（调试）")
    args = p.parse_args()

    import pickle

    with open(args.grasp_pickle, "rb") as f:
        grasps = pickle.load(f)
    grasp_rots = [np.asarray(g.ac_rotmat, dtype=float) for g in grasps]
    print(f"[init] topdown grasps={len(grasp_rots)}")

    robot = PantheraHTSglArm(enable_cc=False)
    reach_pts = build_reach_cache(
        robot, grasp_rots, REACH_CACHE, force=bool(args.rebuild_cache)
    )
    if len(reach_pts) < 30:
        raise RuntimeError("可达点过少，请检查 IK / 摆放区")

    root = Path(args.dataset_root)
    sample_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]

    rng = np.random.default_rng(args.seed)
    # 预先决定哪些样本做随机旋转增强
    rotate_flags = {
        d.name: bool(rng.random() < ROTATE_SAMPLE_FRAC) for d in sample_dirs
    }

    ik_memo = {}
    total_moved = total_rot = 0
    state_hist = {"normal": 0, "inverted": 0, "fallen": 0}
    t0 = time.time()

    for i, d in enumerate(sample_dirs):
        ann_path = d / "annotations.json"
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        # 每样本独立子 rng，便于复现
        s_rng = np.random.default_rng(args.seed + int(d.name) * 9973)
        n_moved, n_rot = patch_sample(
            ann,
            s_rng,
            robot,
            grasp_rots,
            reach_pts,
            ik_memo,
            apply_random_rot=rotate_flags[d.name],
        )
        total_moved += n_moved
        total_rot += n_rot
        for obj in ann.get("objects") or []:
            st = obj.get("state") or "normal"
            state_hist[st] = state_hist.get(st, 0) + 1

        if not args.dry_run:
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=2)
                f.write("\n")

        if (i + 1) % 25 == 0 or i + 1 == len(sample_dirs):
            print(
                f"[{i+1}/{len(sample_dirs)}] moved_parts={total_moved} "
                f"rot_enhanced_objs~={total_rot} states={state_hist}",
                flush=True,
            )

    print("=" * 64)
    print(f"samples={len(sample_dirs)} dry_run={args.dry_run}")
    print(f"parts_relocated={total_moved}")
    print(f"rotate_samples={sum(1 for v in rotate_flags.values() if v)}")
    print(f"state_hist={state_hist}")
    print(f"elapsed={time.time() - t0:.1f}s")
    print("note: RGB/mask 未重渲染；仿真以 annotations.pose_6d/state 为准。")


if __name__ == "__main__":
    main()
