#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""挑战杯桌面固定布局：机械臂禁区 + 收纳盒（仿真用，采集图中不渲染）。

坐标系与 ``environment`` / Panthera-HT 一致：桌面 z=0，臂基座在原点附近。
采集 ``dataset_learn`` 时只把布局写入 JSON，场景里不画盒/不画臂，但零件
摆放会避开这些区域，方便后续仿真直接复用坐标。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

# ---------------------------------------------------------------------------
# 机械臂（Panthera-HT）占用 / 禁摆区
# ---------------------------------------------------------------------------
ROBOT_BASE_POS = np.array([0.0, 0.0, 0.0], dtype=float)
# 基座附近圆盘：零件中心不得进入（米）
ROBOT_KEEP_OUT_RADIUS = 0.14

# ---------------------------------------------------------------------------
# 收纳盒：固定位姿；内腔足以放入库中最大件（约 0.255 m 长）
# ---------------------------------------------------------------------------
STORAGE_BOX_POS = np.array([0.30, -0.20, 0.0], dtype=float)  # 外轮廓底面中心 xy，z=桌面
STORAGE_BOX_SIZE = np.array([0.30, 0.24, 0.12], dtype=float)  # 外廓 L×W×H (m)
STORAGE_BOX_WALL = 0.008
PLACE_TCP_Z = 0.10  # 放入时 TCP 建议高度（略高于盒口）
PLACE_OBJ_Z = 0.010

# 零件相对盒外廓的额外水平间隙
BOX_CLEARANCE = 0.04

# ---------------------------------------------------------------------------
# 零件随机摆放区（在 IK 友好区，且避开臂/盒）
# 经 top-down + 多 yaw 探测：x∈[-0.30,0.35], y∈[-0.35,0.35] 大部分可达；
# 采集时收紧到下列轴对齐矩形，并再扣掉 keep-out。
# ---------------------------------------------------------------------------
PART_SPAWN_XY_MIN = np.array([-0.28, -0.28], dtype=float)
PART_SPAWN_XY_MAX = np.array([0.12, 0.28], dtype=float)
PART_MIN_CENTER_DIST = 0.09
PART_DESK_MARGIN = 0.08


def storage_box_outer_xy_bounds(
    pos=None, size=None, clearance: float = BOX_CLEARANCE
) -> Tuple[np.ndarray, np.ndarray]:
    """返回收纳盒外廓（含间隙）的 xy 轴对齐包围盒 (min, max)。"""
    if pos is None:
        pos = STORAGE_BOX_POS
    if size is None:
        size = STORAGE_BOX_SIZE
    cx, cy = float(pos[0]), float(pos[1])
    hx = 0.5 * float(size[0]) + float(clearance)
    hy = 0.5 * float(size[1]) + float(clearance)
    return (
        np.array([cx - hx, cy - hy], dtype=float),
        np.array([cx + hx, cy + hy], dtype=float),
    )


def point_in_robot_keepout(xy, center=None, radius: float = ROBOT_KEEP_OUT_RADIUS) -> bool:
    c = ROBOT_BASE_POS[:2] if center is None else np.asarray(center, dtype=float)[:2]
    d = np.asarray(xy, dtype=float)[:2] - c
    return float(np.dot(d, d)) < float(radius) ** 2


def point_in_box_keepout(xy, pos=None, size=None, clearance: float = BOX_CLEARANCE) -> bool:
    lo, hi = storage_box_outer_xy_bounds(pos=pos, size=size, clearance=clearance)
    p = np.asarray(xy, dtype=float)[:2]
    return bool(np.all(p >= lo) and np.all(p <= hi))


def point_in_spawn_rect(xy, xy_min=None, xy_max=None) -> bool:
    lo = PART_SPAWN_XY_MIN if xy_min is None else np.asarray(xy_min, dtype=float)
    hi = PART_SPAWN_XY_MAX if xy_max is None else np.asarray(xy_max, dtype=float)
    p = np.asarray(xy, dtype=float)[:2]
    return bool(np.all(p >= lo) and np.all(p <= hi))


def is_valid_part_xy(xy) -> bool:
    """零件中心是否允许：在摆放矩形内，且不进臂/盒禁区。"""
    if not point_in_spawn_rect(xy):
        return False
    if point_in_robot_keepout(xy):
        return False
    if point_in_box_keepout(xy):
        return False
    return True


def sample_part_positions(
    n: int,
    seed=None,
    min_dist: float = PART_MIN_CENTER_DIST,
    max_tries: int = 4000,
    xy_min=None,
    xy_max=None,
) -> list:
    """在安全区内采样 n 个互不重叠的 z=0 摆放点。"""
    rng = np.random.default_rng(seed)
    lo = PART_SPAWN_XY_MIN if xy_min is None else np.asarray(xy_min, dtype=float)
    hi = PART_SPAWN_XY_MAX if xy_max is None else np.asarray(xy_max, dtype=float)
    positions = []
    for _ in range(int(n)):
        placed = False
        for _try in range(int(max_tries)):
            x = float(rng.uniform(lo[0], hi[0]))
            y = float(rng.uniform(lo[1], hi[1]))
            if not is_valid_part_xy((x, y)):
                continue
            ok = True
            for px, py, _pz in positions:
                if (x - px) ** 2 + (y - py) ** 2 < float(min_dist) ** 2:
                    ok = False
                    break
            if ok:
                positions.append(np.array([x, y, 0.0], dtype=float))
                placed = True
                break
        if not placed:
            # 网格回退：仍强制落在合法区
            for gx in np.linspace(lo[0], hi[0], 8):
                for gy in np.linspace(lo[1], hi[1], 8):
                    if not is_valid_part_xy((gx, gy)):
                        continue
                    if all(
                        (gx - px) ** 2 + (gy - py) ** 2 >= float(min_dist) ** 2
                        for px, py, _ in positions
                    ):
                        positions.append(np.array([float(gx), float(gy), 0.0]))
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            raise RuntimeError(
                f"无法在安全区放下第 {len(positions) + 1}/{n} 个零件；"
                f"请减小零件数或增大摆放区"
            )
    return positions


def layout_dict() -> Dict[str, Any]:
    """写入 annotations / scene_layout.json 的固定布局摘要。"""
    box_lo, box_hi = storage_box_outer_xy_bounds()
    return {
        "coordinate_frame": "desk_z0_robot_base_origin",
        "robot": {
            "base_pos_m": ROBOT_BASE_POS.tolist(),
            "keep_out_radius_m": float(ROBOT_KEEP_OUT_RADIUS),
            "rendered_in_dataset": False,
        },
        "storage_box": {
            "pos_m": STORAGE_BOX_POS.tolist(),
            "size_outer_m": STORAGE_BOX_SIZE.tolist(),
            "wall_thickness_m": float(STORAGE_BOX_WALL),
            "clearance_m": float(BOX_CLEARANCE),
            "keepout_xy_min_m": box_lo.tolist(),
            "keepout_xy_max_m": box_hi.tolist(),
            "place_tcp_z_m": float(PLACE_TCP_Z),
            "rendered_in_dataset": False,
            "note": "采集图不渲染；后续 pick-place 仿真按此固定坐标生成盒子",
        },
        "part_spawn": {
            "xy_min_m": PART_SPAWN_XY_MIN.tolist(),
            "xy_max_m": PART_SPAWN_XY_MAX.tolist(),
            "min_center_dist_m": float(PART_MIN_CENTER_DIST),
            "avoids": ["robot_keepout", "storage_box_keepout"],
        },
    }


def save_scene_layout_json(path: Optional[str] = None) -> str:
    if path is None:
        path = os.path.join(_TB_DIR, "dataset_learn", "scene_layout.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layout_dict(), f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    out = save_scene_layout_json()
    print("[layout]", out)
    print(json.dumps(layout_dict(), ensure_ascii=False, indent=2))
