#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EC 喇叭头螺丝装箱：桌面布局 / 料盘格位 / 可达摆放区。

坐标系与 ``tiaozhanbei`` 一致：桌面 z=0，臂基座在原点。
``box.stl`` 本地坐标：角点原点，尺寸约 45×60×12 mm（米制缩放后）。
格位按 STL 实测：短边(X) 6 列 × 长边(Y) 8 行（共 48 格）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EC_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_EC_DIR, os.pardir, os.pardir))
BUGLE_STL = os.path.join(
    _TB_DIR,
    "Part_Model",
    "Screws and bolts",
    "Cap screws",
    "Bugle screws",
    "Bugle screws1_ec.stl",
)
BOX_STL = os.path.join(_TB_DIR, "Part_Model", "Fixtures", "Trays", "box.stl")
PART_CLASS = "螺钉螺栓-喇叭头螺丝"

# 料盘：外廓底面角点落在 BOX_ORIGIN_XY，本地 +x/+y 沿桌面
# slot0 ≈ 夹爪舒适松爪末位；须在基座禁区外（r=0.14）
BOX_ORIGIN_XY = np.array([0.30, 0.0], dtype=float)
BOX_SIZE_M = np.array([0.045, 0.060, 0.012], dtype=float)
# 底面贴桌（勿抬高，否则视觉悬空）
BOX_LIFT_Z = 0.0
BOX_FLOOR_Z = 0.0015  # 格底相对料盘底面（mesh 内腔底）
BOX_CLEARANCE = 0.02  # 零件相对料盘外廓额外间隙


def box_world_floor_z() -> float:
    return float(BOX_LIFT_Z + BOX_FLOOR_Z)

ROBOT_KEEP_OUT_RADIUS = 0.14

# 零件摆放（可达区，避开基座与料盘）
PART_SPAWN_XY_MIN = np.array([-0.22, -0.10], dtype=float)
PART_SPAWN_XY_MAX = np.array([0.10, 0.26], dtype=float)
PART_MIN_CENTER_DIST = 0.028  # 螺丝小，可密一些

# 6×8 格（对齐 box.stl）：本地坐标（米），相对角点；格心 ≈ 半格距
SLOT_COLS = 6
SLOT_ROWS = 8
_SLOT_XS = np.linspace(0.00375, 0.04125, SLOT_COLS)
_SLOT_YS = np.linspace(0.00375, 0.05625, SLOT_ROWS)
SLOT_PITCH_X = float(_SLOT_XS[1] - _SLOT_XS[0]) if SLOT_COLS > 1 else 0.0075
SLOT_PITCH_Y = float(_SLOT_YS[1] - _SLOT_YS[0]) if SLOT_ROWS > 1 else 0.0075


def slot_local_xy(slot_id: int) -> np.ndarray:
    """slot_id: row-major，y 从小到大、同行 x 从小到大。"""
    sid = int(slot_id)
    if sid < 0 or sid >= SLOT_COLS * SLOT_ROWS:
        raise IndexError(f"slot_id={sid} 超出 [0, {SLOT_COLS * SLOT_ROWS})")
    r, c = divmod(sid, SLOT_COLS)
    return np.array([float(_SLOT_XS[c]), float(_SLOT_YS[r])], dtype=float)


def all_slots_local() -> List[Dict[str, Any]]:
    out = []
    for sid in range(SLOT_COLS * SLOT_ROWS):
        xy = slot_local_xy(sid)
        r, c = divmod(sid, SLOT_COLS)
        out.append(
            {
                "slot_id": sid,
                "row": int(r),
                "col": int(c),
                "local_xy_m": xy.tolist(),
                "floor_z_m": float(BOX_FLOOR_Z),
            }
        )
    return out


def box_keepout_xy_bounds(
    origin_xy=None, size=None, clearance: float = BOX_CLEARANCE
) -> Tuple[np.ndarray, np.ndarray]:
    o = BOX_ORIGIN_XY if origin_xy is None else np.asarray(origin_xy, dtype=float)
    s = BOX_SIZE_M if size is None else np.asarray(size, dtype=float)
    c = float(clearance)
    lo = np.array([o[0] - c, o[1] - c], dtype=float)
    hi = np.array([o[0] + s[0] + c, o[1] + s[1] + c], dtype=float)
    return lo, hi


def point_in_robot_keepout(xy) -> bool:
    p = np.asarray(xy, dtype=float)[:2]
    return float(np.dot(p, p)) < float(ROBOT_KEEP_OUT_RADIUS) ** 2


def point_in_box_keepout(xy) -> bool:
    lo, hi = box_keepout_xy_bounds()
    p = np.asarray(xy, dtype=float)[:2]
    return bool(np.all(p >= lo) and np.all(p <= hi))


def point_in_spawn_rect(xy) -> bool:
    p = np.asarray(xy, dtype=float)[:2]
    return bool(np.all(p >= PART_SPAWN_XY_MIN) and np.all(p <= PART_SPAWN_XY_MAX))


def is_valid_part_xy(xy) -> bool:
    if not point_in_spawn_rect(xy):
        return False
    if point_in_robot_keepout(xy):
        return False
    if point_in_box_keepout(xy):
        return False
    return True


def slot_world_xy(slot_id: int, origin_xy=None) -> np.ndarray:
    o = BOX_ORIGIN_XY if origin_xy is None else np.asarray(origin_xy, dtype=float)
    return o[:2] + slot_local_xy(slot_id)


def sample_part_positions(
    n: int,
    seed=None,
    min_dist: float = PART_MIN_CENTER_DIST,
    max_tries: int = 6000,
    reach_pts: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """在安全区采样 n 个互不重叠点；若给 reach_pts 则优先从可达网格抽。"""
    rng = np.random.default_rng(seed)
    positions: List[np.ndarray] = []

    def _ok(x, y) -> bool:
        if not is_valid_part_xy((x, y)):
            return False
        for p in positions:
            if (x - p[0]) ** 2 + (y - p[1]) ** 2 < float(min_dist) ** 2:
                return False
        return True

    cand = None
    if reach_pts is not None and len(reach_pts) > 0:
        mask = np.array([is_valid_part_xy(p) for p in reach_pts], dtype=bool)
        cand = reach_pts[mask]

    for _ in range(int(n)):
        placed = False
        if cand is not None and len(cand) > 0:
            order = rng.permutation(len(cand))
            for idx in order:
                x = float(cand[idx, 0] + rng.uniform(-0.006, 0.006))
                y = float(cand[idx, 1] + rng.uniform(-0.006, 0.006))
                if _ok(x, y):
                    positions.append(np.array([x, y, 0.0], dtype=float))
                    placed = True
                    break
        if not placed:
            for _try in range(int(max_tries)):
                x = float(rng.uniform(PART_SPAWN_XY_MIN[0], PART_SPAWN_XY_MAX[0]))
                y = float(rng.uniform(PART_SPAWN_XY_MIN[1], PART_SPAWN_XY_MAX[1]))
                if _ok(x, y):
                    positions.append(np.array([x, y, 0.0], dtype=float))
                    placed = True
                    break
        if not placed:
            raise RuntimeError(f"无法放下第 {len(positions)+1}/{n} 个螺丝；减小 --n-parts 或间距")
    return positions


def layout_dict() -> Dict[str, Any]:
    lo, hi = box_keepout_xy_bounds()
    return {
        "coordinate_frame": "desk_z0_robot_base_origin",
        "part_class": PART_CLASS,
        "part_stl": os.path.abspath(BUGLE_STL),
        "box_stl": os.path.abspath(BOX_STL),
        "robot": {
            "base_pos_m": [0.0, 0.0, 0.0],
            "keep_out_radius_m": float(ROBOT_KEEP_OUT_RADIUS),
            "rendered_in_dataset": False,
        },
        "storage_box": {
            "stl": os.path.abspath(BOX_STL),
            "origin_xy_m": BOX_ORIGIN_XY.tolist(),
            "size_outer_m": BOX_SIZE_M.tolist(),
            "floor_z_m": float(BOX_FLOOR_Z),
            "clearance_m": float(BOX_CLEARANCE),
            "keepout_xy_min_m": lo.tolist(),
            "keepout_xy_max_m": hi.tolist(),
            "grid": {"cols": SLOT_COLS, "rows": SLOT_ROWS, "n_slots": SLOT_COLS * SLOT_ROWS},
            "slots": all_slots_local(),
            "rendered_in_dataset": False,
            "note": "采集图不渲染料盘；仿真时按 origin_xy 加载 box.stl，按 slot_id 放入格心",
        },
        "part_spawn": {
            "xy_min_m": PART_SPAWN_XY_MIN.tolist(),
            "xy_max_m": PART_SPAWN_XY_MAX.tolist(),
            "min_center_dist_m": float(PART_MIN_CENTER_DIST),
            "avoids": ["robot_keepout", "storage_box_keepout"],
        },
    }


def save_layout_json(path: Optional[str] = None) -> str:
    if path is None:
        path = os.path.join(_EC_DIR, "box_slots.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layout_dict(), f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def sample_pose_for_state(rng: np.random.Generator, state: str):
    """按 Bugle STL 采样桌面姿态（长轴为网格 +Y，非 +Z）。

    - fallen：长轴近水平（平躺）
    - normal：长轴朝 +Z（正立）
    - inverted：长轴朝 -Z（倒立）
    """
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    if state == "fallen":
        # 平躺：roll/pitch≈0 时长轴(Y)贴桌
        return (
            float(rng.uniform(-0.15, 0.15)),
            float(rng.uniform(-0.15, 0.15)),
            yaw,
        )
    if state == "normal":
        # 正立：Rx(+90°) 使 +Y → +Z
        roll = float(0.5 * np.pi + rng.uniform(-0.1, 0.1))
        pitch = float(rng.uniform(-0.1, 0.1))
        return roll, pitch, yaw
    if state == "inverted":
        # 倒立：Rx(-90°) 使 +Y → -Z
        roll = float(-0.5 * np.pi + rng.uniform(-0.1, 0.1))
        pitch = float(rng.uniform(-0.1, 0.1))
        return roll, pitch, yaw
    return float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.15, 0.15)), yaw


def state_from_rpy(roll: float, pitch: float, yaw: float) -> str:
    """由烘焙 rpy 判定姿态。Bugle 长轴为网格 +Y，用法向 ``R[:,1]`` 对世界 Z。"""
    import wrs.basis.robot_math as rm

    R = rm.rotmat_from_euler(float(roll), float(pitch), float(yaw))
    # 旧实现误用 R[2,2]（网格 Z），会把平躺标成 normal/inverted
    long_z = float(R[2, 1])
    if long_z >= 0.70:
        return "normal"
    if long_z <= -0.70:
        return "inverted"
    return "fallen"


def choose_state(rng: np.random.Generator) -> str:
    keys = ["normal", "inverted", "fallen"]
    w = np.array([0.45, 0.25, 0.30], dtype=float)
    return str(rng.choice(keys, p=w / w.sum()))


if __name__ == "__main__":
    p = save_layout_json()
    print("[ec_layout]", p, "n_slots=", SLOT_COLS * SLOT_ROWS)
