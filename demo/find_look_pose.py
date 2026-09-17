# -*- coding: utf-8 -*-
"""搜一个「相机尽量俯视桌面」的 LOOK 位。

为什么要俯视：深度相机只拍得到水果朝着它那半边表面，识别出来的中心天生
偏向相机。相机越斜，这个偏差落在水平方向的分量越大（要靠
SURFACE_TO_CENTER_K 去补，补得再准也是估的）；相机越接近正上方俯视，
偏差就越往高度上跑，而高度有 z_top 直接量得到，比估的可靠。

用法：
    python demo/find_look_pose.py
    python demo/find_look_pose.py --area 0.15 0.50 -0.25 0.25 --top 5

搜出来的关节角抄进 fruit_config.py 的 LOOK_CONF_DEG，然后
    python demo/fruit_real.py --mode look
先干跑看看姿态对不对，再上真机。
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fruit_config import LOOK_CONF_DEG, SURFACE_TO_CENTER_K
from fruit_vision import camera_homomat, load_handeye

# D405 深度视场（度）。手上没存 intrinsics.json 就按标称值算。
D405_HFOV_DEG = 87.0
D405_VFOV_DEG = 58.0
# D405 的有效量程，太近拍不到、太远噪声大。
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 0.55
# TCP 在法兰 +z 方向 13cm（和 panthera_gripper 的 loc_acting_center_pos 一致）
FLANGE_TO_TCP_M = 0.13
# 估残余偏差时假设的水果半径
REF_RADIUS_M = 0.035

_ROBOT = None


def robot():
    global _ROBOT
    if _ROBOT is None:
        from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

        _ROBOT = PantheraHTSglArm(enable_cc=False)
    return _ROBOT


def cam_rotmat(tilt: float, az: float, roll: float) -> np.ndarray:
    """视线偏离竖直 tilt、朝 az 方位倾，再绕视线自转 roll。"""
    v = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)])
    ref = np.cross(v, [0.0, 0.0, 1.0])
    if np.linalg.norm(ref) < 1e-6:
        ref = np.cross(v, [1.0, 0.0, 0.0])
    ref = ref / np.linalg.norm(ref)
    x = ref * math.cos(roll) + np.cross(v, ref) * math.sin(roll)
    x = x / np.linalg.norm(x)
    y = np.cross(v, x)
    return np.column_stack([x, y, v])


def cam_to_tcp(w2c: np.ndarray, t_flange_cam: np.ndarray) -> np.ndarray:
    """想让相机到这个位姿，TCP 该去哪。"""
    t_flange_tcp = np.eye(4)
    t_flange_tcp[2, 3] = FLANGE_TO_TCP_M
    return w2c @ np.linalg.inv(t_flange_cam) @ t_flange_tcp


def solve(w2c: np.ndarray, t_flange_cam: np.ndarray, seed):
    tgt = cam_to_tcp(w2c, t_flange_cam)
    q = robot().ik(tgt_pos=tgt[:3, 3], tgt_rotmat=tgt[:3, :3], seed_jnt_values=seed)
    if q is None:
        q = robot().ik(tgt_pos=tgt[:3, 3], tgt_rotmat=tgt[:3, :3])
    return None if q is None else np.asarray(q, dtype=float).reshape(-1)


def coverage(w2c: np.ndarray, grid: np.ndarray) -> float:
    """桌面目标区有多少比例落在视场里、且距离在量程内。"""
    inv = np.linalg.inv(w2c)
    pc = (inv[:3, :3] @ grid.T).T + inv[:3, 3]
    z = pc[:, 2]
    ok = (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok &= np.abs(pc[:, 0] / z) <= math.tan(math.radians(D405_HFOV_DEG / 2))
        ok &= np.abs(pc[:, 1] / z) <= math.tan(math.radians(D405_VFOV_DEG / 2))
    return float(np.mean(ok))


def jaw_shadow(w2c: np.ndarray, tcp: np.ndarray, table_z: float):
    """夹爪在桌面上挡住的位置。

    相机是装在法兰侧面、顺着夹爪方向看的，所以夹爪永远在画面里，
    在桌上留一块盲区。挑 LOOK 位时要把这块盲区甩到水果区外面。
    """
    cam = w2c[:3, 3]
    d = np.asarray(tcp, dtype=float) - cam
    if d[2] > -1e-6:
        return None
    return cam + d * ((cam[2] - table_z) / -d[2])


def score_pose(w2c: np.ndarray, tcp: np.ndarray, grid, area, table_z: float) -> dict:
    """只看相机位姿就能算的指标，用来在解 IK 之前先筛掉一批。"""
    v = w2c[:3, 2]
    shadow = jaw_shadow(w2c, tcp, table_z)
    x0, x1, y0, y1 = area
    if shadow is None:
        clear = -9.9
    else:
        # 正数=盲区在水果区外面，越大越好
        clear = float(max(x0 - shadow[0], shadow[0] - x1, y0 - shadow[1], shadow[1] - y1))
    tilt = math.degrees(math.acos(float(np.clip(-v[2], -1.0, 1.0))))
    return {
        "tilt": tilt,
        "height": float(w2c[2, 3] - table_z),
        "cover": coverage(w2c, grid),
        "shadow": shadow,
        "clear": clear,
        # 水平残余偏差：视线越竖直，表面→中心那截越不落在水平面上
        "bias_mm": SURFACE_TO_CENTER_K * REF_RADIUS_M * math.sin(math.radians(tilt)) * 1000,
        "img_up": -w2c[:3, 1],
    }


def describe(q, grid, t_flange_cam: np.ndarray, table_z: float, area) -> dict:
    """按 FK 出来的真实位姿算指标（IK 可能只给近似解）。"""
    q = np.asarray(q, dtype=float).reshape(-1)
    robot().fk(q, update=True)
    d = score_pose(
        camera_homomat(q, t_flange_cam), np.asarray(robot().gl_tcp_pos), grid, area, table_z
    )
    lim = np.asarray(robot().manipulator.jnt_ranges)
    d["q"] = q
    d["margin"] = float(np.degrees(np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q))))
    return d


def fmt(tag: str, d: dict) -> str:
    deg = ", ".join(f"{math.degrees(v):.2f}" for v in d["q"])
    s = d["shadow"]
    where = "看不到桌面" if s is None else f"({s[0]:.2f}, {s[1]:.2f})"
    inout = "区外" if d["clear"] > 0 else f"区内，离边 {-d['clear'] * 100:.0f}cm"
    return (
        f"{tag}\n"
        f"  LOOK_CONF_DEG = ({deg})\n"
        f"  视线偏离竖直 {d['tilt']:.1f}°，相机高 {d['height'] * 100:.0f}cm，"
        f"桌面覆盖 {d['cover'] * 100:.0f}%，离限位最近 {d['margin']:.0f}°\n"
        f"  3.5cm 水果的水平残余偏差 ≈ {d['bias_mm']:.0f}mm\n"
        f"  夹爪盲区落在 {where} {inout}"
    )


def rank(d: dict):
    """先要俯视，再要盲区躲开水果区，最后看限位余量。"""
    return (round(d["tilt"], 1), -round(d["clear"], 3), -d["margin"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", nargs=4, type=float, default=[0.15, 0.50, -0.25, 0.25],
                    metavar=("X0", "X1", "Y0", "Y1"), help="桌面上要看全的矩形")
    ap.add_argument("--table-z", type=float, default=0.0, help="桌面世界高度")
    ap.add_argument("--max-tilt", type=float, default=40.0, help="最大容许偏离竖直角度")
    ap.add_argument("--min-cover", type=float, default=1.0, help="桌面覆盖率门槛")
    ap.add_argument("--shadow-margin", type=float, default=0.0,
                    help="夹爪盲区至少要甩到水果区外多少米，负数=允许落在区内")
    ap.add_argument("--aim-sweep", type=float, default=0.08,
                    help="相机瞄准点在区中心附近挪动的范围，挪开才好把盲区甩出去")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    x0, x1, y0, y1 = args.area
    gx, gy = np.meshgrid(np.linspace(x0, x1, 7), np.linspace(y0, y1, 7))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, args.table_z)])
    aim = np.array([(x0 + x1) / 2, (y0 + y1) / 2, args.table_z])

    t_flange_cam = load_handeye()
    seed = np.radians(LOOK_CONF_DEG)

    print(f"[look] 目标区 x[{x0}, {x1}] y[{y0}, {y1}]，桌面 z={args.table_z}")
    print(fmt("[look] 现在的 LOOK 位：", describe(seed, grid, t_flange_cam, args.table_z, args.area)))

    tilts = [t for t in range(0, int(args.max_tilt) + 1, 5)]
    heights = [round(h, 3) for h in np.arange(0.30, 0.56, 0.025)]
    rolls = [math.radians(r) for r in range(0, 360, 10)]
    off = [round(o, 3) for o in np.arange(-args.aim_sweep, args.aim_sweep + 1e-9, 0.04)]
    aims = [aim + np.array([dx, dy, 0.0]) for dx, dy in itertools.product(off, off)]
    found: list[dict] = []
    tried = ikcalls = 0
    for t_deg in tilts:
        tilt = math.radians(t_deg)
        azs = [0.0] if t_deg == 0 else [math.radians(a) for a in range(0, 360, 30)]
        for a, az, h, roll in itertools.product(aims, azs, heights, rolls):
            tried += 1
            rot = cam_rotmat(tilt, az, roll)
            w2c = np.eye(4)
            w2c[:3, :3] = rot
            w2c[:3, 3] = a - rot[:, 2] * (h / math.cos(tilt))
            # 先用相机位姿筛，IK 贵
            pre = score_pose(w2c, cam_to_tcp(w2c, t_flange_cam)[:3, 3], grid, args.area, args.table_z)
            if pre["cover"] < args.min_cover - 1e-9 or pre["clear"] < args.shadow_margin:
                continue
            ikcalls += 1
            q = solve(w2c, t_flange_cam, seed)
            if q is None:
                continue
            d = describe(q, grid, t_flange_cam, args.table_z, args.area)
            # IK 可能给出近似解，按实际 FK 出来的姿态复核
            if (
                d["tilt"] > args.max_tilt + 1e-6
                or d["cover"] < args.min_cover - 1e-9
                or d["clear"] < args.shadow_margin
            ):
                continue
            found.append(d)
        if found:
            print(f"[look] 倾角试到 {t_deg}° 已有 {len(found)} 个解，停止往更斜的方向找")
            break

    print(f"[look] 试了 {tried} 个相机位姿，其中 {ikcalls} 个过了初筛去解 IK")
    if not found:
        print("[look] 没搜到。放宽试试：--shadow-margin -0.05、--max-tilt 60、"
              "--min-cover 0.9，或把 --area 缩小")
        return

    # 关节角相近的算同一个，留排序最靠前的
    uniq: dict[tuple, dict] = {}
    for d in sorted(found, key=rank):
        key = tuple(np.round(np.degrees(d["q"]) / 5).astype(int))
        uniq.setdefault(key, d)
    best = sorted(uniq.values(), key=rank)[: args.top]

    print(f"\n[look] 前 {len(best)} 个候选（越靠前越接近俯视）：")
    for i, d in enumerate(best, 1):
        up = d["img_up"]
        print(fmt(f"\n[{i}]", d))
        print(f"  图像上方朝世界 ({up[0]:+.2f}, {up[1]:+.2f}, {up[2]:+.2f})")
    print("\n[look] 抄进 fruit_config.py 的 LOOK_CONF_DEG，再 python demo/fruit_real.py --mode look 验证")


if __name__ == "__main__":
    main()
