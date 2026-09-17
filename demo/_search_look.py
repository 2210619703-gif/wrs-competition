# -*- coding: utf-8 -*-
"""搜更接近俯视的 LOOK。IK 目标是法兰推出来的 TCP，限位用真机 robot.cfg。"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fruit_config import LOOK_CONF_DEG
from fruit_vision import camera_homomat, flange_pose, format_conf_deg, load_handeye

HW_LIM_DEG = np.array(
    [
        [-140.688, 148.140],
        [0.000, 180.000],
        [-0.036, 205.000],
        [-124.956, 90.972],
        [-87.228, 72.972],
        [-95.004, 90.540],
    ],
    dtype=float,
)
HW_LO = np.radians(HW_LIM_DEG[:, 0])
HW_HI = np.radians(HW_LIM_DEG[:, 1])
MARGIN = np.radians(1.5)

HFOV = math.radians(82.0)
VFOV = math.radians(54.0)
Z_NEAR, Z_FAR = 0.08, 0.55
TCP_ALONG = np.array([0.0, 0.0, 0.13])

FRUITS = np.array(
    [
        [0.208, -0.112, 0.07],
        [0.273, 0.148, 0.07],
        [0.235, -0.164, 0.07],
        [0.299, 0.095, 0.07],
        [0.28, 0.00, 0.06],
        [0.18, -0.20, 0.06],
        [0.18, 0.20, 0.06],
        [0.38, -0.20, 0.06],
        [0.38, 0.20, 0.06],
    ],
    dtype=float,
)


def in_hw(q) -> bool:
    q = np.asarray(q, dtype=float)
    return bool(np.all(q >= HW_LO + MARGIN) and np.all(q <= HW_HI - MARGIN))


def describe(q, Tfc):
    C = camera_homomat(q, Tfc)
    view = C[:3, 2]
    view = view / (np.linalg.norm(view) + 1e-12)
    tilt = math.degrees(math.acos(float(np.clip(-view[2], -1.0, 1.0))))
    cam = C[:3, 3]
    fpos, frot = flange_pose(q)
    tcp = fpos + frot @ TCP_ALONG
    n_in, depths = 0, []
    for p in FRUITS:
        pc = np.linalg.inv(C) @ np.array([p[0], p[1], p[2], 1.0])
        x, y, z = pc[:3]
        if z <= Z_NEAR or z >= Z_FAR:
            continue
        if abs(x / z) <= math.tan(HFOV / 2) and abs(y / z) <= math.tan(VFOV / 2):
            n_in += 1
            depths.append(z)
    table_hit = None
    if view[2] < -0.15:
        t = -cam[2] / view[2]
        table_hit = cam[:2] + t * view[:2]
    return {
        "q": np.asarray(q, dtype=float),
        "deg": format_conf_deg(q),
        "tilt": tilt,
        "cam": cam,
        "view": view,
        "tcp": tcp,
        "flange": fpos,
        "n_in": n_in,
        "n_all": len(FRUITS),
        "depth": float(np.mean(depths)) if depths else 1e9,
        "hit": table_hit,
        "dj2": abs(math.degrees(q[1])),
    }


def score(info, look_q) -> float:
    cam = info["cam"]
    tilt = info["tilt"]
    n_miss = info["n_all"] - info["n_in"]
    z_pen = 0.0
    if cam[2] < 0.30:
        z_pen += 50.0 * (0.30 - cam[2])
    if cam[2] > 0.46:
        z_pen += 30.0 * (cam[2] - 0.46)
    x_pen = 35.0 * abs(cam[0] - 0.26)
    y_pen = 25.0 * abs(cam[1])
    hit_pen = 0.0
    if info["hit"] is not None:
        hit_pen = 40.0 * abs(info["hit"][0] - 0.26) + 20.0 * abs(info["hit"][1])
    tcp_pen = 80.0 if info["tcp"][2] < 0.16 else 0.0
    jump = float(np.linalg.norm(np.degrees(info["q"] - look_q)))
    jump_pen = 0.08 * jump
    depth_pen = 0.0
    if info["depth"] < 0.18:
        depth_pen += 40.0 * (0.18 - info["depth"])
    if info["depth"] > 0.42:
        depth_pen += 20.0 * (info["depth"] - 0.42)
    return (
        2.2 * tilt
        + 16.0 * n_miss
        + z_pen
        + x_pen
        + y_pen
        + hit_pen
        + tcp_pen
        + jump_pen
        + depth_pen
    )


def cam_rot(yaw: float, tilt: float) -> np.ndarray:
    """+Z 先朝下，再绕世界 Y 往 +X 倾 tilt，再绕竖直轴 yaw。"""
    z = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
    yaw_m = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    z = yaw_m @ z
    x = yaw_m @ x
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    return np.column_stack([x, y, z])


def ik_tcp(robot, pos, rot, seeds):
    for seed in seeds:
        q = robot.ik(tgt_pos=np.asarray(pos, dtype=float), tgt_rotmat=rot, seed_jnt_values=seed)
        if q is not None and in_hw(q):
            return np.asarray(q, dtype=float)
    q = robot.ik(tgt_pos=np.asarray(pos, dtype=float), tgt_rotmat=rot)
    if q is not None and in_hw(q):
        return np.asarray(q, dtype=float)
    return None


def main():
    from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

    Tfc = load_handeye()
    Tcf = np.linalg.inv(Tfc)
    T_flange_tcp = np.eye(4)
    T_flange_tcp[:3, 3] = TCP_ALONG
    robot = PantheraHTSglArm(enable_cc=False)
    look = np.radians(LOOK_CONF_DEG)
    base = describe(look, Tfc)
    print("=== 当前 LOOK ===")
    print(
        f"  {base['deg']}\n"
        f"  俯视偏差 {base['tilt']:.1f}°  相机 {np.round(base['cam'],3).tolist()}  "
        f"TCP z={base['tcp'][2]:.3f}  视野 {base['n_in']}/{base['n_all']}  "
        f"落点 {None if base['hit'] is None else np.round(base['hit'],3).tolist()}"
    )

    seeds = [look.copy(), robot.manipulator.home_conf.copy()]
    found = []
    n_ik = 0
    n_ok = 0
    xs = np.linspace(0.18, 0.36, 7)
    ys = np.linspace(-0.08, 0.08, 5)
    zs = np.linspace(0.32, 0.46, 5)
    yaws = [math.radians(a) for a in (0, 30, 60, 90, -30, -60, -90, 180)]
    tilts = [math.radians(a) for a in (0, 8, 15, 22)]
    for x in xs:
        for y in ys:
            for z in zs:
                for yaw in yaws:
                    for tilt in tilts:
                        R = cam_rot(yaw, tilt)
                        Twc = np.eye(4)
                        Twc[:3, :3] = R
                        Twc[:3, 3] = [x, y, z]
                        Twf = Twc @ Tcf
                        Twt = Twf @ T_flange_tcp
                        q = ik_tcp(robot, Twt[:3, 3], Twt[:3, :3], seeds)
                        n_ik += 1
                        if q is None:
                            continue
                        n_ok += 1
                        info = describe(q, Tfc)
                        info["src"] = "ik"
                        found.append(info)
                        seeds = [q] + seeds[:4]

    rng = np.random.default_rng(7)
    for _ in range(8000):
        q = rng.uniform(HW_LO + MARGIN, HW_HI - MARGIN)
        info = describe(q, Tfc)
        if info["tilt"] > 28.0:
            continue
        if info["cam"][2] < 0.28 or info["cam"][2] > 0.50:
            continue
        if info["n_in"] < 6:
            continue
        if info["tcp"][2] < 0.16:
            continue
        info["src"] = "mc"
        found.append(info)

    if not found:
        print("没有找到合格位形")
        return

    found.sort(key=lambda d: score(d, look))
    print(f"候选 {len(found)}（IK {n_ok}/{n_ik} 成功）")
    print("=== 前 8 名 ===")
    seen = []
    shown = 0
    for info in found:
        key = tuple(round(v, 0) for v in np.degrees(info["q"]))
        if any(np.allclose(key, s, atol=6.0) for s in seen):
            continue
        seen.append(key)
        cam = info["cam"]
        hit = info["hit"]
        print(
            f"[{shown+1}] {info['src']:4s}  {info['deg']}\n"
            f"     俯视 {info['tilt']:4.1f}°  视野 {info['n_in']}/{info['n_all']}  "
            f"相机 [{cam[0]:+.3f},{cam[1]:+.3f},{cam[2]:+.3f}]  "
            f"TCP z={info['tcp'][2]:.3f}  "
            f"落点 {None if hit is None else np.round(hit,3).tolist()}  "
            f"score={score(info, look):.1f}"
        )
        shown += 1
        if shown >= 8:
            break

    best = found[0]
    print("\n建议 LOOK_CONF_DEG =", best["deg"])
    print(
        f"相对当前：俯视 {base['tilt']:.1f}° → {best['tilt']:.1f}°，"
        f"视野 {base['n_in']}/{base['n_all']} → {best['n_in']}/{best['n_all']}，"
        f"相机 x {base['cam'][0]:+.3f} → {best['cam'][0]:+.3f}"
    )


if __name__ == "__main__":
    main()
