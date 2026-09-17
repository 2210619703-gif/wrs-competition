# -*- coding: utf-8 -*-
"""水果抓取规划：走 WRS Panthera-HT 仿真规划器（IK + 关节插值）。

不用工业件那套 grasp pickle / dataset_learn。对每个水果先试竖直顶抓，
够不着就按 GRASP_TILT_DEG 越来越斜着下手：
沿接近轴后退 → IK 抓取位 → 竖直抬起 →（可选）插值到 A/B 示教筐。
转场（去下一个水果、夹着水果去料框）会先抬到 TRANSIT_Z_M 再平移，
免得贴着桌面扫过还没抓的水果。
导出的 JSON 和 tiaozhanbei/real/traj_adapter.py 读的格式一致。
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time

import numpy as np

from fruit_config import (
    APPROACH_Z_M,
    APPROACH_Z_STEP_M,
    BIN_HEIGHT_M,
    GRASP_DEPTH_FRAC,
    GRASP_DEPTH_M,
    GRASP_EXTRA_SINK_M,
    GRASP_TILT_DEG,
    JOG_BOUNDS_M,
    JOG_Z_MIN_M,
    MIN_APPROACH_Z_M,
    MIN_GRASP_ABOVE_BOTTOM_M,
    MIN_GRASP_Z_M,
    IK_YAW_OFFSETS_DEG,
    JAW_CLOSE_M,
    LIFT_Z_M,
    OUTPUT_DIR,
    PATH_STEPS,
    REPO_ROOT,
    SIM_JAW_OPEN_MAX,
    bin_conf,
    carry_z_m,
    dest_label,
    exit_bin_z_m,
    look_conf,
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class FruitPlanError(RuntimeError):
    pass


_ROBOT = None
_ROBOT_LOCK = threading.Lock()
_WARM_LOCK = threading.Lock()
_WARMED = False
_WARM_THREAD = None


def _robot():
    global _ROBOT
    if _ROBOT is not None:
        return _ROBOT
    with _ROBOT_LOCK:
        if _ROBOT is not None:
            return _ROBOT
        from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

        robot = PantheraHTSglArm(enable_cc=False)
        ee = robot.end_effector
        jaw_max = float(SIM_JAW_OPEN_MAX)
        ee.jaw_range = np.array([0.0, jaw_max], dtype=float)
        if hasattr(ee, "close_bias"):
            ee.close_bias = 0.0
        _ROBOT = robot
        return robot


def warmup_planner(logger=None):
    """把 URDF / TracIK 加载并跑一次空 IK。第一次大约 10s，之后规划只剩 0.2s。"""
    global _WARMED
    with _WARM_LOCK:
        if _WARMED:
            return _robot()
        t0 = time.perf_counter()
        robot = _robot()
        q = np.zeros(robot.manipulator.n_dof, dtype=float)
        try:
            if conf_is_filled(LOOK_CONF_DEG):
                q = np.asarray(look_conf(), dtype=float).reshape(-1)
        except Exception:
            pass
        robot.fk(q, update=True)
        pos = np.asarray(robot.gl_tcp_pos, dtype=float)
        rot = np.asarray(robot.gl_tcp_rotmat, dtype=float)
        robot.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=q)
        _WARMED = True
        dt = time.perf_counter() - t0
        msg = f"[plan] IK / TracIK 已预热 {dt:.1f}s"
        if logger:
            logger(msg)
        else:
            print(msg)
        return robot


def warmup_planner_async(logger=None):
    """连臂的时候丢到后台，和去观察位 / 拍照重叠。"""
    global _WARM_THREAD
    if _WARMED:
        return None
    if _WARM_THREAD is not None and _WARM_THREAD.is_alive():
        return _WARM_THREAD
    t = threading.Thread(
        target=warmup_planner,
        kwargs={"logger": logger},
        daemon=True,
        name="ik-warmup",
    )
    _WARM_THREAD = t
    t.start()
    if logger:
        logger("[plan] 正在后台加载 URDF / TracIK（约 10s），和拍照重叠")
    return t


def lerp_path(q0, q1, steps: int = PATH_STEPS) -> list[list[float]]:
    a = np.asarray(q0, dtype=float).reshape(-1)
    b = np.asarray(q1, dtype=float).reshape(-1)
    n = max(2, int(steps))
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append((a + (b - a) * t).tolist())
    return out


# 抬高中继点如果和原位形差太多，说明 IK 翻到了另一个臂型，宁可不要这个中继
_MAX_RELAY_JUMP_RAD = math.radians(60.0)


def raise_conf(q, z: float | None = None, *, min_z: float | None = None):
    """把这个位形的 TCP 竖直抬到世界高度 ``z``，姿态不变。

    已经够高、或者抬上去 IK 无解、或者解出来是另一个臂型（关节跳变太大），
    都返回 None，让调用方退回直连。``min_z`` 是允许往下让的下限，放到框
    时不要让到框沿以下。
    """
    robot = _robot()
    q = np.asarray(q, dtype=float).reshape(-1)
    robot.fk(q, update=True)
    pos = np.asarray(robot.gl_tcp_pos, dtype=float).copy()
    rot = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    target = float(carry_z_m() if z is None else z)
    floor = pos[2] if min_z is None else max(pos[2], float(min_z))
    if pos[2] >= target - 1e-4:
        return None
    h = target
    while h > floor + 1e-4:
        cand = _ik([pos[0], pos[1], h], rot, q)
        if cand is not None and np.max(np.abs(cand - q)) <= _MAX_RELAY_JUMP_RAD:
            return cand
        h -= APPROACH_Z_STEP_M
    return None


def translate_tcp(q, delta_xyz):
    """TCP 平移、姿态不变。超工作区就夹到边上。无解返回 None。"""
    warmup_planner()
    robot = _robot()
    q = np.asarray(q, dtype=float).reshape(-1)
    robot.fk(q, update=True)
    pos = np.asarray(robot.gl_tcp_pos, dtype=float).copy()
    rot = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    tgt = pos + np.asarray(delta_xyz, dtype=float).reshape(3)
    b = JOG_BOUNDS_M
    tgt[0] = float(np.clip(tgt[0], b["x"][0], b["x"][1]))
    tgt[1] = float(np.clip(tgt[1], b["y"][0], b["y"][1]))
    tgt[2] = float(np.clip(tgt[2], max(float(JOG_Z_MIN_M), b["z"][0]), b["z"][1]))
    if float(np.linalg.norm(tgt - pos)) < 0.004:
        return None
    cand = _ik(tgt, rot, q)
    if cand is None:
        return None
    if np.max(np.abs(cand - q)) > _MAX_RELAY_JUMP_RAD:
        return None
    return cand


def transit_path(q_from, q_to, *, steps: int = PATH_STEPS, z: float | None = None) -> list[list[float]]:
    """转场路径：先竖直抬到安全高度，在高处平移，再下来。

    直接关节插值会贴着桌面扫过去，夹着水果去料框时会蹭框沿。两头抬不上去
    就退回直连。
    """
    height = float(carry_z_m() if z is None else z)
    floor = max(0.0, height - 0.04)  # 最多让 4cm，不能让到框沿那一档
    up_from = raise_conf(q_from, height, min_z=floor)
    up_to = raise_conf(q_to, height, min_z=floor)
    if up_from is None and up_to is None:
        return lerp_path(q_from, q_to, steps)
    a = q_from if up_from is None else up_from
    b = q_to if up_to is None else up_to
    short = max(2, steps // 3)
    out: list[list[float]] = []
    if up_from is not None:
        out += lerp_path(q_from, a, short)
    out += lerp_path(a, b, steps)
    if up_to is not None:
        out += lerp_path(b, q_to, short)
    return out


def grasp_rot(yaw: float, tilt: float = 0.0) -> np.ndarray:
    """夹爪朝下、绕竖直轴 yaw，再绕手指开合轴偏 tilt。

    tilt=0 是竖直顶抓。tilt>0 时夹爪往 yaw 指的水平方向倾，两根手指仍然等高，
    所以斜着也能平着夹住桌面上的件。
    """
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y = np.cross(z, x)
    n = np.linalg.norm(y)
    if n < 1e-8:
        x = np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        n = np.linalg.norm(y)
    y = y / n
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    rot = np.column_stack([x, y, z])
    if abs(tilt) > 1e-9:
        c, s = math.cos(tilt), math.sin(tilt)
        rot = rot @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return rot


_J6_LIMITS = None


def _j6_limits():
    """真机 J6 软限位。仿真是 ±180°，真机大约 [-95°, 90.5°]。"""
    global _J6_LIMITS
    if _J6_LIMITS is None:
        lo, hi = math.radians(-95.0), math.radians(90.5)
        try:
            from real_config import load_real_config

            cfg = load_real_config()
            lo, hi = cfg.joint_limits[-1]
        except Exception:
            pass
        _J6_LIMITS = (float(lo), float(hi))
    return _J6_LIMITS


def _fit_j6(q, seed):
    """两指夹爪转 180° 是同一个抓法。把 J6 折进真机软限位。"""
    q = np.asarray(q, dtype=float).copy()
    lo, hi = _j6_limits()
    raw = float(q[-1])
    anchor = float(np.asarray(seed, dtype=float).reshape(-1)[-1]) if seed is not None else 0.0
    cands = [raw + k * math.pi for k in range(-4, 5)]
    inside = [c for c in cands if lo - 1e-6 <= c <= hi + 1e-6]
    q[-1] = min(inside or cands, key=lambda c: abs(c - anchor))
    return q


def _ik(tgt_pos, rot, seed):
    robot = _robot()
    q = robot.ik(tgt_pos=np.asarray(tgt_pos, dtype=float), tgt_rotmat=rot, seed_jnt_values=seed)
    if q is None:
        q = robot.ik(tgt_pos=np.asarray(tgt_pos, dtype=float), tgt_rotmat=rot)
    if q is None:
        return None
    return _fit_j6(np.asarray(q, dtype=float), seed)


def ik_grasp(xyz, yaw: float, tilt: float, seed, back: float = 0.0):
    """back>0 时沿接近轴往后退（竖直抓就是正上方，斜抓就是斜后上方）。"""
    rot = grasp_rot(float(yaw), float(tilt))
    tgt = np.asarray(xyz, dtype=float) - float(back) * rot[:, 2]
    return _ik(tgt, rot, seed)


def ik_lift(xyz, yaw: float, tilt: float, seed, up: float):
    """抬起一律竖直向上，先把件拎离桌面。"""
    rot = grasp_rot(float(yaw), float(tilt))
    tgt = np.asarray(xyz, dtype=float) + np.array([0.0, 0.0, float(up)])
    return _ik(tgt, rot, seed)


def _dist_ladder(top: float, floor: float | None = None) -> list[float]:
    """从配置距离往下退的候选。

    件离基座近时「抓取点 + 10cm」经常没解，但同一点近几厘米就有。与其整单
    失败，不如缩短接近段。放到框时 ``floor`` 会抬高，避免又降回蹭框沿的高度。
    """
    lo = MIN_APPROACH_Z_M if floor is None else max(MIN_APPROACH_Z_M, float(floor))
    out = []
    h = float(top)
    while h >= lo - 1e-9:
        out.append(round(h, 4))
        h -= APPROACH_Z_STEP_M
    if not out:
        out.append(round(lo, 4))
    return out


def _robust_top(obj: dict, z_median: float) -> float:
    """可见点云的顶面。95/80 分位会被深度飞点抬高，用中位数 + 半径封顶。"""
    z_top = obj.get("z_top")
    top = float(z_top) if z_top is not None else float(z_median)
    r = float(obj.get("radius") or 0.0)
    if r > 1e-4:
        top = min(top, float(z_median) + r)
    return top


def _fruit_height(obj: dict, top: float) -> float:
    r = float(obj.get("radius") or 0.0)
    z_low = obj.get("z_low")
    h = 2.0 * r if r > 1e-4 else 0.0
    if z_low is not None:
        h = max(h, max(0.0, top - float(z_low)))
    return h


def _grasp_height(obj: dict, z_median: float, en: str) -> float:
    """顶抓 TCP 高度：从稳妥的顶面往下扎到件的中部，而不是上 1/3。

    深度飞点会把 z_top 抬高好几厘米，再只减一个固定半径，合爪点就还在
    橙子上沿。这里用中位数 + 半径封顶，再按件高的比例下探。
    """
    top = _robust_top(obj, z_median)
    height = _fruit_height(obj, top)
    depth_min = float(GRASP_DEPTH_M.get(en, 0.028))
    frac = float(GRASP_DEPTH_FRAC.get(en, 0.55))
    extra = float(GRASP_EXTRA_SINK_M.get(en, 0.0))
    depth = max(depth_min, height * frac) + extra
    grasp_z = top - depth
    floor = float(MIN_GRASP_Z_M)
    z_low = obj.get("z_low")
    if z_low is not None:
        floor = max(floor, float(z_low) + float(MIN_GRASP_ABOVE_BOTTOM_M))
    return max(floor, grasp_z)


def _first_reachable(fn, dists):
    for d in dists:
        q = fn(d)
        if q is not None:
            return q, d
    return None, None


def _pose_candidates(yaw0: float):
    """(yaw, tilt) 候选，先竖直后越来越斜。

    夹爪绕手指轴对称，竖直顶抓 yaw 和 yaw+180 是同一个姿态；但一旦倾斜，
    两者就是朝相反方向下手，所以斜抓要把 +180 也算上。
    """
    for t_deg in GRASP_TILT_DEG:
        tilt = math.radians(float(t_deg))
        for y_deg in IK_YAW_OFFSETS_DEG:
            yield yaw0 + math.radians(float(y_deg)), tilt, float(t_deg)
            if abs(t_deg) > 1e-9:
                yield yaw0 + math.radians(float(y_deg) + 180.0), tilt, float(t_deg)


def solve_grasp(obj: dict, seed, *, place: bool = False):
    p = obj.get("pose_6d") or {}
    xyz = (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))
    yaw0 = float(p.get("yaw") or 0.0)
    en = str(obj.get("yolo_class") or "apple")
    grasp_z = _grasp_height(obj, xyz[2], en)
    grasp_xyz = (xyz[0], xyz[1], grasp_z)
    r = obj.get("radius")
    if r and 2.0 * float(r) > SIM_JAW_OPEN_MAX:
        print(
            f"[plan] 警告：{obj.get('class')} 量出来约 {2.0 * float(r) * 100:.1f}cm 宽，"
            f"夹爪最大才开 {SIM_JAW_OPEN_MAX * 100:.1f}cm，多半是夹不住的"
        )
    approach_ladder = _dist_ladder(APPROACH_Z_M)
    carry = carry_z_m()
    if place:
        lift_up = max(float(LIFT_Z_M), carry - grasp_z)
        lift_floor = max(MIN_APPROACH_Z_M, carry - grasp_z - 0.04)
    else:
        lift_up = float(LIFT_Z_M)
        lift_floor = MIN_APPROACH_Z_M
    lift_ladder = _dist_ladder(lift_up, lift_floor)
    last_err = "IK 无解：竖直到 45° 斜抓都够不着，这个点可能太靠近基座或超出行程"
    for yaw, tilt, tilt_deg in _pose_candidates(yaw0):
        q_g = ik_grasp(grasp_xyz, yaw, tilt, seed)
        if q_g is None:
            continue
        q_a, h_a = _first_reachable(
            lambda d: ik_grasp(grasp_xyz, yaw, tilt, q_g, back=d), approach_ladder
        )
        if q_a is None:
            q_a, h_a = _first_reachable(
                lambda d: ik_grasp(grasp_xyz, yaw, tilt, seed, back=d), approach_ladder
            )
        if q_a is None:
            last_err = (
                f"接近位 IK 失败：抓取点后退 {MIN_APPROACH_Z_M * 100:.0f}cm 起都无解"
            )
            continue
        q_l, h_l = _first_reachable(
            lambda d: ik_lift(grasp_xyz, yaw, tilt, q_g, up=d), lift_ladder
        )
        if q_l is None:
            q_l, h_l = np.asarray(q_a, dtype=float), h_a
        posture = "竖直顶抓" if tilt_deg < 1e-9 else f"斜 {tilt_deg:.0f}° 抓"
        top = _robust_top(obj, xyz[2])
        height = _fruit_height(obj, top)
        raw_top = float(obj.get("z_top", xyz[2]))
        print(
            f"[plan] {obj.get('class')} {posture}，合爪高度 z={grasp_z:.3f}m"
            f"（顶面 {top:.3f}"
            + (f"，封掉飞点 {raw_top:.3f}" if raw_top - top > 0.005 else "")
            + f" 往下 {(top - grasp_z) * 100:.1f}cm"
            + (f" / 件高约 {height * 100:.1f}cm 的 {(top - grasp_z) / height * 100:.0f}%" if height > 1e-4 else "")
            + "）"
        )
        if h_a < APPROACH_Z_M - 1e-9 or h_l < lift_up - 1e-9:
            print(
                f"[plan] {obj.get('class')} 空间不够，接近 "
                f"{h_a * 100:.0f}cm、抬起 {h_l * 100:.0f}cm"
                f"（想要 {APPROACH_Z_M * 100:.0f}/{lift_up * 100:.0f}cm）"
            )
        if place:
            print(
                f"[plan] {obj.get('class')} 放到框：框高 {BIN_HEIGHT_M * 100:.0f}cm，"
                f"转场 TCP ≥ {carry * 100:.0f}cm（抬起 {h_l * 100:.0f}cm）"
            )
        return {
            "q_approach": q_a,
            "q_grasp": q_g,
            "q_lift": q_l,
            "yaw": yaw,
            "tilt_deg": tilt_deg,
            "approach_z": float(h_a),
            "lift_z": float(h_l),
            "jaw_close": float(JAW_CLOSE_M.get(en, 0.04)),
        }
    raise FruitPlanError(f"{obj.get('class')} @ {np.round(xyz, 3).tolist()} {last_err}")


def plan_fruits(objects: list[dict], *, dest: str | None, start_q=None, out_path: str | None = None) -> dict:
    if not objects:
        raise FruitPlanError("没有要抓的水果")
    warmup_planner()
    start = np.asarray(start_q if start_q is not None else look_conf(), dtype=float)
    place_q = None
    dest_name = ""
    if dest:
        place_q, dest_name = bin_conf(dest)

    segments = []
    cursor = start.copy()
    jaw_open = float(SIM_JAW_OPEN_MAX)
    for i, obj in enumerate(objects, start=1):
        sol = solve_grasp(obj, cursor, place=place_q is not None)
        jv: list[list[float]] = []
        ev: list[float | None] = []

        def _append(path, width):
            for q in path:
                if jv and max(abs(a - b) for a, b in zip(jv[-1], q)) < 1e-9:
                    continue
                jv.append(list(map(float, q)))
                ev.append(float(width))

        _append(transit_path(cursor, sol["q_approach"]), jaw_open)
        _append(lerp_path(sol["q_approach"], sol["q_grasp"]), jaw_open)
        carry_start = max(0, len(jv) - 1)
        # 合爪：同一关节角，ev 从开变关
        jv.append(list(map(float, sol["q_grasp"])))
        ev.append(float(sol["jaw_close"]))
        _append(lerp_path(sol["q_grasp"], sol["q_lift"]), sol["jaw_close"])
        release_idx = None
        if place_q is not None:
            _append(transit_path(sol["q_lift"], place_q), sol["jaw_close"])
            place_list = list(map(float, place_q))
            # 适配器松爪发生在 release_idx，carry 不含这一帧。
            # 先再写一个示教位，让合爪走完下探；松爪时 TCP 已经在放件高度，
            # 后面的 retreat 只负责往上抬出框，不会再往下拱。
            jv.append(list(place_list))
            ev.append(float(sol["jaw_close"]))
            release_idx = len(jv) - 1
            jv.append(list(place_list))
            ev.append(jaw_open)
            up = raise_conf(place_q, exit_bin_z_m(), min_z=carry_z_m())
            if up is not None:
                _append(lerp_path(place_q, up, max(8, PATH_STEPS // 3)), jaw_open)
                cursor = np.asarray(up, dtype=float)
                print(
                    f"[plan] {obj.get('class')} 松爪后抬出框 "
                    f"TCP→{exit_bin_z_m() * 100:.0f}cm"
                )
            else:
                cursor = np.asarray(place_q, dtype=float)
                print(f"[plan] {obj.get('class')} 松爪后抬不出框，将从示教位离开")
        else:
            cursor = np.asarray(sol["q_lift"], dtype=float)
        carry_end = len(jv) - 1 if release_idx is None else max(carry_start, release_idx - 1)
        segments.append(
            {
                "index": i,
                "total": len(objects),
                "object": str(obj.get("class") or obj.get("yolo_class")),
                "grasp_mode": (
                    "topdown" if sol["tilt_deg"] < 1e-9 else f"tilt{sol['tilt_deg']:.0f}"
                ),
                "do_place": bool(place_q is not None),
                "jaw_open": jaw_open,
                "jaw_close": float(sol["jaw_close"]),
                "carry_start": int(carry_start),
                "carry_end": int(carry_end),
                "release_idx": release_idx,
                "jv_list": jv,
                "ev_list": ev,
            }
        )
        print(
            f"[plan] [{i}] {segments[-1]['object']} 航点 {len(jv)}"
            + (f" → {dest_label(dest_name)}" if dest_name else "（只抓）")
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = out_path or os.path.join(OUTPUT_DIR, "traj.json")
    payload = {
        "source": "demo/fruit_sim_planner.py",
        "robot": "panthera_ht_6dof",
        "joint_unit": "rad",
        "gripper_unit": "sim_jaw_width_m",
        "sample_id": "fruit",
        "task": dest_name or "pick",
        "home_conf": np.asarray(start, dtype=float).tolist(),
        "segments": segments,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[plan] 仿真规划器已写出 {path}")
    return payload
