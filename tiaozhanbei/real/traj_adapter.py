# -*- coding: utf-8 -*-
"""仿真关节轨迹 → 真机动作脚本。

仿真臂和 fafu 真机是同一条 Panthera-HT 六轴（关节原点与轴向一致），所以
``jv_list`` 里的 6 个弧度值可以直接对应真机 J1~J6。但两边有三个必须处理的差异：

1. 仿真为了让规划更容易成功，把关节范围放宽了（``relax_panthera_joint_ranges_for_demo``），
   有些位形超出 ``robot.cfg`` 的软限位，必须裁剪并报出来；
2. 仿真动画是逐帧的（几百帧），真机不需要这么密，按角度变化抽帧；
3. 仿真夹爪是平移开口（米），真机 M7 是旋转关节（弧度），而且真机没必要逐帧跟随开口，
   只在「合爪抓取」「松爪放下」两个时刻动作。

产物是一个纯数据的 plan（可 JSON 落盘），由 ``real_arm.execute_plan`` 执行。
"""
from __future__ import annotations

import json
import math
import os

from real_config import (
    CLAMP_FAIL_RAD,
    CLAMP_WARN_RAD,
    MAX_STEP_RAD,
    WAYPOINT_MIN_DELTA_RAD,
    WRIST_SHORTEST_PATH,
    GripperMap,
    RealRobotConfig,
)

_TWO_PI = 2.0 * math.pi
_HALF_TURN = math.pi


def load_sim_traj(path: str) -> dict:
    """读取 ``run_agent_pick_place_side_sim.py --export-traj`` 导出的 JSON。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if data.get("joint_unit") not in (None, "rad"):
        raise ValueError(f"只支持弧度轨迹，拿到 joint_unit={data.get('joint_unit')!r}")
    if not data.get("segments"):
        raise ValueError(f"{path} 里没有任何 segment")
    return data


def _max_delta(a, b) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def downsample(waypoints: list[list[float]], min_delta: float = WAYPOINT_MIN_DELTA_RAD):
    """按角度变化抽帧；首末帧一定保留。"""
    if len(waypoints) <= 2:
        return [list(map(float, w)) for w in waypoints]
    kept = [list(map(float, waypoints[0]))]
    for wp in waypoints[1:-1]:
        if _max_delta(wp, kept[-1]) >= min_delta:
            kept.append(list(map(float, wp)))
    last = list(map(float, waypoints[-1]))
    if _max_delta(last, kept[-1]) > 1e-9:
        kept.append(last)
    return kept


def _wrap_options(joint_idx: int, num_joints: int) -> list[float]:
    """该关节有哪些「换个角度写、机械上完全一样」的等价偏移。

    转动关节加减整圈是同一个位形。最后一个关节是腕部滚转，两指夹爪转 180°
    后两根手指互换，夹持效果一样，所以额外允许半圈。
    """
    opts = [0.0, _TWO_PI, -_TWO_PI]
    if joint_idx == num_joints - 1:
        opts += [math.pi, -math.pi]
    return opts


def _choose_wrap(waypoints, cfg: RealRobotConfig) -> list[float]:
    """整条路径统一挑一组等价偏移，让越界总量最小。

    只能整条挑，不能逐点挑，否则路径中间会凭空出现半圈的跳变。
    """
    offsets = []
    for j in range(cfg.num_joints):
        lo, hi = cfg.joint_limits[j]
        col = [float(wp[j]) for wp in waypoints]
        best, best_cost = 0.0, None
        for off in _wrap_options(j, cfg.num_joints):
            cost = sum(max(lo - (v + off), 0.0) + max((v + off) - hi, 0.0) for v in col)
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = off, cost
        offsets.append(best)
    return offsets


def _densify(waypoints, max_step: float = MAX_STEP_RAD):
    """把跨度过大的相邻航点线性插成若干小步，真机才不会一口气甩过去。"""
    if len(waypoints) < 2:
        return [list(map(float, w)) for w in waypoints]
    out = [list(map(float, waypoints[0]))]
    for wp in waypoints[1:]:
        prev = out[-1]
        gap = _max_delta(wp, prev)
        n = int(math.ceil(gap / max_step)) if gap > max_step else 1
        for k in range(1, n + 1):
            t = k / n
            out.append([p + (float(q) - p) * t for p, q in zip(prev, wp)])
    return out


def _shortest_equiv_delta(d: float) -> float:
    """把一段腕部净转角换成等价的最短转角。

    两指夹爪转 180° 后两根手指互换，夹持位形完全一样，所以净转 152° 和
    净转 -28° 是同一个结果 —— 真机没必要走那 152°。
    """
    return d - _HALF_TURN * round(d / _HALF_TURN)


def _monotone_runs(col: list[float], eps: float = 1e-9) -> list[tuple[int, int]]:
    """按拐点把一列角度切成若干单调段，返回 ``[(起, 止)]``（闭区间）。

    平台段（连续相等）不算拐点，会被并进它前面那段 —— 抓取时腕部固定不动
    就是一个平台，必须整段一起折算，否则平台会被拆开、腕姿在下爪途中还在转。
    """
    if len(col) < 2:
        return []
    marks = [0]
    sign = 0
    for i in range(1, len(col)):
        d = col[i] - col[i - 1]
        if abs(d) <= eps:
            continue
        s = 1 if d > 0 else -1
        if sign != 0 and s != sign:
            marks.append(i - 1)
        sign = s
    marks.append(len(col) - 1)
    return [(marks[k], marks[k + 1]) for k in range(len(marks) - 1) if marks[k + 1] > marks[k]]


def _shortest_wrist(flat, cfg: RealRobotConfig) -> tuple[list[float], float, float]:
    """把腕部滚转整列折算到等价的最短行程，形状不变。

    仿真不知道 180° 等价，会让 J6 从 0° 一路摇到 152° 再摇回来（实测单条
    路径 604° 行程，占掉整段动作的绝大部分时间）。而 152° ≡ -28°，真机只
    需要转 28°。

    做法是**按单调段折算净转角**，而不是整列线性重插：每段的净转角换成等价
    最短值，段内各点按原比例缩放。这样平台仍是平台（抓取时腕部保持不动）、
    单调仍单调、和其他关节的相对节奏也不变，只是把多余的整/半圈摘掉。

    首点保持原值不动：机械臂此刻就在那个位形上，改了首点会让路径接不上。

    返回 ``(该列, 原行程, 新行程)``。
    """
    j = cfg.num_joints - 1
    raw = [float(wp[j]) for wp in flat]
    if len(raw) < 2:
        return raw, 0.0, 0.0

    out = list(raw)
    for a, b in _monotone_runs(raw):
        span = raw[b] - raw[a]
        target = _shortest_equiv_delta(span)
        if abs(span) <= 1e-9:
            for i in range(a + 1, b + 1):
                out[i] = out[a]
            continue
        scale = target / span
        for i in range(a + 1, b + 1):
            out[i] = out[a] + (raw[i] - raw[a]) * scale

    travel_old = sum(abs(raw[i] - raw[i - 1]) for i in range(1, len(raw)))
    travel_new = sum(abs(out[i] - out[i - 1]) for i in range(1, len(out)))
    return out, travel_old, travel_new


def _unwrap_symmetric_last(flat, cfg: RealRobotConfig) -> tuple[list[float], float]:
    """把腕部滚转逐点换到等价角度，既保持路径连续，又尽量落在软限位内。

    仿真不知道 180° 等价，抓取时常常把 J6 转到 -170.8° 这种真机够不到的角度，
    而 +9.2° 是同一个夹持位形、真机轻松可达。这里逐点在 ±180° 的等价值里挑
    「离上一点最近且不越界」的那个：结果连续、行程最短、越界最少。

    返回换算后的该列以及最大改动量（用来提示改了多少）。
    """
    j = cfg.num_joints - 1
    lo, hi = cfg.joint_limits[j]
    center = 0.5 * (lo + hi)

    def out_of_range(v: float) -> float:
        return max(lo - v, 0.0) + max(v - hi, 0.0)

    col: list[float] = []
    shifted = 0.0
    prev: float | None = None
    for wp in flat:
        raw = float(wp[j])
        anchor = center if prev is None else prev
        best = min(
            (raw + k * math.pi for k in range(-3, 4)),
            key=lambda c: out_of_range(c) * 100.0 + abs(c - anchor),
        )
        shifted = max(shifted, abs(best - raw))
        col.append(best)
        prev = best
    return col, shifted


def drop_home_prefix(path, home, tol: float = MAX_STEP_RAD) -> tuple[list[list[float]], int]:
    """只丢掉贴着仿真 HOME 的前缀，后面规划过的航点原样保留。

    ``splice_from_current`` 会按「直达终点更短」把接近收成 1 个点，
    避让位到抓取位变成一条 100° 关节直线，机座会抖。从避让位接入时用这个。
    """
    pts = [list(map(float, wp)) for wp in path]
    if not pts:
        return pts, 0
    home_q = list(map(float, home))
    skipped = 0
    while len(pts) > 1 and _max_delta(pts[0], home_q) <= tol:
        pts.pop(0)
        skipped += 1
    return pts, skipped


def splice_from_current(path, current) -> tuple[list[list[float]], int]:
    """从当前位置接入路径，丢掉只为离开 HOME 而存在的前缀。

    对每个航点 i 算 ``距当前 + 从 i 走到终点``，取总和最小的入口。
    当前位置已经接得上（首点够近）时原样返回。
    """
    if len(path) < 2:
        return [list(map(float, wp)) for wp in path], 0
    cur = list(map(float, current))
    pts = [list(map(float, wp)) for wp in path]
    # 已经接在路径头上就原路走，不要因为「直达终点更短」把中间验证过的姿态裁掉
    if _max_delta(cur, pts[0]) <= MAX_STEP_RAD:
        return pts, 0
    remain = [0.0] * len(pts)
    for i in range(len(pts) - 2, -1, -1):
        remain[i] = remain[i + 1] + _max_delta(pts[i], pts[i + 1])
    best_i = 0
    best_c = _max_delta(cur, pts[0]) + remain[0]
    for i in range(1, len(pts)):
        cost = _max_delta(cur, pts[i]) + remain[i]
        if cost < best_c - 1e-9:
            best_i, best_c = i, cost
    return pts[best_i:], best_i


def cut_after_lift(path, min_lift_rad: float) -> list[list[float]]:
    """保留抓取后的抬起段，丢掉后面回 HOME 的尾巴。

    从路径起点（抓取位）量起，走到单关节变化达到 ``min_lift_rad`` 的那一
    点（含）就停。整段都不够这个量就至少留下第二点，避免抬都没抬就走。
    """
    pts = [list(map(float, wp)) for wp in path]
    if len(pts) < 2:
        return pts
    start = pts[0]
    for i in range(1, len(pts)):
        if _max_delta(pts[i], start) >= min_lift_rad:
            return pts[: i + 1]
    return pts[:2]


def path_between(q_from, q_to, max_step: float = MAX_STEP_RAD) -> list[list[float]]:
    """两个位形之间的小步过渡航点（含终点、不含起点）。

    回 HOME 这类「一步跨很远」的动作用它拆开，避免真机一次甩过去。
    """
    return _densify([list(map(float, q_from)), list(map(float, q_to))], max_step)[1:]


def _condition_segment(
    bodies: list[list],
    cfg: RealRobotConfig,
    tag: str,
    warnings: list[str],
    errors: list[str],
    min_delta: float,
) -> list[list[list[float]]]:
    """把一个目标的各段路径一起做适配，产出可以直接下发的航点。

    必须整段一起处理：等价角度换算只有在整条路径上统一决策，接近、搬运、
    撤离三段之间才不会凭空冒出半圈跳变。
    """
    downs = [downsample(body, min_delta) for body in bodies]
    bounds = []
    flat: list[list[float]] = []
    for wps in downs:
        bounds.append((len(flat), len(flat) + len(wps)))
        flat.extend(list(map(float, wp)) for wp in wps)
    if not flat:
        return [[] for _ in bodies]

    # 其余关节整圈等价，整条路径统一平移
    offsets = _choose_wrap(flat, cfg)
    offsets[cfg.num_joints - 1] = 0.0
    for j, off in enumerate(offsets):
        if off:
            warnings.append(
                f"{tag}: J{j + 1} 整体旋转 {math.degrees(off):+.0f}° 到等价位形，"
                "以落进真机软限位"
            )
    flat = [[v + offsets[j] for j, v in enumerate(wp)] for wp in flat]

    # 腕部滚转折算
    last = cfg.num_joints - 1
    if WRIST_SHORTEST_PATH:
        col, t_old, t_new = _shortest_wrist(flat, cfg)
        if t_old - t_new > math.radians(1.0):
            warnings.append(
                f"{tag}: J{last + 1} 按夹爪 180° 对称折算到最短行程，"
                f"{math.degrees(t_old):.0f}° → {math.degrees(t_new):.0f}°"
                "（夹持位形不变；腕部不再走仿真那串中间姿态）"
            )
    else:
        col, shifted = _unwrap_symmetric_last(flat, cfg)
        if shifted > 1e-6:
            warnings.append(
                f"{tag}: J{last + 1} 按夹爪 180° 对称改写等价角度，最大改动 "
                f"{math.degrees(shifted):.0f}°（夹持位形不变）"
            )
    for wp, v in zip(flat, col):
        wp[last] = v

    clamped: list[list[float]] = []
    worst: dict[int, float] = {}
    for wp in flat:
        c, hit = cfg.clamp(wp)
        for j in hit:
            worst[j] = max(abs(wp[j] - c[j]), worst.get(j, 0.0))
        clamped.append(c)
    for j in sorted(worst):
        if worst[j] < CLAMP_WARN_RAD:
            continue
        lo, hi = cfg.joint_limits[j]
        msg = (
            f"{tag}: J{j + 1} 超软限位，最大越界 {math.degrees(worst[j]):.2f}°"
            f"（软限位 [{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]）"
        )
        if worst[j] > CLAMP_FAIL_RAD:
            errors.append(msg + "，裁剪后已不是规划时验证过的位姿，拒绝下发")
        else:
            warnings.append(msg + "，已裁剪")

    out = []
    for start, end in bounds:
        dense = _densify(clamped[start:end])
        for i in range(1, len(dense)):
            d = _max_delta(dense[i], dense[i - 1])
            if d > MAX_STEP_RAD + 1e-9:
                errors.append(
                    f"{tag}: 第 {i} 个航点单关节跨度 {math.degrees(d):.1f}°，插值后仍超上限"
                )
        out.append(dense)
    return out


def _slice_segment(seg: dict) -> list[tuple[str, object]]:
    """把一段 jv_list 按 carry_start / release_idx 切成 路径 / 夹爪 动作序列。"""
    jv = seg["jv_list"]
    n = len(jv)
    cs = max(0, min(int(seg.get("carry_start", 0)), n - 1))
    release = seg.get("release_idx")
    release = None if release is None else max(0, min(int(release), n - 1))

    items: list[tuple[str, object]] = [("approach", jv[: cs + 1])]
    if release is not None and release > cs:
        items.append(("close", None))
        items.append(("carry", jv[cs + 1 : release]))
        items.append(("open", None))
        if release < n - 1:
            items.append(("retreat", jv[release:]))
    else:
        items.append(("close", None))
        if cs + 1 < n:
            items.append(("carry", jv[cs + 1 :]))
    return [(kind, body) for kind, body in items if kind in ("close", "open") or body]


def _check_junctions(steps: list[dict], tag: str, errors: list[str]) -> None:
    """相邻两段路径之间只隔一个夹爪动作，手臂不该动。

    夹着零件时（松爪之前）如果两段接不上，说明前面的等价换算给这两段挑了
    不同的偏移，硬发下去会带着零件甩半圈。松爪之后手上是空的，允许重新调腕。
    """
    released = False
    prev_end = None
    for step in steps:
        if step["kind"] != "path":
            if step["action"] == "move" and prev_end is not None:
                released = True
            continue
        wps = step["waypoints"]
        if not wps:
            continue
        if prev_end is not None and not released:
            d = _max_delta(wps[0], prev_end)
            if d > MAX_STEP_RAD:
                errors.append(
                    f"{tag} {step['phase']}: 与上一段衔接处跨度 {math.degrees(d):.1f}°，"
                    "夹持中不能这样跳变"
                )
        prev_end = wps[-1]
        released = False


def adapt(
    traj: dict,
    cfg: RealRobotConfig,
    gmap: GripperMap,
    min_delta: float = WAYPOINT_MIN_DELTA_RAD,
) -> dict:
    """仿真轨迹 → 真机 plan。裁剪与抽帧都在这里完成，执行阶段不再改数值。"""
    warnings: list[str] = []
    errors: list[str] = []
    segments = []

    for seg in traj["segments"]:
        idx = int(seg.get("index", len(segments) + 1))
        obj = seg.get("object") or f"segment{idx}"
        jaw_open = float(seg.get("jaw_open", gmap.width_max))
        jaw_close = float(seg.get("jaw_close", 0.0))
        tag = f"[{idx}] {obj}"

        for wp in seg["jv_list"]:
            if len(wp) != cfg.num_joints:
                errors.append(
                    f"{tag}: 航点维度 {len(wp)} != 真机关节数 {cfg.num_joints}"
                )
                break

        items = _slice_segment(seg)
        paths = _condition_segment(
            [body for kind, body in items if kind not in ("close", "open")],
            cfg,
            tag,
            warnings,
            errors,
            min_delta,
        )

        # 接近一律张满。仿真若误把 jaw_open 收到几毫米，真机也不得跟着提前收。
        # 合爪只发生在 grasp 那一帧。
        close_w = float(jaw_close)
        tool_like = any(
            k in str(obj).lower()
            for k in ("钳", "扳", "螺丝刀", "plier", "wrench", "screwdriver")
        )
        if tool_like and close_w < 0.012:
            warnings.append(
                f"{tag}: 仿真合爪 {close_w*1000:.1f}mm 过小，工具按 38mm 合"
            )
            close_w = 0.038
        steps: list[dict] = [
            {
                "kind": "gripper",
                "action": "move",
                "angle_rad": float(gmap.angle_max),
                "width_m": float(gmap.width_max),
                "note": "保持全开，到位后再合",
            }
        ]
        raw_frames = 0
        kept_frames = 0
        cursor = 0
        for kind, body in items:
            if kind == "close":
                steps.append(
                    {
                        "kind": "gripper",
                        "action": "grasp",
                        "angle_rad": gmap.width_to_angle(close_w),
                        "width_m": close_w,
                        "note": "力控合爪",
                    }
                )
                continue
            if kind == "open":
                steps.append(
                    {
                        "kind": "gripper",
                        "action": "move",
                        "angle_rad": float(gmap.angle_max),
                        "width_m": float(gmap.width_max),
                        "note": "松爪放下",
                    }
                )
                continue
            wps = paths[cursor]
            cursor += 1
            raw_frames += len(body)
            kept_frames += len(wps)
            steps.append({"kind": "path", "phase": kind, "waypoints": wps})

        _check_junctions(steps, tag, errors)

        segments.append(
            {
                "index": idx,
                "object": obj,
                "grasp_mode": seg.get("grasp_mode", ""),
                "do_place": bool(seg.get("do_place", False)),
                "raw_frames": raw_frames,
                "waypoints": kept_frames,
                "steps": steps,
            }
        )

    home, _ = cfg.clamp(traj.get("home_conf") or [0.0] * cfg.num_joints)
    _bridge_gaps(segments, home, warnings)

    return {
        "source_traj": traj.get("task", ""),
        "sim_source": traj.get("source", ""),
        "sample_id": traj.get("sample_id", ""),
        "robot": traj.get("robot", "panthera_ht_6dof"),
        "joint_unit": "rad",
        "gripper_unit": "rad",
        "gripper_map": gmap.describe(),
        "home_conf": list(traj.get("home_conf") or []),
        "segments": segments,
        "warnings": warnings,
        "errors": errors,
    }


def _bridge_gaps(segments, home, warnings: list[str]) -> None:
    """补上「执行器会瞬移」的地方：HOME → 第一个航点，以及目标与目标之间。

    标准抓放的批量任务在仿真里本来就是连续的，首尾自然接得上。EC 螺丝入格不一样：
    每个 job 撤离后停在格子上方，下一个 job 从别处起步，中间有几十度的空档。

    这里用直线关节插补补齐，避免真机一步甩过去。**但这段过渡没有在仿真里做过
    碰撞检查**，所以同时报警提醒——料框已经放了件时尤其要先空载确认一遍。
    """
    prev = list(home)
    for seg in segments:
        paths = [s for s in seg["steps"] if s["kind"] == "path" and s["waypoints"]]
        if not paths:
            continue
        head = paths[0]
        gap = _max_delta(head["waypoints"][0], prev)
        if gap > MAX_STEP_RAD:
            bridge = _densify([prev, head["waypoints"][0]])[1:-1]
            head["waypoints"] = bridge + head["waypoints"]
            seg["waypoints"] = seg.get("waypoints", 0) + len(bridge)
            warnings.append(
                f"[{seg['index']}] {seg['object']}: 与上一位形差 {math.degrees(gap):.1f}°，"
                f"已插入 {len(bridge)} 个直线过渡航点；这段仿真没做碰撞检查，先空载确认"
            )
        prev = paths[-1]["waypoints"][-1]


def save_plan(plan: dict, path: str) -> str:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


def format_plan(plan: dict) -> str:
    """给终端看的 plan 摘要，dry-run 时就靠它确认要下发什么。"""
    lines = [
        "=" * 62,
        f"真机执行计划  sample={plan.get('sample_id') or '-'}  "
        f"规划来源={plan.get('sim_source') or '-'}  "
        f"目标数={len(plan['segments'])}",
        plan.get("gripper_map", ""),
    ]
    for seg in plan["segments"]:
        lines.append(
            f"  [{seg['index']}] {seg['object']}  "
            f"{'pick-place' if seg['do_place'] else 'pick'}  "
            f"抽帧 {seg['raw_frames']} → {seg['waypoints']}"
        )
        for step in seg["steps"]:
            if step["kind"] == "path":
                wps = step["waypoints"]
                head = ", ".join(f"{math.degrees(v):.1f}" for v in wps[-1]) if wps else ""
                lines.append(f"      path/{step['phase']:8s} {len(wps):3d} 点 → 终点 [{head}]°")
            else:
                lines.append(
                    f"      grip/{step['action']:8s} "
                    f"{math.degrees(step['angle_rad']):6.1f}°  "
                    f"(开口 {step['width_m'] * 1000:.1f}mm) {step['note']}"
                )
    for w in plan.get("warnings", []):
        lines.append(f"  [warn] {w}")
    for e in plan.get("errors", []):
        lines.append(f"  [ERROR] {e}")
    lines.append("=" * 62)
    return "\n".join(lines)
