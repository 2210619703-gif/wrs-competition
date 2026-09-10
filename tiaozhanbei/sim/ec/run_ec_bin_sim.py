#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EC 喇叭头螺丝 → 料盘分格装箱仿真（场景层）。

姿态分流（三种 place 不同）：
  - fallen：place=参考拧腕解 tip=15°/dyaw=-45°/j6≈-49.5°；pick 自选；
  - normal：顶抓；place=夹爪竖直（⊥XY）对准格子下探松爪；
  - inverted：顶抓放到桌面缓冲区 remesh 成 fallen，再按 fallen 入格。

默认 ``inv_last``：先抓 normal/fallen 入 JSON 格子；最后处理 inverted。
清障：挡路为 normal/fallen 则顶抓直接入其格子；inverted 才进缓冲区。
夹爪闭合相对螺头外廓留约 1mm 间隙（EC_JAW_HEAD_CLEARANCE_M）。

用法（仓库根目录）::
    python tiaozhanbei/sim/ec/run_ec_bin_sim.py
    python tiaozhanbei/sim/ec/run_ec_bin_sim.py --task tiaozhanbei/sim/ec/tasks/test03.json
    python tiaozhanbei/sim/ec/run_ec_bin_sim.py --verbose --no-anime
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_TB_DIR = os.path.abspath(os.path.join(_SIM_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs import wd, rm, mgm, mcm
import wrs.basis.data_adapter as da
from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

from tiaozhanbei.sim.environment import MODEL_SCALE, gen_desk, attach_desk_stripe
from tiaozhanbei.sim.ec.ec_layout import (
    BOX_FLOOR_Z,
    BOX_LIFT_Z,
    BOX_ORIGIN_XY,
    BOX_SIZE_M,
    BOX_STL,
    BUGLE_STL,
    PART_CLASS,
    SLOT_COLS,
    SLOT_PITCH_X,
    SLOT_PITCH_Y,
    SLOT_ROWS,
    box_world_floor_z,
    slot_world_xy,
)
from wrs.motion.motion_data import MotionData
import wrs.grasping.grasp as gg

# 复用侧抓/顶抓规划（不改原文件）
import tiaozhanbei.sim.run_agent_pick_place_side_sim as side


DEFAULT_SAMPLE = os.path.join(_TB_DIR, "dataset_learn", "0502")
DEFAULT_TASK = os.path.join(_THIS_DIR, "tasks", "test03.json")
DEFAULT_CACHED_TRAJ = os.path.join(_THIS_DIR, "traj", "ec_all12_traj.json")
DEFAULT_GRASP_TOPDOWN = os.path.join(
    _SIM_DIR, "grasp", "panthera_ht_block_grasps_topdown_yaw.pickle"
)
DEFAULT_GRASP_SIDE = os.path.join(_SIM_DIR, "grasp", "panthera_ht_block_grasps.pickle")
# 动画抽稀：规划帧很密，空格只播关键帧
ANIME_FRAME_STRIDE = 8
ANIME_TARGET_FRAMES = 28
# 终端加速：默认少打日志；--verbose 打开细节
EC_VERBOSE = False
# 桌面缓冲区候选（工作空间内、避开基座禁区与料盘）
EC_BUFFER_XY_CANDS = (
    np.array([0.14, -0.08], dtype=float),
    np.array([0.16, -0.05], dtype=float),
    np.array([0.10, -0.10], dtype=float),
    np.array([0.12, 0.05], dtype=float),
    np.array([0.18, -0.08], dtype=float),
)


def _ec_log(msg: str):
    if EC_VERBOSE:
        print(msg, flush=True)


_EC_UI_MANAGED = False


def _ec_pump_ui(base, n: int = 2):
    """规划阻塞主线程时刷几帧，避免 Panda 窗口被 Windows 标成「未响应」。

    UI 托管模式下右侧已有 HOME 预览窗口，这里刷帧会让规划中的正式窗口
    被 Windows 拉到前台（黑屏跳出），因此直接跳过。
    """
    if base is None or _EC_UI_MANAGED:
        return
    try:
        for _ in range(max(1, int(n))):
            base.taskMgr.step()
    except Exception:
        pass


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    if not text.strip():
        raise ValueError(
            f"任务 JSON 为空: {path}\n"
            "请先保存文件，或: python tiaozhanbei/sim/ec/run_ec_bin_sim.py "
            "--task tiaozhanbei/sim/ec/tasks/test.json"
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"任务 JSON 无法解析: {path}\n{e}") from e


def _norm_state(s) -> str:
    t = str(s or "normal").strip().lower()
    if t in ("upside_down", "inverted", "倒放", "倒置"):
        return "inverted"
    if t in ("fallen", "tipped", "倾倒", "侧倒", "lying", "tilt", "tilted"):
        return "fallen"
    if t in ("normal", "upright", "正放", "正常"):
        return "normal"
    return t


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _parse_slot_token(text: str):
    raw = str(text or "")
    m = re.search(r"第\s*([一二三四五六七八九十两\d]+)\s*个?\s*格", raw)
    if m:
        n = _parse_count_token(m.group(1))
        return None if n is None else max(0, n - 1)
    m = re.search(r"(?:料盘)?格子\s*(\d+)", raw)
    if m:
        return int(m.group(1))
    return None


def _parse_count_token(token: str):
    token = str(token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if token in digits:
        return digits[token]
    return None


def _slot_from_dest_xy(dest):
    if not isinstance(dest, (list, tuple)) or len(dest) < 2:
        return None
    xy = np.asarray(dest[:2], dtype=float)
    best, bd = None, 1e9
    for sid in range(SLOT_COLS * SLOT_ROWS):
        d = float(np.linalg.norm(slot_world_xy(sid) - xy))
        if d < bd:
            bd, best = d, sid
    return best if best is not None and bd < 0.02 else None


def _infer_slot_id(item: dict, fallback=None):
    if not isinstance(item, dict):
        return fallback
    for key in ("slot_id", "slot"):
        if item.get(key) is not None:
            try:
                return int(item.get(key))
            except (TypeError, ValueError):
                pass
    nested = item.get("task") if isinstance(item.get("task"), dict) else None
    if nested is not None:
        got = _infer_slot_id(nested)
        if got is not None:
            return got
    for key in ("destination", "original_cmd", "reason"):
        got = _parse_slot_token(item.get(key) or "")
        if got is not None:
            return got
    got = _slot_from_dest_xy(item.get("dest_coordinate"))
    if got is not None:
        return got
    return fallback


def find_adjust_pose(root: dict):
    """从 tasks[] 里找 adjust_pose；返回 dict 或 None。"""
    if not isinstance(root, dict):
        return None
    node = root["task"] if isinstance(root.get("task"), dict) else root
    for step in node.get("tasks") or []:
        if str(step.get("action") or "").lower() != "adjust_pose":
            continue
        rpy = step.get("rpy") or [0.0, 0.0, float(step.get("angle") or 0.0)]
        while len(rpy) < 3:
            rpy.append(0.0)
        return {
            "adjust_to_normal": True,
            "target_pose_state": _norm_state(
                step.get("target_pose_state") or "normal"
            ),
            "place_yaw": float(rpy[2]),
            "place_rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
        }
    return None


def parse_jobs(data: dict, ann: dict) -> list:
    """统一解析智能体 JSON → [{object_id, slot_id, coordinate, class, state}, ...]。"""
    jobs = []
    adjust_info = find_adjust_pose(data) if isinstance(data, dict) else None

    def _from_item(item: dict, default_slot=None, adjust=None):
        oid = item.get("object_id")
        if oid is None and item.get("id") is not None:
            oid = item.get("id")
        slot = _infer_slot_id(item, default_slot)
        if slot is None and item.get("task") and isinstance(item["task"], dict):
            slot = _infer_slot_id(item["task"], default_slot)
            if oid is None:
                oid = item["task"].get("object_id")
        task = item.get("task") if isinstance(item.get("task"), dict) else item
        coord = task.get("coordinate") or item.get("coordinate")
        cls = task.get("object") or item.get("object") or PART_CLASS
        if oid is None and coord is not None:
            # 用坐标在 annotations 里匹配
            c = np.asarray(coord, dtype=float)[:2]
            best, bd = None, 1e9
            for obj in ann.get("objects") or []:
                p = obj.get("pose_6d") or {}
                xy = np.array([float(p.get("x", 0)), float(p.get("y", 0))])
                d = float(np.linalg.norm(xy - c))
                if d < bd:
                    bd, best = d, obj
            if best is not None and bd < 0.025:
                oid = best.get("id")
                cls = best.get("class") or cls
        if oid is None or slot is None:
            return None
        obj_ann = None
        for obj in ann.get("objects") or []:
            if int(obj.get("id")) == int(oid):
                obj_ann = obj
                break
        pose = (obj_ann or {}).get("pose_6d") or {}
        if coord is None:
            coord = [float(pose.get("x", 0)), float(pose.get("y", 0)), 0.0]
        state = _norm_state(
            (obj_ann or {}).get("state")
            or item.get("state")
            or item.get("pose_state")
            or "normal"
        )
        # 有 adjust_pose 步，或零件本身非正放 → 放置前调成 normal
        adj = adjust if adjust is not None else find_adjust_pose(item)
        need_adj = bool(adj) or state in ("inverted", "fallen")
        place_yaw = float(pose.get("yaw", 0.0) or 0.0)
        place_rpy = [0.0, 0.0, place_yaw]
        if adj:
            place_yaw = float(adj.get("place_yaw", place_yaw))
            place_rpy = list(adj.get("place_rpy") or [0.0, 0.0, place_yaw])
            place_rpy[0], place_rpy[1] = 0.0, 0.0  # 正放：清零 roll/pitch
            place_rpy[2] = place_yaw
        return {
            "object_id": int(oid),
            "slot_id": int(slot),
            "coordinate": [
                float(coord[0]),
                float(coord[1]),
                float(coord[2] if len(coord) > 2 else 0.0),
            ],
            "object": cls,
            "state": state,
            "adjust_to_normal": need_adj,
            "place_yaw": place_yaw,
            "place_rpy": place_rpy,
            "dest_coordinate": slot_world_xy(int(slot)).tolist()
            + [float(box_world_floor_z())],
        }

    if isinstance(data, list):
        for it in data:
            j = _from_item(it if isinstance(it, dict) else {}, adjust=None)
            if j:
                jobs.append(j)
        return jobs

    if not isinstance(data, dict):
        raise TypeError("任务 JSON 应为 object 或 array")

    for src_key, dst_key in (("jobs_json", "jobs"), ("batch_json", "batch")):
        if dst_key not in data or not isinstance(data.get(dst_key), list):
            parsed = _as_list(data.get(src_key))
            if parsed:
                data[dst_key] = parsed

    default_slot = _infer_slot_id(data)

    # 批量任务：jobs / batch（优先于 tasks 步骤展开，避免一步一 job）
    for key in ("jobs", "batch"):
        if key not in data or not isinstance(data.get(key), list):
            continue
        for it in data[key]:
            if not isinstance(it, dict):
                continue
            j = _from_item(
                it,
                default_slot=default_slot,
                adjust=adjust_info or find_adjust_pose(it),
            )
            if j:
                jobs.append(j)
        if jobs:
            return jobs

    # 单任务：顶层或嵌套 task（含 tasks[].adjust_pose）
    j = _from_item(data, default_slot=default_slot, adjust=adjust_info)
    if j:
        return [j]

    # 兼容 tasks 步骤列表：找带 slot_id 的 place/pick
    # （无 jobs/batch 时使用；有 batch 时上面已返回，不会把多步拆成重复 job）
    nested = data.get("task") if isinstance(data.get("task"), dict) else data
    for step in nested.get("tasks") or []:
        if _infer_slot_id(step, default_slot) is None and str(step.get("action") or "").lower() not in (
            "place",
            "pick",
        ):
            continue
        j = _from_item(step, default_slot=default_slot, adjust=adjust_info)
        if j:
            jobs.append(j)
    if jobs:
        # Dify 单件常把 perceive/pick/place 都带同一坐标，只保留一件
        uniq = []
        seen = set()
        for job in jobs:
            key = (job["object_id"], job["slot_id"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(job)
        return uniq

    job_count = data.get("job_count")
    if job_count == 1 or default_slot is not None:
        print(
            "[task] Dify 单件工单缺 object_id/slot_id，未回退到全量入格",
            flush=True,
        )
        return []

    # 兜底：按 annotations 顺序 object_id=i → slot_id=i-1
    print("[task] 未解析到 jobs，使用默认 object_id→slot_id 一一映射", flush=True)
    for obj in ann.get("objects") or []:
        oid = int(obj["id"])
        st = _norm_state(obj.get("state") or "normal")
        yaw = float((obj.get("pose_6d") or {}).get("yaw", 0.0) or 0.0)
        jobs.append(
            {
                "object_id": oid,
                "slot_id": oid - 1,
                "coordinate": [
                    float(obj["pose_6d"]["x"]),
                    float(obj["pose_6d"]["y"]),
                    0.0,
                ],
                "object": obj.get("class") or PART_CLASS,
                "state": st,
                "adjust_to_normal": st in ("inverted", "fallen"),
                "place_yaw": yaw,
                "place_rpy": [0.0, 0.0, yaw],
                "dest_coordinate": slot_world_xy(oid - 1).tolist()
                + [float(box_world_floor_z())],
            }
        )
    return jobs


def load_box_model(origin_xy=None):
    """加载 box.stl，角点对齐 BOX_ORIGIN_XY，底面抬高 BOX_LIFT_Z。"""
    o = BOX_ORIGIN_XY if origin_xy is None else np.asarray(origin_xy, dtype=float)
    mesh = da.trm.load(BOX_STL)
    mesh.apply_scale(np.array([MODEL_SCALE, MODEL_SCALE, MODEL_SCALE]))
    # mesh 本地已从 (0,0,0) 起；平移到世界角点并抬高
    mesh.apply_translation(
        np.array([float(o[0]), float(o[1]), float(BOX_LIFT_Z)])
    )
    model = mcm.CollisionModel(mesh, name="ec_box", rgb=rm.vec(0.55, 0.45, 0.35), alpha=0.55)
    model.pos = rm.vec(0.0, 0.0, 0.0)
    return model


def load_parts_from_ec_ann(ann: dict):
    """按 EC annotations 还原螺丝（支持 roll/pitch/yaw + state）。"""
    colors = [
        rm.vec(0.80, 0.42, 0.36),
        rm.vec(0.42, 0.62, 0.86),
        rm.vec(0.48, 0.75, 0.46),
        rm.vec(0.82, 0.70, 0.38),
        rm.vec(0.65, 0.50, 0.75),
    ]
    part_infos = []
    for i, obj in enumerate(ann.get("objects") or []):
        pose = obj.get("pose_6d") or {}
        roll = float(pose.get("roll", 0.0) or 0.0)
        pitch = float(pose.get("pitch", 0.0) or 0.0)
        yaw = float(pose.get("yaw", 0.0) or 0.0)
        stl = obj.get("stl_path") or BUGLE_STL
        mesh, height, grip_width = side._load_mesh_on_table(stl, yaw, roll=roll, pitch=pitch)
        rgb = colors[i % len(colors)]
        model = mcm.CollisionModel(mesh, name=str(obj.get("class") or PART_CLASS), rgb=rgb)
        pos = rm.vec(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), 0.0)
        model.pos = pos
        model.rotmat = np.eye(3)
        part_infos.append(
            {
                "id": obj.get("id"),
                "class": obj.get("class") or PART_CLASS,
                "state": obj.get("state") or "normal",
                "model": model,
                "pos": np.asarray(pos, dtype=float),
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                # mesh 已烘焙桌面 rpy；入格时要用它们反算竖直，勿被 seat 覆盖
                "bake_roll": roll,
                "bake_pitch": pitch,
                "bake_yaw": yaw,
                "stl_path": stl,
                "height": height,
                "top_z": height,
                "mid_z": 0.5 * height,
                "grip_width": grip_width,
            }
        )
    return part_infos


# Bugle STL 长轴=+Y（平躺）；Rx(+90°) → 长轴竖直到 +Z（入格 upright）
UPRIGHT_OBJ_ROT = rm.rotmat_from_euler(float(np.pi / 2.0), 0.0, 0.0)
EC_BOX_RIM_Z = float(BOX_LIFT_Z + BOX_SIZE_M[2])  # 箱口高度（贴桌时=0.012）
# 入格攻角：fallen/统一臂型用 15°；normal 顶放优先 0°
EC_PLACE_APPROACH_TIP_DEGS = (15.0, 10.0, 5.0, 0.0)
EC_PLACE_TIP_DEGS_TOP = (0.0, 5.0)  # 加速：少 tip，同 pick 上多 dyaw
EC_PLACE_TIP_DEGS_FAST = (15.0, 0.0)
# 放置种子：J2 前伸族（~80°）+ J5 偏正（逆时针相对折叠腕）
EC_PLACE_NATURAL_CONF = np.array(
    [0.0, np.deg2rad(80.0), 0.70, np.deg2rad(40.0), np.deg2rad(20.0), 1.2],
    dtype=float,
)
EC_J1_CW_BIAS = float(np.deg2rad(-20.0))
EC_J1_WIGGLE = float(np.deg2rad(20.0))
EC_HIGH_HOVER_Z = 0.14
# 松爪：指尖略高于箱口；TCP 下限配合竖直/微俯角入格
EC_RELEASE_TIP_ABOVE = 0.004
EC_RELEASE_TCP_FLOOR = float(EC_BOX_RIM_Z + EC_RELEASE_TIP_ABOVE + 0.022)
# 方案 A：松爪硬验收（零件≈竖直立在格心，指尖略高于箱口）
EC_RELEASE_OBJ_XY_TOL = 0.004
# 绝对下限≈16°；具体 tip 候选再用 cos(tip) 放宽，避免 tip=15 被误杀后只剩折腕解
EC_RELEASE_UPRIGHT_MIN = 0.96
# 箱口+18mm：覆盖 IK 抖动（日志里常 0.025~0.028，过严 0.024 会全灭）
EC_RELEASE_TIP_MAX_ABOVE = 0.018
# 稳定抓取：只尝试排序后前 N 个（加速；place 失败不重算 pick）
EC_STABLE_GRASP_TOP_N = 2
EC_FALLEN_GRASP_TOP_N = 4  # fallen place 固定臂型，多试几个 pick 朝向
EC_PLACE_ROT_TOP_N = 6  # 同一 pick 上多试几个 place 朝向
EC_CLEAR_NEIGHBOR_MAX = 1
EC_CLEAR_NEIGHBOR_RADIUS = 0.07
# 夹紧：相对螺头外廓双侧合计留约 1mm 间隙（指面不贴死，减 deep 误报）
EC_JAW_CLOSE_FRAC = 0.98
# 双侧合计间隙；过大则视觉上空隙明显（居中后仍靠此外廓定闭合宽）
EC_JAW_HEAD_CLEARANCE_M = 0.0006
# 优先解族：J2 前伸、J5 不太负（避免折叠肘 + 腕死折）
EC_J2_FORWARD_MIN = float(np.deg2rad(55.0))
EC_J2_FORWARD_MAX = float(np.deg2rad(115.0))
EC_J5_CCW_PREF = float(np.deg2rad(-40.0))  # j5 越大（越正）越好
# 硬拒折腕：腕部连杆自碰未进 CD pair，必须靠关节/法兰高度启发式拦住
# 相对放宽 J5 下界，以覆盖零件1 成功解 j5≈-73.5°
EC_J5_HARD_MIN = float(np.deg2rad(-95.0))
EC_J5_HARD_MAX = float(np.deg2rad(70.0))
# 法兰/夹爪座离桌：图中水平腕 j6≈0 时会穿桌，须抬高验收
EC_EE_DESK_CLEAR_Z = 0.045

# 统一放置臂型 = 调整前零件1成功解（勿被坏解覆盖）
# tip=15° dyaw=-45° j≈[-38, 106.6, 25.5, 15.2, -73.5, -49.5]
EC_PLACE_REF_TIP_DEG = 15.0
EC_PLACE_REF_DYAW = float(-0.25 * np.pi)
EC_PLACE_REF_ARM = np.array(
    [
        0.0,  # J1 占位，实际按格心重算
        np.deg2rad(106.6),
        np.deg2rad(25.5),
        np.deg2rad(15.2),
        np.deg2rad(-73.5),
        np.deg2rad(-49.5),
    ],
    dtype=float,
)
# J6 必须转到约 -50°：|J6|≲20° 时夹爪水平贴桌（用户截图问题）
EC_PLACE_J6_REF = float(EC_PLACE_REF_ARM[5])
EC_PLACE_J6_MIN = float(np.deg2rad(-95.0))
EC_PLACE_J6_MAX = float(np.deg2rad(-25.0))
# 运行时：首件成功后可刷新；之后优先复用该臂型
_PLACE_STYLE = {
    "tip_deg": float(EC_PLACE_REF_TIP_DEG),
    "dyaw": float(EC_PLACE_REF_DYAW),
    "q_arm": EC_PLACE_REF_ARM.copy(),
    "locked": False,
    "kind": None,  # "fallen_ref" | "normal_vertical"
}
# fallen：place 永远用 EC_PLACE_REF_*；此处只记成功抓取偏好（可选）
_FALLEN_TPL = {
    "locked": False,  # 仅表示已有成功 pick 记录，不改变 place
    "grasp_mode": "topdown",
    "ac_pos": None,
    "ac_rotmat": None,
}


def _ec_j6_place_ok(q) -> bool:
    """放置「拧腕安全带」：约 -50° 附近，避免 |J6|≲20° 水平贴桌。

    注意：对齐入格的 ±180° 腕转不走此带；仅作默认 REF / 偏好。
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size < 6:
        return False
    j6 = float(q[5])
    return bool(EC_PLACE_J6_MIN <= j6 <= EC_PLACE_J6_MAX)


def _ec_j6_in_joint_limits(robot, j6: float) -> bool:
    j6_lo, j6_hi = _ec_j6_motion_range(robot)
    return bool(float(j6_lo) - 1e-6 <= float(j6) <= float(j6_hi) + 1e-6)


def lock_place_style_from_q(q_release, tip_deg=None, dyaw=None):
    """锁定放置臂型；J6 不合格时仍保留参考臂型（防锁进穿桌解）。"""
    global _PLACE_STYLE
    q = np.asarray(q_release, dtype=float).reshape(-1)
    if q.size < 6:
        return
    arm = EC_PLACE_REF_ARM.copy()
    if _ec_j6_place_ok(q):
        arm[1:6] = q[1:6]
    else:
        print(
            f"[place-style] 拒绝坏 J6={np.rad2deg(float(q[5])):.1f}°，"
            f"保留参考 j6={np.rad2deg(EC_PLACE_J6_REF):.1f}°",
            flush=True,
        )
    _PLACE_STYLE["q_arm"] = arm
    # 仅 fallen 族：tip/dyaw 固定为调整前成功族
    _PLACE_STYLE["tip_deg"] = float(EC_PLACE_REF_TIP_DEG)
    _PLACE_STYLE["dyaw"] = float(EC_PLACE_REF_DYAW)
    _PLACE_STYLE["kind"] = "fallen_ref"
    if tip_deg is not None and abs(float(tip_deg) - EC_PLACE_REF_TIP_DEG) < 6.0:
        _PLACE_STYLE["tip_deg"] = float(tip_deg)
    if dyaw is not None and abs(float(dyaw) - EC_PLACE_REF_DYAW) < np.deg2rad(20.0):
        _PLACE_STYLE["dyaw"] = float(dyaw)
    _PLACE_STYLE["locked"] = True
    print(
        f"[place-style] 锁定 fallen 臂型 tip={_PLACE_STYLE['tip_deg']:.0f}° "
        f"dyaw={np.rad2deg(_PLACE_STYLE['dyaw']):.0f}° "
        f"j2-6={np.round(np.rad2deg(arm[1:6]),1).tolist()}",
        flush=True,
    )


def pin_fallen_place_style():
    """fallen / inverted→fallen：拧腕参考解。"""
    global _PLACE_STYLE
    _PLACE_STYLE = {
        "tip_deg": float(EC_PLACE_REF_TIP_DEG),
        "dyaw": float(EC_PLACE_REF_DYAW),
        "q_arm": EC_PLACE_REF_ARM.copy(),
        "locked": True,
        "kind": "fallen_ref",
    }


def pin_normal_place_style():
    """normal：竖直顶放（tip=0，夹爪接近轴 ⊥ XY）。"""
    global _PLACE_STYLE
    q = np.asarray(EC_PLACE_NATURAL_CONF, dtype=float).copy()
    _PLACE_STYLE = {
        "tip_deg": 0.0,
        "dyaw": 0.0,
        "q_arm": q,
        "locked": True,
        "kind": "normal_vertical",
    }


def lock_fallen_pick_hint(grasp_mode, grasp):
    """只记录成功抓取偏好；place 不受影响。"""
    global _FALLEN_TPL
    _FALLEN_TPL = {
        "locked": True,
        "grasp_mode": str(grasp_mode or "topdown"),
        "ac_pos": np.asarray(grasp.ac_pos, dtype=float).reshape(3).copy(),
        "ac_rotmat": np.asarray(grasp.ac_rotmat, dtype=float).copy(),
    }
    print(
        f"[fallen] place 固定 tip=15°/dyaw=-45°/j6=-49.5°；"
        f"pick 可调 mode={_FALLEN_TPL['grasp_mode']} "
        f"ac={np.round(_FALLEN_TPL['ac_pos'], 4).tolist()}",
        flush=True,
    )


def _rank_grasps_for_fallen_pick(part_grasps, top_n: int = 3):
    """fallen pick：自行选抓取；若有历史成功 grasp 则略优先相近者。"""
    ranked = _ec_stable_sort_grasps(part_grasps)
    if bool(_FALLEN_TPL.get("locked")) and _FALLEN_TPL.get("ac_rotmat") is not None:
        tp = np.asarray(_FALLEN_TPL["ac_pos"], dtype=float).reshape(3)
        tr = np.asarray(_FALLEN_TPL["ac_rotmat"], dtype=float)

        def _key(g):
            ac_p = np.asarray(g.ac_pos, dtype=float).reshape(3)
            ac_r = np.asarray(g.ac_rotmat, dtype=float)
            sim = float(np.linalg.norm(ac_p - tp)) + 0.35 * float(
                np.linalg.norm(ac_r - tr)
            )
            return (sim, float(np.linalg.norm(ac_p)))

        ranked = sorted(list(part_grasps), key=_key)
    return ranked[: max(1, int(top_n))]


def ec_tight_jaw_close(jaw_close, jaw_open, grip_width) -> float:
    """贴紧但不穿模：以接触搜索宽为准，最多再收 0.3mm。"""
    gw = float(max(grip_width or 0.005, side.JAW_CLOSE_MIN))
    if jaw_close is None:
        jc = float(gw * EC_JAW_CLOSE_FRAC)
    else:
        jc = max(float(jaw_close) - 0.0003, float(gw) * 0.90, side.JAW_CLOSE_MIN)
    return float(np.clip(jc, side.JAW_CLOSE_MIN, float(jaw_open) - 1e-4))


def _ec_grasp_tcp(obj_model, grasp):
    obj_pos = np.asarray(obj_model.pos, dtype=float)
    obj_rot = np.asarray(obj_model.rotmat, dtype=float)
    ac_p = np.asarray(grasp.ac_pos, dtype=float).reshape(3)
    ac_r = np.asarray(grasp.ac_rotmat, dtype=float)
    return obj_pos + obj_rot @ ac_p, obj_rot @ ac_r


def _ec_pick_pose_quality(grasp, obj_model, grasp_mode: str = "topdown") -> bool:
    """夹爪姿态合格：顶抓接近轴朝下、张合轴近似水平（fallen 侧躺可夹直径）。"""
    _, R = _ec_grasp_tcp(obj_model, grasp)
    # 工具 +Z 为接近轴：顶抓应朝下
    approach_down = -float(R[2, 2])
    # 顶抓须接近竖直（拒用户截图中 ~45° 斜抓穿模）
    if str(grasp_mode) == "topdown" and approach_down < 0.85:
        return False
    # 张合方向≈工具 Y：应近水平，否则指尖易戳进螺头
    open_vert = abs(float(R[2, 1]))
    if open_vert > 0.45:
        return False
    tcp_pos, _ = _ec_grasp_tcp(obj_model, grasp)
    # TCP 过低 → 指尖/掌部穿桌或穿零件
    if float(tcp_pos[2]) < 0.006:
        return False
    return True


def _ec_ee_hits_object(robot, obj_model, jaw_width) -> bool:
    """整爪（含掌座）与零件是否 mesh 碰撞。"""
    try:
        robot.end_effector.change_jaw_width(float(jaw_width))
    except Exception:
        pass
    ee = robot.end_effector
    if side._ee_fingers_hit_object(ee, obj_model):
        return True
    # 再查 palm / 其它 cdmesh
    try:
        if ee.is_mesh_collided([obj_model]):
            return True
    except Exception:
        pass
    return False


def _ec_obj_width_along_open(obj_model, tcp_rot) -> float:
    """零件在夹爪张合方向（工具 +Y）上的外廓宽度。"""
    try:
        verts = np.asarray(obj_model.trm_mesh.vertices, dtype=float)
    except Exception:
        return 0.0
    if verts.size == 0:
        return 0.0
    R_obj = np.asarray(obj_model.rotmat, dtype=float)
    pos = np.asarray(obj_model.pos, dtype=float)
    world = (R_obj @ verts.T).T + pos
    open_dir = np.asarray(tcp_rot, dtype=float)[:, 1]
    nrm = float(np.linalg.norm(open_dir))
    if nrm < 1e-9:
        return 0.0
    open_dir = open_dir / nrm
    t = world @ open_dir
    return float(np.max(t) - np.min(t))


def _ec_obj_open_span(obj_model, tcp_rot):
    """张合方向上外廓 (t_min, t_max, mid, width)。"""
    try:
        verts = np.asarray(obj_model.trm_mesh.vertices, dtype=float)
    except Exception:
        return None
    if verts.size == 0:
        return None
    R_obj = np.asarray(obj_model.rotmat, dtype=float)
    pos = np.asarray(obj_model.pos, dtype=float)
    world = (R_obj @ verts.T).T + pos
    open_dir = np.asarray(tcp_rot, dtype=float)[:, 1]
    nrm = float(np.linalg.norm(open_dir))
    if nrm < 1e-9:
        return None
    open_dir = open_dir / nrm
    t = world @ open_dir
    t_min, t_max = float(np.min(t)), float(np.max(t))
    return t_min, t_max, 0.5 * (t_min + t_max), float(t_max - t_min), open_dir


def _ec_center_grasp_along_open(obj_model, grasp, min_shift: float = 2e-4) -> float:
    """把 TCP 沿张合轴（工具 +Y）平移，使零件外廓中线落在两指中间。

    返回平移量（米）；|shift|<min_shift 则视为已居中。
    """
    obj_pos = np.asarray(obj_model.pos, dtype=float)
    obj_rot = np.asarray(obj_model.rotmat, dtype=float)
    ac_p = np.asarray(grasp.ac_pos, dtype=float).reshape(3).copy()
    ac_r = np.asarray(grasp.ac_rotmat, dtype=float)
    tcp_pos = obj_pos + obj_rot @ ac_p
    tcp_rot = obj_rot @ ac_r
    span = _ec_obj_open_span(obj_model, tcp_rot)
    if span is None:
        return 0.0
    _t0, _t1, mid, _w, open_dir = span
    tcp_t = float(np.dot(tcp_pos, open_dir))
    shift = float(mid - tcp_t)
    if abs(shift) < float(min_shift):
        return 0.0
    # 限制单次纠偏，避免把指尖扯出螺头
    shift = float(np.clip(shift, -0.008, 0.008))
    new_tcp = tcp_pos + open_dir * shift
    grasp.ac_pos = (obj_rot.T @ (new_tcp - obj_pos)).reshape(3)
    return shift


def _ec_fit_jaw_close_to_sides(obj_model, grasp, grip_width, jc_contact, jaw_open):
    """闭合宽 = 螺头张合向外廓 + 约 1mm 间隙（厚指 CD 常在 ~10mm 早碰，须按外廓纠正）。"""
    _, tcp_rot = _ec_grasp_tcp(obj_model, grasp)
    w_open = _ec_obj_width_along_open(obj_model, tcp_rot)
    gw = float(max(grip_width or 0.0, side.JAW_CLOSE_MIN))
    w = max(float(w_open), gw)
    gap = float(EC_JAW_HEAD_CLEARANCE_M)
    jc_fit = float(np.clip(w + gap, side.JAW_CLOSE_MIN, float(jaw_open) - 1e-4))
    if jc_contact is None:
        return jc_fit, w
    jc_c = float(jc_contact)
    if jc_c > jc_fit + 0.0005:
        print(
            f"[jaw] 接触偏松 {jc_c:.4f} → 外廓+{gap*1e3:.0f}mm间隙 {jc_fit:.4f} "
            f"(open_w={w:.4f})",
            flush=True,
        )
        return jc_fit, w
    # 不低于「外廓+1mm」；接触更紧时也抬到间隙宽，避免贴死螺头
    jc = float(np.clip(max(jc_c, jc_fit), jc_fit, float(jaw_open) - 1e-4))
    if abs(jc - jc_c) > 1e-4:
        print(
            f"[jaw] 闭合留隙 {jc_c:.4f} → {jc:.4f} "
            f"(外廓{w:.4f}+{gap*1e3:.0f}mm)",
            flush=True,
        )
    return jc, w


def ec_sanitize_pick_grasps(
    robot,
    obj_model,
    part_grasps,
    jaw_open,
    jaw_close_seed,
    grasp_mode: str = "topdown",
    grip_width: float = 0.005,
):
    """逐条验收 pick：姿态合格 + 张开不穿模 + 接触闭合 + 外廓贴紧 + 不深穿模。

    每条 grasp 写入自己的 ``ee_values``（权威闭爪宽）。
    返回的 ``jaw_close`` 只是各 grasp 的中位**种子/回退值**，供尚未选定
    grasp 时的探测用；一旦选定 grasp，规划侧必须改用 ``grasp.ee_values``。
    """
    kept = gg.GraspCollection(end_effector=robot.end_effector)
    jcs = []
    seed = np.asarray(side.HOME_CONF, dtype=float).copy()
    jaw_open = float(jaw_open)
    n_pose = n_open = n_jaw = n_close = n_ik = 0
    squeeze = float(getattr(side, "JAW_CONTACT_SQUEEZE_SCREW", 0.0))
    probe_eps = float(getattr(side, "JAW_CLEARANCE_PROBE", 0.0015))

    for g in part_grasps:
        if not _ec_pick_pose_quality(g, obj_model, grasp_mode=grasp_mode):
            n_pose += 1
            continue
        # 0) 张合方向居中：消除「螺丝贴一侧、另一侧大空隙」
        sh = _ec_center_grasp_along_open(obj_model, g)
        if abs(float(sh)) >= 2e-4:
            print(
                f"[grasp] 张合轴居中 Δ={sh*1e3:.1f}mm "
                f"(mode={grasp_mode})",
                flush=True,
            )
        tcp_pos, tcp_rot = _ec_grasp_tcp(obj_model, g)
        q = side._ik_with_seed(robot, tcp_pos, tcp_rot, seed)
        if q is None:
            n_ik += 1
            continue
        seed = np.asarray(q, dtype=float).copy()
        # 1) 张开：整爪不得穿零件
        robot.goto_given_conf(jnt_values=q, ee_values=float(jaw_open))
        if _ec_ee_hits_object(robot, obj_model, jaw_open):
            n_open += 1
            continue
        # 2) 接触搜索闭合
        jc = side.refine_jaw_close_by_contact(
            robot,
            obj_model,
            g,
            jaw_open=min(jaw_open + 0.005, side.JAW_OPEN_MAX),
            jaw_seed=float(jaw_close_seed),
            squeeze=squeeze,
        )
        if jc is None:
            n_jaw += 1
            continue
        # 3) 按张合方向外廓贴紧（修空隙过大）
        jc_contact = float(jc)
        jc, w_open = _ec_fit_jaw_close_to_sides(
            obj_model, g, grip_width, jc_contact, jaw_open
        )
        # 厚指 CD 常在螺头处早报「深穿」，细螺丝优先外廓贴紧（尖端视觉贴合）
        robot.goto_given_conf(jnt_values=q, ee_values=float(jc) + probe_eps)
        cd_deep = _ec_ee_hits_object(robot, obj_model, float(jc) + probe_eps)
        if cd_deep and float(w_open) > 0.012:
            lo, hi = float(jc), min(float(jc) + 0.008, jaw_open - 1e-4)
            ok_jc = None
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                robot.goto_given_conf(jnt_values=q, ee_values=mid + probe_eps)
                if _ec_ee_hits_object(robot, obj_model, mid + probe_eps):
                    lo = mid
                else:
                    ok_jc = mid
                    hi = mid
            if ok_jc is None:
                n_close += 1
                continue
            jc = float(ok_jc)
        elif cd_deep:
            print(
                f"[jaw] 细件保留外廓贴紧 {jc:.4f}（CD深穿多为厚指误报，"
                f"接触曾={jc_contact:.4f}）",
                flush=True,
            )
        if not side.conf_tip_clears_desk(robot, q, jc):
            n_close += 1
            continue
        g.ee_values = float(jc)
        kept.append(g)
        jcs.append(float(jc))

    if len(kept) == 0:
        print(
            f"[pick-cd] 全部不合格 pose拒={n_pose} 张穿={n_open} "
            f"接触失败={n_jaw} 深穿={n_close} ik失败={n_ik}",
            flush=True,
        )
        return kept, float(jaw_close_seed)

    # 中位仅作种子；真正写入轨迹的是选定 grasp 的 ee_values
    jaw_close_seed_out = float(np.median(np.asarray(jcs, dtype=float)))
    gw = float(max(grip_width or 0.005, side.JAW_CLOSE_MIN))
    jaw_close_seed_out = float(
        np.clip(
            jaw_close_seed_out,
            max(gw * 0.90, side.JAW_CLOSE_MIN),
            jaw_open - 1e-4,
        )
    )
    print(
        f"[pick-cd] 合格 {len(kept)}/{len(list(part_grasps))} "
        f"jaw_close_seed≈{jaw_close_seed_out:.4f} "
        f"(单grasp={np.round(jcs, 4).tolist()}) "
        f"(pose拒={n_pose} 张穿={n_open} 接触失败={n_jaw} 深穿={n_close})",
        flush=True,
    )
    return kept, jaw_close_seed_out


def _local_nudge_for_topdown_ik(
    robot, target, topdown_pickle, max_shift: float = 0.03, step: float = 0.01
):
    """在标注 XY 附近做最小平移，使顶抓至少有一个 IK 解。

    返回新 xy（已写入 target）；失败返回 None。不跨桌面挪缓冲。
    """
    if not topdown_pickle or not os.path.isfile(str(topdown_pickle)):
        return None
    try:
        grasps = gg.GraspCollection.load_from_disk(str(topdown_pickle))
    except Exception:
        return None
    if grasps is None or len(grasps) == 0:
        return None
    m = target.get("model")
    if m is None:
        return None
    xy0 = np.asarray(target.get("pos", m.pos), dtype=float)[:2].copy()
    z0 = float(np.asarray(target.get("pos", m.pos), dtype=float)[2])
    Rm = np.asarray(m.rotmat, dtype=float)
    seed = np.asarray(side.HOME_CONF, dtype=float)

    def _ik_count(xy, n_probe: int = 12) -> int:
        pos = np.array([float(xy[0]), float(xy[1]), z0], dtype=float)
        n_ok = 0
        for i, g in enumerate(grasps):
            if i >= int(n_probe):
                break
            tcp_pos = pos + Rm @ np.asarray(g.ac_pos, dtype=float)
            tcp_rot = Rm @ np.asarray(g.ac_rotmat, dtype=float)
            q = robot.ik(
                tgt_pos=tcp_pos, tgt_rotmat=tcp_rot, seed_jnt_values=seed
            )
            if q is not None:
                n_ok += 1
        return int(n_ok)

    n0 = _ik_count(xy0)
    if n0 >= 5:
        return xy0
    # 优先朝原点（工作空间内侧）微移，再环扫；要足够多 IK 解，避免贴边界「勉强有解但闭爪穿模」
    cands = []
    toward = -xy0
    nrm = float(np.linalg.norm(toward))
    if nrm > 1e-6:
        toward = toward / nrm
        for r in np.arange(step, float(max_shift) + 1e-9, step):
            cands.append((float(r), xy0 + toward * float(r)))
    for r in np.arange(step, float(max_shift) + 1e-9, step):
        for ang in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            cands.append(
                (
                    float(r),
                    xy0
                    + float(r)
                    * np.array([np.cos(ang), np.sin(ang)], dtype=float),
                )
            )
    best_xy = None
    best_key = None  # (-n_ok, r) 最小更好
    for r, xy in cands:
        n_ok = _ik_count(xy)
        if n_ok < 8:
            continue
        # 优先 IK 多；同等时略偏向 ≥2cm，避免贴边界勉强解导致闭爪穿模
        key = (-n_ok, abs(float(r) - 0.02), float(r))
        if best_key is None or key < best_key:
            best_key = key
            best_xy = np.asarray(xy, dtype=float).copy()
    if best_xy is None:
        # 放宽：至少 5 个 IK
        for r, xy in cands:
            n_ok = _ik_count(xy)
            if n_ok < 5:
                continue
            key = (-n_ok, float(r))
            if best_key is None or key < best_key:
                best_key = key
                best_xy = np.asarray(xy, dtype=float).copy()
    if best_xy is None:
        return None
    pos = np.array([float(best_xy[0]), float(best_xy[1]), z0], dtype=float)
    try:
        m.pos = pos
    except Exception:
        pass
    target["pos"] = pos.copy()
    return best_xy


def pick_buffer_xy(part_infos, target=None) -> np.ndarray:
    """选一个远离零件/料盘的缓冲区 XY。"""
    occupied = []
    tid = int(target.get("id", -1)) if target else -1
    for info in part_infos or []:
        if int(info.get("id", -2)) == tid:
            continue
        occupied.append(np.asarray(info["model"].pos, dtype=float)[:2])
    md2 = 0.04 ** 2
    for xy in EC_BUFFER_XY_CANDS:
        if any(float(np.dot(xy - o, xy - o)) < md2 for o in occupied):
            continue
        return np.asarray(xy, dtype=float).copy()
    return np.asarray(EC_BUFFER_XY_CANDS[0], dtype=float).copy()


def remesh_part_as_fallen(info, xy, yaw: float = 0.0, base=None):
    """把零件在桌面 xy 重烘焙为 fallen 平躺（长轴水平；供 inverted 缓冲后第二段）。"""
    stl = info.get("stl_path") or BUGLE_STL
    # Bugle 长轴=+Y：roll=pitch=0 为平躺；旧值 π/2 实际是正立
    roll = 0.0
    pitch = 0.0
    yaw = float(yaw)
    mesh, height, grip_width = side._load_mesh_on_table(
        stl, yaw, roll=roll, pitch=pitch
    )
    old = info.get("model")
    rgb = None
    name = str(info.get("class") or PART_CLASS)
    if old is not None:
        name = str(getattr(old, "name", None) or name)
        try:
            rgb = np.asarray(old.rgb, dtype=float)
        except Exception:
            rgb = None
        try:
            old.detach()
        except Exception:
            pass
    if rgb is None:
        rgb = rm.vec(0.70, 0.55, 0.40)
    model = mcm.CollisionModel(mesh, name=name, rgb=rgb)
    pos = np.array([float(xy[0]), float(xy[1]), 0.0], dtype=float)
    model.pos = pos
    model.rotmat = np.eye(3)
    info["model"] = model
    info["pos"] = pos.copy()
    info["height"] = float(height)
    info["top_z"] = float(height)
    info["mid_z"] = 0.5 * float(height)
    info["grip_width"] = float(grip_width)
    info["state"] = "fallen"
    info["roll"] = info["bake_roll"] = roll
    info["pitch"] = info["bake_pitch"] = pitch
    info["yaw"] = info["bake_yaw"] = yaw
    if base is not None:
        model.attach_to(base)
    print(
        f"[buffer] id={info.get('id')} → fallen @ "
        f"{np.round(pos[:2], 3).tolist()} yaw={yaw:.2f}",
        flush=True,
    )
    return info


def ec_place_base_conf_for_slot(slot_xy, slot_id: int = 0, force_ref: bool = False):
    """格子 → 放置主体构型；fallen 时 force_ref 钉死参考 J2–J6。"""
    p = np.asarray(slot_xy, dtype=float).reshape(-1)[:2]
    j1 = float(np.arctan2(float(p[1]), float(p[0]))) + float(EC_J1_CW_BIAS)
    if force_ref:
        q = EC_PLACE_REF_ARM.copy()
    else:
        q = np.asarray(_PLACE_STYLE["q_arm"], dtype=float).copy()
    q[0] = j1
    _ = slot_id
    return q


EC_J3_MIN = float(np.deg2rad(-5.0))
EC_SEAT_FLOOR_Z = float(box_world_floor_z())
EC_SEAT_CLEARANCE = 1.5e-4
EC_GOOD_RELEASE_TCP_Z = 0.055
EC_ACCEPT_RELEASE_TCP_Z = 0.090


def _ec_conf_self_ok(robot, q, jaw_close) -> bool:
    """构型无自碰且指尖不穿桌。"""
    try:
        robot.goto_given_conf(
            np.asarray(q, dtype=float), ee_values=float(jaw_close)
        )
    except Exception:
        return False
    if bool(robot.is_collided()):
        return False
    return bool(side.conf_tip_clears_desk(robot, q, jaw_close))


def _ec_ee_desk_clear(robot) -> bool:
    """夹爪座/法兰不扎桌（is_collided 查不到腕部自碰与法兰穿桌）。"""
    try:
        z_ee = float(np.asarray(robot.end_effector.pos, dtype=float)[2])
    except Exception:
        z_ee = float(robot.gl_tcp_pos[2])
    return z_ee >= float(EC_EE_DESK_CLEAR_Z)


def _ec_conf_place_ok(robot, q, jaw_close, require_j6: bool = False) -> bool:
    """放置专用：J5 + 自碰/指尖 + 法兰离桌；松爪帧再要求 J6 拧腕。"""
    q = np.asarray(q, dtype=float)
    j5 = float(q[4])
    if j5 < float(EC_J5_HARD_MIN) or j5 > float(EC_J5_HARD_MAX):
        return False
    if require_j6 and (not _ec_j6_place_ok(q)):
        return False
    if not _ec_conf_self_ok(robot, q, jaw_close):
        return False
    return bool(_ec_ee_desk_clear(robot))


def _ec_motion_self_ok(robot, mot, jaw_close, sample_step=None) -> bool:
    """轨迹抽检：任一点自碰则失败（统一用 jaw_close，避免张爪宽度干扰）。"""
    if mot is None:
        return False
    n = len(mot.jv_list)
    if n <= 0:
        return False
    step = max(1, n // 20) if sample_step is None else max(1, int(sample_step))
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    for i in idxs:
        if not _ec_conf_self_ok(robot, mot.jv_list[i], jaw_close):
            return False
    return True


def _ec_motion_place_ok(robot, mot, jaw_close, sample_step=None) -> bool:
    """放置轨迹抽检：含 J5/法兰约束。"""
    if mot is None:
        return False
    n = len(mot.jv_list)
    if n <= 0:
        return False
    step = max(1, n // 20) if sample_step is None else max(1, int(sample_step))
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    for i in idxs:
        if not _ec_conf_place_ok(robot, mot.jv_list[i], jaw_close):
            return False
    return True


def _ec_tipped_place_obj_rot(tip_deg: float, base_obj_rot=None):
    """入格物体姿态：在竖直基础上再低头 tip_deg（接近轴带 −Z）。"""
    base = UPRIGHT_OBJ_ROT if base_obj_rot is None else np.asarray(base_obj_rot)
    tip = float(np.deg2rad(tip_deg))
    if abs(tip) < 1e-9:
        return np.asarray(base, dtype=float).copy()
    # Rx(π/2 - tip)：长轴仍近竖直，夹爪接近方向带俯角
    return rm.rotmat_from_euler(float(np.pi / 2.0) - tip, 0.0, 0.0)


def _ec_place_tcp_xy(slot_xy, grasp, tcp_rot):
    """零件中心在格心时的 TCP XY（完整 grasp 偏置，不对齐截断）。

    松爪对齐的是螺丝/指尖接触处，不是夹爪轨中心；必须用物体中心反算 TCP。
    """
    xy = np.asarray(slot_xy, dtype=float).reshape(-1)[:2]
    if grasp is None:
        return xy.copy()
    ac_r = np.asarray(grasp.ac_rotmat, dtype=float)
    ac_p = np.asarray(grasp.ac_pos, dtype=float).reshape(3)
    R_tcp = np.asarray(tcp_rot, dtype=float)
    R_obj = R_tcp @ ac_r.T
    off = np.asarray(R_obj @ ac_p, dtype=float)[:2]
    return np.array([float(xy[0] + off[0]), float(xy[1] + off[1])], dtype=float)


def _ec_obj_xy_from_tcp(robot, grasp, tcp_rot=None):
    """由当前 TCP 反算持物中心 XY。"""
    ac_r = np.asarray(grasp.ac_rotmat, dtype=float)
    ac_p = np.asarray(grasp.ac_pos, dtype=float).reshape(3)
    R_tcp = (
        np.asarray(tcp_rot, dtype=float)
        if tcp_rot is not None
        else np.asarray(robot.gl_tcp_rotmat, dtype=float)
    )
    R_obj = R_tcp @ ac_r.T
    obj = np.asarray(robot.gl_tcp_pos, dtype=float) - R_obj @ ac_p
    return obj[:2]


def _ec_stable_sort_grasps(part_grasps):
    """稳定、可复现的 grasp 排序：偏置小 → 更顶抓 → 姿态字典序。"""
    ranked = []
    for i, g in enumerate(part_grasps):
        ac_p = np.asarray(g.ac_pos, dtype=float).reshape(3)
        ac_r = np.asarray(g.ac_rotmat, dtype=float)
        # 顶抓：接近轴 ≈ −Z
        topdown = -float(ac_r[2, 2])
        key = (
            float(np.linalg.norm(ac_p)),
            -topdown,
            tuple(np.round(ac_p, 5).tolist()),
            tuple(np.round(ac_r.reshape(-1), 5).tolist()),
            int(i),
        )
        ranked.append((key, g))
    ranked.sort(key=lambda t: t[0])
    return [g for _, g in ranked]


def _ec_vertical_place_tcp_rots(
    grasp, place_obj_rot=None, tip_degs=None, place_kind: str = None
):
    """零件入格 TCP：由 grasp 反算，使松爪时物体长轴近竖直。

    place_kind:
      fallen_ref — 优先 tip=15°/dyaw=-45°
      normal_vertical — 优先 tip=0° 且夹爪接近轴竖直（tip_down≈1）
    """
    base_obj = (
        UPRIGHT_OBJ_ROT if place_obj_rot is None else np.asarray(place_obj_rot)
    )
    ac = np.asarray(grasp.ac_rotmat, dtype=float)
    ac_p = np.asarray(grasp.ac_pos, dtype=float).reshape(3)
    kind = str(place_kind or _PLACE_STYLE.get("kind") or "fallen_ref")
    if tip_degs is not None:
        tip_list = tuple(tip_degs)
    elif kind == "normal_vertical":
        tip_list = (0.0,)
    else:
        tip_list = tuple(
            EC_PLACE_TIP_DEGS_FAST if not EC_VERBOSE else EC_PLACE_APPROACH_TIP_DEGS
        )
    scored = []
    for tip_deg in tip_list:
        place_obj = (
            np.asarray(base_obj, dtype=float).copy()
            if abs(float(tip_deg)) < 1e-9
            else _ec_tipped_place_obj_rot(tip_deg, base_obj_rot=base_obj)
        )
        dyaws = (
            (0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi, 0.25 * np.pi, -0.25 * np.pi)
            if kind == "normal_vertical"
            else (
                # fallen REF：围绕 -45° 多试，避免只剩 1 个朝向导致 J6 无拧腕解
                -0.25 * np.pi,
                -0.20 * np.pi,
                -0.30 * np.pi,
                -0.125 * np.pi,
                -0.375 * np.pi,
                0.0,
                -0.5 * np.pi,
                0.25 * np.pi,
                np.pi,
                0.5 * np.pi,
            )
        )
        for dyaw in dyaws:
            R_obj = rm.rotmat_from_euler(0.0, 0.0, float(dyaw)) @ place_obj
            R_tcp = R_obj @ ac
            align_x = abs(float(R_tcp[0, 0]))
            tip_down = max(0.0, -float(R_tcp[2, 2]))
            upright = abs(float(R_obj[2, 1]))
            off_xy = float(np.linalg.norm((R_obj @ ac_p)[:2]))
            scored.append(
                (upright, off_xy, tip_down, align_x, float(dyaw), float(tip_deg), R_tcp)
            )
    if kind == "normal_vertical":
        # 夹爪竖直 ⊥XY 优先，再物体直立、偏置小
        scored.sort(
            key=lambda t: (
                -float(t[2]),
                abs(float(t[5])),
                -float(t[0]),
                float(t[1]),
            )
        )
    else:
        tip_pref = float(_PLACE_STYLE.get("tip_deg", EC_PLACE_REF_TIP_DEG))
        dyaw_pref = float(_PLACE_STYLE.get("dyaw", EC_PLACE_REF_DYAW))
        scored.sort(
            key=lambda t: (
                abs(float(t[5]) - tip_pref),
                abs(float(t[4]) - dyaw_pref),
                -float(t[2]),
                -float(t[0]),
                float(t[1]),
            )
        )
    return scored


def _sample_clear_xy(occupied_xy, rng=None):
    """在桌面安全区采一个远离 occupied 的空位。"""
    from tiaozhanbei.sim.ec.ec_layout import (
        PART_MIN_CENTER_DIST,
        PART_SPAWN_XY_MAX,
        PART_SPAWN_XY_MIN,
        is_valid_part_xy,
    )

    rng = np.random.default_rng() if rng is None else rng
    occ = [np.asarray(p, dtype=float)[:2] for p in occupied_xy]
    md2 = float(PART_MIN_CENTER_DIST) ** 2
    for _ in range(800):
        xy = np.array(
            [
                float(rng.uniform(PART_SPAWN_XY_MIN[0], PART_SPAWN_XY_MAX[0])),
                float(rng.uniform(PART_SPAWN_XY_MIN[1], PART_SPAWN_XY_MAX[1])),
            ],
            dtype=float,
        )
        if not is_valid_part_xy(xy):
            continue
        if any(float(np.dot(xy - o, xy - o)) < md2 for o in occ):
            continue
        return xy
    return None


def find_blocking_neighbors(target, part_infos, max_n=None, radius=None):
    """目标附近仍在桌面的挡路件（近→远）。已入格（偏料盘）的不算。"""
    max_n = int(EC_CLEAR_NEIGHBOR_MAX if max_n is None else max_n)
    radius = float(EC_CLEAR_NEIGHBOR_RADIUS if radius is None else radius)
    tpos = np.asarray(target["model"].pos, dtype=float)[:2]
    cands = []
    for info in part_infos:
        if int(info.get("id", -1)) == int(target.get("id", -2)):
            continue
        p = np.asarray(info["model"].pos, dtype=float)[:2]
        if float(p[0]) > 0.22:
            continue
        d = float(np.linalg.norm(p - tpos))
        if d < radius:
            cands.append((d, info))
    cands.sort(key=lambda t: t[0])
    return [info for _, info in cands[:max_n]]


def clear_nearest_neighbors_for_pick(target, part_infos, max_n=None, radius=None):
    """兼容旧调用：仅查找挡路件，不再瞬移（清障改走真实缓冲放置）。"""
    blockers = find_blocking_neighbors(
        target, part_infos, max_n=max_n, radius=radius
    )
    if blockers:
        print(
            f"[clear] 检测到挡路件 ids={[b.get('id') for b in blockers]} "
            f"（将尝试抓到缓冲区）",
            flush=True,
        )
    return []


def restore_cleared_neighbors(moved):
    for info, old in moved or []:
        info["model"].pos = np.asarray(old, dtype=float).copy()
        info["pos"] = np.asarray(old, dtype=float).copy()


def resolve_obstructor(
    robot,
    blocker,
    part_infos,
    desk_obstacle,
    box_model,
    grasp_topdown,
    grasp_side,
    use_rrt,
    base=None,
    pending_jobs=None,
    strategy_fallen: str = "auto",
):
    """清障：normal/fallen 有 slot 则直接入格；inverted（或无 job）→ 缓冲区。

    返回 (pkg, finished_job|None)。finished_job 表示该挡路件已入格，应从 pending 移除。
    """
    st = _norm_state(blocker.get("state") or "fallen")
    bid = int(blocker.get("id", -1))
    job = None
    for j in pending_jobs or []:
        if int(j["object_id"]) == bid:
            job = j
            break
    desk_snap = _snapshot_part_pose(blocker)

    # normal / fallen：顶抓直接放入自己的 JSON 格子
    if st in ("normal", "fallen") and job is not None:
        slot_id = int(job["slot_id"])
        print(
            f"[clear] 挡路 id={bid} state={st} → 直接入格 slot={slot_id}",
            flush=True,
        )
        common = dict(
            robot=robot,
            target=blocker,
            part_infos=part_infos,
            desk_obstacle=desk_obstacle,
            slot_id=slot_id,
            grasp_topdown=grasp_topdown,
            grasp_side=grasp_side,
            use_rrt=use_rrt,
            adjust_to_normal=bool(job.get("adjust_to_normal")),
            place_yaw=job.get("place_yaw"),
            box_model=box_model,
            place_surface="box",
            require_upright=True,
            lock_style=True,
            enforce_style=True,
            seat_kind="slot",
        )
        if st == "normal":
            pkg = plan_one_job(
                **common,
                strategy_prefer="topdown",
                pipeline="normal",
                tip_degs=EC_PLACE_TIP_DEGS_TOP,
            )
        else:
            pkg = plan_one_job(
                **common,
                strategy_prefer=str(strategy_fallen or "auto"),
                pipeline="fallen",
            )
        pkg["desk_snap"] = desk_snap
        pkg["skip_pre_anime_restore"] = False
        pkg["is_buffer_clear"] = False
        pkg["cleared_early_slot"] = True
        seat_in_slot(pkg, base, robot, box_model=box_model)
        side._reset_robot_hand(robot, pkg["jaw_open"])
        robot.goto_given_conf(side.HOME_CONF, ee_values=float(pkg["jaw_open"]))
        return pkg, job

    # inverted 或无对应 job：侧夹/侧倾放入缓冲区（勿竖直顶放）
    buf_xy = pick_buffer_xy(part_infos, blocker)
    print(
        f"[clear] 挡路 id={bid} state={st} → 缓冲区 "
        f"{np.round(buf_xy, 3).tolist()}",
        flush=True,
    )
    common = dict(
        robot=robot,
        target=blocker,
        part_infos=part_infos,
        desk_obstacle=desk_obstacle,
        slot_id=0,
        grasp_topdown=grasp_topdown,
        grasp_side=grasp_side,
        use_rrt=use_rrt,
        adjust_to_normal=False,
        place_yaw=float(blocker.get("yaw") or 0.0),
        box_model=box_model,
        place_xy=buf_xy,
        place_surface="desk",
        require_upright=False,
        lock_style=False,
        enforce_style=False,
        tip_degs=(45.0, 40.0, 50.0, 35.0),
        buffer_yaw=float(blocker.get("yaw") or 0.0),
    )
    if st == "inverted":
        pkg = plan_one_job(
            **common,
            strategy_prefer="topdown",
            pipeline="inverted",
            seat_kind="buffer_fallen",
        )
    else:
        # 无 slot job 的 normal/fallen：仍暂存缓冲
        pkg = plan_one_job(
            **common,
            strategy_prefer="topdown",
            pipeline="normal" if st == "normal" else "fallen",
            seat_kind="buffer_clear",
        )
    pkg["desk_snap"] = desk_snap
    pkg["skip_pre_anime_restore"] = False
    pkg["is_buffer_clear"] = True
    seat_in_slot(pkg, base, robot, box_model=box_model)
    side._reset_robot_hand(robot, pkg["jaw_open"])
    robot.goto_given_conf(side.HOME_CONF, ee_values=float(pkg["jaw_open"]))
    return pkg, None


def rank_jobs_inverted_last(jobs) -> list:
    """先 fallen/normal（按 slot_id），再 inverted（按 slot_id）。

    格子映射不变；倒放需缓冲 remesh，放到后面通常更稳。
    """
    def _key(j):
        st = _norm_state(j.get("state") or "normal")
        inv = 1 if st == "inverted" else 0
        return (inv, int(j["slot_id"]), int(j["object_id"]))

    return sorted(list(jobs), key=_key)


def rank_jobs_by_pick_ease(jobs, part_infos) -> list:
    """易抓优先排序；保留每个 job 的 object_id→slot_id（JSON 格子不变）。

    启发：最近邻越远、越靠外围 → 越先抓；inverted 故意靠后。
    """
    if not jobs:
        return []
    by_id = {int(p["id"]): p for p in (part_infos or [])}
    xy = {}
    for j in jobs:
        oid = int(j["object_id"])
        info = by_id.get(oid)
        if info is not None:
            xy[oid] = np.asarray(info.get("pos", info["model"].pos), dtype=float)[:2]
        else:
            c = j.get("coordinate") or [0.0, 0.0]
            xy[oid] = np.asarray(c, dtype=float)[:2]

    def _ease(j):
        oid = int(j["object_id"])
        p = xy[oid]
        others = [
            float(np.linalg.norm(p - xy[int(o["object_id"])]))
            for o in jobs
            if int(o["object_id"]) != oid and int(o["object_id"]) in xy
        ]
        # 仍在桌面上的邻件（已入格的 pos 在料盘，距离大，不挡）
        min_d = min(others) if others else 0.2
        st = _norm_state(
            j.get("state")
            or (by_id.get(oid) or {}).get("state")
            or "normal"
        )
        # inverted 最后处理（缓冲调姿更重）；fallen/normal 先清场
        st_pen = {"inverted": -1.0, "fallen": 0.01, "normal": 0.02}.get(st, 0.0)
        r = float(np.linalg.norm(p))
        # 已入格零件在 box 附近，对桌面目标 min_d 仍用当前 pos
        return (float(min_d) + st_pen + 0.12 * min(r, 0.28), -oid)

    ranked = sorted(list(jobs), key=_ease, reverse=True)
    return ranked


def _ec_ik_slot_rot(
    robot,
    xy,
    z,
    tcp_rot,
    seed,
    jaw_close,
    pos_tol=0.02,
    prefer_style: bool = True,
    desk_relax: bool = False,
):
    """格心/缓冲 IK；箱内优先统一臂型，桌面缓冲以 seed 为主。"""
    xy = np.asarray(xy, dtype=float).reshape(-1)[:2]
    seed0 = np.asarray(seed, dtype=float).copy()
    R = np.asarray(tcp_rot, dtype=float)
    tgt = np.array([float(xy[0]), float(xy[1]), float(z)], dtype=float)
    j1_pref = float(np.arctan2(float(xy[1]), float(xy[0]))) + float(EC_J1_CW_BIAS)
    ref = np.asarray(_PLACE_STYLE["q_arm"], dtype=float).copy()
    seeds = []
    if prefer_style:
        s_ref = ref.copy()
        s_ref[0] = j1_pref
        seeds.append(s_ref)
        for dj1 in (0.0, -EC_J1_WIGGLE, EC_J1_WIGGLE):
            s = ref.copy()
            s[0] = j1_pref + float(dj1)
            seeds.append(s)
    seeds.append(seed0)
    for dj1 in (0.0, -EC_J1_WIGGLE, EC_J1_WIGGLE, -2 * EC_J1_WIGGLE, 2 * EC_J1_WIGGLE):
        s = seed0.copy()
        s[0] = j1_pref + float(dj1)
        seeds.append(s)
    if prefer_style:
        for dj2, dj4, dj5, dj6 in (
            (0.0, 0.0, 0.0, 0.0),
            (5.0, 0.0, 5.0, 0.0),
            (-5.0, 5.0, -5.0, 10.0),
            (10.0, -10.0, 0.0, -10.0),
        ):
            s = ref.copy()
            s[0] = j1_pref
            s[1] += float(np.deg2rad(dj2))
            s[3] += float(np.deg2rad(dj4))
            s[4] += float(np.deg2rad(dj5))
            s[5] += float(np.deg2rad(dj6))
            seeds.append(s)
    else:
        for dj2, dj4, dj5, dj6 in (
            (0.0, 0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0, 0.0),
            (-10.0, 10.0, 0.0, 20.0),
            (0.0, -15.0, 15.0, -20.0),
            (20.0, -20.0, 10.0, 0.0),
            (-15.0, 0.0, -10.0, 30.0),
        ):
            s = seed0.copy()
            s[0] = j1_pref
            s[1] += float(np.deg2rad(dj2))
            s[3] += float(np.deg2rad(dj4))
            s[4] += float(np.deg2rad(dj5))
            s[5] += float(np.deg2rad(dj6))
            seeds.append(s)
    best = None
    best_key = None
    for seed_i in seeds:
        q = robot.ik(tgt_pos=tgt, tgt_rotmat=R, seed_jnt_values=seed_i)
        if q is None:
            continue
        q = np.asarray(q, dtype=float).copy()
        if float(q[2]) < EC_J3_MIN:
            q[2] = EC_J3_MIN
        if not _ec_conf_place_ok(robot, q, jaw_close):
            continue
        err = float(np.linalg.norm(np.asarray(robot.gl_tcp_pos, dtype=float) - tgt))
        if err > float(pos_tol):
            continue
        j2 = float(q[1])
        j6_err = abs(float(q[5]) - float(EC_PLACE_J6_REF))
        # 无拧腕的水平 J6≈0 必须劣后
        j6_bad = 0 if _ec_j6_place_ok(q) else 1
        arm_err = float(np.linalg.norm(q[1:6] - ref[1:6]))
        seed_err = float(np.linalg.norm(q - seed0))
        fwd = 0 if (EC_J2_FORWARD_MIN <= j2 <= EC_J2_FORWARD_MAX) else 1
        if prefer_style:
            key = (j6_bad, j6_err, arm_err, fwd, err)
        elif desk_relax:
            # 缓冲 remesh：不锁拧腕 J6，只求到位
            key = (err, seed_err, fwd)
        else:
            key = (j6_bad, seed_err, err, fwd)
        if best is None or key < best_key:
            best, best_key = q, key
    return best


def _ec_desk_rot_cands(R_pick, prefer_tilt: bool = False, tip_degs=None):
    """桌面缓冲姿态候选。

    prefer_tilt=True（inverted→缓冲）：侧倾优先，竖直顶放仅作兜底，
    松爪后更像「倒下」成 fallen。
    """
    R_pick = np.asarray(R_pick, dtype=float)
    R_top = np.eye(3)
    R_top[:, 2] = np.array([0.0, 0.0, -1.0], dtype=float)
    R_top[:, 0] = np.array([1.0, 0.0, 0.0], dtype=float)
    R_top[:, 1] = np.cross(R_top[:, 2], R_top[:, 0])
    yaws = (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi)
    if prefer_tilt:
        tips = tuple(tip_degs) if tip_degs is not None else (25.0, 15.0, 35.0, 40.0)
        out = []
        # 先试传入的侧倾 TCP，再试竖直基座上的 tip 变体
        out.append(R_pick.copy())
        for tip_deg in tips:
            tip = float(np.deg2rad(float(tip_deg)))
            if abs(tip) < 1e-6:
                continue
            for sgn in (1.0, -1.0):
                Rx = rm.rotmat_from_euler(sgn * tip, 0.0, 0.0)
                Ry = rm.rotmat_from_euler(0.0, sgn * tip, 0.0)
                for yaw in yaws:
                    Rz = rm.rotmat_from_euler(0.0, 0.0, float(yaw))
                    out.append(Rz @ R_top @ Rx)
                    out.append(Rz @ R_top @ Ry)
                    out.append(Rz @ R_pick @ Rx)
                    out.append(Rz @ R_pick @ Ry)
        for yaw in yaws:
            Rz = rm.rotmat_from_euler(0.0, 0.0, float(yaw))
            out.append(Rz @ R_pick)
        # 竖直顶放最后兜底
        out.append(R_top)
        for yaw in yaws:
            Rz = rm.rotmat_from_euler(0.0, 0.0, float(yaw))
            out.append(Rz @ R_top)
        return out
    out = [R_top]
    for yaw in yaws:
        Rz = rm.rotmat_from_euler(0.0, 0.0, float(yaw))
        out.append(Rz @ R_top)
        out.append(Rz @ R_pick)
    out.append(R_pick)
    return out


def _ec_tilt_desk_tcp_cands(R_pick, tip_degs):
    """从持物末 TCP 生成侧倾桌面松爪候选（非 tip=0 顶放）。"""
    R0 = np.asarray(R_pick, dtype=float).copy()
    tips = tuple(tip_degs) if tip_degs is not None else (45.0, 40.0, 50.0, 35.0)
    scored = []
    for tip_deg in tips:
        tip = float(np.deg2rad(float(tip_deg)))
        variants = [R0]
        if abs(tip) >= 1e-6:
            for sgn in (1.0, -1.0):
                variants.append(R0 @ rm.rotmat_from_euler(sgn * tip, 0.0, 0.0))
                variants.append(R0 @ rm.rotmat_from_euler(0.0, sgn * tip, 0.0))
                # 相对竖直接近轴侧倾
                R_top = np.eye(3)
                R_top[:, 2] = np.array([0.0, 0.0, -1.0], dtype=float)
                R_top[:, 0] = np.array([1.0, 0.0, 0.0], dtype=float)
                R_top[:, 1] = np.cross(R_top[:, 2], R_top[:, 0])
                for yaw in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
                    Rz = rm.rotmat_from_euler(0.0, 0.0, float(yaw))
                    variants.append(
                        Rz @ R_top @ rm.rotmat_from_euler(sgn * tip, 0.0, 0.0)
                    )
        for R_tcp in variants:
            tip_down = max(0.0, -float(R_tcp[2, 2]))
            # 偏好侧倾：app↓≈cos(25°)≈0.91，惩罚竖直顶放≈1.0
            scored.append(
                (
                    1.0,
                    0.0,
                    tip_down,
                    abs(float(R_tcp[0, 0])),
                    0.0,
                    float(tip_deg),
                    np.asarray(R_tcp, dtype=float).copy(),
                )
            )
    # 侧倾优先：默认贴 ~45°（app↓≈0.71）；tip_deg 接近 45 加分
    scored.sort(
        key=lambda t: (
            abs(float(t[2]) - 0.71),
            abs(abs(float(t[5])) - 45.0),
            -abs(float(t[5])),
        )
    )
    # 去重（粗）
    uniq = []
    for t in scored:
        R = t[-1]
        if any(float(np.linalg.norm(R - u[-1])) < 1e-3 for u in uniq):
            continue
        uniq.append(t)
        if len(uniq) >= 16:
            break
    return uniq


def _ec_ik_desk_pose(robot, xy, z, R_cands, seeds, jaw_close, pos_tol=0.07):
    """桌面缓冲：多姿态 × 多种子求 IK。"""
    seed_list = [np.asarray(s, dtype=float) for s in seeds if s is not None]
    for R_try in R_cands:
        for seed in seed_list:
            q = _ec_ik_slot_rot(
                robot,
                xy,
                z,
                R_try,
                seed,
                jaw_close,
                pos_tol=pos_tol,
                prefer_style=False,
                desk_relax=True,
            )
            if q is not None:
                return q, np.asarray(R_try, dtype=float)
    return None, None


def ec_build_fallen_ref_place(
    robot,
    q_start,
    seat_xy,
    slot_id: int,
    jaw_close,
    jaw_open,
    desk_obstacle,
    grasp=None,
    target=None,
    place_yaw: float = 0.0,
):
    """fallen 固定 place：以 EC_PLACE_REF_ARM 为松爪构型，只搜 J1/微量 J2 对准格心。

    持物 vs 入格：一致则 J6=REF；反着则转到对齐 J6，并在高位腕转过去。
    J6 改变后必须重搜 J1/J2/J3，否则螺丝中心会偏到邻格。
    """
    _ = slot_id
    # 对准格心（非 lean 后的 seat XY），避免目标本身偏半格
    try:
        slot_xy = slot_world_xy(int(slot_id))
    except Exception:
        slot_xy = np.asarray(seat_xy, dtype=float).reshape(-1)[:2]
    q_start = np.asarray(q_start, dtype=float).copy()
    q_start0 = q_start.copy()
    tip_stop = float(EC_BOX_RIM_Z + EC_RELEASE_TIP_ABOVE)
    tip_min = float(EC_BOX_RIM_Z + 1e-3)
    # 成功样例 tip≈0.016；关节搜可能略高，验收略宽于箱内通用值
    tip_max = float(EC_BOX_RIM_Z + max(EC_RELEASE_TIP_MAX_ABOVE, 0.022))
    tip_accept_soft = max(float(tip_max), float(EC_BOX_RIM_Z + 0.028))
    j1_pref = float(np.arctan2(float(slot_xy[1]), float(slot_xy[0]))) + float(
        EC_J1_CW_BIAS
    )
    # 对准阈值：松爪前优先让螺丝中心落在格心；只保留少量 IK 抖动余量。
    _pitch = float(min(SLOT_PITCH_X, SLOT_PITCH_Y))
    xy_err_aim = min(float(EC_RELEASE_OBJ_XY_TOL), 0.55 * _pitch)
    xy_err_max = max(float(EC_RELEASE_OBJ_XY_TOL), 0.75 * _pitch)
    xy_err_soft = max(float(xy_err_max), min(0.008, 1.05 * _pitch))

    def _fallen_xy_err(q):
        """松爪应对准「螺丝中心→格心」，不是 TCP 中心（否则夹爪偏一格）。"""
        robot.goto_given_conf(q, ee_values=float(jaw_close))
        if grasp is not None:
            obj_xy = _ec_obj_xy_from_tcp(robot, grasp)
            return float(np.linalg.norm(obj_xy - slot_xy))
        return float(
            np.linalg.norm(np.asarray(robot.gl_tcp_pos[:2], dtype=float) - slot_xy)
        )

    def _search_j123(
        j6_fixed,
        err_cap,
        require_tip_band: bool = False,
        require_j6_band: bool = True,
    ):
        """钉死 J4–J6，搜 J1/J2/J3 使螺丝中心对准格心。

        排序：tip 落带优先，再最小 XY 误差（避免为对准把 tip 抬出箱口）。
        require_j6_band=False：允许对齐用的 ±180° 腕角（仿真 J6 全行程）。
        """
        best_q = None
        best_key = None
        j6_fixed = float(j6_fixed)
        for dj1_deg in np.linspace(-45.0, 45.0, 61):
            for dj2_deg in (
                0.0, 4.0, 7.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0,
                -4.0, -8.0, -12.0,
            ):
                for dj3_deg in (
                    0.0, 6.0, -6.0, 12.0, -10.0, 16.0, -14.0, 18.0
                ):
                    q = EC_PLACE_REF_ARM.copy()
                    q[0] = j1_pref + float(np.deg2rad(dj1_deg))
                    q[1] = float(EC_PLACE_REF_ARM[1] + np.deg2rad(dj2_deg))
                    q[2] = float(EC_PLACE_REF_ARM[2] + np.deg2rad(dj3_deg))
                    q[5] = j6_fixed
                    if not _ec_conf_place_ok(
                        robot, q, jaw_close, require_j6=bool(require_j6_band)
                    ):
                        continue
                    tip_z = side.finger_tips_z_min(robot.end_effector)
                    if not np.isfinite(tip_z) or float(tip_z) < tip_min - 0.002:
                        continue
                    in_band = tip_min <= float(tip_z) <= tip_accept_soft
                    if require_tip_band and (not in_band):
                        continue
                    err = _fallen_xy_err(q)
                    if err > float(err_cap):
                        continue
                    tip_pen = abs(float(tip_z) - tip_stop)
                    arm_err = abs(float(q[1] - EC_PLACE_REF_ARM[1])) + abs(
                        float(q[2] - EC_PLACE_REF_ARM[2])
                    )
                    key = (
                        0 if in_band else 1,
                        err,
                        tip_pen,
                        arm_err,
                        abs(dj1_deg),
                    )
                    if best_q is None or key < best_key:
                        best_q, best_key = q.copy(), key
        return best_q

    def _force_tip_band(q_in, j6_fixed, err_allow, require_j6_band: bool = True):
        """tip 过高时优先折 J2/J3 压进验收带（允许 XY 略放宽）。"""
        q0 = np.asarray(q_in, dtype=float).copy()
        q0[5] = float(j6_fixed)
        robot.goto_given_conf(q0, ee_values=float(jaw_close))
        tip0 = float(side.finger_tips_z_min(robot.end_effector))
        if tip_min <= tip0 <= tip_accept_soft:
            return q0
        best = None
        best_key = None
        for dj2 in np.linspace(0.0, 36.0, 37):
            for dj3 in (0.0, 6.0, 12.0, -6.0, 16.0):
                q = q0.copy()
                q[1] = float(q0[1] + np.deg2rad(dj2))
                q[2] = float(EC_PLACE_REF_ARM[2] + np.deg2rad(dj3))
                if not _ec_conf_place_ok(
                    robot, q, jaw_close, require_j6=bool(require_j6_band)
                ):
                    continue
                tz = float(side.finger_tips_z_min(robot.end_effector))
                if not (tip_min <= tz <= tip_accept_soft):
                    continue
                err = _fallen_xy_err(q)
                if err > float(err_allow):
                    continue
                key = (err, abs(dj2), abs(dj3))
                if best is None or key < best_key:
                    best, best_key = q.copy(), key
        return best if best is not None else q0

    def _fine_j1(q_in, j6_fixed, err_cap, require_j6_band: bool = True):
        """在当前臂型附近只拧 J1，再压 XY 误差（保持 tip 落带）。"""
        q0 = np.asarray(q_in, dtype=float).copy()
        q0[5] = float(j6_fixed)
        best = q0
        best_err = _fallen_xy_err(q0)
        for ddeg in np.linspace(-12.0, 12.0, 49):
            q = q0.copy()
            q[0] = float(q0[0] + np.deg2rad(ddeg))
            if not _ec_conf_place_ok(
                robot, q, jaw_close, require_j6=bool(require_j6_band)
            ):
                continue
            tip_z = side.finger_tips_z_min(robot.end_effector)
            if not (
                np.isfinite(tip_z)
                and tip_min <= float(tip_z) <= tip_accept_soft
            ):
                continue
            err = _fallen_xy_err(q)
            if err > float(err_cap):
                continue
            if err + 1e-5 < best_err:
                best, best_err = q.copy(), err
        return best

    def _aim_with_j6(j6_fixed, require_j6_band: bool = None):
        """按给定 J6 找松爪臂型：tip 落带硬约束，再尽量对准格心。

        J6 在放置安全带外（如 ±180° 对齐）时自动关闭 band 硬约束。
        """
        j6_fixed = float(j6_fixed)
        if require_j6_band is None:
            require_j6_band = bool(
                EC_PLACE_J6_MIN <= j6_fixed <= EC_PLACE_J6_MAX
            )
        # tip 落带优先，误差逐步放宽；仍限制在格心附近，避免看起来偏到格边。
        q_b = None
        for cap in (
            xy_err_aim,
            xy_err_max,
            xy_err_soft,
            0.010,
        ):
            q_b = _search_j123(
                j6_fixed,
                cap,
                require_tip_band=True,
                require_j6_band=bool(require_j6_band),
            )
            if q_b is not None:
                break
        if q_b is None:
            # 无落带解：先取较准的高 tip，再强压 tip
            q_b = _search_j123(
                j6_fixed,
                    0.010,
                require_tip_band=False,
                require_j6_band=bool(require_j6_band),
            )
            if q_b is None:
                q_b = _search_j123(
                    j6_fixed,
                    0.012,
                    require_tip_band=False,
                    require_j6_band=bool(require_j6_band),
                )
            if q_b is None:
                return None
            q_b = _force_tip_band(
                q_b,
                j6_fixed,
                err_allow=max(float(xy_err_soft), 0.010),
                require_j6_band=bool(require_j6_band),
            )
        else:
            # 已在落带：勿为「更准」把 tip 抬出带；只细调 J1
            pass
        robot.goto_given_conf(q_b, ee_values=float(jaw_close))
        tip_now = float(side.finger_tips_z_min(robot.end_effector))
        if not (tip_min <= tip_now <= tip_accept_soft):
            q_b = _force_tip_band(
                q_b,
                j6_fixed,
                err_allow=0.025,
                require_j6_band=bool(require_j6_band),
            )
            robot.goto_given_conf(q_b, ee_values=float(jaw_close))
            tip_now = float(side.finger_tips_z_min(robot.end_effector))
            if not (tip_min <= tip_now <= tip_accept_soft):
                return None
        return _fine_j1(
            q_b,
            j6_fixed,
            err_cap=0.025,
            require_j6_band=bool(require_j6_band),
        )

    # fallen place：J1–J5/tip = 原 REF 解；若松爪持物相对入格差≈180°（翻面），
    # 则 J6 = J6_fallen + 翻面角（±180°），不是停在绝对 ±180°、也不是扳回 -49.5°。
    best = _aim_with_j6(EC_PLACE_J6_REF, require_j6_band=True)
    if best is None:
        print("[ec-place] fallen REF 关节搜索无解（J1/J2/J3）", flush=True)
        return None
    q_rel = best
    j6_use = float(EC_PLACE_J6_REF)
    j6_flip_mot = None
    matched_seat_rot = None
    need_place_reorient = False

    if grasp is not None and target is not None:
        tr0, ax0, R_h0, R_s0 = _ec_hold_seat_metrics(
            robot,
            grasp,
            target,
            float(place_yaw or 0.0),
            q_rel,
            jaw_close,
        )
        matched_seat_rot = np.asarray(R_s0, dtype=float).copy()
        # ① 松爪前 vs 入格：够像 → 原 fallen 解，绝不翻面（修 id=2 多此一举）
        if _ec_hold_similar_to_seat(tr0, ax0):
            print(
                f"[j6] 松爪≈入格 tr={tr0:.2f} axis={ax0:.2f} "
                f"→ 直接 fallen REF J6={np.rad2deg(j6_use):.1f}°（不翻面）",
                flush=True,
            )
        elif not _ec_hold_about_180_from_seat(tr0, ax0):
            # 既非相似也非约 180°：仍走原 fallen，勿乱转
            print(
                f"[j6] 非约180°偏差 tr={tr0:.2f} axis={ax0:.2f} "
                f"→ 保持 fallen REF J6={np.rad2deg(j6_use):.1f}°",
                flush=True,
            )
        else:
            # ② 约差 180°：试 J6=fallen±180°，且必须比原解明显更好才采用
            j6_lo, j6_hi = _ec_j6_motion_range(robot)
            j6_fallen = float(EC_PLACE_J6_REF)
            j6_s = float(q_start0[5])
            print(
                f"[j6] 松爪相对入格约差180° tr={tr0:.2f} axis={ax0:.2f} "
                f"→ 试翻面 J6=fallen({np.rad2deg(j6_fallen):.1f}°)±180°",
                flush=True,
            )
            place_best = None
            j6_flip_tgts = []
            for sign in (1.0, -1.0):
                d_flip = float(sign * np.pi)
                j6_ideal = float(j6_fallen + d_flip)
                for wrap in (0.0, 2.0 * np.pi, -2.0 * np.pi):
                    j6_tgt = float(j6_ideal + wrap)
                    if j6_lo - 1e-6 <= j6_tgt <= j6_hi + 1e-6:
                        j6_flip_tgts.append((j6_tgt, d_flip))
                        break
            uniq = []
            for j6_tgt, d_flip in j6_flip_tgts:
                if all(abs(j6_tgt - u[0]) > np.deg2rad(1.0) for u in uniq):
                    uniq.append((j6_tgt, d_flip))
            for j6_tgt, d_flip in uniq:
                # |ΔJ6| 应接近 180°（ang∈[0,π]，完美翻面 ang≈π）
                ang = abs(
                    ((j6_tgt - j6_fallen + np.pi) % (2 * np.pi)) - np.pi
                )
                if abs(float(ang) - np.pi) > np.deg2rad(25.0):
                    continue
                q_aim = _aim_with_j6(j6_tgt, require_j6_band=False)
                if q_aim is None:
                    q_try = best.copy()
                    q_try[5] = j6_tgt
                    if not _ec_conf_place_ok(
                        robot, q_try, jaw_close, require_j6=False
                    ):
                        continue
                    q_aim = _fine_j1(
                        q_try, j6_tgt, err_cap=0.025, require_j6_band=False
                    )
                tr_a, ax_a, R_ha, R_sa = _ec_hold_seat_metrics(
                    robot,
                    grasp,
                    target,
                    float(place_yaw or 0.0),
                    q_aim,
                    jaw_close,
                )
                # 腕转段可选：失败时仍保留翻面松爪解（hover/via 会拧到该 J6）
                mot_f = None
                if abs(j6_tgt - j6_s) >= np.deg2rad(3.0):
                    mot_f = ec_build_inhand_j6_to(
                        robot,
                        q_start0,
                        j6_tgt,
                        jaw_close,
                        desk_obstacle,
                        n_steps=None,
                        quiet=True,
                    )
                arm_d = float(
                    np.linalg.norm(q_aim[1:5] - EC_PLACE_REF_ARM[1:5])
                )
                key = (
                    0 if _ec_hold_matches_seat(tr_a, ax_a) else 1,
                    0 if mot_f is not None else 1,
                    -float(tr_a),
                    -float(ax_a),
                    arm_d,
                )
                if place_best is None or key < place_best[0]:
                    place_best = (
                        key,
                        j6_tgt,
                        d_flip,
                        q_aim.copy(),
                        mot_f,
                        float(tr_a),
                        float(ax_a),
                        R_sa,
                    )
            # 翻面须明显优于原 fallen 解，否则多此一举（id=2）
            use_flip = False
            if place_best is not None:
                _, j6_tgt, d_flip, q_aim, mot_f, tr_a, ax_a, R_sa = place_best
                if float(tr_a) >= float(tr0) + 0.35 or (
                    _ec_hold_similar_to_seat(tr_a, ax_a)
                    and not _ec_hold_similar_to_seat(tr0, ax0)
                ):
                    use_flip = True
            if use_flip:
                j6_flip_mot = mot_f
                if j6_flip_mot is not None:
                    q_start = np.asarray(
                        j6_flip_mot.jv_list[-1], dtype=float
                    ).copy()
                q_rel = q_aim
                j6_use = float(j6_tgt)
                matched_seat_rot = np.asarray(R_sa, dtype=float).copy()
                print(
                    f"[j6] 采用翻面 J6 = "
                    f"{np.rad2deg(j6_fallen):.1f}°{np.rad2deg(d_flip):+.1f}° "
                    f"= {np.rad2deg(j6_tgt):.1f}° "
                    f"tr {tr0:.2f}→{tr_a:.2f}",
                    flush=True,
                )
            else:
                tr_b = (
                    float(place_best[5]) if place_best is not None else float("nan")
                )
                print(
                    f"[j6] 翻面无收益（best_tr={tr_b:.2f} "
                    f"vs fallen_tr={tr0:.2f}）→ 仍用 fallen REF",
                    flush=True,
                )
                q_rel = best.copy()
                j6_use = float(EC_PLACE_J6_REF)
                matched_seat_rot = np.asarray(R_s0, dtype=float).copy()

        # tip 落带；翻面 J6 不要求落在 [-95,-25] 安全带
        j6_in_band = bool(EC_PLACE_J6_MIN <= float(j6_use) <= EC_PLACE_J6_MAX)
        robot.goto_given_conf(q_rel, ee_values=float(jaw_close))
        tip_chk = side.finger_tips_z_min(robot.end_effector)
        tip_bad = (not np.isfinite(tip_chk)) or (
            float(tip_chk) < float(tip_min)
            or float(tip_chk) > float(tip_accept_soft)
        )
        if tip_bad or (
            not _ec_conf_place_ok(
                robot, q_rel, jaw_close, require_j6=bool(j6_in_band)
            )
        ):
            q_fb = _aim_with_j6(
                float(j6_use), require_j6_band=bool(j6_in_band)
            )
            if q_fb is None:
                print("[ec-place] fallen(+翻面) tip 落带失败", flush=True)
                return None
            q_rel = q_fb
            j6_use = float(q_rel[5])
        tr_f, ax_f, R_hf, R_sf = _ec_hold_seat_metrics(
            robot,
            grasp,
            target,
            float(place_yaw or 0.0),
            q_rel,
            jaw_close,
        )
        matched_seat_rot = np.asarray(R_sf, dtype=float).copy()
        print(
            f"[j6] 入格解 "
            f"j={np.round(np.rad2deg(q_rel),1).tolist()} "
            f"tr={tr_f:.2f} axis={ax_f:.2f} "
            f"up={float((R_hf @ np.array([0.0,1.0,0.0]))[2]):.2f} "
            f"flipJ6={not j6_in_band}",
            flush=True,
        )

    # 同 J6（fallen 或 fallen+翻面）下再细瞄格心
    err_now = _fallen_xy_err(q_rel)
    if err_now > float(xy_err_max):
        j6_in_band = bool(EC_PLACE_J6_MIN <= float(q_rel[5]) <= EC_PLACE_J6_MAX)
        q_fine = _aim_with_j6(
            float(q_rel[5]), require_j6_band=bool(j6_in_band)
        )
        if q_fine is not None and _fallen_xy_err(q_fine) + 1e-4 < err_now:
            print(
                f"[ec-place] 细瞄格心 err {err_now*1e3:.1f}→"
                f"{_fallen_xy_err(q_fine)*1e3:.1f}mm",
                flush=True,
            )
            q_rel = q_fine
            j6_use = float(q_rel[5])
            if grasp is not None and target is not None:
                _, _, _, R_sf2 = _ec_hold_seat_metrics(
                    robot,
                    grasp,
                    target,
                    float(place_yaw or 0.0),
                    q_rel,
                    jaw_close,
                )
                matched_seat_rot = np.asarray(R_sf2, dtype=float).copy()

    robot.goto_given_conf(q_rel, ee_values=float(jaw_close))
    R = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
    # hover/下探 TCP：使物体中心落在格心（含 grasp 偏置）
    xy_tcp = _ec_place_tcp_xy(slot_xy, grasp, R)
    z_rel = float(robot.gl_tcp_pos[2])
    tip_rel = side.finger_tips_z_min(robot.end_effector)
    xy_err = _fallen_xy_err(q_rel)
    print(
        f"[ec-place] fallen REF 松爪候选 "
        f"j={np.round(np.rad2deg(q_rel),1).tolist()} "
        f"tip_z={float(tip_rel):.4f} obj_xy_err={xy_err:.4f}",
        flush=True,
    )

    require_j6_hov = bool(_ec_j6_place_ok(q_rel))
    # hover：同腕型抬高，便于从 pick 过渡
    q_hov = None
    for hz in (
        max(z_rel + 0.05, 0.12),
        0.14,
        0.16,
        0.11,
        max(z_rel + 0.03, 0.09),
    ):
        q_try = _ec_ik_slot_rot(
            robot,
            xy_tcp,
            hz,
            R,
            q_rel,
            jaw_close,
            pos_tol=0.05,
            prefer_style=True,
        )
        if q_try is None:
            continue
        q_try = np.asarray(q_try, dtype=float).copy()
        q_try[3:6] = q_rel[3:6]  # 钉死拧腕（含对齐后的 J6）
        q_try[5] = float(j6_use)
        if not _ec_conf_place_ok(
            robot, q_try, jaw_close, require_j6=require_j6_hov
        ):
            continue
        q_hov = q_try
        break
    if q_hov is None:
        q_hov = q_rel.copy()
        q_hov[1] = float(q_rel[1] - np.deg2rad(10.0))
        if not _ec_conf_place_ok(
            robot, q_hov, jaw_close, require_j6=require_j6_hov
        ):
            q_hov = q_rel.copy()

    via_mot = side._jv_lerp_motion(robot, q_start, q_hov, jaw_close, n_steps=18)
    if side.motion_penetrates_desk(
        robot, via_mot, desk_obstacle, jaw_fallback=jaw_close
    ) or (not _ec_motion_place_ok(robot, via_mot, jaw_close)):
        # 高位中转再下到 hover
        q_hi = q_hov.copy()
        q_hi[1] = float(min(q_hov[1], EC_PLACE_REF_ARM[1]) - np.deg2rad(15.0))
        if _ec_conf_place_ok(
            robot, q_hi, jaw_close, require_j6=require_j6_hov
        ):
            via_mot = (
                side._jv_lerp_motion(robot, q_start, q_hi, jaw_close, n_steps=12)
                + side._jv_lerp_motion(robot, q_hi, q_hov, jaw_close, n_steps=12)
            )
        if side.motion_penetrates_desk(
            robot, via_mot, desk_obstacle, jaw_fallback=jaw_close
        ) or (not _ec_motion_place_ok(robot, via_mot, jaw_close)):
            print("[ec-place] fallen REF via 穿桌/折腕，失败", flush=True)
            return None

    desc = side._jv_lerp_motion(robot, q_hov, q_rel, jaw_close, n_steps=14)
    if not _ec_motion_place_ok(robot, desc, jaw_close, sample_step=1):
        print("[ec-place] fallen REF 下探非法，失败", flush=True)
        return None
    robot.goto_given_conf(q_rel, ee_values=float(jaw_close))
    tip_z = side.finger_tips_z_min(robot.end_effector)
    if not np.isfinite(tip_z) or float(tip_z) < tip_min or float(tip_z) > tip_accept_soft:
        print(
            f"[ec-place] fallen REF tip 不合格 tip_z={float(tip_z):.4f} "
            f"(要∈[{tip_min:.3f},{tip_accept_soft:.3f}])",
            flush=True,
        )
        return None
    obj_err = _fallen_xy_err(q_rel)
    # tip 已合格时也尽量要求格心附近，避免松爪前明显靠格边。
    xy_accept = max(float(xy_err_soft), 0.010)
    if obj_err > float(xy_accept):
        print(
            f"[ec-place] fallen REF 螺丝未对准格心 err={obj_err:.4f} "
            f"(要≤{xy_accept:.4f})，丢弃",
            flush=True,
        )
        return None
    print(
        f"[ec-place] OK release j={np.round(np.rad2deg(q_rel),1).tolist()} "
        f"tip_z={float(tip_z):.4f} obj_xy_err={obj_err:.4f} upright=REF",
        flush=True,
    )
    # 松爪多停几帧，动画里能看清「夹爪对着格子」再回原位
    desc.extend(
        [q_rel.copy(), q_rel.copy(), q_rel.copy()],
        ev_list=[float(jaw_open), float(jaw_open), float(jaw_open)],
        mesh_list=[],
    )
    place_mot = via_mot + desc
    if j6_flip_mot is not None:
        place_mot = j6_flip_mot + place_mot
    # 入格 rot：与持物最近的 upright（动画松爪用；搬运中零件始终随爪）
    if matched_seat_rot is not None:
        try:
            place_mot.ec_matched_seat_rot = np.asarray(
                matched_seat_rot, dtype=float
            ).copy()
        except Exception:
            pass
    try:
        place_mot.ec_need_place_reorient = False
    except Exception:
        pass
    return place_mot


def ec_build_normal_vertical_place(
    robot,
    q_start,
    seat_xy,
    slot_id: int,
    jaw_close,
    jaw_open,
    desk_obstacle,
    grasp=None,
):
    """normal 竖直顶放：接近轴 −Z，J6 不锁拧腕；关节搜 + 下探。"""
    _ = slot_id
    slot_xy = np.asarray(seat_xy, dtype=float).reshape(-1)[:2]
    q_start = np.asarray(q_start, dtype=float).copy()
    tip_stop = float(EC_BOX_RIM_Z + EC_RELEASE_TIP_ABOVE)
    tip_min = float(EC_BOX_RIM_Z + 1e-3)
    tip_max = float(EC_BOX_RIM_Z + EC_RELEASE_TIP_MAX_ABOVE + 0.010)
    j1_pref = float(np.arctan2(float(slot_xy[1]), float(slot_xy[0]))) + float(
        EC_J1_CW_BIAS
    )
    # 顶放 TCP：+z 接近朝下
    R = np.eye(3)
    R[:, 2] = np.array([0.0, 0.0, -1.0], dtype=float)
    R[:, 0] = np.array([1.0, 0.0, 0.0], dtype=float)
    R[:, 1] = np.cross(R[:, 2], R[:, 0])

    best = None
    best_key = None
    base = np.asarray(EC_PLACE_NATURAL_CONF, dtype=float).copy()
    for yaw in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
        Rz = rm.rotmat_from_euler(0.0, 0.0, float(yaw))
        R_try = Rz @ R
        xy_goal = _ec_place_tcp_xy(slot_xy, grasp, R_try)
        for hz in (0.10, 0.12, 0.08, 0.14, 0.06):
            for dj1 in (0.0, -0.2, 0.2, -0.35, 0.35):
                seed = base.copy()
                seed[0] = j1_pref + float(dj1)
                q = _ec_ik_slot_rot(
                    robot,
                    xy_goal,
                    hz,
                    R_try,
                    seed,
                    jaw_close,
                    pos_tol=0.05,
                    prefer_style=False,
                    desk_relax=True,
                )
                if q is None:
                    continue
                if not _ec_conf_place_ok(robot, q, jaw_close, require_j6=False):
                    continue
                robot.goto_given_conf(q, ee_values=float(jaw_close))
                app = -float(np.asarray(robot.gl_tcp_rotmat, dtype=float)[2, 2])
                if app < 0.80:
                    continue
                if grasp is not None:
                    err = float(
                        np.linalg.norm(_ec_obj_xy_from_tcp(robot, grasp) - slot_xy)
                    )
                else:
                    err = float(
                        np.linalg.norm(
                            np.asarray(robot.gl_tcp_pos[:2], dtype=float) - slot_xy
                        )
                    )
                if err > 0.014:
                    continue
                tip_z = side.finger_tips_z_min(robot.end_effector)
                key = (err, -app, abs(float(tip_z) - tip_stop) if np.isfinite(tip_z) else 1.0)
                if best is None or key < best_key:
                    best, best_key = (q.copy(), R_try.copy(), float(hz)), key
    if best is None:
        print("[ec-place] normal 竖直 hover IK 失败", flush=True)
        return None
    q_hov, R_use, hover_z = best
    # 复用通用下探（竖直、不强制 J6）
    return ec_build_place_neg_z(
        robot,
        q_start=q_start,
        seat_xy=slot_xy,
        slot_id=int(slot_id),
        jaw_close=jaw_close,
        jaw_open=jaw_open,
        desk_obstacle=desk_obstacle,
        tcp_rot=R_use,
        hover_z=hover_z,
        release_tcp_floor=float(EC_RELEASE_TCP_FLOOR),
        grasp=grasp,
        approach_tip_deg=0.0,
        place_surface="box",
        require_upright=False,
        force_ref_arm=False,
        _skip_normal_shortcut=True,
    )


def ec_build_place_neg_z(
    robot,
    q_start,
    seat_xy,
    slot_id: int,
    jaw_close,
    jaw_open,
    desk_obstacle,
    tcp_rot,
    hover_z=None,
    release_tcp_floor=None,
    ee_align_x=None,
    grasp=None,
    approach_tip_deg=None,
    place_surface: str = "box",
    require_upright: bool = True,
    force_ref_arm: bool = False,
    _skip_normal_shortcut: bool = False,
    target=None,
    place_yaw: float = 0.0,
):
    """放置：低头入格姿态 → J1 先顺时针对准 → 格心上方 → 单调 −Z → 松爪。

    force_ref_arm：fallen 用；J6 按持物 vs 入格一致/反着选择。
    """
    # fallen：直接走固定臂型，避免 grasp→TCP IK 把 J6 拧飞
    if force_ref_arm and str(place_surface or "box").lower() != "desk":
        return ec_build_fallen_ref_place(
            robot,
            q_start=q_start,
            seat_xy=seat_xy,
            slot_id=int(slot_id),
            jaw_close=jaw_close,
            jaw_open=jaw_open,
            desk_obstacle=desk_obstacle,
            grasp=grasp,
            target=target,
            place_yaw=float(place_yaw or 0.0),
        )
    hover_z = float(EC_HIGH_HOVER_Z if hover_z is None else hover_z)
    on_desk = str(place_surface or "box").lower() == "desk"
    if on_desk:
        # 缓冲落地后要 remesh：允许松爪略高，优先能放下
        z_tcp_floor = float(
            0.028 if release_tcp_floor is None else release_tcp_floor
        )
        tip_stop = 0.006
        tip_min = 0.0005
        tip_max = 0.055
        xy_tol = 0.020
        upright_min = 0.0 if not require_upright else float(EC_RELEASE_UPRIGHT_MIN)
    else:
        z_tcp_floor = float(
            EC_RELEASE_TCP_FLOOR if release_tcp_floor is None else release_tcp_floor
        )
        tip_stop = float(EC_BOX_RIM_Z + EC_RELEASE_TIP_ABOVE)
        tip_min = float(EC_BOX_RIM_Z + 1e-3)
        tip_max = float(EC_BOX_RIM_Z + EC_RELEASE_TIP_MAX_ABOVE)
        xy_tol = float(EC_RELEASE_OBJ_XY_TOL)
        upright_min = float(EC_RELEASE_UPRIGHT_MIN)
    # 对齐目标：格子中心（不用 lean）；TCP 由「螺丝在格心」反算
    slot_xy = np.asarray(seat_xy, dtype=float).reshape(-1)[:2]
    R = np.asarray(tcp_rot, dtype=float)
    q_start = np.asarray(q_start, dtype=float).copy()
    if on_desk:
        # 缓冲：螺丝中心对准缓冲 XY（勿只对 TCP，否则松爪与落点差一截）
        xy = _ec_place_tcp_xy(slot_xy, grasp, R)
        q_seed = q_start.copy()
        q_seed[0] = float(np.arctan2(float(slot_xy[1]), float(slot_xy[0])))
    else:
        xy = _ec_place_tcp_xy(slot_xy, grasp, R)
        q_seed = ec_place_base_conf_for_slot(
            xy, slot_id=int(slot_id), force_ref=bool(force_ref_arm)
        )
        if force_ref_arm:
            # 仅 J1 略跟 pick，J2–J6 完全用参考解
            q_seed[0] = 0.25 * float(q_start[0]) + 0.75 * float(q_seed[0])
        else:
            q_seed[0] = 0.35 * float(q_start[0]) + 0.65 * float(q_seed[0])

    tip_d = float(approach_tip_deg) if approach_tip_deg is not None else -1.0
    off_xy = float(np.linalg.norm(xy - slot_xy))
    _ = ee_align_x
    _ec_log(
        f"[ec-place] slot={slot_xy.tolist()} tcp_xy={xy.tolist()} "
        f"|obj-tcp|_xy={off_xy:.3f} tip≈{tip_d:.0f}° surface={place_surface}"
        f"{' REF-ARM' if force_ref_arm else ''}"
    )

    # 1) 高位 hover（桌面：先求低位松爪，再在其上 hover）
    q_hov = None
    q_desk_rel = None
    if on_desk:
        # inverted→缓冲：侧倾优先；require_upright 桌面清障仍可顶放
        desk_tilt = (not require_upright) and (
            approach_tip_deg is None or abs(float(approach_tip_deg)) >= 5.0
        )
        tip_for_desk = None
        if desk_tilt:
            tip_for_desk = (
                (float(approach_tip_deg),)
                if approach_tip_deg is not None
                else (45.0, 40.0, 50.0, 35.0)
            )
        R_desk = _ec_desk_rot_cands(
            R, prefer_tilt=bool(desk_tilt), tip_degs=tip_for_desk
        )
        # 确保传入的侧倾 TCP 排在最前；竖直顶放（app↓≈1）垫后
        if desk_tilt:
            R0 = np.asarray(R, dtype=float)

            def _tip_down(Rm):
                return max(0.0, -float(np.asarray(Rm, dtype=float)[2, 2]))

            rest = [
                x for x in R_desk if float(np.linalg.norm(np.asarray(x) - R0)) > 1e-3
            ]
            tilt_first = [x for x in rest if _tip_down(x) <= 0.96]
            vert_last = [x for x in rest if _tip_down(x) > 0.96]
            R_desk = [R0] + tilt_first + vert_last
        desk_seeds = (q_seed, q_start)
        # 优先直接解出可验收松爪高度；失败再退到高位 hover+步进
        for z_try in (0.055, 0.048, 0.042, 0.038, 0.065, 0.075, 0.090):
            for tol in (0.06, 0.08, 0.10):
                q_try, R_try = _ec_ik_desk_pose(
                    robot, xy, z_try, R_desk, desk_seeds, jaw_close, pos_tol=tol
                )
                if q_try is None:
                    continue
                robot.goto_given_conf(q_try, ee_values=float(jaw_close))
                tip_z = side.finger_tips_z_min(robot.end_effector)
                if grasp is not None:
                    xy_err = float(
                        np.linalg.norm(
                            _ec_obj_xy_from_tcp(robot, grasp) - slot_xy
                        )
                    )
                else:
                    xy_err = float(
                        np.linalg.norm(robot.gl_tcp_pos[:2] - slot_xy)
                    )
                if (
                    np.isfinite(tip_z)
                    and tip_min <= float(tip_z) <= tip_max
                    and xy_err <= xy_tol
                ):
                    q_desk_rel = np.asarray(q_try, dtype=float).copy()
                    R = np.asarray(R_try, dtype=float)
                    print(
                        f"[ec-place] desk 低位松爪 tcp_z={z_try:.3f} "
                        f"tip_z={float(tip_z):.4f} obj_xy_err={xy_err:.4f}",
                        flush=True,
                    )
                    break
            if q_desk_rel is not None:
                break
        hz_list = (0.12, 0.10, 0.14, 0.08, 0.16, 0.18)
        for hz in hz_list:
            q_hov, R_hov = _ec_ik_desk_pose(
                robot,
                xy,
                hz,
                [R] + R_desk if q_desk_rel is not None else R_desk,
                (q_desk_rel, q_seed, q_start)
                if q_desk_rel is not None
                else desk_seeds,
                jaw_close,
                pos_tol=0.08,
            )
            if q_hov is not None:
                hover_z = float(hz)
                if q_desk_rel is None:
                    R = np.asarray(R_hov, dtype=float)
                break
        if q_hov is None and q_desk_rel is not None:
            # 低位已有解：用略抬高的 IK 当 hover，否则直接从松爪起步
            q_up, _ = _ec_ik_desk_pose(
                robot, xy, 0.10, [R], (q_desk_rel,), jaw_close, pos_tol=0.10
            )
            if q_up is not None:
                q_hov = q_up
                hover_z = 0.10
            else:
                q_hov = q_desk_rel.copy()
                hover_z = 0.06
    else:
        hz_list = (hover_z, 0.16, 0.12) if not EC_VERBOSE else (
            hover_z, 0.20, 0.16, 0.22, 0.14, 0.12, 0.10, 0.08
        )
        pos_tols = (0.025, 0.03)
        for hz in hz_list:
            for tol in pos_tols:
                q_hov = _ec_ik_slot_rot(
                    robot,
                    xy,
                    hz,
                    R,
                    q_seed,
                    jaw_close,
                    pos_tol=tol,
                    prefer_style=True,
                )
                if q_hov is not None:
                    hover_z = float(hz)
                    break
                q_hov = _ec_ik_slot_rot(
                    robot,
                    xy,
                    hz,
                    R,
                    q_start,
                    jaw_close,
                    pos_tol=tol,
                    prefer_style=True,
                )
                if q_hov is not None:
                    hover_z = float(hz)
                    break
            if q_hov is not None:
                break
    if q_hov is None:
        print("[ec-place] 高位入格 IK 失败", flush=True)
        return None

    # fallen：贴参考臂型并强制 J6≈-49.5°；normal：只要求接近轴竖直
    ref = (
        EC_PLACE_REF_ARM.copy()
        if force_ref_arm
        else np.asarray(_PLACE_STYLE["q_arm"], dtype=float).copy()
    )
    if force_ref_arm:
        ref[5] = float(EC_PLACE_J6_REF)
    arm_err0 = float(np.linalg.norm(q_hov[1:6] - ref[1:6]))
    if not on_desk and force_ref_arm:
        # 固定 place：优先钉死 REF 臂型（J6≈-49.5°），不要接受乱腕 IK
        j1_slot = float(
            np.arctan2(float(xy[1]), float(xy[0])) + float(EC_J1_CW_BIAS)
        )
        recovered = False
        for hz_try in (hover_z, 0.16, 0.14, 0.12, 0.18, 0.10):
            for blend in (1.0, 0.9, 0.75):
                q_try = ref.copy()
                q_try[0] = j1_slot
                q_try[1:6] = (1.0 - blend) * q_hov[1:6] + blend * ref[1:6]
                q_try[5] = float(EC_PLACE_J6_REF)
                q_re = _ec_ik_slot_rot(
                    robot,
                    xy,
                    hz_try,
                    R,
                    q_try,
                    jaw_close,
                    pos_tol=0.045,
                    prefer_style=True,
                )
                if q_re is None or (not _ec_j6_place_ok(q_re)):
                    continue
                err_re = float(np.linalg.norm(q_re[1:6] - ref[1:6]))
                if err_re > np.deg2rad(55.0):
                    continue
                q_hov = q_re
                hover_z = float(hz_try)
                arm_err0 = err_re
                recovered = True
                break
            if recovered:
                break
        if not recovered:
            # 关节空间直接用 REF：FK 若 XY 够近则采用（固定 place 优先）
            q_ref = ec_place_base_conf_for_slot(xy, slot_id=int(slot_id), force_ref=True)
            q_ref[5] = float(EC_PLACE_J6_REF)
            if _ec_conf_place_ok(robot, q_ref, jaw_close):
                robot.goto_given_conf(q_ref, ee_values=float(jaw_close))
                xy_err = float(np.linalg.norm(robot.gl_tcp_pos[:2] - xy))
                if xy_err <= 0.028:
                    q_hov = q_ref
                    hover_z = float(robot.gl_tcp_pos[2])
                    # 后续下探锁 REF 真实 TCP 朝向，避免再被 grasp 反算 R 带偏 J6
                    R = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
                    arm_err0 = float(np.linalg.norm(q_hov[1:6] - ref[1:6]))
                    recovered = True
                    print(
                        f"[ec-place] fallen 用 REF 关节 hover "
                        f"xy_err={xy_err:.4f} j6={np.rad2deg(q_hov[5]):.1f}°",
                        flush=True,
                    )
        elif _ec_j6_place_ok(q_hov):
            robot.goto_given_conf(q_hov, ee_values=float(jaw_close))
            R = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
        if not _ec_j6_place_ok(q_hov):
            print(
                f"[ec-place] hover J6={np.rad2deg(q_hov[5]):.1f}° 未拧腕，丢弃",
                flush=True,
            )
            return None
        if arm_err0 > np.deg2rad(55.0):
            print(
                f"[ec-place] hover 偏离 fallen 参考 Δ={np.rad2deg(arm_err0):.1f}°，丢弃",
                flush=True,
            )
            return None

    robot.goto_given_conf(q_hov, ee_values=float(jaw_close))
    Rm = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    app_down = -float(Rm[2, 2])
    if (not on_desk) and (not force_ref_arm) and app_down < 0.85:
        print(
            f"[ec-place] normal 夹爪未竖直 app↓={app_down:.2f}<0.85，丢弃",
            flush=True,
        )
        return None
    _ec_log(
        f"[ec-place] hover j={np.round(np.rad2deg(q_hov),1).tolist()} "
        f"tcp_z={float(robot.gl_tcp_pos[2]):.3f} "
        f"app↓={app_down:.2f} armΔ={np.rad2deg(arm_err0):.1f}°"
        f"{' fallen-REF' if force_ref_arm else ' normal-vert'}"
    )

    # pick末 → hover：桌面缓冲走短路径；入格再腕调/J1
    ns = (8, 8, 10) if not EC_VERBOSE else (12, 12, 16)
    if on_desk:
        via_mot = side._jv_lerp_motion(
            robot, q_start, q_hov, jaw_close, n_steps=12
        )
        _ec_log(
            f"[ec-place] desk via: 直达 hover j={np.round(np.rad2deg(q_hov),1).tolist()}"
        )
    else:
        q_adj = q_start.copy()
        q_adj[3:6] = 0.2 * q_start[3:6] + 0.8 * q_hov[3:6]
        q_adj[1:3] = 0.55 * q_start[1:3] + 0.45 * q_hov[1:3]
        q_adj[0] = float(q_start[0])
        if not _ec_conf_place_ok(robot, q_adj, jaw_close):
            q_adj = q_start.copy()
            q_adj[4:6] = q_hov[4:6]
        q_j1 = q_adj.copy()
        j1_swing = float(q_hov[0]) + float(np.deg2rad(-14.0))
        q_j1[0] = float(np.clip(j1_swing, -2.3, 2.3))
        q_j1[1:6] = 0.35 * q_adj[1:6] + 0.65 * q_hov[1:6]
        if _ec_conf_place_ok(robot, q_adj, jaw_close) and _ec_conf_place_ok(
            robot, q_j1, jaw_close
        ):
            via_mot = (
                side._jv_lerp_motion(robot, q_start, q_adj, jaw_close, n_steps=ns[0])
                + side._jv_lerp_motion(robot, q_adj, q_j1, jaw_close, n_steps=ns[1])
                + side._jv_lerp_motion(robot, q_j1, q_hov, jaw_close, n_steps=ns[2])
            )
            _ec_log(
                f"[ec-place] via: 腕调竖直→J1→hover j5={np.rad2deg(q_hov[4]):.1f}°"
            )
        elif _ec_conf_place_ok(robot, q_j1, jaw_close):
            via_mot = side._jv_lerp_motion(
                robot, q_start, q_j1, jaw_close, n_steps=ns[1]
            ) + side._jv_lerp_motion(robot, q_j1, q_hov, jaw_close, n_steps=ns[2])
        else:
            q_j1 = q_adj.copy()
            q_j1[0] = float(q_hov[0])
            if _ec_conf_place_ok(robot, q_j1, jaw_close):
                via_mot = side._jv_lerp_motion(
                    robot, q_start, q_j1, jaw_close, n_steps=ns[1]
                ) + side._jv_lerp_motion(
                    robot, q_j1, q_hov, jaw_close, n_steps=ns[2]
                )
            else:
                via_mot = side._jv_lerp_motion(
                    robot, q_start, q_hov, jaw_close, n_steps=14
                )

    robot.goto_given_conf(q_start, ee_values=float(jaw_close))
    z0 = float(robot.gl_tcp_pos[2])
    zs = []
    for q in via_mot.jv_list:
        robot.goto_given_conf(q, ee_values=float(jaw_close))
        zs.append(float(robot.gl_tcp_pos[2]))
    if len(zs) >= 3 and min(zs[1:-1]) < min(z0, zs[-1]) - 0.025:
        q_hi = _ec_ik_slot_rot(
            robot,
            xy,
            max(hover_z + 0.05, z0 + 0.03),
            R,
            q_hov,
            jaw_close,
            pos_tol=0.035,
            prefer_style=not on_desk,
        )
        if q_hi is not None:
            q_j1b = q_start.copy()
            q_j1b[0] = float(q_hi[0])
            via_mot = (
                side._jv_lerp_motion(robot, q_start, q_j1b, jaw_close, n_steps=12)
                + side._jv_lerp_motion(robot, q_j1b, q_hi, jaw_close, n_steps=12)
                + side._jv_lerp_motion(robot, q_hi, q_hov, jaw_close, n_steps=10)
            )
            print("[ec-place] via 经高位中转，避免先下后上", flush=True)

    if side.motion_penetrates_desk(
        robot, via_mot, desk_obstacle, jaw_fallback=jaw_close
    ) or (not _ec_motion_place_ok(robot, via_mot, jaw_close)):
        print("[ec-place] via 穿桌/折腕/法兰过低，失败", flush=True)
        return None

    # 2) 下探：桌面用预求松爪关节插值；箱内锁定姿态单调 −Z
    robot.goto_given_conf(q_hov, ee_values=float(jaw_close))
    z_cur = float(robot.gl_tcp_pos[2])
    q_cur = np.asarray(q_hov, dtype=float).copy()
    jv_desc = [q_cur.copy()]

    if on_desk and q_desk_rel is not None:
        desc_mot = side._jv_lerp_motion(
            robot, q_hov, q_desk_rel, jaw_close, n_steps=14
        )
        z_prev = z_cur
        for q in desc_mot.jv_list[1:]:
            robot.goto_given_conf(q, ee_values=float(jaw_close))
            z_new = float(robot.gl_tcp_pos[2])
            tip_z = side.finger_tips_z_min(robot.end_effector)
            if not _ec_conf_place_ok(robot, q, jaw_close):
                print("[ec-place] desk 下探非法构型，停止", flush=True)
                break
            if np.isfinite(tip_z) and float(tip_z) < tip_min:
                print(
                    f"[ec-place] 指尖将低于桌面 tip_z={float(tip_z):.4f}",
                    flush=True,
                )
                break
            # 允许小幅起伏；明显上抬则停
            if z_new > z_prev + 0.008:
                break
            q_cur = np.asarray(q, dtype=float).copy()
            z_cur = z_new
            z_prev = z_new
            jv_desc.append(q_cur.copy())
        # 末帧尽量落到预求松爪
        if _ec_conf_place_ok(robot, q_desk_rel, jaw_close):
            robot.goto_given_conf(q_desk_rel, ee_values=float(jaw_close))
            tip_z = side.finger_tips_z_min(robot.end_effector)
            if np.isfinite(tip_z) and float(tip_z) >= tip_min:
                q_cur = np.asarray(q_desk_rel, dtype=float).copy()
                if len(jv_desc) == 0 or np.linalg.norm(jv_desc[-1] - q_cur) > 1e-6:
                    jv_desc.append(q_cur.copy())
    else:
        for _ in range(60):
            tip_z = side.finger_tips_z_min(robot.end_effector)
            if np.isfinite(tip_z) and float(tip_z) <= tip_stop + 1e-4:
                break
            if z_cur <= z_tcp_floor + 1e-4:
                break
            z_next = max(z_tcp_floor, z_cur - 0.006)
            q_new = _ec_ik_slot_rot(
                robot,
                xy,
                z_next,
                R,
                q_cur,
                jaw_close,
                pos_tol=0.04 if on_desk else 0.025,
                prefer_style=not on_desk,
                desk_relax=on_desk,
            )
            if q_new is None and on_desk:
                # 桌面逐步失败时换姿态再试
                q_new, R_new = _ec_ik_desk_pose(
                    robot,
                    xy,
                    z_next,
                    _ec_desk_rot_cands(R),
                    (q_cur, q_seed, q_start),
                    jaw_close,
                    pos_tol=0.08,
                )
                if q_new is not None:
                    R = R_new
            if q_new is None:
                print(f"[ec-place] 下探 IK 停 tcp_z={z_cur:.3f}", flush=True)
                break
            robot.goto_given_conf(q_new, ee_values=float(jaw_close))
            z_new = float(robot.gl_tcp_pos[2])
            if z_new > z_cur + 1e-4:
                print("[ec-place] 拒绝上抬步，停止下探", flush=True)
                break
            tip_z = side.finger_tips_z_min(robot.end_effector)
            if np.isfinite(tip_z) and float(tip_z) < tip_min:
                print(
                    f"[ec-place] 指尖将低于箱口 tip_z={float(tip_z):.4f}",
                    flush=True,
                )
                break
            if not _ec_conf_place_ok(robot, q_new, jaw_close):
                print("[ec-place] 自碰/折腕/法兰过低，停止下探", flush=True)
                break
            if float(np.linalg.norm(robot.gl_tcp_pos[:2] - xy)) > (
                0.018 if on_desk else 0.012
            ):
                print("[ec-place] 偏离格心，停止下探", flush=True)
                break
            q_cur = q_new
            z_cur = z_new
            jv_desc.append(q_cur.copy())
            if np.isfinite(tip_z) and float(tip_z) <= tip_stop + 1e-4:
                break

    robot.goto_given_conf(q_cur, ee_values=float(jaw_close))
    tip_z = side.finger_tips_z_min(robot.end_effector)
    xy_err = float(np.linalg.norm(robot.gl_tcp_pos[:2] - xy))
    Rm = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    desc = MotionData(robot)
    desc.extend(
        jv_desc,
        ev_list=[float(jaw_close)] * len(jv_desc),
        mesh_list=[],
    )
    if not _ec_motion_place_ok(robot, desc, jaw_close, sample_step=1):
        print("[ec-place] 下探轨迹折腕/穿桌，失败", flush=True)
        return None
    z_prev = None
    for q in jv_desc:
        robot.goto_given_conf(q, ee_values=float(jaw_close))
        zz = float(robot.gl_tcp_pos[2])
        if z_prev is not None and zz > z_prev + 1.5e-3:
            print("[ec-place] 轨迹存在上抬，失败", flush=True)
            return None
        z_prev = zz
    # 验收：对准目标 XY；箱内要求竖直；桌面缓冲可放宽
    if not np.isfinite(tip_z) or float(tip_z) > tip_max:
        print(
            f"[ec-place] 松爪过高 tip_z={float(tip_z):.4f}>{tip_max:.3f}，丢弃",
            flush=True,
        )
        return None
    if np.isfinite(tip_z) and float(tip_z) < tip_min:
        print(
            f"[ec-place] 松爪过低 tip_z={float(tip_z):.4f}<{tip_min:.3f}，丢弃",
            flush=True,
        )
        return None
    robot.goto_given_conf(q_cur, ee_values=float(jaw_close))
    Rm = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    if grasp is not None:
        obj_xy = _ec_obj_xy_from_tcp(robot, grasp, Rm)
        obj_err = float(np.linalg.norm(obj_xy - slot_xy))
        ac_r = np.asarray(grasp.ac_rotmat, dtype=float)
        R_obj = Rm @ ac_r.T
        upright_now = abs(float(R_obj[2, 1]))  # 长轴≈物系 +Y
    else:
        obj_xy = np.asarray(robot.gl_tcp_pos[:2], dtype=float)
        obj_err = float(np.linalg.norm(obj_xy - slot_xy))
        upright_now = abs(float(Rm[2, 2]))
    if obj_err > float(xy_tol):
        print(
            f"[ec-place] 未对准目标 obj_xy={np.round(obj_xy,4).tolist()} "
            f"tgt={slot_xy.tolist()} err={obj_err:.4f}，丢弃",
            flush=True,
        )
        return None
    if require_upright and upright_now < float(upright_min):
        print(
            f"[ec-place] 松爪姿态不够竖直 upright={upright_now:.3f}"
            f"<{upright_min:.2f}，丢弃",
            flush=True,
        )
        return None
    q_rel = np.asarray(jv_desc[-1], dtype=float)
    print(
        f"[ec-place] OK release j={np.round(np.rad2deg(q_rel),1).tolist()} "
        f"tip_z={float(tip_z):.4f} upright={upright_now:.3f}",
        flush=True,
    )
    if not _ec_motion_place_ok(robot, via_mot, jaw_close):
        print("[ec-place] via 折腕/穿桌抽检失败", flush=True)
        return None
    # fallen：松爪须拧 J6；normal：松爪须夹爪接近轴竖直
    if force_ref_arm:
        if not _ec_conf_place_ok(robot, q_rel, jaw_close, require_j6=True):
            print(
                f"[ec-place] fallen 松爪非法 j6={np.rad2deg(q_rel[5]):.1f}° "
                f"(要∈[{np.rad2deg(EC_PLACE_J6_MIN):.0f},{np.rad2deg(EC_PLACE_J6_MAX):.0f}])",
                flush=True,
            )
            return None
    else:
        if not _ec_conf_place_ok(robot, q_rel, jaw_close, require_j6=False):
            print(
                f"[ec-place] 松爪构型非法 j5={np.rad2deg(q_rel[4]):.1f}° "
                f"ee_z={float(robot.end_effector.pos[2]):.3f}",
                flush=True,
            )
            return None
        robot.goto_given_conf(q_rel, ee_values=float(jaw_close))
        app_f = -float(np.asarray(robot.gl_tcp_rotmat, dtype=float)[2, 2])
        if (not on_desk) and app_f < 0.85:
            print(
                f"[ec-place] normal 松爪夹爪未竖直 app↓={app_f:.2f}，丢弃",
                flush=True,
            )
            return None
    desc.extend([q_rel], ev_list=[float(jaw_open)], mesh_list=[])
    return via_mot + desc


# 规范入格姿态（与加策略前一致）：长轴竖直 Rx(+π/2)
CANONICAL_ADJUST_NAME = "rx+"


def upright_rot_options(target):
    """规划用调姿候选：规范 rx+ 永远优先；其它仅在规范失败时用于解 IK。

    注意：最终落入 box 的姿态始终是 CANONICAL（UPRIGHT_OBJ_ROT），与此处候选无关。
    """
    h = float(target.get("height") or 0.0)
    gw = float(target.get("grip_width") or 0.0)
    st = str(target.get("state") or "normal").lower()
    opts = [
        (CANONICAL_ADJUST_NAME, UPRIGHT_OBJ_ROT.copy()),
        ("rx-", rm.rotmat_from_euler(float(-np.pi / 2.0), 0.0, 0.0)),
        ("ry+", rm.rotmat_from_euler(0.0, float(np.pi / 2.0), 0.0)),
        ("ry-", rm.rotmat_from_euler(0.0, float(-np.pi / 2.0), 0.0)),
    ]
    if h >= 0.0085 and h >= 0.85 * max(gw, 1e-6):
        opts.append(("keep_tall", np.eye(3)))
    # 规范 rx+ 始终第一；其余按状态排
    if st == "fallen":
        order = [CANONICAL_ADJUST_NAME, "ry+", "ry-", "rx-", "keep_tall"]
    else:
        order = [CANONICAL_ADJUST_NAME, "rx-", "ry+", "ry-", "keep_tall"]
    by_name = {n: R for n, R in opts}
    out = []
    for n in order:
        if n in by_name and all(x[0] != n for x in out):
            out.append((n, by_name[n]))
    return out


def _estimate_release_tcp_z(robot, mot, release_idx, jaw_close):
    """松爪前一帧的 TCP 高度；失败则返回很大值。"""
    try:
        if mot is None or release_idx is None:
            return 1.0
        i = int(max(0, release_idx - 1))
        q = mot.jv_list[i]
        robot.goto_given_conf(q, ee_values=float(jaw_close))
        return float(robot.gl_tcp_pos[2])
    except Exception:
        return 1.0


def seat_rot_undo_bake(info: dict, place_yaw: float = 0.0) -> np.ndarray:
    """桌面 rpy 已烘焙进 mesh 顶点时，求使零件竖直入格的 model.rotmat。

    顶点 ≈ R_bake @ 规范网格；要世界姿态 = Rz(yaw)@Rx(π/2)，则
    rotmat = R_up @ inv(R_bake)。若仍用固定 Rx(π/2)，yaw≠0 的 fallen 会平躺跨格。
    """
    roll = float(info.get("bake_roll", info.get("roll", 0.0)) or 0.0)
    pitch = float(info.get("bake_pitch", info.get("pitch", 0.0)) or 0.0)
    yaw = float(info.get("bake_yaw", info.get("yaw", 0.0)) or 0.0)
    R_bake = np.asarray(rm.rotmat_from_euler(roll, pitch, yaw), dtype=float)
    R_up = np.asarray(
        rm.rotmat_from_euler(float(np.pi / 2.0), 0.0, float(place_yaw)),
        dtype=float,
    )
    try:
        return R_up @ np.linalg.inv(R_bake)
    except np.linalg.LinAlgError:
        return R_up.copy()


def _seat_long_axis_upright(model, rot, min_dot: float = 0.85) -> bool:
    """入格后最长边是否接近世界 +Z（防止平躺验收漏网）。"""
    try:
        verts = np.asarray(model.trm_mesh.vertices, dtype=float)
    except Exception:
        return True
    if verts.shape[0] < 2:
        return True
    world = (np.asarray(rot, dtype=float) @ verts.T).T
    extents = world.max(axis=0) - world.min(axis=0)
    axis = int(np.argmax(extents))
    return axis == 2 and float(extents[2]) >= float(min_dot) * float(np.max(extents))


def _slot_lean_signs(slot_id: int):
    """按格位选择靠壁方向（朝料盘内侧格板），避免全部挤同一外角。"""
    col = int(slot_id) % int(SLOT_COLS)
    row = int(slot_id) // int(SLOT_COLS)
    sx = -1.0 if col >= (int(SLOT_COLS) // 2) else 1.0
    sy = -1.0 if row >= (int(SLOT_ROWS) // 2) else 1.0
    return float(sx), float(sy)


def seat_pose_in_slot(
    slot_id: int, half_xy=(0.0027, 0.0027), place_obj_rot=None, lean_frac=0.0
):
    """入格名义位姿：底面贴格底，默认落在格心（精调阶段再靠对应小格板）。"""
    _ = half_xy
    _ = lean_frac
    xy = slot_world_xy(int(slot_id))
    pos = np.array(
        [float(xy[0]), float(xy[1]), float(EC_SEAT_FLOOR_Z)],
        dtype=float,
    )
    rot = (
        np.asarray(place_obj_rot, dtype=float).copy()
        if place_obj_rot is not None
        else UPRIGHT_OBJ_ROT.copy()
    )
    return pos, rot


def _mesh_world_zmin(model, pos, rot):
    """零件参考 mesh 在位姿下的世界 z 最小（用于贴底、防穿桌/穿盒底）。"""
    try:
        verts = np.asarray(model.trm_mesh.vertices, dtype=float)
    except Exception:
        return float(pos[2])
    if verts.size == 0:
        return float(pos[2])
    world = (np.asarray(rot, dtype=float) @ verts.T).T + np.asarray(pos, dtype=float)
    return float(np.min(world[:, 2]))


def _mesh_world_aabb(model, pos, rot):
    try:
        verts = np.asarray(model.trm_mesh.vertices, dtype=float)
    except Exception:
        p = np.asarray(pos, dtype=float)
        return p.copy(), p.copy()
    world = (np.asarray(rot, dtype=float) @ verts.T).T + np.asarray(pos, dtype=float)
    return world.min(axis=0), world.max(axis=0)


def refine_seat_no_penetration(model, pos, rot, box_model=None, slot_id=None, half_xy=None):
    """几何入格：贴格底，并让零件 AABB 中心尽量对齐格心（间隙≥0）。"""
    _ = box_model
    _ = half_xy
    pos = np.asarray(pos, dtype=float).copy()
    rot = np.asarray(rot, dtype=float).copy()
    floor_z = float(EC_SEAT_FLOOR_Z)
    clr = float(EC_SEAT_CLEARANCE)
    wall_clr = 2.5e-4

    def _snap_floor(p):
        """底面贴格底：过低抬起、悬空压下，保证 zmin≈floor（间隙≥0）。"""
        p = np.asarray(p, dtype=float).copy()
        zmin = _mesh_world_zmin(model, p, rot)
        p[2] += (floor_z + clr) - zmin
        return p

    pos = _snap_floor(pos)

    if slot_id is None:
        return pos, rot

    cxy = slot_world_xy(int(slot_id))
    half_cell = np.array(
        [0.5 * float(SLOT_PITCH_X) - wall_clr, 0.5 * float(SLOT_PITCH_Y) - wall_clr],
        dtype=float,
    )

    def _center_xy(p):
        """AABB 中心对齐格心。"""
        p = np.asarray(p, dtype=float).copy()
        mn, mx = _mesh_world_aabb(model, p, rot)
        mid = 0.5 * (mn[:2] + mx[:2])
        p[0] += cxy[0] - mid[0]
        p[1] += cxy[1] - mid[1]
        return p

    def _fit_xy(p):
        """把 AABB 平移进格内；若零件比格子大则居中。"""
        p = np.asarray(p, dtype=float).copy()
        mn, mx = _mesh_world_aabb(model, p, rot)
        size = mx[:2] - mn[:2]
        for i in range(2):
            lo = cxy[i] - half_cell[i]
            hi = cxy[i] + half_cell[i]
            if size[i] >= (hi - lo) - 1e-9:
                mid = 0.5 * (mn[i] + mx[i])
                p[i] += cxy[i] - mid
            else:
                if mn[i] < lo:
                    p[i] += lo - mn[i]
                if mx[i] > hi:
                    p[i] += hi - mx[i]
                mn, mx = _mesh_world_aabb(model, p, rot)
                if mn[i] < lo:
                    p[i] += lo - mn[i]
                if mx[i] > hi:
                    p[i] += hi - mx[i]
        return p

    # 先居中再夹紧，避免漂到角落；若仍有余裕，非超宽轴保持格心对齐。
    pos = _snap_floor(_fit_xy(_center_xy(pos)))
    mn, mx = _mesh_world_aabb(model, pos, rot)
    size = mx[:2] - mn[:2]
    cell = 2.0 * half_cell
    for axis in (0, 1):
        if float(size[axis]) < float(cell[axis]) - 1e-9:
            mid = 0.5 * (mn[axis] + mx[axis])
            pos[axis] += cxy[axis] - mid
    pos = _snap_floor(_fit_xy(pos))

    mn, mx = _mesh_world_aabb(model, pos, rot)
    mid_xy = 0.5 * (mn[:2] + mx[:2])
    center_err = float(np.linalg.norm(mid_xy - cxy))
    print(
        f"[seat] 几何入格 zmin={mn[2]:.4f} "
        f"xy=[{pos[0]:.4f},{pos[1]:.4f}] "
        f"center_err={center_err:.4f} (格心居中, 间隙≥0)",
        flush=True,
    )
    return pos, rot



def _ec_joint_lerp_motion(robot, q0, q1, jaw_open, n_steps: int = 14):
    """无碰撞插值失败时的关节线性过渡（保证动画能回到原位）。"""
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    q1 = np.asarray(q1, dtype=float).reshape(-1)
    n = max(2, int(n_steps))
    mot = MotionData(robot)
    jvs = []
    evs = []
    for i in range(n + 1):
        t = float(i) / float(n)
        jvs.append((1.0 - t) * q0 + t * q1)
        evs.append(float(jaw_open))
    mot.extend(jvs, ev_list=evs, mesh_list=[])
    return mot


def _ec_append_return_home(
    robot, mot, jaw_open, use_rrt: bool = False, obstacles=None
):
    """放置松爪后只做小抬升；连续 job 不再每件回 HOME。"""
    if mot is None or len(mot.jv_list) < 1:
        return mot
    home = np.asarray(side.HOME_CONF, dtype=float)
    q_end = np.asarray(mot.jv_list[-1], dtype=float)
    if float(np.linalg.norm(q_end - home)) < 1.5e-2:
        return mot
    obs = list(obstacles or [])
    planner = side.ppp.PickPlacePlanner(robot)
    depart = None
    for d in (0.06, 0.04, 0.02):
        try:
            depart = planner.gen_linear_depart_from_given_conf(
                start_jnt_values=q_end,
                direction=rm.const.z_ax,
                distance=float(d),
                ee_values=float(jaw_open),
                granularity=float(side.LINEAR_GRANULARITY),
                obstacle_list=obs,
            )
        except Exception:
            depart = None
        if depart is not None:
            break
    q_from = (
        np.asarray(depart.jv_list[-1], dtype=float)
        if depart is not None
        else q_end
    )
    out = mot
    if depart is not None:
        print("[ppp] 连续任务：松爪后顶部撤离，不回 HOME", flush=True)
        out = out + depart
    else:
        _ = q_from
        print("[ppp] 连续任务：无顶部撤离段，保留放置轨迹，不回 HOME", flush=True)
    return out


def thin_motion(mot, robot, stride=ANIME_FRAME_STRIDE, keep_indices=None, target_n=ANIME_TARGET_FRAMES):
    """抽稀轨迹帧，保留首尾与关键索引（hold/release），并多留闭合夹爪前后帧。"""
    if mot is None:
        return None, {}
    n = len(mot.jv_list)
    if n <= 2:
        return mot, {i: i for i in range(n)}
    keep = {0, n - 1}
    if keep_indices:
        for i in keep_indices:
            if i is None:
                continue
            i = int(np.clip(int(i), 0, n - 1))
            keep.add(i)
            # 闭合/松爪附近多留几帧，避免动画里“夹爪收缩消失”
            for d in range(-2, 2):
                keep.add(int(np.clip(i + d, 0, n - 1)))
    step = max(1, int(stride))
    if target_n and n > int(target_n):
        step = max(step, int(np.ceil(n / float(target_n))))
    keep.update(range(0, n, step))
    idxs = sorted(keep)
    out = MotionData(robot)
    evs = []
    for i in idxs:
        ev = mot.ev_list[i] if i < len(mot.ev_list) else None
        evs.append(ev)
    out.extend(
        jv_list=[mot.jv_list[i] for i in idxs],
        ev_list=evs,
        mesh_list=[],
    )
    remap = {old: new for new, old in enumerate(idxs)}
    _ec_log(f"[anime] 抽稀帧 {n} → {len(idxs)} (stride≈{step})")
    return out, remap


def _ensure_visible_jaw_close(pkg, n_blend: int = 6):
    """动画用：在 hold 前强制写出 jaw_open→jaw_close 过渡，避免瞬间接合。"""
    mot = pkg.get("mot")
    cs = pkg.get("carry_start")
    if mot is None or cs is None:
        return
    jo = float(pkg["jaw_open"])
    jc = float(pkg["jaw_close"])
    cs = int(cs)
    n = len(mot.ev_list)
    if n <= 0 or cs <= 0:
        return
    start = max(0, cs - int(n_blend))
    span = max(1, cs - start)
    for i in range(0, start):
        mot.ev_list[i] = jo
    for i in range(start, min(cs + 1, n)):
        t = float(i - start) / float(span)
        mot.ev_list[i] = float(jo + t * (jc - jo))


def _snapshot_part_pose(info: dict) -> dict:
    m = info["model"]
    return {
        "pos": np.asarray(m.pos, dtype=float).copy(),
        "rotmat": np.asarray(m.rotmat, dtype=float).copy(),
        "state": info.get("state"),
        "info_pos": np.asarray(info.get("pos", m.pos), dtype=float).copy(),
        "bake_roll": float(info.get("bake_roll", info.get("roll", 0.0)) or 0.0),
        "bake_pitch": float(info.get("bake_pitch", info.get("pitch", 0.0)) or 0.0),
        "bake_yaw": float(info.get("bake_yaw", info.get("yaw", 0.0)) or 0.0),
        "stl_path": info.get("stl_path") or BUGLE_STL,
        "class": info.get("class") or PART_CLASS,
    }


def _restore_part_pose(info: dict, snap: dict, base=None):
    """还原位姿；若曾 remesh（如 inverted→fallen），按 bake 重烘焙 mesh。"""
    stl = snap.get("stl_path") or info.get("stl_path") or BUGLE_STL
    roll = float(snap.get("bake_roll", 0.0) or 0.0)
    pitch = float(snap.get("bake_pitch", 0.0) or 0.0)
    yaw = float(snap.get("bake_yaw", 0.0) or 0.0)
    need_remesh = (
        abs(float(info.get("bake_roll", roll)) - roll) > 1e-6
        or abs(float(info.get("bake_pitch", pitch)) - pitch) > 1e-6
        or abs(float(info.get("bake_yaw", yaw)) - yaw) > 1e-6
        or _norm_state(info.get("state")) != _norm_state(snap.get("state"))
    )
    old = info.get("model")
    if need_remesh:
        mesh, height, grip_width = side._load_mesh_on_table(
            stl, yaw, roll=roll, pitch=pitch
        )
        rgb = rm.vec(0.70, 0.55, 0.40)
        name = str(snap.get("class") or info.get("class") or PART_CLASS)
        if old is not None:
            try:
                rgb = np.asarray(old.rgb, dtype=float)
            except Exception:
                pass
            name = str(getattr(old, "name", None) or name)
            try:
                old.detach()
            except Exception:
                pass
        model = mcm.CollisionModel(mesh, name=name, rgb=rgb)
        info["model"] = model
        info["height"] = float(height)
        info["top_z"] = float(height)
        info["mid_z"] = 0.5 * float(height)
        info["grip_width"] = float(grip_width)
        info["bake_roll"] = info["roll"] = roll
        info["bake_pitch"] = info["pitch"] = pitch
        info["bake_yaw"] = info["yaw"] = yaw
        info["stl_path"] = stl
        m = model
    else:
        m = old
        try:
            m.detach()
        except Exception:
            pass
    m.pos = np.asarray(snap["pos"], dtype=float).copy()
    m.rotmat = np.asarray(snap["rotmat"], dtype=float).copy()
    info["pos"] = np.asarray(snap["info_pos"], dtype=float).copy()
    if snap.get("state") is not None:
        info["state"] = snap["state"]
    if base is not None:
        m.attach_to(base)


def _seat_rot_matching_hold(info, Rm, ac_r, place_yaw: float = 0.0) -> np.ndarray:
    """选与当前持物最接近、且长轴朝上的入格 rot（避免松爪瞬间再颠倒 180°）。"""
    ac_r = np.asarray(ac_r, dtype=float)
    Rm = np.asarray(Rm, dtype=float)
    R_hold = Rm @ ac_r.T
    yaw0 = float(place_yaw or 0.0)
    best_R = seat_rot_undo_bake(info, place_yaw=yaw0)
    best_score = -1e9
    for dy in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
        R = seat_rot_undo_bake(info, place_yaw=yaw0 + float(dy))
        # 长轴（mesh +Y）须朝世界 +Z，禁止头朝下
        axis = R @ np.array([0.0, 1.0, 0.0], dtype=float)
        if float(axis[2]) < 0.75:
            continue
        if not _seat_long_axis_upright(info.get("model"), R):
            continue
        score = float(np.trace(R_hold.T @ R))
        if score > best_score:
            best_score = score
            best_R = R
    return np.asarray(best_R, dtype=float).copy()


def _sync_place_pose_from_release_tcp(pkg, robot):
    """用当前松爪 TCP 反算落点；入格姿态优先用已翻正的持物 rot。"""
    ac_p = pkg.get("grasp_ac_pos")
    ac_r = pkg.get("grasp_ac_rotmat")
    if ac_p is None or ac_r is None:
        return False
    ac_p = np.asarray(ac_p, dtype=float).reshape(3)
    ac_r = np.asarray(ac_r, dtype=float)
    tcp = np.asarray(robot.gl_tcp_pos, dtype=float)
    Rm = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    yaw_u = float(pkg.get("place_yaw", 0.0) or 0.0)
    if str(pkg.get("seat_kind") or "") in ("buffer_fallen", "buffer_clear"):
        R_obj = Rm @ ac_r.T
        pos = tcp - R_obj @ ac_p
        rot = np.asarray(pkg.get("place_pose", (pos, np.eye(3)))[1], dtype=float)
        if rot.shape != (3, 3):
            rot = np.eye(3)
        pkg["place_pose"] = (np.asarray(pos, dtype=float).copy(), rot)
        return True
    # 箱格：优先沿用放置段已翻正的持物 / place_pose，避免松爪再颠倒
    rot = None
    info = pkg["target"]
    try:
        held = info.get("model")
        if held is not None:
            R_h = np.asarray(held.rotmat, dtype=float)
            axis = R_h @ np.array([0.0, 1.0, 0.0], dtype=float)
            if float(axis[2]) >= 0.75 and _seat_long_axis_upright(held, R_h):
                rot = R_h.copy()
    except Exception:
        rot = None
    if rot is None:
        try:
            R_p = np.asarray(pkg.get("place_pose", (None, None))[1], dtype=float)
            if R_p.shape == (3, 3):
                axis = R_p @ np.array([0.0, 1.0, 0.0], dtype=float)
                if float(axis[2]) >= 0.75 and _seat_long_axis_upright(
                    info.get("model"), R_p
                ):
                    rot = R_p.copy()
        except Exception:
            rot = None
    if rot is None:
        rot = _seat_rot_matching_hold(info, Rm, ac_r, place_yaw=yaw_u)
    pos = tcp - rot @ ac_p
    pkg["place_pose"] = (np.asarray(pos, dtype=float).copy(), rot)
    pkg["place_yaw"] = yaw_u
    return True


def _rotmat_blend(R0, R1, t: float) -> np.ndarray:
    """两姿态球面插值，t∈[0,1]。"""
    from scipy.spatial.transform import Rotation as SciR
    from scipy.spatial.transform import Slerp

    t = float(np.clip(t, 0.0, 1.0))
    R0 = np.asarray(R0, dtype=float)
    R1 = np.asarray(R1, dtype=float)
    if t <= 1e-9:
        return R0.copy()
    if t >= 1.0 - 1e-9:
        return R1.copy()
    slerp = Slerp([0.0, 1.0], SciR.from_matrix((R0, R1)))
    return np.asarray(slerp(t).as_matrix(), dtype=float)


def _hold_object_with_rot(robot, held_model, jaw_close, ac_p, R_obj, jnt_values=None):
    """按给定世界姿态重贴持物，夹持点仍对 TCP。"""
    ac_p = np.asarray(ac_p, dtype=float).reshape(3)
    R_obj = np.asarray(R_obj, dtype=float)
    if len(robot.end_effector.oiee_list) > 0:
        robot.end_effector.release_all()
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values, ee_values=float(jaw_close))
    tcp = np.asarray(robot.gl_tcp_pos, dtype=float)
    pos = tcp - R_obj @ ac_p
    held_model.pos = pos
    held_model.rotmat = R_obj
    robot.hold(held_model, jaw_width=float(jaw_close))
    return pos.copy(), R_obj.copy()


def _hold_object_seat_upright(robot, pkg, held_model, jaw_close, jnt_values=None):
    """放置段：把持物翻成与最终入格相同的竖直姿态（夹持点仍对 TCP）。

    返回 (ok, flipped, R_hold, R_seat)。
    """
    ac_p = pkg.get("grasp_ac_pos")
    ac_r = pkg.get("grasp_ac_rotmat")
    if ac_p is None or ac_r is None:
        return False, False, None, None
    if str(pkg.get("seat_kind") or "") in ("buffer_fallen", "buffer_clear"):
        return False, False, None, None
    ac_p = np.asarray(ac_p, dtype=float).reshape(3)
    ac_r = np.asarray(ac_r, dtype=float)
    if jnt_values is not None:
        if len(robot.end_effector.oiee_list) > 0:
            robot.end_effector.release_all()
        robot.goto_given_conf(jnt_values=jnt_values, ee_values=float(jaw_close))
    Rm = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    R_hold = Rm @ ac_r.T
    yaw_u = float(pkg.get("place_yaw", 0.0) or 0.0)
    R_seat = _seat_rot_matching_hold(pkg["target"], Rm, ac_r, place_yaw=yaw_u)
    # SO(3) trace≈3 完全对齐；<2.85（约≥20°）即做放置段翻正，避免松爪再颠倒
    flipped = float(np.trace(R_hold.T @ R_seat)) < 2.85
    # 持物长轴未竖直时强制翻正（fallen→入格 / inverted 第二段常见）
    hold_axis = R_hold @ np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(hold_axis[2])) < 0.85:
        flipped = True
    pos, R_use = _hold_object_with_rot(
        robot, held_model, jaw_close, ac_p, R_seat, jnt_values=None
    )
    pkg["place_pose"] = (pos, R_use)
    # 锁定入格 rot，松爪 seat 不再另选颠倒/偏航
    pkg["seat_rot_locked"] = np.asarray(R_seat, dtype=float).copy()
    return True, bool(flipped), R_hold.copy(), R_seat.copy()


def seat_in_slot(pkg, base, robot, box_model=None):
    """松爪后落入格 / 缓冲区。"""
    target = pkg["target"]
    if len(robot.end_effector.oiee_list) > 0:
        robot.end_effector.release_all()

    # inverted 第一段：放到缓冲区并重烘焙为 fallen
    if str(pkg.get("seat_kind") or "") == "buffer_fallen":
        # 落点必须对齐松爪时螺丝中心（place_pose / release_obj_xy），勿只用规划 buffer_xy
        xy = pkg.get("release_obj_xy")
        if xy is None and pkg.get("place_pose") is not None:
            xy = np.asarray(pkg["place_pose"][0], dtype=float)[:2]
        if xy is None:
            xy = pkg.get("buffer_xy")
        if xy is None:
            xy = np.asarray(pkg["place_pose"][0], dtype=float)[:2]
        xy = np.asarray(xy, dtype=float).reshape(-1)[:2]
        yaw = float(pkg.get("buffer_yaw", 0.0) or 0.0)
        remesh_part_as_fallen(target, xy, yaw=yaw, base=base)
        pos = np.asarray(target["pos"], dtype=float).copy()
        pkg["place_pose"] = (pos, np.eye(3))
        pkg["buffer_xy"] = np.asarray(pos[:2], dtype=float).copy()
        return

    # 挡路件暂存缓冲：保持原姿态，只改 XY（不 remesh）
    if str(pkg.get("seat_kind") or "") == "buffer_clear":
        xy = pkg.get("release_obj_xy")
        if xy is None and pkg.get("place_pose") is not None:
            xy = np.asarray(pkg["place_pose"][0], dtype=float)[:2]
        if xy is None:
            xy = pkg.get("buffer_xy")
        if xy is None:
            xy = np.asarray(pkg["place_pose"][0], dtype=float)[:2]
        m = target["model"]
        try:
            m.detach()
        except Exception:
            pass
        z = float(np.asarray(target.get("pos", m.pos), dtype=float)[2])
        pos = np.array([float(xy[0]), float(xy[1]), z], dtype=float)
        rot = np.asarray(m.rotmat, dtype=float).copy()
        m.pos = pos
        m.rotmat = rot
        target["pos"] = pos.copy()
        if base is not None:
            m.attach_to(base)
        pkg["place_pose"] = (pos.copy(), rot)
        print(
            f"[buffer-clear] id={target.get('id')} 暂存 @ "
            f"{np.round(pos[:2], 3).tolist()} state={target.get('state')}",
            flush=True,
        )
        return

    m = target["model"]
    try:
        m.detach()
    except Exception:
        pass
    seat_pos = np.asarray(pkg["place_pose"][0], dtype=float).copy()
    yaw_u = float(pkg.get("place_yaw", 0.0) or 0.0)
    # 优先用放置段锁定/已翻正的 rot，避免松爪后再颠倒一次
    seat_rot = None
    try:
        locked = pkg.get("seat_rot_locked")
        if locked is not None:
            cand_l = np.asarray(locked, dtype=float)
            if cand_l.shape == (3, 3):
                axis_l = cand_l @ np.array([0.0, 1.0, 0.0], dtype=float)
                if float(axis_l[2]) >= 0.70:
                    seat_rot = cand_l.copy()
    except Exception:
        seat_rot = None
    if seat_rot is None:
        try:
            cand0 = np.asarray(pkg["place_pose"][1], dtype=float)
            if cand0.shape == (3, 3) and _seat_long_axis_upright(m, cand0):
                axis = cand0 @ np.array([0.0, 1.0, 0.0], dtype=float)
                if float(axis[2]) >= 0.70:
                    seat_rot = cand0.copy()
        except Exception:
            seat_rot = None
    if seat_rot is None:
        seat_rot = seat_rot_undo_bake(target, place_yaw=yaw_u)
    half_xy = pkg.get("half_xy")
    seat_pos, seat_rot = refine_seat_no_penetration(
        m,
        seat_pos,
        seat_rot,
        box_model=box_model if box_model is not None else pkg.get("box_model"),
        slot_id=pkg.get("slot_id"),
        half_xy=half_xy,
    )
    if not _seat_long_axis_upright(m, seat_rot):
        for dy in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
            cand = seat_rot_undo_bake(target, place_yaw=float(dy))
            p2, r2 = refine_seat_no_penetration(
                m,
                seat_pos,
                cand,
                box_model=box_model if box_model is not None else pkg.get("box_model"),
                slot_id=pkg.get("slot_id"),
                half_xy=half_xy,
            )
            if _seat_long_axis_upright(m, r2):
                seat_pos, seat_rot = p2, r2
                break
    m.pos = seat_pos
    m.rotmat = seat_rot
    m.attach_to(base)
    target["state"] = "normal"
    target["pos"] = seat_pos.copy()
    target["roll"] = float(np.pi / 2.0)
    target["pitch"] = 0.0
    pkg["place_pose"] = (seat_pos, seat_rot)
    zmin = _mesh_world_zmin(m, seat_pos, seat_rot)
    ok_up = _seat_long_axis_upright(m, seat_rot)
    print(
        f"[seat] slot={pkg['slot_id']} pos={np.round(seat_pos, 4).tolist()} "
        f"zmin={zmin:.4f} upright_axis={'OK' if ok_up else 'BAD'}",
        flush=True,
    )


def _ec_hold_seat_metrics(robot, grasp, target, place_yaw, q, jaw_close):
    """持物(grasp一致) vs 入格：返回 (trace, axis_dot, R_hold, R_seat)。"""
    q = np.asarray(q, dtype=float).copy()
    robot.goto_given_conf(q, ee_values=float(jaw_close))
    Rm = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    ac_r = np.asarray(grasp.ac_rotmat, dtype=float)
    R_hold = Rm @ ac_r.T
    R_seat = _seat_rot_matching_hold(
        target, Rm, ac_r, place_yaw=float(place_yaw or 0.0)
    )
    tr = float(np.trace(R_hold.T @ R_seat))
    ah = R_hold @ np.array([0.0, 1.0, 0.0], dtype=float)
    as_ = R_seat @ np.array([0.0, 1.0, 0.0], dtype=float)
    axis_dot = float(np.dot(ah, as_))
    return tr, axis_dot, R_hold, R_seat


def _ec_hold_matches_seat(tr, axis_dot, tr_min: float = 1.9, axis_min: float = 0.70) -> bool:
    """持物与入格同向：长轴同向且姿态接近（非 180° 反着）。"""
    return bool(float(axis_dot) >= float(axis_min) and float(tr) >= float(tr_min))


def _ec_hold_similar_to_seat(tr, axis_dot) -> bool:
    """松爪前一刻与入格姿态足够像 → 直接走原 fallen 解，不必翻面。

    SO(3): trace≈3 完全重合；tr≳1.5（约 <75°）且长轴同向即视为相似。
    id=2 一类 tr≈2.4 不应再转 180°。
    """
    return bool(float(tr) >= 1.5 and float(axis_dot) >= 0.50)


def _ec_hold_about_180_from_seat(tr, axis_dot) -> bool:
    """相对入格约差 180°（反着）→ 才用 J6=fallen_J6±180°。"""
    return bool(float(tr) < 0.5 or float(axis_dot) < 0.0)


def _ec_hold_long_axis_up(R_hold, min_z: float = 0.75) -> bool:
    """持物 mesh +Y（长轴）是否朝世界 +Z（入格前须竖直，禁平躺/倒头）。"""
    axis = np.asarray(R_hold, dtype=float) @ np.array([0.0, 1.0, 0.0], dtype=float)
    return bool(float(axis[2]) >= float(min_z))


def _ec_hold_upright_z(R_hold) -> float:
    """持物长轴（mesh +Y）在世界 +Z 上的分量；≈1 竖直，≈0 平躺。"""
    ah = np.asarray(R_hold, dtype=float) @ np.array([0.0, 1.0, 0.0], dtype=float)
    return float(ah[2])


def _ec_j6_motion_range(robot):
    try:
        j6_lo, j6_hi = [float(x) for x in robot.jlc.jnts[5].motion_range]
    except Exception:
        j6_lo, j6_hi = -np.pi, np.pi
    return float(j6_lo), float(j6_hi)


def ec_build_inhand_j6_to(
    robot,
    q_start,
    j6_target,
    jaw_close,
    desk_obstacle,
    n_steps: int = None,
    quiet: bool = False,
):
    """高位只转 J6 到目标角（夹爪+持物一起转，始终 grasp 一致）。

    仿真 J6∈±180°；大角度（含约 180°）按 ~10°/步插值。
    """
    q0 = np.asarray(q_start, dtype=float).copy()
    j6_tgt = float(j6_target)
    j6_lo, j6_hi = _ec_j6_motion_range(robot)
    if j6_tgt < j6_lo - 1e-6 or j6_tgt > j6_hi + 1e-6:
        return None
    d_j6 = float(j6_tgt - float(q0[5]))
    if abs(d_j6) < np.deg2rad(3.0):
        return None  # 已在目标附近，无需单独腕转段
    if n_steps is None:
        n_steps = int(max(10, min(36, round(abs(np.rad2deg(d_j6)) / 10.0))))
    qs = []
    for i in range(int(n_steps) + 1):
        a = float(i) / float(n_steps)
        q = q0.copy()
        q[5] = (1.0 - a) * float(q0[5]) + a * j6_tgt
        robot.goto_given_conf(q, ee_values=float(jaw_close))
        if not side.conf_tip_clears_desk(robot, q, jaw_close):
            return None
        if not _ec_ee_desk_clear(robot):
            return None
        qs.append(q.copy())
    if desk_obstacle is not None:
        probe = MotionData(robot)
        probe.extend(
            [qs[0], qs[len(qs) // 2], qs[-1]],
            ev_list=[float(jaw_close)] * 3,
            mesh_list=[],
        )
        if side.motion_penetrates_desk(
            robot, probe, desk_obstacle, jaw_fallback=jaw_close
        ):
            return None
    mot = MotionData(robot)
    mot.extend(qs, ev_list=[float(jaw_close)] * len(qs), mesh_list=[])
    if not quiet:
        print(
            f"[j6] 腕转 "
            f"{np.rad2deg(q0[5]):.1f}° → {np.rad2deg(j6_tgt):.1f}° "
            f"(Δ{np.rad2deg(d_j6):.1f}°，夹爪+持物同转)",
            flush=True,
        )
    return mot


def ec_build_inhand_j6_flip(
    robot,
    q_start,
    jaw_close,
    desk_obstacle,
    delta_rad: float = None,
    n_steps: int = 8,
):
    """兼容：按固定角度腕转（优先 ±180°）。新逻辑请用对齐搜索 + ec_build_inhand_j6_to。"""
    if delta_rad is None:
        delta_rad = float(np.pi)
    q0 = np.asarray(q_start, dtype=float).copy()
    j6_lo, j6_hi = _ec_j6_motion_range(robot)
    for sign in (1.0, -1.0):
        target_j6 = float(q0[5] + sign * float(delta_rad))
        if target_j6 < j6_lo - 1e-6 or target_j6 > j6_hi + 1e-6:
            continue
        mot = ec_build_inhand_j6_to(
            robot, q0, target_j6, jaw_close, desk_obstacle, n_steps=n_steps
        )
        if mot is not None:
            return mot
    print("[j6] 固定角腕转失败（限位/穿桌）", flush=True)
    return None


def _ec_choose_fallen_j6_for_seat(
    robot,
    q_rel,
    grasp,
    target,
    place_yaw,
    jaw_close,
    tip_min: float = None,
    tip_max: float = None,
):
    """在松爪臂型上选 J6：先判断持物 vs 入格，反着则腕转（可到 ±180°）。

    仿真 J6∈±180°。对齐搜索走关节全行程，不再被放置安全带 [-95,-25] 卡住。
    安全带仅作同质量解的偏好。matched=持物与入格同向且长轴竖直。

    返回 (j6, matched, tr, axis_dot, R_seat)。
    """
    q0 = np.asarray(q_rel, dtype=float).copy()
    j6_ref = float(q0[5])
    tr0, ax0, R_h0, R_s0 = _ec_hold_seat_metrics(
        robot, grasp, target, place_yaw, q0, jaw_close
    )
    if _ec_hold_matches_seat(tr0, ax0) and _ec_hold_long_axis_up(R_h0):
        return j6_ref, True, tr0, ax0, R_s0

    j6_lo, j6_hi = _ec_j6_motion_range(robot)
    tmin = float(tip_min) if tip_min is not None else None
    tmax = float(tip_max) if tip_max is not None else None

    def _tip_ok(q):
        if tmin is None or tmax is None:
            return True
        robot.goto_given_conf(q, ee_values=float(jaw_close))
        tz = side.finger_tips_z_min(robot.end_effector)
        return bool(np.isfinite(tz) and tmin <= float(tz) <= tmax)

    def _search(cands, prefer_band: bool):
        best = None  # (key, j6, tr, ax, R_seat)
        for j6 in cands:
            j6 = float(np.clip(float(j6), j6_lo, j6_hi))
            q = q0.copy()
            q[5] = float(j6)
            # 对齐：不强制放置安全带，只要求关节/自碰/离桌/tip
            if not _ec_conf_place_ok(robot, q, jaw_close, require_j6=False):
                continue
            if not _tip_ok(q):
                continue
            tr, ax, R_h, R_s = _ec_hold_seat_metrics(
                robot, grasp, target, place_yaw, q, jaw_close
            )
            up = _ec_hold_long_axis_up(R_h)
            matched = _ec_hold_matches_seat(tr, ax) and up
            in_band = _ec_j6_place_ok(q)
            key = (
                0 if matched else 1,
                0 if up else 1,
                -float(ax),
                -float(tr),
                (0 if in_band else 1) if prefer_band else 0,
                abs(float(j6) - float(EC_PLACE_J6_REF)),
            )
            if best is None or key < best[0]:
                best = (key, float(j6), float(tr), float(ax), R_s)
                if matched:
                    break
        return best

    # 1) 明确试 ±180°（用户要的「腕转半圈」，限位内）
    cands = [float(j6_ref), float(EC_PLACE_J6_REF)]
    for sign in (1.0, -1.0):
        cands.append(float(j6_ref + sign * np.pi))
        cands.append(float(EC_PLACE_J6_REF + sign * np.pi))
        cands.append(float(j6_ref + sign * 0.5 * np.pi))
    # 2) 安全带密搜（偏好）
    band = np.linspace(float(EC_PLACE_J6_MIN), float(EC_PLACE_J6_MAX), 29)
    cands.extend(float(x) for x in band)
    best = _search(cands, prefer_band=True)
    matched_b = bool(best is not None and best[0][0] == 0)

    # 3) 全行程 ±180° 密搜（真正放开，不再被 place-band continue 掉）
    if not matched_b:
        print(
            f"[j6] 未对齐 (tr0={tr0:.2f} axis0={ax0:.2f} "
            f"up0={float((R_h0 @ np.array([0.0,1.0,0.0]))[2]):.2f}) "
            f"→ J6 全行程±180°重搜",
            flush=True,
        )
        full = np.linspace(j6_lo, j6_hi, 73)
        best_full = _search(full, prefer_band=False)
        if best_full is not None and (
            best is None or best_full[0] < best[0]
        ):
            best = best_full
            matched_b = best[0][0] == 0

    if best is None:
        print(
            f"[j6] 无对齐解，保持 REF "
            f"J6={np.rad2deg(j6_ref):.1f}° (tr0={tr0:.2f} axis0={ax0:.2f})",
            flush=True,
        )
        return j6_ref, False, tr0, ax0, R_s0
    _, j6_b, tr_b, ax_b, R_sb = best
    q_chk = q0.copy()
    q_chk[5] = float(j6_b)
    _, _, R_hb, _ = _ec_hold_seat_metrics(
        robot, grasp, target, place_yaw, q_chk, jaw_close
    )
    matched_b = bool(
        _ec_hold_matches_seat(tr_b, ax_b) and _ec_hold_long_axis_up(R_hb)
    )
    # 仍明显反着 → 再比相对 REF/当前角的 ±180°（夹爪带着零件转半圈）
    if (not matched_b) and float(tr_b) < 1.0:
        flip_best = None
        for base in (float(j6_ref), float(j6_b), float(EC_PLACE_J6_REF)):
            for sign in (1.0, -1.0):
                j6_flip = float(np.clip(base + sign * np.pi, j6_lo, j6_hi))
                if abs(j6_flip - float(j6_ref)) < np.deg2rad(60.0):
                    continue
                q_f = q0.copy()
                q_f[5] = j6_flip
                if not _ec_conf_place_ok(robot, q_f, jaw_close, require_j6=False):
                    continue
                if not _tip_ok(q_f):
                    continue
                tr_f, ax_f, R_hf, R_sf = _ec_hold_seat_metrics(
                    robot, grasp, target, place_yaw, q_f, jaw_close
                )
                up_f = _ec_hold_long_axis_up(R_hf)
                matched_f = _ec_hold_matches_seat(tr_f, ax_f) and up_f
                key = (
                    0 if matched_f else 1,
                    0 if up_f else 1,
                    -float(tr_f),
                    -float(ax_f),
                    -abs(float(j6_flip) - float(j6_ref)),
                )
                if flip_best is None or key < flip_best[0]:
                    flip_best = (
                        key,
                        float(j6_flip),
                        float(tr_f),
                        float(ax_f),
                        R_sf,
                    )
        if flip_best is not None:
            _, j6_f, tr_f, ax_f, R_sf = flip_best
            if float(tr_f) + 0.05 >= float(tr_b):
                print(
                    f"[j6] 采用≈180°腕转 {np.rad2deg(j6_b):.1f}°→"
                    f"{np.rad2deg(j6_f):.1f}° "
                    f"(tr {tr_b:.2f}→{tr_f:.2f} axis {ax_b:.2f}→{ax_f:.2f})",
                    flush=True,
                )
                j6_b, tr_b, ax_b, R_sb = j6_f, tr_f, ax_f, R_sf
                q_chk = q0.copy()
                q_chk[5] = float(j6_b)
                _, _, R_hb, _ = _ec_hold_seat_metrics(
                    robot, grasp, target, place_yaw, q_chk, jaw_close
                )
                matched_b = bool(
                    _ec_hold_matches_seat(tr_b, ax_b)
                    and _ec_hold_long_axis_up(R_hb)
                )
    print(
        f"[j6] 选角 {np.rad2deg(j6_b):.1f}° "
        f"(tr={tr_b:.2f} axis={ax_b:.2f} matched={matched_b} "
        f"band={_ec_j6_place_ok(q_chk)})",
        flush=True,
    )
    return float(j6_b), bool(matched_b), float(tr_b), float(ax_b), R_sb


def _try_plan_pick_place(
    robot,
    target,
    part_grasps,
    jaw_open,
    jaw_close,
    grasp_mode,
    obstacles,
    desk_obstacle,
    place_obj_rot,
    use_rrt,
    slot_id: int = 0,
    place_xy=None,
    tip_degs=None,
    place_surface: str = "box",
    require_upright: bool = True,
    lock_style: bool = True,
    enforce_style: bool = True,
    use_fallen_tpl: bool = False,
    place_kind: str = None,
    inhand_reorient: bool = False,
    start_conf=None,
):
    """pick → (可选腕转) → 放置姿态 → 单调 −Z 下探松爪。

    use_fallen_tpl / place_kind=fallen_ref：place 钉死拧腕参考解。
    inhand_reorient：pick 后 J6≈180°（inverted 侧抓翻正）。
    place_kind=normal_vertical：夹爪竖直 ⊥XY 对齐格心下降松爪。
    """
    place_obj_rot = (
        UPRIGHT_OBJ_ROT if place_obj_rot is None else np.asarray(place_obj_rot)
    )
    tgt_xy = (
        np.asarray(place_xy, dtype=float).reshape(-1)[:2]
        if place_xy is not None
        else slot_world_xy(int(slot_id))
    )
    on_desk = str(place_surface or "box").lower() == "desk"
    if on_desk:
        # 桌面缓冲绝不用 fallen REF / 竖直顶放（否则高位 IK 常挂）
        use_fallen_tpl = False
        inhand_reorient = False
        place_kind = "desk"
    elif use_fallen_tpl:
        place_kind = "fallen_ref"
        use_fallen_tpl = True
    elif inhand_reorient and str(place_kind or "") != "normal_vertical":
        # inverted 默认：腕转后仍走 fallen REF；显式 normal_vertical 时保留（边缘侧抓）
        place_kind = "fallen_ref"
        use_fallen_tpl = True
    elif place_kind is None:
        place_kind = str(_PLACE_STYLE.get("kind") or "normal_vertical")
        if place_kind == "fallen_ref" and not use_fallen_tpl:
            place_kind = "normal_vertical"
    place_kind = str(place_kind)
    is_fallen_place = (not on_desk) and (
        place_kind == "fallen_ref" or bool(use_fallen_tpl)
    )
    is_normal_place = (not is_fallen_place) and (not on_desk)

    n_grasp = int(
        EC_FALLEN_GRASP_TOP_N
        if (not on_desk)
        and (
            place_kind == "fallen_ref"
            or bool(use_fallen_tpl)
            or bool(inhand_reorient)
        )
        else EC_STABLE_GRASP_TOP_N
    )
    if is_fallen_place:
        pin_fallen_place_style()
        tip_degs = (float(EC_PLACE_REF_TIP_DEG),)
        if inhand_reorient:
            grasps = _ec_stable_sort_grasps(part_grasps)[:n_grasp]
        else:
            grasps = _rank_grasps_for_fallen_pick(part_grasps, top_n=n_grasp)
        # 只空障一轮：邻件由 main 清障，避免双倍 pick
        obs_rounds = [[]]
        print(
            f"[fallen] place=REF tip=15°；J6=一致用REF/反着再腕转；"
            f"pick n={len(grasps)} mode={grasp_mode}"
            f"{' (inverted侧抓)' if inhand_reorient else ''}",
            flush=True,
        )
    else:
        if is_normal_place:
            pin_normal_place_style()
            if tip_degs is None:
                tip_degs = EC_PLACE_TIP_DEGS_TOP
        grasps = _ec_stable_sort_grasps(part_grasps)[:n_grasp]
        obs_rounds = [[]]

    print(
        f"[ppp] pick({grasp_mode}) → place@{np.round(tgt_xy,3).tolist()} "
        f"surface={place_surface} rrt={bool(use_rrt)} "
        f"kind={place_kind} grasps={len(grasps)}",
        flush=True,
    )

    rrt_flags = [False, True] if use_rrt else [False]
    if grasp_mode == "side":
        rrt_flags = [False]

    for rrt_flag in rrt_flags:
        for gi, grasp in enumerate(grasps):
            gc_one = gg.GraspCollection(end_effector=robot.end_effector)
            gc_one.append(grasp)
            # 选定本 grasp 后：唯一权威闭爪宽 = grasp.ee_values（覆盖中位种子）
            jc_use = float(getattr(grasp, "ee_values", jaw_close) or jaw_close)
            jc_use = float(
                np.clip(jc_use, side.JAW_CLOSE_MIN, float(jaw_open) - 1e-4)
            )
            jaw_close = float(jc_use)
            pick_mot = None
            for oi, obs in enumerate(obs_rounds):
                pick_mot = side.plan_pick_lift(
                    robot,
                    obj_cmodel=target["model"],
                    grasp_collection=gc_one,
                    jaw_open=jaw_open,
                    jaw_close=jaw_close,
                    obstacles=obs,
                    desk=desk_obstacle,
                    use_rrt=bool(rrt_flag),
                    grasp_mode=grasp_mode,
                    start_conf=start_conf,
                )
                if pick_mot is not None:
                    if oi > 0:
                        _ec_log(f"[ppp] pick OK grasp#{gi} 障碍回退 obs={len(obs)}")
                    break
            if pick_mot is None:
                continue
            # 抓取 TCP 复验（抬起后物体已不在指间，须在抓取位查穿模）
            if not _ec_pick_pose_quality(grasp, target["model"], grasp_mode):
                print(f"[ppp] pick 姿态不合格 grasp#{gi}，丢弃", flush=True)
                side._reset_robot_hand(robot, jaw_open)
                continue
            tcp_pos, tcp_rot = _ec_grasp_tcp(target["model"], grasp)
            q_g = side._ik_with_seed(robot, tcp_pos, tcp_rot, pick_mot.jv_list[0])
            if q_g is not None:
                # 必须用本 grasp 的闭合宽（含螺头≈1mm 间隙），勿用全局 median
                probe_eps = float(getattr(side, "JAW_CLEARANCE_PROBE", 0.0015))
                probe_w = float(jc_use) + probe_eps
                robot.goto_given_conf(q_g, ee_values=float(jaw_open))
                if _ec_ee_hits_object(robot, target["model"], jaw_open):
                    print(f"[ppp] 抓取位张开穿模 grasp#{gi}，丢弃", flush=True)
                    side._reset_robot_hand(robot, jaw_open)
                    continue
                robot.goto_given_conf(q_g, ee_values=probe_w)
                if _ec_ee_hits_object(robot, target["model"], probe_w):
                    w_open = _ec_obj_width_along_open(target["model"], tcp_rot)
                    gap = float(EC_JAW_HEAD_CLEARANCE_M)
                    # 细螺丝 + 已留间隙：厚指 CD 易误报深穿，放行
                    if float(w_open) <= 0.012 and float(jc_use) + 1e-4 >= float(
                        w_open
                    ) + gap:
                        print(
                            f"[ppp] 抓取位闭合 CD 深穿忽略 grasp#{gi} "
                            f"(细件已留≈{gap*1e3:.0f}mm间隙 jc={jc_use:.4f})",
                            flush=True,
                        )
                    else:
                        print(
                            f"[ppp] 抓取位闭合深穿模 grasp#{gi}，丢弃",
                            flush=True,
                        )
                        side._reset_robot_hand(robot, jaw_open)
                        continue
            q_start = np.asarray(pick_mot.jv_list[-1], dtype=float)
            place_start_raw = len(pick_mot.jv_list)
            # fallen/inverted 入格：不在此盲转 ±180°；
            # 由 ec_build_fallen_ref_place 判断持物 vs 入格，反着再转 J6。
            _ = inhand_reorient
            # 桌面缓冲：侧倾松爪（勿 tip=0 顶放），随后 remesh 成 fallen
            place_mot = None
            place_meta = None
            if str(place_surface).lower() == "desk" and not require_upright:
                robot.goto_given_conf(q_start, ee_values=float(jaw_close))
                R_pick = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
                use_tips = (
                    tip_degs
                    if tip_degs is not None
                    else (45.0, 40.0, 50.0, 35.0)
                )
                # 非零 tip 优先；若全失败再退 tip=0 顶放
                rot_cands = _ec_tilt_desk_tcp_cands(R_pick, use_tips)
                if not any(abs(float(t[5])) >= 5.0 for t in rot_cands):
                    rot_cands.append(
                        (
                            1.0,
                            0.0,
                            1.0,
                            abs(float(R_pick[0, 0])),
                            0.0,
                            0.0,
                            R_pick,
                        )
                    )
                # tip=0 顶放仅作全部侧倾失败后的兜底
                rot_cands.append(
                    (
                        1.0,
                        0.0,
                        1.0,
                        abs(float(R_pick[0, 0])),
                        0.0,
                        0.0,
                        R_pick.copy(),
                    )
                )
                print(
                    f"[ppp] desk buffer 侧倾 place×{len(rot_cands)} "
                    f"tips={list(use_tips)}(+0°兜底)",
                    flush=True,
                )
            elif is_fallen_place:
                # fallen：place 与 grasp 朝向解耦，只走固定 REF 臂型
                rot_cands = [
                    (
                        1.0,
                        0.0,
                        0.26,
                        1.0,
                        float(EC_PLACE_REF_DYAW),
                        float(EC_PLACE_REF_TIP_DEG),
                        np.eye(3),
                    )
                ]
            elif is_normal_place:
                # normal：优先竖直顶放专用路径（不依赖 grasp 反算门禁）
                cand_nv = ec_build_normal_vertical_place(
                    robot,
                    q_start=q_start,
                    seat_xy=tgt_xy,
                    slot_id=int(slot_id),
                    jaw_close=jaw_close,
                    jaw_open=jaw_open,
                    desk_obstacle=desk_obstacle,
                    grasp=grasp,
                )
                if cand_nv is not None:
                    q_rel = np.asarray(cand_nv.jv_list[-1], dtype=float)
                    place_mot = cand_nv
                    place_meta = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, q_rel)
                    print(
                        f"[ppp] place OK tip=0° "
                        f"j5={np.rad2deg(q_rel[4]):.1f}° "
                        f"j6={np.rad2deg(q_rel[5]):.1f}° "
                        f"app↓=vert VERT",
                        flush=True,
                    )
                    rot_cands = []
                else:
                    use_tips = tip_degs if tip_degs is not None else EC_PLACE_TIP_DEGS_TOP
                    rot_cands = _ec_vertical_place_tcp_rots(
                        grasp,
                        place_obj_rot,
                        tip_degs=use_tips,
                        place_kind="normal_vertical",
                    )
                    vert = [t for t in rot_cands if float(t[2]) >= 0.70]
                    rot_cands = (vert if vert else list(rot_cands))[
                        : int(EC_PLACE_ROT_TOP_N)
                    ]
            else:
                use_tips = tip_degs
                pk = "normal_vertical"
                rot_cands = _ec_vertical_place_tcp_rots(
                    grasp,
                    place_obj_rot,
                    tip_degs=use_tips,
                    place_kind=pk,
                )
                n_rot = int(EC_PLACE_ROT_TOP_N)
                if is_normal_place:
                    vert = [t for t in rot_cands if float(t[2]) >= 0.75]
                    rot_cands = (vert if vert else list(rot_cands))[:n_rot]
                else:
                    rot_cands = list(rot_cands)[:n_rot]
            if place_mot is not None:
                # normal 竖直捷径已成功
                pass
            else:
                print(
                    f"[ppp] grasp#{gi} pick OK → 试 place×{len(rot_cands)}（不重算 pick）",
                    flush=True,
                )
            desk_vert_fallback = None  # (mot, meta) 侧倾全失败时用
            for upright, off_xy, tip_down, align_x, dyaw, tip_deg, R_tcp in rot_cands:
                if require_upright and (not is_fallen_place):
                    tip_floor = max(
                        0.85,
                        float(np.cos(np.deg2rad(float(tip_deg)))) - 0.05,
                    )
                    if upright < tip_floor or off_xy > 0.065:
                        continue
                if is_normal_place and float(tip_down) < 0.75:
                    continue
                cand = ec_build_place_neg_z(
                    robot,
                    q_start=q_start,
                    seat_xy=tgt_xy,
                    slot_id=int(slot_id),
                    jaw_close=jaw_close,
                    jaw_open=jaw_open,
                    desk_obstacle=desk_obstacle,
                    tcp_rot=R_tcp,
                    hover_z=float(EC_HIGH_HOVER_Z),
                    release_tcp_floor=(
                        0.030
                        if str(place_surface).lower() == "desk"
                        else float(EC_RELEASE_TCP_FLOOR)
                    ),
                    ee_align_x=align_x,
                    grasp=grasp,
                    approach_tip_deg=tip_deg,
                    place_surface=place_surface,
                    require_upright=(False if is_fallen_place else require_upright),
                    force_ref_arm=bool(is_fallen_place),
                    target=target,
                    place_yaw=0.0,
                )
                if cand is None:
                    if is_normal_place:
                        print(
                            f"[ppp] place 失败 tip={float(tip_deg):.0f}° "
                            f"dyaw={np.rad2deg(float(dyaw)):.0f}° "
                            f"app↓={float(tip_down):.2f}",
                            flush=True,
                        )
                    continue
                q_rel = np.asarray(cand.jv_list[-1], dtype=float)
                ref = np.asarray(
                    EC_PLACE_REF_ARM if is_fallen_place else _PLACE_STYLE["q_arm"],
                    dtype=float,
                )
                # J2–J5 贴 REF；J6 可为对齐入格的转角（允许偏离 REF）
                arm_err = float(np.linalg.norm(q_rel[1:5] - ref[1:5]))
                if is_fallen_place:
                    if arm_err > np.deg2rad(50.0):
                        print(
                            f"[ppp] fallen place 非参考臂型 "
                            f"j2-5 Δ={np.rad2deg(arm_err):.1f}°，跳过",
                            flush=True,
                        )
                        continue
                    if grasp is not None:
                        tr_r, ax_r, R_hr, _ = _ec_hold_seat_metrics(
                            robot,
                            grasp,
                            target,
                            0.0,
                            q_rel,
                            jaw_close,
                        )
                        if (not _ec_hold_matches_seat(tr_r, ax_r)) or (
                            not _ec_hold_long_axis_up(R_hr)
                        ):
                            # 已走「腕转≈180→fallen REF」；此处不丢弃
                            print(
                                f"[ppp] fallen REF 松爪 "
                                f"tr={tr_r:.2f} axis={ax_r:.2f} "
                                f"up={float((R_hr @ np.array([0.0,1.0,0.0]))[2]):.2f} "
                                f"J6={np.rad2deg(q_rel[5]):.1f}°",
                                flush=True,
                            )
                place_mot = cand
                place_meta = (tip_deg, dyaw, upright, off_xy, arm_err, tip_down, q_rel)
                # 用松爪实际 TCP 刷新 app↓（桌面侧倾可能与候选略有差）
                try:
                    robot.goto_given_conf(q_rel, ee_values=float(jaw_close))
                    tip_down_real = max(
                        0.0, -float(robot.gl_tcp_rotmat[2, 2])
                    )
                except Exception:
                    tip_down_real = float(tip_down)
                on_desk_pl = str(place_surface).lower() == "desk"
                tag = (
                    "REF"
                    if is_fallen_place
                    else ("DESK-TILT" if on_desk_pl else "VERT")
                )
                print(
                    f"[ppp] place OK tip={tip_deg:.0f}° "
                    f"j5={np.rad2deg(q_rel[4]):.1f}° "
                    f"j6={np.rad2deg(q_rel[5]):.1f}° "
                    f"app↓={float(tip_down_real):.2f} "
                    f"armΔ={np.rad2deg(arm_err):.1f}° {tag}",
                    flush=True,
                )
                # 桌面缓冲：若仍几乎竖直顶放且还有更侧倾候选，继续试
                if (
                    on_desk_pl
                    and (not require_upright)
                    and float(tip_down_real) > 0.97
                    and abs(float(tip_deg)) >= 5.0
                ):
                    desk_vert_fallback = (cand, place_meta)
                    print(
                        "[ppp] desk 仍近竖直顶放，继续试更侧倾候选…",
                        flush=True,
                    )
                    place_mot = None
                    place_meta = None
                    continue
                break
            if (
                place_mot is None
                and desk_vert_fallback is not None
                and str(place_surface).lower() == "desk"
            ):
                place_mot, place_meta = desk_vert_fallback
                print("[ppp] desk 侧倾不可达，回退竖直顶放兜底", flush=True)
            if place_mot is None:
                side._reset_robot_hand(robot, jaw_open)
                continue

            # 入格 rot / 翻正标志在拼接前取出（+ 运算可能丢掉自定义属性）
            matched_seat_rot = None
            need_place_reorient = False
            try:
                matched_seat_rot = getattr(place_mot, "ec_matched_seat_rot", None)
                need_place_reorient = bool(
                    getattr(place_mot, "ec_need_place_reorient", False)
                )
            except Exception:
                matched_seat_rot = None
                need_place_reorient = False

            mot = pick_mot + place_mot
            if side.motion_penetrates_desk(
                robot, mot, desk_obstacle, jaw_fallback=jaw_open
            ):
                print("[ppp] 整段轨迹穿桌，丢弃", flush=True)
                side._reset_robot_hand(robot, jaw_open)
                continue

            # jaw_close 已在本 grasp 迭代内对齐为 jc_use（唯一权威）
            cs, ce, ri = side.find_carry_range(mot, jaw_close, jaw_open=jaw_open)
            # 持物段统一写成权威闭爪宽，避免 depart/place 拼接后短暂开爪/宽度跳动
            hold_end = int(ri) if ri is not None else int(ce)
            for i in range(int(cs), min(hold_end + 1, len(mot.ev_list))):
                mot.ev_list[i] = float(jaw_close)
            rel_tcp = _estimate_release_tcp_z(robot, mot, ri, jaw_close)
            # 桌面缓冲：松爪后不回 HOME，便于立刻规划下一抓；箱格仍回 HOME
            if str(place_surface).lower() != "desk":
                mot = _ec_append_return_home(
                    robot,
                    mot,
                    jaw_open,
                    use_rrt=bool(use_rrt),
                    obstacles=[],
                )
                side._reset_robot_hand(robot, jaw_open)
                robot.goto_given_conf(side.HOME_CONF, ee_values=float(jaw_open))
            else:
                side._reset_robot_hand(robot, jaw_open)
                q_end = np.asarray(mot.jv_list[-1], dtype=float)
                robot.goto_given_conf(q_end, ee_values=float(jaw_open))
            if place_meta is not None and str(place_surface).lower() != "desk":
                tip_deg, dyaw, *_rest, q_rel = place_meta
                if is_fallen_place:
                    pin_fallen_place_style()
                    lock_fallen_pick_hint(grasp_mode, grasp)
                elif is_normal_place and lock_style:
                    pin_normal_place_style()
                    _PLACE_STYLE["q_arm"] = np.asarray(q_rel, dtype=float).copy()
                    _PLACE_STYLE["tip_deg"] = float(tip_deg)
                    _PLACE_STYLE["dyaw"] = float(dyaw)
            print(
                f"[ppp] OK release_tcp_z={rel_tcp:.3f} jaw_close={jaw_close:.4f}",
                flush=True,
            )
            out = {
                "mot": mot,
                "carry_start": cs,
                "carry_end": ce,
                "release_idx": ri,
                "place_start": int(place_start_raw),
                "grasp_ac_pos": np.asarray(grasp.ac_pos, dtype=float).copy(),
                "grasp_ac_rotmat": np.asarray(grasp.ac_rotmat, dtype=float).copy(),
                "hover_z": float(EC_HIGH_HOVER_Z),
                "release_z": float(EC_RELEASE_TCP_FLOOR),
                "tcp_z_floor": float(EC_RELEASE_TCP_FLOOR),
                "approach": 0.0,
                "release_tcp_z": float(rel_tcp),
                "jaw_close": float(jaw_close),
                "tpl_tip": float(place_meta[0]) if place_meta else None,
                "tpl_dyaw": float(place_meta[1]) if place_meta else None,
                "tpl_q_rel": (
                    np.asarray(place_meta[-1], dtype=float).copy()
                    if place_meta
                    else None
                ),
            }
            if matched_seat_rot is not None:
                try:
                    out["matched_seat_rot"] = np.asarray(
                        matched_seat_rot, dtype=float
                    ).copy()
                except Exception:
                    pass
            out["need_place_reorient"] = bool(need_place_reorient)
            return out

    print("[ppp] pick/place 均失败", flush=True)
    return None


def plan_one_job(
    robot,
    target,
    part_infos,
    desk_obstacle,
    slot_id: int,
    grasp_topdown: str,
    grasp_side: str,
    use_rrt: bool,
    adjust_to_normal: bool = False,
    place_yaw: float = None,
    box_model=None,
    strategy_prefer: str = "auto",
    pipeline: str = None,
    place_xy=None,
    place_surface: str = "box",
    require_upright: bool = True,
    lock_style: bool = True,
    enforce_style: bool = True,
    tip_degs=None,
    seat_kind: str = "slot",
    buffer_yaw: float = 0.0,
    allow_relocate: bool = True,
    start_conf=None,
):
    """按姿态分流规划 pick→place。

    pipeline / state:
      fallen — 顶抓自选 + place=拧腕 REF
      normal — 仅顶抓 + place=夹爪竖直 ⊥XY 下探松爪
      inverted — 顶抓到桌面缓冲 remesh 成 fallen，再 fallen 入格
      buffer_* — 桌面暂存（挡路清障）
    start_conf — 可选起始关节（缓冲松爪后续抓，跳过回 HOME）
    """
    open_w, _ = side.jaw_widths_for_part(target)
    side.configure_gripper_for_demo(robot, jaw_max=max(open_w + 0.005, 0.02))
    side._reset_robot_hand(robot, open_w)
    q_plan0 = (
        np.asarray(start_conf, dtype=float).copy()
        if start_conf is not None
        else np.asarray(side.HOME_CONF, dtype=float).copy()
    )
    robot.goto_given_conf(q_plan0, ee_values=float(open_w))

    _ = place_yaw
    _ = adjust_to_normal
    yaw_use = 0.0
    gw = float(target.get("grip_width") or 0.0055)
    half = 0.5 * max(gw, 0.004)
    half_xy = (half, half)
    st = _norm_state(pipeline or target.get("state") or "fallen")

    seat_rot0 = seat_rot_undo_bake(target, place_yaw=yaw_use)
    if (
        str(seat_kind) in ("buffer_fallen", "buffer_clear")
        and place_xy is not None
    ):
        seat_pos0 = np.array(
            [float(place_xy[0]), float(place_xy[1]), 0.0], dtype=float
        )
        if str(seat_kind) == "buffer_fallen":
            seat_rot0 = np.eye(3)
    else:
        seat_pos0, _ = seat_pose_in_slot(
            slot_id, half_xy=half_xy, place_obj_rot=seat_rot0, lean_frac=0.0
        )
        seat_pos0, seat_rot0 = refine_seat_no_penetration(
            target["model"],
            seat_pos0,
            seat_rot0,
            box_model=box_model,
            slot_id=slot_id,
            half_xy=half_xy,
        )
    try:
        target["model"].pos = np.asarray(target["pos"], dtype=float)
        target["model"].rotmat = np.eye(3)
    except Exception:
        pass

    prefer = str(strategy_prefer or "auto").strip().lower()
    inhand_reorient = st == "inverted" and str(seat_kind) == "slot"
    use_fallen_tpl = (
        st in ("fallen", "inverted")
        and str(seat_kind) not in ("buffer_fallen", "buffer_clear")
    )
    # place 姿态分流：fallen/inverted→slot 用 REF；normal 竖直顶放
    if use_fallen_tpl:
        place_kind = "fallen_ref"
    elif st == "normal" and str(seat_kind) not in (
        "buffer_fallen",
        "buffer_clear",
    ):
        place_kind = "normal_vertical"
    else:
        place_kind = None  # desk buffer 等：不锁 REF

    # 姿态分流抓取
    if inhand_reorient:
        # inverted：侧抓 → 腕转 → fallen REF place
        pin_fallen_place_style()
        grasp_order = ["side"]
        tip_degs = (float(EC_PLACE_REF_TIP_DEG),)
        prefer = "side"
        print(
            "[inverted] 侧抓 + J6 翻正 → place=fallen REF",
            flush=True,
        )
    elif str(seat_kind) == "buffer_fallen":
        # inverted 设计：顶抓到缓冲区，侧倾松爪后 remesh 成 fallen（不转 J6）
        grasp_order = ["topdown"]
        if tip_degs is None:
            tip_degs = (45.0, 40.0, 50.0, 35.0)
        prefer = "topdown"
        print(
            f"[buffer] inverted→fallen：grasp_order={grasp_order} "
            f"desk tip≈45°侧倾（无 J6）",
            flush=True,
        )
    elif st == "normal" or str(seat_kind) == "buffer_clear":
        grasp_order = ["topdown"]
        if tip_degs is None:
            tip_degs = (
                (0.0, 5.0, 15.0)
                if str(seat_kind) == "buffer_clear"
                else EC_PLACE_TIP_DEGS_TOP
            )
        if place_kind == "normal_vertical":
            pin_normal_place_style()
        prefer = "topdown"
    elif use_fallen_tpl:
        # fallen：place=REF；顶抓若持物平躺/反着（约180°）过不了入格校验 → 再试侧抓
        pin_fallen_place_style()
        grasp_order = ["topdown", "side"]
        tip_degs = (float(EC_PLACE_REF_TIP_DEG),)
        prefer = "topdown"
    elif prefer in ("side", "topdown"):
        grasp_order = [prefer]
    else:
        grasp_order = ["topdown", "side"]

    if tip_degs is None:
        if use_fallen_tpl:
            tip_degs = (float(EC_PLACE_REF_TIP_DEG),)
        elif place_kind == "normal_vertical":
            tip_degs = EC_PLACE_TIP_DEGS_TOP
        else:
            tip_degs = (
                EC_PLACE_TIP_DEGS_FAST
                if not EC_VERBOSE
                else EC_PLACE_APPROACH_TIP_DEGS
            )

    adj_opts = upright_rot_options(target)
    # normal / fallen / inverted / 缓冲：只试规范朝向
    if st in ("normal", "fallen", "inverted") or str(seat_kind) in (
        "buffer_fallen",
        "buffer_clear",
    ):
        adj_opts = (
            [adj_opts[0]]
            if adj_opts
            else [(CANONICAL_ADJUST_NAME, UPRIGHT_OBJ_ROT.copy())]
        )

    print(
        f"[strategy] state={st} grasp_order={grasp_order} "
        f"tips={list(tip_degs)} seat={seat_kind} "
        f"place_kind={place_kind or '-'} "
        f"reorient={'J6' if inhand_reorient else '-'}",
        flush=True,
    )

    best = None  # (score, pkg_fields)
    grasp_cache = {}

    for gmode in grasp_order:
        if gmode not in grasp_cache:
            try:
                grasp_cache[gmode] = side.prepare_grasps_topdown_then_side(
                    robot,
                    target,
                    part_infos,
                    desk_obstacle,
                    topdown_pickle=grasp_topdown,
                    antipodal_pickle=grasp_side,
                    force_mode=gmode,
                )
            except Exception as e:
                print(f"[strategy] 抓取准备失败 mode={gmode}: {e}", flush=True)
                grasp_cache[gmode] = None
        prep = grasp_cache[gmode]
        if prep is None:
            continue
        part_grasps, jaw_open, jaw_close, grasp_z, grasp_mode, neighbor_models = prep
        # fallen：多试 grasp；全体再过穿模/姿态验收
        if use_fallen_tpl:
            stable = _rank_grasps_for_fallen_pick(part_grasps, top_n=6)
        else:
            stable = _ec_stable_sort_grasps(part_grasps)
        part_grasps = gg.GraspCollection(end_effector=robot.end_effector)
        for g in stable:
            part_grasps.append(g)
        side.configure_gripper_for_demo(
            robot, jaw_max=min(side.JAW_OPEN_MAX, max(float(jaw_open) + 0.005, 0.02))
        )
        jaw_open = float(min(float(jaw_open), float(robot.end_effector.jaw_range[1])))
        jaw_close = float(
            np.clip(
                float(jaw_close),
                side.JAW_CLOSE_MIN,
                float(robot.end_effector.jaw_range[1]) - 1e-4,
            )
        )
        gw_clamp = float(target.get("grip_width") or 0.005)
        # 硬验收：姿态合格 + 张/闭均不与零件穿模（修用户截图穿模）
        part_grasps, jaw_close = ec_sanitize_pick_grasps(
            robot,
            target["model"],
            part_grasps,
            jaw_open=jaw_open,
            jaw_close_seed=jaw_close,
            grasp_mode=grasp_mode,
            grip_width=gw_clamp,
        )
        if len(part_grasps) == 0:
            print(f"[strategy] pick 验收后无合格 grasp mode={grasp_mode}", flush=True)
            continue
        # 不统一覆盖 ee_values：每条 grasp 用外廓贴紧后的各自闭合宽
        print(
            f"[grasp] 合格 n={len(part_grasps)} jaw_close≈{jaw_close:.4f} "
            f"top_ac={np.round(np.asarray(part_grasps[0].ac_pos,float),4).tolist()} "
            f"mode={grasp_mode} "
            f"g0_jc={float(getattr(part_grasps[0], 'ee_values', jaw_close)):.4f}",
            flush=True,
        )
        robot.goto_given_conf(q_plan0, ee_values=float(jaw_open))
        obstacles = [] if grasp_mode == "side" else list(neighbor_models)

        # 最终入格姿态固定为规范正放（与加策略前一致），与规划用 adjust 解耦
        seat_pos_final = seat_pos0.copy()
        seat_rot_final = seat_rot0.copy()

        # 规范调姿即可；no_align 与 seat_rot 重复，跳过以免双倍 pick
        mode_ok = False
        adj_try = list(adj_opts) if adj_opts else [
            (CANONICAL_ADJUST_NAME, UPRIGHT_OBJ_ROT.copy())
        ]
        for adj_name, R_adj in adj_try:
            if mode_ok and adj_name != CANONICAL_ADJUST_NAME:
                break
            if (
                best is not None
                and str(best[1].get("strategy", "")).endswith(
                    "+" + CANONICAL_ADJUST_NAME
                )
                and adj_name != CANONICAL_ADJUST_NAME
            ):
                break

            align = R_adj is not None
            # 规划放置 TCP 用 undo-bake 竖直，与最终入格一致
            place_obj_rot = seat_rot_final.copy()
            try:
                target["model"].pos = np.asarray(target["pos"], dtype=float)
                target["model"].rotmat = np.eye(3)
            except Exception:
                pass

            # 边缘 fallen：顶抓不可达 → 侧抓。REF 臂型难从侧抓末态接入，改自由竖直入格。
            side_edge = (
                bool(use_fallen_tpl)
                and str(grasp_mode) == "side"
                and str(seat_kind) == "slot"
            )
            need_reorient = bool(inhand_reorient) or bool(side_edge)
            place_kind_use = place_kind
            use_fallen_tpl_use = use_fallen_tpl
            tip_degs_use = tip_degs
            if side_edge:
                place_kind_use = "normal_vertical"
                use_fallen_tpl_use = False
                tip_degs_use = EC_PLACE_TIP_DEGS_TOP
            print(
                f"[strategy] >>> grasp={grasp_mode} adjust={adj_name} "
                f"align={align} reorient={'J6' if need_reorient else '-'} "
                f"place_kind={place_kind_use or '-'} "
                f"(seat=undo-bake upright)",
                flush=True,
            )
            hit = _try_plan_pick_place(
                robot,
                target,
                part_grasps,
                jaw_open,
                jaw_close,
                grasp_mode,
                obstacles,
                desk_obstacle,
                place_obj_rot,
                use_rrt,
                slot_id=int(slot_id),
                place_xy=place_xy,
                tip_degs=tip_degs_use,
                place_surface=place_surface,
                require_upright=require_upright,
                lock_style=lock_style,
                enforce_style=enforce_style,
                use_fallen_tpl=use_fallen_tpl_use,
                place_kind=place_kind_use,
                inhand_reorient=need_reorient,
                start_conf=start_conf,
            )
            if hit is None:
                _ec_log(f"[strategy] fail grasp={grasp_mode} adjust={adj_name}")
                continue

            score = float(hit["release_tcp_z"])
            if grasp_mode == "side":
                score -= 0.005
            if adj_name == CANONICAL_ADJUST_NAME:
                score -= 0.015
            elif adj_name != "no_align":
                score += 0.025
            tag = f"{grasp_mode}+{adj_name}"
            print(
                f"[strategy] OK {tag} release_tcp_z={hit['release_tcp_z']:.3f}",
                flush=True,
            )
            # place 阶段已判定对齐的入格 rot（含 ±180° 择优），覆盖名义 seat
            seat_rot_use = np.asarray(seat_rot_final, dtype=float).copy()
            msr = hit.get("matched_seat_rot")
            if msr is not None:
                try:
                    msr_a = np.asarray(msr, dtype=float)
                    if msr_a.shape == (3, 3):
                        axis_m = msr_a @ np.array([0.0, 1.0, 0.0], dtype=float)
                        if float(axis_m[2]) >= 0.70:
                            seat_rot_use = msr_a.copy()
                            print(
                                "[strategy] 入格 rot←持物对齐解（防松爪 180°颠倒）",
                                flush=True,
                            )
                except Exception:
                    pass
            cand = {
                "mot": hit["mot"],
                "jaw_open": jaw_open,
                "jaw_close": float(hit.get("jaw_close", jaw_close)),
                "place_pose": (seat_pos_final.copy(), seat_rot_use.copy()),
                # 动画松爪入格用：避免被 TCP 反算覆盖成邻格
                "seat_pose_planned": (
                    seat_pos_final.copy(),
                    seat_rot_use.copy(),
                ),
                "seat_rot_locked": seat_rot_use.copy(),
                "place_model": None,
                "place_height": float(target.get("height") or 0.0),
                "place_obj_rot": seat_rot_use.copy(),
                "adjust_to_normal": True,
                "place_yaw": yaw_use,
                "grasp_mode": grasp_mode,
                "carry_start": hit["carry_start"],
                "carry_end": hit["carry_end"],
                "release_idx": hit["release_idx"],
                "place_start": hit.get("place_start"),
                "grasp_ac_pos": hit.get("grasp_ac_pos"),
                "grasp_ac_rotmat": hit.get("grasp_ac_rotmat"),
                "cleared_neighbors": hit.get("cleared_neighbors"),
                "target": target,
                "slot_id": int(slot_id),
                "half_xy": half_xy,
                "box_model": box_model,
                "strategy": tag,
                "release_tcp_z": float(hit["release_tcp_z"]),
                "hover_z": hit["hover_z"],
                "release_z": hit["release_z"],
                "used_tcp_z": hit["tcp_z_floor"],
                "used_app": hit["approach"],
                "seat_kind": str(seat_kind),
                "buffer_xy": (
                    np.asarray(place_xy, dtype=float).copy()
                    if place_xy is not None
                    else None
                ),
                "buffer_yaw": float(buffer_yaw),
                "pipeline_state": st,
                "need_place_reorient": bool(hit.get("need_place_reorient", False)),
            }
            if best is None or score < best[0]:
                best = (score, cand)
            mode_ok = True
            rel_z = float(hit["release_tcp_z"])
            # 规范调姿成功即停止本抓取模式的其它朝向
            if adj_name == CANONICAL_ADJUST_NAME:
                print(
                    "[strategy] 规范 rx+ 成功 → 入格姿态固定为正放，"
                    "不再换其它调姿朝向",
                    flush=True,
                )
                break
            if rel_z <= EC_GOOD_RELEASE_TCP_Z:
                print(
                    f"[strategy] 已达理想靠近 tcp_z≤{EC_GOOD_RELEASE_TCP_Z:.3f}，停止",
                    flush=True,
                )
                break
        if best is not None and best[1]["release_tcp_z"] <= EC_ACCEPT_RELEASE_TCP_Z:
            # 侧抓若已足够近可结束；否则继续试下一抓取模式争取更低
            if str(best[1].get("strategy", "")).startswith("side"):
                break
            if prefer != "auto":
                break

    if best is None:
        # 禁止远距离缓冲挪位（会丢掉数据集）。依次：原位 RRT → 局部微移≤3cm 再顶抓。
        if (
            bool(allow_relocate)
            and str(seat_kind) == "slot"
            and place_xy is None
            and st in ("fallen", "normal")
            and str(place_surface or "box").lower() != "desk"
        ):
            xy = np.asarray(target.get("pos", [0, 0, 0]), dtype=float)[:2]
            if not bool(use_rrt):
                print(
                    f"[retry] id={target.get('id')} state={st} "
                    f"原位 xy={np.round(xy, 3).tolist()} 顶/侧抓未解出；"
                    "默认不开 RRT，改试局部微移",
                    flush=True,
                )
            nudged = _local_nudge_for_topdown_ik(
                robot, target, grasp_topdown, max_shift=0.04, step=0.01
            )
            if nudged is not None and float(np.linalg.norm(np.asarray(nudged)[:2] - xy)) > 1e-4:
                xy1 = np.asarray(nudged, dtype=float)[:2]
                # 标注 fallen 若 bake 极端，微移后按规范 fallen 重烘焙（XY 仍在标注旁）
                if st == "fallen":
                    yaw0 = float(
                        target.get("yaw") or target.get("bake_yaw") or 0.0
                    )
                    remesh_part_as_fallen(target, xy1, yaw=yaw0, base=None)
                    xy1 = np.asarray(target.get("pos"), dtype=float)[:2]
                print(
                    f"[nudge] id={target.get('id')} 标注贴工作空间边界，"
                    f"局部微移 {np.round(xy, 3).tolist()} → "
                    f"{np.round(xy1, 3).tolist()}（≤4cm，非远缓冲）后重试顶抓",
                    flush=True,
                )
            elif nudged is not None:
                # IK 已够但 PPP 仍失败：不再假装微移
                nudged = None
                xy1 = None
            if nudged is not None:
                pick_snap = _snapshot_part_pose(target)
                pkg_n = plan_one_job(
                    robot,
                    target,
                    part_infos,
                    desk_obstacle,
                    slot_id=int(slot_id),
                    grasp_topdown=grasp_topdown,
                    grasp_side=grasp_side,
                    use_rrt=use_rrt,
                    adjust_to_normal=adjust_to_normal,
                    place_yaw=place_yaw,
                    box_model=box_model,
                    strategy_prefer="topdown",
                    pipeline=pipeline,
                    place_xy=None,
                    place_surface=place_surface,
                    require_upright=require_upright,
                    lock_style=lock_style,
                    enforce_style=enforce_style,
                    tip_degs=tip_degs,
                    seat_kind=seat_kind,
                    buffer_yaw=buffer_yaw,
                    allow_relocate=False,
                )
                pkg_n["local_nudge"] = True
                pkg_n["pick_desk_snap"] = pick_snap
                pkg_n["relocated_to_buffer"] = False
                return pkg_n
        raise RuntimeError(
            "PPP 多策略均失败（未挪远缓冲）："
            "顶抓超出工作空间 / 侧抓后无法接入放置 / 邻件阻挡。"
            "已试顶抓+RRT+局部≤4cm 微移。"
        )

    pkg = best[1]
    mot = pkg["mot"]
    jaw_close = pkg["jaw_close"]
    jaw_open = pkg["jaw_open"]
    carry_start, carry_end, release_idx = (
        pkg["carry_start"],
        pkg["carry_end"],
        pkg["release_idx"],
    )
    place_start = pkg.get("place_start")
    n_raw = len(mot.jv_list) if mot is not None else 0
    mot, remap = thin_motion(
        mot,
        robot,
        stride=ANIME_FRAME_STRIDE,
        keep_indices=(
            carry_start,
            carry_end,
            release_idx,
            (int(release_idx) + 1) if release_idx is not None else None,
            (int(release_idx) + 2) if release_idx is not None else None,
            (int(release_idx) + 3) if release_idx is not None else None,
            place_start,
            max(0, n_raw - 1),
        ),
        target_n=int(ANIME_TARGET_FRAMES),
    )

    def _map_idx(i):
        if i is None:
            return None
        if i in remap:
            return remap[i]
        keys = sorted(remap.keys())
        best_i = min(keys, key=lambda k: abs(k - int(i)))
        return remap[best_i]

    pkg["mot"] = mot
    pkg["carry_start"] = _map_idx(carry_start)
    pkg["carry_end"] = _map_idx(carry_end)
    pkg["release_idx"] = _map_idx(release_idx)
    pkg["place_start"] = _map_idx(place_start)
    _ensure_visible_jaw_close(pkg)
    # 桌面缓冲：用松爪 TCP 反算螺丝 XY，供 remesh / seat 对齐松开位置
    if str(pkg.get("seat_kind") or "") in ("buffer_fallen", "buffer_clear"):
        ri = pkg.get("release_idx")
        if ri is not None and 0 <= int(ri) < len(mot.jv_list):
            robot.goto_given_conf(
                jnt_values=mot.jv_list[int(ri)],
                ee_values=float(jaw_open),
            )
            if _sync_place_pose_from_release_tcp(pkg, robot):
                xy_rel = np.asarray(pkg["place_pose"][0], dtype=float)[:2]
                pkg["release_obj_xy"] = xy_rel.copy()
                pkg["buffer_xy"] = xy_rel.copy()
                print(
                    f"[plan] buffer 落点←松爪 obj_xy="
                    f"{np.round(xy_rel, 4).tolist()}",
                    flush=True,
                )
    place_pose = pkg["place_pose"]

    print(
        f"[plan] BEST strategy={pkg['strategy']} mode={pkg['grasp_mode']} "
        f"frames={len(mot)} slot={slot_id} "
        f"seat_xy={np.round(place_pose[0][:2], 4).tolist()} "
        f"seat_z={place_pose[0][2]:.4f} "
        f"release_tcp_z={pkg['release_tcp_z']:.3f} "
        f"hover_z={pkg['hover_z']:.3f} "
        f"carry=[{pkg['carry_start']}..{pkg['carry_end']}] "
        f"release={pkg['release_idx']}",
        flush=True,
    )
    return pkg


def _export_segments_from_job_pkgs(job_pkgs: list) -> list[dict]:
    """把 EC 规划结果转成 ``side.export_planned_trajectory`` 需要的 segment 列表。"""
    segs = []
    n = len(job_pkgs)
    for i, pkg in enumerate(job_pkgs, 1):
        tgt = pkg.get("target") or {}
        mot = pkg["mot"]
        n_frames = len(mot.jv_list)
        segs.append(
            {
                "index": i,
                "total": n,
                "object": str(tgt.get("class") or f"id={tgt.get('id', i)}"),
                "object_id": int(tgt.get("id") or i),
                "slot_id": int(pkg.get("slot_id") or 0),
                "seat_kind": str(pkg.get("seat_kind") or "slot"),
                "grasp_mode": str(pkg.get("grasp_mode") or ""),
                "do_place": True,
                "mot": mot,
                "jaw_open": pkg["jaw_open"],
                "jaw_close": pkg["jaw_close"],
                "carry_start": int(pkg.get("carry_start") or 0),
                "carry_end": int(pkg.get("carry_end") if pkg.get("carry_end") is not None else n_frames - 1),
                "release_idx": pkg.get("release_idx"),
            }
        )
    return segs


def job_pkgs_from_cached_traj(traj_path, robot, part_infos, jobs, by_id, box_model):
    """用已导出的关节轨迹拼回 job_pkgs，跳过规划直接播动画。"""
    with open(traj_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    segs = list(payload.get("segments") or [])
    if not segs:
        raise RuntimeError(f"轨迹为空：{traj_path}")

    stages = []
    for job in jobs:
        tid = int(job["object_id"])
        if tid not in by_id:
            continue
        target = by_id[tid]
        st = _norm_state(target.get("state") or job.get("state") or "normal")
        if st == "inverted":
            stages.append((job, target, "buffer_fallen"))
            stages.append((job, target, "slot"))
        else:
            stages.append((job, target, "slot"))
    if len(stages) != len(segs):
        print(
            f"[replay] 段数不一致 stages={len(stages)} segs={len(segs)}，按较短对齐",
            flush=True,
        )
    n = min(len(stages), len(segs))
    snaps = {}
    pkgs = []
    for i in range(n):
        job, target, seat_kind = stages[i]
        seg = segs[i]
        tid = int(target.get("id"))
        if tid not in snaps:
            snaps[tid] = _snapshot_part_pose(target)
        jv_list = [np.asarray(q, dtype=float).ravel() for q in (seg.get("jv_list") or [])]
        ev_raw = list(seg.get("ev_list") or [])
        ev_list = []
        for k in range(len(jv_list)):
            v = ev_raw[k] if k < len(ev_raw) else None
            ev_list.append(None if v is None else float(v))
        jaw_lo, jaw_hi = (0.0, float(side.JAW_OPEN_MAX))
        try:
            jaw_lo = float(robot.end_effector.jaw_range[0])
            jaw_hi = float(robot.end_effector.jaw_range[1])
        except Exception:
            pass
        jaw_open = float(np.clip(seg.get("jaw_open", jaw_hi), jaw_lo, jaw_hi))
        jaw_close = float(
            np.clip(seg.get("jaw_close", 0.006), jaw_lo + 1e-4, max(jaw_open - 1e-4, jaw_lo))
        )
        ev_list = [
            None if v is None else float(np.clip(v, jaw_lo, jaw_hi)) for v in ev_list
        ]
        mot = MotionData(robot)
        mot.extend(jv_list=jv_list, ev_list=ev_list, mesh_list=[])
        gw = float(target.get("grip_width") or 0.0055)
        half = 0.5 * max(gw, 0.004)
        slot_id = int(job["slot_id"])
        if seat_kind == "buffer_fallen":
            buf = pick_buffer_xy(part_infos, target)
            place_pos = np.array([float(buf[0]), float(buf[1]), 0.02], dtype=float)
            place_rot = np.eye(3)
        else:
            seat_rot = seat_rot_undo_bake(target, place_yaw=0.0)
            place_pos, place_rot = seat_pose_in_slot(
                slot_id, half_xy=(half, half), place_obj_rot=seat_rot, lean_frac=0.0
            )
        n_frames = len(jv_list)
        release_idx = seg.get("release_idx")
        if release_idx is not None:
            release_idx = int(release_idx)
        pkg = {
            "mot": mot,
            "target": target,
            "slot_id": slot_id,
            "seat_kind": seat_kind,
            "jaw_open": jaw_open,
            "jaw_close": jaw_close,
            "carry_start": int(seg.get("carry_start") or 0),
            "carry_end": int(seg.get("carry_end") if seg.get("carry_end") is not None else max(0, n_frames - 1)),
            "release_idx": release_idx,
            "place_start": None,
            "place_pose": (place_pos.copy(), np.asarray(place_rot, dtype=float).copy()),
            "seat_pose_planned": (place_pos.copy(), np.asarray(place_rot, dtype=float).copy()),
            "grasp_mode": str(seg.get("grasp_mode") or "topdown"),
            "strategy": "cached_traj",
            "grasp_ac_pos": None,
            "grasp_ac_rotmat": None,
            "buffer_xy": place_pos[:2].copy() if seat_kind == "buffer_fallen" else None,
            "buffer_yaw": float(target.get("yaw") or target.get("bake_yaw") or 0.0),
            "place_yaw": 0.0,
            "release_tcp_z": 0.05,
            "hover_z": 0.14,
            "desk_snap": snaps[tid],
            "skip_pre_anime_restore": seat_kind == "slot" and _norm_state(
                target.get("state") or ""
            ) == "inverted",
            "half_xy": (half, half),
            "box_model": box_model,
        }
        pkgs.append(pkg)
    print(
        f"[replay] 已加载 {len(pkgs)} 段 / {len({int(p['target']['id']) for p in pkgs})} 件",
        flush=True,
    )
    return pkgs


def apply_job_final(robot, pkg, base):
    """无动画：先到松爪帧对齐落点，再回末帧（HOME）并入格。"""
    mot = pkg["mot"]
    jaw_open = pkg["jaw_open"]
    ri = pkg.get("release_idx")
    if ri is not None and 0 <= int(ri) < len(mot.jv_list):
        j_rel = mot.jv_list[int(ri)]
        robot.goto_given_conf(jnt_values=j_rel, ee_values=float(jaw_open))
        _sync_place_pose_from_release_tcp(pkg, robot)
    jv = mot.jv_list[-1]
    ev = mot.ev_list[-1] if len(mot.ev_list) else jaw_open
    if len(robot.end_effector.oiee_list) > 0:
        robot.end_effector.release_all()
    seat_in_slot(pkg, base, robot)
    robot.goto_given_conf(
        jnt_values=jv, ee_values=float(ev if ev is not None else jaw_open)
    )


def run_space_anime(base, robot, job_pkgs, auto_play=False):
    """空格逐帧播放；auto_play=True 时由任务定时器自动推进。"""
    PRE, CARRY, DONE = 0, 1, 2

    class AnimeData(object):
        __slots__ = (
            "job_i",
            "counter",
            "phase",
            "last_mesh",
            "last_idx",
            "finished_all",
            "pkg",
            "adjusted",
            "carry_on_base",
            "flip_R0",
            "flip_R1",
            "flip_i0",
            "flip_i1",
            "initial_hold_ticks",
        )

        def __init__(self):
            self.job_i = 0
            self.counter = 0
            self.phase = PRE
            self.last_mesh = None
            self.last_idx = -1
            self.finished_all = False
            self.pkg = None
            self.adjusted = False
            self.carry_on_base = False
            self.flip_R0 = None
            self.flip_R1 = None
            self.flip_i0 = None
            self.flip_i1 = None
            self.initial_hold_ticks = int(
                float(side.AUTO_PLAY_INITIAL_HOLD_SEC) / float(side.ANIMATION_INTERVAL)
            )

    ad = AnimeData()

    def _release_last_mesh():
        side.release_model_collection(ad.last_mesh)
        ad.last_mesh = None
        ad.last_idx = -1

    def _start_job(ji: int) -> bool:
        if ji >= len(job_pkgs):
            return False
        ad.pkg = job_pkgs[ji]
        ad.job_i = ji
        ad.counter = 0
        ad.phase = PRE
        ad.adjusted = False
        ad.carry_on_base = False
        ad.flip_R0 = None
        ad.flip_R1 = None
        ad.flip_i0 = None
        ad.flip_i1 = None
        _release_last_mesh()
        # 局部微移开抓：开抓前同步到轨迹对应 XY（仍在标注附近，非远缓冲）
        if ad.pkg.get("pick_desk_snap") is not None and (
            ad.pkg.get("local_nudge") or ad.pkg.get("relocated_to_buffer")
        ):
            _restore_part_pose(
                ad.pkg["target"], ad.pkg["pick_desk_snap"], base=base
            )
            tag = "局部微移" if ad.pkg.get("local_nudge") else "缓冲"
            print(
                f"[anime] pick-sync id={ad.pkg['target'].get('id')} "
                f"→ {tag}开抓位 "
                f"{np.round(ad.pkg['pick_desk_snap']['info_pos'][:2], 3).tolist()}",
                flush=True,
            )
        side._reset_robot_hand(robot, ad.pkg["jaw_open"])
        try:
            robot.end_effector.change_jaw_width(float(ad.pkg["jaw_open"]))
        except Exception:
            pass
        mot = ad.pkg["mot"]
        jv0 = np.asarray(mot.jv_list[0], dtype=float)
        ev0 = mot.ev_list[0] if len(mot.ev_list) else ad.pkg["jaw_open"]
        open_w = float(ad.pkg["jaw_open"])
        # 上一件轨迹末应已回 HOME；仅在明显偏离时补一段关节过渡（不瞬移）
        try:
            q_now = np.asarray(robot.get_jnt_values(), dtype=float)
        except Exception:
            q_now = None
        if q_now is not None and float(np.linalg.norm(q_now - jv0)) > 0.04:
            bridge = _ec_joint_lerp_motion(
                robot, q_now, jv0, open_w, n_steps=12
            )
            for bq, be in zip(bridge.jv_list, bridge.ev_list):
                robot.goto_given_conf(
                    jnt_values=bq, ee_values=float(be if be is not None else open_w)
                )
        else:
            robot.goto_given_conf(
                jnt_values=jv0,
                ee_values=float(ev0 if ev0 is not None else open_w),
            )
        ad.last_mesh = robot.gen_meshmodel(
            toggle_tcp_frame=False, toggle_jnt_frames=False
        )
        ad.last_mesh.attach_to(base)
        ad.last_idx = 0
        tgt = ad.pkg["target"]
        print(
            f"[anime] job {ji+1}/{len(job_pkgs)} id={tgt.get('id')} "
            f"slot={ad.pkg['slot_id']} frames={len(mot)} "
            f"strategy={ad.pkg.get('strategy')} → "
            f"{'自动播放' if auto_play else '按 [Space] 推进'}",
            flush=True,
        )
        return True

    if not _start_job(0):
        print("[anime] 无可用轨迹", flush=True)
        return

    def update(task_obj):
        if ad.finished_all:
            return task_obj.done
        if auto_play and ad.initial_hold_ticks > 0:
            ad.initial_hold_ticks -= 1
            return task_obj.again
        if not auto_play and not base.inputmgr.keymap["space"]:
            return task_obj.again

        # 消费空格（避免连发）
        if base.inputmgr.keymap["space"]:
            base.inputmgr.keymap["space"] = False
        pkg = ad.pkg
        mot = pkg["mot"]
        n_frames = len(mot)
        ad.counter += 1

        if ad.counter >= n_frames:
            _release_last_mesh()
            if ad.phase != DONE:
                seat_in_slot(pkg, base, robot)
                ad.phase = DONE
            # 轨迹末帧应已是 HOME
            try:
                q_end = np.asarray(mot.jv_list[-1], dtype=float)
                robot.goto_given_conf(
                    jnt_values=q_end, ee_values=float(pkg["jaw_open"])
                )
            except Exception:
                pass
            print(
                f"[anime] job {ad.job_i+1} 完成 → slot {pkg['slot_id']}，已回原位",
                flush=True,
            )
            if not _start_job(ad.job_i + 1):
                ad.finished_all = True
                print("[anime] 全部完成，关闭窗口退出", flush=True)
                return task_obj.done
            return task_obj.again

        if ad.counter == ad.last_idx:
            return task_obj.again

        jv = mot.jv_list[ad.counter]
        ev = mot.ev_list[ad.counter] if ad.counter < len(mot.ev_list) else None
        held_model = pkg["target"]["model"]
        jaw_close = float(pkg["jaw_close"])
        jaw_open = float(pkg["jaw_open"])
        carry_start = pkg["carry_start"]
        release_idx = pkg["release_idx"]
        place_start = pkg.get("place_start")
        place_pose = pkg["place_pose"]
        holding_now = len(robot.end_effector.oiee_list) > 0
        # 动画用 base 跟随零件，不 hold → CARRY 可安全设闭合宽
        if holding_now:
            robot.goto_given_conf(jnt_values=jv)
        elif ad.phase == CARRY or (
            carry_start is not None and ad.counter >= int(carry_start)
        ):
            robot.goto_given_conf(jnt_values=jv, ee_values=jaw_close)
        elif ev is not None:
            robot.goto_given_conf(jnt_values=jv, ee_values=float(ev))
        else:
            robot.goto_given_conf(jnt_values=jv, ee_values=float(jaw_open))

        def _pose_from_grasp_tcp():
            """由当前 TCP + grasp 算零件世界位姿（与夹爪刚性固连）。"""
            ac_p = pkg.get("grasp_ac_pos")
            ac_r = pkg.get("grasp_ac_rotmat")
            if ac_p is None or ac_r is None:
                return None, None
            ac_p = np.asarray(ac_p, dtype=float).reshape(3)
            ac_r = np.asarray(ac_r, dtype=float)
            tcp = np.asarray(robot.gl_tcp_pos, dtype=float)
            Rm = np.asarray(robot.gl_tcp_rotmat, dtype=float)
            R_obj = Rm @ ac_r.T
            pos = tcp - R_obj @ ac_p
            return pos, R_obj

        def _follow_tcp_on_base():
            """动画搬运：零件始终挂在 base，每帧按 grasp 贴 TCP（爪+件一起动）。

            不用 robot.hold：oiee 相对位姿 + gen_meshmodel 常导致「爪空、螺丝留桌面」。
            禁止零件单独 slerp（那会看起来飞出夹爪自己转）。
            """
            pos, R_obj = _pose_from_grasp_tcp()
            if pos is None:
                return False
            if len(robot.end_effector.oiee_list) > 0:
                try:
                    robot.end_effector.release_all()
                except Exception:
                    pass
            try:
                held_model.detach()
            except Exception:
                pass
            held_model.pos = pos
            held_model.rotmat = R_obj
            held_model.attach_to(base)
            return True

        if carry_start is not None and ad.phase == PRE and ad.counter >= carry_start:
            if pkg.get("grasp_ac_pos") is None:
                tcp = np.asarray(robot.gl_tcp_pos, dtype=float)
                rm_tcp = np.asarray(robot.gl_tcp_rotmat, dtype=float)
                pos0 = np.asarray(held_model.pos, dtype=float)
                r0 = np.asarray(held_model.rotmat, dtype=float)
                pkg["grasp_ac_pos"] = r0.T @ (tcp - pos0)
                pkg["grasp_ac_rotmat"] = r0.T @ rm_tcp
            robot.goto_given_conf(jnt_values=jv, ee_values=jaw_close)
            try:
                robot.end_effector.change_jaw_width(jaw_close)
            except Exception:
                pass
            ok_f = _follow_tcp_on_base()
            ad.phase = CARRY
            ad.carry_on_base = True
            ad.flip_R0 = None
            ad.flip_R1 = None
            ad.flip_i0 = None
            ad.flip_i1 = None
            print(
                f"[anime] carry-follow id={pkg['target'].get('id')} "
                f"jaw_close={jaw_close:.4f} ok={ok_f}",
                flush=True,
            )

        if (
            place_start is not None
            and ad.phase == CARRY
            and (not ad.adjusted)
            and ad.counter >= int(place_start)
        ):
            ad.adjusted = True
            upright = abs(float(held_model.rotmat[2, 1]))
            print(
                f"[anime] place-carry id={pkg['target'].get('id')} "
                f"upright={upright:.2f} (爪+件随TCP/J6，不单独转零件)",
                flush=True,
            )

        # CARRY 全程：零件按 grasp 贴 TCP（J6 腕转时一起转）
        if ad.phase == CARRY and pkg.get("grasp_ac_pos") is not None:
            _follow_tcp_on_base()

        if (
            release_idx is not None
            and ad.phase == CARRY
            and ad.counter >= release_idx
        ):
            robot.goto_given_conf(jnt_values=jv, ee_values=jaw_close)
            _follow_tcp_on_base()
            # 箱格入格：用规划好的格位/竖直姿态，勿用松爪 TCP 覆盖
            # （拧腕后 TCP 反算常偏到邻格，看起来「没对齐」）
            planned = pkg.get("seat_pose_planned") or pkg.get("place_pose")
            if (
                str(pkg.get("seat_kind") or "slot") == "slot"
                and planned is not None
            ):
                try:
                    pp, pr = planned
                    pkg["place_pose"] = (
                        np.asarray(pp, dtype=float).copy(),
                        np.asarray(pr, dtype=float).copy(),
                    )
                    axis = np.asarray(pr, dtype=float) @ np.array(
                        [0.0, 1.0, 0.0], dtype=float
                    )
                    if float(axis[2]) >= 0.70:
                        pkg["seat_rot_locked"] = np.asarray(pr, dtype=float).copy()
                except Exception:
                    _sync_place_pose_from_release_tcp(pkg, robot)
            else:
                pos_now, R_now = _pose_from_grasp_tcp()
                if pos_now is not None:
                    pkg["place_pose"] = (pos_now.copy(), R_now.copy())
                _sync_place_pose_from_release_tcp(pkg, robot)
            seat_in_slot(pkg, base, robot)
            robot.goto_given_conf(jnt_values=jv, ee_values=jaw_open)
            try:
                robot.end_effector.change_jaw_width(jaw_open)
            except Exception:
                pass
            ad.phase = DONE
            place_pose = pkg["place_pose"]
            print(
                f"[anime] place slot={pkg['slot_id']} "
                f"@ {np.round(place_pose[0], 4).tolist()} "
                f"(规划格位) → 继续回原位",
                flush=True,
            )

        _release_last_mesh()
        ad.last_mesh = robot.gen_meshmodel(
            toggle_tcp_frame=False, toggle_jnt_frames=False
        )
        ad.last_mesh.attach_to(base)
        ad.last_idx = ad.counter
        return task_obj.again

    # 先刷几帧再进 run，避免规划卡死后首屏仍灰屏「未响应」
    _ec_pump_ui(base, n=4)
    if not auto_play:
        print(
            "[sim] 窗口应已可交互；点一下 Panda 窗口再按 [Space] 推进",
            flush=True,
        )
    base.taskMgr.doMethodLater(
        side.ANIMATION_INTERVAL, update, "ec_bin_space_anime", appendTask=True
    )
    base.run()


def main():
    p = argparse.ArgumentParser(description="EC Bugle screw → 料盘分格仿真")
    p.add_argument("--task", default=DEFAULT_TASK, help="智能体任务 JSON")
    p.add_argument("--sample-dir", default=DEFAULT_SAMPLE, help="数据集样本目录 dataset_learn/0502")
    p.add_argument("--grasp-pickle", default=DEFAULT_GRASP_TOPDOWN)
    p.add_argument("--side-grasp-pickle", default=DEFAULT_GRASP_SIDE)
    p.add_argument("--no-anime", action="store_true", help="只规划不播动画")
    p.add_argument(
        "--export-traj",
        default="",
        help="规划成功后把关节轨迹导出到该 JSON 并退出（供 tiaozhanbei/real 真机执行模块使用）",
    )
    p.add_argument(
        "--load-traj",
        default="",
        help="直接播放已导出的关节轨迹 JSON，跳过规划",
    )
    p.add_argument(
        "--joint-ranges",
        default="",
        help="收紧 6 个关节范围（度），JSON 形如 [[lo,hi],...] 或指向该内容的文件；"
        "导出真机轨迹时用它把搜索限制在真机软限位内",
    )
    p.add_argument("--auto-play", action="store_true", help="自动推进动画帧；默认仍可用空格手动推进")
    p.add_argument("--wait-start-file", default="", help="UI 贴合 Panda 窗口后创建该文件，仿真再开始播放")
    p.add_argument("--preview-window-only", action="store_true", help="仅显示 EC HOME 初始场景窗口，不进入规划/动画")
    p.add_argument("--ready-file", default="", help="EC HOME 预览窗口准备好后写入该文件")
    p.add_argument("--no-rrt", action="store_true", help="禁用 RRT（默认已禁用，兼容旧参数）")
    p.add_argument("--rrt", action="store_true", help="启用 RRT（更慢，接近困难时再开）")
    p.add_argument("--verbose", action="store_true", help="打印规划细节")
    p.add_argument("--stripe", default="vert_gray")
    p.add_argument("--max-jobs", type=int, default=0, help="只跑前 N 个 job；0=全部")
    p.add_argument(
        "--object-id",
        type=int,
        nargs="+",
        default=None,
        help="只跑指定零件 id，例如 --object-id 6 或 --object-id 4 5 6",
    )
    p.add_argument(
        "--strategy",
        default="auto",
        choices=("auto", "side", "topdown"),
        help="fallen 策略: auto/side/topdown；normal/inverted 第一段强制顶抓",
    )
    p.add_argument(
        "--pick-order",
        default="inv_last",
        choices=("inv_last", "slot", "json", "easy"),
        help="顺序: inv_last=先fallen/normal再inverted(默认); "
        "slot=纯slot_id; json=任务列表序; easy=易抓优先(inverted靠后)",
    )
    args = p.parse_args()

    global EC_VERBOSE, _PLACE_STYLE, _FALLEN_TPL, _EC_UI_MANAGED
    EC_VERBOSE = bool(args.verbose)
    # UI 托管：正式窗口保持屏外隐藏，直到 UI 贴合后再显示并播放
    _EC_UI_MANAGED = bool(getattr(args, "wait_start_file", "") or "")

    # 每次运行重置；place 姿态按 pipeline 分流，不在此全局锁 fallen REF
    _PLACE_STYLE = {
        "tip_deg": 0.0,
        "dyaw": 0.0,
        "q_arm": np.asarray(EC_PLACE_NATURAL_CONF, dtype=float).copy(),
        "locked": False,
        "kind": None,
    }
    _FALLEN_TPL = {
        "locked": False,
        "grasp_mode": "topdown",
        "ac_pos": None,
        "ac_rotmat": None,
    }
    ann_path = os.path.join(args.sample_dir, "annotations.json")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(
            f"缺少 {ann_path}，请先运行: python tiaozhanbei/dataset_gen/generate_ec_dataset.py"
        )
    ann = load_json(ann_path)
    if args.preview_window_only:
        side.configure_hidden_start_window(args)
        base = wd.World(
            cam_pos=[1.1, -0.85, 0.8],
            lookat_pos=[0.12, -0.05, 0.05],
            w=1280,
            h=720,
        )
        sys.modules["__main__"].base = base
        desk, legs = gen_desk()
        desk.attach_to(base)
        for leg in legs:
            leg.attach_to(base)
        attach_desk_stripe(base, style=args.stripe, seed=0)
        box = load_box_model()
        box.attach_to(base)
        for info in load_parts_from_ec_ann(ann):
            info["model"].attach_to(base)
        robot = PantheraHTSglArm(enable_cc=True)
        if getattr(args, "joint_ranges", ""):
            side.apply_joint_ranges_deg(robot, side.parse_joint_ranges_arg(args.joint_ranges))
        robot.goto_given_conf(side.HOME_CONF)
        side.render_planning_preview(base, robot)
        ready_file = getattr(args, "ready_file", "") or ""
        if ready_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(ready_file)), exist_ok=True)
                with open(ready_file, "w", encoding="utf-8") as f:
                    f.write("ready\n")
            except Exception as e:
                print(f"[preview] ready-file 写入失败：{e}", flush=True)
        print("[preview] EC live HOME scene ready", flush=True)
        base.run()
        return
    if os.path.isfile(args.task):
        task_data = load_json(args.task)
    else:
        print(f"[task] 未找到 {args.task}，将用 object_id→slot_id 默认映射", flush=True)
        task_data = {}

    jobs = parse_jobs(task_data, ann)
    if str(args.pick_order) == "inv_last":
        jobs = rank_jobs_inverted_last(jobs)
    elif str(args.pick_order) == "slot":
        jobs = sorted(jobs, key=lambda j: (int(j["slot_id"]), int(j["object_id"])))
    if args.object_id:
        want = {int(x) for x in args.object_id}
        jobs = [j for j in jobs if int(j["object_id"]) in want]
        print(f"[task] 仅 object_id={sorted(want)} → {len(jobs)} 个 job", flush=True)
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    use_rrt = bool(args.rrt) and (not args.no_rrt)
    print(
        f"[task] n_jobs={len(jobs)} sample={args.sample_dir} "
        f"pick_order={args.pick_order} rrt={use_rrt} verbose={EC_VERBOSE}",
        flush=True,
    )
    for j in jobs:
        print(
            f"[task]   slot={j['slot_id']} obj={j['object_id']} "
            f"state={j['state']}",
            flush=True,
        )

    # 略缩小窗口：规划阶段长时间占主线程时，1920 全屏+卡通描边更易假死灰屏
    side.configure_hidden_start_window(args)
    base = wd.World(
        cam_pos=[1.1, -0.85, 0.8],
        lookat_pos=[0.12, -0.05, 0.05],
        w=1280,
        h=720,
    )
    sys.modules["__main__"].base = base
    base.toggle_mesh = False
    try:
        base.win.setCloseRequestEvent("ec-keep-alive")
        base.accept(
            "ec-keep-alive",
            lambda: print("[sim] 规划中忽略关窗，继续跑完 12 件", flush=True),
        )
    except Exception:
        pass
    mgm.gen_frame(ax_length=0.1).attach_to(base)
    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style=args.stripe, seed=0)
    desk_obstacle = side.make_desk_obstacle()

    box = load_box_model()
    box.attach_to(base)
    print(
        f"[scene] box @ origin_xy={BOX_ORIGIN_XY.tolist()} "
        f"lift_z={BOX_LIFT_Z:.3f} size={BOX_SIZE_M.tolist()}",
        flush=True,
    )
    print("[scene] 加载 12 个螺丝网格…", flush=True)
    part_infos = load_parts_from_ec_ann(ann)
    by_id = {int(info["id"]): info for info in part_infos}
    for info in part_infos:
        info["model"].attach_to(base)

    print("[scene] 加载机械臂…", flush=True)
    robot = PantheraHTSglArm(enable_cc=True)
    # 单臂构造默认 jaw_range=[0, 0.012]；演示开口是 55mm。
    # 规划路径会在每件里 configure；回放跳过规划，必须在这里先放开。
    side.configure_gripper_for_demo(robot, jaw_max=side.JAW_OPEN_MAX)
    print("[scene] 机械臂就绪", flush=True)
    if getattr(args, "joint_ranges", ""):
        side.apply_joint_ranges_deg(robot, side.parse_joint_ranges_arg(args.joint_ranges))
    robot.goto_given_conf(side.HOME_CONF)
    if _EC_UI_MANAGED:
        # UI 已在右侧显示 HOME 预览窗口：正式窗口规划期间不刷帧，避免黑屏跳出
        preview_mesh = robot.gen_meshmodel(
            toggle_tcp_frame=False, toggle_jnt_frames=False
        )
        preview_mesh.attach_to(base)
        print("[sim] 场景已挂上；规划期间保持 UI 的 HOME 预览画面", flush=True)
    else:
        # 规划阶段先渲染 HOME 场景，避免耗时规划时 Panda 窗口停在黑屏。
        preview_mesh = side.render_planning_preview(base, robot)
        print(
            "[sim] 场景已挂上；随后规划会在当前 HOME 画面后继续执行",
            flush=True,
        )

    job_pkgs = []
    ok_n = fail_n = 0
    total_n = len(jobs)
    load_traj = str(getattr(args, "load_traj", "") or "").strip()
    if load_traj:
        # 缓存轨迹是 12 件全量；不跟当前口令子集走，避免只播前几段。
        jobs = rank_jobs_inverted_last(
            [
                {
                    "object_id": int(obj["id"]),
                    "slot_id": int(obj.get("slot_id", int(obj["id"]) - 1)),
                    "state": _norm_state(obj.get("state") or "normal"),
                }
                for obj in (ann.get("objects") or [])
                if obj.get("id") is not None
            ]
        )
        total_n = len(jobs)
        print(
            f"[replay] 使用缓存轨迹，跳过规划 → {load_traj} "
            f"(按标注 {total_n} 件 inv_last)",
            flush=True,
        )
        job_pkgs = job_pkgs_from_cached_traj(
            load_traj, robot, part_infos, jobs, by_id, box
        )
        ok_n = len({int(p["target"]["id"]) for p in job_pkgs})
        fail_n = 0
        pending = []
        planned_i = 0
    else:
        pending = list(jobs)
        planned_i = 0
    while pending:
        _ec_pump_ui(base, n=1)
        if str(args.pick_order) == "easy":
            pending = rank_jobs_by_pick_ease(pending, part_infos)
            order_tag = "easy"
        elif str(args.pick_order) == "inv_last":
            pending = rank_jobs_inverted_last(pending)
            order_tag = "inv_last"
        elif str(args.pick_order) == "slot":
            pending = sorted(
                pending, key=lambda j: (int(j["slot_id"]), int(j["object_id"]))
            )
            order_tag = "slot"
        else:
            order_tag = "json"
        job = pending.pop(0)
        planned_i += 1
        tid = int(job["object_id"])
        if tid not in by_id:
            print(f"[skip] object_id={tid} 不在场景中", flush=True)
            fail_n += 1
            continue
        target = by_id[tid]
        desk_snap = _snapshot_part_pose(target)
        # 优先用场景当前 state（缓冲 remesh 后可能已是 fallen）
        st = _norm_state(target.get("state") or job.get("state") or "normal")
        print(
            f"\n=== plan {planned_i}/{total_n} slot={job['slot_id']} "
            f"id={tid} state={st} (order={order_tag}) ===",
            flush=True,
        )
        try:
            common = dict(
                robot=robot,
                target=target,
                part_infos=part_infos,
                desk_obstacle=desk_obstacle,
                slot_id=int(job["slot_id"]),
                grasp_topdown=args.grasp_pickle,
                grasp_side=args.side_grasp_pickle,
                use_rrt=use_rrt,
                adjust_to_normal=bool(job.get("adjust_to_normal")),
                place_yaw=job.get("place_yaw"),
                box_model=box,
            )

            def _plan_slot_for_state(state_name):
                if state_name == "inverted":
                    # 设计：顶抓到缓冲区 → remesh fallen → 再按 fallen 入格
                    buf_xy = pick_buffer_xy(part_infos, target)
                    print(
                        f"[inverted] 顶抓缓冲 @ {np.round(buf_xy, 3).tolist()} "
                        f"（≈45°侧倾松爪，无 J6）→ fallen → slot",
                        flush=True,
                    )
                    common_buf = dict(common)
                    try:
                        pkg_buf = plan_one_job(
                            **common_buf,
                            strategy_prefer="topdown",
                            pipeline="inverted",
                            place_xy=buf_xy,
                            place_surface="desk",
                            require_upright=False,
                            lock_style=False,
                            enforce_style=False,
                            tip_degs=(45.0, 40.0, 50.0, 35.0),
                            seat_kind="buffer_fallen",
                            buffer_yaw=float(
                                target.get("yaw")
                                or target.get("bake_yaw")
                                or 0.0
                            ),
                        )
                        pkg_buf["desk_snap"] = desk_snap
                        pkg_buf["skip_pre_anime_restore"] = False
                        seat_in_slot(pkg_buf, base, robot, box_model=box)
                        # place 后不回 HOME：从缓冲末态直接规划下一抓
                        q_cont = np.asarray(
                            pkg_buf["mot"].jv_list[-1], dtype=float
                        ).copy()
                        side._reset_robot_hand(robot, pkg_buf["jaw_open"])
                        robot.goto_given_conf(
                            q_cont, ee_values=float(pkg_buf["jaw_open"])
                        )
                        buf_snap = _snapshot_part_pose(target)
                        print(
                            "[inverted] 缓冲完成 → 直接规划 fallen→入格"
                            "（跳过 HOME）",
                            flush=True,
                        )
                        pkg = plan_one_job(
                            **common,
                            strategy_prefer=args.strategy,
                            pipeline="fallen",
                            place_surface="box",
                            require_upright=True,
                            lock_style=True,
                            enforce_style=True,
                            seat_kind="slot",
                            start_conf=q_cont,
                        )
                        pkg["desk_snap"] = buf_snap
                        pkg["skip_pre_anime_restore"] = True
                        seat_in_slot(pkg, base, robot, box_model=box)
                        side._reset_robot_hand(robot, pkg["jaw_open"])
                        robot.goto_given_conf(
                            side.HOME_CONF,
                            ee_values=float(pkg["jaw_open"]),
                        )
                        return [pkg_buf, pkg]
                    except Exception as e_buf:
                        # 缓冲失败：尝试侧抓 + J6 直入格；再不行原位 remesh
                        print(
                            f"[inverted] 缓冲 PPP 失败 → 试侧抓+J6入格 "
                            f"({e_buf})",
                            flush=True,
                        )
                        try:
                            pkg = plan_one_job(
                                **common,
                                strategy_prefer="side",
                                pipeline="inverted",
                                place_surface="box",
                                require_upright=True,
                                lock_style=True,
                                enforce_style=True,
                                seat_kind="slot",
                            )
                            pkg["desk_snap"] = desk_snap
                            seat_in_slot(pkg, base, robot, box_model=box)
                            side._reset_robot_hand(robot, pkg["jaw_open"])
                            robot.goto_given_conf(
                                side.HOME_CONF,
                                ee_values=float(pkg["jaw_open"]),
                            )
                            return [pkg]
                        except Exception as e_side:
                            print(
                                f"[inverted] 侧抓入格也失败 → 原位 remesh 兜底 "
                                f"({e_side})",
                                flush=True,
                            )
                            yaw0 = float(
                                target.get("yaw")
                                or target.get("bake_yaw")
                                or 0.0
                            )
                            xy0 = np.asarray(
                                desk_snap.get("info_pos", target["pos"]),
                                dtype=float,
                            )[:2]
                            remesh_part_as_fallen(
                                target, xy0, yaw=yaw0, base=base
                            )
                            fallen_snap = _snapshot_part_pose(target)
                            pkg = plan_one_job(
                                **common,
                                strategy_prefer=args.strategy,
                                pipeline="fallen",
                                place_surface="box",
                                require_upright=True,
                                lock_style=True,
                                enforce_style=True,
                                seat_kind="slot",
                            )
                            pkg["desk_snap"] = fallen_snap
                            pkg["skip_pre_anime_restore"] = False
                            seat_in_slot(pkg, base, robot, box_model=box)
                            side._reset_robot_hand(robot, pkg["jaw_open"])
                            robot.goto_given_conf(
                                side.HOME_CONF,
                                ee_values=float(pkg["jaw_open"]),
                            )
                            print(
                                "[inverted] 注意: 原位 remesh 兜底，无缓冲/"
                                "J6 翻正动画，直接 fallen→入格",
                                flush=True,
                            )
                            return [pkg]
                if state_name == "normal":
                    pkg = plan_one_job(
                        **common,
                        strategy_prefer="topdown",
                        pipeline="normal",
                        tip_degs=EC_PLACE_TIP_DEGS_TOP,
                        place_surface="box",
                        require_upright=True,
                        lock_style=True,
                        enforce_style=True,
                        seat_kind="slot",
                    )
                else:
                    pkg = plan_one_job(
                        **common,
                        strategy_prefer=args.strategy,
                        pipeline="fallen",
                        place_surface="box",
                        require_upright=True,
                        lock_style=True,
                        enforce_style=True,
                        seat_kind="slot",
                    )
                # 原位快照供动画前总还原；relocate 时开抓用 pick_desk_snap（见 plan_one_job）
                pkg["desk_snap"] = desk_snap
                seat_in_slot(pkg, base, robot, box_model=box)
                side._reset_robot_hand(robot, pkg["jaw_open"])
                robot.goto_given_conf(
                    side.HOME_CONF, ee_values=float(pkg["jaw_open"])
                )
                return [pkg]

            try:
                pkgs = _plan_slot_for_state(st)
            except Exception as e0:
                # 入格失败：清障后再试
                # normal/fallen 挡路 → 直接入其 JSON 格；inverted → 缓冲区
                blockers = find_blocking_neighbors(target, part_infos)
                if not blockers:
                    raise
                print(
                    f"[clear] id={tid} 入格失败 → 处理挡路 "
                    f"{[b.get('id') for b in blockers]} ({e0})",
                    flush=True,
                )
                clear_pkgs = []
                for blk in blockers:
                    try:
                        cpkg, fin_job = resolve_obstructor(
                            robot,
                            blk,
                            part_infos,
                            desk_obstacle,
                            box,
                            args.grasp_pickle,
                            args.side_grasp_pickle,
                            use_rrt,
                            base=base,
                            pending_jobs=pending,
                            strategy_fallen=args.strategy,
                        )
                        clear_pkgs.append(cpkg)
                        by_id[int(blk["id"])] = blk
                        if fin_job is not None:
                            pending = [
                                j
                                for j in pending
                                if int(j["object_id"])
                                != int(fin_job["object_id"])
                            ]
                            ok_n += 1
                            print(
                                f"[clear] 挡路 id={fin_job['object_id']} "
                                f"已提前入格，从队列移除",
                                flush=True,
                            )
                    except Exception as ce:
                        print(
                            f"[clear] 挡路 id={blk.get('id')} 失败: {ce}",
                            flush=True,
                        )
                if not clear_pkgs:
                    raise
                job_pkgs.extend(clear_pkgs)
                desk_snap = _snapshot_part_pose(target)
                pkgs = _plan_slot_for_state(st)
                job_pkgs.extend(pkgs)
                ok_n += 1
                print(f"[plan] 已完成 {ok_n}/{total_n}，剩余 {len(pending)}", flush=True)
                _ec_pump_ui(base, n=3)
                continue

            job_pkgs.extend(pkgs)
            ok_n += 1
            print(f"[plan] 已完成 {ok_n}/{total_n}，剩余 {len(pending)}", flush=True)
            _ec_pump_ui(base, n=3)
        except Exception as e:
            fail_n += 1
            _restore_part_pose(target, desk_snap, base=base)
            # remesh 后 model 可能已换：尽量回到桌面快照
            try:
                by_id[tid] = target
            except Exception:
                pass
            side._reset_robot_hand(robot, side.jaw_widths_for_part(target)[0])
            robot.goto_given_conf(side.HOME_CONF)
            print(f"[fail] job object_id={tid}: {e}", flush=True)

    # 动画前：每个 object 只还原「第一段」快照（通常是原始桌面）。
    # inverted 两段 [缓冲, 入格] 若都 restore，后一段会把件留在缓冲/格子里，
    # 空格前就像「少了搬运过程」。skip_pre_anime_restore 的段不再覆盖首段。
    restored_ids = set()
    for pkg in job_pkgs:
        tid = int(pkg.get("target", {}).get("id", -1))
        if tid in restored_ids:
            continue
        if pkg.get("skip_pre_anime_restore"):
            continue
        snap = pkg.get("desk_snap")
        if snap is not None:
            _restore_part_pose(pkg["target"], snap, base=base)
            restored_ids.add(tid)

    print(f"\n[plan] ok={ok_n} fail={fail_n} / {len(jobs)}", flush=True)
    if not job_pkgs:
        print(
            "[done] 无成功轨迹 → 窗口中仅显示场景/HOME 姿态；"
            "空格无动画。请查看上方 [fail]/[cd] 日志。",
            flush=True,
        )
        base.run()
        return

    try:
        preview_mesh.detach()
    except Exception:
        pass
    side.release_model_collection(preview_mesh)

    if getattr(args, "export_traj", ""):
        if fail_n or ok_n < len(jobs):
            print(
                f"[export] 跳过：未满 {ok_n}/{len(jobs)}（fail={fail_n}）",
                flush=True,
            )
        else:
            sid = os.path.basename(os.path.normpath(args.sample_dir))
            side.export_planned_trajectory(
                args.export_traj,
                sid,
                args.task,
                _export_segments_from_job_pkgs(job_pkgs),
                source="run_ec_bin_sim",
            )
        if not args.auto_play:
            base.destroy()
            return

    if args.no_anime:
        for pkg in job_pkgs:
            apply_job_final(robot, pkg, base)
        robot.gen_meshmodel(toggle_tcp_frame=False).attach_to(base)
        print("[sim] --no-anime：规划成功，关闭窗口退出", flush=True)
        base.run()
        return

    print("=" * 60)
    if args.auto_play:
        side.wait_for_ui_animation_start(args)
        print("[sim] 操作: 自动播放（多件会连续播放）；仍可按 [Space] 手动加速")
    else:
        print("[sim] 操作: 按 [Space] 推进下一帧（多件会连续播放）")
    print("=" * 60)
    # 起始静态姿态
    side._reset_robot_hand(robot, job_pkgs[0]["jaw_open"])
    robot.goto_given_conf(side.HOME_CONF, ee_values=float(job_pkgs[0]["jaw_open"]))
    run_space_anime(base, robot, job_pkgs, auto_play=bool(args.auto_play))


if __name__ == "__main__":
    main()
