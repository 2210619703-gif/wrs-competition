#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按智能体 API 任务 JSON 做 Panthera-HT pick / pick-place 仿真（顶抓 + 侧向回退）。

基于 ``run_agent_pick_place_sim.py``（不改原文件），增加：
1. 先用 ``panthera_ht_block_grasps_topdown_yaw.pickle`` 顶抓；
2. 若全部 IK/穿桌失败，再加载 ``panthera_ht_block_grasps.pickle`` 对踵侧向/倾斜抓取；
3. 侧抓保留接近方向（hand-z），邻件只做抓取位姿预筛；近基座用快速关节插值规划（避免 ADPlanner 卡死）。

运行（仓库根目录）::
    python tiaozhanbei/sim/run_agent_pick_place_side_sim.py
    python tiaozhanbei/sim/run_agent_pick_place_side_sim.py --task tiaozhanbei/tasks/test06.json
    python tiaozhanbei/sim/run_agent_pick_place_side_sim.py --task tiaozhanbei/tasks/test06.json --no-anime
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_SIM_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Windows：先加载机器人/WRS，再碰 Panda taskMgr
from wrs import wd, rm, mgm, mcm, gg, ppp, mmd
import wrs.basis.data_adapter as da
from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

from tiaozhanbei.sim.environment import (
    MODEL_SCALE,
    DESK_XY,
    DESK_THICKNESS,
    gen_desk,
    attach_desk_stripe,
)
from tiaozhanbei.sim.scene_layout import (
    STORAGE_BOX_POS,
    STORAGE_BOX_SIZE,
    STORAGE_BOX_WALL,
    PLACE_TCP_Z,
    PLACE_OBJ_Z,
)

# ---------------------------------------------------------------------------
# 布局 / 防穿桌 / 收纳盒
# ---------------------------------------------------------------------------
HOME_CONF = np.array([0.0, 0.30, 0.30, 0.0, 0.20, 0.0], dtype=float)


def relax_panthera_joint_ranges_for_demo(robot) -> None:
    """演示规划用：放宽 Panthera 关节范围，特别是腕部 J5/J6。

    EC 仿真会在放置阶段对 J5/J6 做更宽的姿态搜索。标准 pick-place
    也需要类似余量，否则连续入框时容易在边界落点被 IK/线性接近提前拒绝。
    """
    jlc = getattr(robot, "jlc", None)
    jnts = getattr(jlc, "jnts", None)
    if not jnts or len(jnts) < 6:
        return
    ranges_deg = [
        (-180.0, 180.0),
        (-90.0, 135.0),
        (-150.0, 150.0),
        (-180.0, 180.0),
        (-120.0, 100.0),
        (-180.0, 180.0),
    ]
    for jnt, (lo, hi) in zip(jnts[:6], ranges_deg):
        try:
            old = np.asarray(jnt.motion_range, dtype=float)
            new = np.array([np.deg2rad(lo), np.deg2rad(hi)], dtype=float)
            if old.shape == (2,):
                new[0] = min(float(old[0]), float(new[0]))
                new[1] = max(float(old[1]), float(new[1]))
            jnt.motion_range = new
        except Exception:
            continue


def arm_joints(robot):
    """拿到 6 个手臂关节对象。

    ``PantheraHTSglArm`` 上没有 ``jlc``，运动链挂在 ``manipulator`` 下面。
    """
    for holder in (robot, getattr(robot, "manipulator", None)):
        jnts = getattr(getattr(holder, "jlc", None), "jnts", None)
        if jnts and len(jnts) >= 6:
            return jnts[:6]
    return None


def apply_joint_ranges_deg(robot, ranges_deg) -> None:
    """把 6 个关节范围硬设成给定值（度），覆盖仿真默认的宽松范围。

    仿真默认范围比真机软限位宽不少。要把规划结果拿去驱动真机时，必须让规划阶段
    就在真机能到的范围内搜索，否则规划出来的位姿真机根本到不了。
    """
    jnts = arm_joints(robot)
    if jnts is None:
        raise RuntimeError("找不到机器人的 6 个手臂关节，无法收紧关节范围")
    for jnt, (lo, hi) in zip(jnts, ranges_deg):
        try:
            jnt.motion_range = np.array([np.deg2rad(lo), np.deg2rad(hi)], dtype=float)
        except Exception:
            continue
    print(
        "[limits] 关节范围已收紧为 "
        + " ".join(f"J{i + 1}[{lo:.0f},{hi:.0f}]" for i, (lo, hi) in enumerate(ranges_deg)),
        flush=True,
    )


def parse_joint_ranges_arg(spec: str):
    """解析 ``--joint-ranges``：6 组 ``lo,hi``（度），或指向同样内容的 JSON 文件。"""
    if not spec:
        return None
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    else:
        data = json.loads(spec)
    ranges = [(float(lo), float(hi)) for lo, hi in data]
    if len(ranges) != 6:
        raise ValueError(f"--joint-ranges 需要 6 组上下限，收到 {len(ranges)} 组")
    return ranges


def facing_j1_home(xy, home=None, stretch_defaults=True):
    """J1 转到目标方位；其余关节取 ``home``（默认 HOME）。

    ``stretch_defaults``：仅在未提供自定义 home 时略伸展 J2/J3/J5，避免折臂支解。
    """
    use_default = home is None
    q = np.asarray(HOME_CONF if use_default else home, dtype=float).copy()
    p = np.asarray(xy, dtype=float).reshape(-1)[:2]
    q[0] = float(np.arctan2(float(p[1]), float(p[0])))
    if use_default and stretch_defaults:
        q[1] = float(max(q[1], 0.50))
        q[2] = float(max(q[2], 0.60))
        q[4] = float(max(q[4], 0.25))
    return q
DEFAULT_GRASP_TOPDOWN = os.path.join(
    _SIM_DIR, "grasp", "panthera_ht_block_grasps_topdown_yaw.pickle"
)
DEFAULT_GRASP_ANTIPODAL = os.path.join(
    _SIM_DIR, "grasp", "panthera_ht_block_grasps.pickle"
)
# 兼容原参数名
DEFAULT_GRASP_PICKLE = DEFAULT_GRASP_TOPDOWN
DEFAULT_TASK = os.path.join(_TB_DIR, "tasks", "delivery_agent_auto.json")
DEFAULT_DATASET = os.path.join(_TB_DIR, "dataset_learn")
DEFAULT_CACHED_TRAJ = os.path.join(_SIM_DIR, "traj", "0501_all_parts_traj.json")
MAX_SIDE_GRASPS_ADAPT = 120

DESK_TOP_Z = 0.0
# 指尖最低点相对桌面
TIP_TOUCH_EPS = 0.0005          # 数值容差：tip_z ≥ -0.5mm 视为贴桌、不穿模
TIP_PENETRATE_MAX = 0.002       # 顶抓预筛宽松阈值；侧抓轨迹用 TIP_FLOOR_STRICT
TIP_FLOOR_STRICT = DESK_TOP_Z   # 侧抓要求：指尖始终 ≥ 0（允许 TIP_TOUCH_EPS）
APPROACH_CLEARANCE = 0.08
MIN_APPROACH_Z = 0.08
# 钳子：规划仍按指尖贴桌接近（PPP 才过得去），合爪前再沿 -z 下探这段，
# 让夹持面落到柄中段，而不是在件顶上空合。
PLIER_SINK_M = 0.010
PLIER_TIP_FLOOR = -0.014
HOVER_Z = 0.12  # pick 后抬起高度（无收纳盒）
FINGERTIP_LIFT_STEP = 0.002     # 方案2：小步抬升
FINGERTIP_LIFT_ITERS = 40

JAW_OPEN_MAX = 0.055
JAW_HARDWARE_MAX = 0.08  # Panthera 模型上限；门禁用此值判断「物理不可外夹」
JAW_CLOSE_MIN = 0.0005
JAW_WIDTH_MARGIN = 0.006
# 宽截面（钳柄并夹约 39mm）指面比连杆原点更厚，只留 6mm 张到 45mm 仍判穿模。
JAW_WIDTH_MARGIN_WIDE = 0.016
JAW_WIDE_SECTION = 0.028
# 舒适夹持宽度（评分用）
GRASP_COMFORT_W_MIN = 0.006
GRASP_COMFORT_W_MAX = 0.042
MIN_PART_HEIGHT_TOPDOWN = 0.003  # 过薄垫片：顶抓仍尝试但抬高 TCP
# 方案 A：接触搜索后略再合一点，视觉上贴紧（过大易穿模）
JAW_CONTACT_SQUEEZE = 0.0008
JAW_CONTACT_BIN_ITERS = 18
# 闭合后略张开仍碰撞 → 指面已嵌入零件（深穿模），必须剔除
JAW_CLEARANCE_PROBE = 0.0015
# 螺丝等细件：接触后不再额外合拢，避免尖端穿进螺头
JAW_CONTACT_SQUEEZE_SCREW = 0.0

PICK_APPROACH_DIST = 0.08
PICK_DEPART_DIST = 0.10
PLACE_APPROACH_DIST = 0.06
PLACE_DEPART_DIST = 0.08
LINEAR_GRANULARITY = 0.05
MAX_PPP_GRASPS = 24  # 侧向候选更多
ANIMATION_INTERVAL = 0.03
AUTO_PLAY_FRAME_STEP = 1
AUTO_PLAY_INITIAL_HOLD_SEC = 1

# 收纳盒：与 dataset_learn/scene_layout.json 固定布局一致
DEFAULT_BOX_XY = np.array(
    [float(STORAGE_BOX_POS[0]), float(STORAGE_BOX_POS[1])], dtype=float
)


def _topdown_rotmat(yaw=0.0):
    """夹爪朝下的旋转；yaw 绕世界 z。"""
    return rm.rotmat_from_euler(np.pi, 0.0, 0.0) @ rm.rotmat_from_euler(
        0.0, 0.0, float(yaw)
    )


def place_tcp_rot_candidates(grasp, obj_rot, pick_rot=None, place_obj_rot=None):
    """放置 TCP 朝向候选：盒位 XY 可达，但同一 grasp 姿态未必在该处有 IK。

    若给 ``place_obj_rot``（期望放置时物体世界旋转），优先
    ``place_obj_rot @ grasp.ac_rotmat``，使持物随腕部转到目标姿态
    （例如平躺螺丝在空中竖起后再入格）。
    """
    obj_rot = np.asarray(obj_rot, dtype=float)
    ac_r = np.asarray(grasp.ac_rotmat, dtype=float)
    if pick_rot is None:
        pick_rot = obj_rot @ ac_r
    pick_rot = np.asarray(pick_rot, dtype=float)
    yaw = float(np.arctan2(pick_rot[1, 0], pick_rot[0, 0]))
    cands = []
    if place_obj_rot is not None:
        R_place = np.asarray(place_obj_rot, dtype=float)
        # 目标物体姿态 + 绕世界 z 的若干 yaw，提高 IK 成功率
        for dyaw in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi, 0.25 * np.pi, -0.25 * np.pi):
            R_obj = rm.rotmat_from_euler(0.0, 0.0, float(dyaw)) @ R_place
            cands.append(R_obj @ ac_r)
    cands.extend(
        [
            pick_rot,  # 带着抓取姿态平移到盒上
            _topdown_rotmat(yaw),  # 俯视 + 抓取 yaw（多数 topdown grasp 可用）
            _topdown_rotmat(0.0),
            obj_rot @ ac_r,  # 旧：物体位姿复用 grasp（common-grasp 语义）
        ]
    )
    # 去重（近似）
    uniq = []
    for R in cands:
        if all(float(np.linalg.norm(R - U)) > 1e-6 for U in uniq):
            uniq.append(R)
    return uniq


def load_task(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    # 兼容 {"task": {...}} 或直接 task 对象
    if isinstance(data, dict) and "task" in data and "action" not in data:
        return data["task"]
    return data


def coerce_multi_object_task_for_single_sim(task: dict) -> dict:
    """当前动画器是单目标 pick-place；全量任务先取第一个真实零件演示。"""
    action = str(task.get("action") or "").lower()
    obj_name = str(task.get("object") or "")
    if action != "multi_agent_pack" and obj_name not in {"所有零件", "所有的零件", "全部零件"}:
        return task

    for step in task.get("tasks") or []:
        step_obj = str(step.get("object") or "")
        if str(step.get("action") or "").lower() != "pick":
            continue
        if not step_obj or step_obj.startswith("料箱"):
            continue
        concrete = dict(step)
        concrete["task_id"] = f"{task.get('task_id', 'multi_agent_pack')}:{step.get('task_id', 'part')}"
        concrete["action"] = "pick_place"
        concrete["source_task_id"] = task.get("task_id")
        concrete["source_action"] = task.get("action")
        concrete["source_object"] = task.get("object")
        concrete["tasks"] = [
            dict(item)
            for item in (task.get("tasks") or [])
            if str(item.get("object") or "") == step_obj
            and str(item.get("action") or "").lower() in {"perceive", "pick", "adjust_pose", "move", "place"}
        ]
        print(
            f"[task] multi-agent/full-parts task adapted to first concrete part: {step_obj}",
            flush=True,
        )
        return concrete

    return task


def _steps_from_batch_item(task: dict, item: dict) -> list[dict]:
    obj = str(item.get("object") or task.get("object") or "")
    coord = item.get("coordinate") or task.get("coordinate") or [0.0, 0.0, 0.0]
    dest = item.get("dest_coordinate") or task.get("dest_coordinate") or [0.3, -0.2, 0.0]
    destination = item.get("destination") or task.get("destination") or "收纳盒"
    pose_state = item.get("pose_state") or task.get("pose_state") or "normal"
    actions = item.get("steps") or ["perceive", "pick", "move", "place"]
    steps = []
    for idx, action in enumerate(actions, start=1):
        step_coord = dest if str(action).lower() in {"move", "place"} else coord
        steps.append(
            {
                "task_id": f"t{idx:03d}",
                "step": idx,
                "action": action,
                "object": obj,
                "coordinate": step_coord,
                "dest_coordinate": dest,
                "destination": destination,
                "angle": item.get("angle", task.get("angle", 0.0)),
                "rpy": item.get("rpy", task.get("rpy", [0.0, 0.0, item.get("angle", 0.0)])),
                "retry": 0,
                "max_retry": 3,
                "status": "pending",
                "reason": item.get("reason", "batch 子任务"),
                "original_cmd": task.get("original_cmd"),
                "pose_state": pose_state,
            }
        )
    return steps


def expand_multi_pick_place_tasks(task: dict) -> list[dict]:
    """把多目标任务展开为单进程内连续执行的多个 pick-place 子任务。"""
    action = str(task.get("action") or "").lower()
    batch = task.get("batch")
    batch_items = batch if isinstance(batch, list) else []
    if not batch_items:
        batch_json = task.get("batch_json")
        if isinstance(batch_json, str) and batch_json.strip():
            try:
                parsed = json.loads(batch_json)
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                batch_items = parsed

    grouped: list[tuple[str, list[dict]]] = []
    if batch_items:
        for item in batch_items:
            if not isinstance(item, dict):
                continue
            obj = str(item.get("object") or "")
            if not obj or obj.startswith("料箱"):
                continue
            grouped.append((obj, _steps_from_batch_item(task, item)))
    elif action in {"multi_pick_place", "multi_agent_pack"} or str(task.get("object") or "") in {
        "所有零件",
        "所有的零件",
        "全部零件",
    }:
        for step in task.get("tasks") or []:
            obj = str(step.get("object") or "")
            step_action = str(step.get("action") or "").lower()
            if not obj or obj.startswith("料箱"):
                continue
            if step_action not in {"perceive", "pick", "adjust_pose", "move", "place"}:
                continue
            if not grouped or grouped[-1][0] != obj:
                grouped.append((obj, []))
            grouped[-1][1].append(step)

    tasks: list[dict] = []
    for index, (obj, steps) in enumerate(grouped, start=1):
        pick_step = next(
            (step for step in steps if str(step.get("action") or "").lower() == "pick"),
            steps[0] if steps else {},
        )
        if not pick_step:
            continue
        subtask = dict(pick_step)
        subtask["task_id"] = f"{task.get('task_id', 'multi_pick_place')}_{index:02d}"
        subtask["action"] = "pick_place"
        subtask["object"] = obj
        subtask["tasks"] = [dict(step) for step in steps]
        subtask["task_sequence_desc"] = [
            f"{index}.{obj}:{'>'.join(str(step.get('action') or '') for step in steps)}"
        ]
        tasks.append(subtask)
    return tasks


def _norm_state(s) -> str:
    t = str(s or "normal").strip().lower()
    if t in ("upside_down", "inverted", "倒放", "倒置"):
        return "inverted"
    if t in ("fallen", "tipped", "倾倒", "侧倒", "lying"):
        return "fallen"
    if t in ("normal", "upright", "正放", "正常"):
        return "normal"
    return t


def parse_adjust_pose(task: dict, part_state: str = None):
    """解析 tasks[].adjust_pose；或零件非正放时默认放置前调成 normal。

    返回 dict:
      adjust_to_normal, place_yaw, place_rpy, source
    """
    part_state = _norm_state(part_state or task.get("state") or task.get("pose_state"))
    place_yaw = None
    for key in ("rpy", "angle"):
        if key == "rpy" and task.get("rpy"):
            rpy = task["rpy"]
            if len(rpy) >= 3:
                place_yaw = float(rpy[2])
            break
        if key == "angle" and task.get("angle") is not None:
            place_yaw = float(task["angle"])
    adj_step = None
    for step in task.get("tasks") or []:
        if str(step.get("action") or "").lower() == "adjust_pose":
            adj_step = step
            break
    if adj_step is not None:
        rpy = adj_step.get("rpy") or [0.0, 0.0, float(adj_step.get("angle") or 0.0)]
        while len(rpy) < 3:
            rpy.append(0.0)
        yaw = float(rpy[2])
        return {
            "adjust_to_normal": True,
            "target_pose_state": _norm_state(
                adj_step.get("target_pose_state") or "normal"
            ),
            "place_yaw": yaw,
            "place_rpy": [0.0, 0.0, yaw],
            "source": "tasks.adjust_pose",
        }
    if part_state in ("inverted", "fallen"):
        yaw = float(place_yaw if place_yaw is not None else 0.0)
        return {
            "adjust_to_normal": True,
            "target_pose_state": "normal",
            "place_yaw": yaw,
            "place_rpy": [0.0, 0.0, yaw],
            "source": f"state={part_state}",
        }
    return {
        "adjust_to_normal": False,
        "target_pose_state": "normal",
        "place_yaw": float(place_yaw if place_yaw is not None else 0.0),
        "place_rpy": [0.0, 0.0, float(place_yaw if place_yaw is not None else 0.0)],
        "source": None,
    }


def build_upright_part_model(stl_path: str, yaw: float, rgb=None, name=None):
    """重建“正放”零件：优先竖直站立（长轴沿 Z），否则 roll=pitch=0。

    螺丝/销钉类 STL 长轴常在局部 XY，仅清零 roll/pitch 仍会平躺。
    """
    yaw = float(yaw)
    flat_mesh, flat_h, flat_g = _load_mesh_on_table(stl_path, yaw, roll=0.0, pitch=0.0)
    # 若平放时高度不是主尺寸，尝试绕 X 竖起（长轴→Z）
    ext = np.sort(np.asarray(flat_mesh.extents, dtype=float))
    roll_u, pitch_u = 0.0, 0.0
    mesh, height, grip_w = flat_mesh, flat_h, flat_g
    if float(ext[2]) > float(ext[0]) * 1.35 and float(flat_h) < 0.7 * float(ext[2]):
        mesh, height, grip_w = _load_mesh_on_table(
            stl_path, yaw, roll=float(np.pi / 2.0), pitch=0.0
        )
        roll_u = float(np.pi / 2.0)
    if rgb is None:
        rgb = rm.vec(0.80, 0.42, 0.36)
    model = mcm.CollisionModel(mesh, name=str(name or "part_upright"), rgb=rgb)
    model.pos = rm.vec(0.0, 0.0, 0.0)
    model.rotmat = np.eye(3)
    model._upright_rpy = (roll_u, pitch_u, yaw)  # 调试用
    return model, float(height), float(grip_w)


def _ppp_obstacle_rounds(grasp_mode, neighbor_models):
    """接近段用的障碍轮次。

    邻件已经在抓取点预筛过。从 HOME 插值时若把邻件当路径障碍，细长工具的
    AABB 会挡住中间构型（三件并排抓螺丝刀必挂）。侧抓本来就不带邻件；
    顶抓先空障碍，失败再带邻件。
    """
    if str(grasp_mode) == "side":
        return [[]]
    nbs = list(neighbor_models or [])
    if not nbs:
        return [[]]
    return [[], nbs]


def _dest_is_null(destination) -> bool:
    if destination is None:
        return True
    s = str(destination).strip().lower()
    return s in ("", "null", "none", "nil")


def has_place_target(task: dict) -> bool:
    """是否执行 pick-place（看 destination / dest_coordinate / tasks，不单看顶层 action）。"""
    if not _dest_is_null(task.get("destination")):
        return True
    for step in task.get("tasks") or []:
        if str(step.get("action") or "").lower() == "place":
            return True
        if not _dest_is_null(step.get("destination")):
            return True
    dest = np.asarray(task.get("dest_coordinate") or [0, 0, 0], dtype=float)
    pick = np.asarray(task.get("coordinate") or [0, 0, 0], dtype=float)
    if dest.shape[0] >= 2 and float(np.linalg.norm(dest[:2])) > 1e-3:
        if float(np.linalg.norm(dest[:2] - pick[:2])) > 1e-3:
            return True
    return False


def resolve_box_xy(task: dict) -> np.ndarray:
    """优先任务 dest_coordinate；无效时用 scene_layout 固定盒位。"""
    dest = np.asarray(task.get("dest_coordinate") or DEFAULT_BOX_XY, dtype=float)
    if dest.shape[0] < 2 or float(np.linalg.norm(dest[:2])) < 1e-3:
        return DEFAULT_BOX_XY.copy()
    return np.array([float(dest[0]), float(dest[1])], dtype=float)


def make_storage_box(center_xy, size=None, wall_t=STORAGE_BOX_WALL):
    """开口朝上的简易收纳盒（底+四壁），中心落在 dest_coordinate xy。"""
    if size is None:
        size = STORAGE_BOX_SIZE
    cx, cy = float(center_xy[0]), float(center_xy[1])
    L, W, H = float(size[0]), float(size[1]), float(size[2])
    t = float(wall_t)
    rgb = rm.vec(0.55, 0.42, 0.32)
    alpha = 0.55
    parts = [
        mcm.gen_box(
            xyz_lengths=rm.vec(L, W, t),
            pos=rm.vec(cx, cy, t / 2.0),
            rgb=rgb,
            alpha=alpha,
        )
    ]
    wall_z = t + (H - t) / 2.0
    wall_h = max(H - t, t)
    parts.extend(
        [
            mcm.gen_box(
                xyz_lengths=rm.vec(L, t, wall_h),
                pos=rm.vec(cx, cy + W / 2.0 - t / 2.0, wall_z),
                rgb=rgb,
                alpha=alpha,
            ),
            mcm.gen_box(
                xyz_lengths=rm.vec(L, t, wall_h),
                pos=rm.vec(cx, cy - W / 2.0 + t / 2.0, wall_z),
                rgb=rgb,
                alpha=alpha,
            ),
            mcm.gen_box(
                xyz_lengths=rm.vec(t, W - 2.0 * t, wall_h),
                pos=rm.vec(cx + L / 2.0 - t / 2.0, cy, wall_z),
                rgb=rgb,
                alpha=alpha,
            ),
            mcm.gen_box(
                xyz_lengths=rm.vec(t, W - 2.0 * t, wall_h),
                pos=rm.vec(cx - L / 2.0 + t / 2.0, cy, wall_z),
                rgb=rgb,
                alpha=alpha,
            ),
        ]
    )
    return parts


def place_goal_pose(box_xy, obj_cmodel, box_floor_z=None):
    """放置位姿：保留原 yaw/旋转，XY 移到盒心，底面抬到盒底板上沿。"""
    if box_floor_z is None:
        box_floor_z = float(STORAGE_BOX_WALL)
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    # 桌面加载时底面在 z=0，obj_pos[2] 为原点高度；盒内再加底板厚度
    pos = np.array(
        [
            float(box_xy[0]),
            float(box_xy[1]),
            float(obj_pos[2]) + float(box_floor_z) + float(PLACE_OBJ_Z),
        ],
        dtype=float,
    )
    rot = np.asarray(obj_cmodel.rotmat, dtype=float).copy()
    return pos, rot


class BoxPlacementState:
    """轻量料框占用模型：只做几何近似，不引入动力学掉落。"""

    def __init__(self, center_xy, size=None, wall_t=STORAGE_BOX_WALL):
        if size is None:
            size = STORAGE_BOX_SIZE
        self.center_xy = np.asarray(center_xy, dtype=float).reshape(-1)[:2]
        self.size = np.asarray(size, dtype=float).reshape(-1)[:3]
        self.wall_t = float(wall_t)
        self.items: list[dict] = []
        inner_l = max(0.02, float(self.size[0]) - 2.0 * self.wall_t)
        inner_w = max(0.02, float(self.size[1]) - 2.0 * self.wall_t)
        self.inner_min = self.center_xy - np.array([inner_l / 2.0, inner_w / 2.0], dtype=float)
        self.inner_max = self.center_xy + np.array([inner_l / 2.0, inner_w / 2.0], dtype=float)
        self.floor_z = float(self.wall_t)
        self.max_stack_z = float(self.size[2]) + self.floor_z

    def _part_extents(self, model) -> np.ndarray:
        try:
            bounds = np.asarray(model.objtrm.bounds, dtype=float)
            ext = bounds[1] - bounds[0]
        except Exception:
            try:
                bounds = np.asarray(model.trm_mesh.bounds, dtype=float)
                ext = bounds[1] - bounds[0]
            except Exception:
                ext = np.array([0.04, 0.02, 0.015], dtype=float)
        return np.maximum(np.asarray(ext, dtype=float), 0.005)

    def _footprint_radius(self, ext: np.ndarray) -> float:
        return float(np.clip(0.55 * max(float(ext[0]), float(ext[1])), 0.012, 0.055))

    def _fits_xy(self, xy: np.ndarray, radius: float) -> bool:
        return bool(
            self.inner_min[0] + radius <= xy[0] <= self.inner_max[0] - radius
            and self.inner_min[1] + radius <= xy[1] <= self.inner_max[1] - radius
        )

    def _clearance(self, xy: np.ndarray, radius: float) -> float:
        if not self.items:
            return float("inf")
        return min(
            float(np.linalg.norm(xy - item["xy"]) - radius - item["radius"])
            for item in self.items
        )

    def _candidate_xy(self, radius: float) -> list[np.ndarray]:
        cx, cy = self.center_xy
        wall_clearance = 0.025
        span_x = max(0.0, (self.inner_max[0] - self.inner_min[0]) / 2.0 - radius - wall_clearance)
        span_y = max(0.0, (self.inner_max[1] - self.inner_min[1]) / 2.0 - radius - wall_clearance)
        offsets = [
            (0.0, 0.0),
            (-0.45, 0.0),
            (0.0, -0.35),
            (0.45, 0.0),
            (0.0, 0.35),
            (-0.75, 0.0),
            (0.75, 0.0),
            (-0.45, -0.35),
            (-0.45, 0.35),
            (0.45, -0.35),
            (0.45, 0.35),
        ]
        cands = []
        for ox, oy in offsets:
            xy = np.array([cx + ox * span_x, cy + oy * span_y], dtype=float)
            if self._fits_xy(xy, radius):
                cands.append(xy)
        return cands

    def choose_pose(
        self,
        model,
        yaw: float = 0.0,
        allow_tilt: bool = True,
        candidate_offset: int = 0,
    ):
        ext = self._part_extents(model)
        radius = self._footprint_radius(ext)
        height = float(np.clip(ext[2], 0.008, 0.08))
        min_clearance = max(0.004, radius * 0.25)
        rot = rm.rotmat_from_euler(0.0, 0.0, float(yaw))

        feasible_idx = 0
        for xy in self._candidate_xy(radius):
            clearance = self._clearance(xy, radius)
            if clearance >= min_clearance:
                if feasible_idx < int(candidate_offset):
                    feasible_idx += 1
                    continue
                z = self.floor_z + float(PLACE_OBJ_Z)
                pos = np.array([xy[0], xy[1], z], dtype=float)
                return pos, rot, {
                    "mode": "grid",
                    "support": None,
                    "radius": radius,
                    "height": height,
                    "tilt": 0.0,
                    "clearance": clearance,
                    "candidate_offset": int(candidate_offset),
                }

        if not self.items:
            raise RuntimeError("料框内部空间不足，无法生成放置点")

        support = min(self.items, key=lambda item: item["top_z"])
        direction = self.center_xy - support["xy"]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = direction / norm
        xy = support["xy"] + direction * min(radius * 0.35, 0.018)
        xy = np.minimum(np.maximum(xy, self.inner_min + radius), self.inner_max - radius)
        if not self._fits_xy(xy, radius):
            raise RuntimeError("料框叠放候选点超出边界")

        base_z = float(support["top_z"])
        tilt = 0.0
        if allow_tilt:
            slender = max(float(ext[0]), float(ext[1])) > max(float(ext[2]) * 1.8, 0.025)
            tilt = float(np.deg2rad(10.0 if slender else 5.0))
            sign = -1.0 if len(self.items) % 2 else 1.0
            rot = rm.rotmat_from_euler(sign * tilt, 0.0, float(yaw))
        z = base_z + float(PLACE_OBJ_Z)
        if z + height > self.max_stack_z + 0.04:
            raise RuntimeError("料框叠放高度超过允许范围")
        pos = np.array([xy[0], xy[1], z], dtype=float)
        return pos, rot, {
            "mode": "stack",
            "support": support["object"],
            "radius": radius,
            "height": height,
            "tilt": tilt,
            "clearance": self._clearance(xy, radius),
        }

    def register(self, obj_name: str, pos, rot, meta: dict) -> None:
        xy = np.asarray(pos, dtype=float).reshape(-1)[:2]
        height = float(meta.get("height", 0.02))
        item = {
            "object": str(obj_name),
            "xy": xy,
            "pos": np.asarray(pos, dtype=float).copy(),
            "rot": np.asarray(rot, dtype=float).copy(),
            "radius": float(meta.get("radius", 0.02)),
            "height": height,
            "top_z": float(pos[2]) + height,
            "mode": meta.get("mode"),
        }
        self.items.append(item)


def choose_box_place_pose(
    box_state: BoxPlacementState,
    obj_cmodel,
    yaw: float = 0.0,
    candidate_offset: int = 0,
):
    return box_state.choose_pose(
        obj_cmodel,
        yaw=yaw,
        allow_tilt=True,
        candidate_offset=candidate_offset,
    )


def list_sample_ids(dataset_root: str):
    ids = [
        d
        for d in os.listdir(dataset_root)
        if d.isdigit() and os.path.isdir(os.path.join(dataset_root, d))
    ]
    return sorted(ids, key=lambda x: int(x))


def find_sample_for_task(dataset_root: str, task: dict):
    """按 object + coordinate 在 dataset_learn 中定位样本与目标标注。"""
    cls = str(task.get("object") or "")
    coord = np.asarray(task.get("coordinate") or [0, 0, 0], dtype=float)
    best = None
    best_dist = 1e9
    for sid in list_sample_ids(dataset_root):
        ann_path = os.path.join(dataset_root, sid, "annotations.json")
        if not os.path.isfile(ann_path):
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        for obj in ann.get("objects") or []:
            if obj.get("class") != cls:
                continue
            pose = obj.get("pose_6d") or {}
            xy = np.array([float(pose.get("x", 0.0)), float(pose.get("y", 0.0))])
            dist = float(np.linalg.norm(xy - coord[:2]))
            if dist < best_dist:
                best_dist = dist
                best = (sid, obj, ann)
    if best is None or best_dist > 0.02:
        raise RuntimeError(
            f"未在 {dataset_root} 找到与任务匹配的零件 "
            f"object={cls} xy={coord[:2].tolist()} (最近距离={best_dist:.4f}m)"
        )
    return best


def _resolve_stl_path(stl_path: str) -> str:
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


def _load_mesh_on_table(stl_path: str, yaw: float, roll: float = 0.0, pitch: float = 0.0):
    """毫米 STL → 米制；按 roll/pitch/yaw 旋转后底面贴 z=0。

    返回 (mesh, height, grip_width)：
    grip_width 取旋转前薄边尺寸，避免长条件 AABB 被 yaw 撑大后算错夹宽。
    """
    stl_path = _resolve_stl_path(stl_path)
    mesh = da.trm.load(stl_path)
    mesh.apply_scale(np.array([MODEL_SCALE, MODEL_SCALE, MODEL_SCALE]))
    dims = np.sort(np.asarray(mesh.extents, dtype=float))
    # 扁平长条：高≈最小边，夹宽≈次小边（齿条约 4mm 高 / 6mm 宽 / 250mm 长）
    height = float(max(dims[0], 1e-4))
    grip_width = float(max(dims[1], 1e-4))

    bounds = mesh.bounds
    xy_center = (bounds[0, :2] + bounds[1, :2]) / 2.0
    mesh.apply_translation(np.array([-xy_center[0], -xy_center[1], 0.0]))
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_euler(float(roll), float(pitch), float(yaw))
    mesh.apply_transform(homomat)
    z_min = mesh.bounds[0, 2]
    mesh.apply_translation(np.array([0.0, 0.0, -z_min]))
    # 贴桌后真实高度
    height = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])
    return mesh, height, grip_width


def load_parts_from_annotations(ann: dict, highlight_id=None):
    """按 annotations.json 还原桌面零件；返回 part_infos 列表。"""
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
        mesh, height, grip_width = _load_mesh_on_table(
            obj.get("stl_path") or "", yaw, roll=roll, pitch=pitch
        )
        rgb = colors[i % len(colors)]
        if highlight_id is not None and obj.get("id") == highlight_id:
            rgb = rm.vec(0.95, 0.55, 0.15)
        model = mcm.CollisionModel(mesh, name=str(obj.get("class")), rgb=rgb)
        pos = rm.vec(float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), 0.0)
        model.pos = pos
        model.rotmat = np.eye(3)  # 旋转已烘焙进 mesh
        part_infos.append(
            {
                "id": obj.get("id"),
                "class": obj.get("class"),
                "state": _norm_state(obj.get("state") or "normal"),
                "model": model,
                "pos": np.asarray(pos, dtype=float),
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "height": height,
                "top_z": height,
                "mid_z": 0.5 * height,
                "grip_width": grip_width,
                "visible_pixels": int(obj.get("visible_pixels") or 0),
                "stl_path": obj.get("stl_path") or "",
                "rgb": rgb,
            }
        )
    return part_infos


def configure_gripper_for_demo(robot, jaw_max: float):
    """演示用放大夹爪开口，并修好指面碰撞/闭合零点。"""
    ee = robot.end_effector
    jaw_max = float(np.clip(jaw_max, 0.012, JAW_OPEN_MAX))
    ee.jaw_range = np.array([0.0, jaw_max], dtype=float)
    # 方案 A 关键：默认 close_bias=0.006 会让 jaw_width≈0 时机械开口仍约 6mm，
    # 细零件永远“夹不拢”，接触搜索也碰不到。
    ee.close_bias = 0.0
    ee._command_max = jaw_max + ee.close_bias
    if hasattr(ee, "jlc") and len(ee.jlc.jnts) > 1:
        ee.jlc.jnts[0].motion_range = np.array([0.0, ee._command_max / 2.0], dtype=float)
        ee.jlc.jnts[1].motion_range = np.array([0.0, ee._command_max], dtype=float)
    # Panthera 只填了 cdelements；is_mesh_collided 读的是 cdmesh_elements（原先为空）
    if hasattr(ee, "cdelements") and ee.cdelements:
        ee.cdmesh_elements = list(ee.cdelements)
    if hasattr(ee, "_calibrate_opening_direction"):
        ee._calibrate_opening_direction()
    ee.change_jaw_width(jaw_max)


def _ee_fingers_hit_object(ee, obj_cmodel) -> bool:
    """指面–零件接触检测（兼容 cdmesh_elements 未同步的情况）。"""
    if ee.is_mesh_collided([obj_cmodel]):
        return True
    for jnt in getattr(ee, "jlc", None).jnts[:2] if getattr(ee, "jlc", None) else []:
        cm = getattr(jnt.lnk, "cmodel", None)
        if cm is None:
            continue
        try:
            if cm.is_mcdwith(obj_cmodel):
                return True
        except Exception:
            continue
    return False


def _part_is_plier(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        name = str(obj.get("class") or "")
    else:
        name = str(getattr(obj, "name", "") or "")
    low = name.lower()
    return "plier" in low or "钳子" in name or ("钳" in name and "钳工" not in name)


def grasp_heights_for_part(info):
    """按零件高度给初始 grasp_z；防穿桌由指尖 z_min 抬升完成，不再写死高余量。"""
    prior = grasp_prior_for_class(info.get("class"))
    top_z = float(info.get("top_z", 0.02))
    mid_z = float(info.get("mid_z", top_z * 0.5))
    height = float(info.get("height", 0.02))
    approach_z = max(MIN_APPROACH_Z, top_z + APPROACH_CLEARANCE)
    frac = float(prior.get("grasp_z_frac") or 0.0)
    if frac > 0.0:
        grasp_z = max(0.006, float(height) * frac)
    elif height <= 0.03:
        grasp_z = max(mid_z, top_z)
    else:
        grasp_z = top_z - 0.12 * height
    grasp_z = max(0.002, float(grasp_z))
    return approach_z, grasp_z


def append_tcp_sink(robot, mot, jaw_width, sink_m, *, n_steps=8, tip_floor=PLIER_TIP_FLOOR):
    """接近到位后沿 -z 再下探，不改 PPP 的贴桌接近。"""
    if mot is None or sink_m <= 1e-6:
        return mot
    q = np.asarray(mot.jv_list[-1], dtype=float)
    robot.goto_given_conf(jnt_values=q, ee_values=float(jaw_width))
    pos = np.asarray(robot.gl_tcp_pos, dtype=float).copy()
    rot = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
    seed = q
    added = 0
    for i in range(1, int(max(1, n_steps)) + 1):
        tgt = pos + np.array([0.0, 0.0, -float(sink_m) * i / n_steps], dtype=float)
        q2 = _ik_with_seed(robot, tgt, rot, seed)
        if q2 is None:
            break
        robot.goto_given_conf(jnt_values=q2, ee_values=float(jaw_width))
        tip = finger_tips_z_min(robot.end_effector)
        if np.isfinite(tip) and float(tip) < float(tip_floor):
            break
        mot.extend([q2], [float(jaw_width)], [None])
        seed = np.asarray(q2, dtype=float)
        added += 1
    if added:
        print(
            f"[grasp] 合爪前下探 {added} 步，约 {1000.0 * sink_m * added / n_steps:.0f}mm",
            flush=True,
        )
    return mot


def finger_tips_z_min(ee) -> float:
    """左右指 mesh 顶点在世界系的最低 z（指尖下探量）。"""
    z_min = np.inf
    jlc = getattr(ee, "jlc", None)
    if jlc is None:
        return z_min
    for jnt in jlc.jnts[:2]:
        cm = getattr(jnt.lnk, "cmodel", None)
        if cm is None or getattr(cm, "trm_mesh", None) is None:
            continue
        verts = np.asarray(cm.trm_mesh.vertices, dtype=float)
        if verts.size == 0:
            continue
        rot = np.asarray(cm.rotmat, dtype=float)
        pos = np.asarray(cm.pos, dtype=float)
        wz = verts @ rot.T + pos
        z_min = min(z_min, float(np.min(wz[:, 2])))
    return float(z_min)


def _ik_with_seed(robot, tcp_pos, tcp_rot, seed_jnt=None):
    """方案2：优先带 seed 求 IK，失败再无 seed 重试。"""
    q = robot.ik(
        tgt_pos=tcp_pos,
        tgt_rotmat=tcp_rot,
        seed_jnt_values=seed_jnt,
    )
    if q is None and seed_jnt is not None:
        q = robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
    return q


def raise_grasp_to_fingertip_desk(robot, obj_cmodel, grasp, jaw_width):
    """沿 +z 小步抬高 TCP，使指尖尽量贴桌。

    - 理想：tip_z ≥ -TIP_TOUCH_EPS
    - 回退（方案1）：tip_z ≥ -TIP_PENETRATE_MAX（最多下探约 2mm）仍可接受
    - 方案2：每步 IK 使用上一构型作 seed
    - 抬升时若指面深穿进零件（jaw+间隙仍撞），停止抬升并回退到上一步
    """
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    ac_pos = np.asarray(grasp.ac_pos, dtype=float).copy()
    ac_rot = np.asarray(grasp.ac_rotmat, dtype=float)
    tip_ideal = DESK_TOP_Z - TIP_TOUCH_EPS
    tip_floor = DESK_TOP_Z - TIP_PENETRATE_MAX
    jw = float(jaw_width)
    jaw_max = float(robot.end_effector.jaw_range[1])
    jw = float(min(jw, jaw_max))
    jw_probe = float(min(jw + JAW_CLEARANCE_PROBE, jaw_max))

    tcp_pos = obj_pos + obj_rot @ ac_pos
    tcp_rot = obj_rot @ ac_rot
    seed = np.asarray(robot.get_jnt_values(), dtype=float).copy()
    q = _ik_with_seed(robot, tcp_pos, tcp_rot, seed)
    if q is None:
        return None, "ik0"
    seed = np.asarray(q, dtype=float).copy()
    robot.goto_given_conf(jnt_values=q, ee_values=jw)
    z_min = finger_tips_z_min(robot.end_effector)
    if not np.isfinite(z_min):
        return None, "tip"
    # 低 TCP 时指尖还在桌下/件里，先抬再判，不要一上来就 obj_pen。
    robot.goto_given_conf(jnt_values=q, ee_values=jw_probe)
    clear = not _ee_fingers_hit_object(robot.end_effector, obj_cmodel)
    robot.goto_given_conf(jnt_values=q, ee_values=jw)

    best_ac = ac_pos.copy()
    best_z = float(z_min)
    best_q = seed.copy()
    have_clear = bool(clear)

    if z_min >= tip_ideal and have_clear:
        grasp.ac_pos = ac_pos
        return grasp, "ideal"

    for _ in range(FINGERTIP_LIFT_ITERS):
        if z_min >= tip_ideal and have_clear:
            break
        need = (tip_ideal - z_min) + 1e-4
        step = float(min(FINGERTIP_LIFT_STEP, max(need, 5e-4)))
        if z_min >= tip_ideal and not have_clear:
            step = float(FINGERTIP_LIFT_STEP)
        trial_tcp = tcp_pos + np.array([0.0, 0.0, step], dtype=float)
        trial_ac = obj_rot.T @ (trial_tcp - obj_pos)
        q2 = _ik_with_seed(robot, trial_tcp, tcp_rot, seed)
        if q2 is None:
            step *= 0.5
            if step < 4e-4:
                break
            trial_tcp = tcp_pos + np.array([0.0, 0.0, step], dtype=float)
            trial_ac = obj_rot.T @ (trial_tcp - obj_pos)
            q2 = _ik_with_seed(robot, trial_tcp, tcp_rot, seed)
            if q2 is None:
                break
        robot.goto_given_conf(jnt_values=q2, ee_values=jw_probe)
        hit = _ee_fingers_hit_object(robot.end_effector, obj_cmodel)
        if hit and have_clear:
            break
        seed = np.asarray(q2, dtype=float).copy()
        ac_pos = trial_ac
        tcp_pos = trial_tcp
        robot.goto_given_conf(jnt_values=q2, ee_values=jw)
        z_min = finger_tips_z_min(robot.end_effector)
        if not np.isfinite(z_min):
            break
        if not hit:
            if not have_clear or z_min > best_z:
                best_z = float(z_min)
                best_ac = ac_pos.copy()
                best_q = seed.copy()
            have_clear = True
        if z_min >= tip_ideal and have_clear:
            grasp.ac_pos = ac_pos
            return grasp, "ideal"

    # 方案1 回退：接受最佳可达且 tip_z 在 2mm 以内，且不深穿模
    robot.goto_given_conf(jnt_values=best_q, ee_values=jw_probe)
    if _ee_fingers_hit_object(robot.end_effector, obj_cmodel):
        return None, "obj_pen"
    if best_z >= tip_floor:
        grasp.ac_pos = best_ac
        robot.goto_given_conf(jnt_values=best_q, ee_values=jw)
        tag = "ideal" if best_z >= tip_ideal else "soft"
        return grasp, tag
    return None, "deep"


def lift_grasps_for_fingertip_desk(robot, grasp_collection, obj_cmodel, jaw_width):
    """对 grasp 集合做指尖贴桌抬升；允许 2mm 软接受，带 IK seed 小步抬升。"""
    kept = gg.GraspCollection(end_effector=robot.end_effector)
    n_fail = 0
    n_soft = 0
    fail_tags = {}
    z_samples = []
    robot.goto_given_conf(HOME_CONF)
    for g in grasp_collection:
        g2 = gg.Grasp(
            ee_values=g.ee_values,
            ac_pos=np.asarray(g.ac_pos, dtype=float).copy(),
            ac_rotmat=np.asarray(g.ac_rotmat, dtype=float).copy(),
        )
        raised, tag = raise_grasp_to_fingertip_desk(
            robot, obj_cmodel, g2, jaw_width
        )
        if raised is None:
            n_fail += 1
            fail_tags[tag] = fail_tags.get(tag, 0) + 1
            continue
        if tag == "soft":
            n_soft += 1
        obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
        obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
        tcp = obj_pos + obj_rot @ np.asarray(raised.ac_pos, dtype=float)
        q = _ik_with_seed(
            robot,
            tcp,
            obj_rot @ np.asarray(raised.ac_rotmat, dtype=float),
            seed_jnt=robot.get_jnt_values(),
        )
        if q is not None:
            robot.goto_given_conf(jnt_values=q, ee_values=float(jaw_width))
            z_samples.append(finger_tips_z_min(robot.end_effector))
        kept.append(raised)
    z_msg = ""
    if z_samples:
        z_msg = f" tip_z=[{min(z_samples):.4f},{max(z_samples):.4f}]"
    soft_msg = f" soft={n_soft}" if n_soft else ""
    fail_msg = f" reasons={fail_tags}" if fail_tags else ""
    print(
        f"[cd] 指尖贴桌抬升: 保留 {len(kept)}/{len(grasp_collection)} "
        f"（失败 {n_fail}{soft_msg}{fail_msg}）{z_msg}",
        flush=True,
    )
    if len(kept) == 0 and fail_tags.get("ik0", 0) == len(grasp_collection):
        pos = np.asarray(obj_cmodel.pos, dtype=float)
        print(
            f"[cd] 提示: 全部初始 IK 失败，零件可能超出工作空间 "
            f"xy=[{pos[0]:.3f},{pos[1]:.3f}]（如桌面边缘轴承）。"
            f"1/2 只能改善「有解但抬升难」的情况；此位置需换样本或侧向抓取。",
            flush=True,
        )
    return kept


def jaw_widths_for_part(info):
    """接近默认张满；闭合宽只作种子，最终由接触搜索 refine。"""
    w = float(info.get("grip_width", 0.01))
    open_w = float(JAW_OPEN_MAX)
    close_w = float(
        np.clip(max(w * 0.98, JAW_CLOSE_MIN), JAW_CLOSE_MIN, open_w - 1e-4)
    )
    return open_w, close_w


def _section_width_m(info, sites=None) -> float:
    """夹持截面宽：优先用采样点，其次零件薄边。"""
    if sites:
        return float(max(sites[0][1], 1e-4))
    return float(max(info.get("grip_width") or 0.01, 1e-4))


def sane_jaw_close(jc, info, sites=None) -> float:
    """接触搜索没碰到会退回 0.5mm。钳子这种 39mm 截面绝不能按最小值硬合。"""
    section = _section_width_m(info, sites)
    jc = float(jc)
    if section > 0.012 and jc < 0.5 * section:
        fixed = float(
            np.clip(section * 0.98, JAW_CLOSE_MIN, JAW_OPEN_MAX - 1e-4)
        )
        print(
            f"[jaw] 接触宽 {jc:.4f} 远小于截面 {section:.3f}m，改用 {fixed:.4f}",
            flush=True,
        )
        return fixed
    return float(np.clip(jc, JAW_CLOSE_MIN, JAW_OPEN_MAX - 1e-4))


def approach_jaw_width(target, part_infos, jaw_close) -> float:
    """接近一律张满。长条工具的 AABB 会假重叠，按邻件收口会把 55mm 收到 7mm。"""
    _ = (target, part_infos, jaw_close)
    full = float(JAW_OPEN_MAX)
    print(f"[jaw] 接近张满 {full:.3f}m，到位后再合", flush=True)
    return full


def refine_jaw_close_by_contact(
    robot, obj_cmodel, grasp, jaw_open, jaw_seed, squeeze=None
):
    """方案 A：在抓取 TCP 上对 jaw_width 二分，找「指面 mesh 刚碰到零件」的开口。

    Panthera 的 jaw_width 按连杆原点间距换算，与肉眼指面间隙不一致；
    用 ``end_effector.is_mesh_collided([obj])`` 直接对齐视觉贴合。

    若半开/张开仍撞零件，返回 None（调用方应丢弃该 grasp），不再退回估宽硬夹。
    """
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    tcp_pos = obj_pos + obj_rot @ np.asarray(grasp.ac_pos, dtype=float)
    tcp_rot = obj_rot @ np.asarray(grasp.ac_rotmat, dtype=float)
    q = _ik_with_seed(robot, tcp_pos, tcp_rot, seed_jnt=robot.get_jnt_values())
    if q is None:
        print("[jaw] 接触搜索失败：IK 无解", flush=True)
        return None

    jaw_hi = float(min(jaw_open, max(jaw_seed * 3.0, jaw_seed + 0.02)))
    jaw_hi = float(np.clip(jaw_hi, JAW_CLOSE_MIN + 1e-3, float(robot.end_effector.jaw_range[1])))
    jaw_lo = float(JAW_CLOSE_MIN)
    if squeeze is None:
        squeeze = float(JAW_CONTACT_SQUEEZE)

    ee = robot.end_effector
    # 张开时不应与零件碰撞；仍撞 = TCP/姿态已穿模
    robot.goto_given_conf(jnt_values=q, ee_values=jaw_hi)
    if _ee_fingers_hit_object(ee, obj_cmodel):
        print(
            f"[jaw] 接触搜索：张开 {jaw_hi:.4f} 仍撞零件 → 丢弃该 grasp",
            flush=True,
        )
        return None

    # 找「仍碰撞」的最大开口 ≈ 指面刚贴上
    best_hit = None
    lo, hi = jaw_lo, jaw_hi
    for _ in range(JAW_CONTACT_BIN_ITERS):
        mid = 0.5 * (lo + hi)
        robot.goto_given_conf(jnt_values=q, ee_values=mid)
        if _ee_fingers_hit_object(ee, obj_cmodel):
            best_hit = mid
            lo = mid
        else:
            hi = mid

    if best_hit is None:
        print(
            f"[jaw] 接触搜索：合到 {jaw_lo:.4f} 仍不碰，使用最小值",
            flush=True,
        )
        return float(jaw_lo)

    # 螺丝等：squeeze=0，停在接触边界，避免尖端再挤进螺头
    jaw_close = float(best_hit - float(squeeze))
    jaw_close = float(np.clip(jaw_close, jaw_lo, jaw_hi - 1e-4))
    # 间隙探针：再张 JAW_CLEARANCE_PROBE 仍撞 = 深穿模
    robot.goto_given_conf(jnt_values=q, ee_values=float(jaw_close + JAW_CLEARANCE_PROBE))
    if _ee_fingers_hit_object(ee, obj_cmodel):
        print(
            f"[jaw] 接触宽 {jaw_close:.4f} 仍深穿模（+{JAW_CLEARANCE_PROBE:.4f} 仍撞）→ 丢弃",
            flush=True,
        )
        return None
    print(
        f"[jaw] 方案A 接触闭合: seed={jaw_seed:.4f} → contact={best_hit:.4f} "
        f"→ close={jaw_close:.4f}",
        flush=True,
    )
    return jaw_close


def filter_grasps_grip_clearance(
    robot, grasp_collection, obj_cmodel, jaw_close, probe=None
):
    """剔除闭合附近仍深穿模的 grasp：jaw_close+probe 时指面不得撞零件。"""
    probe = float(JAW_CLEARANCE_PROBE if probe is None else probe)
    kept = gg.GraspCollection(end_effector=robot.end_effector)
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    n_drop = 0
    n_ik = 0
    seed = np.asarray(HOME_CONF, dtype=float).copy()
    jw_probe = float(jaw_close) + probe
    for g in grasp_collection:
        tcp_pos = obj_pos + obj_rot @ np.asarray(g.ac_pos, dtype=float)
        tcp_rot = obj_rot @ np.asarray(g.ac_rotmat, dtype=float)
        q = _ik_with_seed(robot, tcp_pos, tcp_rot, seed)
        if q is None:
            n_ik += 1
            continue
        seed = np.asarray(q, dtype=float).copy()
        robot.goto_given_conf(jnt_values=q, ee_values=jw_probe)
        if _ee_fingers_hit_object(robot.end_effector, obj_cmodel):
            n_drop += 1
            continue
        kept.append(g)
    print(
        f"[cd] 闭合间隙: 保留 {len(kept)}/{len(grasp_collection)} "
        f"（probe_jaw={jw_probe:.4f} 深穿模剔除 {n_drop} ik失败 {n_ik}）",
        flush=True,
    )
    return kept


def apply_jaw_close_to_grasps(grasp_collection, jaw_close):
    for g in grasp_collection:
        g.ee_values = float(jaw_close)
    return grasp_collection


def make_desk_obstacle():
    """真实桌面碰撞体：顶面 z=0（不再下沉）。穿桌主约束改用指尖 z_min。"""
    return mcm.gen_box(
        xyz_lengths=rm.vec(float(DESK_XY[0]), float(DESK_XY[1]), float(DESK_THICKNESS)),
        pos=rm.vec(0.0, 0.0, -float(DESK_THICKNESS) / 2.0),
        rgb=rm.vec(0.7, 0.7, 0.7),
        alpha=0.01,
    )


def _world_mesh_vertices(model):
    """CollisionModel 顶点 → 世界坐标。"""
    mesh = getattr(model, "trm_mesh", None)
    if mesh is None:
        return np.zeros((0, 3), dtype=float)
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.size == 0:
        return np.zeros((0, 3), dtype=float)
    R = np.asarray(model.rotmat, dtype=float)
    p = np.asarray(model.pos, dtype=float)
    return verts @ R.T + p


def grasp_prior_for_class(class_name: str) -> dict:
    """按类别关键字微调采样；不写死单个 STL。"""
    name = str(class_name or "").lower()
    # 默认：长条多点、避开两端
    prior = {
        "kind": "generic",
        "n_along": 7,
        "band_half": 0.014,
        "end_trim": 0.18,  # 长轴两端剔除比例
        "force_center": False,
        "prefer_mid": True,
        "thin_first": True,
        "max_sites": 5,
        "compact_len": 0.05,
    }
    screw_keys = ("螺丝", "螺栓", "screw", "bolt", "铆钉", "rivet")
    not_screw = ("螺丝刀", "改锥", "screwdriver")
    # 钳子单独成类：最宽处是转轴/钳口，最细处是尖嘴，两者都不是该抓的。
    plier_keys = ("钳子", "钳", "plier")
    tool_keys = (
        "扳手", "螺丝刀", "改锥",
        "扭矩", "工具", "wrench", "screwdriver", "torque", "tool",
    )
    compact_keys = ("螺母", "垫圈", "垫片", "nut", "washer", "齿轮", "gear")
    bearing_keys = ("轴承", "滚轮", "bearing", "roller", "凸轮滚轮")

    if any(k in name for k in bearing_keys):
        prior.update(
            kind="bearing",
            n_along=5,
            end_trim=0.12,
            force_center=False,
            prefer_mid=True,
            thin_first=True,
            max_sites=4,
        )
    elif any(k in name for k in screw_keys) and not any(k in name for k in not_screw):
        prior.update(
            kind="screw",
            n_along=9,
            end_trim=0.22,  # 强避开头/尾
            prefer_mid=True,
            thin_first=True,
            max_sites=6,
        )
    elif any(k in name for k in plier_keys):
        prior.update(
            kind="plier",
            n_along=9,
            end_trim=0.28,  # 尖嘴和柄尾都裁掉
            prefer_mid=True,
            thin_first=False,  # 最细的是尖嘴，不能追
            max_sites=5,
            grasp_z_frac=0.42,  # 柄厚中段，不要贴顶
        )
    elif any(k in name for k in tool_keys):
        prior.update(
            kind="tool",
            n_along=8,
            end_trim=0.18,
            prefer_mid=True,
            thin_first=True,
            max_sites=5,
        )
    elif any(k in name for k in compact_keys):
        prior.update(
            kind="compact",
            force_center=True,
            n_along=3,
            end_trim=0.1,
            prefer_mid=True,
            thin_first=False,
            max_sites=2,
            compact_len=0.08,
        )
    return prior


def part_xy_span(info) -> float:
    """水平外接尺度（近似外径/长边），用于过大零件门禁。"""
    verts = _world_mesh_vertices(info["model"])
    if verts.shape[0] < 2:
        return float(info.get("grip_width", 0.02))
    xy = verts[:, :2]
    return float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))


def _score_grasp_site(local_w, frac, height, prior):
    """分数越大越好。"""
    score = 0.0
    # 舒适宽度
    if GRASP_COMFORT_W_MIN <= local_w <= GRASP_COMFORT_W_MAX:
        score += 3.0
    elif local_w < GRASP_COMFORT_W_MIN:
        score += 0.5 + local_w / max(GRASP_COMFORT_W_MIN, 1e-6)
    elif local_w <= JAW_OPEN_MAX - 0.004:
        score += 1.5 - (local_w - GRASP_COMFORT_W_MAX) * 10.0
    else:
        score -= 5.0  # 夹不住

    if prior.get("thin_first", True):
        score += max(0.0, 0.05 - local_w) * 20.0

    # 中段优先（螺丝杆/把手）
    if prior.get("prefer_mid", True):
        score += 2.0 * (1.0 - abs(float(frac) - 0.5) * 2.0)

    # 高度：过薄略扣分但仍可用
    if height < MIN_PART_HEIGHT_TOPDOWN:
        score -= 0.5
    elif height < 0.01:
        score += 0.3
    else:
        score += 0.8
    return float(score)


def candidate_grasp_sites(info, grasp_z, prior=None, band_half=None):
    """沿零件水平长轴采样夹持点，按多因子分数排序。

    返回 list[(contact_xyz, local_width, opening_xy_unit)]。
    """
    if prior is None:
        prior = grasp_prior_for_class(info.get("class"))
    n_along = int(prior.get("n_along", 7))
    end_trim = float(prior.get("end_trim", 0.18))
    band = float(band_half if band_half is not None else prior.get("band_half", 0.014))
    compact_len = float(prior.get("compact_len", 0.05))
    max_sites = int(prior.get("max_sites", 5))

    model = info["model"]
    verts = _world_mesh_vertices(model)
    height = float(info.get("height") or info.get("top_z") or grasp_z)
    # 薄件：TCP 略抬，避免指尖贴桌穿模。钳子要抓柄中段，不能再抬。
    z_boost = 0.0
    if prior.get("kind") != "plier" and height < MIN_PART_HEIGHT_TOPDOWN * 2:
        z_boost = 0.006
    gz = float(grasp_z) + z_boost

    def _center_site(opening=None):
        op = (
            np.array([1.0, 0.0], dtype=float)
            if opening is None
            else np.asarray(opening, dtype=float)
        )
        return (
            np.array(
                [float(info["pos"][0]), float(info["pos"][1]), gz],
                dtype=float,
            ),
            float(info.get("grip_width", 0.02)),
            op,
        )

    if verts.shape[0] < 12 or prior.get("force_center"):
        site = _center_site()
        print(
            f"[grasp] 夹持点: 中心 "
            f"(prior={prior.get('kind')} w={site[1]:.3f}m)",
            flush=True,
        )
        return [site]

    xy = verts[:, :2]
    c = xy.mean(axis=0)
    xy0 = xy - c
    cov = (xy0.T @ xy0) / max(len(xy0), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = np.asarray(eigvecs[:, int(np.argmax(eigvals))], dtype=float)
    minor = np.asarray(eigvecs[:, int(np.argmin(eigvals))], dtype=float)
    mn = float(np.linalg.norm(minor))
    if mn < 1e-9:
        minor = np.array([-major[1], major[0]], dtype=float)
        mn = float(np.linalg.norm(minor))
    minor = minor / max(mn, 1e-12)

    s = xy0 @ major
    s_min, s_max = float(np.min(s)), float(np.max(s))
    length = s_max - s_min
    if length < compact_len:
        site = _center_site(minor)
        print(
            f"[grasp] 夹持点: 紧凑件中心 "
            f"(L={length:.3f}m prior={prior.get('kind')})",
            flush=True,
        )
        return [site]

    lo_f, hi_f = end_trim, 1.0 - end_trim
    raw = []
    for frac in np.linspace(lo_f, hi_f, int(max(3, n_along))):
        s0 = s_min + float(frac) * length
        mask = np.abs(s - s0) <= band
        if int(np.count_nonzero(mask)) < 6:
            mask = np.abs(s - s0) <= band * 2.0
        if int(np.count_nonzero(mask)) < 4:
            continue
        slice_v = verts[mask]
        t = (slice_v[:, :2] - c) @ minor
        local_w = float(np.max(t) - np.min(t))
        cx = float(np.mean(slice_v[:, 0]))
        cy = float(np.mean(slice_v[:, 1]))
        z_med = float(np.median(slice_v[:, 2]))
        if prior.get("kind") == "plier":
            cz = float(np.clip(min(z_med, gz), 0.006, max(0.008, 0.50 * height)))
        else:
            cz = float(np.clip(max(z_med, gz * 0.5), 0.006, max(gz, 0.012)))
        sc = _score_grasp_site(local_w, frac, height, prior)
        raw.append(
            (
                sc,
                np.array([cx, cy, cz], dtype=float),
                local_w,
                minor.copy(),
                float(frac),
            )
        )

    if not raw:
        return [_center_site(minor)]

    # 扳手/螺丝刀：最宽切片多半是头部，夹持点往对侧柄上靠。
    if prior.get("kind") == "tool" and len(raw) >= 3:
        head_frac = max(raw, key=lambda r: r[2])[4]
        rescored = []
        for sc, cxyz, lw, op, frac in raw:
            sc = float(sc) + 2.5 * abs(float(frac) - float(head_frac))
            rescored.append((sc, cxyz, lw, op, frac))
        raw = rescored

    # 钳子：最宽是转轴/钳口，最细是尖嘴。抓身子（两柄中段），两端都扣分。
    if prior.get("kind") == "plier" and len(raw) >= 3:
        head_frac = max(raw, key=lambda r: r[2])[4]
        rescored = []
        for sc, cxyz, lw, op, frac in raw:
            sc = float(sc)
            # 越靠近两端（尖嘴或柄尾）扣得越多
            endness = max(0.0, abs(float(frac) - 0.5) * 2.0 - 0.20)
            sc -= 4.5 * endness
            # 躲开最宽的钳口
            sc += 2.0 * abs(float(frac) - float(head_frac))
            # 柄的截面通常 14~32mm；再宽就是钳头
            if 0.014 <= float(lw) <= 0.032:
                sc += 2.8
            elif float(lw) > 0.036:
                sc -= 2.0
            rescored.append((sc, cxyz, lw, op, frac))
        raw = rescored

    raw.sort(key=lambda x: -x[0])
    # 可夹宽度优先保留；若全不可夹仍返回高分点供门禁报错
    fit = [r for r in raw if r[2] <= JAW_OPEN_MAX - 0.004]
    pool = fit if fit else raw
    chosen = pool[:max_sites]
    print(
        f"[grasp] 夹持点候选: {len(chosen)}/{len(raw)} "
        f"prior={prior.get('kind')} L={length:.3f}m "
        f"best_w={chosen[0][2]:.3f}m score={chosen[0][0]:.2f} "
        f"(AABB宽={float(info.get('grip_width', 0)):.3f}m)",
        flush=True,
    )
    return [(c, w, op) for _, c, w, op, _ in chosen]


def check_graspability_gate(info, sites=None):
    """可抓性门禁：过大/无可用截面则明确失败。

    返回 (ok, reason, meta)。
    """
    prior = grasp_prior_for_class(info.get("class"))
    _, grasp_z = grasp_heights_for_part(info)
    if sites is None:
        sites = candidate_grasp_sites(info, grasp_z, prior=prior)
    span = part_xy_span(info)
    height = float(info.get("height") or info.get("top_z") or 0.0)
    best_w = float(sites[0][1]) if sites else float(info.get("grip_width", 1.0))
    min_w = float(min((s[1] for s in sites), default=best_w))
    meta = {
        "prior": prior.get("kind"),
        "span": span,
        "height": height,
        "best_local_w": best_w,
        "min_local_w": min_w,
        "n_sites": len(sites or []),
    }
    print(
        f"[gate] prior={meta['prior']} span={span:.3f}m "
        f"min_local_w={min_w:.3f}m height={height:.3f}m",
        flush=True,
    )

    # 所有候选局部宽都超过硬件开口 → 不可外夹
    if min_w > JAW_HARDWARE_MAX - 0.002:
        return (
            False,
            f"零件局部最细截面 {min_w:.3f}m 仍大于夹爪上限 "
            f"{JAW_HARDWARE_MAX:.3f}m，无法外夹",
            meta,
        )
    # 水平跨度远大于开口且最细截面也夹不住仿真开口
    if span > JAW_HARDWARE_MAX * 1.15 and min_w > JAW_OPEN_MAX - 0.002:
        return (
            False,
            f"零件水平跨度 {span:.3f}m 过大且无可夹截面"
            f"（最细 {min_w:.3f}m > 仿真开口 {JAW_OPEN_MAX:.3f}m）",
            meta,
        )
    # 轴承类：跨度过大直接拒绝（避免硬抓外圆穿模）
    if prior.get("kind") == "bearing" and span > JAW_OPEN_MAX + 0.015:
        if min_w > JAW_OPEN_MAX - 0.004:
            return (
                False,
                f"轴承类零件跨度 {span:.3f}m / 最细宽 {min_w:.3f}m "
                f"超出可外夹范围",
                meta,
            )
    if height < 1e-4:
        return False, "零件高度异常，无法抓取", meta
    return True, "", meta


def filter_grasps_open_not_penetrating(
    robot, grasp_collection, obj_cmodel, jaw_open, jaw_close=None
):
    """剔除「张到夹爪上限仍穿进零件」的 grasp（TCP 扎进实体/夹在过粗处）。

    仅用 ``jaw_open`` 判断过严：截面中心 TCP 在半开时指面常贴着零件。
    """
    _ = jaw_close
    kept = gg.GraspCollection(end_effector=robot.end_effector)
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    jaw_max = float(robot.end_effector.jaw_range[1])
    probe = float(min(jaw_max, max(float(jaw_open), jaw_max * 0.95)))
    n_drop = 0
    n_ik = 0
    seed = np.asarray(HOME_CONF, dtype=float).copy()
    robot.goto_given_conf(HOME_CONF)
    for g in grasp_collection:
        tcp_pos = obj_pos + obj_rot @ np.asarray(g.ac_pos, dtype=float)
        tcp_rot = obj_rot @ np.asarray(g.ac_rotmat, dtype=float)
        q = _ik_with_seed(robot, tcp_pos, tcp_rot, seed)
        if q is None:
            n_ik += 1
            continue
        seed = np.asarray(q, dtype=float).copy()
        robot.goto_given_conf(jnt_values=q, ee_values=probe)
        if _ee_fingers_hit_object(robot.end_effector, obj_cmodel):
            n_drop += 1
            continue
        kept.append(g)
    print(
        f"[cd] 张满不穿模: 保留 {len(kept)}/{len(grasp_collection)} "
        f"（probe={probe:.3f} 穿模剔除 {n_drop} ik失败 {n_ik}）",
        flush=True,
    )
    return kept


def adapt_pickle_grasps_to_part(src_gc, info, robot, sites=None, prior=None):
    """把参考块 pickle 的开合方位迁到目标零件（多夹持点，按评分顺序）。"""
    if prior is None:
        prior = grasp_prior_for_class(info.get("class"))
    _, grasp_z = grasp_heights_for_part(info)
    if sites is None:
        sites = candidate_grasp_sites(info, grasp_z, prior=prior)
    # 用最高分截面估开合（已按分数排序）
    best_w = float(sites[0][1]) if sites else float(info.get("grip_width", 0.02))
    info_local = dict(info)
    info_local["grip_width"] = float(
        np.clip(best_w, 0.004, max(float(info.get("grip_width", best_w)), best_w))
    )
    jaw_open, jaw_close = jaw_widths_for_part(info_local)
    model = info["model"]
    gripper = robot.end_effector
    world_gc = gg.GraspCollection(end_effector=gripper)

    max_sites = min(int(prior.get("max_sites", 5)), len(sites))
    for contact, local_w, open_xy in sites[:max_sites]:
        if local_w > JAW_OPEN_MAX - 0.002:
            continue
        jc = float(
            np.clip(max(local_w * 0.98, JAW_CLOSE_MIN), JAW_CLOSE_MIN, jaw_open - 1e-4)
        )
        opening_pref = np.array(
            [float(open_xy[0]), float(open_xy[1]), 0.0], dtype=float
        )
        on = float(np.linalg.norm(opening_pref))
        if on < 1e-9:
            opening_pref = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            opening_pref = opening_pref / on

        # 只保留沿薄方向的开合（沿长轴夹会夹到整段扳手）
        opening_list = [opening_pref, -opening_pref]
        for g in src_gc:
            opening = np.asarray(g.ac_rotmat, dtype=float)[:, 0].copy()
            opening = np.array([opening[0], opening[1], 0.0], dtype=float)
            n = np.linalg.norm(opening)
            if n < 1e-9:
                continue
            opening = opening / n
            if abs(float(np.dot(opening[:2], opening_pref[:2]))) < 0.65:
                continue
            opening_list.append(opening)

        seen = []
        for opening in opening_list:
            if any(abs(float(np.dot(opening[:2], s[:2]))) > 0.98 for s in seen):
                continue
            seen.append(opening)
            approaching = np.array([0.0, 0.0, -1.0], dtype=float)
            grasp = gripper.grip_at_by_twovecs(
                jaw_center_pos=contact,
                approaching_direction=approaching,
                thumb_opening_direction=opening,
                jaw_width=jc,
            )
            world_gc.append(grasp)

    obj_gc = gg.grasp_collection_world_to_object_frame(
        world_gc,
        np.asarray(model.pos, dtype=float),
        np.asarray(model.rotmat, dtype=float),
    )
    print(
        f"[grasp] 顶抓适配: n={len(obj_gc)} jaw_close≈{jaw_close:.4f} "
        f"sites={max_sites} best_local_w={best_w:.3f} (源yaw×点)",
        flush=True,
    )
    return obj_gc, jaw_open, jaw_close, grasp_z


def adapt_antipodal_grasps_to_part(
    src_gc,
    info,
    robot,
    max_n=MAX_SIDE_GRASPS_ADAPT,
    prefer_side=True,
    sites=None,
    prior=None,
):
    """对踵/侧向抓取适配：多夹持点 + 保留接近方向（不压成纯顶抓）。"""
    if prior is None:
        prior = grasp_prior_for_class(info.get("class"))
    _, grasp_z = grasp_heights_for_part(info)
    if sites is None:
        sites = candidate_grasp_sites(info, grasp_z, prior=prior)
    best_w = float(sites[0][1]) if sites else float(info.get("grip_width", 0.02))
    info_local = dict(info)
    info_local["grip_width"] = float(
        np.clip(best_w, 0.004, max(float(info.get("grip_width", best_w)), best_w))
    )
    jaw_open, jaw_close = jaw_widths_for_part(info_local)
    model = info["model"]
    gripper = robot.end_effector
    items = list(src_gc)
    pos_xy = np.asarray(info["pos"][:2], dtype=float)
    r_xy = float(np.linalg.norm(pos_xy))
    r_hat = pos_xy / r_xy if r_xy > 1e-6 else np.array([1.0, 0.0])
    if prefer_side:

        def _side_key(g):
            R = np.asarray(g.ac_rotmat, dtype=float)
            app = R[:, 2]
            return (
                abs(float(R[2, 2])),
                float(app[0] * r_hat[0] + app[1] * r_hat[1]),
            )

        items.sort(key=_side_key)
    max_sites = min(int(prior.get("max_sites", 4)), len(sites))
    per_site = max(8, int(max_n) // max(1, max_sites))
    if len(items) > per_site:
        items = items[:per_site]

    world_gc = gg.GraspCollection(end_effector=gripper)
    for contact, local_w, open_xy in sites[:max_sites]:
        if local_w > JAW_OPEN_MAX - 0.002:
            continue
        jc = float(
            np.clip(max(local_w * 0.98, JAW_CLOSE_MIN), JAW_CLOSE_MIN, jaw_open - 1e-4)
        )
        open_pref = np.array(
            [float(open_xy[0]), float(open_xy[1]), 0.0], dtype=float
        )
        onp = float(np.linalg.norm(open_pref))
        if onp > 1e-9:
            open_pref /= onp
        for g in items:
            approaching = np.asarray(g.ac_rotmat, dtype=float)[:, 2].copy()
            opening = np.asarray(g.ac_rotmat, dtype=float)[:, 0].copy()
            an = float(np.linalg.norm(approaching))
            on = float(np.linalg.norm(opening))
            if an < 1e-9 or on < 1e-9:
                continue
            approaching /= an
            opening /= on
            opening = opening - approaching * float(np.dot(opening, approaching))
            on = float(np.linalg.norm(opening))
            if on < 1e-9:
                continue
            opening /= on
            # 侧抓也偏好沿薄方向开合，避免沿长轴劈进模型
            if onp > 1e-9 and abs(float(np.dot(opening[:2], open_pref[:2]))) < 0.35:
                continue
            grasp = gripper.grip_at_by_twovecs(
                jaw_center_pos=contact,
                approaching_direction=approaching,
                thumb_opening_direction=opening,
                jaw_width=float(jc),
            )
            world_gc.append(grasp)

    obj_gc = gg.grasp_collection_world_to_object_frame(
        world_gc,
        np.asarray(model.pos, dtype=float),
        np.asarray(model.rotmat, dtype=float),
    )
    print(
        f"[grasp] 侧向/对踵适配: n={len(obj_gc)} jaw_close≈{jaw_close:.4f} "
        f"sites={max_sites} best_local_w={best_w:.3f} "
        f"(每点≤{per_site}/{len(src_gc)})",
        flush=True,
    )
    return obj_gc, jaw_open, jaw_close, grasp_z


def filter_grasps_ik_neighbors_desk(
    robot, grasp_collection, obj_cmodel, jaw_width, neighbor_models=None
):
    """侧抓预筛：IK 有解 + 指尖 z≥0 + 夹爪不撞邻件。"""
    neighbor_models = neighbor_models or []
    kept = gg.GraspCollection(end_effector=robot.end_effector)
    n_ik = n_desk = n_nb = 0
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    tip_floor = TIP_FLOOR_STRICT - TIP_TOUCH_EPS
    robot.goto_given_conf(HOME_CONF)
    seed = np.asarray(HOME_CONF, dtype=float).copy()
    for g in grasp_collection:
        tcp_pos = obj_pos + obj_rot @ np.asarray(g.ac_pos, dtype=float)
        tcp_rot = obj_rot @ np.asarray(g.ac_rotmat, dtype=float)
        q = _ik_with_seed(robot, tcp_pos, tcp_rot, seed)
        if q is None:
            n_ik += 1
            continue
        seed = np.asarray(q, dtype=float).copy()
        robot.goto_given_conf(jnt_values=q, ee_values=float(jaw_width))
        z_min = finger_tips_z_min(robot.end_effector)
        if np.isfinite(z_min) and float(z_min) < tip_floor:
            n_desk += 1
            continue
        hit_nb = False
        for nb in neighbor_models:
            if _ee_fingers_hit_object(robot.end_effector, nb):
                hit_nb = True
                break
        if hit_nb:
            n_nb += 1
            continue
        kept.append(
            gg.Grasp(
                ee_values=g.ee_values,
                ac_pos=np.asarray(g.ac_pos, dtype=float).copy(),
                ac_rotmat=np.asarray(g.ac_rotmat, dtype=float).copy(),
            )
        )
    print(
        f"[cd] 侧抓预筛: 保留 {len(kept)}/{len(grasp_collection)} "
        f"（ik失败={n_ik} 穿桌={n_desk} 撞邻件={n_nb}；要求 tip_z≥0）",
        flush=True,
    )
    return kept


def filter_grasps_clear_neighbors(
    robot, grasp_collection, obj_cmodel, jaw_width, neighbor_models=None
):
    """顶抓阶段附加：去掉夹爪与邻件碰撞的 grasp。"""
    neighbor_models = neighbor_models or []
    if not neighbor_models:
        return grasp_collection
    kept = gg.GraspCollection(end_effector=robot.end_effector)
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    n_drop = 0
    robot.goto_given_conf(HOME_CONF)
    for g in grasp_collection:
        tcp_pos = obj_pos + obj_rot @ np.asarray(g.ac_pos, dtype=float)
        tcp_rot = obj_rot @ np.asarray(g.ac_rotmat, dtype=float)
        q = robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot, seed_jnt_values=HOME_CONF)
        if q is None:
            n_drop += 1
            continue
        robot.goto_given_conf(jnt_values=q, ee_values=float(jaw_width))
        if any(_ee_fingers_hit_object(robot.end_effector, nb) for nb in neighbor_models):
            n_drop += 1
            continue
        kept.append(g)
    print(
        f"[cd] 邻件预筛: 保留 {len(kept)}/{len(grasp_collection)}（剔除 {n_drop}）",
        flush=True,
    )
    return kept


def prepare_grasps_topdown_then_side(
    robot,
    target,
    part_infos,
    desk_obstacle,
    topdown_pickle,
    antipodal_pickle,
    force_mode=None,
):
    """先顶抓，失败则对踵侧抓；返回 grasps/jaw/mode/neighbors。

    force_mode:
      None — 顶抓优先，失败再侧抓（默认）
      \"topdown\" — 只用顶抓
      \"side\" — 跳过顶抓，只用侧抓/对踵
    """
    neighbors = [
        p["model"]
        for p in part_infos
        if p.get("id") != target.get("id") and p.get("model") is not None
    ]
    prior = grasp_prior_for_class(target.get("class"))
    _, gz0 = grasp_heights_for_part(target)
    sites = candidate_grasp_sites(target, gz0, prior=prior)
    ok_gate, gate_reason, _gate_meta = check_graspability_gate(target, sites=sites)
    if not ok_gate:
        raise RuntimeError(f"可抓性门禁拒绝: {gate_reason}")
    force = str(force_mode or "").strip().lower() or None
    if force not in (None, "topdown", "side"):
        raise ValueError(f"force_mode 无效: {force_mode}")

    def _finalize(part_grasps, jaw_open, jaw_close, grasp_z, mode):
        if len(part_grasps) == 0:
            return None
        jaw_seed = float(jaw_close)
        prior_kind = str(prior.get("kind") or "")
        squeeze = (
            float(JAW_CONTACT_SQUEEZE_SCREW)
            if prior_kind == "screw"
            else float(JAW_CONTACT_SQUEEZE)
        )
        configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)
        # 逐条 grasp 找可用接触宽；张开仍穿模的直接丢掉
        hits = []
        for g in part_grasps:
            jc = refine_jaw_close_by_contact(
                robot,
                target["model"],
                g,
                jaw_open=JAW_OPEN_MAX,
                jaw_seed=jaw_seed,
                squeeze=squeeze,
            )
            if jc is None:
                continue
            hits.append((g, float(jc)))
        if not hits:
            print("[jaw] 全部 grasp 接触搜索失败（穿模/无解）", flush=True)
            return None
        real_hits = [h for h in hits if h[1] > JAW_CLOSE_MIN + 1e-4]
        pool = real_hits if real_hits else hits
        refined = float(np.median([jc for _, jc in pool]))
        jaw_close = sane_jaw_close(refined, target, sites)
        kept_for_refine = gg.GraspCollection(end_effector=robot.end_effector)
        for g, _jc in pool:
            g.ee_values = float(jaw_close)
            kept_for_refine.append(g)
        part_grasps = kept_for_refine
        apply_jaw_close_to_grasps(part_grasps, jaw_close)
        jaw_open = approach_jaw_width(target, part_infos, jaw_close)
        configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)
        jaw_open = float(min(jaw_open, float(robot.end_effector.jaw_range[1])))
        jaw_close = float(
            np.clip(
                jaw_close,
                JAW_CLOSE_MIN,
                float(robot.end_effector.jaw_range[1]) - 1e-4,
            )
        )
        apply_jaw_close_to_grasps(part_grasps, jaw_close)
        robot.goto_given_conf(HOME_CONF)
        part_grasps = filter_grasps_open_not_penetrating(
            robot,
            part_grasps,
            target["model"],
            jaw_open=jaw_open,
            jaw_close=jaw_close,
        )
        part_grasps = filter_grasps_grip_clearance(
            robot, part_grasps, target["model"], jaw_close=jaw_close
        )
        if mode == "topdown":
            part_grasps = lift_grasps_for_fingertip_desk(
                robot, part_grasps, target["model"], jaw_width=jaw_open
            )
            # 抬升后姿态变了：再验间隙，并按抬升后姿态重估夹距
            part_grasps = filter_grasps_grip_clearance(
                robot, part_grasps, target["model"], jaw_close=jaw_close
            )
            if len(part_grasps) > 0:
                jc2 = refine_jaw_close_by_contact(
                    robot,
                    target["model"],
                    part_grasps[0],
                    jaw_open=JAW_OPEN_MAX,
                    jaw_seed=jaw_close,
                    squeeze=squeeze,
                )
                if jc2 is not None:
                    jaw_close = sane_jaw_close(jc2, target, sites)
                    apply_jaw_close_to_grasps(part_grasps, jaw_close)
                    part_grasps = filter_grasps_grip_clearance(
                        robot, part_grasps, target["model"], jaw_close=jaw_close
                    )
            part_grasps = filter_grasps_against_desk(
                robot, part_grasps, target["model"], desk_obstacle
            )
        else:
            part_grasps = filter_grasps_ik_neighbors_desk(
                robot,
                part_grasps,
                target["model"],
                jaw_width=jaw_open,
                neighbor_models=neighbors,
            )
        part_grasps = filter_grasps_clear_neighbors(
            robot,
            part_grasps,
            target["model"],
            jaw_width=jaw_open,
            neighbor_models=neighbors,
        )
        if len(part_grasps) == 0:
            return None
        g0 = part_grasps[0]
        grasp_z = float(
            (
                np.asarray(target["model"].pos, dtype=float)
                + np.asarray(target["model"].rotmat, dtype=float)
                @ np.asarray(g0.ac_pos, dtype=float)
            )[2]
        )
        return part_grasps, jaw_open, jaw_close, grasp_z, mode, neighbors

    # ----- A. 顶抓 -----
    if force != "side":
        print(f"[grasp] 阶段A 顶抓 pickle={topdown_pickle}", flush=True)
        src_top = gg.GraspCollection.load_from_disk(topdown_pickle)
        part_grasps, jaw_open, jaw_close, grasp_z = adapt_pickle_grasps_to_part(
            src_top, target, robot, sites=sites, prior=prior
        )
        if len(part_grasps) > 0:
            robot.goto_given_conf(HOME_CONF)
            part_grasps = filter_grasps_open_not_penetrating(
                robot,
                part_grasps,
                target["model"],
                jaw_open=jaw_open,
                jaw_close=jaw_close,
            )
            configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)
            part_grasps = lift_grasps_for_fingertip_desk(
                robot, part_grasps, target["model"], jaw_width=JAW_OPEN_MAX
            )
            part_grasps = filter_grasps_clear_neighbors(
                robot,
                part_grasps,
                target["model"],
                jaw_width=jaw_open,
                neighbor_models=neighbors,
            )
            done = _finalize(part_grasps, jaw_open, jaw_close, grasp_z, "topdown")
            if done is not None:
                print("[grasp] 使用顶抓成功", flush=True)
                return done
        if force == "topdown":
            raise RuntimeError("强制顶抓失败（无可用 grasp / IK/穿桌/邻件）")
        print("[grasp] 顶抓不可用，进入阶段B 对踵侧向/倾斜…", flush=True)
    else:
        print("[grasp] 强制侧抓：跳过顶抓", flush=True)

    # ----- B. 侧向/对踵 -----
    print(f"[grasp] 阶段B 对踵 pickle={antipodal_pickle}", flush=True)
    src_side = gg.GraspCollection.load_from_disk(antipodal_pickle)
    part_grasps, jaw_open, jaw_close, grasp_z = adapt_antipodal_grasps_to_part(
        src_side,
        target,
        robot,
        prefer_side=True,
        sites=sites,
        prior=prior,
    )
    if len(part_grasps) == 0:
        raise RuntimeError("侧向/对踵适配后无 grasp")
    robot.goto_given_conf(HOME_CONF)
    part_grasps = filter_grasps_open_not_penetrating(
        robot,
        part_grasps,
        target["model"],
        jaw_open=jaw_open,
        jaw_close=jaw_close,
    )
    if len(part_grasps) == 0:
        # 对踵 pickle 的倾斜接近在桌面低位经常全无 IK。补一组竖直顶抓方向。
        print("[grasp] 侧向接近全无 IK，补竖直开口方向", flush=True)
        vert_gc, jaw_open, jaw_close, grasp_z = adapt_pickle_grasps_to_part(
            gg.GraspCollection.load_from_disk(topdown_pickle),
            target,
            robot,
            sites=sites,
            prior=prior,
        )
        configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)
        vert_gc = lift_grasps_for_fingertip_desk(
            robot, vert_gc, target["model"], jaw_width=JAW_OPEN_MAX
        )
        part_grasps = filter_grasps_open_not_penetrating(
            robot,
            vert_gc,
            target["model"],
            jaw_open=max(float(jaw_open), JAW_OPEN_MAX * 0.95),
            jaw_close=jaw_close,
        )
    part_grasps = filter_grasps_ik_neighbors_desk(
        robot,
        part_grasps,
        target["model"],
        jaw_width=max(float(jaw_open), JAW_OPEN_MAX * 0.95),
        neighbor_models=neighbors,
    )
    done = _finalize(part_grasps, jaw_open, jaw_close, grasp_z, "side")
    if done is None:
        if force == "side":
            raise RuntimeError("强制侧抓失败（IK/穿桌/邻件碰撞）")
        raise RuntimeError("顶抓与侧向抓取均失败（IK/穿桌/邻件碰撞）")
    print("[grasp] 使用侧向/对踵抓取成功", flush=True)
    return done


def filter_grasps_against_desk(robot, grasp_collection, obj_cmodel, desk):
    """按指尖 z_min 筛：允许轻触 z=0，拒绝深入穿桌。"""
    _ = desk
    kept = gg.GraspCollection(end_effector=robot.end_effector)
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    n_reject = 0
    for grasp in grasp_collection:
        tcp_pos = obj_pos + obj_rot @ np.asarray(grasp.ac_pos, dtype=float)
        tcp_rot = obj_rot @ np.asarray(grasp.ac_rotmat, dtype=float)
        q = robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
        if q is None:
            n_reject += 1
            continue
        robot.goto_given_conf(jnt_values=q, ee_values=grasp.ee_values)
        tip_z = finger_tips_z_min(robot.end_effector)
        floor = PLIER_TIP_FLOOR if _part_is_plier(obj_cmodel) else (DESK_TOP_Z - TIP_PENETRATE_MAX)
        if tip_z < floor:
            n_reject += 1
            continue
        kept.append(grasp)
    print(
        f"[cd] 指尖桌面预筛: 保留 {len(kept)}/{len(grasp_collection)}（剔除 {n_reject}）",
        flush=True,
    )
    return kept


def tip_clears_desk(robot, floor=None) -> bool:
    """当前构型指尖最低点是否不低于桌面（默认 z≥0，容差 TIP_TOUCH_EPS）。"""
    if floor is None:
        floor = TIP_FLOOR_STRICT
    tip_z = finger_tips_z_min(robot.end_effector)
    if not np.isfinite(tip_z):
        return True
    return float(tip_z) >= float(floor) - TIP_TOUCH_EPS


def conf_tip_clears_desk(robot, jnt_values, jaw_width, floor=None) -> bool:
    robot.goto_given_conf(
        jnt_values=np.asarray(jnt_values, dtype=float),
        ee_values=float(jaw_width),
    )
    return tip_clears_desk(robot, floor=floor)


def motion_penetrates_desk(
    robot, mot, desk, jaw_fallback=None, floor=None, sample_all=False
):
    """轨迹抽检：指尖低于桌面则判穿桌。

    ``floor`` 默认宽松（顶抓兼容 -2mm）；侧抓请传 ``TIP_FLOOR_STRICT`` 且
    ``sample_all=True`` 逐帧检查。
    """
    _ = desk
    if mot is None:
        return False
    if floor is None:
        floor = DESK_TOP_Z - TIP_PENETRATE_MAX
    n = len(mot.jv_list)
    step = 1 if sample_all else max(1, n // 24)
    for i in range(0, n, step):
        jv = mot.jv_list[i]
        ev = mot.ev_list[i] if i < len(mot.ev_list) else jaw_fallback
        if ev is not None and len(robot.end_effector.oiee_list) == 0:
            robot.goto_given_conf(jnt_values=jv, ee_values=ev)
        else:
            robot.goto_given_conf(jnt_values=jv)
        tip_z = finger_tips_z_min(robot.end_effector)
        if np.isfinite(tip_z) and tip_z < float(floor) - TIP_TOUCH_EPS:
            return True
    return False


def strip_motion_meshes(mot):
    if mot is None or not hasattr(mot, "mesh_list"):
        return
    for i, mesh in enumerate(list(mot.mesh_list)):
        if mesh is None:
            continue
        try:
            mesh.detach()
        except Exception:
            pass
        for cm in list(getattr(mesh, "cm_list", [])):
            try:
                cm.remove()
            except Exception:
                pass
        for gm in list(getattr(mesh, "gm_list", [])):
            try:
                gm.remove()
            except Exception:
                pass
        if hasattr(mesh, "cm_list"):
            mesh.cm_list.clear()
        if hasattr(mesh, "gm_list"):
            mesh.gm_list.clear()
        mot.mesh_list[i] = None


def release_model_collection(mesh):
    if mesh is None:
        return
    try:
        mesh.detach()
    except Exception:
        pass
    for cm in list(getattr(mesh, "cm_list", [])):
        try:
            cm.remove()
        except Exception:
            pass
    for gm in list(getattr(mesh, "gm_list", [])):
        try:
            gm.remove()
        except Exception:
            pass
    if hasattr(mesh, "cm_list"):
        mesh.cm_list.clear()
    if hasattr(mesh, "gm_list"):
        mesh.gm_list.clear()


def wait_for_ui_animation_start(args) -> None:
    """Block autoplay until the UI has attached the Panda3D window."""
    start_file = getattr(args, "wait_start_file", "") or ""
    if not start_file:
        return
    print("[sim] 动画就绪: 等待 UI 贴合窗口后开始播放", flush=True)
    started_at = time.time()
    while not os.path.exists(start_file):
        if time.time() - started_at > 2.5:
            print("[sim] UI 启动信号等待超时，直接开始播放动画", flush=True)
            return
        time.sleep(0.05)
    try:
        os.remove(start_file)
    except Exception:
        pass
    print("[sim] UI 贴合完成，开始播放动画", flush=True)


def configure_hidden_start_window(args) -> None:
    """只有真机 ``--export-traj``（且不播动画）才离屏。

    界面 ``--wait-start-file`` 不要把规划窗藏到屏外：加载机械臂碰撞网格时
    没有正常 GPU 窗口会卡很久，看起来像死在「加载机械臂」。
    """
    export_only = bool(getattr(args, "export_traj", "") or "") and not bool(
        getattr(args, "auto_play", False)
    )
    if not export_only:
        return
    try:
        from panda3d.core import loadPrcFileData
    except Exception:
        return
    loadPrcFileData(
        "",
        "\n".join(
            [
                "window-type offscreen",
                "win-origin -32000 -32000",
                "undecorated 1",
                "win-size 64 64",
                "sync-video 0",
            ]
        ),
    )


def render_planning_preview(base, robot):
    preview_mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
    preview_mesh.attach_to(base)
    try:
        for _ in range(6):
            base.graphicsEngine.renderFrame()
    except Exception:
        pass
    return preview_mesh


def safe_home_jaw_width(robot, preferred: float = 0.055) -> float:
    try:
        lo, hi = robot.end_effector.jaw_range
        return float(np.clip(preferred, float(lo), float(hi) - 1e-4))
    except Exception:
        return float(preferred)


def _pick_approach_direction(pick_tcp_rot, grasp_mode):
    """顶抓沿世界 -z；侧抓沿 TCP 接近轴（hand-z），避免侧向姿态硬竖直下压。"""
    if grasp_mode == "side":
        return None  # ADPlanner：None → goal_tcp_rotmat 的 loc_z
    return -rm.const.z_ax


def _side_approach_distances():
    """侧抓线性接近距离候选（近基座区只试短行程）。"""
    return (0.02, 0.03)


def _jv_lerp_motion(robot, q_start, q_goal, ee_value, n_steps=12):
    """无碰撞检查的关节插值（侧抓近基座时 ADPlanner 插值极慢/易卡）。"""
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_goal, dtype=float)
    jv_list = []
    ev_list = []
    for t in np.linspace(0.0, 1.0, int(max(2, n_steps))):
        jv_list.append((1.0 - t) * q0 + t * q1)
        ev_list.append(float(ee_value))
    mot = mmd.MotionData(robot)
    # mesh_list=[]：不生成 mesh，避免规划阶段极慢
    mot.extend(jv_list, ev_list=ev_list, mesh_list=[])
    return mot


def _jv_path_tip_safe(
    robot, q_start, q_goal, jaw_width, desk, via_tcp_pos=None, via_tcp_rot=None
):
    """关节插值；若中段穿桌，经高位 via 点绕行，保证 tip_z≥0。"""
    direct = _jv_lerp_motion(robot, q_start, q_goal, jaw_width, n_steps=16)
    if not motion_penetrates_desk(
        robot,
        direct,
        desk,
        jaw_fallback=jaw_width,
        floor=TIP_FLOOR_STRICT,
        sample_all=True,
    ):
        return direct
    if via_tcp_pos is None or via_tcp_rot is None:
        return None
    q_via = _ik_with_seed(robot, via_tcp_pos, via_tcp_rot, q_start)
    if q_via is None:
        q_via = _ik_with_seed(robot, via_tcp_pos, via_tcp_rot, q_goal)
    if q_via is None:
        return None
    if not conf_tip_clears_desk(robot, q_via, jaw_width):
        return None
    a = _jv_lerp_motion(robot, q_start, q_via, jaw_width, n_steps=12)
    b = _jv_lerp_motion(robot, q_via, q_goal, jaw_width, n_steps=12)
    full = a + b
    if motion_penetrates_desk(
        robot,
        full,
        desk,
        jaw_fallback=jaw_width,
        floor=TIP_FLOOR_STRICT,
        sample_all=True,
    ):
        return None
    return full


def _ik_tcp_cleared(robot, tcp_pos, tcp_rot, jaw_width, seed, z_boost_max=0.06):
    """求 IK；若指尖穿桌则把 TCP 沿 +z 抬高再试。"""
    pos = np.asarray(tcp_pos, dtype=float).copy()
    base_z = float(pos[2])
    seed = np.asarray(seed, dtype=float).copy()
    for boost in np.linspace(0.0, float(z_boost_max), 7):
        pos[2] = base_z + float(boost)
        q = _ik_with_seed(robot, pos, tcp_rot, seed)
        if q is None:
            continue
        if conf_tip_clears_desk(robot, q, jaw_width):
            return np.asarray(q, dtype=float), pos.copy()
        seed = np.asarray(q, dtype=float).copy()
    return None, None


def _repair_conf_tip(robot, q, jaw_width, seed_home=None):
    """穿桌构型修复：抬 TCP / 向 home 混合，使 tip_z≥0。"""
    q = np.asarray(q, dtype=float).copy()
    if conf_tip_clears_desk(robot, q, jaw_width):
        return q
    if seed_home is None:
        seed_home = HOME_CONF
    seed_home = np.asarray(seed_home, dtype=float)
    robot.goto_given_conf(q, ee_values=float(jaw_width))
    tip = finger_tips_z_min(robot.end_effector)
    p, Rr = robot.end_effector.gl_acting_center_pose
    p = np.asarray(p, dtype=float)
    Rr = np.asarray(Rr, dtype=float)
    need = max(0.003, (-float(tip) if np.isfinite(tip) else 0.02) + 0.003)
    robot.goto_given_conf(seed_home, ee_values=float(jaw_width))
    _, home_R = robot.end_effector.gl_acting_center_pose
    home_R = np.asarray(home_R, dtype=float)
    for rot_try in (Rr, home_R, _topdown_rotmat(0.0)):
        for dz in np.linspace(need, need + 0.08, 10):
            pp = p.copy()
            pp[2] += float(dz)
            q2 = _ik_with_seed(robot, pp, rot_try, q)
            if q2 is None:
                continue
            if conf_tip_clears_desk(robot, q2, jaw_width):
                return np.asarray(q2, dtype=float)
    for a in np.linspace(0.1, 0.85, 10):
        q2 = (1.0 - a) * q + a * seed_home
        if conf_tip_clears_desk(robot, q2, jaw_width):
            return np.asarray(q2, dtype=float)
    return None


def _joint_approach_tip_safe(robot, q_grasp, jaw_width, desk, n_samples=30):
    """home→抓取：关节插值 + 穿桌点修复，全程 tip_z≥0。"""
    q0 = np.asarray(HOME_CONF, dtype=float)
    q1 = np.asarray(q_grasp, dtype=float)
    if not conf_tip_clears_desk(robot, q1, jaw_width):
        return None
    waypoints = []
    for t in np.linspace(0.0, 1.0, int(max(8, n_samples))):
        q = (1.0 - t) * q0 + t * q1
        if conf_tip_clears_desk(robot, q, jaw_width):
            waypoints.append(np.asarray(q, dtype=float))
            continue
        q_fix = _repair_conf_tip(robot, q, jaw_width, seed_home=q0)
        if q_fix is None:
            return None
        waypoints.append(q_fix)
    if np.linalg.norm(waypoints[-1] - q1) > 1e-6:
        waypoints.append(q1.copy())

    q_chain = [waypoints[0]]
    for nxt in waypoints[1:]:
        prev = q_chain[-1]
        mid = 0.5 * (prev + nxt)
        if not conf_tip_clears_desk(robot, mid, jaw_width):
            mid_fix = _repair_conf_tip(robot, mid, jaw_width, seed_home=q0)
            if mid_fix is None:
                return None
            q_chain.append(mid_fix)
        q_chain.append(np.asarray(nxt, dtype=float))

    mot = mmd.MotionData(robot)
    mot.extend(
        q_chain,
        ev_list=[float(jaw_width)] * len(q_chain),
        mesh_list=[],
    )
    if motion_penetrates_desk(
        robot,
        mot,
        desk,
        jaw_fallback=jaw_width,
        floor=TIP_FLOOR_STRICT,
        sample_all=True,
    ):
        return None
    return mot


def _raise_tcp_until_tip_clear(
    robot, tcp_pos, tcp_rot, jaw_width, seed_jnt, max_lift=0.04
):
    """沿 +z 抬 TCP，直到指尖 z≥0；返回 (q, tcp_pos) 或 (None, None)。"""
    pos = np.asarray(tcp_pos, dtype=float).copy()
    rot = np.asarray(tcp_rot, dtype=float)
    seed = np.asarray(seed_jnt, dtype=float).copy()
    q = _ik_with_seed(robot, pos, rot, seed)
    if q is None:
        return None, None
    if conf_tip_clears_desk(robot, q, jaw_width):
        return q, pos
    lifted = 0.0
    while lifted < float(max_lift):
        lifted += FINGERTIP_LIFT_STEP
        pos[2] = float(tcp_pos[2]) + lifted
        q = _ik_with_seed(robot, pos, rot, seed)
        if q is None:
            continue
        seed = np.asarray(q, dtype=float).copy()
        if conf_tip_clears_desk(robot, q, jaw_width):
            return q, pos
    return None, None


def plan_side_pick_lift(
    robot, obj_cmodel, grasp_collection, jaw_open, jaw_close, desk, sink_m=0.0
):
    """侧抓专用：短 hand-z 接近 + 闭合 + 抬起；全程指尖 z≥0。"""
    if len(grasp_collection) > MAX_PPP_GRASPS:
        trimmed = gg.GraspCollection(end_effector=robot.end_effector)
        for i, g in enumerate(grasp_collection):
            if i >= MAX_PPP_GRASPS:
                break
            trimmed.append(g)
        grasp_collection = trimmed

    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    print(
        f"[ppp] side-pick 快速规划 grasps={len(grasp_collection)} "
        f"（约束 tip_z≥{TIP_FLOOR_STRICT}）…",
        flush=True,
    )
    t0 = time.time()
    for gi, grasp in enumerate(grasp_collection):
        print(f"[ppp] try grasp#{gi} …", flush=True)
        robot.goto_given_conf(HOME_CONF, ee_values=float(jaw_open))
        pick_tcp_pos = obj_rot @ np.asarray(grasp.ac_pos, dtype=float) + obj_pos
        pick_tcp_rot = obj_rot @ np.asarray(grasp.ac_rotmat, dtype=float)
        app = np.asarray(pick_tcp_rot[:, 2], dtype=float)

        q_grasp, pick_tcp_pos = _raise_tcp_until_tip_clear(
            robot, pick_tcp_pos, pick_tcp_rot, jaw_open, HOME_CONF
        )
        if q_grasp is None:
            print(f"[ppp] grasp#{gi} 抓取位指尖无法抬到 z≥0", flush=True)
            continue
        # 闭合时指尖也可能更低，再验一次
        if not conf_tip_clears_desk(robot, q_grasp, jaw_close):
            q2, pos2 = _raise_tcp_until_tip_clear(
                robot, pick_tcp_pos, pick_tcp_rot, jaw_close, q_grasp
            )
            if q2 is None:
                print(f"[ppp] grasp#{gi} 闭合后指尖穿桌", flush=True)
                continue
            q_grasp, pick_tcp_pos = q2, pos2

        pick_mot = _joint_approach_tip_safe(
            robot, q_grasp, jaw_open, desk, n_samples=30
        )
        if pick_mot is None:
            print(f"[ppp] grasp#{gi} 接近段无法保证 tip_z≥0", flush=True)
            continue
        if sink_m > 0:
            pick_mot = append_tcp_sink(robot, pick_mot, jaw_open, sink_m)
            robot.goto_given_conf(
                jnt_values=np.asarray(pick_mot.jv_list[-1], dtype=float),
                ee_values=float(jaw_open),
            )
            pick_tcp_pos = np.asarray(robot.gl_tcp_pos, dtype=float).copy()
        q_grasp = np.asarray(pick_mot.jv_list[-1], dtype=float).copy()

        pick_mot.extend(
            [np.asarray(pick_mot.jv_list[-1], dtype=float).copy()],
            [float(jaw_close)],
            mesh_list=[],
        )
        tip_floor = PLIER_TIP_FLOOR if sink_m > 0 else None
        if not conf_tip_clears_desk(
            robot, pick_mot.jv_list[-1], jaw_close, floor=tip_floor
        ):
            print(f"[ppp] grasp#{gi} 闭合帧穿桌", flush=True)
            continue

        # 近基座侧抓：+z / 回撤，且抬起段也不得穿桌
        depart = None
        for lift in (0.04, 0.06, 0.08, 0.03, float(PICK_DEPART_DIST)):
            up_pos = pick_tcp_pos + np.array([0.0, 0.0, float(lift)])
            q_up = _ik_with_seed(robot, up_pos, pick_tcp_rot, q_grasp)
            if q_up is None:
                continue
            if not conf_tip_clears_desk(robot, q_up, jaw_close):
                continue
            dep = _jv_lerp_motion(robot, q_grasp, q_up, jaw_close, n_steps=8)
            if motion_penetrates_desk(
                robot,
                dep,
                desk,
                jaw_fallback=jaw_close,
                floor=TIP_FLOOR_STRICT,
                sample_all=True,
            ):
                continue
            depart = dep
            break
        if depart is None:
            for retreat in (0.03, 0.05):
                back_pos = pick_tcp_pos - app * float(retreat)
                back_pos[2] = max(float(back_pos[2]), float(pick_tcp_pos[2]) + 0.02)
                q_back = _ik_with_seed(robot, back_pos, pick_tcp_rot, q_grasp)
                if q_back is None:
                    continue
                if not conf_tip_clears_desk(robot, q_back, jaw_close):
                    continue
                dep = _jv_lerp_motion(
                    robot, q_grasp, q_back, jaw_close, n_steps=8
                )
                if motion_penetrates_desk(
                    robot,
                    dep,
                    desk,
                    jaw_fallback=jaw_close,
                    floor=TIP_FLOOR_STRICT,
                    sample_all=True,
                ):
                    continue
                depart = dep
                break
        if depart is None:
            print(
                f"[ppp] grasp#{gi} 无抬起解，仅用接近+闭合",
                flush=True,
            )
            mot = pick_mot
        else:
            mot = pick_mot + depart

        if motion_penetrates_desk(
            robot,
            mot,
            desk,
            jaw_fallback=jaw_open,
            floor=TIP_FLOOR_STRICT,
            sample_all=True,
        ):
            print(f"[ppp] grasp#{gi} 全轨迹穿桌，丢弃", flush=True)
            continue

        tip_samples = []
        for i in range(0, len(mot.jv_list), max(1, len(mot.jv_list) // 8)):
            ev = mot.ev_list[i] if i < len(mot.ev_list) else jaw_open
            robot.goto_given_conf(mot.jv_list[i], ee_values=float(ev))
            tip_samples.append(finger_tips_z_min(robot.end_effector))
        tip_min = min(tip_samples) if tip_samples else float("nan")
        print(
            f"[ppp] OK grasp#{gi} frames={len(mot)} "
            f"tip_z_min≈{tip_min:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        return mot
    print(f"[ppp] side-pick 全部失败 ({time.time() - t0:.1f}s)", flush=True)
    return None


def plan_pick_lift(
    robot,
    obj_cmodel,
    grasp_collection,
    jaw_open,
    jaw_close,
    obstacles,
    desk,
    use_rrt=True,
    grasp_mode="topdown",
    start_conf=None,
    sink_m=0.0,
):
    """PPP 风格 pick：approach → 闭合 → +z 抬起（不要求 place common-grasp）。

    ``gen_pick_and_place`` 需要抓取/放置共有 grasp；API ``action=pick`` 无放置盒，
    远位零件抬起位姿常无 common-gid。这里逐条 grasp 做 approach+depart。
    ``start_conf``：可选起始关节（如缓冲松爪末态），默认 HOME。
    """
    if len(grasp_collection) > MAX_PPP_GRASPS:
        trimmed = gg.GraspCollection(end_effector=robot.end_effector)
        for i, g in enumerate(grasp_collection):
            if i >= MAX_PPP_GRASPS:
                break
            trimmed.append(g)
        grasp_collection = trimmed

    planner = ppp.PickPlacePlanner(robot)
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    q_start0 = (
        np.asarray(start_conf, dtype=float).copy()
        if start_conf is not None
        else np.asarray(HOME_CONF, dtype=float).copy()
    )
    app_dists = (
        _side_approach_distances()
        if grasp_mode == "side"
        else (0.12, PICK_APPROACH_DIST, 0.05, 0.03)
    )

    def _after_grasp(pick_mot, rrt_flag):
        if sink_m > 0:
            pick_mot = append_tcp_sink(robot, pick_mot, jaw_open, sink_m)
        pick_mot.extend(
            [pick_mot.jv_list[-1]],
            [float(jaw_close)],
            [None],
        )
        depart = planner.gen_linear_depart_from_given_conf(
            start_jnt_values=pick_mot.jv_list[-1],
            direction=rm.const.z_ax,
            distance=PICK_DEPART_DIST,
            ee_values=None,
            granularity=LINEAR_GRANULARITY,
            obstacle_list=obstacles,
        )
        if depart is None:
            return None
        # 侧抓：抬起即可，省去回 home（插值/RRT 在近基座区极慢且易卡死）
        # 从缓冲末态续抓：也不强制回 HOME，交给后续 place 接
        if grasp_mode == "side" or start_conf is not None:
            return pick_mot + depart
        retract = None
        if rrt_flag:
            retract = planner.rrtc_planner.plan(
                start_conf=depart.jv_list[-1],
                goal_conf=HOME_CONF,
                obstacle_list=obstacles,
                max_time=4.0,
                smoothing_n_iter=50,
            )
        if retract is None:
            retract = planner.im_planner.gen_interplated_between_given_conf(
                start_jnt_values=depart.jv_list[-1],
                end_jnt_values=HOME_CONF,
                obstacle_list=obstacles,
                ee_values=None,
            )
        if retract is None:
            return pick_mot + depart
        return pick_mot + depart + retract

    def _try_one(grasp, rrt_flag):
        robot.goto_given_conf(q_start0)
        pick_tcp_pos = obj_rot @ np.asarray(grasp.ac_pos, dtype=float) + obj_pos
        pick_tcp_rot = obj_rot @ np.asarray(grasp.ac_rotmat, dtype=float)
        app_dir = _pick_approach_direction(pick_tcp_rot, grasp_mode)
        obj_list = [] if grasp_mode == "side" else [obj_cmodel]
        for dist in app_dists:
            pick_mot = planner.gen_approach(
                goal_tcp_pos=pick_tcp_pos,
                goal_tcp_rotmat=pick_tcp_rot,
                start_jnt_values=q_start0,
                linear_direction=app_dir,
                linear_distance=float(dist),
                ee_values=jaw_open,
                granularity=LINEAR_GRANULARITY,
                obstacle_list=obstacles,
                object_list=obj_list,
                use_rrt=rrt_flag,
            )
            if pick_mot is None:
                continue
            full = _after_grasp(pick_mot, rrt_flag)
            if full is not None:
                return full
        # 侧抓：直达抓取构型（目标物不进障碍，避免张开夹爪被判碰零件）
        if grasp_mode != "side":
            return None
        q_goal = robot.ik(
            tgt_pos=pick_tcp_pos,
            tgt_rotmat=pick_tcp_rot,
            seed_jnt_values=q_start0,
        )
        if q_goal is None:
            return None
        robot.goto_given_conf(q_start0, ee_values=float(jaw_open))
        if rrt_flag:
            start2grasp = planner.rrtc_planner.plan(
                start_conf=q_start0,
                goal_conf=q_goal,
                obstacle_list=list(obstacles),
                max_time=3.0,
                smoothing_n_iter=20,
            )
        else:
            start2grasp = planner.im_planner.gen_interplated_between_given_conf(
                start_jnt_values=q_start0,
                end_jnt_values=q_goal,
                obstacle_list=list(obstacles),
                ee_values=float(jaw_open),
            )
        if start2grasp is None:
            return None
        return _after_grasp(start2grasp, rrt_flag)

    def _accept(mot):
        if mot is None:
            return None
        floor = DESK_TOP_Z - (abs(PLIER_TIP_FLOOR) if sink_m > 0 else TIP_PENETRATE_MAX)
        if motion_penetrates_desk(
            robot, mot, desk, jaw_fallback=jaw_open, floor=floor
        ):
            print("[cd] 轨迹夹爪穿桌，丢弃该 grasp", flush=True)
            return None
        return mot

    # 侧抓：只用直线（RRT 近基座易误报 goal collision 且极慢）
    if grasp_mode == "side":
        rrt_flags = [False]
    else:
        rrt_flags = [True, False] if use_rrt else [False]
    print(
        f"[ppp] pick 规划（grasps={len(grasp_collection)}, "
        f"mode={grasp_mode}, rrt_order={rrt_flags}）…",
        flush=True,
    )
    t0 = time.time()
    for rrt_flag in rrt_flags:
        if grasp_mode == "side" and rrt_flag:
            print("[ppp] 直线失败，改 RRT…", flush=True)
        elif grasp_mode != "side" and not rrt_flag and use_rrt:
            print("[ppp] RRT 全失败，改直线…", flush=True)
        for gi, grasp in enumerate(grasp_collection):
            print(f"[ppp] try grasp#{gi} rrt={rrt_flag} …", flush=True)
            mot = _accept(_try_one(grasp, rrt_flag))
            if mot is not None:
                print(
                    f"[ppp] OK grasp#{gi} rrt={rrt_flag} frames={len(mot)} "
                    f"({time.time() - t0:.1f}s)",
                    flush=True,
                )
                strip_motion_meshes(mot)
                gc.collect()
                return mot
            print(
                f"[ppp] grasp#{gi} rrt={rrt_flag} 失败，下一条…",
                flush=True,
            )
    print(f"[ppp] 全部失败 ({time.time() - t0:.1f}s)", flush=True)
    return None


def _reset_robot_hand(robot, jaw_open):
    if len(robot.end_effector.oiee_list) > 0:
        robot.end_effector.release_all()
    robot.goto_given_conf(HOME_CONF, ee_values=float(jaw_open))


def plan_pick_place(
    robot,
    obj_cmodel,
    grasp_collection,
    jaw_open,
    jaw_close,
    place_pose,
    obstacles,
    desk,
    use_rrt=True,
    grasp_mode="topdown",
    place_tcp_z=None,
    place_approach_dist=None,
    check_desk_penetration=True,
    place_obj_rot=None,
    align_tcp_to_place_obj=False,
    release_obj_z=None,
    release_tcp_floor=None,
    place_hover_z=None,
    place_base_conf=None,
    start_conf=None,
    sink_m=0.0,
    fast=False,
):
    """与 ``plan_pick_lift`` 相同的 pick 段，再追加 place（不用 gen_pick_and_place）。

    原因：``gen_pick_and_place`` 的 common-grasp 常选到与仅 pick 不同的条；
    钻头等零件上那些 grasp 在张开时已撞零件，动画会表现为夹爪穿模。

    place_tcp_z / place_approach_dist: 可选覆盖默认放置高度与下探距离
    （小料盘如 EC box 需更低，避免高空松爪像“掉下去”）。
    check_desk_penetration: 为 False 时跳过整段穿桌检查（仅调试用）。
    place_obj_rot: 期望放置时物体的世界旋转；配合 align_tcp_to_place_obj
    时按 grasp 反推 TCP，使持物在空中转到该姿态后再松爪。
    release_obj_z: align 时先到 place_pose 悬停调姿，再尽量下探到该物体高度后松爪。
    release_tcp_floor: 调姿后下探允许的最低 TCP z（默认 0.055；EC 小料盘可更低）。
    place_hover_z: 若给定，先高位（该 z）正着到位（优先世界 -Z 接近），再短距下探。
    place_base_conf: 放置主体构型（J2–J6）；J1 仍按格子方位改写，作为 via/IK 种子。
    fast: 仿真界面连放加速（更粗步长、更少 grasp/接近距离）；真机导出不要开。
    """
    max_grasps = 2 if fast else MAX_PPP_GRASPS
    gran = 0.09 if fast else LINEAR_GRANULARITY
    if len(grasp_collection) > max_grasps:
        trimmed = gg.GraspCollection(end_effector=robot.end_effector)
        for i, g in enumerate(grasp_collection):
            if i >= max_grasps:
                break
            trimmed.append(g)
        grasp_collection = trimmed

    place_pos, place_rot = place_pose
    place_pos = np.asarray(place_pos, dtype=float)
    place_rot = np.asarray(place_rot, dtype=float)
    if place_obj_rot is None and place_rot is not None:
        # place_pose 的旋转若非单位阵，视为期望物体姿态
        if float(np.linalg.norm(place_rot - np.eye(3))) > 1e-6:
            place_obj_rot = place_rot
    app_dist = float(
        PLACE_APPROACH_DIST if place_approach_dist is None else place_approach_dist
    )
    # 未按物体姿态对齐时：沿用固定 TCP 高度；对齐时 TCP 由 grasp+物体位姿推出
    tcp_z = float(PLACE_TCP_Z if place_tcp_z is None else place_tcp_z)
    align_tcp = bool(align_tcp_to_place_obj and place_obj_rot is not None)
    planner = ppp.PickPlacePlanner(robot)
    obj_pos = np.asarray(obj_cmodel.pos, dtype=float)
    obj_rot = np.asarray(obj_cmodel.rotmat, dtype=float)
    q_start0 = (
        np.asarray(start_conf, dtype=float).copy()
        if start_conf is not None
        else np.asarray(HOME_CONF, dtype=float).copy()
    )

    def _accept(mot):
        if mot is None:
            return None
        floor = DESK_TOP_Z - (abs(PLIER_TIP_FLOOR) if sink_m > 0 else TIP_PENETRATE_MAX)
        if check_desk_penetration and motion_penetrates_desk(
            robot, mot, desk, jaw_fallback=jaw_open, floor=floor
        ):
            print("[cd] 轨迹夹爪穿桌，丢弃", flush=True)
            return None
        return mot

    def _pick_segment(grasp, rrt_flag):
        """与 plan_pick_lift 一致：approach(open) → close → +z depart。"""
        _reset_robot_hand(robot, jaw_open)
        robot.goto_given_conf(q_start0, ee_values=float(jaw_open))
        pick_tcp_pos = obj_rot @ np.asarray(grasp.ac_pos, dtype=float) + obj_pos
        pick_tcp_rot = obj_rot @ np.asarray(grasp.ac_rotmat, dtype=float)
        app_dir = _pick_approach_direction(pick_tcp_rot, grasp_mode)
        app_dists = (
            _side_approach_distances()
            if grasp_mode == "side"
            else ((PICK_APPROACH_DIST, 0.05) if fast else (0.12, PICK_APPROACH_DIST, 0.05, 0.03))
        )
        pick_mot = None
        obj_list = [] if grasp_mode == "side" else [obj_cmodel]
        for dist in app_dists:
            pick_mot = planner.gen_approach(
                goal_tcp_pos=pick_tcp_pos,
                goal_tcp_rotmat=pick_tcp_rot,
                start_jnt_values=q_start0,
                linear_direction=app_dir,
                linear_distance=float(dist),
                ee_values=float(jaw_open),
                granularity=gran,
                obstacle_list=obstacles,
                object_list=obj_list,
                use_rrt=rrt_flag,
            )
            if pick_mot is not None:
                break
        if pick_mot is None and grasp_mode == "side":
            q_goal = robot.ik(
                tgt_pos=pick_tcp_pos,
                tgt_rotmat=pick_tcp_rot,
                seed_jnt_values=q_start0,
            )
            if q_goal is not None:
                robot.goto_given_conf(q_start0, ee_values=float(jaw_open))
                if rrt_flag:
                    pick_mot = planner.rrtc_planner.plan(
                        start_conf=q_start0,
                        goal_conf=q_goal,
                        obstacle_list=list(obstacles),
                        max_time=6.0,
                        smoothing_n_iter=50,
                    )
                else:
                    pick_mot = planner.im_planner.gen_interplated_between_given_conf(
                        start_jnt_values=q_start0,
                        end_jnt_values=q_goal,
                        obstacle_list=list(obstacles),
                        ee_values=float(jaw_open),
                    )
        if pick_mot is None:
            return None
        if sink_m > 0:
            pick_mot = append_tcp_sink(robot, pick_mot, jaw_open, sink_m)
        pick_mot.extend(
            [pick_mot.jv_list[-1]],
            [float(jaw_close)],
            [None],
        )
        # ee_values=None 时 MotionData 会写入 robot.get_ee_values()；
        # 先同步为闭合，避免后续帧被误标成张开。
        robot.goto_given_conf(pick_mot.jv_list[-1], ee_values=float(jaw_close))
        depart = planner.gen_linear_depart_from_given_conf(
            start_jnt_values=pick_mot.jv_list[-1],
            direction=rm.const.z_ax,
            distance=PICK_DEPART_DIST,
            ee_values=None,
            granularity=gran,
            obstacle_list=obstacles,
        )
        if depart is None:
            return None
        return pick_mot + depart

    def _place_segment(grasp, start_jnt, rrt_flag):
        """抬起后平移到盒上方放下。

        盒坐标本身可达；失败通常是「同一 grasp 姿态在放置点无 IK」。
        align_tcp 时：TCP = place_obj_rot @ grasp，持物在途中转到竖直再松爪。
        """
        ac_pos = np.asarray(grasp.ac_pos, dtype=float)
        ac_rot = np.asarray(grasp.ac_rotmat, dtype=float)
        pick_tcp_rot = obj_rot @ ac_rot
        rot_cands = place_tcp_rot_candidates(
            grasp,
            obj_rot,
            pick_rot=pick_tcp_rot,
            place_obj_rot=place_obj_rot if align_tcp else None,
        )
        # 与 rot_cands 对齐的 TCP 位置；align 时按物体目标位姿+grasp 推
        pos_cands = []
        R_obj_cands = []
        if align_tcp and place_obj_rot is not None:
            R_base = np.asarray(place_obj_rot, dtype=float)
            yaw_list = (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi, 0.25 * np.pi, -0.25 * np.pi)
            for dyaw in yaw_list:
                R_obj = rm.rotmat_from_euler(0.0, 0.0, float(dyaw)) @ R_base
                tcp_p = R_obj @ ac_pos + place_pos
                if place_tcp_z is not None and float(tcp_p[2]) < float(tcp_z):
                    tcp_p = tcp_p + np.array(
                        [0.0, 0.0, float(tcp_z) - float(tcp_p[2])], dtype=float
                    )
                pos_cands.append(tcp_p)
                R_obj_cands.append(R_obj)
            # 只保留「物体竖直」候选，避免退化成平躺平移后松爪换 mesh
            rot_cands = list(rot_cands[: len(yaw_list)])
            pos_cands = pos_cands[: len(rot_cands)]
            R_obj_cands = R_obj_cands[: len(rot_cands)]
            # 优先：TCP 靠近格心，且接近轴更接近世界 -Z（“正着”）
            def _cand_key(i):
                xy_err = float(np.linalg.norm(pos_cands[i][:2] - place_pos[:2]))
                # rot 第 3 列多为接近方向；越接近 -Z 越好
                down = -float(rot_cands[i][2, 2])
                return (xy_err, -down)

            order = sorted(range(len(pos_cands)), key=_cand_key)
            rot_cands = [rot_cands[i] for i in order]
            pos_cands = [pos_cands[i] for i in order]
            R_obj_cands = [R_obj_cands[i] for i in order]
            # 高位正着到位：把接近点抬到 place_hover_z
            if place_hover_z is not None:
                hz = float(place_hover_z)
                for i in range(len(pos_cands)):
                    if float(pos_cands[i][2]) < hz:
                        pos_cands[i] = pos_cands[i].copy()
                        pos_cands[i][2] = hz
        else:
            hover_z = float(
                tcp_z
                if place_hover_z is None
                else max(float(tcp_z), float(place_hover_z))
            )
            fixed = np.array(
                [float(place_pos[0]), float(place_pos[1]), hover_z],
                dtype=float,
            )
            pos_cands = [fixed] * len(rot_cands)
            R_obj_cands = [None] * len(rot_cands)
            if fast:
                rot_cands = rot_cands[:2]
                pos_cands = pos_cands[:2]
                R_obj_cands = R_obj_cands[:2]

        robot.goto_given_conf(start_jnt, ee_values=float(jaw_close))
        place_mot = None
        used_R_obj = None
        used_tcp_rot = None
        # 高位正着：先 J1 转到目标方位，再优先世界 -Z 接近下探
        approach_start = np.asarray(start_jnt, dtype=float).copy()
        via_face = None
        if place_hover_z is not None or place_base_conf is not None:
            q_face = facing_j1_home(
                place_pos[:2],
                home=place_base_conf,
                stretch_defaults=place_base_conf is None,
            )
            print(
                f"[ppp] 放置主体构型 via j1={np.rad2deg(q_face[0]):.1f}° "
                f"q[1:]=[{', '.join(f'{np.rad2deg(v):.0f}' for v in q_face[1:])}]deg "
                f"再接近格子",
                flush=True,
            )
            via_face = planner.im_planner.gen_interplated_between_given_conf(
                start_jnt_values=start_jnt,
                end_jnt_values=q_face,
                obstacle_list=obstacles,
                ee_values=float(jaw_close),
            )
            if via_face is None and rrt_flag:
                via_face = planner.rrtc_planner.plan(
                    start_conf=start_jnt,
                    goal_conf=q_face,
                    obstacle_list=obstacles,
                    max_time=3.0,
                    smoothing_n_iter=30,
                )
            if via_face is not None:
                approach_start = q_face
            else:
                print("[ppp] J1 转向过渡失败，从 pick 末构型直接接近", flush=True)
            app_dirs = (-rm.const.z_ax, None)
        elif align_tcp:
            app_dirs = (None, -rm.const.z_ax)
        else:
            app_dirs = (-rm.const.z_ax,)
        j1_tgt = float(np.arctan2(float(place_pos[1]), float(place_pos[0])))
        for prefer_j1 in ((True, False) if place_hover_z is not None else (False,)):
            for ri, place_tcp_rot in enumerate(rot_cands):
                place_tcp_pos = pos_cands[ri]
                if prefer_j1:
                    q_goal = robot.ik(
                        tgt_pos=place_tcp_pos,
                        tgt_rotmat=place_tcp_rot,
                        seed_jnt_values=approach_start,
                    )
                    if q_goal is not None:
                        j1_err = abs(
                            float(
                                np.arctan2(
                                    np.sin(q_goal[0] - j1_tgt),
                                    np.cos(q_goal[0] - j1_tgt),
                                )
                            )
                        )
                        # 拒绝明显折回去的支解，优先 J1 朝向目标
                        if j1_err > np.deg2rad(55.0):
                            continue
                for app_dir in app_dirs:
                    place_mot = planner.gen_approach(
                        goal_tcp_pos=place_tcp_pos,
                        goal_tcp_rotmat=place_tcp_rot,
                        start_jnt_values=approach_start,
                        linear_direction=app_dir,
                        linear_distance=float(app_dist),
                        ee_values=None,
                        granularity=gran,
                        obstacle_list=obstacles,
                        object_list=[],
                        use_rrt=rrt_flag,
                    )
                    if place_mot is not None:
                        used_R_obj = R_obj_cands[ri]
                        used_tcp_rot = place_tcp_rot
                        break
                if place_mot is not None:
                    if via_face is not None:
                        place_mot = via_face + place_mot
                    print(
                        f"[ppp] place 候选#{ri} @ "
                        f"xy={np.round(place_tcp_pos[:2], 4).tolist()} "
                        f"z={float(place_tcp_pos[2]):.3f}"
                        f"{' align_obj' if align_tcp else ''}"
                        f"{' via_J1' if via_face is not None else ''}"
                        f"{' prefer_J1' if prefer_j1 else ''}",
                        flush=True,
                    )
                    break
            if place_mot is not None:
                break
        if place_mot is None:
            return None

        # 调姿到位后沿 -Z 逐步 IK 下探靠近料盘，再松爪（短步，避免长直线卡死）
        if (
            align_tcp
            and release_obj_z is not None
            and float(release_obj_z) < float(place_pos[2]) - 1e-4
        ):
            try:
                q_cur = np.asarray(place_mot.jv_list[-1], dtype=float)
                robot.goto_given_conf(q_cur, ee_values=float(jaw_close))
                tcp0 = np.asarray(robot.gl_tcp_pos, dtype=float).copy()
                tcp_r = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
                hover_tcp_z = float(tcp0[2])
                floor_z = (
                    0.055
                    if release_tcp_floor is None
                    else float(release_tcp_floor)
                )
                goal_tcp_z = max(float(release_obj_z) + 0.012, floor_z)
                tip_floor = DESK_TOP_Z - TIP_TOUCH_EPS  # 不允许穿桌
                best_q = q_cur
                best_tcp_z = hover_tcp_z
                # 每次降 5mm；IK 成功且指尖仍≥桌面才接受
                z_try = hover_tcp_z
                while z_try > goal_tcp_z + 1e-4:
                    z_next = max(goal_tcp_z, z_try - 0.005)
                    tcp_p = tcp0.copy()
                    tcp_p[2] = float(z_next)
                    q_new = robot.ik(
                        tgt_pos=tcp_p,
                        tgt_rotmat=tcp_r,
                        seed_jnt_values=best_q,
                    )
                    if q_new is None:
                        break
                    q_new = np.asarray(q_new, dtype=float)
                    robot.goto_given_conf(q_new, ee_values=float(jaw_close))
                    tip_z = finger_tips_z_min(robot.end_effector)
                    if np.isfinite(tip_z) and float(tip_z) < float(tip_floor):
                        print(
                            f"[ppp] 下探止于指尖贴桌 tip_z={float(tip_z):.4f} "
                            f"(tcp_z={z_next:.3f} 不采用)",
                            flush=True,
                        )
                        break
                    best_q = q_new
                    best_tcp_z = float(z_next)
                    z_try = z_next
                if best_tcp_z < hover_tcp_z - 1e-4:
                    n_lerp = max(8, int(round((hover_tcp_z - best_tcp_z) / 0.005)))
                    descend = _jv_lerp_motion(
                        robot, q_cur, best_q, jaw_close, n_steps=n_lerp
                    )
                    place_mot = place_mot + descend
                    print(
                        f"[ppp] 调姿后下探靠近 box: tcp_z "
                        f"{hover_tcp_z:.3f} → {best_tcp_z:.3f} 再松爪",
                        flush=True,
                    )
                else:
                    print(
                        f"[ppp] 无法下探(目标 tcp_z≤{goal_tcp_z:.3f})，"
                        f"当前 {hover_tcp_z:.3f} 松爪",
                        flush=True,
                    )
            except Exception as e:
                print(f"[ppp] 下探异常，悬停松爪: {e}", flush=True)

        place_mot.extend(
            [place_mot.jv_list[-1]],
            [float(jaw_open)],
            [None],
        )
        # 同步到松爪构型，便于读 TCP；低位不再做耗时直线撤离/回 HOME
        try:
            robot.goto_given_conf(
                place_mot.jv_list[-1], ee_values=float(jaw_open)
            )
            tcp_now_z = float(np.asarray(robot.gl_tcp_pos, dtype=float)[2])
        except Exception:
            tcp_now_z = float(tcp_z)
        if align_tcp and tcp_now_z < 0.09:
            print(
                f"[ppp] 低位松爪 tcp_z={tcp_now_z:.3f}，跳过撤离/回 HOME",
                flush=True,
            )
            return place_mot
        if not rrt_flag:
            print("[ppp] stable mode: 跳过耗时松爪直线搜索，先求顶部撤离 IK…", flush=True)
            try:
                q_release = np.asarray(place_mot.jv_list[-1], dtype=float)
                robot.goto_given_conf(q_release, ee_values=float(jaw_open))
                tcp_up = np.asarray(robot.gl_tcp_pos, dtype=float).copy()
                tcp_rot = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
                tcp_up[2] += 0.06
                q_up = robot.ik(
                    tgt_pos=tcp_up,
                    tgt_rotmat=tcp_rot,
                    seed_jnt_values=q_release,
                )
            except Exception:
                q_up = None
            if q_up is None:
                q_up = facing_j1_home(place_pos[:2], stretch_defaults=True)
                print("[ppp] stable mode: 精确顶部 IK 未收敛，改用近似顶部安全撤离构型", flush=True)
            top_depart = _jv_lerp_motion(
                robot,
                place_mot.jv_list[-1],
                q_up,
                jaw_open,
                n_steps=8,
            )
            print("[ppp] stable mode: 顶部撤离 ok，连续任务不回 HOME", flush=True)
            return place_mot + top_depart
        release_depart = None
        for d in (float(PLACE_DEPART_DIST), 0.04, 0.02):
            print(f"[ppp] 尝试松爪撤离 +Z {float(d):.3f}m…", flush=True)
            release_depart = planner.gen_linear_depart_from_given_conf(
                start_jnt_values=place_mot.jv_list[-1],
                direction=rm.const.z_ax,
                distance=float(d),
                ee_values=float(jaw_open),
                granularity=gran,
                obstacle_list=obstacles,
            )
            if release_depart is not None:
                print(f"[ppp] 松爪撤离 +Z {float(d):.3f}m ok", flush=True)
                break
        if release_depart is None:
            print("[ppp] 松爪撤离失败，仅返回放置段", flush=True)
            return place_mot
        retract = None
        if rrt_flag:
            retract = planner.rrtc_planner.plan(
                start_conf=release_depart.jv_list[-1],
                goal_conf=HOME_CONF,
                obstacle_list=obstacles,
                max_time=3.0,
                smoothing_n_iter=30,
            )
        if retract is None:
            print("[ppp] 插值回 HOME…", flush=True)
            retract = planner.im_planner.gen_interplated_between_given_conf(
                start_jnt_values=release_depart.jv_list[-1],
                end_jnt_values=HOME_CONF,
                obstacle_list=obstacles,
                ee_values=float(jaw_open),
            )
        full = place_mot + release_depart
        if retract is not None:
            full = full + retract
        else:
            print("[ppp] 回 HOME 未完成，返回放置+撤离", flush=True)
        return full

    print(
        f"[ppp] pick→place 规划（grasps={len(grasp_collection)}, mode={grasp_mode}, "
        f"rrt={use_rrt}）→ place_xy={np.round(place_pos[:2], 4).tolist()} "
        f"tcp_z={tcp_z:.3f} approach={app_dist:.3f}"
        f"{' align_obj=True' if align_tcp else ''}",
        flush=True,
    )
    t0 = time.time()
    # 侧抓优先直线；place 仍可 RRT/直线降级
    if grasp_mode == "side":
        pick_rrt_flags = [False, True] if use_rrt else [False]
    else:
        pick_rrt_flags = [True, False] if use_rrt else [False]
    place_rrt_flags = [True, False] if use_rrt else [False]
    for pick_rrt in pick_rrt_flags:
        if use_rrt and not pick_rrt:
            print("[ppp] RRT pick 全失败，改直线 pick…", flush=True)
        for gi, grasp in enumerate(grasp_collection):
            pick = _pick_segment(grasp, pick_rrt)
            if pick is None:
                print(
                    f"[ppp] grasp#{gi} pick_rrt={pick_rrt} 失败，下一条…",
                    flush=True,
                )
                continue
            placed = False
            for place_rrt in place_rrt_flags:
                place = _place_segment(grasp, pick.jv_list[-1], place_rrt)
                mot = _accept(pick + place) if place is not None else None
                _reset_robot_hand(robot, jaw_open)
                if mot is not None:
                    print(
                        f"[ppp] OK grasp#{gi} pick_rrt={pick_rrt} "
                        f"place_rrt={place_rrt} frames={len(mot)} "
                        f"({time.time() - t0:.1f}s)",
                        flush=True,
                    )
                    try:
                        mot.grasp_ac_pos = np.asarray(grasp.ac_pos, dtype=float).copy()
                        mot.grasp_ac_rotmat = np.asarray(grasp.ac_rotmat, dtype=float).copy()
                    except Exception:
                        pass
                    strip_motion_meshes(mot)
                    gc.collect()
                    return mot
            print(
                f"[ppp] grasp#{gi} pick_rrt={pick_rrt} 成功但 place 失败…",
                flush=True,
            )
    print(f"[ppp] pick-place 全部失败 ({time.time() - t0:.1f}s)", flush=True)
    return None


def find_carry_range(mot, jaw_close, jaw_open=None):
    """闭合夹爪起 hold；张开帧须出现在一段非开爪帧之后（避免误判接近段）。

    用「足够闭合」判定（≤ open/close 中点），避免 grasp 实际 ee_values 与
    集合中位 jaw_close 差 >0.5mm 时漏掉桌面闭爪、误匹配到空中二次闭合。
    """
    cs = None
    jc = float(jaw_close)
    jo = float(jaw_open) if jaw_open is not None else None
    # 闭合阈值：略宽于精确匹配，兼容 per-grasp 微调宽
    if jo is not None and jo > jc + 1e-6:
        closed_max = 0.5 * (jo + jc)
    else:
        closed_max = jc + 5e-4

    def _ev_closed(ev) -> bool:
        if ev is None:
            return False
        try:
            return float(ev) <= closed_max + 1e-9
        except (TypeError, ValueError):
            return False

    def _ev_open(ev) -> bool:
        if ev is None or jo is None:
            return False
        try:
            return abs(float(ev) - jo) < 5e-3
        except (TypeError, ValueError):
            return False

    for i, ev in enumerate(mot.ev_list):
        if _ev_closed(ev):
            cs = i
            break
    if cs is None:
        cs = max(0, len(mot.jv_list) // 3)
    ce = len(mot.jv_list) - 1
    release_idx = None
    if jo is not None:
        seen_gap = False  # 闭合后至少经历 None/闭合，才认后续张开为 place
        for i in range(cs + 1, len(mot.ev_list)):
            ev = mot.ev_list[i]
            if ev is None:
                seen_gap = True
                continue
            if _ev_closed(ev):
                seen_gap = True
                continue
            if seen_gap and _ev_open(ev):
                release_idx = i
                ce = max(cs, i - 1)
                break
    return cs, ce, release_idx


def export_planned_trajectory(
    out_path: str,
    sid,
    task_path: str,
    segments: list[dict],
    source: str = "run_agent_pick_place_side_sim",
) -> None:
    """把规划好的关节轨迹写成纯数值 JSON，供真机执行模块 ``tiaozhanbei/real`` 读取。

    ``segments`` 每项需带 ``mot``（``mmd.MotionData``）与该段的夹爪/carry 信息。
    导出内容里 ``jv_list`` 是 6 关节弧度，``ev_list`` 是仿真夹爪开口（米），
    真机侧再做限位裁剪与开口→夹爪角度换算。

    ``source`` 标明轨迹是哪个仿真脚本规划的（标准抓放 / EC 螺丝入格），
    真机侧只做记录，不改变执行逻辑。
    """
    payload = {
        "source": source,
        "robot": "panthera_ht_6dof",
        "joint_unit": "rad",
        "gripper_unit": "sim_jaw_width_m",
        "sample_id": str(sid),
        "task": os.path.abspath(task_path) if task_path else "",
        "home_conf": np.asarray(HOME_CONF, dtype=float).tolist(),
        "segments": [],
    }
    total_frames = 0
    for seg in segments:
        mot = seg["mot"]
        jv_list = [np.asarray(q, dtype=float).ravel().tolist() for q in mot.jv_list]
        ev_raw = list(getattr(mot, "ev_list", []) or [])
        ev_list = []
        for i in range(len(jv_list)):
            v = ev_raw[i] if i < len(ev_raw) else None
            ev_list.append(None if v is None else float(v))
        release_idx = seg.get("release_idx")
        payload["segments"].append(
            {
                "index": int(seg.get("index", 1)),
                "total": int(seg.get("total", len(segments))),
                "object": str(seg.get("object") or ""),
                "grasp_mode": str(seg.get("grasp_mode") or ""),
                "do_place": bool(seg.get("do_place", False)),
                "jaw_open": float(seg.get("jaw_open", JAW_OPEN_MAX)),
                "jaw_close": float(seg.get("jaw_close", JAW_CLOSE_MIN)),
                "carry_start": int(seg.get("carry_start", 0)),
                "carry_end": int(seg.get("carry_end", len(jv_list) - 1)),
                "release_idx": None if release_idx is None else int(release_idx),
                "jv_list": jv_list,
                "ev_list": ev_list,
            }
        )
        total_frames += len(jv_list)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[export] 轨迹已导出 -> {out_path} "
        f"segments={len(payload['segments'])} frames={total_frames}",
        flush=True,
    )


def _target_ann_for_task_in_sample(ann: dict, task: dict, sid: str) -> dict:
    cls = str(task.get("object") or "")
    coord = np.asarray(task.get("coordinate") or [0, 0, 0], dtype=float)
    best = None
    best_dist = float("inf")
    for obj in ann.get("objects") or []:
        if obj.get("class") != cls:
            continue
        pose = obj.get("pose_6d") or {}
        xy = np.array([float(pose.get("x", 0.0)), float(pose.get("y", 0.0))])
        dist = float(np.linalg.norm(xy - coord[:2]))
        if dist < best_dist:
            best = obj
            best_dist = dist
    if best is None or best_dist >= 0.03:
        raise RuntimeError(f"样本 {sid} 中找不到任务零件: {cls}")
    return best


def run_multi_pick_place_in_one_world(args, source_task: dict, tasks: list[dict]) -> None:
    first_task = tasks[0]
    if args.sample_id:
        sid = args.sample_id
        ann_path = os.path.join(args.dataset_root, sid, "annotations.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
    else:
        sid, _first_ann, ann = find_sample_for_task(args.dataset_root, first_task)

    print(
        f"[task] single-world batch pick-place targets={len(tasks)} "
        f"source_action={source_task.get('action')}",
        flush=True,
    )
    for i, task in enumerate(tasks, start=1):
        print(
            f"[task]   {i}/{len(tasks)} object={task.get('object')} "
            f"coord={np.round(np.asarray(task.get('coordinate') or [0, 0, 0], dtype=float), 4).tolist()}",
            flush=True,
        )

    print(f"[grasp] topdown={args.grasp_pickle}  side={args.side_grasp_pickle}", flush=True)

    configure_hidden_start_window(args)
    base = wd.World(
        cam_pos=[1.65, -1.25, 1.05],
        lookat_pos=[0.03, -0.03, 0.08],
        w=1920,
        h=1080,
    )
    sys.modules["__main__"].base = base
    mgm.gen_frame(ax_length=0.12).attach_to(base)

    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style=args.stripe, seed=0)
    desk_obstacle = make_desk_obstacle()

    part_infos = load_parts_from_annotations(ann, highlight_id=None)
    for info in part_infos:
        info["model"].attach_to(base)

    do_place = any(has_place_target(task) for task in tasks)
    box_parts = []
    box_state = None
    if do_place:
        box_xy = resolve_box_xy(first_task)
        box_state = BoxPlacementState(box_xy)
        box_parts = make_storage_box(box_xy)
        for bp in box_parts:
            bp.attach_to(base)
        print(
            f"[scene] 收纳盒 size={STORAGE_BOX_SIZE.tolist()} "
            f"@ xy={np.round(box_xy, 4).tolist()}",
            flush=True,
        )

    robot = PantheraHTSglArm(enable_cc=True)
    relax_panthera_joint_ranges_for_demo(robot)
    if getattr(args, "joint_ranges", ""):
        apply_joint_ranges_deg(robot, parse_joint_ranges_arg(args.joint_ranges))
    robot.goto_given_conf(HOME_CONF)
    planning_preview_mesh = render_planning_preview(base, robot)
    sim_ui_batch = bool(getattr(args, "sim_ui_batch", False)) and not str(
        getattr(args, "export_traj", "") or ""
    ).strip()
    batch_use_rrt = False
    if sim_ui_batch:
        print(
            "[ppp] sim-ui batch: 连续抓放不回 HOME，直线失败再开 RRT，粗步长加速（不影响真机导出）",
            flush=True,
        )
    elif not args.no_rrt:
        print("[ppp] batch stable mode: 连续多件演示禁用 RRT，优先快速直线规划/失败跳过", flush=True)

    segments = []
    current_conf = np.asarray(HOME_CONF, dtype=float).copy()
    for index, task in enumerate(tasks, start=1):
        try:
            target_ann = _target_ann_for_task_in_sample(ann, task, sid)
            target = next((info for info in part_infos if info.get("id") == target_ann.get("id")), None)
            if target is None:
                raise RuntimeError("目标零件加载失败")

            part_state = _norm_state(
                target.get("state")
                or task.get("state")
                or task.get("pose_state")
                or target_ann.get("state")
            )
            target["state"] = part_state
            adjust_info = parse_adjust_pose(task, part_state=part_state)
            task_do_place = has_place_target(task)
            task_box_xy = resolve_box_xy(task) if task_do_place else None
            adjust_to_normal = False
            if task_do_place and adjust_info.get("adjust_to_normal"):
                print(
                    f"[adjust] {index}/{len(tasks)} 保持松爪前姿态入框，不再自动翻正",
                    flush=True,
                )
            place_yaw = float(
                adjust_info.get("place_yaw")
                if adjust_info.get("place_yaw") is not None
                else target.get("yaw", 0.0)
            )
            if abs(place_yaw) < 1e-12 and target.get("yaw") is not None:
                place_yaw = float(target["yaw"])

            place_model = None
            if adjust_to_normal:
                stl = target.get("stl_path") or target_ann.get("stl_path") or ""
                place_model, up_h, _ = build_upright_part_model(
                    stl,
                    place_yaw,
                    rgb=target.get("rgb"),
                    name=f"{target.get('class')}_upright_{index}",
                )
                print(
                    f"[adjust] {index}/{len(tasks)} {part_state} → normal "
                    f"(source={adjust_info.get('source')}, place_yaw={place_yaw:.3f}, "
                    f"upright_h={up_h:.3f}m)",
                    flush=True,
                )

            configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)

            (
                part_grasps,
                jaw_open,
                jaw_close,
                grasp_z,
                grasp_mode,
                neighbor_models,
            ) = prepare_grasps_topdown_then_side(
                robot,
                target,
                part_infos,
                desk_obstacle,
                topdown_pickle=args.grasp_pickle,
                antipodal_pickle=args.side_grasp_pickle,
            )
            configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)
            jaw_max_for_segment = float(robot.end_effector.jaw_range[1])
            jaw_open = float(min(float(jaw_open), float(robot.end_effector.jaw_range[1])))
            jaw_close = float(
                np.clip(
                    float(jaw_close),
                    JAW_CLOSE_MIN,
                    float(robot.end_effector.jaw_range[1]) - 1e-4,
                )
            )
            apply_jaw_close_to_grasps(part_grasps, jaw_close)
            robot.goto_given_conf(HOME_CONF, ee_values=float(jaw_open))

            obstacle_rounds = _ppp_obstacle_rounds(grasp_mode, neighbor_models)
            if sim_ui_batch and len(obstacle_rounds) > 1:
                obstacle_rounds = obstacle_rounds[:1]
            hover_pos = np.array(
                [float(target["pos"][0]), float(target["pos"][1]), float(HOVER_Z)],
                dtype=float,
            )
            place_ref = (
                place_model
                if (task_do_place and adjust_to_normal and place_model is not None)
                else target["model"]
            )
            place_meta = {}
            place_pose = None

            def _plan_with(obs, use_rrt):
                sink = PLIER_SINK_M if _part_is_plier(target) else 0.0
                if task_do_place:
                    return plan_pick_place(
                        robot,
                        obj_cmodel=target["model"],
                        grasp_collection=part_grasps,
                        jaw_open=jaw_open,
                        jaw_close=jaw_close,
                        place_pose=place_pose,
                        obstacles=obs,
                        desk=desk_obstacle,
                        use_rrt=use_rrt,
                        grasp_mode=grasp_mode,
                        start_conf=current_conf,
                        place_tcp_z=0.13,
                        place_approach_dist=0.055,
                        sink_m=sink,
                        fast=sim_ui_batch,
                    )
                if grasp_mode == "side":
                    return plan_side_pick_lift(
                        robot,
                        obj_cmodel=target["model"],
                        grasp_collection=part_grasps,
                        jaw_open=jaw_open,
                        jaw_close=jaw_close,
                        desk=desk_obstacle,
                        sink_m=sink,
                    )
                return plan_pick_lift(
                    robot,
                    obj_cmodel=target["model"],
                    grasp_collection=part_grasps,
                    jaw_open=jaw_open,
                    jaw_close=jaw_close,
                    obstacles=obs,
                    desk=desk_obstacle,
                    use_rrt=use_rrt,
                    grasp_mode=grasp_mode,
                    sink_m=sink,
                )

            mot = None
            place_attempts = 6 if (task_do_place and box_state is not None) else 1
            rrt_tries = (False, True) if (sim_ui_batch and not args.no_rrt) else (batch_use_rrt,)
            for place_try in range(place_attempts):
                if task_do_place:
                    if box_state is not None:
                        try:
                            place_pose_pos, place_pose_rot, place_meta = choose_box_place_pose(
                                box_state,
                                place_ref,
                                yaw=place_yaw,
                                candidate_offset=place_try,
                            )
                        except Exception as e:
                            print(
                                f"[box] target {index}/{len(tasks)} 没有更多可用放置点: {e}",
                                flush=True,
                            )
                            break
                        place_pose = (place_pose_pos, place_pose_rot)
                        print(
                            f"[box] target {index}/{len(tasks)} slot={place_meta['mode']} "
                            f"try={place_try + 1}/{place_attempts} "
                            f"object={target['class']} pos={np.round(place_pose_pos, 4).tolist()} "
                            f"tilt={float(place_meta.get('tilt', 0.0)):.3f} "
                            f"support={place_meta.get('support')}",
                            flush=True,
                        )
                    else:
                        place_meta = {"mode": "single", "radius": 0.02, "height": 0.02}
                        place_pose = place_goal_pose(task_box_xy, place_ref)
                else:
                    place_meta = {}
                    place_pose = None

                for use_rrt in rrt_tries:
                    if use_rrt:
                        print(
                            f"[ppp] batch {index}/{len(tasks)} 直线放置失败，"
                            f"仿真界面连放开 RRT 再试…",
                            flush=True,
                        )
                    for oi, obs in enumerate(obstacle_rounds):
                        if oi > 0:
                            print(
                                f"[ppp] batch {index}/{len(tasks)} 障碍轮次#{oi} "
                                f"邻件×{len(obs)} 再试…",
                                flush=True,
                            )
                        mot = _plan_with(obs, use_rrt)
                        if mot is not None:
                            break
                    if mot is not None:
                        break
                if mot is not None:
                    break
                if task_do_place and box_state is not None and place_try + 1 < place_attempts:
                    print(
                        f"[box] target {index}/{len(tasks)} 当前放置点规划失败，换下一个料框空位…",
                        flush=True,
                    )
            if mot is None:
                raise RuntimeError("PPP 规划失败：请检查零件尺寸/可达性或换 grasp pickle")

            carry_start, carry_end, release_idx = find_carry_range(
                mot, jaw_close, jaw_open=jaw_open if task_do_place else None
            )
            current_conf = np.asarray(mot.jv_list[-1], dtype=float).copy()
            print(
                f"[ppp] batch {index}/{len(tasks)} 成功 object={target['class']} "
                f"frames={len(mot)} grasp={grasp_mode}",
                flush=True,
            )
            place_roll = float(place_meta.get("tilt", 0.0)) if task_do_place else 0.0
            place_pitch = 0.0
            if task_do_place and box_state is not None and place_pose is not None:
                box_state.register(target["class"], place_pose[0], place_pose[1], place_meta)
                print(
                    f"[box] registered obstacle object={target['class']} "
                    f"mode={place_meta.get('mode')} count={len(box_state.items)}",
                    flush=True,
                )
            segments.append(
                {
                    "index": index,
                    "total": len(tasks),
                    "task": task,
                    "target": target,
                    "mot": mot,
                    "jaw_max": jaw_max_for_segment,
                    "jaw_open": jaw_open,
                    "jaw_close": jaw_close,
                    "grasp_z": grasp_z,
                    "grasp_mode": grasp_mode,
                    "do_place": task_do_place,
                    "place_pose": place_pose,
                    "place_model": place_model,
                    "adjust": adjust_to_normal,
                    "place_yaw": place_yaw,
                    "place_roll": place_roll,
                    "place_pitch": place_pitch,
                    "place_meta": place_meta,
                    "hover_pos": hover_pos,
                    "carry_start": carry_start,
                    "carry_end": carry_end,
                    "release_idx": release_idx,
                    "grasp_ac_pos": getattr(mot, "grasp_ac_pos", None),
                    "grasp_ac_rotmat": getattr(mot, "grasp_ac_rotmat", None),
                }
            )
        except Exception as e:
            print(
                f"[SIM] 跳过目标 {index}/{len(tasks)} object={task.get('object')}：{e}",
                flush=True,
            )
            _reset_robot_hand(robot, float(robot.end_effector.jaw_range[1]))

    if not segments:
        raise RuntimeError("批量任务没有任何目标规划成功，无法启动仿真")

    release_model_collection(planning_preview_mesh)
    print("=" * 60)
    print(f"[sim] sample={sid} single-world batch 成功目标={len(segments)}/{len(tasks)}", flush=True)
    for seg in segments:
        place_pose = seg["place_pose"]
        place_txt = (
            np.round(place_pose[0][:2], 4).tolist()
            if seg["do_place"] and place_pose is not None
            else None
        )
        print(
            f"[sim]   {seg['index']}/{seg['total']} object={seg['target']['class']} "
            f"mode={'pick-place' if seg['do_place'] else 'pick'} place_xy={place_txt}",
            flush=True,
        )
    if args.auto_play:
        print("[sim] 操作: 自动播放连续抓取；仍可按 [Space] 手动加速")
    else:
        print("[sim] 操作: 按 [Space] 推进连续抓取帧")
    print("=" * 60)

    if args.export_traj:
        export_planned_trajectory(
            args.export_traj,
            sid,
            args.task,
            [
                {
                    "index": seg["index"],
                    "total": seg["total"],
                    "object": seg["target"]["class"],
                    "grasp_mode": seg["grasp_mode"],
                    "do_place": seg["do_place"],
                    "mot": seg["mot"],
                    "jaw_open": seg["jaw_open"],
                    "jaw_close": seg["jaw_close"],
                    "carry_start": seg["carry_start"],
                    "carry_end": seg["carry_end"],
                    "release_idx": seg["release_idx"],
                }
                for seg in segments
            ],
        )
        if not args.auto_play:
            base.destroy()
            return

    first_seg = segments[0]
    configure_gripper_for_demo(
        robot,
        jaw_max=max(float(first_seg["jaw_max"]), float(first_seg["jaw_open"]) + 0.005),
    )
    _reset_robot_hand(robot, first_seg["jaw_open"])
    static_mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
    static_mesh.attach_to(base)

    if args.no_anime:
        for seg in segments:
            if seg["do_place"] and seg["place_pose"] is not None:
                if seg["adjust"] and seg["place_model"] is not None:
                    try:
                        seg["target"]["model"].detach()
                    except Exception:
                        pass
                    pm = seg["place_model"]
                    pm.pos = seg["place_pose"][0]
                    pm.rotmat = seg["place_pose"][1]
                    pm.attach_to(base)
                    seg["target"]["model"] = pm
                    seg["target"]["state"] = "normal"
                else:
                    seg["target"]["model"].pos = seg["place_pose"][0]
                    seg["target"]["model"].rotmat = seg["place_pose"][1]
        print("[sim] --no-anime：批量规划成功，退出", flush=True)
        base.destroy()
        return

    static_mesh.detach()
    release_model_collection(static_mesh)

    PRE, CARRY, DONE = 0, 1, 2

    class BatchAnimeData(object):
        __slots__ = (
            "seg_i",
            "counter",
            "phase",
            "last_mesh",
            "last_idx",
            "finished",
            "preview_saved",
            "initial_hold_ticks",
        )

        def __init__(self):
            self.seg_i = 0
            self.counter = 0
            self.phase = PRE
            self.last_mesh = None
            self.last_idx = -1
            self.finished = False
            self.preview_saved = False
            self.initial_hold_ticks = int(AUTO_PLAY_INITIAL_HOLD_SEC / ANIMATION_INTERVAL)

    ad = BatchAnimeData()

    def release_last_mesh():
        release_model_collection(ad.last_mesh)
        ad.last_mesh = None
        ad.last_idx = -1

    def save_preview_once():
        if ad.preview_saved or not args.preview_out:
            return
        ad.preview_saved = True
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.preview_out)), exist_ok=True)
            for _ in range(3):
                base.graphicsEngine.renderFrame()
            ok = base.win.saveScreenshot(args.preview_out)
            print(f"[preview] saved={args.preview_out} ok={ok}", flush=True)
        except Exception as e:
            print(f"[preview] 保存失败：{e}", flush=True)

    def release_held_to_pose(seg, pos_xyz):
        target = seg["target"]
        held_model = target["model"]
        try:
            release_rot = np.asarray(held_model.rotmat, dtype=float).copy()
        except Exception:
            release_rot = None
        ac_p = seg.get("grasp_ac_pos")
        ac_r = seg.get("grasp_ac_rotmat")
        if seg["do_place"] and ac_p is not None and ac_r is not None:
            try:
                tcp_rot = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
                release_rot = tcp_rot @ np.asarray(ac_r, dtype=float).T
            except Exception:
                pass
        if len(robot.end_effector.oiee_list) > 0:
            robot.end_effector.release_all()
        if seg["do_place"]:
            # 位置必须落在料框内部规划空位；姿态继承松爪前一刻持物姿态。
            pos_xyz = np.asarray(pos_xyz, dtype=float).copy()
        pos_v = rm.vec(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
        final_rot = (
            release_rot
            if seg["do_place"] and release_rot is not None
            else np.asarray(target["model"].rotmat, dtype=float)
        )
        if seg["adjust"] and seg["place_model"] is not None:
            try:
                target["model"].detach()
            except Exception:
                pass
            pm = seg["place_model"]
            pm.pos = pos_v
            pm.rotmat = final_rot
            pm.attach_to(base)
            target["model"] = pm
            target["state"] = "normal"
            target["roll"] = float(seg.get("place_roll", 0.0))
            target["pitch"] = float(seg.get("place_pitch", 0.0))
            target["yaw"] = float(seg["place_yaw"])
            target["pos"] = np.asarray(pos_xyz, dtype=float)
            print(
                f"[adjust] release into box using held rotation "
                f"@ {np.round(pos_xyz, 4).tolist()}",
                flush=True,
            )
            print(
                f"[box] settled object={target['class']} mode={seg.get('place_meta', {}).get('mode')} "
                f"rot=held_release",
                flush=True,
            )
            return
        target["model"].pos = pos_v
        target["model"].rotmat = final_rot
        target["model"].attach_to(base)
        target["pos"] = np.asarray(pos_xyz, dtype=float)
        if seg["do_place"]:
            print(
                f"[box] settled object={target['class']} mode={seg.get('place_meta', {}).get('mode')} "
                f"rot=held_release",
                flush=True,
            )

    def start_segment(seg):
        configure_gripper_for_demo(
            robot,
            jaw_max=max(float(seg["jaw_max"]), float(seg["jaw_open"]) + 0.005),
        )
        _reset_robot_hand(robot, seg["jaw_open"])
        mot = seg["mot"]
        jv0 = mot.jv_list[0]
        ev0 = mot.ev_list[0] if len(mot.ev_list) else None
        if ev0 is not None:
            robot.goto_given_conf(jnt_values=jv0, ee_values=ev0)
        else:
            robot.goto_given_conf(jnt_values=jv0)
        release_last_mesh()
        ad.last_mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
        ad.last_mesh.attach_to(base)
        ad.last_idx = 0
        ad.counter = 0
        ad.phase = PRE
        print(
            f"[sim] 开始连续目标 {seg['index']}/{seg['total']}: {seg['target']['class']}",
            flush=True,
        )

    def update(task_obj):
        if ad.finished:
            return task_obj.done
        save_preview_once()
        if args.auto_play and ad.initial_hold_ticks > 0:
            ad.initial_hold_ticks -= 1
            return task_obj.again
        seg = segments[ad.seg_i]
        mot = seg["mot"]
        n_frames = len(mot)

        if args.auto_play:
            ad.counter += AUTO_PLAY_FRAME_STEP
        elif base.inputmgr.keymap["space"]:
            ad.counter += 1
        if ad.counter >= n_frames:
            release_last_mesh()
            if ad.phase == CARRY:
                drop = seg["place_pose"][0] if seg["do_place"] and seg["place_pose"] is not None else np.array(
                    [seg["hover_pos"][0], seg["hover_pos"][1], 0.01], dtype=float
                )
                release_held_to_pose(seg, drop)
                ad.phase = DONE
            print(
                f"[sim] 连续目标 {seg['index']}/{seg['total']} 完成: {seg['target']['class']}",
                flush=True,
            )
            ad.seg_i += 1
            if ad.seg_i >= len(segments):
                ad.finished = True
                print("[sim] 连续抓取全部完成", flush=True)
                return task_obj.done
            start_segment(segments[ad.seg_i])
            return task_obj.again

        if ad.counter == ad.last_idx:
            return task_obj.again

        jv = mot.jv_list[ad.counter]
        ev = mot.ev_list[ad.counter] if ad.counter < len(mot.ev_list) else None
        holding_now = len(robot.end_effector.oiee_list) > 0
        if holding_now or ad.phase == CARRY:
            robot.goto_given_conf(jnt_values=jv)
        elif ev is not None:
            robot.goto_given_conf(jnt_values=jv, ee_values=ev)
        else:
            robot.goto_given_conf(jnt_values=jv)

        target = seg["target"]
        held_model = target["model"]
        if seg["carry_start"] is not None and ad.phase == PRE and ad.counter >= seg["carry_start"]:
            held_model.detach()
            if len(robot.end_effector.oiee_list) == 0:
                robot.hold(held_model, jaw_width=seg["jaw_close"])
            ad.phase = CARRY
            print("[sim] hold:", target["class"], flush=True)

        if (
            seg["do_place"]
            and seg["place_pose"] is not None
            and seg["release_idx"] is not None
            and ad.phase == CARRY
            and ad.counter >= seg["release_idx"]
        ):
            release_held_to_pose(seg, seg["place_pose"][0])
            ad.phase = DONE
            print(
                f"[sim] place into box @ {np.round(seg['place_pose'][0], 4).tolist()}"
                + (" (adjusted→normal)" if seg["adjust"] else ""),
                flush=True,
            )
            if ev is not None:
                robot.goto_given_conf(jnt_values=jv, ee_values=ev)

        release_last_mesh()
        ad.last_mesh = robot.gen_meshmodel(
            toggle_tcp_frame=False, toggle_jnt_frames=False
        )
        ad.last_mesh.attach_to(base)
        ad.last_idx = ad.counter
        return task_obj.again

    start_segment(first_seg)
    if args.auto_play and AUTO_PLAY_INITIAL_HOLD_SEC > 0:
        robot.goto_given_conf(HOME_CONF, ee_values=safe_home_jaw_width(robot))
        release_last_mesh()
        ad.last_mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
        ad.last_mesh.attach_to(base)
        ad.last_idx = 0
    if args.auto_play:
        wait_for_ui_animation_start(args)
    base.taskMgr.doMethodLater(
        ANIMATION_INTERVAL, update, "agent_pick_place_side_batch_anime", appendTask=True
    )
    base.run()


def save_home_scene_preview(args) -> None:
    if not args.sample_id:
        raise RuntimeError("--preview-only 需要指定 --sample-id")
    sid = str(args.sample_id).zfill(4) if str(args.sample_id).isdigit() else str(args.sample_id)
    ann_path = os.path.join(args.dataset_root, sid, "annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    out_path = args.preview_out
    if not out_path:
        out_path = os.path.join(args.dataset_root, sid, "sim_home_preview.jpg")
    print(f"[preview] home sample={sid} out={out_path}", flush=True)

    base = wd.World(
        cam_pos=[1.65, -1.25, 1.05],
        lookat_pos=[0.03, -0.03, 0.08],
        w=1920,
        h=1080,
    )
    sys.modules["__main__"].base = base
    mgm.gen_frame(ax_length=0.12).attach_to(base)

    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style=args.stripe, seed=0)

    for info in load_parts_from_annotations(ann, highlight_id=None):
        info["model"].attach_to(base)

    storage = (ann.get("scene_layout") or {}).get("storage_box") or {}
    xyz = storage.get("pos_m") or [float(DEFAULT_BOX_XY[0]), float(DEFAULT_BOX_XY[1]), 0.0]
    for bp in make_storage_box(np.asarray(xyz[:2], dtype=float)):
        bp.attach_to(base)

    robot = PantheraHTSglArm(enable_cc=False)
    configure_gripper_for_demo(robot, jaw_max=0.055)
    robot.goto_given_conf(HOME_CONF)
    try:
        robot.end_effector.change_jaw_width(0.055)
    except Exception:
        pass
    robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    for _ in range(6):
        base.graphicsEngine.renderFrame()
    ok = base.win.saveScreenshot(out_path)
    print(f"[preview] saved={out_path} ok={ok}", flush=True)
    try:
        base.destroy()
    except Exception:
        pass


def show_home_scene_window(args) -> None:
    if not args.sample_id:
        raise RuntimeError("--preview-window-only 需要指定 --sample-id")
    sid = str(args.sample_id).zfill(4) if str(args.sample_id).isdigit() else str(args.sample_id)
    ann_path = os.path.join(args.dataset_root, sid, "annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    print(f"[preview] live HOME scene sample={sid}", flush=True)
    base = wd.World(
        cam_pos=[1.65, -1.25, 1.05],
        lookat_pos=[0.03, -0.03, 0.08],
        w=1920,
        h=1080,
    )
    sys.modules["__main__"].base = base
    mgm.gen_frame(ax_length=0.12).attach_to(base)

    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style=args.stripe, seed=0)

    part_infos = load_parts_from_annotations(ann, highlight_id=None)
    for info in part_infos:
        info["model"].attach_to(base)

    storage = (ann.get("scene_layout") or {}).get("storage_box") or {}
    xyz = storage.get("pos_m") or [float(DEFAULT_BOX_XY[0]), float(DEFAULT_BOX_XY[1]), 0.0]
    for bp in make_storage_box(np.asarray(xyz[:2], dtype=float)):
        bp.attach_to(base)

    robot = PantheraHTSglArm(enable_cc=True)
    relax_panthera_joint_ranges_for_demo(robot)
    robot.goto_given_conf(HOME_CONF)
    try:
        robot.end_effector.change_jaw_width(0.055)
    except Exception:
        pass
    robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)

    print("[preview] live HOME scene ready", flush=True)
    ready_file = getattr(args, "ready_file", "") or ""
    if ready_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(ready_file)), exist_ok=True)
            with open(ready_file, "w", encoding="utf-8") as f:
                f.write("ready\n")
        except Exception as e:
            print(f"[preview] ready-file 写入失败：{e}", flush=True)
    base.run()


def run_worker(args) -> None:
    request_dir = os.path.abspath(args.worker_dir)
    ready_file = os.path.abspath(args.worker_ready_file) if args.worker_ready_file else ""
    os.makedirs(request_dir, exist_ok=True)
    for name in os.listdir(request_dir):
        if name.endswith(".json") or name.endswith(".tmp"):
            try:
                os.unlink(os.path.join(request_dir, name))
            except Exception:
                pass
    if ready_file:
        os.makedirs(os.path.dirname(ready_file), exist_ok=True)
        with open(ready_file, "w", encoding="utf-8") as f:
            f.write("ready\n")
    print(f"[worker] ready request_dir={request_dir}", flush=True)

    while True:
        requests = sorted(
            name
            for name in os.listdir(request_dir)
            if name.endswith(".json")
            and not name.endswith(".done.json")
            and not name.endswith(".error.json")
        )
        if not requests:
            time.sleep(0.1)
            continue

        req_path = os.path.join(request_dir, requests[0])
        try:
            with open(req_path, "r", encoding="utf-8") as f:
                req = json.load(f)
            task_path = str(req["task"])
            argv = [sys.argv[0], "--task", task_path, "--auto-play"]
            sample_id = req.get("sample_id")
            if sample_id:
                argv.extend(["--sample-id", str(sample_id)])
            dataset_root = req.get("dataset_root")
            if dataset_root:
                argv.extend(["--dataset-root", str(dataset_root)])
            wait_start_file = req.get("wait_start_file")
            if wait_start_file:
                argv.extend(["--wait-start-file", str(wait_start_file)])
            if req.get("no_rrt"):
                argv.append("--no-rrt")
            if req.get("sim_ui_batch"):
                argv.append("--sim-ui-batch")

            print(f"[worker] task-start request={os.path.basename(req_path)} task={task_path}", flush=True)
            old_argv = sys.argv[:]
            try:
                sys.argv = argv
                main()
                code = 0
            finally:
                sys.argv = old_argv
            os.replace(req_path, req_path + ".done.json")
            print(f"[worker] task-finished request={os.path.basename(req_path)} code={code}", flush=True)
        except Exception as e:
            try:
                with open(req_path + ".error.json", "w", encoding="utf-8") as f:
                    json.dump({"error": str(e)}, f, ensure_ascii=False, indent=2)
                os.unlink(req_path)
            except Exception:
                pass
            print(
                f"[worker] task-finished request={os.path.basename(req_path)} code=1 error={e}",
                flush=True,
            )


def main():
    p = argparse.ArgumentParser(
        description="智能体 API → Panthera-HT pick/place（顶抓+侧向回退）"
    )
    p.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="智能体 API 任务 JSON（默认 tiaozhanbei/tasks/test06.json）",
    )
    p.add_argument("--dataset-root", default=DEFAULT_DATASET)
    p.add_argument(
        "--grasp-pickle",
        default=DEFAULT_GRASP_TOPDOWN,
        help="顶抓 pickle（失败后仍会尝试对踵侧抓）",
    )
    p.add_argument(
        "--side-grasp-pickle",
        default=DEFAULT_GRASP_ANTIPODAL,
        help="对踵/侧向 pickle",
    )
    p.add_argument("--sample-id", default=None, help="强制样本号；默认按坐标匹配")
    p.add_argument("--no-anime", action="store_true")
    p.add_argument(
        "--export-traj",
        default="",
        help="规划成功后把关节轨迹导出到该 JSON 并退出（供 tiaozhanbei/real 真机执行模块使用）",
    )
    p.add_argument(
        "--joint-ranges",
        default="",
        help="收紧 6 个关节范围（度），JSON 形如 [[lo,hi],...] 或指向该内容的文件；"
        "导出真机轨迹时用它把搜索限制在真机软限位内",
    )
    p.add_argument("--auto-play", action="store_true", help="自动推进动画帧；默认仍可用空格手动推进")
    p.add_argument("--wait-start-file", default="", help="UI 贴合 Panda 窗口后创建该文件，仿真再开始播放")
    p.add_argument("--preview-out", default="", help="保存动画开始前的仿真初始截图")
    p.add_argument("--preview-only", action="store_true", help="仅保存 HOME 初始场景截图后退出")
    p.add_argument("--preview-window-only", action="store_true", help="仅显示 HOME 初始场景窗口，不进入规划/动画")
    p.add_argument("--ready-file", default="", help="HOME 预览窗口准备好后写入该文件")
    p.add_argument("--worker-dir", default="", help="常驻 worker 请求目录；为空则按普通单次仿真运行")
    p.add_argument("--worker-ready-file", default="", help="worker 初始化完成后写入该文件")
    p.add_argument("--no-rrt", action="store_true")
    p.add_argument(
        "--sim-ui-batch",
        action="store_true",
        help="仅仿真界面连放：每件放完回 HOME，放置直线失败再开 RRT。"
        "真机 --export-traj 不要加此开关。",
    )
    p.add_argument("--stripe", default="vert_gray")
    args = p.parse_args()

    if args.preview_only:
        save_home_scene_preview(args)
        return
    if args.preview_window_only:
        show_home_scene_window(args)
        return
    if args.worker_dir:
        run_worker(args)
        return

    raw_task = load_task(args.task)
    multi_tasks = expand_multi_pick_place_tasks(raw_task)
    if len(multi_tasks) > 1:
        run_multi_pick_place_in_one_world(args, raw_task, multi_tasks)
        return

    task = coerce_multi_object_task_for_single_sim(raw_task)
    do_place = has_place_target(task)
    box_xy = resolve_box_xy(task) if do_place else None
    print(
        f"[task] id={task.get('task_id')} action={task.get('action')} "
        f"object={task.get('object')} mode={'pick-place' if do_place else 'pick'}",
        flush=True,
    )
    if do_place:
        print(
            f"[task] destination={task.get('destination')} "
            f"box_xy={np.round(box_xy, 4).tolist()}",
            flush=True,
        )
        seq = task.get("task_sequence_desc") or []
        for line in seq:
            print(f"[task]   {line}", flush=True)


    if args.sample_id:
        sid = args.sample_id
        ann_path = os.path.join(args.dataset_root, sid, "annotations.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        target_ann = None
        coord = np.asarray(task.get("coordinate") or [0, 0, 0], dtype=float)
        for obj in ann.get("objects") or []:
            if obj.get("class") != task.get("object"):
                continue
            pose = obj.get("pose_6d") or {}
            xy = np.array([float(pose.get("x", 0.0)), float(pose.get("y", 0.0))])
            if np.linalg.norm(xy - coord[:2]) < 0.02:
                target_ann = obj
                break
        if target_ann is None:
            raise RuntimeError(f"样本 {sid} 中找不到任务零件")
    else:
        sid, target_ann, ann = find_sample_for_task(args.dataset_root, task)

    print(
        f"[dataset] sample={sid} target_id={target_ann.get('id')} "
        f"class={target_ann.get('class')}",
        flush=True,
    )

    print(
        f"[grasp] topdown={args.grasp_pickle}  side={args.side_grasp_pickle}",
        flush=True,
    )

    configure_hidden_start_window(args)
    base = wd.World(
        cam_pos=[1.65, -1.25, 1.05],
        lookat_pos=[0.03, -0.03, 0.08],
        w=1920,
        h=1080,
    )
    sys.modules["__main__"].base = base  # RRT 调试/mesh 依赖
    mgm.gen_frame(ax_length=0.12).attach_to(base)

    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style=args.stripe, seed=0)
    desk_obstacle = make_desk_obstacle()

    part_infos = load_parts_from_annotations(ann, highlight_id=None)
    target = None
    for info in part_infos:
        if info.get("id") == target_ann.get("id"):
            target = info
            break
    if target is None:
        raise RuntimeError("目标零件加载失败")

    # 任务 pose_state / annotations.state → 是否放置前调成正放
    part_state = _norm_state(
        target.get("state")
        or task.get("state")
        or task.get("pose_state")
        or target_ann.get("state")
    )
    target["state"] = part_state
    adjust_info = parse_adjust_pose(task, part_state=part_state)
    adjust_to_normal = bool(do_place and adjust_info.get("adjust_to_normal"))
    place_yaw = float(
        adjust_info.get("place_yaw")
        if adjust_info.get("place_yaw") is not None
        else target.get("yaw", 0.0)
    )
    if abs(place_yaw) < 1e-12 and target.get("yaw") is not None:
        place_yaw = float(target["yaw"])

    for info in part_infos:
        info["model"].attach_to(base)

    box_parts = []
    if do_place:
        box_parts = make_storage_box(box_xy)
        for bp in box_parts:
            bp.attach_to(base)
        print(
            f"[scene] 收纳盒 size={STORAGE_BOX_SIZE.tolist()} "
            f"@ xy={np.round(box_xy, 4).tolist()}",
            flush=True,
        )

    place_model = None
    if adjust_to_normal:
        stl = target.get("stl_path") or target_ann.get("stl_path") or ""
        place_model, up_h, _ = build_upright_part_model(
            stl,
            place_yaw,
            rgb=target.get("rgb"),
            name=f"{target.get('class')}_upright",
        )
        print(
            f"[adjust] {part_state} → normal "
            f"(source={adjust_info.get('source')}, place_yaw={place_yaw:.3f}, "
            f"upright_h={up_h:.3f}m)",
            flush=True,
        )

    robot = PantheraHTSglArm(enable_cc=True)
    relax_panthera_joint_ranges_for_demo(robot)
    if getattr(args, "joint_ranges", ""):
        apply_joint_ranges_deg(robot, parse_joint_ranges_arg(args.joint_ranges))

    configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)
    robot.goto_given_conf(HOME_CONF)
    planning_preview_mesh = render_planning_preview(base, robot)

    (
        part_grasps,
        jaw_open,
        jaw_close,
        grasp_z,
        grasp_mode,
        neighbor_models,
    ) = prepare_grasps_topdown_then_side(
        robot,
        target,
        part_infos,
        desk_obstacle,
        topdown_pickle=args.grasp_pickle,
        antipodal_pickle=args.side_grasp_pickle,
    )
    configure_gripper_for_demo(robot, jaw_max=JAW_OPEN_MAX)
    jaw_open = float(min(float(jaw_open), float(robot.end_effector.jaw_range[1])))
    jaw_close = float(
        np.clip(
            float(jaw_close),
            JAW_CLOSE_MIN,
            float(robot.end_effector.jaw_range[1]) - 1e-4,
        )
    )
    apply_jaw_close_to_grasps(part_grasps, jaw_close)
    robot.goto_given_conf(HOME_CONF, ee_values=float(jaw_open))

    # 侧抓 / 顶抓：邻件已预筛；接近段先不把邻件当路径障碍
    obstacle_rounds = _ppp_obstacle_rounds(grasp_mode, neighbor_models)
    print(
        f"[cd] grasp_mode={grasp_mode} PPP 障碍轮次="
        f"{[len(o) for o in obstacle_rounds]}（桌面仍用指尖 z_min；侧抓接近沿 hand-z）",
        flush=True,
    )

    hover_pos = np.array(
        [float(target["pos"][0]), float(target["pos"][1]), float(HOVER_Z)],
        dtype=float,
    )
    if do_place:
        # 调姿时用正放 mesh 算放置高度；抓取仍用原姿态零件
        place_ref = place_model if (adjust_to_normal and place_model is not None) else target["model"]
        place_pose = place_goal_pose(box_xy, place_ref)
    else:
        place_pose = None

    def _plan_with(obs):
        sink = PLIER_SINK_M if _part_is_plier(target) else 0.0
        if do_place:
            return plan_pick_place(
                robot,
                obj_cmodel=target["model"],
                grasp_collection=part_grasps,
                jaw_open=jaw_open,
                jaw_close=jaw_close,
                place_pose=place_pose,
                obstacles=obs,
                desk=desk_obstacle,
                use_rrt=not args.no_rrt,
                grasp_mode=grasp_mode,
                sink_m=sink,
            )
        if grasp_mode == "side":
            return plan_side_pick_lift(
                robot,
                obj_cmodel=target["model"],
                grasp_collection=part_grasps,
                jaw_open=jaw_open,
                jaw_close=jaw_close,
                desk=desk_obstacle,
                sink_m=sink,
            )
        return plan_pick_lift(
            robot,
            obj_cmodel=target["model"],
            grasp_collection=part_grasps,
            jaw_open=jaw_open,
            jaw_close=jaw_close,
            obstacles=obs,
            desk=desk_obstacle,
            use_rrt=not args.no_rrt,
            grasp_mode=grasp_mode,
            sink_m=sink,
        )

    mot = None
    for oi, obs in enumerate(obstacle_rounds):
        if oi > 0:
            print(f"[ppp] 障碍轮次#{oi} 邻件×{len(obs)} 再试…", flush=True)
        mot = _plan_with(obs)
        if mot is not None:
            break
    if mot is None:
        raise RuntimeError("PPP 规划失败：请检查零件尺寸/可达性或换 grasp pickle")

    release_model_collection(planning_preview_mesh)
    print(f"[ppp] 成功 frames={len(mot)}", flush=True)
    carry_start, carry_end, release_idx = find_carry_range(
        mot, jaw_close, jaw_open=jaw_open if do_place else None
    )

    print("=" * 60)
    print(
        f"[sim] sample={sid} object={target['class']} "
        f"mode={'pick-place' if do_place else 'pick'} grasp={grasp_mode}"
    )
    print(
        f"[sim] pick_xy={np.round(target['pos'][:2], 4).tolist()} "
        f"grasp_z={grasp_z:.3f} lift={PICK_DEPART_DIST:.3f}m"
    )
    if do_place:
        print(
            f"[sim] place_xy={np.round(place_pose[0][:2], 4).tolist()} "
            f"place_z={place_pose[0][2]:.3f}"
            + (" adjust→normal" if adjust_to_normal else "")
        )
    print(
        f"[sim] jaw_open={jaw_open:.3f} jaw_close={jaw_close:.3f} "
        f"carry=[{carry_start}..{carry_end}] release={release_idx}"
    )
    if args.auto_play:
        print("[sim] 操作: 自动播放；仍可按 [Space] 手动加速")
    else:
        print("[sim] 操作: 按 [Space] 推进下一帧")
    print("=" * 60)

    if args.export_traj:
        export_planned_trajectory(
            args.export_traj,
            sid,
            args.task,
            [
                {
                    "index": 1,
                    "total": 1,
                    "object": target["class"],
                    "grasp_mode": grasp_mode,
                    "do_place": do_place,
                    "mot": mot,
                    "jaw_open": jaw_open,
                    "jaw_close": jaw_close,
                    "carry_start": carry_start,
                    "carry_end": carry_end,
                    "release_idx": release_idx,
                }
            ],
        )
        if not args.auto_play:
            base.destroy()
            return

    _reset_robot_hand(robot, jaw_open)
    static_mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
    static_mesh.attach_to(base)

    if args.no_anime:
        if do_place and place_pose is not None:
            if adjust_to_normal and place_model is not None:
                try:
                    target["model"].detach()
                except Exception:
                    pass
                place_model.pos = place_pose[0]
                place_model.rotmat = place_pose[1]
                place_model.attach_to(base)
                target["model"] = place_model
                target["state"] = "normal"
                print(
                    f"[adjust] --no-anime 已按 normal 放入 "
                    f"@ {np.round(place_pose[0], 4).tolist()}",
                    flush=True,
                )
            else:
                target["model"].pos = place_pose[0]
                target["model"].rotmat = place_pose[1]
        print("[sim] --no-anime：规划成功，退出", flush=True)
        base.destroy()
        return

    static_mesh.detach()
    release_model_collection(static_mesh)

    held_ref = {
        "model": target["model"],
        "place_model": place_model,
        "adjust": adjust_to_normal,
        "grasp_ac_pos": getattr(mot, "grasp_ac_pos", None),
        "grasp_ac_rotmat": getattr(mot, "grasp_ac_rotmat", None),
    }
    n_frames = len(mot)
    PRE, CARRY, DONE = 0, 1, 2

    class AnimeData(object):
        __slots__ = (
            "counter",
            "phase",
            "last_mesh",
            "last_idx",
            "finished",
            "preview_saved",
            "initial_hold_ticks",
        )

        def __init__(self):
            self.counter = 0
            self.phase = PRE
            self.last_mesh = None
            self.last_idx = -1
            self.finished = False
            self.preview_saved = False
            self.initial_hold_ticks = int(AUTO_PLAY_INITIAL_HOLD_SEC / ANIMATION_INTERVAL)

    ad = AnimeData()

    def release_held_to_pose(pos_xyz):
        try:
            release_rot = np.asarray(held_ref["model"].rotmat, dtype=float).copy()
        except Exception:
            release_rot = None
        ac_p = held_ref.get("grasp_ac_pos")
        ac_r = held_ref.get("grasp_ac_rotmat")
        if do_place and ac_p is not None and ac_r is not None:
            try:
                tcp_rot = np.asarray(robot.gl_tcp_rotmat, dtype=float).copy()
                release_rot = tcp_rot @ np.asarray(ac_r, dtype=float).T
            except Exception:
                pass
        if len(robot.end_effector.oiee_list) > 0:
            robot.end_effector.release_all()
        pos_v = rm.vec(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
        final_rot = (
            release_rot
            if do_place and release_rot is not None
            else np.asarray(held_ref["model"].rotmat, dtype=float)
        )
        if held_ref["adjust"] and held_ref["place_model"] is not None:
            try:
                held_ref["model"].detach()
            except Exception:
                pass
            pm = held_ref["place_model"]
            pm.pos = pos_v
            pm.rotmat = final_rot
            pm.attach_to(base)
            held_ref["model"] = pm
            target["model"] = pm
            target["state"] = "normal"
            target["roll"] = 0.0
            target["pitch"] = 0.0
            target["yaw"] = float(place_yaw)
            target["pos"] = np.asarray(pos_xyz, dtype=float)
            print(
                f"[adjust] release into box using held rotation "
                f"@ {np.round(pos_xyz, 4).tolist()}",
                flush=True,
            )
            return
        held_ref["model"].pos = pos_v
        held_ref["model"].rotmat = final_rot
        held_ref["model"].attach_to(base)
        target["pos"] = np.asarray(pos_xyz, dtype=float)

    def release_last_mesh():
        release_model_collection(ad.last_mesh)
        ad.last_mesh = None
        ad.last_idx = -1

    def save_preview_once():
        if ad.preview_saved or not args.preview_out:
            return
        ad.preview_saved = True
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.preview_out)), exist_ok=True)
            for _ in range(3):
                base.graphicsEngine.renderFrame()
            ok = base.win.saveScreenshot(args.preview_out)
            print(f"[preview] saved={args.preview_out} ok={ok}", flush=True)
        except Exception as e:
            print(f"[preview] 保存失败：{e}", flush=True)

    def update(task_obj):
        if ad.finished:
            return task_obj.done
        save_preview_once()
        if args.auto_play and ad.initial_hold_ticks > 0:
            ad.initial_hold_ticks -= 1
            return task_obj.again
        if args.auto_play:
            ad.counter += AUTO_PLAY_FRAME_STEP
        elif base.inputmgr.keymap["space"]:
            ad.counter += 1
        if ad.counter >= n_frames:
            release_last_mesh()
            if ad.phase == CARRY:
                drop = place_pose[0] if do_place else np.array(
                    [hover_pos[0], hover_pos[1], 0.01], dtype=float
                )
                release_held_to_pose(drop)
                ad.phase = DONE
            ad.finished = True
            print(
                f"[sim] 动画结束（{'pick-place' if do_place else 'pick'} 完成）",
                flush=True,
            )
            return task_obj.done
        if ad.counter == ad.last_idx:
            return task_obj.again

        jv = mot.jv_list[ad.counter]
        ev = mot.ev_list[ad.counter] if ad.counter < len(mot.ev_list) else None
        holding_now = len(robot.end_effector.oiee_list) > 0
        # 持物时禁止改 ee_values，否则 WRS 抛 The hand is holding objects!
        if holding_now or ad.phase == CARRY:
            robot.goto_given_conf(jnt_values=jv)
        elif ev is not None:
            robot.goto_given_conf(jnt_values=jv, ee_values=ev)
        else:
            robot.goto_given_conf(jnt_values=jv)

        held_model = held_ref["model"]
        if carry_start is not None and ad.phase == PRE and ad.counter >= carry_start:
            held_model.detach()
            if len(robot.end_effector.oiee_list) == 0:
                robot.hold(held_model, jaw_width=jaw_close)
            ad.phase = CARRY
            print("[sim] hold:", target["class"], flush=True)

        if (
            do_place
            and release_idx is not None
            and ad.phase == CARRY
            and ad.counter >= release_idx
        ):
            release_held_to_pose(place_pose[0])
            ad.phase = DONE
            print(
                f"[sim] place into box @ {np.round(place_pose[0], 4).tolist()}"
                + (" (adjusted→normal)" if adjust_to_normal else ""),
                flush=True,
            )
            # 释放后允许后续帧开爪 / 离开
            if ev is not None:
                robot.goto_given_conf(jnt_values=jv, ee_values=ev)

        release_last_mesh()
        ad.last_mesh = robot.gen_meshmodel(
            toggle_tcp_frame=False, toggle_jnt_frames=False
        )
        ad.last_mesh.attach_to(base)
        ad.last_idx = ad.counter
        return task_obj.again

    jv0 = mot.jv_list[0]
    ev0 = mot.ev_list[0] if len(mot.ev_list) else None
    if ev0 is not None:
        robot.goto_given_conf(jnt_values=jv0, ee_values=ev0)
    else:
        robot.goto_given_conf(jnt_values=jv0)
    ad.last_mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
    ad.last_mesh.attach_to(base)
    ad.last_idx = 0

    if args.auto_play and AUTO_PLAY_INITIAL_HOLD_SEC > 0:
        robot.goto_given_conf(HOME_CONF, ee_values=safe_home_jaw_width(robot))
        release_last_mesh()
        ad.last_mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
        ad.last_mesh.attach_to(base)
        ad.last_idx = 0

    if args.auto_play:
        wait_for_ui_animation_start(args)
    base.taskMgr.doMethodLater(
        ANIMATION_INTERVAL, update, "agent_pick_place_side_anime", appendTask=True
    )
    base.run()


if __name__ == "__main__":
    main()
