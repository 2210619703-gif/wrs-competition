#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把标准抓放规划轨迹叠成半透明机械臂（类似 pick_multi_block 的 render_full_trajectory）。

默认对应 ``agent_demo_ui.py`` 里::

    FORCE_START_SAMPLE = "0001"
    FORCE_START_SCENE = ""

即 ``dataset_learn/0001`` 中的螺丝刀放入收纳盒。不走 ``ec_bin``。

仓库根目录::

    python tiaozhanbei/sim/render_full_trajectory.py
    python tiaozhanbei/sim/render_full_trajectory.py --stage pick --max-meshes 8

默认 ``full``：从闭爪抓起叠到放入，机械臂与工件同步半透明叠影，约 12～16 帧。

螺丝入格（0502、正放 1 颗）::

    python tiaozhanbei/sim/render_full_trajectory.py --scene slot
    python tiaozhanbei/sim/render_full_trajectory.py --scene slot --object-id 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_SIM_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs import wd, rm, mgm  # noqa: E402
from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import (  # noqa: E402
    PantheraHTSglArm,
)

from tiaozhanbei.sim.environment import attach_desk_stripe, gen_desk  # noqa: E402
from tiaozhanbei.sim import run_agent_pick_place_side_sim as side  # noqa: E402
from tiaozhanbei.sim.ec import run_ec_bin_sim as ec  # noqa: E402

STAGE_FULL = "full"
STAGE_APPROACH = "approach"
STAGE_PICK = "pick"
STAGE_DEPART = "depart"
STAGE_PLACE = "place"
STAGE_CHOICES = (STAGE_FULL, STAGE_APPROACH, STAGE_PICK, STAGE_DEPART, STAGE_PLACE)

SCENE_BOX = "box"
SCENE_SLOT = "slot"
SCENE_CHOICES = (SCENE_BOX, SCENE_SLOT)

DEFAULT_SAMPLE = "0001"
DEFAULT_OBJECT = "螺丝刀"
DEFAULT_SLOT_SAMPLE = "0502"


def _load_ann(dataset_root: str, sample_id: str) -> dict:
    path = os.path.join(dataset_root, sample_id, "annotations.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_target(part_infos: list, hint: str):
    hint = (hint or "").strip()
    if hint:
        for info in part_infos:
            name = str(info.get("class") or "")
            if hint in name or name in hint:
                return info
        by_id = [info for info in part_infos if str(info.get("id")) == hint]
        if by_id:
            return by_id[0]
    raise RuntimeError(
        f"场景里找不到「{hint}」。已有类别："
        + ", ".join(str(i.get("class")) for i in part_infos)
    )


def _stage_range(n: int, carry_start: int, carry_end: int, stage: str):
    start, end = 0, n - 1
    cs = int(carry_start) if carry_start is not None else start
    ce = int(carry_end) if carry_end is not None else end
    cs = max(start, min(cs, end))
    ce = max(cs, min(ce, end))
    pre_end = max(start, cs - 1)
    pre_mid = (start + pre_end) // 2
    hold_mid = (cs + ce) // 2
    pick_lo = min(pre_mid + 1, pre_end)
    pick_hi = pre_end
    if pick_lo > pick_hi:
        pick_lo, pick_hi = cs, min(cs + 1, end)
    if stage == STAGE_FULL:
        # 从闭爪抓取叠到放入料框；不含接近桌面的空爪 approach
        return cs, end
    if stage == STAGE_APPROACH:
        return start, pre_mid
    if stage == STAGE_PICK:
        # 必须包含闭爪抓取帧，否则叠影停在接近、螺丝刀还在桌上
        return pick_lo, min(max(pick_hi, cs), end)
    if stage == STAGE_DEPART:
        return cs, hold_mid
    if stage == STAGE_PLACE:
        return min(hold_mid + 1, end), end
    raise ValueError(stage)


def _sample_indices(lo: int, hi: int, max_meshes: int, keyframes=()) -> list[int]:
    if hi < lo:
        return []
    ids = list(range(lo, hi + 1))
    if max_meshes > 0 and len(ids) > max_meshes:
        pick = np.linspace(0, len(ids) - 1, max_meshes, dtype=int)
        ids = [ids[i] for i in pick]
    extra = [k for k in keyframes if k is not None and lo <= int(k) <= hi]
    ids = sorted(set(int(i) for i in list(ids) + extra))
    return ids


def _held_object_pose(robot, ac_pos, ac_rot):
    """由当前 TCP 与抓取相对位姿反推物体世界位姿。"""
    tcp_pos = np.asarray(robot.gl_tcp_pos, dtype=float)
    tcp_rot = np.asarray(robot.gl_tcp_rotmat, dtype=float)
    ac_pos = np.asarray(ac_pos, dtype=float)
    ac_rot = np.asarray(ac_rot, dtype=float)
    obj_rot = tcp_rot @ ac_rot.T
    obj_pos = tcp_pos - obj_rot @ ac_pos
    return obj_pos, obj_rot


def _overlay_object(src_model, pos, rot, alpha, base):
    ghost = src_model.copy()
    ghost.pos = rm.vec(float(pos[0]), float(pos[1]), float(pos[2]))
    ghost.rotmat = np.asarray(rot, dtype=float)
    _set_mesh_alpha(ghost, alpha)
    ghost.attach_to(base)
    return ghost


def _set_mesh_alpha(mesh, alpha: float) -> None:
    try:
        mesh.alpha = float(alpha)
        return
    except Exception:
        pass
    for cm in getattr(mesh, "cm_list", []) or []:
        try:
            cm.alpha = float(alpha)
        except Exception:
            try:
                cm.setColor(cm.getColor()[0], cm.getColor()[1], cm.getColor()[2], float(alpha))
            except Exception:
                pass


def plan_screwdriver_motion(base, sample_id: str, object_hint: str, dataset_root: str, use_rrt: bool):
    ann = _load_ann(dataset_root, sample_id)
    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style="vert_gray", seed=0)
    desk_obstacle = side.make_desk_obstacle()

    part_infos = side.load_parts_from_annotations(ann, highlight_id=None)
    for info in part_infos:
        info["model"].attach_to(base)
    target = _pick_target(part_infos, object_hint)
    print(f"[render] sample={sample_id} target={target.get('class')} id={target.get('id')}", flush=True)

    box_xy = np.asarray(side.STORAGE_BOX_POS[:2], dtype=float)
    box_state = side.BoxPlacementState(box_xy)
    for bp in side.make_storage_box(box_xy):
        bp.attach_to(base)

    robot = PantheraHTSglArm(enable_cc=True)
    side.relax_panthera_joint_ranges_for_demo(robot)
    robot.goto_given_conf(side.HOME_CONF)

    open_w, _ = side.jaw_widths_for_part(target)
    side.configure_gripper_for_demo(robot, jaw_max=max(open_w + 0.005, 0.02))
    (
        part_grasps,
        jaw_open,
        jaw_close,
        _grasp_z,
        grasp_mode,
        neighbor_models,
    ) = side.prepare_grasps_topdown_then_side(
        robot,
        target,
        part_infos,
        desk_obstacle,
        topdown_pickle=side.DEFAULT_GRASP_TOPDOWN,
        antipodal_pickle=side.DEFAULT_GRASP_ANTIPODAL,
    )
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
    side.apply_jaw_close_to_grasps(part_grasps, jaw_close)
    robot.goto_given_conf(side.HOME_CONF, ee_values=float(jaw_open))

    place_yaw = float(target.get("yaw", 0.0) or 0.0)
    place_pose = None
    mot = None
    for place_try in range(6):
        place_pose_pos, place_pose_rot, place_meta = side.choose_box_place_pose(
            box_state,
            target["model"],
            yaw=place_yaw,
            candidate_offset=place_try,
        )
        place_pose = (place_pose_pos, place_pose_rot)
        print(
            f"[render] place try={place_try + 1} mode={place_meta.get('mode')} "
            f"pos={np.round(place_pose_pos, 4).tolist()} grasp={grasp_mode}",
            flush=True,
        )
        obstacle_rounds = [[]] if grasp_mode == "side" else [list(neighbor_models)]
        for obs in obstacle_rounds:
            mot = side.plan_pick_place(
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
                start_conf=side.HOME_CONF,
                place_tcp_z=0.13,
                place_approach_dist=0.055,
            )
            if mot is not None:
                break
        if mot is not None:
            break
        print("[render] 当前放置点失败，换料框空位…", flush=True)

    if mot is None:
        raise RuntimeError("螺丝刀 pick-place 规划失败")

    carry_start, carry_end, _release = side.find_carry_range(
        mot, jaw_close, jaw_open=jaw_open
    )
    print(
        f"[render] frames={len(mot.jv_list)} carry=[{carry_start}..{carry_end}] "
        f"grasp={grasp_mode} ac={'ok' if getattr(mot, 'grasp_ac_pos', None) is not None else 'missing'}",
        flush=True,
    )
    return robot, mot, target, place_pose, carry_start, carry_end, jaw_open, jaw_close


def _pick_normal_screw(part_infos: list, object_id=None):
    if object_id is not None:
        oid = int(object_id)
        for info in part_infos:
            if int(info.get("id")) == oid:
                return info
        raise RuntimeError(f"场景里没有 object_id={oid}")
    normals = [
        p
        for p in part_infos
        if str(p.get("state") or "").strip().lower() == "normal"
    ]
    if not normals:
        raise RuntimeError("0502 里没有 state=normal 的螺丝")
    return normals[0]


def plan_normal_slot_motion(
    base,
    sample_id: str,
    dataset_root: str,
    object_id=None,
    slot_id=None,
    use_rrt: bool = False,
):
    """0502 正放螺丝 → 料盘一格。其余螺丝留在桌上，只叠这一颗的轨迹。"""
    ann = _load_ann(dataset_root, sample_id)
    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style="vert_gray", seed=0)
    desk_obstacle = side.make_desk_obstacle()

    box = ec.load_box_model()
    box.attach_to(base)

    part_infos = ec.load_parts_from_ec_ann(ann)
    for info in part_infos:
        info["model"].attach_to(base)

    target = _pick_normal_screw(part_infos, object_id=object_id)
    oid = int(target.get("id"))
    sid = int(slot_id) if slot_id is not None else oid - 1
    print(
        f"[render] sample={sample_id} screw id={oid} state={target.get('state')} "
        f"slot={sid}",
        flush=True,
    )

    robot = PantheraHTSglArm(enable_cc=True)
    side.relax_panthera_joint_ranges_for_demo(robot)
    robot.goto_given_conf(side.HOME_CONF)

    pkg = ec.plan_one_job(
        robot,
        target,
        part_infos,
        desk_obstacle,
        slot_id=sid,
        grasp_topdown=ec.DEFAULT_GRASP_TOPDOWN,
        grasp_side=ec.DEFAULT_GRASP_SIDE,
        use_rrt=bool(use_rrt),
        pipeline="normal",
        box_model=box,
        strategy_prefer="topdown",
        seat_kind="slot",
        lock_style=True,
        enforce_style=True,
        require_upright=True,
    )
    mot = pkg["mot"]
    place_pose = pkg["place_pose"]
    carry_start = pkg["carry_start"]
    carry_end = pkg["carry_end"]
    jaw_open = float(pkg["jaw_open"])
    jaw_close = float(pkg["jaw_close"])
    # 规划过程会挪 mesh；叠影前把原件放回桌上
    target["model"].pos = np.asarray(target["pos"], dtype=float)
    target["model"].rotmat = np.eye(3)
    print(
        f"[render] frames={len(mot.jv_list)} carry=[{carry_start}..{carry_end}] "
        f"strategy={pkg.get('strategy')} ac={'ok' if pkg.get('grasp_ac_pos') is not None else 'missing'}",
        flush=True,
    )
    if pkg.get("grasp_ac_pos") is not None:
        mot.grasp_ac_pos = pkg["grasp_ac_pos"]
        mot.grasp_ac_rotmat = pkg["grasp_ac_rotmat"]
    return robot, mot, target, place_pose, carry_start, carry_end, jaw_open, jaw_close


def _overlay_motion(
    base,
    robot,
    mot,
    target,
    place_pose,
    carry_start,
    carry_end,
    jaw_open,
    jaw_close,
    stage: str,
    max_meshes: int,
    alpha_min: float,
    alpha_max: float,
    object_label: str,
):
    obj_model = target["model"]
    table_pos = np.asarray(obj_model.pos, dtype=float).copy()
    table_rot = np.asarray(obj_model.rotmat, dtype=float).copy()
    ac_pos = getattr(mot, "grasp_ac_pos", None)
    ac_rot = getattr(mot, "grasp_ac_rotmat", None)
    cs = int(carry_start or 0)
    ce = int(carry_end if carry_end is not None else cs)

    n = len(mot.jv_list)
    lo, hi = _stage_range(n, carry_start, carry_end, stage)
    indices = _sample_indices(lo, hi, max_meshes, keyframes=(cs, ce, n - 1))
    if not indices:
        raise RuntimeError(f"stage={stage!r} 没有可渲染帧")
    print(
        f"[render] stage={stage} overlay={len(indices)} frames [{lo}..{hi}] "
        f"pick_key={cs} place_key={ce}",
        flush=True,
    )

    ev_list = list(getattr(mot, "ev_list", []) or [])
    denom = max(len(indices) - 1, 1)
    holding = False
    for order, frame_id in enumerate(indices):
        jv = np.asarray(mot.jv_list[frame_id], dtype=float)
        ev = ev_list[frame_id] if frame_id < len(ev_list) else None
        if ev is None:
            ev = jaw_open
        in_carry = cs <= frame_id <= ce
        if holding or in_carry:
            robot.goto_given_conf(jv)
        else:
            robot.goto_given_conf(jv, ee_values=float(ev))

        if in_carry and not holding:
            ghost = obj_model.copy()
            ghost.pos = table_pos
            ghost.rotmat = table_rot
            try:
                ghost.detach()
            except Exception:
                pass
            try:
                robot.hold(ghost, jaw_width=float(jaw_close))
                holding = True
            except Exception as e:
                print(f"[render] hold 失败，改用抓取相对位姿: {e}", flush=True)

        elif holding and not in_carry:
            try:
                robot.end_effector.release_all()
            except Exception:
                pass
            holding = False
            robot.goto_given_conf(jv, ee_values=float(ev))

        mesh = robot.gen_meshmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
        alpha = alpha_min + (alpha_max - alpha_min) * (order / denom)
        _set_mesh_alpha(mesh, alpha)
        mesh.attach_to(base)

        if holding:
            continue
        if frame_id < cs:
            continue
        if ac_pos is not None and ac_rot is not None and frame_id <= ce:
            obj_pos, obj_rot = _held_object_pose(robot, ac_pos, ac_rot)
        elif place_pose is not None:
            obj_pos, obj_rot = place_pose[0], place_pose[1]
        else:
            obj_pos, obj_rot = table_pos, table_rot
        try:
            _overlay_object(obj_model, obj_pos, obj_rot, alpha, base)
        except Exception as e:
            print(f"[render] {object_label}叠影失败 frame={frame_id}: {e}", flush=True)

    if holding:
        try:
            robot.end_effector.release_all()
        except Exception:
            pass
    obj_model.pos = table_pos
    obj_model.rotmat = table_rot


def render_trajectory(
    stage: str = STAGE_FULL,
    max_meshes: int = 16,
    alpha_min: float = 0.10,
    alpha_max: float = 0.95,
    sample_id: str = DEFAULT_SAMPLE,
    object_hint: str = DEFAULT_OBJECT,
    dataset_root: str | None = None,
    use_rrt: bool = False,
    scene: str = SCENE_BOX,
    object_id=None,
    slot_id=None,
):
    if stage not in STAGE_CHOICES:
        raise ValueError(f"stage 必须是 {STAGE_CHOICES}")
    if scene not in SCENE_CHOICES:
        raise ValueError(f"scene 必须是 {SCENE_CHOICES}")
    dataset_root = dataset_root or side.DEFAULT_DATASET

    if scene == SCENE_SLOT:
        cam_pos = [1.15, -0.90, 0.82]
        lookat_pos = [0.12, -0.05, 0.04]
    else:
        cam_pos = [1.65, -1.25, 1.05]
        lookat_pos = [0.03, -0.03, 0.08]
    base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos, w=1600, h=900)
    sys.modules["__main__"].base = base
    mgm.gen_frame(ax_length=0.12).attach_to(base)

    if scene == SCENE_SLOT:
        robot, mot, target, place_pose, carry_start, carry_end, jaw_open, jaw_close = (
            plan_normal_slot_motion(
                base,
                sample_id,
                dataset_root,
                object_id=object_id,
                slot_id=slot_id,
                use_rrt=use_rrt,
            )
        )
        label = "螺丝"
    else:
        robot, mot, target, place_pose, carry_start, carry_end, jaw_open, jaw_close = (
            plan_screwdriver_motion(
                base, sample_id, object_hint, dataset_root, use_rrt=use_rrt
            )
        )
        label = "螺丝刀"

    _overlay_motion(
        base,
        robot,
        mot,
        target,
        place_pose,
        carry_start,
        carry_end,
        jaw_open,
        jaw_close,
        stage,
        max_meshes,
        alpha_min,
        alpha_max,
        label,
    )
    base.run()


def main():
    p = argparse.ArgumentParser(description="渲染抓放规划轨迹叠影（入盒或入格）")
    p.add_argument(
        "--scene",
        default=SCENE_BOX,
        choices=SCENE_CHOICES,
        help="box=0001 螺丝刀入盒；slot=0502 正放螺丝入一格",
    )
    p.add_argument("--sample-id", default=None, help="默认 box→0001，slot→0502")
    p.add_argument("--object", default=DEFAULT_OBJECT, help="box 模式的类别名片段")
    p.add_argument("--object-id", type=int, default=None, help="slot 模式指定螺丝 id，默认第一颗 normal")
    p.add_argument("--slot-id", type=int, default=None, help="slot 模式目标格，默认 object_id-1")
    p.add_argument("--dataset-root", default=side.DEFAULT_DATASET)
    p.add_argument("--stage", default=STAGE_FULL, choices=STAGE_CHOICES)
    p.add_argument("--max-meshes", type=int, default=12, help="叠影帧数；<=0 用该段全部帧")
    p.add_argument("--alpha-min", type=float, default=0.10, help="pick 起点透明度（虚）")
    p.add_argument("--alpha-max", type=float, default=0.95, help="place 终点透明度（实）")
    p.add_argument("--rrt", action="store_true", help="规划时启用 RRT（更慢，默认关）")
    args = p.parse_args()
    scene = args.scene
    sample_id = args.sample_id
    if not sample_id:
        sample_id = DEFAULT_SLOT_SAMPLE if scene == SCENE_SLOT else DEFAULT_SAMPLE
    if str(sample_id).isdigit():
        sample_id = str(sample_id).zfill(4)
    render_trajectory(
        stage=args.stage,
        max_meshes=args.max_meshes,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        sample_id=sample_id,
        object_hint=args.object,
        dataset_root=args.dataset_root,
        use_rrt=bool(args.rrt),
        scene=scene,
        object_id=args.object_id,
        slot_id=args.slot_id,
    )


if __name__ == "__main__":
    main()
