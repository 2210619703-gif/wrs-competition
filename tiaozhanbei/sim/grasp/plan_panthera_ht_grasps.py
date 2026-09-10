#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 Panthera-HT 的 PantheraGripper 生成参考抓取 pickle（物体坐标系）。

与 pick_multi_block/grasp/plan_block_grasps.py 同套路，但末端改为
``wrs.robot_sim.end_effectors.grippers.panthera_gripper.PantheraGripper``，
夹爪参数默认与 ``PantheraHTSglArm`` 一致。

输出（默认写到本目录）::
    panthera_ht_block_grasps.pickle           # 对踵采样全集
    panthera_ht_block_grasps_topdown.pickle   # 仅保留爪尖朝下

运行（仓库根目录）::
    python tiaozhanbei/sim/grasp/plan_panthera_ht_grasps.py --no-vis
    python tiaozhanbei/sim/grasp/plan_panthera_ht_grasps.py --mode both --no-vis
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs import wd, rm, mcm, mgm, gpa, gg
from wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper import PantheraGripper

# 与 PantheraHTSglArm 一致
DEFAULT_JAW_MAX = 0.012
DEFAULT_CLOSE_BIAS = 0.006
# 参考方块须小于夹爪最大开口（对踵规划会跳过 jaw_width > jaw_range[1]）
DEFAULT_BLOCK_EDGE = 0.008

ANTIPODAL_PICKLE = "panthera_ht_block_grasps.pickle"
TOPDOWN_PICKLE = "panthera_ht_block_grasps_topdown.pickle"  # antipodal 滤出的顶抓
TOPDOWN_YAW_PICKLE = "panthera_ht_block_grasps_topdown_yaw.pickle"  # 顶面 yaw 采样

ANGLE_BETWEEN_CONTACT_NORMALS = rm.radians(175)
ROTATION_INTERVAL = rm.radians(30)
MAX_SAMPLES = 80
MIN_DIST_BETWEEN_SAMPLED_CONTACT_POINTS = 0.003
CONTACT_OFFSET = 0.0005

DEFAULT_YAW_SAMPLES = 12
DEFAULT_TOPDOWN_TOL_DEG = 20.0
MAX_VIS_GRASPS = 40


def make_gripper(jaw_max: float, close_bias: float) -> PantheraGripper:
    return PantheraGripper(
        jaw_range=np.array([0.0, float(jaw_max)], dtype=float),
        close_bias=float(close_bias),
        use_palm_mesh=True,
    )


def make_block(edge: float):
    return mcm.gen_box(
        xyz_lengths=rm.vec(edge, edge, edge),
        rgb=rm.const.tab20_list[0],
        alpha=1.0,
    )


def plan_antipodal(gripper, block, in_object_frame: bool = True):
    return gpa.plan_gripper_grasps(
        gripper=gripper,
        obj_cmodel=block,
        angle_between_contact_normals=ANGLE_BETWEEN_CONTACT_NORMALS,
        rotation_interval=ROTATION_INTERVAL,
        max_samples=MAX_SAMPLES,
        min_dist_between_sampled_contact_points=MIN_DIST_BETWEEN_SAMPLED_CONTACT_POINTS,
        contact_offset=CONTACT_OFFSET,
        toggle_dbg=False,
        in_object_frame=in_object_frame,
    )


def plan_topdown_yaw(
    gripper,
    block_edge: float,
    jaw_width: float,
    yaw_samples: int = DEFAULT_YAW_SAMPLES,
) -> gg.GraspCollection:
    """桌面顶抓：接触点在方块顶面中心，绕世界 z 采样 yaw（物体系，物体在原点）。"""
    if not (gripper.jaw_range[0] <= jaw_width <= gripper.jaw_range[1]):
        raise ValueError(
            f"jaw_width={jaw_width} 超出夹爪范围 "
            f"[{gripper.jaw_range[0]}, {gripper.jaw_range[1]}]"
        )
    collection = gg.GraspCollection(end_effector=gripper)
    contact_pos = np.array([0.0, 0.0, block_edge / 2.0], dtype=float)
    approaching = np.array([0.0, 0.0, -1.0], dtype=float)
    opening = np.array([1.0, 0.0, 0.0], dtype=float)
    n_yaw = max(int(yaw_samples), 1)
    for k in range(n_yaw):
        yaw = 2.0 * np.pi * k / n_yaw
        rot_z = rm.rotmat_from_axangle(rm.const.z_ax, yaw)
        grasp = gripper.grip_at_by_twovecs(
            jaw_center_pos=contact_pos,
            approaching_direction=rot_z.dot(approaching),
            thumb_opening_direction=rot_z.dot(opening),
            jaw_width=float(jaw_width),
        )
        collection.append(grasp)
    return collection


def filter_topdown(src: gg.GraspCollection, tol_deg: float) -> gg.GraspCollection:
    cos_thr = float(np.cos(np.radians(tol_deg)))
    out = gg.GraspCollection(end_effector=src.end_effector)
    for grasp in src:
        # PantheraGripper acting-center +z 为接近轴（与 WRSGripper 约定相同）
        gripper_z = grasp.ac_rotmat[:, 2]
        if float(-gripper_z[2]) >= cos_thr:
            out.append(grasp)
    return out


def visualize(gripper, block, collection, max_vis: int):
    base = wd.World(cam_pos=rm.vec(0.25, 0.25, 0.18), lookat_pos=rm.vec(0, 0, 0.01))
    mgm.gen_frame().attach_to(base)
    block.attach_to(base)
    n_show = min(max_vis, len(collection)) if max_vis > 0 else len(collection)
    print(f"[vis] 显示前 {n_show}/{len(collection)} 个 grasp")
    for i in range(n_show):
        g = collection[i]
        gripper.grip_at_by_pose(
            jaw_center_pos=g.ac_pos,
            jaw_center_rotmat=g.ac_rotmat,
            jaw_width=g.ee_values,
        )
        gripper.gen_meshmodel(alpha=0.15, toggle_tcp_frame=False).attach_to(base)
    base.run()


def main():
    parser = argparse.ArgumentParser(description="Panthera-HT 参考抓取 pickle 生成")
    parser.add_argument(
        "--mode",
        choices=("antipodal", "topdown", "both"),
        default="both",
        help="antipodal=对踵采样; topdown=顶面 yaw 采样; both=两者都做",
    )
    parser.add_argument("--block-edge", type=float, default=DEFAULT_BLOCK_EDGE,
                        help=f"参考方块边长 (m)，默认 {DEFAULT_BLOCK_EDGE}")
    parser.add_argument("--jaw-max", type=float, default=DEFAULT_JAW_MAX,
                        help=f"夹爪最大开口 (m)，默认 {DEFAULT_JAW_MAX}（与 PantheraHTSglArm 一致）")
    parser.add_argument("--close-bias", type=float, default=DEFAULT_CLOSE_BIAS,
                        help=f"close_bias (m)，默认 {DEFAULT_CLOSE_BIAS}")
    parser.add_argument("--topdown-jaw", type=float, default=None,
                        help="顶抓固定夹距；默认取 min(block_edge*0.95, jaw_max)")
    parser.add_argument("--yaw-samples", type=int, default=DEFAULT_YAW_SAMPLES)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOPDOWN_TOL_DEG,
                        help="从 antipodal 结果滤 topdown 的角度容差（度）")
    parser.add_argument("--out-dir", type=str, default=_THIS_DIR)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--max-vis", type=int, default=MAX_VIS_GRASPS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gripper = make_gripper(args.jaw_max, args.close_bias)
    block = make_block(args.block_edge)
    print(
        f"[plan] PantheraGripper jaw_range=[0, {args.jaw_max}] "
        f"close_bias={args.close_bias} block={args.block_edge}m"
    )

    antipodal_path = os.path.join(args.out_dir, ANTIPODAL_PICKLE)
    topdown_path = os.path.join(args.out_dir, TOPDOWN_PICKLE)
    topdown_yaw_path = os.path.join(args.out_dir, TOPDOWN_YAW_PICKLE)
    vis_collection = None

    if args.mode in ("antipodal", "both"):
        print("[plan] 对踵抓取规划中…")
        antipodal = plan_antipodal(gripper, block, in_object_frame=True)
        print(f"[plan] antipodal 共 {len(antipodal)} 个 grasp")
        antipodal.save_to_disk(file_name=antipodal_path)
        print(f"[plan] 已保存 -> {antipodal_path}")

        filtered = filter_topdown(antipodal, args.tol)
        print(f"[plan] antipodal 中 topdown 过滤保留 {len(filtered)}/{len(antipodal)}")
        if len(filtered) > 0:
            filtered.save_to_disk(file_name=topdown_path)
            print(f"[plan] 已保存 topdown(滤) -> {topdown_path}")
            vis_collection = filtered
        else:
            print("[plan] antipodal 无 topdown grasp（可放宽 --tol 或改用 --mode topdown）")
            vis_collection = antipodal if len(antipodal) else None

    if args.mode in ("topdown", "both"):
        jaw_w = args.topdown_jaw
        if jaw_w is None:
            jaw_w = float(min(args.block_edge * 0.95, args.jaw_max))
        print(f"[plan] 顶面 yaw 采样，jaw_width={jaw_w:.4f}m, n={args.yaw_samples}")
        yaw_gc = plan_topdown_yaw(
            gripper,
            block_edge=args.block_edge,
            jaw_width=jaw_w,
            yaw_samples=args.yaw_samples,
        )
        yaw_gc.save_to_disk(file_name=topdown_yaw_path)
        print(f"[plan] 顶面 yaw 共 {len(yaw_gc)} 个 grasp -> {topdown_yaw_path}")
        # 仅 topdown 模式且尚未有滤出文件时，同步写一份到 TOPDOWN_PICKLE 约定名
        if args.mode == "topdown" or not os.path.isfile(topdown_path):
            yaw_gc.save_to_disk(file_name=topdown_path)
            print(f"[plan] 同步写入 -> {topdown_path}")
        vis_collection = yaw_gc

    if args.no_vis or vis_collection is None or len(vis_collection) == 0:
        print("[plan] 完成（无可视化）")
        return

    visualize(gripper, block, vis_collection, args.max_vis)


if __name__ == "__main__":
    main()
