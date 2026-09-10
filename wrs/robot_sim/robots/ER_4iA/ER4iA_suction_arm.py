#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# wrs/robot_sim/robots/ER_4iA -> 仓库根目录 (上溯 4 级)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.ER_4iA.ER4iA import ER4iA
from wrs.robot_sim.robots.LRMate_200iD import LRMate200iDSglArm
# 复用无 STL 依赖的程序化吸盘 (接口与 MVFLN40 兼容)
from pick_multi_block.new_plan.sim_suction import SimSuction

# ER4iA J6(腕心) -> 官方法兰面 的偏置 (沿法兰 +x), 与 ER4iA.py 的 loc_tcp_pos 对应
FLANGE_TO_FACE = np.array([0.07, 0.0, 0.0])


class ER4iASuctionArm(sari.SglArmRobotInterface):
    """ER-4iA 单臂 + 吸盘。TCP = 吸盘接触中心（法兰面起 170 mm，与现场 UTOOL Z 一致）。

    安装方式与实机相同：吸盘轴线沿法兰外向（WRS 法兰 +X）。默认 home 取 J5=-90°，
    使吸盘接近轴朝世界 -Z（与实机工作姿态一致）。相对旧版额外绕工具轴 yaw 180°，
    以对齐 FANUC UTOOL WPR=0 的 TCP 横滚/偏航。
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name="er4ia_suction_arm", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        self.manipulator = ER4iA(pos=self.pos,
                                 rotmat=self.rotmat,
                                 name=name + "_arm",
                                 enable_cc=True)
        # 默认姿态：腕部下弯，吸盘朝世界 -Z（对应实机俯拍/抓取姿态）
        self.manipulator.home_conf = np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0])

        # Roty(90°): 吸盘局部 +z → 法兰外向 +x（同轴安装，长度 170 mm）
        # Rotz(180°): 对齐示教器 UTOOL WPR=0 的 TCP x/y，消除 verify 中 ~180° 姿态误差
        self.ee_offset_rotmat = np.dot(
            rm.rotmat_from_euler(0, np.pi / 2, 0),
            rm.rotmat_from_euler(0, 0, np.pi),
        )

        # 吸盘装在"法兰面"(腕心 + 0.07 沿工具外向), 而非腕心。
        face_pos = self.manipulator.gl_flange_pos + \
            self.manipulator.gl_flange_rotmat @ FLANGE_TO_FACE
        self.end_effector = SimSuction(
            pos=face_pos,
            rotmat=np.dot(self.manipulator.gl_flange_rotmat, self.ee_offset_rotmat),
            name=name + "_suction",
            enable_cc=True,
        )

        # PickPlacePlanner 缺省会读 jaw_range[1]; 吸盘无夹距, 给 [0,0] 即可。
        self.end_effector.jaw_range = np.array([0.0, 0.0])

        # 把吸盘接触中心作为 manipulator 的 TCP:
        #   位置 = 法兰面偏置 + 旋转后的接触中心（总长仍为 0.07+0.17 m）
        #   朝向 = ee_offset @ 接触中心朝向
        self.manipulator.loc_tcp_pos = FLANGE_TO_FACE + np.dot(
            self.ee_offset_rotmat, self.end_effector.loc_acting_center_pos)
        self.manipulator.loc_tcp_rotmat = np.dot(
            self.ee_offset_rotmat, self.end_effector.loc_acting_center_rotmat)

        # 初始化到 home，使打开仿真时吸盘即朝 -Z
        self.goto_given_conf(self.manipulator.home_conf)

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """碰撞检测设置 (沿用 LRMate 吸盘臂策略)。

        - 关闭自碰撞对 (AABB 在折叠构型下会误报, IK 已用 motion_range 约束)。
        - 基座 ``toggle_extcd=False``, 避免与桌面永远相撞。
        - 吸盘 mesh 作为 dynamic_ext_list, 参与外部障碍物碰撞。
        """
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0], toggle_extcd=False)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        # 吸盘本体通过 cdmesh_elements 暴露需要参与碰撞的链节
        ee_cces = [self.cc.add_cce(cde) for cde in self.end_effector.cdmesh_elements]
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = ee_cces

    def update_end_effector(self, ee_values=None):
        """重写父类: 末端固定到"法兰面 + ee_offset", 保证工具位姿正确。"""
        if self.end_effector is None:
            return
        if ee_values is not None:
            self.end_effector.change_ee_values(ee_values=ee_values)
        gl_flange_pos = self.manipulator.gl_flange_pos + \
            self.manipulator.gl_flange_rotmat @ FLANGE_TO_FACE
        gl_flange_rotmat = np.dot(self.manipulator.gl_flange_rotmat, self.ee_offset_rotmat)
        self.end_effector.fix_to(pos=gl_flange_pos, rotmat=gl_flange_rotmat)

    def goto_given_conf(self, jnt_values, ee_values=None):
        return super().goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        result = super().fk(jnt_values, toggle_jacobian, update)
        if update:
            self.update_end_effector()
        return result

    # ----------------- 与两指夹爪 API 兼容的 no-op 包装 -----------------
    def get_jaw_width(self):
        """吸盘没有真正的夹距; 为兼容旧调用返回 0.0。"""
        return 0.0

    def change_jaw_width(self, jaw_width):
        """no-op: 吸盘没有可调夹距。"""
        _ = jaw_width


# if __name__ == '__main__':
#     import wrs.visualization.panda.world as wd
#     import wrs.modeling.geometric_model as mgm
#
#     base = wd.World(cam_pos=[1.6, 0, 1.0], lookat_pos=[0, 0, 0.35])
#     mgm.gen_frame(ax_length=0.2).attach_to(base)
#
#     robot = ER4iASuctionArm(enable_cc=True)
#     robot2 = LRMate200iDSglArm(enable_cc=True)
#     joint = robot.rand_conf()
#     joint1 = np.array([0,0,0,0,0,0])
#     print("Random Joint:", joint)
#
#     pos, rot = robot.fk(joint)
#     print("TCP Pos:", pos)
#     print("TCP Rot:\n", rot)
#     joint_ik = robot.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=joint)
#     print("IK Solved Joint:", joint_ik)
#
#     robot.goto_given_conf(joint_ik)
#     robot.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
#
#     robot2.goto_given_conf(joint1)
#     robot2.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
#     base.run()

if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[1.6, 0, 1.0], lookat_pos=[0, 0, 0.35])
    mgm.gen_frame(ax_length=0.2).attach_to(base)

    robot = ER4iASuctionArm()
    pos, rot = robot.fk(robot.manipulator.home_conf, update=True)
    print("home TCP pos:", pos)
    print("home approach (TCP z-axis):", rot[:, 2])
    robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    base.run()
