#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import wrs.modeling.geometric_model as gm
import wrs.basis.robot_math as rm  # 【新增】引入底层数学库
import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.LRMate_200iD.LR_200iD import LRMate200iD
from wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v6 import WRSGripper6


class LRMate200iDSglArm(sari.SglArmRobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name="lrmate200id_arm", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        self.manipulator = LRMate200iD(pos=self.pos,
                                       rotmat=self.rotmat,
                                       name=name + "_arm",
                                       enable_cc=True)
        self.manipulator.home_conf = np.zeros(6)

        R_old = rm.rotmat_from_euler(0, 1.5708, 0)
        R_new = rm.rotmat_from_euler(1.5708, 0, 0)
        self.ee_offset_rotmat = np.dot(R_new.T, R_old)
        self.end_effector = WRSGripper6(
            pos=self.manipulator.gl_flange_pos,
            rotmat=np.dot(self.manipulator.gl_flange_rotmat, self.ee_offset_rotmat),
            name=name + "_gripper")

        self.manipulator.loc_tcp_pos = np.dot(self.ee_offset_rotmat, self.end_effector.loc_acting_center_pos)
        self.manipulator.loc_tcp_rotmat = np.dot(self.ee_offset_rotmat, self.end_effector.loc_acting_center_rotmat)

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """碰撞检测设置。

        重要说明（关于自碰撞）：
            LRMate 的链节 cdprim 默认使用 AABB（轴对齐包围盒）。在小臂折叠
            到工作空间较低的构型时，远端链节（ml3/ml4/ee）的 AABB 会与近端
            链节（mlb/ml0/ml1/ml2）的 AABB 大量重叠，导致 ``is_collided`` 永远
            返回 True（即便 mesh 实际并不相交）。LRMate 的 IK 已经过 motion_range
            过滤，正常返回的关节解都不会让真实 mesh 自相交，所以这里关闭自碰撞
            对，仅保留外部障碍物碰撞检测。

        关于外部碰撞（重要）：
            机器人基座 (mlb) 必须 ``toggle_extcd=False``，否则基座的 cdprim
            会永远和桌面 cdprim 重叠（基座坐在桌面上），导致 ``is_collided`` 永远
            返回 True。
        """
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0], toggle_extcd=False)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        # end-effector
        ee_cces = [self.cc.add_cce(cde) for cde in self.end_effector.cdelements]
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = ee_cces

    def update_end_effector(self, ee_values=None):
        """重写父类方法：在父类 EE 位姿基础上叠加 ee_offset_rotmat。

        这样 ``goto_given_conf``/``restore_state``/``goto_home_conf`` 等所有
        会调用 ``update_end_effector`` 的路径都能得到正确的末端朝向。
        """
        if self.end_effector is None:
            return
        if ee_values is not None:
            self.end_effector.change_ee_values(ee_values=ee_values)
        gl_flange_pos = self.manipulator.gl_flange_pos
        gl_flange_rotmat = np.dot(self.manipulator.gl_flange_rotmat, self.ee_offset_rotmat)
        self.end_effector.fix_to(pos=gl_flange_pos, rotmat=gl_flange_rotmat)

    def goto_given_conf(self, jnt_values, ee_values=None):
        return super().goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        result = super().fk(jnt_values, toggle_jacobian, update)
        if update:
            self.update_end_effector()
        return result

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    base = wd.World(cam_pos=[2.0, 2.0, 1.5], lookat_pos=[0, 0, 0.3])
    mgm.gen_frame(ax_length=1).attach_to(base)
    robot = LRMate200iDSglArm(enable_cc=True)
    robot.change_jaw_width(0.15)

    joint = np.array([0,0.2,0,-0.5,0,0])
    print("Random Joint:", joint)

    pos, rot = robot.fk(joint)
    print("TCP Pos:", pos)
    print("TCP Rot:\n", rot)
    # gm.gen_sphere(pos=pos, radius=0.02, rgb=np.array([1, 0, 0])).attach_to(base)
    joint_ik = robot.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=joint)
    print("IK Solved Joint:", joint_ik)

    robot.goto_given_conf(joint_ik)
    robot.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False, alpha=0.1).attach_to(base)
    robot.show_cdprim()
    print(robot.is_collided(toggle_contacts=True))
    pos_collided = np.array([6.46312535e-02, -2.60499073e-04,  7.13353574e-01])
    gm.gen_sphere(pos=pos, radius=0.02, rgb=np.array([0, 1, 0])).attach_to(base)
    base.run()