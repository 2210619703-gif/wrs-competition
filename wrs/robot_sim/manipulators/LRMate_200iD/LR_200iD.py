#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/4/13 12:59
# @Author : ZhangXi
import os
import sys
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
from wrs import rrtc
import wrs.modeling.collision_model as mcm

# ==========================================
# 尝试加载 Trac IK 模块
# ==========================================
try:
    from trac_ik import TracIK

    is_trac_ik = True
    print("Trac IK module loaded successfully")
except Exception as e:
    print(
        f"Trac IK not available ({e}); using JLC numerical IK fallback. "
        "Optional: pip install pytracik"
    )
    is_trac_ik = False


class LRMate200iD(mi.ManipulatorInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name='LRMate200iD', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)

        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "Base.STL"))
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([0.2, 0.2, 0.2, 1])

        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0.042741])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2.9671, 2.9671])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "J1.STL"))
        self.jlc.jnts[0].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([1.0, 1.0, 0.0, 1])

        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([0.05, 0, 0.28726])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, -1, 0])
        self.jlc.jnts[1].motion_range = np.array([-2.5307, 1.7453])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "J2.STL"))
        self.jlc.jnts[1].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([1.0, 1.0, 0.0, 1])

        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([0, 0, 0.33])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, -1, 0])
        self.jlc.jnts[2].motion_range = np.array([-1.2217, 3.5779])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "J3.STL"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([1.0, 1.0, 0.0, 1])

        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([0.088001, 0, 0.035027])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[3].loc_motion_ax = np.array([-1, 0, 0])
        self.jlc.jnts[3].motion_range = np.array([-3.1416, 3.1416])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "J4.STL"))
        self.jlc.jnts[3].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(.0, .0, .0)
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([1.0, 1.0, 0.0, 1])

        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0.2454, 0, 0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, -1, 0])
        self.jlc.jnts[4].motion_range = np.array([-2.1817, 2.1817])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "J5.STL"))
        self.jlc.jnts[4].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0, .0, .0)
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([1.0, 1.0, 0.0, 1])

        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0.05, 0, 0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(1.5708, 0, 0)  # 改为绕 X 轴
        self.jlc.jnts[5].loc_motion_ax = np.array([-1, 0, 0])  # 改为负 X 轴
        self.jlc.jnts[5].motion_range = np.array([-2 * np.pi, 2 * np.pi])
        # self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "J6.STL"))
        # self.jlc.jnts[5].lnk.loc_pos = np.array([0, 0, 0])
        # self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(.0, .0, .0)
        # self.jlc.jnts[5].lnk.cmodel.rgba = np.array([0.2, 0.2, 0.2, 1])

        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)

        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)

        # 初始化 TracIK 求解器
        if is_trac_ik:
            urdf_path = os.path.join(current_file_dir, "LRMate-200iD.urdf")
            # 根据提供的 URDF 文件，基座名称为 "Base"，第六轴为 "J6"
            self._ik_solver = TracIK("Base", "J6", urdf_path,
                                     timeout=0.005,
                                     solver_type="Distance")
        else:
            self._ik_solver = None

        # set up cc
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        # l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        from_list = [l3, l4]
        into_list = [lb, l0]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray,
           seed_jnt_values=None, option: str = "empty", toggle_dbg: bool = False):
        """
        Solve the inverse kinematics for the end‑effector.
        """
        # Transform target pose into the wrist coordinate frame
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos

        if is_trac_ik and self._ik_solver is not None:
            # convert to the anchor base frame for Trac IK
            anchor_inv_homomat = np.linalg.inv(rm.homomat_from_posrot(
                self.jlc.anchor.pos, self.jlc.anchor.rotmat))
            tgt_homomat = anchor_inv_homomat.dot(rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos, tgt_rotmat = tgt_homomat[:3, 3], tgt_homomat[:3, :3]

            seed_jnt_values = self.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
            return self._ik_solver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values)
        else:
            # Fall back to numerical IK provided by the JLC
            return self.jlc.ik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values,
                               toggle_dbg=toggle_dbg)

    def is_collided(self, obstacle_list=[], other_robot_list=[], **kwargs):
        """
        重写 is_collided 以吸收 RRT 规划器传递的 'other_robot_list' 参数。
        ManipulatorInterface 原生不接受此参数。
        """
        return super().is_collided(obstacle_list=obstacle_list, **kwargs)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.modeling.geometric_model as gm
    import numpy as np

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0.2])
    mgm.gen_frame(ax_length=.2).attach_to(base)
    arm = LRMate200iD(enable_cc=True)

    joint = np.array([0,0,0,0,0,0])
    arm.goto_given_conf(joint)
    arm.gen_meshmodel(alpha=1).attach_to(base)
    # print(arm.is_collided())
    base.run()