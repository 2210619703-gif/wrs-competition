#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/7/6 17:25
# @Author : ZhangXi
import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.collision_model as mcm

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

class ER4iA(mi.ManipulatorInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name='ER4iA', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)

        # anchor (base_link)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base.stl"))
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([0.2, 0.2, 0.2, 1])

        # first joint and link (J1)
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0.35])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2.9671, 2.9671])  # [-170, 170] deg
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j1.stl"))
        self.jlc.jnts[0].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([0.95, 0.85, 0.0, 1])

        # second joint and link (J2)
        self.jlc.jnts[1].loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].motion_range = np.array([-1.9199, 2.0944])  # [-110, 120] deg
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j2.stl"))
        self.jlc.jnts[1].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([0.95, 0.85, 0.0, 1])

        # third joint and link (J3)
        self.jlc.jnts[2].loc_pos = np.array([0, 0, 0.26])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, -1, 0])
        self.jlc.jnts[2].motion_range = np.array([-1.2043, 3.5779])  # [-69, 205] deg
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j3.stl"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([0.95, 0.85, 0.0, 1])

        # fourth joint and link (J4)
        self.jlc.jnts[3].loc_pos = np.array([0, 0, 0.02])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[3].loc_motion_ax = np.array([-1, 0, 0])
        self.jlc.jnts[3].motion_range = np.array([-3.3161, 3.3161])  # [-190, 190] deg
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j4.stl"))
        self.jlc.jnts[3].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([0.95, 0.85, 0.0, 1])

        # fifth joint and link (J5)
        self.jlc.jnts[4].loc_pos = np.array([0.29, 0, 0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, -1, 0])
        self.jlc.jnts[4].motion_range = np.array([-2.0944, 2.0944])  # [-120, 120] deg
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j5.stl"))
        self.jlc.jnts[4].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([0.2, 0.2, 0.2, 1])

        # sixth joint and link (J6)
        self.jlc.jnts[5].loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[5].loc_motion_ax = np.array([-1, 0, 0])
        self.jlc.jnts[5].motion_range = np.array([-6.2832, 6.2832])  # [-360, 360] deg
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j6.stl"))
        self.jlc.jnts[5].lnk.loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([0.2, 0.2, 0.2, 1])

        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)

        # tcp (法兰中心)
        self.loc_tcp_pos = np.array([0.07, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # 若需与 FANUC 出厂法兰坐标系(UTOOL0, 工具 z 轴朝法兰外)一致, 改用:
        # self.loc_tcp_rotmat = rm.rotmat_from_euler(np.pi, -np.pi / 2, 0)

        # 初始化 TracIK 求解器 (与 LRMate200iD 一致, 避免数值 IK 在 PPP/RRT 中过慢)
        if is_trac_ik:
            urdf_path = os.path.join(current_file_dir, "ER-4iA.urdf")
            # URDF 基座链名为 "Base", 第六轴链名为 "J6"
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
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        # 远端连杆(小臂/腕) vs 近端(基座/大臂) 自碰撞检测
        from_list = [l3, l4,l5]
        into_list = [lb, l0, l1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray,
           seed_jnt_values=None, option: str = "empty", toggle_dbg: bool = False):
        """求解末端逆运动学 (优先 TracIK, 无则回退 JLC 数值解)。"""
        # 将目标位姿从 TCP 变换到腕部(法兰)坐标系
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos

        if is_trac_ik and self._ik_solver is not None:
            # 变换到 anchor 基座坐标系供 TracIK 使用
            anchor_inv_homomat = np.linalg.inv(rm.homomat_from_posrot(
                self.jlc.anchor.pos, self.jlc.anchor.rotmat))
            tgt_homomat = anchor_inv_homomat.dot(rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos, tgt_rotmat = tgt_homomat[:3, 3], tgt_homomat[:3, :3]

            seed_jnt_values = self.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
            return self._ik_solver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values)
        else:
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

    base = wd.World(cam_pos=[1.6, 0, 1.0], lookat_pos=[0, 0, 0.35])
    mgm.gen_frame(ax_length=0.2).attach_to(base)
    arm = ER4iA(enable_cc=True)

    joint = np.array([0,0,0,0,0,0])
    # joint = arm.rand_conf()
    arm.goto_given_conf(joint)
    pos, rot = arm.fk(jnt_values=joint)
    print(pos, rot)
    arm.gen_meshmodel(alpha=1).attach_to(base)
    print(arm.is_collided())
    base.run()