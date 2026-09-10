import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.modeling.model_collection as mmc

class Omnipicker(gpi.GripperInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),ik_solver='a',name='Omnipicker',cdmesh_type=mcm.const.CDMeshType.DEFAULT):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        self.coupling.loc_flange_pose_list[0] = [np.zeros(3), np.eye(3)]
        # jaw range
        self.jaw_range = np.array([.0, 0.15])
        self.jlc_narrow = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=8, name=name)
        self.jlc_wide = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=8, name=name)
        # base link
        self.jlc_narrow.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_link.STL"))
        self.jlc_narrow.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)

        # joint 0 - hand_narrow1_joint
        self.jlc_narrow.jnts[0].loc_pos = np.array([0, -0.0195, 0.0565])
        self.jlc_narrow.jnts[0].loc_rotmat = rm.rotmat_from_euler(-2.9951, -1.5708, -0.15964)
        self.jlc_narrow.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        # self.jlc.jnts[0].motion_range = np.array([-3.14, 3.14])
        self.jlc_narrow.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "narrow1_Link.STL"))

        # joint 1 - hand_narrow2_joint
        self.jlc_narrow.jnts[1].loc_pos = np.array([0.030852, 0.018551, 0])
        self.jlc_narrow.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc_narrow.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        # self.jlc.jnts[1].motion_range = np.array([-3.14, 3.14])
        self.jlc_narrow.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "narrow2_Link.STL"))

        # joint 2 - hand_narrow3_joint
        self.jlc_narrow.jnts[2].loc_pos = np.array([0.018118, -0.01574, 0])
        self.jlc_narrow.jnts[2].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc_narrow.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        # self.jlc.jnts[2].motion_range = np.array([-3.14, 3.14])
        self.jlc_narrow.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "narrow3_Link.STL"))

        # joint 4 - hand_wide1_joint
        self.jlc_wide.jnts[1].loc_pos = np.array([0, 0.0195, 0.0565])
        self.jlc_wide.jnts[1].loc_rotmat = rm.rotmat_from_euler(-2.9951, -1.5708, -0.15964)
        self.jlc_wide.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc_wide.jnts[1].motion_range = np.array([-3.14, 3.14])
        self.jlc_wide.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wide1_Link.STL"))

        # joint 5 - hand_wide2_joint
        self.jlc_wide.jnts[2].loc_pos = np.array([0.030852, -0.018551, 0])
        self.jlc_wide.jnts[2].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc_wide.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc_wide.jnts[2].motion_range = np.array([-3.14, 3.14])
        self.jlc_wide.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wide2_Link.STL"))

        # joint 6 - hand_wide3_joint
        self.jlc_wide.jnts[3].loc_pos = np.array([0.018118, 0.01574, 0])
        self.jlc_wide.jnts[3].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc_wide.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc_wide.jnts[3].motion_range = np.array([-3.14, 3.14])
        self.jlc_wide.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wide3_Link.STL"))


        self.jlc_narrow.finalize(ik_solver=ik_solver, identifier_str=name)
        self.jlc_wide.finalize(ik_solver=ik_solver, identifier_str=name)
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.jlc_narrow.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.jlc_wide.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def get_jaw_width(self):
        return self.jlc_wide.jnts[1].motion_value

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
            jnt_values = [0.0] * 9
            jnt_values[0] = side_jawwidth
            jnt_values[1] = jaw_width
            self.jlc_narrow.goto_given_conf(jnt_values=jnt_values)
            self.jlc_wide.goto_given_conf(jnt_values = jnt_values)
        else:
            raise ValueError("The angle parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='yumi_gripper_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc_narrow.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        self.jlc_wide.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='yumi_gripper_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_root_frame=False,
                                    toggle_flange_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc_narrow.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc_wide.gen_meshmodel(rgb=rgb,
                                      alpha=alpha,
                                      toggle_flange_frame=False,
                                      toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_cdmesh=toggle_cdmesh,
                                      toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self._gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh, toggle_frame=toggle_jnt_frames)
        return m_col

if __name__ == '__main__':
        import wrs.visualization.panda.world as wd

        # 初始化仿真世界
        base = wd.World(cam_pos=[1, 0.5, 0.3], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)

        # 创建 YumiGripper 对象
        grpr = Omnipicker()

        # 可视化夹爪模型（透明度设为 0.5）
        grpr.gen_meshmodel(alpha=1).attach_to(base)

        # 运行仿真
        base.run()

