#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/21 11:08
# @Author : ZhangXi
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/6/7 11:11
# @Author : ZhangXi
import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.modeling.model_collection as mmc


class PigInjector(gpi.GripperInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='a', name='injector',
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        real_offset = np.array([-0.012, -0.013, 0])
        self.coupling.loc_flange_pose_list[0] = [real_offset, np.eye(3)]
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=0, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "injector.stl"),
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._back_mdl_cdnp,
            ex_radius=.015)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.75, .75, .75, 1])
        self.jlc.finalize()
        # self.loc_acting_center_pos = np.array([0.135161, 0, 0.298])
        self.loc_acting_center_pos = np.array([.160161, -.0017, .30380530]) + real_offset

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='yumi_gripper_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
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
                      name='hand_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_root_frame=False,
                                    toggle_flange_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc.gen_meshmodel(rgb=rgb,
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

    def get_jaw_width(self):
        return .1

    def change_ee_values(self, ee_values):
        """
        interface
        :param ee_values:
        :return:
        """
        return ...

    def _back_mdl_cdnp(self, ex_radius):
        collision_node = CollisionNode('htw_robot_body')
        collision_primitive_c0 = CollisionBox(Point3(.155, 0, 0.18),
                                              x=0.045 + ex_radius, y=.05 + ex_radius, z=.1 + ex_radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.054, .0, .1),
                                              x=0.0625 + ex_radius, y=0.015 + ex_radius, z=0.015 + ex_radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0, 0, 0.025),
                                              x=0.01 + ex_radius, y=.01 + ex_radius, z=.05 + ex_radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_c3 = CollisionBox(Point3(0.265, 0, 0.17),
                                              x=0.08 + ex_radius, y=.015 + ex_radius, z=.03 + ex_radius)
        collision_node.addSolid(collision_primitive_c3)
        collision_primitive_c4 = CollisionBox(Point3(-0.05, 0, 0),
                                              x=0.03 + ex_radius, y=.03 + ex_radius, z=.0003 + ex_radius)
        collision_node.addSolid(collision_primitive_c4)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(collision_node)
        return cdprim


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    # Initialize simulation world
    base = wd.World(cam_pos=[1, 0.5, 0.3], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # Create Omnipicker object
    grpr = PigInjector()

    # Visualize gripper model
    cm = grpr.gen_meshmodel(toggle_tcp_frame=True, alpha=1)
    for i in cm.cm_list:
        i.show_cdprim()
        i.attach_to(base)
    # Run simulation
    base.run()