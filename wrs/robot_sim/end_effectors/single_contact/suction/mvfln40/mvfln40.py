import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.single_contact.single_contact_interface as si


class MVFLN40(si.SCTInterface):
    """
    MVFLN40 吸盘：与 ORSD 一致，使用 Anchor + n_dof=0 的 JLChain，避免已废弃的 coupling.jnts API。
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name='mvfln40',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir = os.path.dirname(__file__)
        flange_gl_pos, flange_gl_rot = self.coupling.gl_flange_pose_list[0]
        self.jlc = rkjlc.JLChain(pos=flange_gl_pos,
                                 rotmat=flange_gl_rot,
                                 n_dof=0,
                                 name=f'{name}_jlc')
        # 法兰到吸盘连杆的偏置（与原 mesh_file 链上 jnts[1] loc_pos 一致）
        self.jlc.anchor.loc_flange_pose_list[0][0] = np.array([0., 0., 0.068])
        mesh_path = os.path.join(this_dir, 'meshes', 'mvfln40.stl')
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=mesh_path, cdmesh_type=self.cdmesh_type, name='mvfln40_mesh')
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.55, .55, .55, 1])
        self.jlc.finalize()
        # 吸盘接触中心：在 coupling 局部系下 z+0.068m
        cpl_lp, cpl_lr = self.coupling.loc_flange_pose_list[0]
        self.loc_acting_center_pos = cpl_lr @ np.array([0., 0., 0.068]) + cpl_lp
        self.loc_acting_center_rotmat = np.eye(3)
        self.suction_center_pos = self.loc_acting_center_pos.copy()
        self.cdmesh_elements = [self.jlc.anchor.lnk_list[0]]
        _ = enable_cc  # 保留参数以兼容旧调用；碰撞网格由 cdmesh_elements 提供

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.fix_to(pos=self._pos, rotmat=self._rotmat)
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0],
                        self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frame=False,
                       toggle_connjnt=False,
                       name='suction_stickmodel'):
        _ = toggle_connjnt
        mm_collection = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False,
                                     toggle_flange_frame=False).attach_to(mm_collection)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frame,
                                toggle_flange_frame=False).attach_to(mm_collection)
        if toggle_tcp_frame:
            suction_center_gl_pos = self._rotmat @ self.suction_center_pos + self._pos
            suction_center_gl_rotmat = self._rotmat @ self.loc_acting_center_rotmat
            mgm.gen_dashed_stick(spos=self._pos,
                                 epos=suction_center_gl_pos,
                                 radius=.0062,
                                 rgba=[.5, 0, 1, 1],
                                 type='round').attach_to(mm_collection)
            mgm.gen_myc_frame(pos=suction_center_gl_pos,
                              rotmat=suction_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection

    def gen_meshmodel(self,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=None,
                      name='mvfln40_meshmodel'):
        rgb, alpha = None, None
        if rgba is not None:
            rgba_arr = np.asarray(rgba, dtype=float)
            rgb = rgba_arr[:3]
            alpha = float(rgba_arr[3]) if rgba_arr.size >= 4 else 1.0
        mm_collection = mmc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_root_frame=False,
                                    toggle_flange_frame=False).attach_to(mm_collection)
        self.jlc.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frame).attach_to(mm_collection)
        if toggle_tcp_frame:
            suction_center_gl_pos = self._rotmat @ self.suction_center_pos + self._pos
            suction_center_gl_rotmat = self._rotmat @ self.loc_acting_center_rotmat
            mgm.gen_dashed_stick(spos=self._pos,
                                 epos=suction_center_gl_pos,
                                 radius=.0062,
                                 rgba=[.5, 0, 1, 1],
                                 type='round').attach_to(mm_collection)
            mgm.gen_myc_frame(pos=suction_center_gl_pos,
                              rotmat=suction_center_gl_rotmat).attach_to(mm_collection)
        self._gen_oiee_meshmodel(mm_collection, rgb=rgb, alpha=alpha)
        return mm_collection


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    grpr = MVFLN40(enable_cc=True)
    grpr.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    base.run()
