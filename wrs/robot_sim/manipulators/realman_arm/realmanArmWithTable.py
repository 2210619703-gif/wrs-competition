import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jl as rkjl

class Realman(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name='Realman', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # 初始化anchor并设置给jlc
        self._init_anchor_with_links()


        #底部支架
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "/home/nvidia/Desktop/package/wrs/0000_examples/objects/box_0.5_0.5_0.68.stl"))
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.anchor.lnk_list[0].loc_pos = np.array([0,0,0 ])
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        # anchor
        self.jlc.anchor.lnk_list[1].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_link.STL"))
        self.jlc.anchor.lnk_list[1].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.anchor.lnk_list[1].loc_pos = np.array([0,0,0])   
        self.jlc.anchor.lnk_list[1].cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        
        
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([0,0,0.2405])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-3.107,3.107])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link1.STL"))
        self.jlc.jnts[0].lnk.loc_pos = np.array([0,0,0])
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0,0,0)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -1.5708, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-2.269,2.269])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link2.STL"))
        self.jlc.jnts[1].lnk.loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0,0,0)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([0.256, .0, .0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0, 0, 1.5708)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-2.256,2.356])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link3.STL"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0,0,0])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0,0,0)
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([0,-0.21,0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5708,0,0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-3.107,3.107])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link4.STL"))
        self.jlc.jnts[3].lnk.loc_pos = np.array([0,0,0])
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(.0,.0,.0)
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2.234,2.234])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link5.STL"))
        self.jlc.jnts[4].lnk.loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(0, .0, .0)
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([.0, -0.144, .0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(1.5708, 0, 0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-6.283,6.283])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link6.STL"))
        self.jlc.jnts[5].lnk.loc_pos = np.array([.0,.0,.0])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(.0, .0, .0)
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([0.79216,0.81961,0.93333,1])
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
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
        from_list = [l3, l4, l5]
        into_list = [lb, l0]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def _init_anchor_with_links(self):
        """完全重新初始化anchor及其链接"""
        # 创建新的anchor实例
        new_anchor = rkjl.Anchor(
            name=f"{self.name}_anchor",
            pos=self.jlc.anchor.pos,  # 保持原有位置
            rotmat=self.jlc.anchor.rotmat,  # 保持原有旋转
            n_flange=1,  # 法兰数量
            n_lnk=2  # 想要设置的链接数量
        )
        
        # # 创建链接
        # link1 = rkjl.Link(
        #     name="table",
        #     loc_pos=np.array([0,0,0.34]),
        #     loc_rotmat=np.eye(3)
        # )
        
        # link2 = rkjl.Link(
        #     name="flange",
        #     loc_pos=np.array([0,0,0.68]),
        #     loc_rotmat=np.eye(3)
        # )
        
        # 设置链接列表（这会自动更新n_lnk）
        # new_anchor.lnk_list = [link1, link2]
        
        # 替换jlc中的anchor
        self.jlc.anchor = new_anchor
    

if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import time

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    arm = Realman(enable_cc=True)

    # 随机生成关节角度
    rand_conf = arm.rand_conf()

    # 打印当前关节角度
    print("当前关节角度:", rand_conf)

    # 进行前向运动学计算
    pos, rotmat = arm.fk(rand_conf, update=True)

    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm.gen_stickmodel()
    arm_mesh.attach_to(base)
    base.run()
