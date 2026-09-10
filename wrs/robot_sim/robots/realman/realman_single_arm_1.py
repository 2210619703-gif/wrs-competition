import math
import numpy as np
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.realman_arm.realmanArm as rman
# import wrs.robot_sim.robots.realman.realman as rman
from wrs import wd, rm, mgm
import wrs.modeling.collision_model as mcm
# import wrs.robot_sim.end_effectors.grippers.omnipicker.omnipicker as omni
import wrs.robot_sim.end_effectors.grippers.hand.hand as hand
import os


class Realman(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="realman_arm", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 6
        home_conf[2] = math.pi / 2
        home_conf[4] = math.pi / 6
        ###初始化机械臂
        self.manipulator = rman.Realman(pos=self.pos, rotmat=self.rotmat, name=name + "_arm", enable_cc=False)
        ###hand路径
        ###/home/nvidia/Desktop/package/wrs/0000_examples/objects/hand.stl
        # hand = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "/home/nvidia/Desktop/package/wrs/0000_examples/objects/hand.stl"))
        # hand.pos = self.manipulator.gl_flange_pos
        # hand.rotmat = self.manipulator.gl_flange_rotmat
        # self.end_effector = hand

        ### 将夹爪绑定到机械臂的法兰盘上
        self.end_effector = hand.Hand(pos=self.manipulator.gl_flange_pos,
                                      rotmat=self.manipulator.gl_flange_rotmat, name="hand")
        # tool center point 设置工具中心点（TCP）为夹爪的作用中心
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        # self.manipulator.home_conf = home_conf
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # 仅设置机械臂的碰撞检测，不涉及末端执行器
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0], toggle_extcd=False)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        mlee = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0],toggle_extcd=False)
        from_list = [ml3, ml4, ml5,]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = []  # 移除末端执行器的碰撞对象

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        ###更新夹爪位姿
        self.update_end_effector()  # 更新夹爪位姿（继承自机械臂法兰盘）

    def get_jaw_width(self):
        return .1

if __name__ == '__main__':
    import time
    import wrs.basis.robot_math as rm
    import wrs.modeling.collision_model as mcm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot = Realman(enable_cc=True)

    tgt_pos = np.array([.3, .1, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print(jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
        robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    robot.show_cdprim()
    base.run()
