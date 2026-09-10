import math
import numpy as np
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.realman.realman_single_arm as rman
import wrs.robot_sim.end_effectors.grippers.Omnipicker_gripper.Omnipicker_gripper as opg
from wrs import wd, rm, mgm, mcm  # 添加 mcm 用于生成立体框

class Realman(sari.SglArmRobotInterface):

    local_pos_offset = np.array([0, 0, 0.08])
    local_rotmat_offset = rm.rotmat_from_axangle([1, 0, 0], math.pi)

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="realman", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 6
        home_conf[2] = math.pi / 2
        home_conf[4] = math.pi / 6
        self.manipulator = rman.RealmanArm(pos=self.pos, rotmat=self.rotmat, name=name + "_arm", enable_cc=False)
        self.end_effector = opg.Omnipicker(pos=self.manipulator.gl_flange_pos,
                                           rotmat=self.manipulator.gl_flange_rotmat, name="opg_" + name)
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        self.manipulator.home_conf = home_conf
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0], toggle_extcd=False)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [ml3, ml4, ml5]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = []

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def update_end_effector(self, ee_values=None):
        fl_pos = self.manipulator.gl_flange_pos
        fl_rotmat = self.manipulator.gl_flange_rotmat
        ee_pos = fl_pos + fl_rotmat @ self.local_pos_offset
        ee_rotmat = fl_rotmat @ self.local_rotmat_offset
        self.end_effector.fix_to(pos=ee_pos, rotmat=ee_rotmat)

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import time

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = Realman(enable_cc=True)

    robot.change_jaw_width(.04)

    tgt_pos = np.array([.3, .1, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print("IK解：", jnt_values)

    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.update_end_effector()
        robot.gen_meshmodel(alpha=1, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
        #robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
        robot.end_effector.gen_meshmodel().attach_to(base)
        mgm.gen_frame(pos=robot.end_effector.pos, rotmat=robot.end_effector.rotmat).attach_to(base)

    # tic = time.time()
    # result, contacts = robot.is_collided(obstacle_list=[box], toggle_contacts=True)
    # print("是否发生碰撞：", result)
    # toc = time.time()
    # print("检测耗时：", toc - tic)
    # for pnt in contacts:
    #     mgm.gen_sphere(pnt).attach_to(base)

    base.run()
