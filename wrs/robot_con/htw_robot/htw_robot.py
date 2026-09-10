'''
Author: wang yining
Date: 2025-08-01 13:25:49
LastEditTime: 2025-08-01 13:17:47
FilePath: /remote_fs/wrs/robot_con/htw_robot/htw_robot.py
Description: 
e-mail: wangyining0408@outlook.com
'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/5/28 17:00
# @Author : ZhangXi
import atexit
import time
import numpy as np
import wrs.robot_con.realman.realman as rm
from htw.api_wrapper import LiftMotorApiWrapper, PitchMotorAPIWrapper
from Robotic_Arm.rm_robot_interface import *

try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False

#########################
LIFT_MOTOR_OFFEST = 470556750
LIFT_MOTOR_RESOLUTION = 8388608
LIFT_MOTOR_LEAD = 10


#########################

def cal_lift_motor_pos(lift_motor_api: LiftMotorApiWrapper):
    return int((-lift_motor_api.get_current_position() + LIFT_MOTOR_OFFEST) / LIFT_MOTOR_RESOLUTION * LIFT_MOTOR_LEAD)


def cal_lift_motor_set_pos(value):
    return -int(value * LIFT_MOTOR_RESOLUTION * 10) + LIFT_MOTOR_OFFEST


class HTWRobotController:
    def __init__(
            self,
            ip_rgt_arm: str = "192.168.3.18",
            ip_lft_arm: str = "192.168.0.18",
    ):
        """
        :param ip_rgt_arm: 右臂 IP 地址
        :param ip_lft_arm: 左臂 IP 地址
        """
        self.rgt_arm_x = rm.RealmanArmController(ip=ip_rgt_arm, port=8080, multiple_thread=True)
        self.lft_arm_x = rm.RealmanArmController(ip=ip_lft_arm, port=8080, multiple_thread=False)
        self.lift_motor = LiftMotorApiWrapper()
        self.pitch_motor = PitchMotorAPIWrapper()
        atexit.register(self.cleanup)

    def cleanup(self):
        print("HTW ROBOT Exit Hook")
        self.lift_motor.disable_motor()
        self.pitch_motor.disable_motor()

    def _lift_motor_get_joint_value(self) -> float:
        return cal_lift_motor_pos(lift_motor_api=self.lift_motor) / 1000

    def _pitch_motor_get_joint_value(self):
        for i in range(3):
            p = self.pitch_motor.get_position()
            if p is not None:
                return np.radians(p)
            time.sleep(.1)
        print("Cannot get position correctly")
        return None

    def set_lift_motor_joint_value(self, pos: float, blocking=True):
        p = cal_lift_motor_set_pos(pos)
        return self.lift_motor.set_position(p, blocking=blocking)

    def set_pitch_motor_joint_value(self, pos: float, blocking=True):
        return self.pitch_motor.set_position(pos, blocking=blocking)

    def get_joint_values(self):
        joint_values = np.array([
            self._lift_motor_get_joint_value(),
            self._pitch_motor_get_joint_value(),
            *self.rgt_arm_x.get_joint_values(),
            *self.lft_arm_x.get_joint_values()
        ])
        return joint_values

    def move_j(self, joint_angles: list, speed: int = 5, is_block=True):
        """
        控制左臂运动到指定关节角度位置
        :param joint_angles: 目标关节角度列表，单位：度
        :param speed: 运动速度，单位：度/秒
        :param is_block: 是否阻塞等待运动完成
        """
        result = self.lft_arm_x.move_j(joint_angles, speed=speed, is_block=is_block)
        if result == 0:
            print(f"✅ 成功移动到关节位置: {joint_angles}")
        else:
            print("❌ 关节运动失败")

    def __del__(self):
        del self.lift_motor
        del self.pitch_motor

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("HTWRobot Controller Delete")
        del self.lift_motor
        del self.pitch_motor


if __name__ == "__main__":
    # 创建机器人控制器实例
    rbt_x = HTWRobotController()
    # rbt_x.lift_motor.enable_motor()
    # if not rbt_x.pitch_motor.is_enabled:
    #     rbt_x.pitch_motor.enable_motor()
    # rbt_x.set_pitch_motor_joint_value(5, blocking=True)
    # rbt_x.lift_motor.enable_motor()
    # rbt_x.set_lift_motor_joint_value(4.8)
    # print("S")
    # rbt_x.lift_motor.enable_motor()
    # rbt_x.set_lift_motor_joint_value(4.8, )

    print(rbt_x.get_joint_values())
    # pos = rbt_x._lift_motor_get_joint_value()
    # rbt_x.set_lift_motor_joint_value(pos - 0.03)
    # target_joints = [-80, -90, -10, 80, 12, -180]
    # rbt_x.move_j(target_joints)
    # joint = rbt_x.get_joint_values()
    # info = rbt_x.get_joint_values()
    # print(info)
    # print(f"当前升降电机位置（米）：{pos}")
    # print(f"当前左手手臂的位置（米）：{joint}")
#
