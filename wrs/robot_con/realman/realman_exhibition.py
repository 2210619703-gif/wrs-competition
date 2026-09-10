#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/9/5 16:08
# @Author : ZhangXi

from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np
import wrs.basis.robot_math as rm


class RealmanArmController:
    def __init__(
            self,
            ip: str = "192.168.1.18",
            port: int = 8080,
            has_gripper=False,
            multiple_thread=True,
    ):
        """
        :param ip: 机器人的 IP 地址
        :param port: 机器人的端口号

        :param has_gripper: 是否安装夹爪
        """
        # 初始化 Realman 机器人控制类
        if multiple_thread:
            mode = rm_thread_mode_e.RM_TRIPLE_MODE_E
        else:
            mode = None
        self.arm = RoboticArm(mode)
        self.handle = self.arm.rm_create_robot_arm(ip, port)  # 创建机器人连接
        print(f"连接ID: {self.handle.id}")  # 打印连接 ID
        time.sleep(.1)

        self._has_gripper = has_gripper
        if has_gripper:
            # 如果安装了夹爪，初始化夹爪控制逻辑
            pass

    def move_j(self,
               joint_angles: np.ndarray or list,
               is_radians=True,
               speed: int = 5,
               r: int = 0,
               connect: int = 0,
               block: int = 1):
        """
        控制机器人运动到指定的关节角度位置
        :param joint_angles: 目标关节角度，六个角度值，单位为度
        :param speed: 运动速度，单位为度/秒
        :param accel: 加速度
        :param decel: 减速度
        """
        if is_radians:
            joint_angles = np.degrees(joint_angles)
        result = self.arm.rm_movej(joint_angles, speed, r, connect, block)
        if result == 0:
            print(f"成功移动到关节位置: {joint_angles}")
        else:
            print(f"关节运动失败: {result}")

    def move_p(self, pose: list, speed: int = 5, accel: int = 0, decel: int = 0, connect: int = 1):
        """
        控制机器人移动到指定的位姿
        :param pose: 目标位姿，包含位置和姿态，六个元素：[x, y, z, roll, pitch, yaw]
        :param speed: 运动速度
        :param accel: 加速度
        :param decel: 减速度
        :param connect: 是否进行连接，1 表示连接，0 表示不连接
        """
        result = self.arm.rm_movej_p(pose, speed, accel, decel, connect)
        if result == 0:
            print(f"成功移动到位姿: {pose}")
        else:
            print("位姿运动失败")

    def move_l(self,
               pos: np.ndarray,
               rotmat: np.ndarray,
               is_euler=False,
               speed: int = 10,
               r=0,
               connect: int = 0,
               block: int = 1):
        """
        控制机器人移动到指定的位姿
        :param pose: 目标位姿，包含位置和姿态，六个元素：[x, y, z, roll, pitch, yaw]
        :param speed: 运动速度
        :param accel: 加速度
        :param decel: 减速度
        :param connect: 是否进行连接，1 表示连接，0 表示不连接
        """
        if not is_euler:
            if rotmat.shape != (3, 3):
                raise ValueError("rotmat must be a 3x3 rotation matrix")
            euler = rm.rotmat_to_euler(rotmat)
        else:
            euler = rotmat
        pose = np.concatenate((pos, euler)).tolist()  # 将位置和姿态合并为一个列表
        result = self.arm.rm_movel(pose, speed, r, connect, block)
        if result == 0:
            # print(f"成功移动到位姿: {pose}")
            return True
        else:
            # print("位姿运动失败")
            return False

if __name__ == "__main__":
    armx = RealmanArmController(ip="192.168.3.18", port=8080, multiple_thread=True)

    # 定义 5 个位置和姿态（这里随便写了，你需要替换成实际的 pos）
    positions = [
        [0.40, 0.20, 0.30, 0, 0, 0],
        [0.45, 0.20, 0.30, 0, 0, 0],
        [0.50, 0.20, 0.30, 0, 0, 0],
        [0.50, 0.25, 0.30, 0, 0, 0],
        [0.45, 0.25, 0.30, 0, 0, 0],
    ]

    for pose in positions:
        armx.move_p(pose,speed=10)
        time.sleep(2)
