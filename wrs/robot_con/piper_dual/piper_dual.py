#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/23 00:23
# @Author : ZhangXi
import platform
import threading
import time
import numpy as np



try:
    from wrs.robot_con.piper.piper import PiperArmController
except ImportError:
    # 如果您没有将文件命名为 piper_arm_controller.py，我们假设您知道如何导入它
    print("Error: Could not import PiperArmController. Please ensure the class is importable.")
    exit()


def get_can_names():
    """根据操作系统返回左臂和右臂的 CAN 接口名称。"""
    system = platform.system()
    if system == "Linux":
        # 假设左臂是 can0，右臂是 can1
        return "can0", "can1"
    elif system == "Windows":
        # 假设左臂是 "0"，右臂是 "1"
        return "0", "1"
    else:
        # 其他系统，默认为 Linux 设置，可能需要根据实际情况调整
        print(f"Warning: Unsupported system '{system}'. Using 'can0' and 'can1'.")
        return "can0", "can1"


class DualPiperArmController:
    """AgileX Piper 双臂控制器的高级封装。"""

    def __init__(self, left_has_gripper=True, right_has_gripper=True):
        self.left_arm_name, self.right_arm_name = get_can_names()

        self.left_arm: PiperArmController = None
        self.right_arm: PiperArmController = None

        print(f"Initializing Left Arm on: {self.left_arm_name}")
        print(f"Initializing Right Arm on: {self.right_arm_name}")

        # 使用多线程同时初始化两个臂，避免一个臂的连接/配置时间阻塞另一个
        left_thread = threading.Thread(
            target=self._init_arm,
            args=('left', self.left_arm_name, left_has_gripper)
        )
        right_thread = threading.Thread(
            target=self._init_arm,
            args=('right', self.right_arm_name, right_has_gripper)
        )

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()

        if self.left_arm is None or self.right_arm is None:
            print("Error: One or both arms failed to initialize.")
            self.close_all()
            raise RuntimeError("Failed to initialize dual arm system.")

        print("\nBoth arms initialized successfully.")

    def _init_arm(self, arm_id, can_name, has_gripper):
        """线程内辅助函数，用于初始化单个机械臂。"""
        try:
            arm = PiperArmController(can_name=can_name, has_gripper=has_gripper, auto_enable=False)
            arm.enable()  # 手动调用 enable，确保初始化过程中的日志清晰

            if arm_id == 'left':
                self.left_arm = arm
                print(f"Left Arm ({can_name}) enabled.")
            else:
                self.right_arm = arm
                print(f"Right Arm ({can_name}) enabled.")

        except Exception as e:
            print(f"Failed to initialize {arm_id} arm on {can_name}: {e}")

    def go_home_both(self):
        """同时将两个机械臂移动到零位 (home position)。"""
        print("\nMoving both arms to home position...")

        target_joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        speed = 10  # 较慢速度以确保安全

        # 使用线程同时发送 move_j 命令
        left_thread = threading.Thread(
            target=self.left_arm.move_j,
            args=(target_joint_angles,),
            kwargs={'speed': speed, 'block': True}
        )
        right_thread = threading.Thread(
            target=self.right_arm.move_j,
            args=(target_joint_angles,),
            kwargs={'speed': speed, 'block': True}
        )

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()
        print("Both arms reached home position.")

    def close_all(self):
        """关闭所有机械臂的连接并禁用电机。"""
        print("\nShutting down both arms...")

        def safe_disable_close(arm, name):
            if arm:
                try:
                    arm.disable()
                    arm.close_connection()
                    print(f"{name} arm disabled and connection closed.")
                except Exception as e:
                    print(f"Error closing {name} arm: {e}")

        left_thread = threading.Thread(target=safe_disable_close, args=(self.left_arm, 'Left'))
        right_thread = threading.Thread(target=safe_disable_close, args=(self.right_arm, 'Right'))

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()
        print("Dual arm system shut down complete.")

    # 您可以添加更多双臂协调动作的方法...


# --- 主程序示例 ---
if __name__ == "__main__":

    dual_arm = None
    try:
        # 1. 初始化双臂控制器
        dual_arm = DualPiperArmController(left_has_gripper=True, right_has_gripper=True)

        # 2. 协调动作示例：两臂同时回到零位
        dual_arm.go_home_both()
        time.sleep(1)

        # 3. 独立动作示例：左臂执行一个抓取动作，右臂移动到另一个位置
        print("\nStarting independent actions...")


        # 左臂：张开 -> 闭合
        def left_arm_gripper_action():
            print("Left arm gripper opening...")
            dual_arm.left_arm.open_gripper(width=0.08)
            time.sleep(0.5)
            print("Left arm gripper closing...")
            dual_arm.left_arm.close_gripper()


        # 右臂：移动到某个目标关节位置
        def right_arm_move_action():
            target_jnt = [0.1, -0.6, 0.7, 0.2, 0.1, 0.0]
            print(f"Right arm moving to: {np.degrees(target_jnt)}")
            dual_arm.right_arm.move_j(target_jnt, speed=20, block=True)
            print("Right arm move complete.")


        left_action_thread = threading.Thread(target=left_arm_gripper_action)
        right_action_thread = threading.Thread(target=right_arm_move_action)

        left_action_thread.start()
        right_action_thread.start()

        left_action_thread.join()
        right_action_thread.join()
        print("Independent actions complete.")

        time.sleep(2)

    except RuntimeError as e:
        print(f"Fatal error during execution: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        # 4. 最终关闭
        if dual_arm:
            dual_arm.go_home_both()
            time.sleep(1)
            dual_arm.close_all()