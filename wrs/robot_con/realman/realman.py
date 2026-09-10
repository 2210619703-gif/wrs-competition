from Robotic_Arm.rm_robot_interface import *

class RealmanArmController:
    def __init__(self, ip: str = "192.168.1.18", port: int = 8080, has_gripper=False):
        """
        :param ip: 机器人的 IP 地址
        :param port: 机器人的端口号1

        :param has_gripper: 是否安装夹爪
        """
        # 初始化 Realman 机器人控制类
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)  # 创建机器人连接
        print(f"连接ID: {self.handle.id}")  # 打印连接 ID

        self._has_gripper = has_gripper
        if has_gripper:
            # 如果安装了夹爪，初始化夹爪控制逻辑
            pass

    def move_j(self, joint_angles: list, speed: int = 5, accel: int = 0, decel: int = 0):
        """
        控制机器人运动到指定的关节角度位置
        :param joint_angles: 目标关节角度，六个角度值，单位为度
        :param speed: 运动速度，单位为度/秒
        :param accel: 加速度
        :param decel: 减速度
        """
        result = self.arm.rm_movej(joint_angles, speed, accel, decel, 1)
        if result == 0:
            print(f"成功移动到关节位置: {joint_angles}")
        else:
            print("关节运动失败")

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

    def get_joint_values(self):
        """
        获取当前机器人的关节角度值
        :return: 返回关节角度列表
        """
        joint_values = self.arm.rm_get_joint_degree()
        print(f"当前关节值: {joint_values}")
        return joint_values

    def get_pose(self):
        """
        获取当前机器人的位姿
        :return: 返回位姿信息，只包含位置和姿态
        """
        pose_data = self.arm.rm_get_current_arm_state()  # 获取原始数据
        pose = pose_data[1].get('pose', [0, 0, 0, 0, 0, 0])  # 从元组中提取字典，然后获取 'pose'

        print(f"当前位姿: {pose}")
        return (0, {'pose': pose})  # 只返回 'pose'

    def close_connection(self):
        """
        删除机器人连接，释放资源
        """
        self.arm.rm_delete_robot_arm()
        print("机器人连接已删除")




if __name__ == "__main__":
    # 创建 RealMan 机器人控制器实例
    realman_controller = RealmanArmController(ip="192.168.1.18", port=8080, has_gripper=False)

    try:
        while True:
            print("\n=== 机器人控制选项 ===")
            print("1. 移动到关节位置")
            print("2. 按位姿运动")
            print("3. 获取关节角度")
            print("4. 获取当前位姿")
            print("5. 退出程序")
            choice = input("请选择操作(1-5): ")

            if choice == "1":
                joint_angles = input("输入关节角度(以逗号分隔，例如: 0,10,80,0,90,0): ")
                joint_angles = [float(x) for x in joint_angles.split(",")]
                realman_controller.move_j(joint_angles)

            elif choice == "2":
                pose = input("输入目标位姿(以逗号分隔，例如: 0.3,0,0.3,3.14,0,0): ")
                pose = [float(x) for x in pose.split(",")]
                realman_controller.move_p(pose)

            elif choice == "3":
                realman_controller.get_joint_values()

            elif choice == "4":
                realman_controller.get_pose()

            elif choice == "5":
                print("退出程序")
                break

            else:
                print("无效的选择，请重试！")

    except KeyboardInterrupt:
        print("\n程序中断")

    finally:
        # 关闭连接
        realman_controller.close_connection()
