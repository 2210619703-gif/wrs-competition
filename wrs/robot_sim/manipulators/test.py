from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
print(handle.id)

# 关节阻塞运动到[0, 20, 70, 0, 90, 0]
print(arm.rm_movej([0, 20, 70, 0, 90, 0], 5, 0, 0, 1))

arm.rm_delete_robot_arm()