'''
Author: wang yining
Date: 2025-08-01 14:36:22
LastEditTime: 2025-08-01 15:17:37
FilePath: /remote_fs/wrs/drivers/devices/chassis/chassis_control.py
Description: 通过直行,自旋,横移三种方式控制底盘
e-mail: wangyining0408@outlook.com
'''
from wrs.drivers.devices.chassis.chassis1 import control_chassis


class EasyChassisController:
    def go_straight(self, distance, advance=True, speed=0.2):
        '''
        description: 直行模式,默认前进advance为True,反之为后退,distance单位为m,speed单位为m/s,默认0.1m/s
        return {*}
        '''
        if advance:
            gear_position = 3  # 前进
        else:
            gear_position = 1  # 倒退

        # 计算run_time
        # 默认速度为10,假设为0.1m/s
        run_time = distance / speed

        control_chassis(
            run_time=run_time,
            water_pump=False,
            remote_enable=True,
            gear_position=gear_position,
            target_speed=speed * 100,  # 这里假设 10对应0.1m/s, 100对应1m/s
            chassis_mode=1,
            target_angle=0
        )

        # 发送停止信号
        control_chassis(
            run_time=1,
            water_pump=False,
            remote_enable=False,
            gear_position=2,
            target_speed=0,
            chassis_mode=7,
            target_angle=0
        )

    def move_horizontally(self, distance, go_left=True, speed=0.2):
        '''
        description: 横移模式,默认前进go_left为True,反之为向右,distance单位为m,speed单位为m/s,默认0.1m/s
        return {*}
        '''
        if go_left:
            gear_position = 3  # 向左
        else:
            gear_position = 1  # 向右

        # 计算run_time
        # 默认速度为10,假设为0.1m/s
        run_time = distance / speed

        control_chassis(
            run_time=run_time,
            water_pump=False,
            remote_enable=True,
            gear_position=gear_position,
            target_speed=speed * 100,  # 这里假设 10对应0.1m/s, 100对应1m/s
            chassis_mode=6,
            target_angle=0
        )

        # 发送停止信号
        control_chassis(
            run_time=1,
            water_pump=False,
            remote_enable=False,
            gear_position=2,
            target_speed=0,
            chassis_mode=7,
            target_angle=0
        )

    def spin(self, angle, anticlockwise=True, speed=40):
        '''
        description: 自旋,不确定speed的作用
        return {*}
        '''

        if anticlockwise:
            gear_position = 3  # 逆时针
        else:
            gear_position = 1  # 顺时针
        # 假设speed 30表示30度/s
        run_time = angle / speed
        # run_time = 3
        start_time = time.time()
        end_time = time.time()
        interval = end_time - start_time
        while interval < run_time:
            control_chassis(
                run_time=1,
                water_pump=False,
                remote_enable=True,
                gear_position=gear_position,
                target_speed=speed,
                chassis_mode=4,
                target_angle=angle
            )
            end_time = time.time()
            interval = end_time - start_time

        # 发送停止信号
        control_chassis(
            run_time=1,
            water_pump=False,
            remote_enable=False,
            gear_position=2,
            target_speed=0,
            chassis_mode=7,
            target_angle=0
        )
    def open_water_pump(self,is_open = True):
        water_pump = is_open
        control_chassis(
            run_time=1,
            water_pump=water_pump,
            remote_enable=True,
            gear_position=2,
            target_speed=0,
            chassis_mode=7,
            target_angle=0
        )

if __name__ == "__main__":
    import time

    # 创建底盘控制器实例
    controller = EasyChassisController()
    start_time = time.time()

    # # while True:
    # print("=== 开始底盘控制测试 ===")
    # # print(time.time)
    # # 测试1: 前进3米（使用默认速度）
    # # print("\n测试1: 前进3米")
    # # controller.go_straight(distance=1, advance=True,speed=0.2)
    # end_time = time.time()
    # interval = end_time - start_time
    # print(f'时间间隔{interval:.6f}')
    # time.sleep(3)
    # # 测试2: 后退2米（指定速度0.2m/s）
    # # print("\n测试2: 后退2米")
    # # controller.go_straight(distance=8, advance=True, speed=0.2)
    # # time.sleep(3)
    # # print(time.time)
    # # time.sleep(3)
    # # 测试3: 向左横移1.5米
    # print("\n测试3: 向左横移1.5米")
    # # controller.move_horizontally(distance=1.5, go_left=True)
    # # time.sleep(3)
    # # 测试4: 向右横移1米（指定速度0.15m/s）
    # print("\n测试4: 向右横移1米")
    # # controller.move_horizontally(distance=1.0, go_left=False, speed=0.15)
    # # time.sleep(3)
    # # 测试5: 顺时针旋转90度
    # print("\n测试5: 顺时针旋转90度")
    # while True:
    #     time.sleep(3)
    #     controller.spin(angle=90, anticlockwise=False)
    #     time.sleep(3)
    #     controller.spin(angle=90, anticlockwise=True)
    # # time.sleep(3)
    # # 测试6: 逆时针旋转45度
    # print("\n测试6: 逆时针旋转45度")
    # # controller.spin(angle=90, anticlockwise=False)
    # # time.sleep(3)
    # print("\n=== 所有测试完成 ===")
