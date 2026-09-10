#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : ZhangXi（修改：ChatGPT）
import can
import struct
import time
import subprocess
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ChassisControl:
    """
    EAM控制信号数据结构（新协议）
    """
    water_pump: bool = False
    remote_enable: bool = False
    target_gear: int = 0
    chassis_mode: int = 0  # 0:无效, 1:直行, 2:横移, 3:自旋
    target_speed: float = 0.0  # 单位：0.1m/s
    target_angle: int = 0      # 有符号 -1024~1024

    def validate(self):
        if not 0 <= self.chassis_mode <= 8:
            raise ValueError("chassis_mode 超出范围")
        if not 0.0 <= self.target_speed <= 100.0:
            raise ValueError("目标速度需在0~100")
        if not -1024 <= self.target_angle <= 1024:
            raise ValueError("角度超出范围")


class ChassisController:
    ARBITRATION_ID = 0x115
    DATA_LENGTH = 8

    def __init__(self, bus: can.BusABC):
        self.bus = bus

    def encode(self, control: ChassisControl) -> bytes:
        control.validate()
        data = bytearray(self.DATA_LENGTH)

        # Byte 0: 控制标志 + 挡位（bit0-bit1: 控制标志；bit4-bit5: 挡位）
        byte0 = 0
        if control.water_pump:
            byte0 |= 0x01  # bit0
        if control.remote_enable:
            byte0 |= 0x02  # bit1
        byte0 |= (control.target_gear & 0x03) << 4  # bit4~5
        data[0] = byte0

        # Byte 1 和 Byte 2 保留 0
        data[1] = 0x00
        data[2] = 0x00

        # Byte 3~4: 目标速度，单位0.1，小端序（低字节 Byte4，高字节 Byte3）
        speed_int = int(control.target_speed * 10)
        data[4] = speed_int & 0xFF  # 低字节
        data[3] = (speed_int >> 8) & 0xFF  # 高字节

        # Byte 5: 底盘控制模式
        data[5] = control.chassis_mode & 0xFF

        # Byte 6~7: 目标转向角度（有符号 + 1024 偏移），小端序
        angle_encoded = control.target_angle + 1024  # 0~2048
        data[6] = angle_encoded & 0xFF  # 低字节
        data[7] = (angle_encoded >> 8) & 0xFF  # 高字节

        return bytes(data)

    def send(self, control: ChassisControl):
        data = self.encode(control)
        msg = can.Message(arbitration_id=self.ARBITRATION_ID, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
            print(f"[发送成功] ID=0x{msg.arbitration_id:X} 数据={msg.data.hex()}")
        except can.CanError as e:
            print(f"[发送失败] 错误: {e}")


def configure_can(channel, bitrate, sample_point=0.808):
    """
    配置 CAN 接口
    """
    print(f"配置 CAN 接口: {channel}...")
    try:
        subprocess.run(f"sudo ip link set {channel} down", shell=True, check=True)
        subprocess.run(
            f"sudo ip link set {channel} type can bitrate {bitrate} sample-point {sample_point}",
            shell=True, check=True)
        subprocess.run(f"sudo ip link set {channel} up", shell=True, check=True)
        print(f"{channel} 配置成功，Bitrate={bitrate} bps，SamplePoint={sample_point}")
    except subprocess.CalledProcessError as e:
        print(f"配置失败: {e}")
        raise
    except FileNotFoundError:
        print("找不到 'sudo' 或 'ip' 命令，请确认运行环境为 Linux 并已安装 iproute2 工具。")
        raise


if __name__ == "__main__":
    CAN_CHANNEL = 'can1'
    CAN_BITRATE = 500000
    SAMPLE_POINT = 0.810

    # ✅ 用户配置部分（只改下面这两个值就行）：
    mode = 3             # 1=直行, 2=横移, 3=自旋
    target_value = 45  # 单位：米 or 度，取决于 mode
    target_speed = 2  # 单位：A，对应协议值=100，≈ 0.1 m/s or 0.1 deg/s
    # =========================================

    try:
        configure_can(CAN_CHANNEL, CAN_BITRATE, SAMPLE_POINT)

        with can.Bus(interface='socketcan', channel=CAN_CHANNEL, bitrate=CAN_BITRATE) as bus:
            controller = ChassisController(bus)

            # ✅ 将 target_speed 转换为协议值（乘10）
            protocol_speed_value = int(target_speed * 10)

            # ✅ 计算运动所需时间（单位：秒）
            if mode in [1, 2]:
                duration = target_value / (target_speed * 0.1)  # 米 / (米/秒)
            elif mode == 3:
                duration = target_value / (target_speed * 0.1)  # 角度 / (角度/秒)
            else:
                raise ValueError("模式错误，仅支持 1=直行，2=横移，3=自旋")

            print(f"[准备执行] 模式={mode}, 目标值={target_value}, 电流={target_speed}A, 时长={duration:.2f}s")

            # ✅ 启控阶段
            ctrl_on = ChassisControl(
                water_pump=False,
                remote_enable=False,
                target_gear=1,
                chassis_mode=mode,
                target_speed=target_speed,
                target_angle=target_value
            )
            controller.send(ctrl_on)
            print("[启控] 报文已发送")
            time.sleep(duration)  # 控制持续时间

            # ✅ 自动失能
            ctrl_off = ChassisControl(
                water_pump=False,
                remote_enable=False,
                target_gear=0,
                chassis_mode=0,
                target_speed=0,
                target_angle=0
            )
            controller.send(ctrl_off)
            print("[失能] 控制结束")

            # ✅ 打印动作结果
            if mode in [1, 2]:
                print(f"[结果] 预计直线移动 {target_value:.2f} 米")
            else:
                print(f"[结果] 预计旋转 {target_value:.2f} 度")

    except KeyboardInterrupt:
        print("用户终止程序")
    except Exception as e:
        print(f"运行出错: {e}")

