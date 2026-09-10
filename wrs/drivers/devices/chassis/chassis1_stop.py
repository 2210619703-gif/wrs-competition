#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/6/30 10:59
# @Author : ZhangXi
import can
import struct
import time
import subprocess
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ChassisControl:
    """
    EAM控制信号数据结构
    根据协议：0x115 (ID 277)，周期20ms，数据长度8字节
    """
    water_pump: bool = False  # 清水泵开启 (1 bit)
    remote_enable: bool = False  # 遥控使能 (1 bit)
    gear_position: int = 0  # 目标挡位 (2 bits): 0=无效, 1=R, 2=N, 3=D
    target_speed: float = 0.0  # 目标速度 (16 bits, 0.1精度)
    target_angle: int = 0  # 目标转向角 (16 bits, 有符号)

    # 数值范围验证
    def validate(self):
        if not 0 <= self.gear_position <= 3:
            raise ValueError("挡位值需在0-3范围内")
        # 注意：协议文档中的目标速度单位是0.1A，但这里似乎应为0.1km/h或类似单位，暂时按数值范围验证
        if not 0.0 <= self.target_speed <= 100.0:
            raise ValueError("目标速度需在0-100范围内")
        # Corrected range based on image: -1024 to 1024
        if not -1024 <= self.target_angle <= 1024:
            raise ValueError("转向角需在-1024~1024范围内")


class ChassisController:
    # CAN协议常量
    ARBITRATION_ID = 0x115  # CAN消息ID
    DATA_LENGTH = 8  # 数据长度

    def __init__(self, bus: can.BusABC):
        self.bus = bus

    def encode(self, control: ChassisControl) -> bytes:
        """
        将控制信号编码为CAN数据帧
        协议结构:
        [0]: WaterPump(bit0) | RemoteEnable(bit1) | GearPos(bit4-5)
        [5-6]: 目标速度 (16 bits, LSB)
        [6-7]: 目标转向角 (16 bits, 有符号, LSB) - Adjusted to fit 8-byte frame
        """
        # 验证数据范围
        control.validate()

        # 创建8字节缓冲区
        data = bytearray(self.DATA_LENGTH)

        # Byte 0: Combination of bit-field signals
        byte0 = 0
        if control.water_pump:
            byte0 |= 0x01  # bit0
        if control.remote_enable:
            byte0 |= 0x02  # bit1
        byte0 |= (control.gear_position & 0x03) << 4  # bit4-5
        data[0] = byte0

        # Target speed conversion (unit: 0.1/bit)
        speed_int = int(control.target_speed * 10)
        # According to image: Start Byte 5, Bit Length 16, Motorola LSB (Little-Endian)
        data[4:6] = speed_int.to_bytes(2, 'big', signed=False)

        # Target steering angle (unit: 1/bit, signed)
        # According to image: Start Byte 7, Bit Length 16. This requires 9 bytes.
        # Assuming for an 8-byte frame, it should start at Byte 6 and span 16 bits (bytes 6 and 7).
        # 将 -1024 ~ +1024 映射到 0 ~ 2048
        angle_encoded = control.target_angle + 1024
        data[6:8] = angle_encoded.to_bytes(2, 'big', signed=False)

        return bytes(data)

    def decode(self, data: bytes) -> ChassisControl:
        """
        从CAN数据帧解析控制信号
        数据长度需为8字节
        """
        if len(data) < self.DATA_LENGTH:
            raise ValueError("数据长度不足8字节")

        control = ChassisControl()

        # Parse Byte 0 (bit-level operations)
        control.water_pump = bool(data[0] & 0x01)
        control.remote_enable = bool(data[0] & 0x02)
        control.gear_position = (data[0] >> 4) & 0x03

        # Parse target speed (Byte 5-6)
        speed_int = int.from_bytes(data[4:6], 'big', signed=False)
        control.target_speed = round(speed_int * 0.1, 1)

        # Parse steering angle (Byte 6-7) - Adjusted to fit 8-byte frame
        angle_encoded = int.from_bytes(data[6:8], 'big', signed=False)
        control.target_angle = angle_encoded - 1024

    def send(self, control: ChassisControl):
        """发送控制信号到CAN总线，增加发送日志"""
        data = self.encode(control)
        msg = can.Message(
            arbitration_id=self.ARBITRATION_ID,
            data=data,
            is_extended_id=False
        )
        try:
            self.bus.send(msg)
            print(f"[发送成功] ID=0x{msg.arbitration_id:X} 数据={msg.data.hex()}")
        except can.CanError as e:
            print(f"[发送失败] ID=0x{msg.arbitration_id:X} 错误: {e}")

    def receive(self) -> Tuple[can.Message, ChassisControl]:
        """接收并解析CAN消息"""
        msg = self.bus.recv()
        if msg and msg.arbitration_id == self.ARBITRATION_ID:
            return msg, self.decode(msg.data)
        return None, None

def configure_can(channel, bitrate, sample_point=0.808):
    """
    配置 CAN 接口，设置 Bitrate 和 Sample Point
    """
    print(f"正在配置 {channel}...")
    try:
        # 配置 IP link 和 bitrate 设置
        subprocess.run(f"sudo ip link set {channel} down", shell=True, check=True)
        subprocess.run(
            f"sudo ip link set {channel} type can bitrate {bitrate} sample-point {sample_point}",
            shell=True, check=True)
        subprocess.run(f"sudo ip link set {channel} up", shell=True, check=True)
        print(f"{channel} 配置成功，Bitrate 设置为 {bitrate} bps，Sample Point 设置为 {sample_point}.")
2000000000070400    except subprocess.CalledProcessError as e:
        print(f"配置 {channel} 时出错: {e}")
        raise
    except FileNotFoundError:
        print("错误: 'sudo' 或 'ip' 命令未找到。请确保在Linux环境并已安装iproute2。")
        raise



if __name__ == "__main__":
    CAN_CHANNEL = 'can1'
    CAN_BITRATE = 500000
    SAMPLE_POINT = 0.810

    try:
        configure_can(CAN_CHANNEL, CAN_BITRATE, SAMPLE_POINT)
        print("开始发送控制信号...")
        with can.Bus(interface='socketcan', channel=CAN_CHANNEL, bitrate=CAN_BITRATE, fd=False) as bus:
            controller = ChassisController(bus)

            ctrl_signal = ChassisControl(
                water_pump=False,
                remote_enable=False,
                gear_position=0,   # D档
                target_speed=0,
                target_angle=0
            )
            
            controller.send(ctrl_signal)
            print(f"[单次发送] 速度={ctrl_signal.target_speed}, 转向={ctrl_signal.target_angle}")
        # 5. 退出with块后，CAN连接会自动关闭
        print("指令已发送，CAN连接已断开")
    except KeyboardInterrupt:
        print("\n已手动中断发送。")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
