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
    chassis_mode: int = 0 #目标底盘模式 0=无效 1=直行 4=自旋 6=横移
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
        if not 0 <= self.chassis_mode <= 8:
            raise ValueError("底盘模式需在0-8范围内")


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
        data[3:5] = speed_int.to_bytes(2, 'big', signed=False)

        data[5] = control.chassis_mode & 0xFF
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
    except subprocess.CalledProcessError as e:
        print(f"配置 {channel} 时出错: {e}")
        raise
    except FileNotFoundError:
        print("错误: 'sudo' 或 'ip' 命令未找到。请确保在Linux环境并已安装iproute2。")
        raise


import time
import can
from typing import Union

def control_chassis(run_time: float, water_pump: bool, remote_enable: bool, 
                   gear_position: int, target_speed: float, 
                   chassis_mode: int, target_angle: float) -> None:
    """
    控制底盘运动
    
    参数:
        run_time: 运行时间(秒)，必须为正数
        water_pump: 水泵开关，布尔值
        remote_enable: 远程使能，布尔值
        gear_position: 档位位置,0-3整数,档位 -> 0:无效 1:R档 2:N档 3:D档
        target_speed: 目标速度(m/s),0-10范围内
        chassis_mode: 底盘模式,0-7整数,底盘模式 -> 0:无效 1:直行 2:前驱阿克曼 3:斜移 4:自旋 5:小转弯 6:横移 7:驻车 8:超出范围
        target_angle: 目标角度(度),-180到180范围内
    """
    # 参数验证
    if not isinstance(run_time, (int, float)) or run_time <= 0:
        raise ValueError("run_time 必须是正数")
    if not isinstance(water_pump, bool):
        raise TypeError("water_pump 必须是布尔值")
    if not isinstance(remote_enable, bool):
        raise TypeError("remote_enable 必须是布尔值")
    if not isinstance(gear_position, int) or gear_position not in {0, 1, 2, 3}:
        raise ValueError("gear_position 必须是0-3的整数")
    if not isinstance(target_speed, (int, float)) or not 0 <= target_speed <= 10:
        raise ValueError("target_speed 必须是0-10范围内的数字")
    if not isinstance(chassis_mode, int) or not 0 <= chassis_mode <= 7:
        raise ValueError("chassis_mode 必须是0-7的整数")
    if not isinstance(target_angle, (int, float)) or not -180 <= target_angle <= 180:
        raise ValueError("target_angle 必须是-180到180范围内的数字")

    CAN_CHANNEL = 'can1'
    CAN_BITRATE = 500000
    SAMPLE_POINT = 0.810

    try:
        configure_can(CAN_CHANNEL, CAN_BITRATE, SAMPLE_POINT)
        print("开始发送控制信号...")
        with can.Bus(interface='socketcan', channel=CAN_CHANNEL, bitrate=CAN_BITRATE, fd=False) as bus:
            controller = ChassisController(bus)

            ctrl_signal = ChassisControl(
                water_pump=water_pump,
                remote_enable=remote_enable,
                gear_position=gear_position,   
                target_speed=target_speed,
                chassis_mode=chassis_mode,
                target_angle=target_angle
            )

            controller.send(ctrl_signal)

            time.sleep(run_time)

            # 发送停止信号
            ctrl_signal = ChassisControl(
                water_pump=False,
                remote_enable=False,
                gear_position=2,   
                target_speed=0,
                chassis_mode=7,
                target_angle=0  
            )

            controller.send(ctrl_signal)
            
    except KeyboardInterrupt:
        print("\n已手动中断发送。")
        # 中断时也发送停止信号
        ctrl_signal = ChassisControl(
            water_pump=False,
            remote_enable=False,
            gear_position=2,   
            target_speed=0,
            chassis_mode=7,
            target_angle=0
        )
        controller.send(ctrl_signal)
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        raise

if __name__ == "__main__":
    pass
    #以速度5直行1秒
    #control_chassis(run_time=1,water_pump=False,remote_enable=True,gear_position=3,target_speed=5,chassis_mode=1,target_angle=0)
    #停止
    control_chassis(run_time=0.1,water_pump=False,remote_enable=False,gear_position=2,target_speed=0,chassis_mode=7,target_angle=0)

