"""
Created on 2025/3/28 
Author: Hao Chen (chen960216@gmail.com)
"""
import can
import struct
import time
import subprocess

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import time


class OmniPickerCANController:
    def __init__(self, channel='can0', bitrate=5000000, can_node_id=0x01):
        """
        初始化 OmniPicker CANFD 控制器
        """
        self.channel = channel
        self.bitrate = bitrate
        self.can_node_id = can_node_id  # Gripper CAN node ID
        self.configure_can0()
        self.bus = self.init_canfd_interface()

    def init_canfd_interface(self):
        """
        初始化 CANFD 设备
        """
        try:
            bus = can.interface.Bus(bustype='socketcan', channel=self.channel, fd=True, bitrate=self.bitrate)
            print("CANFD 设备初始化成功。")
            return bus
        except Exception as e:
            print("CANFD 设备初始化失败：", e)
            return None

    def configure_can0(self):
        """
        配置 CAN0 接口和电阻，设置 Bitrate
        """
        try:
            # 切换到 root 用户并配置 CAN0 电阻
            # subprocess.run("echo 0 > /sys/class/tz_gpio/can_ohms_0/value", shell=True, check=True)
            # print("CAN0 终端电阻去除。")
            print("请以管理员权限手动去除 CAN0 终端电阻：echo 0 > /sys/class/tz_gpio/can_ohms_0/value")

            # 配置 IP link 和 bitrate 设置
            subprocess.run("sudo ip link set can0 down", shell=True, check=True)  # 先断开连接
            subprocess.run(
                f"sudo ip link set can0 type can bitrate 1000000 sample-point 0.80 dbitrate {self.bitrate} dsample-point 0.8 fd on",
                shell=True, check=True)  # berr-reporting on restart-ms 100
            subprocess.run("sudo ip link set can0 up", shell=True, check=True)  # 重新启动 can0
            print("CAN0 配置成功，Bitrate 设置为 1Mbps，数据域 5Mbps。")
        except subprocess.CalledProcessError as e:
            print(f"配置 CAN0 时出错: {e}")

    def send_can_message(self, can_id, data):
        """
        发送 CANFD 消息
        """
        msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
            print(f"发送 CAN 消息: ID={hex(can_id)}, 数据={data}")
        except can.CanError:
            print("CAN 消息发送失败。")

    def set_gripper_command(self, pos_cmd, force_cmd, vel_cmd, acc_cmd, dec_cmd):
        """
        设置夹爪控制指令
        """
        # 设置每个命令的值在 0 - 255 范围内
        pos_cmd = max(0, min(255, pos_cmd))  # 目标位置，0为夹紧，255为完全张开
        force_cmd = max(0, min(255, force_cmd))  # 目标力矩
        vel_cmd = max(0, min(255, vel_cmd))  # 目标速度
        acc_cmd = max(0, min(255, acc_cmd))  # 目标加速度
        dec_cmd = max(0, min(255, dec_cmd))  # 目标减速度

        # 发送控制命令帧
        data = [0x00, pos_cmd, force_cmd, vel_cmd, acc_cmd, dec_cmd, 0x00, 0x00]
        self.send_can_message(self.can_node_id, data)

    def get_gripper_status(self, timeout=5.0):
        """
        监听 CAN 消息，获取夹爪状态
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.bus.recv(timeout=timeout)
            if msg and msg.arbitration_id == self.can_node_id:
                # 状态数据：故障代码、状态、位置、速度、力矩
                fault_code = msg.data[0]
                state = msg.data[1]
                pos = msg.data[2]
                vel = msg.data[3]
                force = msg.data[4]
                print(f"Fault Code: {fault_code}, State: {state}, Pos: {pos}, Vel: {vel}, Force: {force}")
                return {
                    "fault_code": fault_code,
                    "state": state,
                    "pos": pos,
                    "vel": vel,
                    "force": force
                }
        print("超时未收到状态数据")
        return None

    def start_ui(self):
        """
        启动 GUI 界面。
        """
        root = Tk()
        root.title("OmniPicker 控制界面")

        # Add control elements
        Label(root, text="Position (0-255)").grid(row=0, column=0)
        position_scroll = Scrollbar(root, orient="horizontal")
        position_scroll.grid(row=0, column=1)
        position_scroll.set(0, 255)

        Label(root, text="Force (0-255)").grid(row=1, column=0)
        force_scroll = Scrollbar(root, orient="horizontal")
        force_scroll.grid(row=1, column=1)
        force_scroll.set(0, 255)

        Label(root, text="Velocity (0-255)").grid(row=2, column=0)
        velocity_scroll = Scrollbar(root, orient="horizontal")
        velocity_scroll.grid(row=2, column=1)
        velocity_scroll.set(0, 255)

        Label(root, text="Acceleration (0-255)").grid(row=3, column=0)
        acceleration_scroll = Scrollbar(root, orient="horizontal")
        acceleration_scroll.grid(row=3, column=1)
        acceleration_scroll.set(0, 255)

        Label(root, text="Deceleration (0-255)").grid(row=4, column=0)
        deceleration_scroll = Scrollbar(root, orient="horizontal")
        deceleration_scroll.grid(row=4, column=1)
        deceleration_scroll.set(0, 255)

        # Message display area
        self.message_display = Text(root, height=10, width=50)
        self.message_display.grid(row=5, column=0, columnspan=2)

        status_label = StringVar()
        Label(root, textvariable=status_label).grid(row=6, column=0, columnspan=2)

        def send_command():
            # Get scrollbar values and send the command
            pos = int(position_scroll.get())
            force = int(force_scroll.get())
            vel = int(velocity_scroll.get())
            acc = int(acceleration_scroll.get())
            dec = int(deceleration_scroll.get())
            self.set_gripper_command(pos, force, vel, acc, dec)
            status_label.set("命令已发送，等待确认...")

        send_button = Button(root, text="发送命令", command=send_command)
        send_button.grid(row=7, column=0, columnspan=2)

        root.mainloop()

    def main(self):
        if not self.bus:
            return
        self.start_ui()


class OmniPickerUI(tk.Tk):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller
        self.title("OmniPicker Motor Control")
        self.geometry("500x500")
        self.config(bg="#f0f0f0")

        # Flag to ensure the first two commands are for initialization only
        self.initialized = False

        # Add a beautiful header
        self.header_label = tk.Label(self, text="OmniPicker Control", font=("Arial", 20, "bold"), bg="#f0f0f0")
        self.header_label.pack(pady=10)

        # Control Panel Frame
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(pady=20)

        # Position Control
        self.pos_label = tk.Label(self.control_frame, text="Position (0-255):", font=("Arial", 12))
        self.pos_label.grid(row=0, column=0, padx=10, pady=5)
        self.pos_value_label = tk.Label(self.control_frame, text="0", font=("Arial", 12))
        self.pos_value_label.grid(row=0, column=2, padx=10)
        self.pos_slider = ttk.Scale(self.control_frame, from_=0, to_=255, orient="horizontal",
                                    command=self.update_position)
        self.pos_slider.set(0)
        self.pos_slider.grid(row=0, column=1)

        # Force Control
        self.force_label = tk.Label(self.control_frame, text="Force (0-255):", font=("Arial", 12))
        self.force_label.grid(row=1, column=0, padx=10, pady=5)
        self.force_value_label = tk.Label(self.control_frame, text="0", font=("Arial", 12))
        self.force_value_label.grid(row=1, column=2, padx=10)
        self.force_slider = ttk.Scale(self.control_frame, from_=0, to_=255, orient="horizontal",
                                      command=self.update_force)
        self.force_slider.set(0)
        self.force_slider.grid(row=1, column=1)

        # Velocity Control
        self.vel_label = tk.Label(self.control_frame, text="Velocity (0-255):", font=("Arial", 12))
        self.vel_label.grid(row=2, column=0, padx=10, pady=5)
        self.vel_value_label = tk.Label(self.control_frame, text="0", font=("Arial", 12))
        self.vel_value_label.grid(row=2, column=2, padx=10)
        self.vel_slider = ttk.Scale(self.control_frame, from_=0, to_=255, orient="horizontal",
                                    command=self.update_velocity)
        self.vel_slider.set(0)
        self.vel_slider.grid(row=2, column=1)

        # Acceleration Control
        self.acc_label = tk.Label(self.control_frame, text="Acceleration (0-255):", font=("Arial", 12))
        self.acc_label.grid(row=3, column=0, padx=10, pady=5)
        self.acc_value_label = tk.Label(self.control_frame, text="0", font=("Arial", 12))
        self.acc_value_label.grid(row=3, column=2, padx=10)
        self.acc_slider = ttk.Scale(self.control_frame, from_=0, to_=255, orient="horizontal",
                                    command=self.update_acceleration)
        self.acc_slider.set(0)
        self.acc_slider.grid(row=3, column=1)

        # Deceleration Control
        self.dec_label = tk.Label(self.control_frame, text="Deceleration (0-255):", font=("Arial", 12))
        self.dec_label.grid(row=4, column=0, padx=10, pady=5)
        self.dec_value_label = tk.Label(self.control_frame, text="0", font=("Arial", 12))
        self.dec_value_label.grid(row=4, column=2, padx=10)
        self.dec_slider = ttk.Scale(self.control_frame, from_=0, to_=255, orient="horizontal",
                                    command=self.update_deceleration)
        self.dec_slider.set(0)
        self.dec_slider.grid(row=4, column=1)

        # Control Button
        self.control_button = ttk.Button(self, text="Send Command", command=self.send_command)
        self.control_button.pack(pady=10)

        # Status Labels
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(pady=10)

        self.fault_code_label = tk.Label(self.status_frame, text="Fault Code: ", font=("Arial", 12))
        self.fault_code_label.grid(row=0, column=0, padx=10, pady=5)

        self.state_label = tk.Label(self.status_frame, text="State: ", font=("Arial", 12))
        self.state_label.grid(row=1, column=0, padx=10, pady=5)

        self.pos_status_label = tk.Label(self.status_frame, text="Position: ", font=("Arial", 12))
        self.pos_status_label.grid(row=2, column=0, padx=10, pady=5)

        self.vel_status_label = tk.Label(self.status_frame, text="Velocity: ", font=("Arial", 12))
        self.vel_status_label.grid(row=3, column=0, padx=10, pady=5)

        self.force_status_label = tk.Label(self.status_frame, text="Force: ", font=("Arial", 12))
        self.force_status_label.grid(row=4, column=0, padx=10, pady=5)

    def update_position(self, value):
        self.pos_cmd = int(float(value))
        self.pos_value_label.config(text=str(self.pos_cmd))

    def update_force(self, value):
        self.force_cmd = int(float(value))
        self.force_value_label.config(text=str(self.force_cmd))

    def update_velocity(self, value):
        self.vel_cmd = int(float(value))
        self.vel_value_label.config(text=str(self.vel_cmd))

    def update_acceleration(self, value):
        self.acc_cmd = int(float(value))
        self.acc_value_label.config(text=str(self.acc_cmd))

    def update_deceleration(self, value):
        self.dec_cmd = int(float(value))
        self.dec_value_label.config(text=str(self.dec_cmd))

    def send_command(self):
        # Initialize position and force, and first 2 sends won't execute
        if not self.initialized:
            self.controller.set_gripper_command(0, 0, 0, 0, 0)  # Initialize with zero values
            self.controller.set_gripper_command(255, 255, 255, 255, 255)  # Initialize with max values
            self.initialized = True
            self.update_status()
            return

        # Send the command after initialization
        self.controller.set_gripper_command(self.pos_cmd, self.force_cmd, self.vel_cmd, self.acc_cmd, self.dec_cmd)
        self.update_status()

    def update_status(self):
        # Retrieve and update the gripper status
        if self.initialized:
            status = self.controller.get_gripper_status(timeout=1)
            if status:
                fault_code = status["fault_code"]
                state = status["state"]
                pos = status["pos"]
                vel = status["vel"]
                force = status["force"]

                fault_codes = {
                    0: "No Fault",
                    1: "Overheat Warning",
                    2: "Over-Speed Warning",
                    3: "Initialization Fault",
                    4: "Over-Limit Detection Warning"
                }

                states = {
                    0: "Target Position Reached",
                    1: "Gripper Moving",
                    2: "Gripper Stuck",
                    3: "Object Dropped"
                }

                # Update labels with the new values
                self.fault_code_label.config(text=f"Fault Code: {fault_codes.get(fault_code, 'Unknown')}")
                self.state_label.config(text=f"State: {states.get(state, 'Unknown')}")
                self.pos_status_label.config(text=f"Position: {pos}")
                self.vel_status_label.config(text=f"Velocity: {vel}")
                self.force_status_label.config(text=f"Force: {force}")


if __name__ == "__main__":
    controller = OmniPickerCANController()
    app = OmniPickerUI(controller)
    app.mainloop()
