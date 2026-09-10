#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/6/13 16:14
# @Author : ZhangXi
import subprocess
import os
import webbrowser
import time

# 管理员密码
sudo_password = "nvidia"

# 要停止的服务列表
services = [
    "pitch_motor_api.service",
    "lift_motor_api.service",
    "modbus_rtu_server_api_with_elegripper.service"
]

# 停止服务
for svc in services:
    print(f"Stopping service: {svc}")
    cmd = f"sudo -S systemctl stop {svc}"
    subprocess.run(cmd, input=sudo_password + "\n", shell=True, text=True)

# 等待服务关闭
time.sleep(2.5)

# 启动 exhibition.py
print("Launching exhibition UI...")
exhibition_script = os.path.expanduser("~/Desktop/package/wrs/htw/exhibition.py")
subprocess.Popen(["python3", exhibition_script])



