# -*- coding: utf-8 -*-
"""读真机关节角，并用 WRS FK 算出法兰世界坐标。

用来填 ``fruit_config.py`` 里的 LOOK / A / B 示教位。只读、不动臂::

    python demo/read_joints.py
    python demo/read_joints.py --watch

手推或示教器摆到目标位后再跑。输出里有可直接粘贴的 ``LOOK_CONF_DEG = (...)``。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

import numpy as np

from fruit_config import HANDEYE_JSON, REAL_DIR, REPO_ROOT
from fruit_vision import describe_arm_state, format_conf_deg, load_handeye_raw

if REAL_DIR not in sys.path:
    sys.path.insert(0, REAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from real_arm import RealArm, RealArmError  # noqa: E402
from real_config import RealConfigError, load_real_config  # noqa: E402


def snapshot_text(arm: RealArm) -> str:
    q = arm.joint_values()
    r2cam = None
    if os.path.isfile(HANDEYE_JSON):
        mat = load_handeye_raw()
        if not np.allclose(mat, np.eye(4)):
            r2cam = mat
    return describe_arm_state(q, r2cam=r2cam)


def connect_readonly(port=None, logger=print) -> RealArm:
    cfg = load_real_config()
    logger(f"[real] robot.cfg = {cfg.cfg_path}")
    logger(f"[real] port={port or cfg.port} baud={cfg.baudrate}")
    arm = RealArm(cfg, dry_run=False, speed=1, port=port, logger=logger, use_servo=False)
    arm.connect()
    return arm


def main() -> int:
    p = argparse.ArgumentParser(description="读真机关节角和法兰世界坐标")
    p.add_argument("--port", default=None, help="串口；默认 robot.cfg 的 auto")
    p.add_argument("--watch", action="store_true", help="持续刷新，Ctrl+C 停")
    p.add_argument("--hz", type=float, default=2.0, help="--watch 刷新频率，默认 2")
    args = p.parse_args()

    try:
        arm = connect_readonly(args.port)
    except (RealConfigError, RealArmError, RuntimeError) as e:
        print(f"[real] 连不上：{e}")
        return 1

    try:
        if not args.watch:
            print(snapshot_text(arm))
            print("\n抄到 fruit_config.py 时改名字：")
            print(f"  LOOK_CONF_DEG        = {format_conf_deg(arm.joint_values())}")
            print(f"  BIN_A_CONF_DEG       = {format_conf_deg(arm.joint_values())}")
            print(f"  BIN_B_CONF_DEG       = {format_conf_deg(arm.joint_values())}")
            print(f"  SLOT_LEFT_CONF_DEG   = {format_conf_deg(arm.joint_values())}")
            print(f"  SLOT_RIGHT_CONF_DEG  = {format_conf_deg(arm.joint_values())}")
            return 0
        print("[real] 持续读数，Ctrl+C 结束")
        period = 1.0 / max(0.2, float(args.hz))
        while True:
            print("-" * 48)
            print(snapshot_text(arm))
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[real] 已停")
        return 0
    finally:
        try:
            arm.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
