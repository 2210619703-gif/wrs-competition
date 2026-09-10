# -*- coding: utf-8 -*-
"""最小真机动作：连上 USB，读关节角，可选让 J2 慢转 5° 再回去。

不经过任务 JSON、不规划、不抓零件。用来确认串口、上电、使能是通的。

默认只读、不动::

    python tiaozhanbei/real/smoke_move.py

确认能读到角度后再动（会先问一次）::

    python tiaozhanbei/real/smoke_move.py --nudge

速度已经压到 8。手放在急停 / Ctrl+C 上。
"""
from __future__ import annotations

import argparse
import math
import sys

from real_arm import RealArm, RealArmError
from real_config import RealConfigError, load_real_config


def _fmt(q) -> str:
    return "[" + ", ".join(f"{math.degrees(v):7.2f}" for v in q) + "]°"


def main() -> int:
    p = argparse.ArgumentParser(description="真机最小动作：读状态，可选 J2 ±5°")
    p.add_argument("--nudge", action="store_true", help="J2 往前 5°，停一下，再回去")
    p.add_argument("--delta-deg", type=float, default=5.0, help="J2 摆动幅度，默认 5")
    p.add_argument("--speed", type=int, default=8, help="move_j 速度，默认 8")
    p.add_argument("--port", default=None, help="串口；默认按 robot.cfg 的 auto")
    p.add_argument("--joint", type=int, default=2, help="动哪个关节（1~6），默认 J2")
    args = p.parse_args()

    try:
        cfg = load_real_config()
    except RealConfigError as e:
        print(f"[real] 配置错误：{e}")
        return 2

    j = int(args.joint) - 1
    if j < 0 or j >= cfg.num_joints:
        print(f"[real] --joint 只能是 1~{cfg.num_joints}")
        return 2

    print(f"[real] robot.cfg = {cfg.cfg_path}")
    print(f"[real] port={args.port or cfg.port} baud={cfg.baudrate}")
    print("[real] 连接中…")

    arm = RealArm(cfg, dry_run=False, speed=args.speed, port=args.port)
    try:
        with arm:
            q0 = arm.joint_values()
            print(f"[real] 当前关节 {_fmt(q0)}")
            if not args.nudge:
                print("[real] 只读成功。要看动作再加 --nudge")
                return 0

            delta = math.radians(args.delta_deg)
            target = list(q0)
            target[j] = q0[j] + delta
            target, hit = cfg.clamp(target)
            if hit:
                print(f"[real] 目标超出软限位，已裁剪 J{[i + 1 for i in hit]}")

            ans = input(
                f"\n  即将 J{j + 1} 从 {math.degrees(q0[j]):.1f}° "
                f"转到 {math.degrees(target[j]):.1f}°，再回去。"
                f" 回车开始，输入 q 取消: "
            ).strip().lower()
            if ans == "q":
                print("[real] 已取消")
                return 0

            print(f"[real] J{j + 1} +{args.delta_deg:.1f}°")
            arm.move_j(target, note="nudge", allow_large=True)
            print(f"[real] 到位 {_fmt(arm.joint_values())}")
            print(f"[real] 回到起点")
            arm.move_j(q0, note="back", allow_large=True)
            print(f"[real] 结束 {_fmt(arm.joint_values())}")
    except (RealArmError, RuntimeError, KeyboardInterrupt) as e:
        print(f"[real] 中止：{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
