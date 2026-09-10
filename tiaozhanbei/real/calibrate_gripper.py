# -*- coding: utf-8 -*-
"""标定「夹爪角度 → 实际开口」，生成 ``gripper_calib.json``。

仿真里夹爪是平移开口（毫米级宽度），真机 M7 是旋转关节（0~105°），两者之间
没有现成公式。不标定的话 ``real_config.GripperMap`` 只能按线性猜，细小零件很容易
夹不住或夹过头。

流程：脚本把夹爪依次停在若干角度，你用卡尺量两指之间的开口，按毫米输入，
最后写成一张分段线性表。整个过程只动夹爪，不动手臂。

    python tiaozhanbei/real/calibrate_gripper.py            # 预览要走哪些角度
    python tiaozhanbei/real/calibrate_gripper.py --execute  # 真机标定
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from real_config import (
    GRIPPER_CALIB_FILE,
    RealConfigError,
    load_real_config,
)
from real_arm import RealArm


def _ask_width_mm(angle_deg: float) -> float | None:
    while True:
        raw = input(f"  夹爪在 {angle_deg:.1f}°，量到的开口(mm)，回车跳过： ").strip()
        if not raw:
            return None
        try:
            val = float(raw)
        except ValueError:
            print("  请输入数字，比如 12.5")
            continue
        if val < 0:
            print("  开口不能是负数")
            continue
        return val


def main() -> int:
    p = argparse.ArgumentParser(description="标定 fafu 夹爪角度与实际开口的对应关系")
    p.add_argument("--execute", action="store_true", help="真的驱动夹爪；不加只预览角度序列")
    p.add_argument("--points", type=int, default=8, help="采样点数量")
    p.add_argument("--port", default=None)
    p.add_argument("--out", default=GRIPPER_CALIB_FILE)
    p.add_argument("--settle", type=float, default=0.8, help="每点停稳时间(秒)")
    args = p.parse_args()

    try:
        cfg = load_real_config()
    except RealConfigError as e:
        print(f"[calib] 配置错误：{e}")
        return 2

    lo, hi = cfg.gripper_limits
    n = max(2, int(args.points))
    angles = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    print(f"[calib] 夹爪软限位 {math.degrees(lo):.1f}° ~ {math.degrees(hi):.1f}°")
    print("[calib] 采样角度： " + ", ".join(f"{math.degrees(a):.1f}°" for a in angles))

    if not args.execute:
        print("[calib] 预览模式结束。确认夹爪周围无遮挡后，加 --execute 开始标定。")
        return 0

    arm = RealArm(cfg, dry_run=False, port=args.port)
    points: list[dict] = []
    with arm:
        for angle in angles:
            arm.gripper_move(angle, note="(标定)")
            time.sleep(max(0.0, args.settle))
            width_mm = _ask_width_mm(math.degrees(angle))
            if width_mm is None:
                continue
            points.append(
                {"width_m": round(width_mm / 1000.0, 5), "angle_deg": round(math.degrees(angle), 3)}
            )
        arm.gripper_move(lo, note="(标定结束，合爪)")

    if len(points) < 2:
        print("[calib] 有效采样点不足 2 个，没有写文件。")
        return 1

    points.sort(key=lambda d: d["width_m"])
    payload = {
        "note": "仿真夹爪开口(米) → fafu M7 角度(度)，由 calibrate_gripper.py 生成",
        "gripper_motor_id": cfg.gripper_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": points,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[calib] 已写入 {args.out}（{len(points)} 个点）")
    print("[calib] 之后 run_task_real.py 会自动用这张表换算夹爪角度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
