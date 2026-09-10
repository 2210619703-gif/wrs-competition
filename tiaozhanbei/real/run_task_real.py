# -*- coding: utf-8 -*-
"""真机执行模块主入口：任务 JSON → 仿真规划 → 限位适配 → 下发 fafu 机械臂。

默认是 **dry-run**，只打印会下发什么，不碰硬件::

    python tiaozhanbei/real/run_task_real.py --task tiaozhanbei/tasks/delivery_agent_auto.json

确认无误后再加 ``--execute`` 真的驱动机械臂（低速）::

    python tiaozhanbei/real/run_task_real.py --task ... --execute --speed 8

螺丝入格（EC）走同一个入口，只是换个场景::

    python tiaozhanbei/real/run_task_real.py --scene ec

已经导出过轨迹的话可以跳过规划，直接复用::

    python tiaozhanbei/real/run_task_real.py --traj tiaozhanbei/real/outputs/traj.json --execute
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from export_sim_traj import (
    OUTPUT_DIR,
    SCENES,
    ExportError,
    default_task_for,
    export_trajectory,
)
from real_arm import RealArm, RealArmError, execute_plan
from real_config import (
    DEFAULT_MOVE_SPEED,
    GripperMap,
    RealConfigError,
    load_real_config,
)
from traj_adapter import adapt, format_plan, load_sim_traj, save_plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="在 fafu 真机上执行决策模块生成的任务 JSON")
    src = p.add_argument_group("任务来源")
    src.add_argument(
        "--scene",
        default="standard",
        choices=SCENES,
        help="standard=抓放入收纳盒；ec=螺丝入格",
    )
    src.add_argument("--task", default=None, help="决策模块产出的任务 JSON；默认按场景取示例")
    src.add_argument("--sample-id", default=None, help="standard 场景的样本号")
    src.add_argument("--dataset-root", default=None, help="standard 场景的数据集根目录")
    src.add_argument("--sample-dir", default=None, help="ec 场景的样本目录")
    src.add_argument("--traj", default=None, help="直接复用已导出的轨迹 JSON，跳过仿真规划")
    src.add_argument(
        "--relaxed-limits",
        action="store_true",
        help="规划时用仿真放宽后的关节范围。默认按 robot.cfg 软限位规划，"
        "结果才可能真的下发到真机",
    )

    out = p.add_argument_group("产物")
    out.add_argument("--traj-out", default=os.path.join(OUTPUT_DIR, "traj.json"))
    out.add_argument("--plan-out", default=os.path.join(OUTPUT_DIR, "plan.json"))

    run = p.add_argument_group("执行")
    run.add_argument("--execute", action="store_true", help="真的驱动机械臂；不加就是 dry-run")
    run.add_argument("--speed", type=int, default=DEFAULT_MOVE_SPEED, help="move_j 速度 (0,100]")
    run.add_argument("--port", default=None, help="串口，默认按 robot.cfg（auto 自动扫描）")
    run.add_argument("--min-delta-deg", type=float, default=1.0, help="抽帧阈值，越大航点越少")
    run.add_argument("--no-home-first", action="store_true")
    run.add_argument("--no-home-after", action="store_true")
    run.add_argument("--stop-on-fail", action="store_true", help="某个目标失败就整批停下")
    return p


def main() -> int:
    args = build_parser().parse_args()

    try:
        cfg = load_real_config()
    except RealConfigError as e:
        print(f"[real] 配置错误：{e}")
        return 2
    print(f"[real] robot.cfg = {cfg.cfg_path}")
    print(
        "[real] 软限位(°) "
        + " ".join(
            f"J{i + 1}[{lo:.0f},{hi:.0f}]" for i, (lo, hi) in enumerate(cfg.limits_deg())
        )
    )

    traj_path = args.traj
    if not traj_path:
        try:
            traj_path = export_trajectory(
                args.task or default_task_for(args.scene),
                args.traj_out,
                scene=args.scene,
                sample_id=args.sample_id,
                dataset_root=args.dataset_root,
                sample_dir=args.sample_dir,
                joint_ranges_deg=None if args.relaxed_limits else cfg.limits_deg(),
            )
        except ExportError as e:
            print(f"[real] 规划失败：{e}")
            return 1
    traj = load_sim_traj(traj_path)

    gmap = GripperMap.from_config(cfg)
    plan = adapt(traj, cfg, gmap, min_delta=math.radians(args.min_delta_deg))
    plan_path = save_plan(plan, args.plan_out)
    print(format_plan(plan))
    print(f"[real] plan 已保存：{plan_path}")

    if plan["errors"]:
        print("[real] plan 有致命问题，未下发。请先解决上面的 [ERROR]。")
        return 3

    if args.execute and gmap.source == "linear":
        print(
            "[real] 提醒：夹爪用的是线性默认映射，还没标定。"
            "建议先跑 calibrate_gripper.py 生成 gripper_calib.json。"
        )

    # dry-run 也走同一套执行流程，只是换成不连硬件的虚拟臂，
    # 这样每条指令都会打印出来，执行器本身的逻辑也顺带验证了。
    arm = RealArm(cfg, dry_run=not args.execute, speed=args.speed, port=args.port)
    try:
        with arm:
            summary = execute_plan(
                plan,
                arm,
                go_home_first=not args.no_home_first,
                go_home_after=not args.no_home_after,
                stop_on_fail=args.stop_on_fail,
            )
    except (RealArmError, RuntimeError) as e:
        print(f"[real] 执行中止：{e}")
        return 1

    if not args.execute:
        print("[real] 以上是 dry-run，没有下发任何指令。确认无误后加 --execute 真机执行。")
    return 0 if summary["ok"] == summary["total"] else 4


if __name__ == "__main__":
    sys.exit(main())
