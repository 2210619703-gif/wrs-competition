# -*- coding: utf-8 -*-
"""调用仿真规划器，把任务 JSON 规划成关节轨迹 JSON。

真机不自己做规划：抓取位姿搜索、IK、防穿桌、入框避让这些逻辑都在仿真脚本里，
重写一遍必然会和仿真跑出不一样的结果。这里用子进程调它们的 ``--export-traj``，
只取规划结果。

两条仿真链路都支持，导出的 JSON 格式一致（靠 ``source`` 字段区分）：

- ``standard``：``sim/run_agent_pick_place_side_sim.py``，标准抓放入收纳盒
- ``ec``：``sim/ec/run_ec_bin_sim.py``，螺丝入格

单独用::

    python tiaozhanbei/real/export_sim_traj.py \
        --task tiaozhanbei/tasks/delivery_agent_auto.json \
        --out tiaozhanbei/real/outputs/traj.json

    python tiaozhanbei/real/export_sim_traj.py --scene ec \
        --task tiaozhanbei/sim/ec/tasks/test03.json \
        --out tiaozhanbei/real/outputs/traj_ec.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from real_config import REAL_DIR, REPO_ROOT, TB_DIR, load_real_config

SIM_SCRIPT = os.path.join(TB_DIR, "sim", "run_agent_pick_place_side_sim.py")
EC_SCRIPT = os.path.join(TB_DIR, "sim", "ec", "run_ec_bin_sim.py")
DEFAULT_TASK = os.path.join(TB_DIR, "tasks", "delivery_agent_auto.json")
EC_DEFAULT_TASK = os.path.join(TB_DIR, "sim", "ec", "tasks", "test03.json")
OUTPUT_DIR = os.path.join(REAL_DIR, "outputs")

SCENES = ("standard", "ec")


class ExportError(RuntimeError):
    pass


def default_task_for(scene: str) -> str:
    return EC_DEFAULT_TASK if scene == "ec" else DEFAULT_TASK


def export_trajectory(
    task: str,
    out_path: str,
    *,
    scene: str = "standard",
    sample_id: str | None = None,
    dataset_root: str | None = None,
    sample_dir: str | None = None,
    joint_ranges_deg=None,
    python_exe: str | None = None,
    timeout: float = 900.0,
    echo: bool = True,
    logger=print,
) -> str:
    """跑一次仿真规划，返回导出的轨迹 JSON 路径。

    ``scene`` 选 ``"standard"``（抓放入收纳盒）或 ``"ec"``（螺丝入格），
    分别对应两个仿真脚本。两者导出的 JSON 结构一样，真机侧不用区分。

    ``joint_ranges_deg`` 传真机软限位（6 组 ``(lo, hi)``，度）时，规划阶段就只在
    真机能到的范围内搜索。不传的话仿真会用放宽后的演示范围，规划更容易成功，
    但结果多半没法直接下发到真机。
    """
    if scene not in SCENES:
        raise ExportError(f"scene 只能是 {SCENES}，收到 {scene!r}")
    script = EC_SCRIPT if scene == "ec" else SIM_SCRIPT
    if not os.path.isfile(script):
        raise ExportError(f"找不到仿真脚本：{script}")
    if not os.path.isfile(task):
        raise ExportError(f"找不到任务 JSON：{task}")

    out_path = os.path.abspath(out_path)
    if os.path.exists(out_path):
        os.remove(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        python_exe or sys.executable,
        script,
        "--task",
        os.path.abspath(task),
        "--export-traj",
        out_path,
        "--no-anime",
    ]
    if scene == "ec":
        # EC 按样本目录定位场景，没有 --sample-id / --dataset-root
        if sample_dir:
            cmd += ["--sample-dir", os.path.abspath(sample_dir)]
    else:
        if sample_id:
            cmd += ["--sample-id", str(sample_id)]
        if dataset_root:
            cmd += ["--dataset-root", os.path.abspath(dataset_root)]
    if joint_ranges_deg:
        ranges = [[round(float(lo), 4), round(float(hi), 4)] for lo, hi in joint_ranges_deg]
        cmd += ["--joint-ranges", json.dumps(ranges)]

    logger(f"[export] 规划中：{' '.join(cmd[1:])}")
    env = os.environ.copy()
    # Windows 管道默认是本机代码页（GBK）。子进程按 GBK 写、这边按 UTF-8 读，
    # 中文就会变成「�ؽڷ�Χ」。强制子进程用 UTF-8。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    tail: list[str] = []
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            tail.append(line)
            del tail[:-40]
            if echo:
                logger(f"  | {line}")
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise ExportError(f"仿真规划超时（>{timeout:.0f}s）")

    if not os.path.isfile(out_path):
        raise ExportError(
            f"仿真没有导出轨迹（exit={proc.returncode}），最后输出：\n  "
            + "\n  ".join(tail[-15:])
        )
    logger(f"[export] 轨迹已生成：{out_path}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="把任务 JSON 规划成真机可用的关节轨迹")
    p.add_argument(
        "--scene",
        default="standard",
        choices=SCENES,
        help="standard=抓放入收纳盒；ec=螺丝入格",
    )
    p.add_argument("--task", default=None, help="默认按 --scene 取各自的示例任务")
    p.add_argument("--sample-id", default=None, help="standard 场景的样本号")
    p.add_argument("--dataset-root", default=None, help="standard 场景的数据集根目录")
    p.add_argument("--sample-dir", default=None, help="ec 场景的样本目录")
    p.add_argument("--out", default=os.path.join(OUTPUT_DIR, "traj.json"))
    p.add_argument("--python", default=None, help="跑仿真用的解释器，默认用当前这个")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument(
        "--relaxed-limits",
        action="store_true",
        help="用仿真放宽后的关节范围规划（更容易成功，但结果通常不能下发真机）",
    )
    args = p.parse_args()

    try:
        export_trajectory(
            args.task or default_task_for(args.scene),
            args.out,
            scene=args.scene,
            sample_id=args.sample_id,
            dataset_root=args.dataset_root,
            sample_dir=args.sample_dir,
            joint_ranges_deg=None if args.relaxed_limits else load_real_config().limits_deg(),
            python_exe=args.python,
            timeout=args.timeout,
        )
    except ExportError as e:
        print(f"[export] 失败：{e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
