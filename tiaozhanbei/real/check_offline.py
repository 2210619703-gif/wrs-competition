# -*- coding: utf-8 -*-
"""不接硬件的自检：上真机之前先把这个跑绿。

检查 SDK 是否就位、``robot.cfg`` 能不能解析、当前解释器有没有匹配的
``fafu_motor`` 扩展、夹爪标定表在不在，以及（可选）一条已导出的轨迹能不能
安全地适配成真机动作。

``--vision`` 会一并检查视觉侧：YOLO 权重、类名映射表、手眼标定文件、
pyrealsense2，并用合成数据验证「深度 → 相机系 → 世界系」这条坐标链算得对不对。

    python tiaozhanbei/real/check_offline.py
    python tiaozhanbei/real/check_offline.py --vision
    python tiaozhanbei/real/check_offline.py --traj tiaozhanbei/real/outputs/traj.json
    python tiaozhanbei/real/check_offline.py --import-sdk
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

from real_config import (
    EXIT_HOME_CONF,
    GRIPPER_CALIB_FILE,
    SIM_HOME_CONF,
    ZERO_CONF,
    GripperMap,
    RealConfigError,
    find_sdk_dir,
    load_real_config,
)

_OK, _WARN, _FAIL = "[ ok ]", "[warn]", "[FAIL]"


class Checker:
    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def ok(self, msg: str) -> None:
        print(f"{_OK} {msg}")

    def warn(self, msg: str) -> None:
        self.warned += 1
        print(f"{_WARN} {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        print(f"{_FAIL} {msg}")


def check_sdk(ck: Checker) -> str | None:
    sdk = find_sdk_dir(required=False)
    if not sdk:
        ck.fail("找不到 fafu SDK（fafu_robot_python），设置环境变量 FAFU_SDK_DIR 指过去")
        return None
    ck.ok(f"SDK 目录 {sdk}")

    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    pyds = [os.path.basename(p) for p in glob.glob(os.path.join(sdk, "fafu_motor*.pyd"))]
    pyds += [os.path.basename(p) for p in glob.glob(os.path.join(sdk, "fafu_motor*.so"))]
    if not pyds:
        ck.fail("SDK 目录下没有 fafu_motor 扩展模块，需要先按 SDK README 编译")
    elif any(tag in name for name in pyds):
        ck.ok(f"扩展模块匹配当前解释器 {tag}：{', '.join(pyds)}")
    else:
        ck.fail(
            f"当前解释器是 {tag}，但只有 {', '.join(pyds)}。"
            "换一个匹配的 Python，或重新编译扩展"
        )
    return sdk


def check_config(ck: Checker):
    try:
        cfg = load_real_config()
    except RealConfigError as e:
        ck.fail(str(e))
        return None
    ck.ok(f"robot.cfg 解析成功：{cfg.cfg_path}")
    ck.ok(
        f"关节 {list(cfg.joint_ids)}，夹爪 M{cfg.gripper_id}，"
        f"port={cfg.port} baud={cfg.baudrate}"
    )
    for i, (lo, hi) in enumerate(cfg.limits_deg()):
        if hi - lo < 1.0:
            ck.fail(f"J{i + 1} 软限位区间只有 {hi - lo:.2f}°，配置多半写错了")
    ck.ok(
        "软限位(°) "
        + " ".join(f"J{i + 1}[{lo:.0f},{hi:.0f}]" for i, (lo, hi) in enumerate(cfg.limits_deg()))
    )

    home, hit = cfg.clamp(SIM_HOME_CONF[: cfg.num_joints])
    if hit:
        ck.fail(
            f"仿真 HOME 位形超出真机软限位（J{[j + 1 for j in hit]}），"
            "真机起始位形需要重新确认"
        )
    else:
        ck.ok("仿真 HOME 位形在真机软限位内：" + ", ".join(f"{math.degrees(v):.1f}°" for v in home))

    zero, hit = cfg.clamp(ZERO_CONF[: cfg.num_joints])
    if hit:
        ck.fail(
            f"电机零位超出真机软限位（J{[j + 1 for j in hit]}），"
            "ZERO_CONF 需要重新确认"
        )
    else:
        ck.ok("电机零位在真机软限位内：" + ", ".join(f"{math.degrees(v):.1f}°" for v in zero))

    rest, hit = cfg.clamp(EXIT_HOME_CONF[: cfg.num_joints])
    if hit:
        ck.fail(
            f"退出静置位超出真机软限位（J{[j + 1 for j in hit]}），"
            "EXIT_HOME_CONF_DEG 需要重新确认"
        )
    else:
        ck.ok("退出静置位在真机软限位内：" + ", ".join(f"{math.degrees(v):.1f}°" for v in rest))
    return cfg


def check_gripper(ck: Checker, cfg) -> GripperMap:
    gmap = GripperMap.from_config(cfg)
    if os.path.isfile(GRIPPER_CALIB_FILE):
        ck.ok(f"夹爪标定表已就位：{gmap.describe()}")
    else:
        ck.warn(
            "还没有 gripper_calib.json，暂时用线性映射。"
            "上真机前建议跑 calibrate_gripper.py --execute"
        )
    return gmap


def check_traj(ck: Checker, cfg, gmap: GripperMap, traj_path: str) -> None:
    from traj_adapter import adapt, format_plan, load_sim_traj

    try:
        traj = load_sim_traj(traj_path)
    except Exception as e:
        ck.fail(f"轨迹读取失败：{e}")
        return
    plan = adapt(traj, cfg, gmap)
    print(format_plan(plan))
    if plan["errors"]:
        ck.fail(f"轨迹适配出 {len(plan['errors'])} 个致命问题，不能下发")
    elif plan["warnings"]:
        ck.warn(f"轨迹可以下发，但有 {len(plan['warnings'])} 条告警，先看清楚")
    else:
        ck.ok("轨迹适配通过，可以 dry-run 了")


def check_vision(ck: Checker) -> None:
    """视觉侧自检：不接相机也能跑完。"""
    import numpy as np

    from vision_config import (
        HANDEYE_JSON,
        ClassMap,
        VisionConfigError,
        default_weights,
        describe_handeye,
        load_handeye,
        points_in_workspace,
        transform_points,
    )

    try:
        weights = default_weights()
        ck.ok(f"YOLO 权重 {os.path.basename(weights)}")
    except VisionConfigError as e:
        ck.fail(str(e))
        weights = None

    try:
        cmap = ClassMap()
        ck.ok(cmap.describe())
        if cmap.unmapped:
            ck.warn(f"有 {len(cmap.unmapped)} 类没映射，用到这些类时会被跳过")
        # 抽查几个常用类，确认 STL 真的存在
        probes = ("screwdrivers", "wrenches", "hex_nuts")
        for en in probes:
            hit = cmap.resolve(en)
            if hit is None:
                ck.warn(f"类 {en} 解析不到 STL，识别到它会被跳过")
            else:
                ck.ok(f"{en} → {hit['zh']}  STL {os.path.basename(hit['stl_path'])}")
    except VisionConfigError as e:
        ck.fail(str(e))

    if os.path.isfile(HANDEYE_JSON):
        try:
            ck.ok(f"手眼标定已就位：{describe_handeye(load_handeye())}")
        except VisionConfigError as e:
            ck.fail(str(e))
    else:
        ck.warn(
            "还没有 handeye_d405.json。相机装好后必须先跑 hand_eye_calib.py，"
            "否则世界坐标全是错的"
        )

    from d405_camera import deproject, pyrealsense_available

    if pyrealsense_available():
        ck.ok("pyrealsense2 可用")
    else:
        ck.warn("没装 pyrealsense2，连不了 D405（离线调试不受影响）")

    # 坐标链自检：合成一张深度图，反投影再变到世界系，看能不能还原已知位置
    intr = {"fx": 600.0, "fy": 600.0, "cx": 159.5, "cy": 119.5}
    h, w = 240, 320
    cam_h = 0.60
    depth = np.full((h, w), cam_h, dtype=np.float32)   # 平桌面
    # 在画面中央摆一个 20mm 高的台阶，代表一个零件
    depth[110:130, 150:170] = cam_h - 0.020
    pts = deproject(depth, intr)
    # 相机在 (0.35, 0, 0.60) 正对下方：相机 +Z→世界 -Z，+X→世界 +Y，+Y→世界 +X
    w2c = np.eye(4)
    w2c[:3, :3] = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    w2c[:3, 3] = np.array([0.35, 0.0, cam_h])
    world = transform_points(w2c, pts.reshape(-1, 3))
    desk_z = float(np.median(world[:, 2]))
    step = world.reshape(h, w, 3)[110:130, 150:170].reshape(-1, 3)
    step_z = float(np.median(step[:, 2]))
    step_xy = np.median(step[:, :2], axis=0)
    if abs(desk_z) < 1e-6 and abs(step_z - 0.020) < 1e-6:
        ck.ok(
            f"坐标链自检通过：桌面 z={desk_z:+.4f}m，20mm 台阶 z={step_z:.4f}m，"
            f"台阶中心 xy=({step_xy[0]:+.4f},{step_xy[1]:+.4f})"
        )
    else:
        ck.fail(f"坐标链自检失败：桌面 z={desk_z:+.6f}（应为 0），台阶 z={step_z:.6f}（应为 0.020）")

    kept = points_in_workspace(world)
    if len(kept):
        ck.ok(f"工作区过滤保留 {len(kept)}/{len(world)} 点")
    else:
        ck.warn("工作区过滤把合成点全滤掉了，检查 vision_config.WORKSPACE_BOUNDS_M")

    # mask → pose_6d 自检：造一个长条形零件，位置和朝向都已知，看能不能还原
    from vision_to_annotations import pose_from_world_points, world_points_for_mask

    depth2 = np.full((h, w), cam_h, dtype=np.float32)
    # 沿图像 u 方向的长条（40x8 像素），高出桌面 6mm
    v0, v1, u0, u1 = 116, 124, 130, 190
    depth2[v0:v1, u0:u1] = cam_h - 0.006
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[v0:v1, u0:u1] = 1
    pts2 = deproject(depth2, intr)
    world2 = world_points_for_mask(mask, pts2, w2c)
    pose = pose_from_world_points("机器工具-螺丝刀", world2)["pose_6d"]

    # 相机 +X → 世界 +Y，所以图像里沿 u 的长条在世界系里是沿 Y 的，yaw≈90°
    u_mid, v_mid = 0.5 * (u0 + u1 - 1), 0.5 * (v0 + v1 - 1)
    exp_y = (u_mid - intr["cx"]) / intr["fx"] * cam_h
    exp_x = 0.35 + (v_mid - intr["cy"]) / intr["fy"] * cam_h
    err_mm = math.hypot(pose["x"] - exp_x, pose["y"] - exp_y) * 1000.0
    yaw_deg = abs(math.degrees(pose["yaw"]))
    if err_mm < 1.0 and abs(yaw_deg - 90.0) < 2.0 and abs(pose["z"] - 0.006) < 5e-4:
        ck.ok(
            f"mask→位姿自检通过：xy 误差 {err_mm:.2f}mm，yaw={yaw_deg:.1f}°(应≈90)，"
            f"高度 {pose['z'] * 1000:.1f}mm(应≈6)"
        )
    else:
        ck.fail(
            f"mask→位姿自检失败：xy 误差 {err_mm:.2f}mm，yaw={yaw_deg:.1f}°(应≈90)，"
            f"高度 {pose['z'] * 1000:.1f}mm(应≈6)"
        )


def main() -> int:
    p = argparse.ArgumentParser(description="真机执行模块离线自检")
    p.add_argument("--traj", default=None, help="顺便校验一条已导出的仿真轨迹")
    p.add_argument("--vision", action="store_true", help="一并检查视觉侧（权重/类名表/手眼/坐标链）")
    p.add_argument("--import-sdk", action="store_true", help="尝试真的 import SDK（不连串口）")
    args = p.parse_args()

    ck = Checker()
    print("=" * 62)
    check_sdk(ck)
    cfg = check_config(ck)
    if cfg is not None:
        gmap = check_gripper(ck, cfg)
        if args.traj:
            check_traj(ck, cfg, gmap, args.traj)

    if args.vision:
        print("-" * 62)
        check_vision(ck)

    if args.import_sdk:
        try:
            from real_config import ensure_sdk_on_path

            ensure_sdk_on_path()
            import fafu_robot_controller  # noqa: F401

            ck.ok("import fafu_robot_controller 成功")
        except Exception as e:
            ck.fail(f"import SDK 失败：{type(e).__name__}: {e}")

    print("=" * 62)
    if ck.failed:
        print(f"自检未通过：{ck.failed} 项失败，{ck.warned} 项告警")
        return 1
    print(f"自检通过（{ck.warned} 项告警）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
