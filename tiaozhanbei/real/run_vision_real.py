# -*- coding: utf-8 -*-
"""视觉到真机的一条命令：相机识别 → 仿真规划 → 限位适配 → 下发。

    相机拍一帧 → YOLO 出 mask → 手眼变到世界系 → annotations.json
      → 仿真按这份标注规划关节轨迹（--export-traj）
      → traj_adapter 裁限位/抽帧/补过渡
      → dry-run 打印，或 --execute 真的驱动机械臂

默认行为全部写在 ``real_config.py`` 的「运行模式」一节，平时改那里就行::

    python tiaozhanbei/real/run_vision_real.py

配置里三个模式：``vision`` 只识别、``interactive`` 识别一次后循环下指令、
``once`` 跑完就退。``EXECUTE_ON_REAL`` 决定是 dry-run 还是真下发。

命令行仍可临时覆盖，每个开关都有 ``--xxx`` / ``--no-xxx``::

    python tiaozhanbei/real/run_vision_real.py --mode vision        # 这次只识别
    python tiaozhanbei/real/run_vision_real.py --no-execute         # 这次只 dry-run
    python tiaozhanbei/real/run_vision_real.py --mode once --pick 螺丝刀

先存一帧再离线反复试，不占相机::

    python tiaozhanbei/real/d405_camera.py --save-dir tiaozhanbei/real/outputs/frame01
    python tiaozhanbei/real/run_vision_real.py --frame-dir tiaozhanbei/real/outputs/frame01

注意：**执行阶段不看相机**。轨迹是按拍照那一刻的零件位置开环规划的，
桌面动过就要重新跑一次识别，否则会抓空或撞件。

连相机拍照前会先把机械臂退到避让位（``vision`` 模式同样如此），避免臂被当成零件。
桌面上的零件还在画面里。离线帧或 ``--reuse-vision`` 不挪臂。不想动臂用 ``--no-park``。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from export_sim_traj import OUTPUT_DIR, ExportError, export_trajectory
from real_arm import RealArm, RealArmError, execute_plan, release_at_place
from real_config import (
    CONTINUOUS_PATH,
    DEFAULT_MOVE_SPEED,
    DEST_NAME,
    DEST_XYZ,
    EXECUTE_ON_REAL,
    HOME_ON_EXIT,
    PARK_BEFORE_VISION,
    PARK_RELEASE_DELAY_S,
    PARK_SETTLE_S,
    PICK_ONLY,
    PICK_TARGETS,
    PARK_CONF,
    REAL_DIR,
    REPO_ROOT,
    START_FROM_PARK,
    RELEASE_MODE_ON_CLOSE,
    RELEASE_MODES,
    RUN_MODE,
    RUN_MODES,
    USE_SERVO_J,
    GripperMap,
    RealConfigError,
    load_real_config,
)
from traj_adapter import adapt, format_plan, load_sim_traj, save_plan
from vision_config import YOLO_CONF, YOLO_IMGSZ, YOLO_IOU, VisionConfigError, describe_handeye, load_handeye
from vision_to_annotations import (
    VISION_DIR,
    VisionError,
    capture_frame,
    detect_to_annotations,
    print_objects,
    save_annotations,
    save_preview,
)

DEFAULT_DEST_XY = tuple(DEST_XYZ)
DEFAULT_DESTINATION = DEST_NAME
# 仿真把这个值当成「没有放置目标」，据此跳过建收纳盒。
NO_DEST = "null"


def _describe(obj: dict) -> str:
    """``[3]机器工具-螺丝刀(screwdrivers)``；复用数据集标注时没有英文类名，就不显示。"""
    en = obj.get("yolo_class")
    return f"[{obj.get('id')}]{obj.get('class')}" + (f"({en})" if en else "")


def select_targets(ann: dict, pick: list[str], want_all: bool) -> list[dict]:
    """按用户给的名字挑出要抓的零件。

    ``pick`` 支持三种写法：中文类名片段（``螺丝刀``）、YOLO 英文类名
    （``screwdrivers``）、或 ``id:3`` 直接指编号。片段匹配是为了不用敲全
    ``机器工具-螺丝刀``。
    """
    objects = ann.get("objects") or []
    if want_all:
        return list(objects)

    chosen: list[dict] = []
    for token in pick:
        key = token.strip()
        if not key:
            continue
        hit = None
        if key.lower().startswith("id:"):
            try:
                target_id = int(key.split(":", 1)[1])
            except ValueError:
                target_id = -1
            hit = next((o for o in objects if int(o.get("id", -1)) == target_id), None)
        if hit is None:
            low = key.lower()
            hit = next(
                (
                    o
                    for o in objects
                    if key in str(o.get("class", ""))
                    or low == str(o.get("yolo_class", "")).lower()
                    or low in str(o.get("yolo_class", "")).lower()
                ),
                None,
            )
        if hit is None:
            raise VisionError(
                f"识别结果里没有 {key!r}。当前可选：" + "、".join(_describe(o) for o in objects)
            )
        if hit not in chosen:
            chosen.append(hit)
    return chosen


def _steps_for(obj: dict, dest, destination: str, pick_only: bool = False) -> list[dict]:
    coord = [obj["pose_6d"]["x"], obj["pose_6d"]["y"], 0.0]
    yaw = float(obj["pose_6d"]["yaw"])
    common = {
        "object": obj["class"],
        # destination 置空是仿真识别「只抓不放」的开关：它据此完全不建收纳盒，
        # 也就不会把盒子当障碍参与规划。用字符串 "null" 而不是 None，
        # 因为仿真展开 batch 时对 None 会回退成默认收纳盒。
        "destination": NO_DEST if pick_only else destination,
        "dest_coordinate": [0.0, 0.0, 0.0] if pick_only else list(dest),
        "angle": yaw,
        "rpy": [0.0, 0.0, yaw],
        "retry": 0,
        "max_retry": 3,
        "status": "pending",
        "pose_state": obj.get("state", "normal"),
    }
    steps = [
        {"task_id": "t000", "step": 1, "action": "perceive", "coordinate": coord, **common},
        {"task_id": "t001", "step": 2, "action": "pick", "coordinate": coord, **common},
    ]
    if pick_only:
        return steps
    return steps + [
        {"task_id": "t002", "step": 3, "action": "move", "coordinate": list(dest), **common},
        {"task_id": "t003", "step": 4, "action": "place", "coordinate": list(dest), **common},
    ]


def build_task(
    ann: dict,
    targets: list[dict],
    dest,
    destination: str,
    cmd: str,
    pick_only: bool = False,
) -> dict:
    """按识别结果拼出仿真能读的任务 JSON。

    单目标走普通 pick_place；多目标用 ``batch``，仿真会在**一个**进程里
    连续抓放，而不是每个零件起一次仿真。

    ``pick_only`` 只抓起来悬停，不放：现场收纳盒位置还没定的时候用，
    避免仿真按一个假盒位规划、结果盒子还挡在路上。
    """
    if not targets:
        raise VisionError("没有选中任何零件")

    dest_xyz = [0.0, 0.0, 0.0] if pick_only else list(dest)
    dest_name = NO_DEST if pick_only else destination

    if len(targets) == 1:
        obj = targets[0]
        yaw = float(obj["pose_6d"]["yaw"])
        task = {
            "task_id": "real001",
            "action": "pick" if pick_only else "pick_place",
            "object": obj["class"],
            "destination": dest_name,
            "coordinate": [obj["pose_6d"]["x"], obj["pose_6d"]["y"], 0.0],
            "dest_coordinate": dest_xyz,
            "angle": yaw,
            "rpy": [0.0, 0.0, yaw],
            "pose_state": obj.get("state", "normal"),
            "retry": 0,
            "max_retry": 3,
            "status": "valid",
            "original_cmd": cmd,
            "tasks": _steps_for(obj, dest, destination, pick_only=pick_only),
        }
    else:
        task = {
            "task_id": "real_batch",
            "action": "multi_pick" if pick_only else "multi_pick_place",
            "object": "所有零件",
            "destination": dest_name,
            "dest_coordinate": dest_xyz,
            "coordinate": [targets[0]["pose_6d"]["x"], targets[0]["pose_6d"]["y"], 0.0],
            "status": "valid",
            "original_cmd": cmd,
            "batch": [
                {
                    "object": o["class"],
                    "coordinate": [o["pose_6d"]["x"], o["pose_6d"]["y"], 0.0],
                    "dest_coordinate": dest_xyz,
                    "destination": dest_name,
                    "pose_state": o.get("state", "normal"),
                    "angle": float(o["pose_6d"]["yaw"]),
                    "rpy": [0.0, 0.0, float(o["pose_6d"]["yaw"])],
                    "reason": f"视觉识别 {o.get('yolo_class')}",
                    "steps": ["perceive", "pick"]
                    if pick_only
                    else ["perceive", "pick", "move", "place"],
                }
                for o in targets
            ],
        }

    return {
        "scene": "standard",
        "scene_source": "real_d405",
        "agent_source": "真机视觉（D405 + YOLO-seg）",
        "user_cmd": cmd,
        "task": task,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="D405 识别 → 仿真规划 → fafu 真机执行",
        epilog=(
            "默认行为全部写在 real_config.py 的「运行模式」一节，"
            "平时改那里就行，不用在命令行敲开关。"
            f"当前：模式={RUN_MODE} 下发={'是' if EXECUTE_ON_REAL else '否(dry-run)'} "
            f"连续={'是' if CONTINUOUS_PATH else '否'} "
            f"只抓={'是' if PICK_ONLY else '否'} 速度={DEFAULT_MOVE_SPEED} "
            f"刹车={RELEASE_MODE_ON_CLOSE}"
        ),
    )

    vis = p.add_argument_group("视觉")
    vis.add_argument("--frame-dir", default="", help="用存下的帧，不连相机")
    vis.add_argument("--resolution", default="mid", choices=("mid", "high"))
    vis.add_argument("--serial", default=None)
    vis.add_argument("--frames", type=int, default=5, help="曝光稳住后连拍帧数（深度中位数 + YOLO 投票）")
    vis.add_argument("--weights", default="", help="YOLO 权重；默认用本目录的")
    vis.add_argument("--handeye", default="", help="手眼 JSON；默认 handeye_d405.json")
    vis.add_argument("--conf", type=float, default=YOLO_CONF)
    vis.add_argument("--iou", type=float, default=YOLO_IOU)
    vis.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help="YOLO 输入边长，需与实时预览一致")
    vis.add_argument("--sample-id", default="real01")
    vis.add_argument("--vision-root", default=VISION_DIR)
    vis.add_argument("--reuse-vision", action="store_true",
                     help="直接用上次的 annotations.json，不重新识别")

    tsk = p.add_argument_group("任务（默认取 real_config.py）")
    tsk.add_argument("--mode", choices=RUN_MODES, default=RUN_MODE,
                     help=f"vision=只识别 / interactive=循环下指令 / once=跑完就退 / "
                          f"zero=送到电机零位；默认 {RUN_MODE}")
    tsk.add_argument("--pick", action="append", default=None,
                     help="要抓的零件，可重复；支持中文片段 / YOLO 英文类名 / id:N")
    tsk.add_argument("--all", action="store_true", help="抓识别到的全部零件")
    tsk.add_argument("--task", default="", help="直接用现成任务 JSON，跳过自动生成")
    tsk.add_argument("--dest", type=float, nargs=3, default=list(DEFAULT_DEST_XY),
                     metavar=("X", "Y", "Z"), help="放置点世界坐标（米）")
    tsk.add_argument("--destination", default=DEFAULT_DESTINATION)
    tsk.add_argument(
        "--pick-only", dest="pick_only", action=argparse.BooleanOptionalAction,
        default=PICK_ONLY,
        help="只抓起来悬停，不放；仿真不会建收纳盒（现场盒位还没定时用）",
    )
    # 老写法保留：--no-plan == --mode vision，-i == --mode interactive
    tsk.add_argument("--no-plan", action="store_true", help="等价于 --mode vision")
    tsk.add_argument("-i", "--interactive", action="store_true",
                     help="等价于 --mode interactive")

    out = p.add_argument_group("产物")
    out.add_argument("--task-out", default=os.path.join(OUTPUT_DIR, "task_from_vision.json"))
    out.add_argument("--traj-out", default=os.path.join(OUTPUT_DIR, "traj_vision.json"))
    out.add_argument("--plan-out", default=os.path.join(OUTPUT_DIR, "plan_vision.json"))

    run = p.add_argument_group("执行（默认取 real_config.py）")
    run.add_argument("--execute", dest="execute", action=argparse.BooleanOptionalAction,
                     default=EXECUTE_ON_REAL,
                     help="真的驱动机械臂；--no-execute 就是 dry-run 只打印")
    run.add_argument(
        "--continuous", dest="continuous", action=argparse.BooleanOptionalAction,
        default=CONTINUOUS_PATH,
        help="整段路径连续下发，动作不再一点一停；--no-continuous 改回逐点阻塞（慢但每步确认到位）",
    )
    run.add_argument("--speed", type=int, default=DEFAULT_MOVE_SPEED)
    run.add_argument("--port", default=None)
    run.add_argument("--min-delta-deg", type=float, default=1.0)
    run.add_argument("--relaxed-limits", action="store_true",
                     help="规划时用仿真放宽后的关节范围（结果通常不能下发真机）")
    run.add_argument("--stop-on-fail", action="store_true")
    run.add_argument(
        "--home-first", dest="home_first", action=argparse.BooleanOptionalAction,
        default=not START_FROM_PARK,
        help="抓取前先回仿真 HOME。默认从避让位直接接，不必绕回去",
    )
    run.add_argument("--no-home-after", action="store_true")
    run.add_argument(
        "--park", dest="park", action=argparse.BooleanOptionalAction,
        default=PARK_BEFORE_VISION,
        help="拍照前退到避让位；--no-park 关掉（臂在画面里会被当成零件）",
    )
    run.add_argument(
        "--servo", dest="servo", action=argparse.BooleanOptionalAction,
        default=USE_SERVO_J,
        help="用 servo_j 在线流式喂帧（动作最连贯）；--no-servo 退回 move_j",
    )
    run.add_argument(
        "--release-mode", choices=RELEASE_MODES, default=RELEASE_MODE_ON_CLOSE,
        help="断开时的刹车档：stop=完全松开(会沉) / brake=阻尼保持(不耗电) / "
             f"hold=主动顶住(耗电发热)；默认 {RELEASE_MODE_ON_CLOSE}",
    )
    return p


class ArmSession:
    """整个流程共用一条真机连接，中途绝不断开。

    为什么必须这样：电机靠位置模式（0x0A）通电顶住才不下垂，而这个状态是
    「连着」才有的。``close_connection`` 本身不发任何释放指令，但一旦关掉串口、
    进程退出，调试板就没有主机再刷位置帧了，重力立刻把臂带下去。
    所以 ``RELEASE_MODE_ON_CLOSE`` 设成 "hold" 也救不回来——那条 set_motor_mode
    在 move_j 之后本来就是 no-op（电机早就在 0x0A），掉的是断开本身。

    以前是「连上 → 挪到避让位 → 断开 → 去识别」，断开那一下臂就沉了。
    现在避让、识别、执行全程共用这一条连接，只在最后收尾时才断。
    """

    def __init__(self, args, cfg) -> None:
        self.args = args
        self.cfg = cfg
        self._arm: RealArm | None = None
        self.holding = False

    def get(self) -> RealArm:
        """拿到已连接的臂；第一次调用才真正连。"""
        if self._arm is None:
            arm = RealArm(
                self.cfg,
                dry_run=False,
                speed=self.args.speed,
                port=self.args.port,
                stream_path=self.args.continuous,
                release_mode=self.args.release_mode,
                use_servo=self.args.servo,
            )
            arm.connect()
            self._arm = arm
        return self._arm

    @property
    def connected(self) -> bool:
        return self._arm is not None

    def estop(self) -> None:
        if self._arm is not None:
            try:
                self._arm.estop()
            except Exception as e:
                print(f"[real] 急停失败：{e}")

    def close(self) -> None:
        if self._arm is None:
            return
        if HOME_ON_EXIT and getattr(self.args, "mode", None) != "zero":
            # 避让位是悬在半空的，断开后关节只剩刹车档能保持的力气。
            # 先送回静置位再张爪：路上还夹着件不会半路掉，到位后松开。
            try:
                print("[real] 退出前送回静置位并松开夹爪")
                self._arm.go_exit_home(open_jaw=True)
            except Exception as e:
                print(f"[real] 回静置位失败（就地断开）：{e}")
        else:
            try:
                self._arm.open_jaw(note="(退出松开)")
            except Exception as e:
                print(f"[real] 退出松爪失败：{e}")
        print(
            f"[real] 收尾断开（刹车档 {self.args.release_mode}）。"
            "注意：断开后板子不再刷位置帧，臂会松到刹车档能保持的程度，"
            "手上有件、或臂悬在半空时先扶一下。"
        )
        try:
            self._arm.close()
        finally:
            self._arm = None

    def __enter__(self) -> "ArmSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # 出异常时不要再让臂走位，先急停就地停住
            self.estop()
            print("[real] 异常退出，跳过回 HOME")
            try:
                self._arm.close() if self._arm is not None else None
            finally:
                self._arm = None
            return
        self.close()


def park_before_vision(args, cfg, session: ArmSession) -> None:
    """拍照前把臂退到相机拍不到的位置。

    机械臂停在画面里会被 YOLO 当成零件。避让只挪臂，桌面上的零件还在，
    所以识别的是零件不是空手。``vision`` 模式同样要先避让，否则拍到的还是臂。
    离线帧 / 复用上次识别 不连真机。

    挪完**不断开**：连接交给 session 一直持着，不然臂会当场沉下去。
    """
    if args.reuse_vision or getattr(args, "frame_dir", "") or not args.park:
        return
    print("[real] 识别前先退到避让位（相机拍不到机械臂，桌面零件仍可见）")
    session.get().go_park()
    time.sleep(PARK_SETTLE_S)
    print("[real] 避让位保持中，识别/规划期间主进程只续帧，不再猛拉或急坠")


def run_vision(args) -> tuple[dict, object | None]:
    """拿到 annotations（现拍或复用上次的）。"""
    sample_dir = os.path.join(os.path.abspath(args.vision_root), args.sample_id)
    ann_path = os.path.join(sample_dir, "annotations.json")

    if args.reuse_vision:
        if not os.path.isfile(ann_path):
            raise VisionError(f"--reuse-vision 但找不到 {ann_path}")
        with open(ann_path, "r", encoding="utf-8-sig") as f:
            ann = json.load(f)
        print(f"[vision] 复用上次识别结果 {ann_path}")
        return ann, None

    if getattr(args, "frame_dir", ""):
        w2c = load_handeye(args.handeye or None)
        print(f"[vision] 手眼 {describe_handeye(w2c)}")
        frame, vote_frames = capture_frame(args)
        ann = detect_to_annotations(
            frame,
            w2c,
            weights=args.weights or None,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            vote_frames=vote_frames,
        )
        print_objects(ann)
        saved = save_annotations(ann, args.vision_root, args.sample_id, frame=frame)
        print(f"[vision] annotations -> {saved}")
        prev = save_preview(ann, frame, os.path.join(sample_dir, "detect_preview.jpg"))
        print(f"[vision] 预览图 -> {prev}（先看一眼框和坐标对不对，再往下走）")
        return ann, frame

    # 连相机识别会占很久 GIL（YOLO）。放进子进程，主进程才能继续给避让位续帧，
    # 否则臂会沉下去，下一帧再把避让位打回去就是猛抬、看门狗再猛坠。
    print("[vision] 识别放到子进程，主进程继续给避让位续帧")
    script = os.path.join(REAL_DIR, "vision_to_annotations.py")
    cmd = [
        sys.executable,
        script,
        "--sample-id",
        str(args.sample_id),
        "--out-root",
        os.path.abspath(args.vision_root),
        "--resolution",
        str(args.resolution),
        "--frames",
        str(args.frames),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--imgsz",
        str(args.imgsz),
    ]
    if getattr(args, "weights", ""):
        cmd.extend(["--weights", args.weights])
    if getattr(args, "handeye", ""):
        cmd.extend(["--handeye", args.handeye])
    if getattr(args, "serial", None):
        cmd.extend(["--serial", args.serial])
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode not in (0, 4):
        raise VisionError(f"识别子进程退出码 {result.returncode}")
    if not os.path.isfile(ann_path):
        raise VisionError(f"识别子进程没写出 {ann_path}")
    with open(ann_path, "r", encoding="utf-8-sig") as f:
        ann = json.load(f)
    print_objects(ann)
    prev = os.path.join(sample_dir, "detect_preview.jpg")
    if os.path.isfile(prev):
        print(f"[vision] 预览图 -> {prev}（先看一眼框和坐标对不对，再往下走）")
    return ann, None


def plan_and_run(args, cfg, ann, targets, session, *, task_path: str = "") -> int:
    """选定目标 → 生成任务 JSON → 仿真规划 → 限位适配 → dry-run/下发。

    抽出来是为了交互模式能对同一份识别结果反复调用，不用重启脚本。
    """
    if not task_path:
        if args.pick_only:
            cmd = "把" + "、".join(str(o.get("class")) for o in targets) + "抓起来"
        elif len(targets) == len(ann.get("objects") or []):
            cmd = "把所有零件放进收纳盒"
        else:
            cmd = "、".join(str(o.get("class")) for o in targets)
        payload = build_task(
            ann, targets, args.dest, args.destination, cmd, pick_only=args.pick_only
        )
        if args.pick_only:
            print("[real] --pick-only：仿真不建收纳盒，抓起来抬到悬停高度就结束")
        task_path = os.path.abspath(args.task_out)
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(
            f"[real] 任务 JSON -> {task_path}（{len(targets)} 个目标"
            + ("，单进程连续抓放）" if len(targets) > 1 else "）")
        )

    try:
        traj_path = export_trajectory(
            task_path,
            args.traj_out,
            scene="standard",
            sample_id=args.sample_id,
            dataset_root=args.vision_root,
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

    if args.execute:
        # 复用 session 的连接：每条指令都断开重连的话，断开的那一下臂就沉了
        arm = session.get()
        try:
            summary = execute_plan(
                plan,
                arm,
                go_home_first=args.home_first,
                go_home_after=not args.no_home_after and not args.park,
                park_after=args.park,
                stop_on_fail=args.stop_on_fail,
            )
        except (RealArmError, RuntimeError) as e:
            print(f"[real] 执行中止：{e}")
            session.estop()
            return 1
    else:
        # dry-run 不碰硬件，单独开一个虚拟臂，不影响 session 的连接
        arm = RealArm(
            cfg,
            dry_run=True,
            speed=args.speed,
            port=args.port,
            stream_path=args.continuous,
            release_mode=args.release_mode,
            use_servo=args.servo,
        )
        if (not args.home_first) and args.park:
            arm.set_virtual_q(PARK_CONF)
        with arm:
            summary = execute_plan(
                plan,
                arm,
                go_home_first=args.home_first,
                go_home_after=not args.no_home_after and not args.park,
                park_after=args.park,
                stop_on_fail=args.stop_on_fail,
            )
        print(
            "[real] 以上是 dry-run，没有下发任何指令。\n"
            "[real] 先确认：预览图里的框对不对、世界坐标像不像、有没有限位告警；"
            "再把 real_config.py 的 EXECUTE_ON_REAL 改成 True。"
        )
    if args.execute and summary["ok"] == summary["total"] and summary["total"] > 0:
        session.holding = True
    return 0 if summary["ok"] == summary["total"] else 4


def _release_held_before_next(session: ArmSession, args) -> None:
    """上一项抓完还夹着时，终端输入了下一项才到料框停顿松爪。"""
    if not session.holding:
        return
    if not args.execute:
        print("[real] dry-run：下一项任务前会先到放置位停顿再松爪")
        session.holding = False
        return
    delay = max(0.0, float(PARK_RELEASE_DELAY_S))
    print(
        f"[real] 上一项还夹着，先到放置位（料框）停 {delay:.1f}s 再松爪，然后抓下一个"
    )
    release_at_place(session.get(), delay_s=delay, park_after=False)
    session.holding = False


_INTERACTIVE_HELP = """\
[real] 交互模式。直接敲零件名就抓，不用再退出重跑：
         螺丝刀            抓这个（支持中文片段 / 英文类名 / id:3）
         螺丝刀 扳手       依次抓多个
         all               抓识别到的全部
         look              重新拍照识别（桌面动过就用这个）
         list              列出当前识别到的零件
         q                 退出
       抓完夹着停在避让位；输入下一项（或 look）时才到料框停 3s 松爪。"""


def interactive_loop(args, cfg, ann, session) -> int:
    """识别一次，然后循环收指令。每条指令直接走规划+执行。"""
    print(_INTERACTIVE_HELP)
    last = 0
    while True:
        try:
            # 管道喂进来的首行可能带 BOM，会让 "list" 匹配不上
            line = input("\n[real] 抓什么> ").strip().lstrip("\ufeff").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[real] 退出交互模式。")
            return last
        if not line:
            continue
        low = line.lower()
        if low in ("q", "quit", "exit", "退出"):
            print("[real] 退出交互模式。" + ("臂会先回静置位、松开夹爪再断开。" if HOME_ON_EXIT else "会先松开夹爪再断开。"))
            return last
        if low in ("h", "help", "?", "帮助"):
            print(_INTERACTIVE_HELP)
            continue
        if low in ("list", "ls", "列表"):
            print_objects(ann)
            continue
        if low in ("look", "重拍", "重新识别"):
            from d405_camera import CameraError

            try:
                _release_held_before_next(session, args)
            except (RealArmError, RuntimeError) as e:
                print(f"[real] 重拍前松爪失败：{e}")
                session.estop()
                last = 1
                continue
            try:
                park_before_vision(args, cfg, session)
                ann, _frame = run_vision(args)
            except (VisionConfigError, VisionError, CameraError) as e:
                print(f"[vision] 重新识别失败：{e}")
                continue
            if not ann.get("objects"):
                print("[real] 这一帧没识别到可用零件。")
            continue

        want_all = low in ("all", "全部", "所有")
        tokens = [] if want_all else line.split()
        try:
            targets = select_targets(ann, tokens, want_all)
        except VisionError as e:
            print(f"[real] {e}")
            continue
        if not targets:
            print("[real] 没选中任何零件。")
            continue
        print("[real] 本次目标：" + "、".join(_describe(o) for o in targets))
        try:
            _release_held_before_next(session, args)
        except (RealArmError, RuntimeError) as e:
            print(f"[real] 下一项任务前松爪失败：{e}")
            session.estop()
            last = 1
            continue
        last = plan_and_run(args, cfg, ann, targets, session)
        # 执行完臂停在避让位，桌面没动的话可以直接接下一条指令
        if last != 0:
            print(f"[real] 上一条指令没有完全成功（code={last}），可以重试或先 look。")


def resolve_mode(args) -> str:
    """老写法（--no-plan / -i）映射到 --mode，并处理 --pick 的缺省。

    --pick 的 argparse 默认是 None（而不是 []），这样才能区分「命令行没给」
    和「命令行明确给了空」，前者回落到 real_config.PICK_TARGETS。
    """
    if args.no_plan:
        args.mode = "vision"
    elif args.interactive:
        args.mode = "interactive"
    if args.pick is None:
        args.pick = list(PICK_TARGETS)
    # once 模式没指定抓什么就抓全部
    if args.mode == "once" and not args.pick and not args.task:
        args.all = True
    return args.mode


def main() -> int:
    args = build_parser().parse_args()
    mode = resolve_mode(args)
    import real_config as _rc

    print(f"[real] 读到配置 {_rc.__file__}  RUN_MODE={_rc.RUN_MODE!r} → 本次模式={mode}")
    print(
        f"[real] 模式={mode} 下发={'是' if args.execute else '否(dry-run)'} "
        f"连续={'是' if args.continuous else '否'} 只抓={'是' if args.pick_only else '否'} "
        f"避让位={'是' if args.park else '否'} 速度={args.speed} "
        f"刹车={args.release_mode} 运动={'servo_j' if args.servo else 'move_j'}"
        "  ← 这些默认值在 real_config.py 的「运行模式」一节"
    )

    try:
        cfg = load_real_config()
    except RealConfigError as e:
        print(f"[real] 配置错误：{e}")
        return 2
    print(f"[real] robot.cfg = {cfg.cfg_path}")
    print(
        "[real] 软限位(°) "
        + " ".join(f"J{i + 1}[{lo:.0f},{hi:.0f}]" for i, (lo, hi) in enumerate(cfg.limits_deg()))
    )

    from d405_camera import CameraError

    if mode == "zero":
        if not args.execute:
            print("[real] dry-run：会把臂送到电机零位 [0, 0, 0, 0, 0, 0]°，这次没下发")
            return 0
        print("[real] 送到电机零位（全 0°），不识别、不规划")
        with ArmSession(args, cfg) as session:
            session.get().go_zero(open_jaw=True)
            print("[real] 已在零位")
        return 0

    # 一条连接贯穿全程：避让 → 识别 → 执行都用它，只在收尾时断开。
    # 中途断开会让电机松掉，臂当场沉下去。
    with ArmSession(args, cfg) as session:
        try:
            park_before_vision(args, cfg, session)
            ann, _frame = run_vision(args)
        except (VisionConfigError, VisionError, CameraError) as e:
            print(f"[vision] 失败：{e}")
            return 1

        if not ann.get("objects"):
            print("[real] 没识别到可用零件，不往下走。")
            return 4

        approx = [o for o in ann["objects"] if o.get("stl_is_approx")]
        if approx:
            print(
                "[real] 提醒：以下零件用的是近似 STL（权重里的泛类），抓取宽度可能有偏差："
                + "、".join(f"{o['class']}({o['yolo_class']})" for o in approx)
            )

        if mode == "vision":
            print("[real] 模式 vision：只做识别，已停在这里。看一眼预览图再决定下一步。")
            return 0

        if mode == "interactive":
            return interactive_loop(args, cfg, ann, session)

        if args.task:
            return plan_and_run(args, cfg, ann, [], session, task_path=args.task)

        try:
            targets = select_targets(ann, args.pick, args.all)
        except VisionError as e:
            print(f"[real] {e}")
            return 1
        return plan_and_run(args, cfg, ann, targets, session)


if __name__ == "__main__":
    sys.exit(main())
