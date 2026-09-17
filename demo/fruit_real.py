# -*- coding: utf-8 -*-
"""水果真机入口：眼在手上识别 + WRS 仿真规划器 + servo_j 下发。

用法（仓库根目录）::

    python demo/fruit_real.py --mode look
    python demo/fruit_real.py --mode zero
    python demo/fruit_real.py --mode bin_a
    python demo/fruit_real.py --mode vision
    python demo/fruit_real.py --mode once --cmd "把苹果放进A框"

默认 RUN_MODE / 是否真下发看 ``fruit_config.py``。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from fruit_cmd import cmd_quota, describe_cmd, parse_fruit_cmd
from fruit_config import (
    CAMERA_RESOLUTION,
    DEFAULT_CMD,
    DEFAULT_MOVE_SPEED,
    EXECUTE_ON_REAL,
    HOME_ON_EXIT,
    JAW_OPEN_ON_LOOK,
    JOG_AXIS,
    JOG_STEP_M,
    JOG_ZH,
    WANDER_STEPS,
    LOOK_SETTLE_S,
    LOOK_SHOW_LIVE,
    MAX_MOVE_SPEED,
    MAX_PICK_ROUNDS,
    OUTPUT_DIR,
    RECAPTURE_EACH_FRUIT,
    REAL_DIR,
    RELEASE_MODE_ON_CLOSE,
    RELEASE_MODES,
    REPO_ROOT,
    RUN_MODE,
    RUN_MODES,
    SKIP_RETRY_RADIUS_M,
    SNAPSHOT_FRAMES,
    USE_SERVO_J,
    FruitConfigError,
    bin_a_conf,
    bin_b_conf,
    slot_left_conf,
    slot_right_conf,
    conf_is_filled,
    dest_label,
    look_conf,
    BIN_A_CONF_DEG,
    BIN_B_CONF_DEG,
    LOOK_CONF_DEG,
    SLOT_LEFT_CONF_DEG,
    SLOT_RIGHT_CONF_DEG,
    FRUIT_ZH,
)
from fruit_sim_planner import FruitPlanError, plan_fruits, translate_tcp, warmup_planner_async
from fruit_tts import announce_action, announce_segment, speak
from fruit_vision import FruitVisionError, capture_and_detect, new_sample_id, warmup_yolo_async

if REAL_DIR not in sys.path:
    sys.path.insert(0, REAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from real_arm import RealArm, RealArmError, execute_plan  # noqa: E402
from real_config import GripperMap, RealConfigError, load_real_config  # noqa: E402
from traj_adapter import adapt, format_plan, save_plan  # noqa: E402


def select_fruits(ann: dict, cmd: dict) -> list[dict]:
    objs = sorted(
        list(ann.get("objects") or []),
        key=lambda o: float(o.get("score") or 0),
        reverse=True,
    )
    if not objs:
        return []
    counts = dict(cmd.get("counts") or {})
    n = cmd.get("count")
    want = list(cmd.get("fruits") or [])
    if counts:
        for en in counts:
            if en not in want:
                want.append(en)
    if cmd.get("all"):
        pool = list(objs)
    elif want:
        pool = [o for o in objs if o.get("yolo_class") in set(want)]
    else:
        return []
    if counts:
        used = {en: 0 for en in counts}
        picked: list[dict] = []
        for o in pool:
            en = o.get("yolo_class")
            if en in used and used[en] < counts[en]:
                picked.append(o)
                used[en] += 1
        for o in pool:
            en = o.get("yolo_class")
            if o in picked:
                continue
            if en in want and en not in counts:
                picked.append(o)
        short = [f"{FRUIT_ZH.get(en, en)}要{need}个只看到{used[en]}个" for en, need in counts.items() if used[en] < need]
        if short:
            print("[cmd] " + "；".join(short) + "，有多少抓多少")
        return picked
    if n:
        hit = pool[: int(n)]
        if len(hit) < int(n):
            print(f"[cmd] 只要 {int(n)} 个，画面里这类只有 {len(hit)} 个")
        return hit
    return pool


def connect_arm(cfg, *, execute: bool, speed: int, port=None, logger=print) -> RealArm:
    arm = RealArm(
        cfg,
        dry_run=not execute,
        speed=max(1, min(int(speed), MAX_MOVE_SPEED)),
        port=port,
        logger=logger,
        release_mode=RELEASE_MODE_ON_CLOSE,
        use_servo=USE_SERVO_J,
    )
    arm.connect()
    warmup_planner_async(logger=logger)
    warmup_yolo_async(logger=logger)
    return arm


def go_named(arm: RealArm, name: str) -> None:
    if name == "look":
        arm.go_conf(look_conf(), label="观察位", open_jaw=JAW_OPEN_ON_LOOK)
        arm.hold_here("观察位")
    elif name == "zero":
        # 用 go_zero 而不是自己拼 go_conf：它到位后会把实测角和残差打出来，
        # 不然只能凭感觉判断有没有真的回到零位。
        arm.go_zero(open_jaw=True)
        arm.hold_here("零位")
    elif name == "bin_a":
        arm.go_conf(bin_a_conf(), label="A 收纳筐", open_jaw=False)
        arm.hold_here("A 收纳筐")
    elif name == "bin_b":
        arm.go_conf(bin_b_conf(), label="B 收纳筐", open_jaw=False)
        arm.hold_here("B 收纳筐")
    elif name == "slot_left":
        arm.go_conf(slot_left_conf(), label="左边空位", open_jaw=False)
        arm.hold_here("左边空位")
    elif name == "slot_right":
        arm.go_conf(slot_right_conf(), label="右边空位", open_jaw=False)
        arm.hold_here("右边空位")
    else:
        raise FruitConfigError(f"未知位形 {name}")


def _jog_delta(name: str) -> np.ndarray:
    axis = np.asarray(JOG_AXIS[name], dtype=float)
    return axis * float(JOG_STEP_M)


def run_motion(arm: RealArm, cmd: dict, *, execute: bool) -> dict:
    """点动 / 自己晃几下。不拍照、不抓水果。"""
    import random

    if cmd.get("action") == "wander":
        speak("我自己动一动")
        dirs = [random.choice(list(JOG_AXIS)) for _ in range(int(WANDER_STEPS))]
    else:
        dirs = list(cmd.get("dirs") or [])
        if not dirs:
            speak("没听清往哪边")
            raise FruitPlanError("没听清往哪边")
        speak("收到指令，" + describe_cmd(cmd))

    q = np.asarray(arm.joint_values(), dtype=float)
    moved = 0
    for name in dirs:
        zh = JOG_ZH.get(name, name)
        speak(f"往{zh}")
        print(f"[jog] 往{zh}  {float(JOG_STEP_M)*100:.0f}cm")
        if not execute:
            moved += 1
            continue
        nxt = translate_tcp(q, _jog_delta(name))
        if nxt is None and cmd.get("action") == "wander":
            order = list(JOG_AXIS)
            random.shuffle(order)
            for alt in order:
                if alt == name:
                    continue
                nxt = translate_tcp(q, _jog_delta(alt))
                if nxt is not None:
                    zh = JOG_ZH.get(alt, alt)
                    speak(f"往{zh}")
                    name = alt
                    break
        if nxt is None:
            speak("这个方向够不着")
            print(f"[jog] 往{zh} IK 无解或已到边")
            continue
        arm.go_conf(nxt, label=f"往{zh}", open_jaw=False)
        q = np.asarray(nxt, dtype=float)
        moved += 1
    if moved:
        speak("动完了")
    else:
        speak("这边动不了")
        raise FruitPlanError("点动失败")
    return {"ok": moved, "total": len(dirs), "ann": None, "preview": None}


def run_vision(arm: RealArm, *, sample_id=None, serial=None, grab=None) -> tuple[dict, object]:
    speak("正在去观察位拍照")
    go_named(arm, "look")
    time.sleep(LOOK_SETTLE_S)
    q = arm.joint_values()
    print(f"[vision] 当前关节(°) {[round(np.degrees(v), 2) for v in q]}")
    speak("正在进行识别")
    sample_id = sample_id or new_sample_id()
    ann, _frame, preview = capture_and_detect(
        q,
        sample_id=sample_id,
        serial=serial,
        frames=SNAPSHOT_FRAMES,
        grab=grab,
    )
    for obj in ann.get("objects") or []:
        p = obj["pose_6d"]
        print(
            f"  [{obj['id']}] {obj['class']}({obj['yolo_class']}) "
            f"xyz=({p['x']:+.3f},{p['y']:+.3f},{p['z']:+.3f}) score={obj['score']:.2f}"
        )
    objs = ann.get("objects") or []
    if not objs:
        print("[vision] 没看到香蕉/苹果/橙子")
        speak("没有看到香蕉、苹果或橙子")
    else:
        names = [str(o.get("class") or "水果") for o in objs]
        counts: dict[str, int] = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        parts = [f"{n}个{k}" if n > 1 else k for k, n in counts.items()]
        speak("看到" + "、".join(parts))
    return ann, preview


def _plan_and_run(arm: RealArm, cfg, targets: list[dict], dest, *, execute: bool) -> dict:
    speak("正在规划动作")
    q0 = arm.joint_values()
    traj = plan_fruits(targets, dest=dest, start_q=q0)
    gmap = GripperMap.from_config(cfg)
    plan = adapt(traj, cfg, gmap)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plan_path = save_plan(plan, os.path.join(OUTPUT_DIR, "plan.json"))
    print(format_plan(plan))
    print(f"[real] plan -> {plan_path}")
    if plan.get("errors"):
        speak("规划失败，没有可执行的路径")
        raise RealArmError("plan 有致命问题，未下发")
    announce_action(targets, dest, execute=execute)
    if not execute:
        print("[real] dry-run，没有下发。fruit_config.EXECUTE_ON_REAL=True 或勾界面后再跑。")
        return {"ok": 0, "total": 0, "dry_run": True, "plan": plan}
    summary = execute_plan(
        plan,
        arm,
        go_home_first=False,
        go_home_after=False,
        park_after=False,
        place_after=False,
        stop_on_fail=False,
        on_segment=lambda seg: announce_segment(seg, dest),
    )
    ok = int(summary.get("ok") or 0)
    total = int(summary.get("total") or 0)
    if dest and ok:
        where = dest_label(dest)
        speak(f"已经放到{where}" if ok == total else f"放到{where}完成了{ok}个")
    elif ok:
        speak("抓取完成")
    return {**summary, "plan": plan}


def _obj_xy(obj: dict) -> tuple[float, float]:
    p = obj.get("pose_6d") or {}
    return float(p.get("x", 0.0)), float(p.get("y", 0.0))


def _near_any(obj: dict, spots: list[tuple[float, float]]) -> bool:
    x, y = _obj_xy(obj)
    return any(math.hypot(x - sx, y - sy) <= SKIP_RETRY_RADIUS_M for sx, sy in spots)


def _back_to_look(arm: RealArm) -> None:
    """离开料框先竖直抬高，再去观察位，避免关节直连扫框沿。"""
    try:
        from fruit_sim_planner import exit_bin_z_m, raise_conf

        q = arm.joint_values()
        up = raise_conf(q, exit_bin_z_m())
        if up is not None:
            speak("正在离开料框")
            arm.go_conf(up, label="抬出料框", open_jaw=True)
        speak("正在回观察位")
        go_named(arm, "look")
    except Exception as e:
        print(f"[real] 回观察位失败：{e}")
        try:
            go_named(arm, "look")
        except Exception as e2:
            print(f"[real] 回观察位仍失败：{e2}")


def run_task(
    arm: RealArm,
    cfg,
    cmd_text: str,
    ann: dict | None = None,
    *,
    execute: bool,
    recapture: bool | None = None,
    grab=None,
    on_preview=None,
) -> dict:
    cmd = parse_fruit_cmd(cmd_text)
    if cmd.get("action") in ("jog", "wander"):
        return run_motion(arm, cmd, execute=execute)
    if not cmd.get("all") and not (cmd.get("fruits") or cmd.get("counts")):
        speak("没听清是橙子、苹果还是香蕉，请再说一遍")
        raise FruitPlanError("没听清水果种类")
    print(f"[cmd] {describe_cmd(cmd)}")
    speak("收到指令，" + describe_cmd(cmd))
    dest = cmd.get("dest") if cmd.get("action") == "pick_place" else None
    if dest == "LEFT" and not conf_is_filled(SLOT_LEFT_CONF_DEG):
        speak("还没示教左边空位")
        raise FruitConfigError("还没填 SLOT_LEFT_CONF_DEG（左边空位）")
    if dest == "RIGHT" and not conf_is_filled(SLOT_RIGHT_CONF_DEG):
        speak("还没示教右边空位")
        raise FruitConfigError("还没填 SLOT_RIGHT_CONF_DEG（右边空位）")
    recap = RECAPTURE_EACH_FRUIT if recapture is None else bool(recapture)
    preview = None

    def _see(photo):
        nonlocal preview
        preview = photo
        if on_preview is not None and photo is not None:
            on_preview(photo)

    quota = cmd_quota(cmd)
    if quota:
        print(f"[cmd] 数量上限 {quota}")

    if not recap:
        if ann is None:
            ann, preview = run_vision(arm, grab=grab)
            _see(preview)
        targets = select_fruits(ann, cmd)
        if not targets:
            speak("指令里的水果当前画面没有")
            raise FruitPlanError("指令里的水果当前画面没有")
        print(f"[real] 一次拍完批量做：{len(targets)} 个目标，按识别置信度顺序")
        out = _plan_and_run(arm, cfg, targets, dest, execute=execute)
        if execute:
            _back_to_look(arm)
        return {**out, "ann": ann, "preview": preview}

    # 逐个重拍：抓走一个就回观察位重新看，坐标永远是最新的
    if not execute:
        print("[real] dry-run 不会真把水果拿走，重拍会一直看到同一个，只演示第一个")
    ok = total = 0
    skip: list[tuple[float, float]] = []
    last_ann = None
    hit_limit = True
    for rnd in range(1, MAX_PICK_ROUNDS + 1):
        if quota is not None and ok >= quota:
            hit_limit = False
            break
        last_ann, preview = run_vision(arm, grab=grab)
        _see(preview)
        targets = [o for o in select_fruits(last_ann, cmd) if not _near_any(o, skip)]
        if quota is not None:
            targets = targets[: max(0, quota - ok)]
        if not targets:
            hit_limit = False
            break
        obj = targets[0]
        left = f"，指令还要 {quota - ok} 个" if quota is not None else ""
        print(f"[real] 第 {rnd} 轮：画面还剩 {len(targets)} 个{left}，这轮做 {obj.get('class')}")
        try:
            out = _plan_and_run(arm, cfg, [obj], dest, execute=execute)
        except (FruitPlanError, RealArmError) as e:
            print(f"[real] {obj.get('class')} 这轮跳过：{e}")
            speak(f"{obj.get('class')} 这轮做不了，先跳过")
            skip.append(_obj_xy(obj))
            continue
        if out.get("dry_run"):
            return {**out, "ann": last_ann, "preview": preview}
        total += int(out.get("total", 0))
        ok += int(out.get("ok", 0))
        if int(out.get("ok", 0)) < int(out.get("total", 0)):
            # 没抓走，下一轮还会看到它，记下来免得原地死循环
            skip.append(_obj_xy(obj))
    if hit_limit:
        print(f"[real] 到了 MAX_PICK_ROUNDS={MAX_PICK_ROUNDS} 上限，收工")
    if total == 0 and not skip:
        raise FruitPlanError("指令里的水果当前画面没有")
    print(f"[real] 逐个重拍收工：成功 {ok}/{total}，跳过 {len(skip)}")
    if execute:
        _back_to_look(arm)
    return {"ok": ok, "total": total, "ann": last_ann, "preview": preview}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="水果真机：观察位 / 零位 / A·B 框 / 识别 / 抓放")
    p.add_argument("--mode", default=RUN_MODE, choices=RUN_MODES)
    p.add_argument("--cmd", default="", help="once 模式的口令")
    p.add_argument("--execute", action="store_true", help="真下发（覆盖配置里的 False）")
    p.add_argument("--no-execute", action="store_true", help="只 dry-run")
    p.add_argument("--speed", type=int, default=DEFAULT_MOVE_SPEED)
    p.add_argument("--port", default=None)
    p.add_argument("--serial", default=None)
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="look / bin_a / bin_b 到位后不等回车，直接回静置位",
    )
    p.add_argument(
        "--recapture",
        action="store_true",
        help="每抓完一个就回观察位重拍（覆盖配置里的 RECAPTURE_EACH_FRUIT）",
    )
    p.add_argument("--no-recapture", action="store_true", help="一次拍完批量做")
    p.add_argument(
        "--live",
        action="store_true",
        help="look 到位后开实时 YOLO 画面（覆盖配置里的 LOOK_SHOW_LIVE）",
    )
    p.add_argument("--no-live", action="store_true", help="look 到位后只等回车")
    return p


def resolve_live(args) -> bool:
    if args.no_live:
        return False
    if args.live:
        return True
    return bool(LOOK_SHOW_LIVE)


def resolve_recapture(args) -> bool | None:
    if args.no_recapture:
        return False
    if args.recapture:
        return True
    return None


def show_live_at_look(arm: RealArm, args) -> bool:
    """在观察位开实时 YOLO 画面，关窗才回静置位。开不起来返回 False。

    复用已经连上的这条臂，所以窗口里按 j 读关节角不会再抢一次串口。
    """
    try:
        from d405_camera import CameraError, D405Camera, pyrealsense_available

        from fruit_yolo_realtime import live_yolo_preview
    except Exception as e:  # 缺 pyrealsense2 / ultralytics / Pillow 都算
        print(f"[look] 实时画面不可用：{e}")
        return False
    if not pyrealsense_available():
        print("[look] 没装 pyrealsense2，开不了实时画面：pip install pyrealsense2")
        return False
    try:
        with D405Camera(CAMERA_RESOLUTION, args.serial, logger=print) as cam:
            print("[look] 实时画面：q / Esc 关窗回静置位，j / 8 打印关节角")
            live_yolo_preview(cam, arm=arm)
        return True
    except (CameraError, FileNotFoundError, OSError) as e:
        print(f"[look] 实时画面打不开：{e}")
        return False


def wait_at_pose(args) -> None:
    if args.no_wait or not getattr(sys.stdin, "isatty", lambda: False)():
        return
    try:
        input("[fruit] 已到位，保持中。看完按回车回静置位…")
    except (EOFError, KeyboardInterrupt):
        print()


def resolve_execute(args) -> bool:
    if args.no_execute:
        return False
    if args.execute:
        return True
    return bool(EXECUTE_ON_REAL)


def main() -> int:
    args = build_parser().parse_args()
    execute = resolve_execute(args)
    recap = resolve_recapture(args)
    recap = RECAPTURE_EACH_FRUIT if recap is None else recap
    print(
        f"[fruit] RUN_MODE={args.mode}  execute={execute}  servo_j={USE_SERVO_J}  "
        f"多个水果={'逐个重拍' if recap else '一次拍完批量做'}"
    )
    print(
        "[fruit] 示教位 "
        f"观察={'已填' if conf_is_filled(LOOK_CONF_DEG) else '未填'} "
        f"A={'已填' if conf_is_filled(BIN_A_CONF_DEG) else '未填'} "
        f"B={'已填' if conf_is_filled(BIN_B_CONF_DEG) else '未填'} "
        f"左={'已填' if conf_is_filled(SLOT_LEFT_CONF_DEG) else '未填'} "
        f"右={'已填' if conf_is_filled(SLOT_RIGHT_CONF_DEG) else '未填'}"
    )
    try:
        cfg = load_real_config()
    except RealConfigError as e:
        print(f"[fruit] {e}")
        return 2

    arm = None
    try:
        if args.mode == "interactive":
            print("[fruit] interactive 请跑 python demo/fruit_demo_ui.py")
            return 0
        arm = connect_arm(cfg, execute=execute, speed=args.speed, port=args.port)
        if args.mode in ("look", "zero", "bin_a", "bin_b", "slot_left", "slot_right"):
            go_named(arm, args.mode)
            # 这几个模式就是为了站在那儿核对示教位 / 对相机。不停一下的话，
            # 下面的 finally 会立刻把臂送回静置位，根本来不及看。
            shown = False
            if args.mode == "look" and not args.no_wait and resolve_live(args):
                shown = show_live_at_look(arm, args)
            if not shown:
                wait_at_pose(args)
            return 0
        if args.mode == "vision":
            run_vision(arm, serial=args.serial)
            return 0
        if args.mode == "once":
            run_task(
                arm,
                cfg,
                args.cmd or DEFAULT_CMD,
                execute=execute,
                recapture=resolve_recapture(args),
            )
            return 0
        print(f"[fruit] 未知模式 {args.mode}")
        return 2
    except (FruitConfigError, FruitVisionError, FruitPlanError, RealArmError, ValueError) as e:
        print(f"[fruit] {e}")
        return 1
    finally:
        if arm is not None:
            if HOME_ON_EXIT and args.mode not in ("zero",):
                try:
                    from real_config import EXIT_HOME_CONF

                    arm.go_conf(EXIT_HOME_CONF, label="静置位", open_jaw=True)
                except Exception as e:
                    print(f"[fruit] 回静置位失败：{e}")
            arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
