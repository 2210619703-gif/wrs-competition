# -*- coding: utf-8 -*-
"""D405 实时预览，界面和按键对齐 ``tiaozhanbei/real/d405_camera.py``。

彩色图已经是 BGR，深度与彩色对齐。YOLO 只用本目录 ``yolo26s-seg.pt`` 的
香蕉 / 苹果 / 橙子（COCO 46/47/49）。这一步不连机械臂、不用手眼。

    python demo/fruit_yolo_realtime.py --live
    python demo/fruit_yolo_realtime.py --yolo
    python demo/fruit_yolo_realtime.py --live --yolo

不传参数时默认 ``--yolo``。按键和工业件 live 一样：

===========  ====================================================
q / Esc      退出
j / 8        打印当前真机关节角 + 法兰世界坐标
[ / ]        调绿色置信度阈值
- / =        调 YOLO 输入尺寸
s            保存当前画面
===========  ====================================================

这一步默认会连机械臂（只读关节，不下发）。窗口焦点在画面上时按 ``j``。
不用臂时加 ``--no-arm``。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from fruit_config import (
    CAMERA_RESOLUTION,
    DEMO_DIR as _DEMO_DIR,
    FRUIT_CLASS_IDS,
    FRUIT_ZH,
    REAL_DIR,
    REPO_ROOT,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
    YOLO_WEIGHTS,
)
from fruit_vision import format_conf_deg, load_yolo, put_labels_cn
from read_joints import connect_readonly, snapshot_text

if REAL_DIR not in sys.path:
    sys.path.insert(0, REAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from d405_camera import (  # noqa: E402
    RESOLUTIONS,
    CameraError,
    D405Camera,
    _colorize_depth,
    pyrealsense_available,
)


def print_joints(arm) -> None:
    if arm is None:
        print("[real] 没接机械臂。去掉 --no-arm 再开，或先 python demo/read_joints.py")
        return
    try:
        q = arm.joint_values()
        print("-" * 48)
        print("[real] " + snapshot_text(arm).replace("\n", "\n[real] "))
        print(f"[real] 可粘贴  LOOK_CONF_DEG       = {format_conf_deg(q)}")
        print(f"[real] 可粘贴  BIN_A_CONF_DEG      = {format_conf_deg(q)}")
        print(f"[real] 可粘贴  BIN_B_CONF_DEG      = {format_conf_deg(q)}")
        print(f"[real] 可粘贴  SLOT_LEFT_CONF_DEG  = {format_conf_deg(q)}")
        print(f"[real] 可粘贴  SLOT_RIGHT_CONF_DEG = {format_conf_deg(q)}")
    except Exception as e:
        print(f"[real] 读关节失败：{e}")


def live_preview(cam: D405Camera, arm=None) -> None:
    """彩色 + 深度并排，和 ``d405_camera.live_preview`` 同一套窗口。"""
    import cv2

    win = "D405 live  (q 退出  j 打印关节)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[cam] 实时预览已打开，按 q 或 Esc 关闭；j / 8 打印关节角")
    while True:
        frame = cam.capture(timeout_ms=2000)
        depth = _colorize_depth(frame.depth_m)
        view = np.hstack([frame.color, depth])
        valid = frame.valid_ratio() * 100.0
        cv2.putText(
            view,
            f"{frame.width}x{frame.height}  depth {valid:.0f}%",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("j"), ord("J"), ord("8")):
            print_joints(arm)
    cv2.destroyWindow(win)


def live_yolo_preview(
    cam: D405Camera,
    *,
    weights: str | None = None,
    conf: float = YOLO_CONF,
    iou: float = YOLO_IOU,
    imgsz: int = YOLO_IMGSZ,
    arm=None,
) -> None:
    """彩色实时画面 + 水果 YOLO。布局/按键抄 ``d405_camera.live_yolo_preview``。"""
    import cv2

    path = weights or YOLO_WEIGHTS
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到水果分割权重：{path}")
    print(f"[cam] 加载 YOLO {os.path.basename(path)} …")
    model = load_yolo(path)
    names = model.names or {}
    print(
        f"[cam] 任务={getattr(model, 'task', None)}  水果={list(FRUIT_CLASS_IDS)}  "
        f"conf={conf:.2f}  imgsz={imgsz}\n"
        "[cam] q 退出；j / 8 打印关节角；[ / ] 调阈值；- / = 调输入尺寸；s 保存"
    )
    win = "D405 + YOLO fruit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_print = 0.0
    floor = 0.08
    save_i = 0
    save_dir = os.path.join(_DEMO_DIR, "outputs", "live")
    while True:
        frame = cam.capture(timeout_ms=2000)
        t0 = time.perf_counter()
        result = model.predict(
            frame.color,
            conf=min(conf, floor),
            iou=iou,
            imgsz=imgsz,
            classes=list(FRUIT_CLASS_IDS),
            verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        view = frame.color.copy()
        boxes = result.boxes
        masks = result.masks
        n_all = 0 if boxes is None else len(boxes)
        n_pass = 0
        hits = []
        labels = []
        for i in range(n_all):
            x1, y1, x2, y2 = (int(v) for v in boxes[i].xyxy[0].cpu().numpy())
            score = float(boxes[i].conf)
            en = str(names.get(int(boxes[i].cls), int(boxes[i].cls)))
            if en not in FRUIT_ZH and en not in ("banana", "apple", "orange"):
                continue
            zh = FRUIT_ZH.get(en, en)
            passed = score >= conf
            n_pass += int(passed)
            color = (0, 220, 0) if passed else (0, 180, 255)
            if masks is not None and i < len(masks.data):
                m = masks.data[i].cpu().numpy()
                if m.shape[:2] != view.shape[:2]:
                    m = cv2.resize(m, (view.shape[1], view.shape[0]), interpolation=cv2.INTER_LINEAR)
                contours, _ = cv2.findContours(
                    (m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(view, contours, -1, color, 2)
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2 if passed else 1)
            labels.append((x1, max(2, y1 - 22), f"{zh} {score:.2f}", color))
            hits.append(f"{zh} {score:.2f}")
        view = put_labels_cn(view, labels)
        cv2.putText(
            view,
            f"{frame.width}x{frame.height}  n={n_pass}/{n_all}  conf={conf:.2f}  "
            f"imgsz={imgsz}  {infer_ms:.0f}ms",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        now = time.perf_counter()
        if hits and now - last_print > 0.8:
            print("[yolo] " + "  |  ".join(hits[:8]))
            last_print = now
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("["):
            conf = max(0.05, round(conf - 0.05, 2))
            print(f"[yolo] conf -> {conf:.2f}")
        elif key == ord("]"):
            conf = min(0.95, round(conf + 0.05, 2))
            print(f"[yolo] conf -> {conf:.2f}")
        elif key == ord("-"):
            imgsz = max(640, imgsz - 128)
            print(f"[yolo] imgsz -> {imgsz}")
        elif key in (ord("="), ord("+")):
            imgsz = min(1280, imgsz + 128)
            print(f"[yolo] imgsz -> {imgsz}")
        elif key == ord("s"):
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"live_{save_i:06d}.jpg")
            cv2.imwrite(out, view)
            print(f"[cam] 已保存 {out}")
            save_i += 1
        elif key in (ord("j"), ord("J"), ord("8")):
            print_joints(arm)
    cv2.destroyWindow(win)


def main() -> int:
    p = argparse.ArgumentParser(description="水果 D405 实时预览（对齐 d405_camera.py --live）")
    p.add_argument("--resolution", default=CAMERA_RESOLUTION, choices=tuple(RESOLUTIONS))
    p.add_argument("--serial", default=None)
    p.add_argument("--live", action="store_true", help="彩色+深度并排，按 q 退出")
    p.add_argument("--yolo", action="store_true", help="彩色上叠加水果 YOLO（隐含 --live）")
    p.add_argument("--weights", default="", help="默认 demo/yolo26s-seg.pt")
    p.add_argument("--conf", type=float, default=YOLO_CONF)
    p.add_argument("--iou", type=float, default=YOLO_IOU)
    p.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    p.add_argument("--port", default=None, help="机械臂串口；默认 robot.cfg")
    p.add_argument("--no-arm", action="store_true", help="不连机械臂（j 键无效）")
    args = p.parse_args()

    if not args.live and not args.yolo:
        args.yolo = True

    if not pyrealsense_available():
        print("[cam] 没装 pyrealsense2，无法连相机。pip install pyrealsense2")
        return 2

    arm = None
    if not args.no_arm:
        try:
            arm = connect_readonly(args.port)
            print("[real] 已连机械臂。窗口里按 j 或 8 打印当前关节角 / 法兰坐标")
        except Exception as e:
            print(f"[real] 机械臂未连接（相机照常开）：{e}")
            arm = None

    try:
        with D405Camera(args.resolution, args.serial, logger=print) as cam:
            if args.yolo:
                live_yolo_preview(
                    cam,
                    weights=args.weights or None,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    arm=arm,
                )
                return 0
            live_preview(cam, arm=arm)
            return 0
    except FileNotFoundError as e:
        print(f"[cam] {e}")
        return 2
    except CameraError as e:
        print(f"[cam] 失败：{e}")
        return 1
    except KeyboardInterrupt:
        print("\n[cam] 已退出")
        return 0
    finally:
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
