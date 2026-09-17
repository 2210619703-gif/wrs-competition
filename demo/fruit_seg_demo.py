"""COCO YOLO26s-seg 水果分割演示：只保留香蕉 / 苹果 / 橙子。

权重默认本目录 ``yolo26s-seg.pt``（官方 80 类分割，水果只有这三种）。
推理时用 ``classes=[46,47,49]`` 丢掉人、杯子等其余类。

用法（仓库根目录或本目录均可）::

  python demo/fruit_seg_demo.py --image 某张图.jpg
  python demo/fruit_seg_demo.py --webcam
  python demo/fruit_seg_demo.py --realsense

按键（实时）: q 退出  s 保存  [ / ] 调置信度
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fruit_vision import put_labels_cn
DEFAULT_WEIGHTS = ROOT / "yolo26s-seg.pt"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt"
SAVE_DIR = ROOT / "outputs"

# COCO 80 类里仅有的三种水果
FRUIT_IDS = (46, 47, 49)
FRUIT_EN = ("banana", "apple", "orange")
FRUIT_ZH = {"banana": "香蕉", "apple": "苹果", "orange": "橙子"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO26s-seg 香蕉/苹果/橙子分割演示")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", type=Path, default=None, help="对一张图推理并保存")
    src.add_argument("--webcam", action="store_true", help="笔记本摄像头")
    src.add_argument("--realsense", action="store_true", help="Intel RealSense（优先 D405）")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="空=自动；GPU 填 0")
    p.add_argument("--out", type=Path, default=None, help="--image 时的保存路径")
    p.add_argument("--camera", type=int, default=0, help="OpenCV 摄像头编号")
    p.add_argument("--serial", default="", help="RealSense 序列号；空则自动选")
    return p.parse_args()


def ensure_weights(path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    print(f"[demo] 本地没有 {path.name}，正在下载官方 YOLO26s-seg …")
    from ultralytics import YOLO

    path.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        import os

        os.chdir(path.parent)
        YOLO(path.name)
    finally:
        os.chdir(cwd)
    if not path.is_file():
        raise FileNotFoundError(
            f"下载失败。请把 yolo26s-seg.pt 放到 {path.parent}，"
            f"或从 {WEIGHTS_URL} 自行下载。"
        )
    return path


def load_model(weights: Path, device: str):
    import torch
    from ultralytics import YOLO

    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[demo] 权重: {weights}")
    print(f"[demo] 设备: {device}  只检测: 香蕉/苹果/橙子")
    model = YOLO(str(weights))
    names = model.names or {}
    for cid, en in zip(FRUIT_IDS, FRUIT_EN):
        got = names.get(cid, "")
        if str(got) != en:
            print(f"[WARN] COCO id {cid} 期望 {en}，实际 {got!r}")
    return model, device


def predict_fruit(model, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int, device: str):
    return model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=list(FRUIT_IDS),
        verbose=False,
    )[0]


def draw_overlay(frame: np.ndarray, result, fps: float | None, conf: float) -> np.ndarray:
    vis = result.plot()
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    names = result.names or {}
    title = "fruit-seg  banana/apple/orange"
    if fps is not None:
        title += f"  {fps:.1f} FPS"
    title += f"  conf={conf:.2f}  n={n}"
    lines = [title]
    if boxes is not None and n:
        counts: dict[str, int] = {}
        for cls_id in boxes.cls.tolist():
            en = names.get(int(cls_id), str(int(cls_id)))
            zh = FRUIT_ZH.get(en, en)
            counts[zh] = counts.get(zh, 0) + 1
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"{name} x{cnt}" if cnt > 1 else name)
    y = 8
    labels = []
    for i, text in enumerate(lines):
        labels.append((12, y, text, (0, 255, 0) if i == 0 else (240, 240, 240)))
        y += 26
    return put_labels_cn(vis, labels, size=20)


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读图: {path}")
    return img


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"无法编码: {path}")
    buf.tofile(str(path))


def run_image(args: argparse.Namespace, model, device: str) -> None:
    img = imread_unicode(args.image)
    result = predict_fruit(model, img, args.conf, args.iou, args.imgsz, device)
    vis = draw_overlay(img, result, None, args.conf)
    out = args.out or (SAVE_DIR / f"{args.image.stem}_fruit.jpg")
    imwrite_unicode(out, vis)
    n = 0 if result.boxes is None else len(result.boxes)
    print(f"[demo] 检出 {n} 个水果，已保存 {out}")
    cv2.imshow("fruit-seg", vis)
    print("[demo] 按任意键关闭窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(args: argparse.Namespace, model, device: str) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"打不开摄像头 {args.camera}")
    print("[demo] 摄像头已开。q 退出  s 保存  [ / ] 调置信度")
    live_loop(args, model, device, frames=lambda: _webcam_frame(cap))
    cap.release()


def _webcam_frame(cap):
    ok, frame = cap.read()
    return frame if ok else None


def run_realsense(args: argparse.Namespace, model, device: str) -> None:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        raise SystemExit("没有检测到 Intel RealSense。")
    listed = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        listed.append((serial, name))
        print(f"[demo] 相机: {name}  SN={serial}")
    chosen = None
    if args.serial:
        chosen = next((item for item in listed if item[0] == args.serial), None)
        if chosen is None:
            raise SystemExit(f"找不到序列号 {args.serial}")
    else:
        chosen = next((item for item in listed if "D405" in item[1].upper()), listed[0])
    serial, name = chosen
    print(f"[demo] 使用: {name}  SN={serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    opened = False
    last_err = None
    for w, h, f in ((1280, 720, 30), (848, 480, 30), (640, 480, 30)):
        try:
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            pipeline.start(config)
            print(f"[demo] 已打开 {w}x{h}@{f}")
            opened = True
            break
        except Exception as exc:
            last_err = exc
            try:
                pipeline.stop()
            except Exception:
                pass
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
    if not opened:
        raise SystemExit(f"无法打开 RealSense 彩色流: {last_err}")

    def grab():
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    print("[demo] RealSense 已开。q 退出  s 保存  [ / ] 调置信度")
    try:
        live_loop(args, model, device, frames=grab)
    finally:
        pipeline.stop()


def live_loop(args: argparse.Namespace, model, device: str, frames) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fps = 0.0
    save_i = 0
    last_vis = None
    try:
        while True:
            frame = frames()
            if frame is None:
                continue
            result = predict_fruit(model, frame, args.conf, args.iou, args.imgsz, device)
            now = time.perf_counter()
            dt = now - t0
            t0 = now
            if dt > 1e-6:
                fps = 1.0 / dt
            last_vis = draw_overlay(frame, result, fps, args.conf)
            cv2.imshow("fruit-seg", last_vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("]"):
                args.conf = min(0.9, round(args.conf + 0.05, 2))
                print(f"[demo] conf={args.conf}")
            if key == ord("["):
                args.conf = max(0.05, round(args.conf - 0.05, 2))
                print(f"[demo] conf={args.conf}")
            if key == ord("s") and last_vis is not None:
                out = SAVE_DIR / f"live_{save_i:06d}.jpg"
                imwrite_unicode(out, last_vis)
                print(f"[demo] 已保存 {out}")
                save_i += 1
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    weights = ensure_weights(args.weights)
    model, device = load_model(weights, args.device)
    if args.image is not None:
        if not args.image.exists():
            raise FileNotFoundError(args.image)
        run_image(args, model, device)
        return
    if args.realsense:
        run_realsense(args, model, device)
        return
    run_webcam(args, model, device)


if __name__ == "__main__":
    main()
