"""RealSense D405 实时分割检测（实机 113 类 YOLO-seg）。

本脚本只用实机权重，不要用仿真 `yolo26s-seg-fine.pt`（124 中文类）。

用法（conda activate wrs）:
  python live_detect_realsense_d405.py
  python live_detect_realsense_d405.py --weights best.pt --conf 0.35
  python live_detect_realsense_d405.py --width 848 --height 480

按键:
  q / Esc  退出
  s        保存当前标注图到 output/live_d405/
  d        开/关深度窗
  [ / ]    降低 / 提高置信度
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "output" / "live_d405"

def _looks_like_real_classes(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "acorn_nuts" in text and "binding_screws" in text


def real_weight_candidates() -> list[Path]:
    # 本目录 best.pt 是 113 类实机权重（train-real-merged-seg-2）
    paths: list[Path] = []
    local = ROOT / "best.pt"
    if local.is_file() and _looks_like_real_classes(ROOT / "classes.txt"):
        paths.append(local)
    paths.extend(
        [
            ROOT / "runs" / "segment" / "train-real-merged-seg-2" / "weights" / "best.pt",
        ]
    )
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RealSense D405 + 实机 YOLO-seg 实时检测")
    p.add_argument("--weights", type=Path, default=None, help="实机 YOLO-seg 权重")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="空=自动；GPU 填 0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", default="", help="指定相机序列号；空则优先 D405")
    p.add_argument("--skip", type=int, default=0, help="每隔 N 帧再推理一次，CPU 可设 1 或 2")
    return p.parse_args()


def resolve_weights(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else ROOT / explicit
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for path in real_weight_candidates():
        if path.is_file():
            return path
    raise FileNotFoundError(
        "找不到实机权重。请把 113 类 best.pt 放到本目录，或用 --weights 指定。"
    )


def warn_if_sim_weights(model) -> None:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        sample = " ".join(str(names[i]) for i in range(min(3, len(names))))
        n = len(names)
    else:
        sample = " ".join(str(x) for x in list(names)[:3])
        n = len(names)
    if n == 124 or any("\u4e00" <= ch <= "\u9fff" for ch in sample):
        print(
            f"[WARN] 当前权重像仿真 124 中文类（nc={n}, 例: {sample}）。"
            "实机请用本目录 best.pt。"
        )


def find_d405_serial(preferred: str = "") -> tuple[str, str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        raise SystemExit("没有检测到 Intel RealSense。请接 D405，并用 USB3 口。")

    listed = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        listed.append((serial, name))
        print(f"[INFO] 相机: {name}  SN={serial}")

    if preferred:
        for serial, name in listed:
            if serial == preferred:
                return serial, name
        raise SystemExit(f"找不到序列号 {preferred}")

    for serial, name in listed:
        if "D405" in name.upper():
            return serial, name
    serial, name = listed[0]
    print(f"[WARN] 未看到名称含 D405 的设备，改用: {name}")
    return serial, name


def start_pipeline(serial: str, width: int, height: int, fps: int):
    import pyrealsense2 as rs

    attempts = [(width, height, fps)]
    for w, h, f in ((1280, 720, 30), (848, 480, 30), (640, 480, 30)):
        if (w, h, f) not in attempts:
            attempts.append((w, h, f))

    last_err = None
    for w, h, f in attempts:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
        try:
            profile = pipeline.start(config)
            print(f"[INFO] 已打开 {w}x{h}@{f}")
            return pipeline, profile, rs
        except Exception as exc:
            last_err = exc
            try:
                pipeline.stop()
            except Exception:
                pass
            print(f"[WARN] 打不开 {w}x{h}@{f}: {exc}")
    raise SystemExit(f"无法打开 RealSense 彩色/深度流: {last_err}")


def draw_overlay(frame: np.ndarray, result, fps: float, conf: float) -> np.ndarray:
    vis = result.plot()
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    names = result.names or {}
    lines = [f"D405  {fps:.1f} FPS  conf={conf:.2f}  n={n}"]
    if boxes is not None and n:
        counts: dict[str, int] = {}
        for cls_id in boxes.cls.tolist():
            key = names.get(int(cls_id), str(int(cls_id)))
            counts[key] = counts.get(key, 0) + 1
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1])[:8]:
            lines.append(f"{name} x{cnt}" if cnt > 1 else name)
    y = 28
    for i, text in enumerate(lines):
        cv2.putText(
            vis,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if i == 0 else 0.6,
            (0, 255, 0) if i == 0 else (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return vis


def main() -> None:
    args = parse_args()
    weights = resolve_weights(args.weights)
    print(f"[INFO] 权重: {weights}")

    try:
        import pyrealsense2 as rs  # noqa: F401
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖。请在 conda wrs 中安装:\n"
            "  pip install pyrealsense2 ultralytics opencv-python\n"
            f"原始错误: {exc}"
        ) from exc

    device = args.device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 推理设备: {device}")

    model = YOLO(str(weights))
    warn_if_sim_weights(model)

    serial, cam_name = find_d405_serial(args.serial)
    print(f"[INFO] 使用: {cam_name}  SN={serial}")
    pipeline, profile, rs = start_pipeline(serial, args.width, args.height, args.fps)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        try:
            depth_sensor.set_option(
                rs.option.visual_preset, int(rs.rs400_visual_preset.high_density)
            )
        except Exception:
            pass

    align = rs.align(rs.stream.color)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    show_depth = False
    last_vis = None
    frame_i = 0
    t0 = time.perf_counter()
    fps = 0.0
    save_i = 0

    print("D405 工作距离大约 7–50 cm，零件请放近一些。")
    print("q 退出  s 保存  d 深度窗  [ / ] 调置信度")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())

            run = args.skip <= 0 or (frame_i % (args.skip + 1) == 0)
            if run or last_vis is None:
                result = model.predict(
                    source=color,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )[0]
                now = time.perf_counter()
                dt = now - t0
                t0 = now
                if dt > 1e-6:
                    fps = 1.0 / dt
                last_vis = draw_overlay(color, result, fps, args.conf)

            cv2.imshow("D405 live detect", last_vis)
            if show_depth and depth_frame:
                depth = np.asanyarray(depth_frame.get_data())
                depth_vis = cv2.convertScaleAbs(depth, alpha=0.08)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("D405 depth", depth_color)

            key = cv2.waitKey(1) & 0xFF
            frame_i += 1
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth
                if not show_depth:
                    try:
                        cv2.destroyWindow("D405 depth")
                    except cv2.error:
                        pass
            if key == ord("]"):
                args.conf = min(0.9, round(args.conf + 0.05, 2))
                print(f"[INFO] conf={args.conf}")
            if key == ord("["):
                args.conf = max(0.05, round(args.conf - 0.05, 2))
                print(f"[INFO] conf={args.conf}")
            if key == ord("s") and last_vis is not None:
                out = SAVE_DIR / f"{save_i:06d}.png"
                cv2.imwrite(str(out), last_vis)
                print(f"[INFO] 已保存 {out}")
                save_i += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
