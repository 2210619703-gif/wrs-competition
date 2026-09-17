# -*- coding: utf-8 -*-
"""RealSense D405 采集封装（固定俯视）。

给出**与彩色图逐像素对齐**的深度和相机系点云，这样 YOLO 在彩色图上出的 mask
可以直接拿去索引 3D 点。

为什么不直接用 ``wrs.drivers.devices.realsense``：那边把 ``pc.calculate()``
的顶点按深度图顺序摊平，再靠「深度和彩色同分辨率」硬 reshape 成彩色图形状。
积木那种 50mm 的目标无所谓，但螺丝、垫圈只有几毫米，深度与彩色之间的视差
会把点采到旁边去。这里用 ``rs.align`` 把深度对齐到彩色帧再反投影，位置更准。

没有相机也能用：``--save-dir`` 存下的帧可以用 ``load_frame()`` 离线复现，
方便先把识别和坐标链调通。

自检 / 实时预览 / 存一帧::

    python tiaozhanbei/real/d405_camera.py --live
    python tiaozhanbei/real/d405_camera.py --live --yolo
    python tiaozhanbei/real/d405_camera.py --probe
    python tiaozhanbei/real/d405_camera.py --save-dir tiaozhanbei/real/outputs/frame01
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from vision_config import (
    CAMERA_AE_SETTLE_LUMA,
    CAMERA_AE_SETTLE_MAX_FRAMES,
    CAMERA_AE_SETTLE_WINDOW,
    CAMERA_RESOLUTION,
    CAMERA_SERIAL,
    CAMERA_WARMUP_FRAMES,
    MAX_DEPTH_M,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
    VisionConfigError,
    default_weights,
)

RESOLUTIONS = {"mid": (848, 480), "high": (1280, 720)}
FPS = 30


class CameraError(RuntimeError):
    """相机打不开、取帧超时等。"""


@dataclass
class CameraFrame:
    """一帧对齐好的数据。

    ``points`` 是 HxWx3 的相机系坐标（米），和 ``color`` 逐像素对应，
    无效深度处为 0。世界系坐标由手眼矩阵再变换一次得到。
    """

    color: np.ndarray            # HxWx3, BGR, uint8
    depth_m: np.ndarray          # HxW, float32, 米
    points: np.ndarray           # HxWx3, float32, 相机系，米
    intrinsics: dict             # fx, fy, cx, cy

    @property
    def width(self) -> int:
        return int(self.color.shape[1])

    @property
    def height(self) -> int:
        return int(self.color.shape[0])

    def valid_ratio(self) -> float:
        """有效深度像素占比，用来判断相机是不是拍到了东西。"""
        return float(np.count_nonzero(self.depth_m > 0) / self.depth_m.size)

    def save(self, out_dir: str) -> str:
        import cv2

        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "color.png"), self.color)
        np.save(os.path.join(out_dir, "depth_m.npy"), self.depth_m.astype(np.float32))
        with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
            json.dump(self.intrinsics, f, ensure_ascii=False, indent=2)
        return out_dir


def deproject(depth_m: np.ndarray, intr: dict) -> np.ndarray:
    """对齐后的深度图 → HxWx3 相机系点云（针孔模型，无效处为 0）。"""
    h, w = depth_m.shape[:2]
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    bad = (z <= 0) | (z > MAX_DEPTH_M) | ~np.isfinite(z)
    z = np.where(bad, 0.0, z)
    pts = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=-1)
    pts[bad] = 0.0
    return pts


def load_frame(frame_dir: str) -> CameraFrame:
    """读回 :meth:`CameraFrame.save` 存下的帧，用于无相机离线调试。"""
    import cv2

    frame_dir = os.path.abspath(frame_dir)
    color_path = os.path.join(frame_dir, "color.png")
    depth_path = os.path.join(frame_dir, "depth_m.npy")
    intr_path = os.path.join(frame_dir, "intrinsics.json")
    for p in (color_path, depth_path, intr_path):
        if not os.path.isfile(p):
            raise CameraError(f"帧目录不完整，缺 {os.path.basename(p)}：{frame_dir}")
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth = np.load(depth_path).astype(np.float32)
    with open(intr_path, "r", encoding="utf-8-sig") as f:
        intr = json.load(f)
    if color.shape[:2] != depth.shape[:2]:
        raise CameraError(
            f"彩色 {color.shape[:2]} 和深度 {depth.shape[:2]} 尺寸不一致：{frame_dir}"
        )
    return CameraFrame(color=color, depth_m=depth, points=deproject(depth, intr), intrinsics=intr)


def pyrealsense_available() -> bool:
    try:
        import pyrealsense2  # noqa: F401
    except Exception:
        return False
    return True


class D405Camera:
    """D405 的薄封装，用 ``with`` 保证管道一定关掉。"""

    def __init__(
        self,
        resolution: str = CAMERA_RESOLUTION,
        serial: str | None = CAMERA_SERIAL,
        *,
        logger=print,
    ) -> None:
        if resolution not in RESOLUTIONS:
            raise VisionConfigError(f"分辨率只能是 {tuple(RESOLUTIONS)}，收到 {resolution!r}")
        self.resolution = resolution
        self.serial = serial
        self.log = logger
        self._pipe = None
        self._align = None
        self._depth_scale = 0.001
        self._intr: dict | None = None

    def __enter__(self) -> "D405Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise CameraError(
                f"导入 pyrealsense2 失败：{e}\n请先 pip install pyrealsense2（需要 USB3）。"
            ) from e

        w, h = RESOLUTIONS[self.resolution]
        pipe = rs.pipeline()
        conf = rs.config()
        if self.serial:
            conf.enable_device(str(self.serial))
        conf.enable_stream(rs.stream.depth, w, h, rs.format.z16, FPS)
        conf.enable_stream(rs.stream.color, w, h, rs.format.bgr8, FPS)
        try:
            profile = pipe.start(conf)
        except Exception as e:
            raise CameraError(
                f"D405 打不开（{w}x{h}）：{e}\n"
                "检查：USB3 口、相机是否被别的程序占用、--serial 是否正确。"
            ) from e

        self._pipe = pipe
        self._align = rs.align(rs.stream.color)
        self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        try:
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        except Exception:
            pass

        # 前几帧自动曝光还没稳定，深度也容易大面积无效，先丢掉
        for _ in range(max(1, int(CAMERA_WARMUP_FRAMES))):
            try:
                pipe.wait_for_frames(timeout_ms=5000)
            except Exception:
                break

        frames = self._align.process(pipe.wait_for_frames(timeout_ms=5000))
        cf = frames.get_color_frame()
        i = cf.profile.as_video_stream_profile().intrinsics
        self._intr = {"fx": float(i.fx), "fy": float(i.fy), "cx": float(i.ppx), "cy": float(i.ppy)}
        self.log(
            f"[cam] D405 就绪 {w}x{h} depth_scale={self._depth_scale:.6f} "
            f"fx={i.fx:.1f} fy={i.fy:.1f} cx={i.ppx:.1f} cy={i.ppy:.1f}"
        )

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
            self.log("[cam] 已关闭")

    @property
    def intrinsics(self) -> dict:
        if self._intr is None:
            raise CameraError("相机还没 open()")
        return dict(self._intr)

    def capture(self, timeout_ms: int = 5000) -> CameraFrame:
        """取一帧，深度已对齐到彩色帧。"""
        if self._pipe is None:
            raise CameraError("相机还没 open()")
        try:
            frames = self._align.process(self._pipe.wait_for_frames(timeout_ms=timeout_ms))
        except Exception as e:
            raise CameraError(f"取帧失败/超时：{e}") from e
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise CameraError("这一帧缺深度或彩色数据")
        color = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        intr = self.intrinsics
        return CameraFrame(color=color, depth_m=depth_m, points=deproject(depth_m, intr), intrinsics=intr)

    def wait_exposure_stable(self, timeout_ms: int = 5000) -> CameraFrame:
        """等到彩色图平均亮度不再跳，再交给 YOLO。

        开机后 AE 还在爬的那几帧，同一桌面有时只检出 1 个件。live 能看到 3 个
        是因为连续推理等到了曝光稳住。这里用亮度窗口代替盲等。
        """
        window = max(3, int(CAMERA_AE_SETTLE_WINDOW))
        limit = max(window, int(CAMERA_AE_SETTLE_MAX_FRAMES))
        tol = float(CAMERA_AE_SETTLE_LUMA)
        lumas: list[float] = []
        last = None
        t0 = time.perf_counter()
        for _ in range(limit):
            last = self.capture(timeout_ms=timeout_ms)
            lumas.append(float(last.color.mean()))
            if len(lumas) >= window:
                chunk = lumas[-window:]
                if (max(chunk) - min(chunk)) <= tol:
                    self.log(
                        f"[cam] 曝光已稳定 luma={lumas[-1]:.1f} "
                        f"（{len(lumas)} 帧，{time.perf_counter() - t0:.1f}s）"
                    )
                    return last
        span = (max(lumas[-window:]) - min(lumas[-window:])) if lumas else 0.0
        self.log(
            f"[cam] 曝光未完全稳住 luma={lumas[-1]:.1f} Δ={span:.1f}，"
            f"用当前帧继续（等了 {time.perf_counter() - t0:.1f}s）"
        )
        return last if last is not None else self.capture(timeout_ms=timeout_ms)

    def capture_stack(self, n: int = 5, timeout_ms: int = 5000) -> list[CameraFrame]:
        """曝光稳住后再连拍 n 帧，给多帧投票和深度中位数用。"""
        n = max(1, int(n))
        self.wait_exposure_stable(timeout_ms=timeout_ms)
        frames = [self.capture(timeout_ms=timeout_ms) for _ in range(n)]
        self.log(f"[cam] 已连拍 {len(frames)} 帧，交给 YOLO 投票")
        return frames

    def capture_median(self, n: int = 5, timeout_ms: int = 5000) -> CameraFrame:
        """连拍取深度中位数，压掉 D405 的逐帧噪声。"""
        return frames_to_median(self.capture_stack(n, timeout_ms=timeout_ms))


def frames_to_median(frames: list[CameraFrame]) -> CameraFrame:
    """多帧深度取中位数，彩色用最后一帧（曝光通常最稳）。"""
    if not frames:
        raise CameraError("没有可合并的帧")
    if len(frames) == 1:
        return frames[0]
    stack = np.stack([f.depth_m for f in frames], axis=0)
    stack = np.where(stack > 0, stack, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    last = frames[-1]
    return CameraFrame(
        color=last.color,
        depth_m=med,
        points=deproject(med, last.intrinsics),
        intrinsics=last.intrinsics,
    )


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    import cv2

    vis = np.clip(depth_m / max(MAX_DEPTH_M, 1e-6), 0.0, 1.0)
    vis = np.where(depth_m > 0, vis, 0.0)
    u8 = (vis * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def live_preview(cam: D405Camera) -> None:
    """弹出窗口播彩色 + 深度，按 q / Esc 退出。"""
    import cv2

    win = "D405 live  (q 退出)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[cam] 实时预览已打开，按 q 或 Esc 关闭")
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
    cv2.destroyWindow(win)


def live_yolo_preview(
    cam: D405Camera,
    *,
    weights: str | None = None,
    conf: float = YOLO_CONF,
    iou: float = YOLO_IOU,
    imgsz: int = YOLO_IMGSZ,
) -> None:
    """彩色实时画面 + YOLO 框。用来对着桌面调位置/灯光，看权重能不能认出零件。

    推理用较低的底限把弱候选也画出来：绿色 = 达到当前阈值（正式识别会留下），
    橙色 = 模型看见了但分数不够。``[`` / ``]`` 调绿色阈值，``-`` / ``=`` 调输入尺寸。
    """
    import time

    import cv2
    from ultralytics import YOLO

    path = weights or default_weights()
    print(f"[cam] 加载 YOLO {os.path.basename(path)} …")
    model = YOLO(str(path))
    names = model.names or {}
    print(
        f"[cam] 任务={getattr(model, 'task', None)}  类数={len(names)}  "
        f"conf={conf:.2f}  imgsz={imgsz}\n"
        "[cam] q 退出；[ / ] 调阈值；- / = 调输入尺寸"
    )
    win = "D405 + YOLO"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_print = 0.0
    floor = 0.08
    while True:
        frame = cam.capture(timeout_ms=2000)
        t0 = time.perf_counter()
        result = model.predict(
            frame.color, conf=min(conf, floor), iou=iou, imgsz=imgsz, verbose=False
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        view = frame.color.copy()
        boxes = result.boxes
        masks = result.masks
        n_all = 0 if boxes is None else len(boxes)
        n_pass = 0
        for i in range(n_all):
            x1, y1, x2, y2 = (int(v) for v in boxes[i].xyxy[0].cpu().numpy())
            score = float(boxes[i].conf)
            cls_name = str(names.get(int(boxes[i].cls), int(boxes[i].cls)))
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
            cv2.putText(
                view,
                f"{cls_name} {score:.2f}",
                (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
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
        if n_all and now - last_print > 0.8:
            hits = [
                f"{names.get(int(b.cls), int(b.cls))} {float(b.conf):.2f}"
                for b in boxes
            ]
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
    cv2.destroyWindow(win)


def main() -> int:
    p = argparse.ArgumentParser(description="D405 自检 / 实时预览 / 存一帧")
    p.add_argument("--resolution", default=CAMERA_RESOLUTION, choices=tuple(RESOLUTIONS))
    p.add_argument("--serial", default=CAMERA_SERIAL)
    p.add_argument("--frames", type=int, default=5, help="曝光稳住后连拍取深度中位数的帧数")
    p.add_argument("--save-dir", default="", help="存下 color.png / depth_m.npy / intrinsics.json")
    p.add_argument("--probe", action="store_true", help="只打印一帧的统计信息")
    p.add_argument("--live", action="store_true", help="弹出窗口看实时画面，按 q 退出")
    p.add_argument("--yolo", action="store_true", help="实时画面上叠加 YOLO 识别框（隐含 --live）")
    p.add_argument("--weights", default="", help="YOLO 权重；默认用本目录的 real_multi.pt")
    p.add_argument("--conf", type=float, default=YOLO_CONF, help="YOLO 置信度阈值")
    p.add_argument("--iou", type=float, default=YOLO_IOU)
    p.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help="YOLO 输入边长，越大越容易看到小零件")
    args = p.parse_args()

    if not pyrealsense_available():
        print("[cam] 没装 pyrealsense2，无法连相机。pip install pyrealsense2")
        return 2

    try:
        with D405Camera(args.resolution, args.serial) as cam:
            if args.yolo:
                live_yolo_preview(
                    cam,
                    weights=args.weights or None,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                )
                return 0
            if args.live:
                live_preview(cam)
                return 0
            frame = cam.capture_median(args.frames)
    except CameraError as e:
        print(f"[cam] 失败：{e}")
        return 1
    except VisionConfigError as e:
        print(f"[cam] {e}")
        return 2
    except KeyboardInterrupt:
        print("\n[cam] 已退出")
        return 0

    valid = frame.depth_m[frame.depth_m > 0]
    print(
        f"[cam] {frame.width}x{frame.height} 有效深度 {frame.valid_ratio() * 100:.1f}%  "
        + (
            f"距离 {valid.min():.3f}~{valid.max():.3f}m 中位 {np.median(valid):.3f}m"
            if valid.size
            else "全帧无有效深度（检查镜头遮挡 / 距离是否过近）"
        )
    )
    if args.save_dir:
        print(f"[cam] 已保存 -> {frame.save(args.save_dir)}")
    elif not args.probe:
        print("[cam] 加 --save-dir 可把这一帧存下来离线用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
