"""
导出工作流 vision_objects JSON（单行字符串）。

用法:
  # 单张 dataset 图片（需成对 depth）
  python export_vision_objects.py --image dataset/images/val/000000.png

  # 输出写入文件
  python export_vision_objects.py --image dataset/images/val/000000.png --out vision.json

  # 无物体时输出 []
  python export_vision_objects.py --image ... --conf 0.99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wrs-main"))

from ultralytics import YOLO  # noqa: E402
from wrs import wd  # noqa: E402

from industrial_class_config import class_names_dict  # noqa: E402
from test_industrial_detection import (  # noqa: E402
    DEFAULT_WEIGHTS_8,
    DEFAULT_WEIGHTS_MERGED4,
    RESOLUTION,
    apply_camera_from_meta,
    create_vcam,
    load_meta_annotations,
    resolve_depth_path,
    resolve_meta_path,
    run_detection_with_pose,
    setup_class_set,
)
from industrial_pose_utils import vision_objects_json  # noqa: E402


def build_detect_args(ns: argparse.Namespace) -> argparse.Namespace:
    """补齐 run_detection_with_pose 所需参数字段。"""
    defaults = dict(
        conf=0.38,
        yolo_iou=0.60,
        max_det=12,
        dedup_iou=0.55,
        no_global_dedup=False,
        smooth=False,
        pose_dedup_dist=0.055,
        center_dist=28.0,
        axis_len=0.045,
        pose=True,
    )
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def parse_args():
    p = argparse.ArgumentParser(description="导出 vision_objects JSON 字符串")
    p.add_argument("--image", required=True, help="RGB 图片路径")
    p.add_argument("--class-set", choices=["8class", "merged4"], default="8class")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.38)
    p.add_argument("--out", type=Path, default=None, help="写入文件；默认打印到 stdout")
    p.add_argument("--pretty", action="store_true", help="多行缩进 JSON（调试用）")
    p.add_argument(
        "--format",
        choices=["internal", "downstream3d"],
        default="downstream3d",
        help="internal=世界坐标; downstream3d=下游三维测试格式",
    )
    return p.parse_args()


def main():
    args = parse_args()
    setup_class_set(args.class_set)
    if args.weights is None:
        args.weights = DEFAULT_WEIGHTS_MERGED4 if args.class_set == "merged4" else DEFAULT_WEIGHTS_8
    if not args.weights.exists():
        raise FileNotFoundError(f"权重不存在: {args.weights}")

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {img_path}")

    depth_path = resolve_depth_path(img_path)
    if depth_path is None:
        raise FileNotFoundError(f"未找到对应 depth: {img_path}")

    import cv2
    import numpy as np

    img_bgr = cv2.imread(str(img_path))
    depth = np.load(str(depth_path))
    meta_path = resolve_meta_path(img_path)
    camera, _ = load_meta_annotations(meta_path) if meta_path else (None, [])

    args.pose = True
    args = build_detect_args(args)
    model = YOLO(str(args.weights))
    base = wd.World(w=RESOLUTION[0], h=RESOLUTION[1], auto_rotate=False)
    vcam = create_vcam(base)
    if camera is not None:
        apply_camera_from_meta(vcam, camera)
    base.graphicsEngine.renderFrame()

    poses, _, _ = run_detection_with_pose(model, img_bgr, depth, vcam, args)
    base.destroy()

    payload = vision_objects_json(
        poses,
        indent=2 if args.pretty else None,
        fmt=args.format,
        depth=depth,
    )
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        print(f"[INFO] 已写入 {args.out.resolve()} ({len(poses)} poses, format={args.format})", file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()