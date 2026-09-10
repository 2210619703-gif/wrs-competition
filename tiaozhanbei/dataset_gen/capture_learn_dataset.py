#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按学习标注格式采集桌面零件数据。

目录结构（默认 sample=0001）：
  tiaozhanbei/dataset_learn/0001/
    0001.jpg                 相机 RGB（1280x720）
    0001_labeled.png         bbox + 名称预览（给人看）
    annotations.json         image/width/height/objects[...]
    masks/
      0001_01.png            单物体二值 mask（与 JSON id 一一对应）
      0001_02.png
      ...
    depth_m.npy / depth_preview.png / segmentation.png / pointcloud.ply

annotations.json 中每个 object 含：
  id, class, bbox[x1,y1,x2,y2], polygon, mask_path, pose_6d
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tiaozhanbei.sim.environment import build_scene, list_desk_stripe_styles
from tiaozhanbei.dataset_gen.sim_data_camera import SimDataCamera


def parse_args():
    parser = argparse.ArgumentParser(description="采集学习格式数据集（每物体独立 mask + 6D 位姿）")
    parser.add_argument(
        "--sample-id",
        default="0001",
        help="样本编号，用于图片名与 mask 文件名前缀",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认 tiaozhanbei/dataset_learn/{sample_id}",
    )
    parser.add_argument("--width", type=int, default=1280, help="图像宽度（学习示例为 1280）")
    parser.add_argument("--height", type=int, default=720, help="图像高度（学习示例为 720）")
    parser.add_argument("--fov", type=float, default=45.0, help="虚拟相机视场角")
    parser.add_argument("--stride", type=int, default=2, help="点云采样步长")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（桌面条纹等）")
    parser.add_argument(
        "--desk-stripe",
        default=None,
        choices=list_desk_stripe_styles(),
        help="桌面条纹样式；不指定则按 --seed 随机选择",
    )
    parser.add_argument("--show", action="store_true", help="采集完成后保留窗口")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(_TB_DIR, "dataset_learn", args.sample_id)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    scene = build_scene(show_frame=False, seed=args.seed, desk_stripe=args.desk_stripe)
    base = scene["base"]
    part_infos = scene["part_infos"]
    desk_stripe = scene["desk_stripe"]

    camera = SimDataCamera(
        base=base,
        cam_pos=np.array([0.65, -0.95, 0.75]),
        lookat_pos=np.array([0.0, 0.0, 0.03]),
        resolution=np.array([args.width, args.height]),
        fov=args.fov,
        near=0.01,
        far=5.0,
    )

    meta = camera.capture_learn_format(
        part_infos=part_infos,
        output_dir=output_dir,
        sample_id=args.sample_id,
        pointcloud_stride=args.stride,
    )
    meta["seed"] = args.seed
    meta["desk_stripe"] = {
        "style": desk_stripe["style"],
        "texture_path": desk_stripe["texture_path"],
    }
    with open(os.path.join(output_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    n_vis = sum(1 for o in meta["objects"] if (o.get("visible_pixels") or 0) > 0)
    print(f"[learn] 数据已保存到: {os.path.abspath(output_dir)}")
    print(f"[learn] 图像: {meta['image']} ({meta['width']}x{meta['height']})")
    print(f"[learn] 桌面条纹: {desk_stripe['style']} (seed={args.seed})")
    print(f"[learn] 物体总数: {len(meta['objects'])}，可见: {n_vis}")
    print("[learn] 主要文件: annotations.json, masks/*.png, *.jpg")

    if args.show:
        base.run()
    else:
        base.destroy()


if __name__ == "__main__":
    main()
