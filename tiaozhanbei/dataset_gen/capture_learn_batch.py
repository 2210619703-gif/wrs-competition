#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量采集学习格式数据：Part_Model 牌堆不重复覆盖。

新约定（与 ``scene_layout`` 对齐）：
- 收纳盒 / 机械臂坐标固定，写入 annotations，但**不渲染**进图像；
- 零件只在安全区随机摆放（IK 友好，避开臂与盒）；
- 每场景零件个数可随机；桌面纹理更丰富。

示例：
  python tiaozhanbei/dataset_gen/capture_learn_batch.py --num-samples 40 --parts-min 4 --parts-max 12 --fresh --seed 0
  python tiaozhanbei/dataset_gen/capture_learn_batch.py --num-samples 5 --parts-per-scene 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tiaozhanbei.sim.environment import build_scene, list_desk_stripe_styles
from tiaozhanbei.dataset_gen.part_sampler import (
    DEFAULT_MANIFEST_PATH,
    coverage_stats,
    load_or_init_manifest,
    mark_sample_done,
    next_sample_id,
    pop_next_batch,
    save_manifest,
)
from tiaozhanbei.sim.scene_layout import layout_dict, save_scene_layout_json
from tiaozhanbei.dataset_gen.sim_data_camera import SimDataCamera


def parse_args():
    parser = argparse.ArgumentParser(
        description="Part_Model 安全区批量采集（固定收纳盒布局元数据）"
    )
    parser.add_argument("--num-samples", type=int, default=2, help="本次最多采集多少张")
    parser.add_argument(
        "--parts-per-scene",
        type=int,
        default=None,
        help="固定每场景零件数；与 --parts-min/max 互斥优先本项",
    )
    parser.add_argument("--parts-min", type=int, default=4, help="每场景最少零件数")
    parser.add_argument("--parts-max", type=int, default=12, help="每场景最多零件数")
    parser.add_argument("--seed", type=int, default=0, help="牌堆洗牌与条纹随机种子")
    parser.add_argument(
        "--dataset-root",
        default=os.path.join(_TB_DIR, "dataset_learn"),
        help="样本输出根目录",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST_PATH,
        help="进度 manifest.json 路径",
    )
    parser.add_argument(
        "--new-round",
        action="store_true",
        help="强制新开一轮（重新洗牌，忽略旧牌堆）",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="清空 dataset-root 下数字样本目录并重置 manifest 后重采",
    )
    parser.add_argument(
        "--desk-stripe",
        default=None,
        choices=list_desk_stripe_styles(),
        help="固定桌面条纹；不指定则每样本按 seed+sample 随机",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def _resolve_parts_range(args):
    if args.parts_per_scene is not None:
        n = int(args.parts_per_scene)
        return n, n
    lo = max(1, int(args.parts_min))
    hi = max(lo, int(args.parts_max))
    return lo, hi


def _fresh_dataset(dataset_root: str, manifest_path: str):
    """删除数字样本目录与旧 manifest，保留其它文件。"""
    if os.path.isdir(dataset_root):
        for name in os.listdir(dataset_root):
            path = os.path.join(dataset_root, name)
            if os.path.isdir(path) and name.isdigit():
                shutil.rmtree(path)
                print(f"[fresh] removed {path}")
    if os.path.isfile(manifest_path):
        os.remove(manifest_path)
        print(f"[fresh] removed {manifest_path}")


def capture_one(sample_id, part_entries, output_dir, seed, desk_stripe, args, scene_layout):
    """采集单张样本并返回 meta。"""
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    scene = build_scene(
        show_frame=False,
        seed=seed,
        desk_stripe=desk_stripe,
        part_entries=part_entries,
        use_safe_layout=True,
    )
    base = scene["base"]
    part_infos = scene["part_infos"]
    stripe = scene["desk_stripe"]

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
        sample_id=sample_id,
        pointcloud_stride=args.stride,
    )
    meta["seed"] = seed
    meta["desk_stripe"] = {
        "style": stripe["style"],
        "texture_path": stripe["texture_path"],
    }
    meta["part_paths"] = [e["rel_path"] for e in part_entries]
    meta["scene_layout"] = scene_layout
    meta["n_parts"] = len(part_entries)
    base.destroy()
    return meta


def main():
    args = parse_args()
    parts_min, parts_max = _resolve_parts_range(args)
    # manifest 用上限占位；实际每张再随机取 n∈[min,max]
    parts_per_scene_manifest = parts_max

    os.makedirs(args.dataset_root, exist_ok=True)
    if args.fresh:
        _fresh_dataset(args.dataset_root, args.manifest)

    layout = layout_dict()
    layout_path = save_scene_layout_json(
        os.path.join(args.dataset_root, "scene_layout.json")
    )
    print(f"[layout] wrote {layout_path}")
    print(
        f"[layout] box_pos={layout['storage_box']['pos_m']} "
        f"size={layout['storage_box']['size_outer_m']}"
    )
    print(
        f"[layout] spawn_xy=[{layout['part_spawn']['xy_min_m']} .. "
        f"{layout['part_spawn']['xy_max_m']}]"
    )

    manifest = load_or_init_manifest(
        seed=args.seed,
        parts_per_scene=parts_per_scene_manifest,
        manifest_path=args.manifest,
        force_new_round=args.new_round or args.fresh,
    )
    if not (manifest.get("deck_remaining") or []) and not (args.new_round or args.fresh):
        print("[batch] 当前轮牌堆已空，自动新开一轮继续覆盖采集")
        manifest = load_or_init_manifest(
            seed=args.seed,
            parts_per_scene=parts_per_scene_manifest,
            manifest_path=args.manifest,
            force_new_round=True,
        )
    manifest["parts_min"] = int(parts_min)
    manifest["parts_max"] = int(parts_max)

    rng = np.random.default_rng(int(args.seed) + 7)
    captured = 0
    for _ in range(args.num_samples):
        n_parts = int(rng.integers(parts_min, parts_max + 1))
        batch = pop_next_batch(manifest, n=n_parts)
        if batch is None:
            print("[batch] 本轮牌堆已抽完，覆盖完成")
            break
        # 牌堆尾部不足最小个数时结束，避免出现过稀样本
        if len(batch) < parts_min:
            print(
                f"[batch] 剩余零件仅 {len(batch)} < parts_min={parts_min}，结束本轮"
            )
            # 退回牌堆，避免“用掉却不建样本”
            rem = list(manifest.get("deck_remaining") or [])
            rem = [e["rel_path"] for e in batch] + rem
            manifest["deck_remaining"] = rem
            save_manifest(manifest, args.manifest)
            break

        sample_id = next_sample_id(args.dataset_root)
        output_dir = os.path.join(args.dataset_root, sample_id)
        sample_seed = int(args.seed) * 100000 + int(sample_id)

        print(
            f"[batch] 采集 {sample_id}: {len(batch)} 零件, "
            f"剩余牌堆 {len(manifest.get('deck_remaining') or [])}"
        )
        meta = capture_one(
            sample_id=sample_id,
            part_entries=batch,
            output_dir=output_dir,
            seed=sample_seed,
            desk_stripe=args.desk_stripe,
            args=args,
            scene_layout=layout,
        )
        meta["manifest_round"] = manifest.get("round")
        with open(os.path.join(output_dir, "annotations.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        mark_sample_done(
            manifest,
            sample_id,
            batch,
            manifest_path=args.manifest,
            extra={
                "desk_stripe": meta["desk_stripe"]["style"],
                "seed": sample_seed,
                "n_parts": len(batch),
            },
        )
        captured += 1
        n_vis = sum(1 for o in meta["objects"] if (o.get("visible_pixels") or 0) > 0)
        print(
            f"[batch] 完成 {sample_id}: 可见 {n_vis}/{len(meta['objects'])}, "
            f"条纹={meta['desk_stripe']['style']}"
        )

    save_manifest(manifest, args.manifest)
    stats = coverage_stats(manifest)
    print(
        f"[batch] 本次采集 {captured} 张 | round={stats['round']} "
        f"used={stats['used']}/{stats['library_size']} remaining={stats['remaining']} "
        f"complete={stats['complete']}"
    )
    print(
        f"[batch] parts_range=[{parts_min},{parts_max}] "
        f"textures={len(list_desk_stripe_styles())}"
    )


if __name__ == "__main__":
    main()
