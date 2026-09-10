#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part_Model 牌堆抽样：同一轮内不重复、抽完即全覆盖，支持断点续采。

进度写入 dataset_learn/manifest.json。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from tiaozhanbei.sim.environment import scan_part_library

_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_GEN_DIR, os.pardir))
DEFAULT_MANIFEST_PATH = os.path.join(_TB_DIR, "dataset_learn", "manifest.json")


def library_index():
    """rel_path -> entry 的快查表。"""
    return {e["rel_path"]: e for e in scan_part_library()}


def _new_deck(seed: int) -> List[str]:
    """按分类轮询洗牌：各类内部打乱后再 round-robin 入堆，场景内类别更均匀。"""
    entries = scan_part_library()
    rng = np.random.default_rng(seed)
    by_cat: Dict[str, List[str]] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e["rel_path"])
    cats = sorted(by_cat.keys())
    for cat in cats:
        rng.shuffle(by_cat[cat])
    deck: List[str] = []
    while any(by_cat[c] for c in cats):
        for cat in cats:
            if by_cat[cat]:
                deck.append(by_cat[cat].pop(0))
    return deck


def init_manifest(
    seed: int = 0,
    parts_per_scene: int = 10,
    round_id: int = 1,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> Dict[str, Any]:
    """创建新一轮牌堆 manifest 并落盘。"""
    deck = _new_deck(seed)
    manifest = {
        "seed": int(seed),
        "parts_per_scene": int(parts_per_scene),
        "round": int(round_id),
        "library_size": len(deck),
        "deck_remaining": deck,
        "used": [],
        "samples": {},
    }
    save_manifest(manifest, manifest_path)
    return manifest


def load_manifest(manifest_path: str = DEFAULT_MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any], manifest_path: str = DEFAULT_MANIFEST_PATH) -> None:
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_or_init_manifest(
    seed: int = 0,
    parts_per_scene: int = 10,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    force_new_round: bool = False,
) -> Dict[str, Any]:
    """加载已有 manifest；不存在或 force_new_round 时新建。

    若已有 manifest 且 seed/parts_per_scene 与请求不一致，则新开一轮（round+1）。
    """
    existing = load_manifest(manifest_path)
    if existing is None:
        return init_manifest(
            seed=seed,
            parts_per_scene=parts_per_scene,
            round_id=1,
            manifest_path=manifest_path,
        )

    need_new = force_new_round or (
        int(existing.get("seed", seed)) != int(seed)
        or int(existing.get("parts_per_scene", parts_per_scene)) != int(parts_per_scene)
    )
    if need_new:
        return init_manifest(
            seed=seed,
            parts_per_scene=parts_per_scene,
            round_id=int(existing.get("round", 1)) + 1,
            manifest_path=manifest_path,
        )
    # 参数一致：续采；若牌堆已空且仍要采，由调用方决定是否 force_new_round
    existing["parts_per_scene"] = int(parts_per_scene)
    return existing


def _proportional_quotas(total: int, weights: List[int]) -> List[int]:
    """按权重把 total 个名额分给各类（最大余数法）。"""
    if total <= 0 or not weights:
        return [0] * len(weights)
    s = sum(weights)
    if s <= 0:
        return [0] * len(weights)
    raw = [total * w / s for w in weights]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - base[i], weights[i]), reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def pop_next_batch(
    manifest: Dict[str, Any],
    n: Optional[int] = None,
) -> Optional[List[Dict[str, Any]]]:
    """从牌堆弹出下一批零件；按各类剩余比例分配，避免大类堆到最后。"""
    if n is None:
        n = int(manifest.get("parts_per_scene", 10))
    remaining = list(manifest.get("deck_remaining") or [])
    if not remaining:
        return None

    index = library_index()
    by_cat: Dict[str, List[str]] = {}
    for rel in remaining:
        entry = index.get(rel)
        cat = entry["category"] if entry else "?"
        by_cat.setdefault(cat, []).append(rel)

    cats = sorted([c for c, items in by_cat.items() if items])
    take = min(int(n), len(remaining))
    quotas = _proportional_quotas(take, [len(by_cat[c]) for c in cats])

    batch_rels: List[str] = []
    for cat, q in zip(cats, quotas):
        for _ in range(q):
            if by_cat[cat]:
                batch_rels.append(by_cat[cat].pop(0))

    # 配额取整误差时补齐
    while len(batch_rels) < take:
        cats_left = [c for c in cats if by_cat[c]]
        if not cats_left:
            break
        cat = max(cats_left, key=lambda c: len(by_cat[c]))
        batch_rels.append(by_cat[cat].pop(0))

    taken = set(batch_rels)
    manifest["deck_remaining"] = [r for r in remaining if r not in taken]

    batch = []
    for rel in batch_rels:
        entry = index.get(rel)
        if entry is not None:
            batch.append(entry)
    if not batch:
        return None
    return batch


def mark_sample_done(
    manifest: Dict[str, Any],
    sample_id: str,
    part_entries: List[Dict[str, Any]],
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """记录本样本用过的零件，并写回 manifest。"""
    rels = [e["rel_path"] for e in part_entries]
    used = manifest.setdefault("used", [])
    used.extend(rels)
    samples = manifest.setdefault("samples", {})
    record = {"part_paths": rels, "n_parts": len(rels)}
    if extra:
        record.update(extra)
    samples[str(sample_id)] = record
    save_manifest(manifest, manifest_path)


def coverage_stats(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """返回本轮覆盖统计。"""
    lib_size = int(manifest.get("library_size") or 0)
    used = manifest.get("used") or []
    remaining = manifest.get("deck_remaining") or []
    return {
        "round": manifest.get("round"),
        "library_size": lib_size,
        "used": len(used),
        "remaining": len(remaining),
        "n_samples": len(manifest.get("samples") or {}),
        "complete": len(remaining) == 0 and lib_size > 0 and len(used) >= lib_size,
    }


def next_sample_id(dataset_root: str, width: int = 4) -> str:
    """在 dataset_root 下找下一个未占用的数字编号 sample_id。"""
    os.makedirs(dataset_root, exist_ok=True)
    existing = set()
    for name in os.listdir(dataset_root):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path) and name.isdigit():
            existing.add(int(name))
    i = 1
    while i in existing:
        i += 1
    return f"{i:0{width}d}"
