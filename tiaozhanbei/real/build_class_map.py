# -*- coding: utf-8 -*-
"""生成 ``yolo_class_map.json``：YOLO 英文类名 → 仿真中文类名 + STL 路径。

真机权重 ``real_multi.pt`` 的类名是英文（``screwdrivers``），而仿真/决策模块用的是
中文（``机器工具-螺丝刀``），STL 还要能在 ``tiaozhanbei/Part_Model`` 里找到。
不对上这张表，识别结果进了仿真会因为找不到 STL 直接报废。

绝大多数类名把下划线换成空格就能对上 Part_Model 的叶目录，剩下十几个在
``OVERRIDES`` 里手写。中文名的构造规则和 ``tiaozhanbei/sim/environment.py``
里的 ``class_name_from_rel_path`` 一致：``<顶层目录中文>-<叶目录中文>``。

权重换了、或者 Part_Model 加了新零件，重跑一次::

    python tiaozhanbei/real/build_class_map.py

本脚本只在改动时手动跑，运行期不依赖它。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

from real_config import REAL_DIR, TB_DIR

PART_MODEL_DIR = Path(TB_DIR) / "Part_Model"
ENV_PY = Path(TB_DIR) / "sim" / "environment.py"
OUT_JSON = Path(REAL_DIR) / "yolo_class_map.json"

# 自动匹配对不上的类：YOLO 类名 → Part_Model 叶目录相对路径。
# 结尾的 1 是权重训练时的编号后缀；带「或」的是权重把多个目录合成了一类。
OVERRIDES: dict[str, str] = {
    "brake_lining_clutch_facing_rivets": "Rivets/Brake lining  Clutch facing rivets",
    "cylindrical_head_bolts1": "Screws and bolts/bolts/Cylindrical head bolts",
    "g_nuts": "Nuts/Clip-on nuts (Sheet metal or speed nuts)/G-nuts",
    "hex_head_bolts1": "Screws and bolts/bolts/Hex head bolts",
    "round_head_bolts1": "Screws and bolts/bolts/Round head bolts",
    "self_locking_nuts": "Nuts/Self-locking nuts",
    "semi_tubular_rivets": "Rivets/Semi-tubular rivets",
    "socket_screws_socket_head_screws": "Screws and bolts/Cap screws/Socket screws, Socket head screws",
    "t_bolts": "Screws and bolts/bolts/T-Bolts",
    "t_nuts": "Nuts/T-nuts",
    "u_bolts": "Screws and bolts/bolts/U-Bolts",
    "washers_screws": "Screws and bolts/Cap screws/Washer screws",
    "plastic_countersunk_or_plastic_fillister_screws": (
        "Screws and bolts/Special screws (Drywall, plastic, tapping, wood)"
        "/Plastic screws/Plastic countersunk or flat screws"
    ),
    # 权重里的泛类，没有对应目录，按最常见的形态兜底
    "nut": "Nuts/Hex nuts",
    "screw": "Screws and bolts/Cap screws/Hexagonal screws",
}

# 同名叶目录出现在多个分类下时，指定优先用哪个（手工具优先于电动工具）
PREFER_DIRS: dict[str, str] = {
    "screwdrivers": "Machines & Tools/Mechanical tools/Screwdrivers",
}

# 这些是「合成类 / 泛类」，中文名对得上但形状只是近似，规划前要留意
AMBIGUOUS = {
    "nut",
    "screw",
    "plastic_countersunk_or_plastic_fillister_screws",
}


def load_zh_maps() -> tuple[dict[str, str], dict[str, str]]:
    """从 environment.py 源码里取出两张中文名表。

    直接 import 会连带把 WRS / Panda3D 拉进来，真机这边不需要，所以用 AST 取。
    """
    src = ENV_PY.read_text(encoding="utf-8")

    def grab(var: str) -> dict[str, str]:
        m = re.search(var + r"\s*=\s*(\{.*?\n\})", src, re.S)
        if not m:
            raise RuntimeError(f"{ENV_PY} 里找不到 {var}")
        return ast.literal_eval(m.group(1))

    return grab("CATEGORY_ZH"), grab("LEAF_DIR_ZH")


def scan_leaf_dirs(root: Path) -> dict[str, list[Path]]:
    """叶目录名（小写）→ 该名字对应的相对路径列表（可能重名）。"""
    out: dict[str, list[Path]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if not any(f.lower().endswith(".stl") for f in filenames):
            continue
        rel = Path(dirpath).relative_to(root)
        out.setdefault(rel.name.lower(), []).append(rel)
    for paths in out.values():
        paths.sort(key=lambda p: p.as_posix().lower())
    return out


def first_stl(root: Path, rel_dir: Path) -> str | None:
    stls = sorted(
        (p for p in (root / rel_dir).iterdir() if p.suffix.lower() == ".stl"),
        key=lambda p: p.name.lower(),
    )
    return stls[0].name if stls else None


def zh_class_for(rel_dir: Path, category_zh: dict, leaf_zh: dict) -> str:
    parts = rel_dir.parts
    top, leaf = parts[0], parts[-1]
    return f"{category_zh.get(top, top)}-{leaf_zh.get(leaf, leaf)}"


def resolve(en_name: str, leaves: dict[str, list[Path]]) -> Path | None:
    if en_name in PREFER_DIRS:
        return Path(PREFER_DIRS[en_name])
    if en_name in OVERRIDES:
        return Path(OVERRIDES[en_name])
    cands = leaves.get(en_name.replace("_", " ").strip().lower())
    return cands[0] if cands else None


def main() -> int:
    p = argparse.ArgumentParser(description="生成 YOLO 类名 → 仿真类名/STL 映射表")
    p.add_argument("--weights", default="", help="YOLO 权重；默认取 vision_config 的默认权重")
    p.add_argument("--out", default=str(OUT_JSON))
    args = p.parse_args()

    from vision_config import default_weights

    weights = Path(args.weights) if args.weights else Path(default_weights())
    if not weights.is_file():
        print(f"[map] 找不到权重：{weights}")
        return 2
    if not PART_MODEL_DIR.is_dir():
        print(f"[map] 找不到 STL 库：{PART_MODEL_DIR}")
        return 2

    from ultralytics import YOLO

    names = YOLO(str(weights)).names or {}
    category_zh, leaf_zh = load_zh_maps()
    leaves = scan_leaf_dirs(PART_MODEL_DIR)

    classes: dict[str, dict] = {}
    unmapped: list[str] = []
    no_stl: list[str] = []
    for en in sorted(str(v) for v in names.values()):
        rel_dir = resolve(en, leaves)
        if rel_dir is None:
            unmapped.append(en)
            continue
        if not (PART_MODEL_DIR / rel_dir).is_dir():
            print(f"[map] 覆盖表里的目录不存在，请修正 OVERRIDES：{en} -> {rel_dir}")
            unmapped.append(en)
            continue
        stl_name = first_stl(PART_MODEL_DIR, rel_dir)
        if stl_name is None:
            no_stl.append(en)
            continue
        entry = {
            "zh": zh_class_for(rel_dir, category_zh, leaf_zh),
            "stl_dir": rel_dir.as_posix(),
            "stl": stl_name,
        }
        if en in AMBIGUOUS:
            entry["approx"] = True
        classes[en] = entry

    payload = {
        "note": "YOLO 英文类名 → 仿真中文类名 + STL。由 build_class_map.py 生成，可手改。",
        "weights": weights.name,
        "part_model_root": PART_MODEL_DIR.as_posix(),
        "classes": classes,
        "unmapped": unmapped,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[map] {len(classes)}/{len(names)} 类已映射 -> {out}")
    approx = [k for k, v in classes.items() if v.get("approx")]
    if approx:
        print(f"[map] 形状近似（合成类/泛类）{len(approx)} 个：{', '.join(approx)}")
    if no_stl:
        print(f"[map] 目录里没有 STL：{', '.join(no_stl)}")
    if unmapped:
        print(f"[map] 未映射 {len(unmapped)} 个（用到时请补 OVERRIDES）：{', '.join(unmapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
