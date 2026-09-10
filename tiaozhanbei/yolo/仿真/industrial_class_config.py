"""工业工具检测类别配置：8 类原始 / 4 类合并。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"

# ---------------------------------------------------------------------------
# 8 类（原始 STL 文件夹 → 训练标签）
# ---------------------------------------------------------------------------

CLASS_NAMES_8: list[str] = [
    "bearing",
    "gear",
    "machine_tool",
    "nut",
    "rivet",
    "roller_bearing",
    "screw_bolt",
    "washer",
]

FOLDER_TO_CLASS_8: dict[str, int] = {
    "Bearings": 0,
    "Gears": 1,
    "Machines & Tools": 2,
    "Nuts": 3,
    "Rivets": 4,
    "Roller bearings": 5,
    "Screws and bolts": 6,
    "Washers": 7,
}

# ---------------------------------------------------------------------------
# 4 类合并（降低 nut↔screw_bolt、bearing↔washer 等混淆）
#   round_part  : bearing + roller_bearing + washer
#   gear        : gear
#   machine_tool: machine_tool
#   fastener    : nut + rivet + screw_bolt
# ---------------------------------------------------------------------------

CLASS_NAMES_MERGED4: list[str] = [
    "round_part",
    "gear",
    "machine_tool",
    "fastener",
]

MAP_8_TO_MERGED4: dict[int, int] = {
    0: 0,  # bearing
    5: 0,  # roller_bearing
    7: 0,  # washer
    1: 1,  # gear
    2: 2,  # machine_tool
    3: 3,  # nut
    4: 3,  # rivet
    6: 3,  # screw_bolt
}

# 数据生成时提高小零件 / 难检类采样（8 类 id）
FOCUS_CLASS_IDS_8: set[int] = {0, 2, 3, 4, 5, 7}
FOCUS_CLASS_WEIGHTS_8: dict[int, float] = {
    0: 3.0,   # bearing
    2: 4.0,   # machine_tool
    3: 2.5,   # nut
    4: 2.5,   # rivet
    5: 3.0,   # roller_bearing
    7: 2.5,   # washer
}

CLASS_SETS = {
    "8class": {
        "names": CLASS_NAMES_8,
        "folder_to_class": FOLDER_TO_CLASS_8,
        "yaml": DATASET_ROOT / "industrial_tools.yaml",
        "map_from_8": {i: i for i in range(8)},
    },
    "merged4": {
        "names": CLASS_NAMES_MERGED4,
        "folder_to_class": FOLDER_TO_CLASS_8,
        "yaml": DATASET_ROOT / "industrial_tools_merged4.yaml",
        "map_from_8": MAP_8_TO_MERGED4,
    },
}


def get_class_set(name: str) -> dict:
    if name not in CLASS_SETS:
        raise ValueError(f"未知 class-set: {name}，可选 {list(CLASS_SETS)}")
    return CLASS_SETS[name]


def class_names_dict(class_set: str) -> dict[int, str]:
    cfg = get_class_set(class_set)
    return {i: n for i, n in enumerate(cfg["names"])}


def map_class_id(old_id: int, class_set: str) -> int:
    return get_class_set(class_set)["map_from_8"][old_id]


def map_class_name(old_id: int, class_set: str) -> str:
    new_id = map_class_id(old_id, class_set)
    return get_class_set(class_set)["names"][new_id]


def write_yaml(class_set: str, output_dir: Path | None = None) -> Path:
    cfg = get_class_set(class_set)
    root = output_dir or DATASET_ROOT
    yaml_path = root / cfg["yaml"].name if output_dir else cfg["yaml"]
    lines = [
        f"# industrial class-set: {class_set}",
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for idx, name in enumerate(cfg["names"]):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path
