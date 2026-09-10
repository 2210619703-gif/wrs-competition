"""
自然语言指令解析：对齐项目书 / 答疑示例
- 「给我一把螺丝刀」
- 「取出左侧的扳手」
- 「帮我把滚柱放到料箱的第三个格子中」
- 「帮我把零件最多的区域装箱」
- 「帮我搬运一下料箱盒」
"""
import math
import re
from collections import defaultdict

from common_config import (
    PLACE_KEYWORDS,
    STORAGE_BOX_NAME,
    STORAGE_BOX_XYZ,
    default_destination,
    parse_intent,
    slot_xyz,
    wants_place,
)


def parse_slot_id(cmd: str):
    """解析「第N个格子 / 第N格」."""
    patterns = [
        r"第\s*([一二三四五六123456])\s*个?\s*格子?",
        r"格子\s*([123456])",
        r"slot\s*([123456])",
    ]
    cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
    for pat in patterns:
        m = re.search(pat, cmd)
        if m:
            raw = m.group(1)
            if raw in cn_map:
                return cn_map[raw]
            return int(raw)
    return None


def parse_spatial(cmd: str):
    if "左侧" in cmd or "左边" in cmd:
        return "left"
    if "右侧" in cmd or "右边" in cmd:
        return "right"
    if "中间" in cmd:
        return "center"
    if "最近" in cmd:
        return "nearest"
    if "最大" in cmd:
        return "largest"
    if "最靠前" in cmd:
        return "front"
    if "最靠后" in cmd:
        return "back"
    return None


def extract_object_hint(cmd: str, class_names):
    """从指令中匹配数据集里的完整 class 名。"""
    # 优先完整匹配
    for name in sorted(class_names, key=len, reverse=True):
        if name and name in cmd:
            return name
    # 细分名匹配（「大类-细分」的后半段）
    for name in class_names:
        if "-" in name:
            sub = name.split("-", 1)[1]
            if sub and sub in cmd:
                return name
    # 常见口语映射
    aliases = {
        "螺丝刀": "螺丝刀",
        "扳手": "扳手",
        "滚柱": "滚子",
        "滚子": "滚子",
        "螺母": "螺母",
        "螺丝": "螺丝",
        "螺栓": "螺栓",
        "轴承": "轴承",
        "齿轮": "齿轮",
        "垫圈": "垫圈",
        "铆钉": "铆钉",
        "台钳": "台钳",
        "砂光": "砂光",
        "刷子": "刷子",
    }
    for alias, key in aliases.items():
        if alias in cmd:
            for name in class_names:
                if key in name:
                    return name
    return None


def extract_all_object_hints(cmd: str, class_names):
    from common_config import extract_named_mentions

    return extract_named_mentions(cmd, class_names)


def select_by_spatial(objs, spatial: str):
    if not objs:
        return None
    if spatial == "largest":
        return max(objs, key=lambda o: o.get("visible_pixels", 0))
    if spatial == "nearest":
        return min(
            objs,
            key=lambda o: math.sqrt(o["xyz"][0] ** 2 + o["xyz"][1] ** 2),
        )
    if spatial == "left":
        return min(objs, key=lambda o: o["xyz"][0])
    if spatial == "right":
        return max(objs, key=lambda o: o["xyz"][0])
    if spatial == "center":
        mean_x = sum(o["xyz"][0] for o in objs) / len(objs)
        return min(objs, key=lambda o: abs(o["xyz"][0] - mean_x))
    if spatial == "front":
        return max(objs, key=lambda o: o["xyz"][1])
    if spatial == "back":
        return min(objs, key=lambda o: o["xyz"][1])
    return max(objs, key=lambda o: o.get("visible_pixels", 0))


def resolve_destination_from_cmd(cmd: str):
    """
    返回 (destination_name, dest_xyz, slot_id|None)。
    仅拿起时三者均为 None。
    """
    if not wants_place(cmd):
        return None, None, None
    slot_id = parse_slot_id(cmd)
    if slot_id is not None:
        return f"料箱格子{slot_id}", slot_xyz(slot_id), slot_id
    if "料箱" in cmd or "收纳盒" in cmd or "收纳箱" in cmd or any(k in cmd for k in PLACE_KEYWORDS):
        name, xyz = default_destination()
        return name, xyz, None
    return STORAGE_BOX_NAME, list(STORAGE_BOX_XYZ), None


def cluster_parts_by_grid(parts_3d, cell: float = 0.20):
    """按 XY 网格聚类，返回 {grid_key: [parts...]}。"""
    buckets = defaultdict(list)
    for p in parts_3d:
        xyz = p.get("xyz") or [0, 0, 0]
        key = (math.floor(float(xyz[0]) / cell), math.floor(float(xyz[1]) / cell))
        buckets[key].append(p)
    return buckets


def select_densest_region(parts_3d, cell: float = 0.20):
    """
    选零件最多的区域。
    返回 (region_parts, region_key, region_count)
    """
    if not parts_3d:
        return [], None, 0
    buckets = cluster_parts_by_grid(parts_3d, cell=cell)
    best_key = max(buckets.keys(), key=lambda k: (len(buckets[k]), -abs(k[0]), -abs(k[1])))
    region = list(buckets[best_key])
    # 区内按距原点近→远，便于仿真依次抓
    region.sort(key=lambda o: math.hypot(o["xyz"][0], o["xyz"][1]))
    return region, best_key, len(region)


def choose_target(cmd: str, parts_3d):
    """
    根据指令选择目标零件。
    parts_3d 项需含 class/xyz/visible_pixels/rpy
    返回 (obj, reason, status)
    """
    if not parts_3d:
        return None, "无有效零件", "error"

    class_names = [p["class"] for p in parts_3d]
    hint = extract_object_hint(cmd, class_names)
    spatial = parse_spatial(cmd)

    candidates = list(parts_3d)
    if hint:
        candidates = [p for p in parts_3d if hint in p["class"] or p["class"] in hint]
        if not candidates:
            candidates = list(parts_3d)

    if spatial and candidates:
        target = select_by_spatial(candidates, spatial)
        reason = ""
        status = "valid" if hint or spatial else "invalid"
        if not hint and spatial:
            reason = f"按空间关系[{spatial}]选择零件{target['class']}"
        return target, reason, status if hint or spatial else "invalid"

    if hint:
        target = max(candidates, key=lambda o: o.get("visible_pixels", 0))
        return target, "", "valid"

    # 模糊指令 / 抓取指定工业零件
    target = max(parts_3d, key=lambda o: o.get("visible_pixels", 0))
    reason = f"指令未明确指定零件名称，已匹配最大可见零件{target['class']}"
    return target, reason, "invalid"
