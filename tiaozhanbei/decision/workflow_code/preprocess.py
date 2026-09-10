import json
import re

STORAGE_BOX_NAME = "收纳盒"
STORAGE_BOX_XYZ = [0.30, -0.20, 0.0]
PLACE_KEYWORDS = ("放到", "放入", "放置", "放进", "移到", "移动到", "装箱")
BIN_SLOT_OFFSETS = {
    0: [-0.08, 0.00, 0.0],
    1: [-0.08, 0.00, 0.0],
    2: [0.00, 0.00, 0.0],
    3: [0.08, 0.00, 0.0],
    4: [-0.08, -0.08, 0.0],
    5: [0.00, -0.08, 0.0],
    6: [0.08, -0.08, 0.0],
    7: [-0.08, 0.08, 0.0],
    8: [0.00, 0.08, 0.0],
    9: [0.08, 0.08, 0.0],
    10: [-0.08, -0.16, 0.0],
    11: [0.00, -0.16, 0.0],
    12: [0.08, -0.16, 0.0],
}


def wants_place(cmd: str) -> bool:
    text = cmd or ""
    if "搬运" in text:
        return False
    if any(k in text for k in PLACE_KEYWORDS):
        return True
    if "收纳盒" in text or "收纳箱" in text or "料箱" in text or "格子" in text or "入格" in text:
        return True
    return False


CN_COUNT = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}
CN_NUM_PAT = r"(?:\d+|十二|十一|十|[零一二两三四五六七八九])"
CLASS_ALIASES = (
    "螺丝刀",
    "扳手",
    "滚柱",
    "滚子",
    "螺母",
    "螺栓",
    "螺丝",
    "轴承",
    "齿轮",
    "垫圈",
    "铆钉",
    "台钳",
    "砂光",
    "刷子",
    "喇叭头",
)


def _to_count(raw):
    raw = (raw or "").strip()
    if str(raw).isdigit():
        return int(raw)
    if raw in CN_COUNT:
        return CN_COUNT[raw]
    if raw.startswith("十") and len(raw) == 2 and raw[1] in CN_COUNT:
        return 10 + CN_COUNT[raw[1]]
    return None


def parse_part_count(cmd: str):
    text = cmd or ""
    for pat in (
        rf"随便拿\s*({CN_NUM_PAT})\s*个",
        rf"前\s*({CN_NUM_PAT})\s*个",
        rf"(?:拿|取|抓)(?:取)?\s*({CN_NUM_PAT})\s*个(?:零件|工件|螺丝|螺钉)?",
        rf"({CN_NUM_PAT})\s*个(?:零件|工件|螺丝|螺钉)",
    ):
        m = re.search(pat, text)
        if not m:
            continue
        n = _to_count(m.group(1))
        if n is not None and 1 <= n <= 50:
            return n
    return None


def wants_pack_all_box(cmd: str) -> bool:
    text = cmd or ""
    if "多智能体" in text or "多机械臂" in text:
        return False
    all_parts = (
        "所有零件" in text
        or "全部零件" in text
        or "所有的零件" in text
        or "所有螺丝" in text
        or "全部螺丝" in text
        or (("所有" in text or "全部" in text) and ("零件" in text or "螺丝" in text))
    )
    if not all_parts:
        return False
    return bool(
        "收纳盒" in text
        or "收纳箱" in text
        or "放进" in text
        or "放入" in text
        or "放到" in text
    )


def extract_named_mentions(cmd: str, class_names=None):
    text = cmd or ""
    if not text:
        return []
    occupied = [False] * len(text)
    found = []

    def _mark(start, end, name):
        if start < 0 or end > len(text) or start >= end:
            return
        if any(occupied[start:end]):
            return
        for i in range(start, end):
            occupied[i] = True
        found.append((start, name))

    names = [n for n in (class_names or []) if n]
    for name in sorted(names, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(name, start)
            if i < 0:
                break
            _mark(i, i + len(name), name)
            start = i + len(name)
    for alias in sorted(CLASS_ALIASES, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(alias, start)
            if i < 0:
                break
            mapped = alias
            for name in names:
                if alias == "螺丝" and "螺丝刀" in name:
                    continue
                if name == alias or name.endswith("-" + alias) or alias in name:
                    mapped = name
                    break
            _mark(i, i + len(alias), mapped)
            start = i + len(alias)
    found.sort(key=lambda x: x[0])
    out = []
    for _, name in found:
        if name not in out:
            out.append(name)
    return out


def detect_intent(cmd: str, class_names=None) -> str:
    text = cmd or ""
    if "多智能体" in text or "多机械臂" in text:
        return "multi_agent_pack"
    if wants_pack_all_box(text):
        return "pack_all_box"
    if ("所有" in text or "全部" in text) and "装箱" in text:
        return "multi_agent_pack"
    if "搬运" in text or "搬走" in text:
        return "transport_box"
    if parse_slot_id(text) is not None and not (
        "螺丝" in text or "螺钉" in text or "喇叭头" in text
    ):
        if wants_place(text):
            return "pick_and_place"
        return "pick_only"
    if "对应格子" in text or "入格" in text:
        return "pack_bin_slots"
    if ("依次" in text or "逐个" in text or "一个个" in text) and (
        "格子" in text or "料盘" in text or "料箱" in text
    ):
        return "pack_bin_slots"
    if "格子" in text and ("螺丝" in text or "螺钉" in text or "喇叭头" in text):
        return "pack_bin_slots"
    names = class_names or []
    screw_n = sum(1 for n in names if n and any(k in str(n) for k in ("螺丝", "螺钉", "喇叭头")))
    if screw_n >= 6 and ("螺丝" in text or "螺钉" in text) and parse_part_count(text) is None:
        if "收纳盒" not in text and "收纳箱" not in text:
            if wants_place(text) or "随便" in text or "拿" in text:
                return "pack_bin_slots"
    if parse_part_count(text) is not None:
        return "pack_n_parts"
    if len(extract_named_mentions(text, class_names)) >= 2:
        return "multi_named_pick_place"
    if ("最多" in text and "区域" in text) or ("装箱" in text and ("区域" in text or "最多" in text)):
        return "pack_densest_region"
    if wants_place(text):
        return "pick_and_place"
    return "pick_only"


def slot_xyz(slot_id):
    base = list(STORAGE_BOX_XYZ)
    offset = BIN_SLOT_OFFSETS.get(int(slot_id), BIN_SLOT_OFFSETS[2])
    return [base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]]


def parse_slot_id(cmd: str):
    cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "零": 0}
    for pat in [
        r"第\s*([零一二三四五六七八九十0-9]+)\s*个?\s*格子?",
        r"格子\s*([0-9]+)",
        r"slot\s*([0-9]+)",
    ]:
        m = re.search(pat, cmd or "")
        if m:
            raw = m.group(1)
            if raw in cn_map:
                return cn_map[raw]
            if str(raw).isdigit():
                return int(raw)
    return None


def _strip_json_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def jobs_to_objects(jobs: list) -> list:
    """把 example_bin.json 的 jobs 转成工作流 objects 视觉格式。"""
    objects = []
    for j in jobs or []:
        if not isinstance(j, dict):
            continue
        coord = list(j.get("coordinate") or [0.0, 0.0, 0.0])
        while len(coord) < 3:
            coord.append(0.0)
        rpy = list(j.get("rpy") or [0.0, 0.0, float(j.get("angle") or 0.0)])
        while len(rpy) < 3:
            rpy.append(0.0)
        oid = j.get("object_id", j.get("id", len(objects) + 1))
        cls = j.get("object") or j.get("class") or ""
        if not cls:
            continue
        objects.append(
            {
                "id": oid,
                "class": cls,
                "bbox": j.get("bbox") or [0, 0, 10, 10],
                "visible_pixels": int(j.get("visible_pixels") or 1000),
                "pose_6d": {
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                    "roll": float(rpy[0]),
                    "pitch": float(rpy[1]),
                    "yaw": float(rpy[2]),
                },
                "state": j.get("state"),
                "pose_state": j.get("pose_state") or j.get("state"),
                "slot_id": j.get("slot_id"),
                "dest_coordinate": j.get("dest_coordinate"),
            }
        )
    return objects


def normalize_xyz(values, default=None):
    default = default or STORAGE_BOX_XYZ
    if not isinstance(values, (list, tuple)):
        return list(default)
    nums = [float(v) for v in values]
    while len(nums) < 3:
        nums.append(0.0)
    return nums[:3]


def resolve_storage_xyz(data: dict):
    """优先 target_container，其次 scene_layout.storage_box（pos_m / origin_xy_m），最后默认收纳盒。"""
    tc = data.get("target_container") or {}
    if isinstance(tc, dict) and tc.get("xyz") is not None:
        xyz = normalize_xyz(tc["xyz"])
        if max(abs(v) for v in xyz) > 1e-6:
            return xyz

    storage = (data.get("scene_layout") or {}).get("storage_box") or {}
    if isinstance(storage, dict):
        if storage.get("pos_m"):
            return normalize_xyz(storage["pos_m"])
        # 仿真同学新标注：origin_xy_m + floor_z_m
        if storage.get("origin_xy_m"):
            ox, oy = storage["origin_xy_m"][:2]
            z = float(storage.get("floor_z_m", 0.0))
            return [float(ox), float(oy), z]

    return list(STORAGE_BOX_XYZ)


def _slot_world_from_layout(storage: dict, slot: dict) -> list:
    ox, oy = storage.get("origin_xy_m") or [STORAGE_BOX_XYZ[0], STORAGE_BOX_XYZ[1]]
    lx, ly = slot.get("local_xy_m") or [0.0, 0.0]
    z = float(slot.get("floor_z_m", storage.get("floor_z_m", 0.0015)))
    return [float(ox) + float(lx), float(oy) + float(ly), z]


def _attach_slot_dests(objects: list, data: dict) -> list:
    """若标注带 scene_layout.storage_box.slots，给 objects 补 slot_id / dest_coordinate。"""
    storage = (data.get("scene_layout") or {}).get("storage_box") or {}
    slots = storage.get("slots") or []
    if not slots:
        return objects
    by_id = {int(s["slot_id"]): s for s in slots if s.get("slot_id") is not None}
    out = []
    for i, obj in enumerate(objects):
        item = dict(obj)
        if item.get("dest_coordinate") is None:
            sid = item.get("slot_id")
            if sid is None:
                try:
                    sid = int(item.get("id") or (i + 1)) - 1
                except Exception:
                    sid = i
                item["slot_id"] = sid
            slot = by_id.get(int(sid))
            if slot is not None:
                item["dest_coordinate"] = _slot_world_from_layout(storage, slot)
                item.setdefault("destination", f"料盘格子{sid}")
        out.append(item)
    return out


def main(user_cmd: str, vision_objects: str) -> dict:
    if isinstance(vision_objects, str):
        raw = _strip_json_fence(vision_objects)
        data = json.loads(raw) if raw.strip() else {}
    else:
        data = vision_objects or {}

    if isinstance(data, list):
        # 可能是 objects 列表，也可能是 jobs 列表
        if data and isinstance(data[0], dict) and ("coordinate" in data[0] or "object_id" in data[0]) and "pose_6d" not in data[0]:
            data = {"objects": jobs_to_objects(data), "target_container": {"xyz": STORAGE_BOX_XYZ}}
        else:
            data = {"objects": data, "target_container": {"xyz": STORAGE_BOX_XYZ}}

    # 兼容搭档 example_bin.json：顶层是 jobs 而不是 objects
    if not data.get("objects") and data.get("jobs"):
        data = dict(data)
        data["objects"] = jobs_to_objects(data.get("jobs") or [])

    # annotations.json：用 scene_layout.slots 补全每件 dest
    if data.get("objects"):
        data = dict(data)
        data["objects"] = _attach_slot_dests(list(data.get("objects") or []), data)

    # —— 闭环回传模式：嵌套对象改成 JSON 字符串，避免 Dify Depth limit ——
    mode = str(data.get("mode") or "").lower()
    if mode == "reflect" or (data.get("post_vision_objects") and data.get("last_task")):
        reflect_payload = {
            "__reflect__": True,
            "last_task_json": json.dumps(data.get("last_task") or {}, ensure_ascii=False),
            "memory_json": json.dumps(
                data.get("memory")
                or {
                    "finished_parts": [],
                    "fail_record": [],
                    "current_retry": 0,
                    "max_retry": 3,
                    "box_full": False,
                    "box_capacity": 6,
                },
                ensure_ascii=False,
            ),
            "post_vision_json": json.dumps(data.get("post_vision_objects") or {}, ensure_ascii=False),
            "user_cmd": user_cmd,
            "class": "__reflect__",
            "xyz": [0.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0],
            "visible_pixels": 1,
            "id": -1,
            "pose_state": "normal",
        }
        llm_context = json.dumps(
            {
                "user_cmd": user_cmd,
                "intent": "reflect",
                "mode": "reflect",
                "objects": [],
            },
            ensure_ascii=False,
        )
        return {
            "parts_3d": [reflect_payload],
            "target_xyz": list(STORAGE_BOX_XYZ),
            "llm_context": llm_context,
        }

    parts_3d = []
    for obj in data.get("objects", []):
        if not isinstance(obj, dict):
            continue
        # 有 pose_6d 即可；无 bbox/pixels 时给默认值（example_bin 转换后也适用）
        p = obj.get("pose_6d")
        if not isinstance(p, dict):
            continue
        pixels = int(obj.get("visible_pixels") or 0)
        if pixels <= 0:
            pixels = 1000
        bbox = obj.get("bbox")
        if bbox is None:
            bbox = [0, 0, 10, 10]
        # 姿态：正常/倒放/倾倒；兼容 example_bin 的 inverted/fallen
        _pose_aliases = {
            "normal": "normal",
            "正常": "normal",
            "upside_down": "upside_down",
            "倒放": "upside_down",
            "倒置": "upside_down",
            "inverted": "upside_down",
            "tilt": "tilt",
            "tilted": "tilt",
            "倾倒": "tilt",
            "倾斜": "tilt",
            "fallen": "tilt",
        }
        pose_state = obj.get("pose_state") or obj.get("state") or ""
        if pose_state:
            ps = str(pose_state).strip()
            pose_state = _pose_aliases.get(ps) or _pose_aliases.get(ps.lower()) or "normal"
        else:
            oid = int(obj.get("id") or 0)
            rem = oid % 5
            if rem == 1:
                pose_state = "upside_down"
            elif rem == 2:
                pose_state = "tilt"
            else:
                pose_state = "normal"
        item = {
            "id": obj["id"],
            "class": obj["class"],
            "xyz": [float(p["x"]), float(p["y"]), float(p["z"])],
            "rpy": [float(p["roll"]), float(p["pitch"]), float(p["yaw"])],
            "visible_pixels": pixels,
            "pose_state": pose_state,
        }
        if obj.get("slot_id") is not None:
            item["slot_id"] = obj.get("slot_id")
        if obj.get("dest_coordinate") is not None:
            item["dest_coordinate"] = list(obj.get("dest_coordinate"))
        parts_3d.append(item)

    do_place = wants_place(user_cmd or "")
    intent = detect_intent(user_cmd or "", [p["class"] for p in parts_3d])
    if intent in (
        "pack_densest_region",
        "transport_box",
        "multi_agent_pack",
        "pack_bin_slots",
        "pack_n_parts",
        "pack_all_box",
        "multi_named_pick_place",
    ):
        do_place = True
    slot_id = parse_slot_id(user_cmd or "") if do_place and intent != "pack_bin_slots" else None
    if intent == "pack_bin_slots":
        # 全量入格：目标点用各零件自己的 dest；这里放默认收纳盒供兼容
        target_xyz = resolve_storage_xyz(data)
        dest_name = "料盘格子"
    elif do_place and slot_id is not None:
        target_xyz = slot_xyz(slot_id)
        dest_name = f"料箱格子{slot_id}"
    elif do_place:
        target_xyz = resolve_storage_xyz(data)
        dest_name = STORAGE_BOX_NAME
    else:
        target_xyz = resolve_storage_xyz(data)  # 仍输出默认点，供下游兼容
        dest_name = None

    llm_context = json.dumps(
        {
            "user_cmd": user_cmd,
            "intent": intent,
            "objects": [
                {
                    "id": o["id"],
                    "class": o["class"],
                    "xyz": o["xyz"],
                    "yaw": o["rpy"][2],
                    "visible_pixels": o["visible_pixels"],
                    "pose_state": o.get("pose_state", "normal"),
                    "slot_id": o.get("slot_id"),
                }
                for o in parts_3d
            ],
            "target_container_name": dest_name,
            "target_container_xyz": target_xyz if do_place else None,
            "slot_id": slot_id,
        },
        ensure_ascii=False,
    )

    # 只能返回节点声明的 3 个输出，否则 Dify 报 Not all output parameters are validated
    return {
        "parts_3d": parts_3d,
        "target_xyz": target_xyz if do_place else [0.0, 0.0, 0.0],
        "llm_context": llm_context,
    }
