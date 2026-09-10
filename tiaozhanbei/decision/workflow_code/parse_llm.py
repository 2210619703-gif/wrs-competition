import json
import math
import re

STORAGE_BOX_NAME = "收纳盒"
STORAGE_BOX_XYZ = [0.30, -0.20, 0.0]
PLACE_AREA_XYZ = [0.55, -0.35, 0.0]
BOX_CAPACITY = 6
PLACE_KEYWORDS = ("放到", "放入", "放置", "放进", "移到", "移动到", "装箱")
TRANSPORT_KEYWORDS = ("搬运", "搬走", "移送料箱", "搬料箱")
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
    10: [-0.04, -0.04, 0.0],
    11: [0.04, -0.04, 0.0],
}


def _shallow_task_for_dify(task: dict) -> dict:
    """避免 Dify 代码节点 Depth limit：tasks/batch 改成 JSON 字符串。"""
    if not isinstance(task, dict):
        return task
    out = dict(task)
    tasks = out.get("tasks")
    if isinstance(tasks, list) and tasks:
        out["tasks_json"] = json.dumps(tasks, ensure_ascii=False)
        out["actions_flat"] = [t.get("action") for t in tasks if isinstance(t, dict)]
        out["tasks_count"] = len(tasks)
        out["tasks"] = []
    if out.get("batch") is not None:
        out["batch_json"] = json.dumps(out["batch"], ensure_ascii=False)
        out.pop("batch", None)
    if out.get("agents") is not None:
        out["agents_json"] = json.dumps(out["agents"], ensure_ascii=False)
        out.pop("agents", None)
    return out


def wants_transport(cmd: str) -> bool:
    text = cmd or ""
    if any(k in text for k in TRANSPORT_KEYWORDS):
        return True
    if "搬运" in text and ("料箱" in text or "盒子" in text or "盒" in text):
        return True
    return False


def wants_pack_region(cmd: str) -> bool:
    text = cmd or ""
    if "最多" in text and ("区域" in text or "区" in text):
        return True
    if "装箱" in text and ("区域" in text or "最多" in text):
        return True
    if "零件最多" in text:
        return True
    return False


def wants_place(cmd: str) -> bool:
    text = cmd or ""
    if wants_transport(text):
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
        rf"任意(?:拿|取|抓)?\s*({CN_NUM_PAT})\s*个",
        rf"前\s*({CN_NUM_PAT})\s*个",
        rf"演示前\s*({CN_NUM_PAT})",
        rf"跑前\s*({CN_NUM_PAT})",
        rf"(?:拿|取|抓)(?:取)?\s*({CN_NUM_PAT})\s*个(?:零件|工件|螺丝|螺钉)?",
        rf"({CN_NUM_PAT})\s*个(?:零件|工件|螺丝|螺钉)",
        rf"执行\s*({CN_NUM_PAT})\s*个",
    ):
        m = re.search(pat, text)
        if not m:
            continue
        span = text[max(0, m.start() - 1) : m.end() + 2]
        if "格子" in span and "第" in text[max(0, m.start() - 4) : m.start()]:
            continue
        n = _to_count(m.group(1))
        if n is not None and 1 <= n <= 50:
            return n
    return None


def parse_object_index(cmd: str):
    """第3个螺丝 / 第3颗。不含「第N个格子」。"""
    text = cmd or ""
    m = re.search(rf"第\s*({CN_NUM_PAT})\s*[个颗枚](?!\s*格子)", text)
    if not m:
        return None
    return _to_count(m.group(1))


def wants_into_slots(cmd: str) -> bool:
    text = cmd or ""
    if "收纳盒" in text or "收纳箱" in text:
        return False
    if "对应格子" in text or "入格" in text:
        return True
    if "料盘" in text:
        return True
    if "格子" in text and ("螺丝" in text or "螺钉" in text or "喇叭头" in text):
        return True
    if ("依次" in text or "逐个" in text or "一个个" in text) and (
        "格子" in text or "料盘" in text or "料箱" in text
    ):
        return True
    return False


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
    if wants_into_slots(text):
        return False
    return bool(
        "收纳盒" in text
        or "收纳箱" in text
        or "放进" in text
        or "放入" in text
        or "放到" in text
    )


def wants_multi_agent_cmd(cmd: str) -> bool:
    text = cmd or ""
    if "多智能体" in text or "多机械臂" in text:
        return True
    if wants_pack_all_box(text):
        return False
    if ("所有" in text or "全部" in text) and "装箱" in text:
        return True
    return False


def wants_pack_bin_slots_cmd(cmd: str) -> bool:
    text = cmd or ""
    if parse_object_index(text) is not None and parse_slot_id(text) is not None:
        return False
    if re.search(r"第\s*[零一二三四五六七八九十0-9]+\s*个?\s*格子?", text) and parse_object_index(text) is None:
        return False
    return wants_into_slots(text)


def is_ec_scene(class_names=None, parts_3d=None) -> bool:
    parts = parts_3d or []
    slotted = sum(1 for p in parts if isinstance(p, dict) and p.get("slot_id") is not None)
    if slotted >= 3:
        return True
    names = list(class_names or []) or [p.get("class") for p in parts if isinstance(p, dict)]
    n = sum(1 for n in names if n and any(k in str(n) for k in ("螺丝", "螺钉", "喇叭头")))
    return n >= 6


def parse_high_intent(cmd: str, class_names=None, parts_3d=None) -> str:
    text = cmd or ""
    ec = is_ec_scene(class_names, parts_3d)
    if wants_multi_agent_cmd(text):
        return "multi_agent_pack"
    if wants_pack_all_box(text):
        return "pack_all_box"
    if wants_transport(cmd):
        return "transport_box"
    # 第N个螺丝 → 第M格：单件
    if parse_object_index(text) is not None:
        if wants_place(text) or wants_into_slots(text) or parse_slot_id(text) is not None:
            return "pick_and_place"
        return "pick_only"
    if parse_slot_id(text) is not None:
        if wants_place(text):
            return "pick_and_place"
        return "pick_only"
    if wants_pack_bin_slots_cmd(text):
        return "pack_bin_slots"
    # 螺丝场景：没说数量、也没说收纳盒 → 完整 EC batch 入格
    if ec and ("螺丝" in text or "螺钉" in text or "喇叭头" in text) and parse_part_count(text) is None:
        if "收纳盒" not in text and "收纳箱" not in text and "搬运" not in text:
            if wants_place(text) or "随便" in text or "拿" in text or "放" in text:
                return "pack_bin_slots"
    n = parse_part_count(text)
    if n is not None:
        if wants_into_slots(text):
            return "pack_bin_slots"
        return "pack_n_parts"
    names = extract_named_mentions(text, class_names)
    if len(names) >= 2:
        return "multi_named_pick_place"
    if wants_pack_region(cmd):
        return "pack_densest_region"
    if wants_place(cmd):
        return "pick_and_place"
    return "pick_only"


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


def slot_xyz(slot_id):
    base = list(STORAGE_BOX_XYZ)
    offset = BIN_SLOT_OFFSETS.get(int(slot_id), BIN_SLOT_OFFSETS[2])
    return [base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]]


def repair_broken_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return ""
    balance = 0
    cut_pos = None
    for idx, char in enumerate(text[start:]):
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            if balance == 0:
                cut_pos = start + idx + 1
                break
    if cut_pos:
        return text[start:cut_pos]
    return ""


def strip_think_blocks(text: str) -> str:
    """去掉模型思考标签，避免吃到 think 里残缺 JSON。"""
    if not text:
        return ""
    patterns = [
        r"<\|think\|>.*?</\|think\|>",
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    # 未闭合的思考块：从标签起删到文末或下一个 JSON 之前
    text = re.sub(r"<\|?think\|?>[\s\S]*?(?=\{|$)", "", text, flags=re.IGNORECASE)
    return text.strip()


def find_json_candidates(text: str):
    """按括号匹配找出所有完整 {...} 候选，优先靠后的。"""
    cands = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        balance = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                balance += 1
            elif text[j] == "}":
                balance -= 1
                if balance == 0:
                    cands.append(text[i : j + 1])
                    i = j + 1
                    break
        else:
            break
    return cands


def extract_json_text(text: str) -> str:
    if not text:
        return ""
    text = strip_think_blocks(text.strip())

    block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if block:
        repaired = repair_broken_json(block.group(1).strip())
        if repaired:
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass

    cands = find_json_candidates(text)
    # 从后往前找第一个能 loads 的完整对象（避开 think 里残缺草稿）
    for cand in reversed(cands):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and ("action" in obj or "object" in obj or "task_id" in obj):
                return cand
        except Exception:
            continue
    for cand in reversed(cands):
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue

    repaired = repair_broken_json(text)
    if repaired:
        try:
            json.loads(repaired)
            return repaired
        except Exception:
            pass
    return ""


def normalize_xyz(values):
    if not isinstance(values, (list, tuple)):
        return list(STORAGE_BOX_XYZ)
    nums = [float(v) for v in values]
    while len(nums) < 3:
        nums.append(0.0)
    return nums[:3]


def is_valid_desk_coord(coord) -> bool:
    if not isinstance(coord, list) or len(coord) != 3:
        return False
    if not all(isinstance(v, (int, float)) for v in coord):
        return False
    x, y, z = coord
    if max(abs(x), abs(y), abs(z)) > 5.0:
        return False
    return abs(x) <= 2.0 and abs(y) <= 2.0 and abs(z) <= 1.0


def find_max_pixel_obj(parts_3d):
    if not parts_3d:
        return None
    return max(parts_3d, key=lambda p: p["visible_pixels"])


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


def parse_spatial(cmd: str):
    if any(k in (cmd or "") for k in ("左侧", "左边")):
        return "left"
    if any(k in (cmd or "") for k in ("右侧", "右边")):
        return "right"
    if "中间" in (cmd or ""):
        return "center"
    if "最近" in (cmd or ""):
        return "nearest"
    if "最大" in (cmd or ""):
        return "largest"
    return None


def select_by_spatial(objs, spatial: str):
    if spatial == "largest":
        return max(objs, key=lambda o: o.get("visible_pixels", 0))
    if spatial == "nearest":
        return min(objs, key=lambda o: math.sqrt(o["xyz"][0] ** 2 + o["xyz"][1] ** 2))
    if spatial == "left":
        return min(objs, key=lambda o: o["xyz"][0])
    if spatial == "right":
        return max(objs, key=lambda o: o["xyz"][0])
    if spatial == "center":
        mean_x = sum(o["xyz"][0] for o in objs) / len(objs)
        return min(objs, key=lambda o: abs(o["xyz"][0] - mean_x))
    return max(objs, key=lambda o: o.get("visible_pixels", 0))


def find_object(parts_3d, object_name: str):
    if not object_name:
        return None
    for p in parts_3d:
        if p.get("class") == object_name:
            return p
    for p in parts_3d:
        cls = p.get("class", "")
        if object_name in cls or cls in object_name:
            return p
    aliases = (
        "螺丝刀",
        "扳手",
        "滚柱",
        "滚子",
        "螺母",
        "螺丝",
        "喇叭头",
        "轴承",
        "齿轮",
        "台钳",
        "刷子",
    )
    for alias in aliases:
        if alias in object_name:
            for p in parts_3d:
                if alias in p.get("class", ""):
                    return p
    return None


def resolve_dest(destination, target_xyz, user_hint=""):
    if not wants_place(user_hint):
        # 仅拿起：无放置目标
        dest_name = str(destination or "").strip()
        if dest_name and dest_name not in ("", "null", "None", "无", STORAGE_BOX_NAME):
            if "收纳" in dest_name or "料箱" in dest_name or "格子" in dest_name:
                pass  # 仍可能被 LLM 误写，下面再清
            else:
                return None, None, None
        return None, None, None
    slot_id = parse_slot_id(user_hint or "")
    if slot_id is not None:
        return f"料箱格子{slot_id}", slot_xyz(slot_id), slot_id
    dest_name = str(destination or "").strip()
    if dest_name in ("", "null", "None", "无"):
        dest_name = STORAGE_BOX_NAME
    if dest_name == STORAGE_BOX_NAME or "收纳" in dest_name or "料箱" in dest_name:
        if "格子" in dest_name:
            return dest_name, normalize_xyz(target_xyz), None
        return STORAGE_BOX_NAME, list(STORAGE_BOX_XYZ), None
    xyz = normalize_xyz(target_xyz)
    if max(abs(v) for v in xyz) < 1e-6:
        return STORAGE_BOX_NAME, list(STORAGE_BOX_XYZ), None
    return dest_name, xyz, None


def default_error_task(reason: str) -> dict:
    return {
        "task_id": "error",
        "action": "idle",
        "object": "",
        "coordinate": [0.0, 0.0, 0.0],
        "dest_coordinate": None,
        "destination": None,
        "angle": 0.0,
        "rpy": [0.0, 0.0, 0.0],
        "retry": 0,
        "max_retry": 3,
        "status": "error",
        "reason": reason,
        "slot_id": None,
    }


# 与 pose_state.py / example_bin 对齐：inverted→倒放，fallen→倾倒
_POSE_ALIASES = {
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


def _normalize_pose(raw):
    if raw is None:
        return "normal"
    t = str(raw).strip()
    if not t:
        return "normal"
    if t in _POSE_ALIASES:
        return _POSE_ALIASES[t]
    low = t.lower()
    return _POSE_ALIASES.get(low, "normal")


def _needs_reorient(pose_state):
    return _normalize_pose(pose_state) in ("upside_down", "tilt")


def _infer_pose(obj):
    pose_state = obj.get("pose_state") or obj.get("state") or ""
    if isinstance(pose_state, str) and pose_state.strip():
        key = pose_state.strip()
        low = key.lower()
        if key in _POSE_ALIASES or low in _POSE_ALIASES:
            return _normalize_pose(pose_state)
    oid = int(obj.get("id") or 0)
    rem = oid % 5
    if rem == 1:
        return "upside_down"
    if rem == 2:
        return "tilt"
    return "normal"


def _densest_region(parts_3d, cell=0.20):
    from collections import defaultdict

    buckets = defaultdict(list)
    for p in parts_3d:
        xyz = p.get("xyz") or [0, 0, 0]
        key = (math.floor(float(xyz[0]) / cell), math.floor(float(xyz[1]) / cell))
        buckets[key].append(p)
    if not buckets:
        return [], None
    best = max(buckets.keys(), key=lambda k: (len(buckets[k]), -abs(k[0]), -abs(k[1])))
    region = list(buckets[best])
    region.sort(key=lambda o: math.hypot(o["xyz"][0], o["xyz"][1]))
    return region, best


def _one_part_steps(part, dest, dest_name, user_cmd, step0=0, skip_on_fail=False):
    obj = part.get("class", "")
    pick_xyz = list(part.get("xyz") or [0, 0, 0])
    rpy = list(part.get("rpy") or [0, 0, 0])
    angle = float(rpy[2] if len(rpy) > 2 else 0)
    pose_state = _infer_pose(part)
    steps = []
    step = step0
    step += 1
    steps.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "perceive",
            "object": obj,
            "coordinate": pick_xyz,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": rpy,
            "retry": 0,
            "max_retry": 3,
            "status": "pending",
            "reason": "识别并判断正常/倒放/倾倒",
            "original_cmd": user_cmd,
            "pose_state": pose_state,
        }
    )
    step += 1
    steps.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "pick",
            "object": obj,
            "coordinate": pick_xyz,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": rpy,
            "retry": 0,
            "max_retry": 3,
            "status": "pending",
            "reason": "",
            "original_cmd": user_cmd,
        }
    )
    place_rpy = rpy
    pose_state = _normalize_pose(pose_state)
    if _needs_reorient(pose_state):
        step += 1
        steps.append(
            {
                "task_id": f"t{step:03d}",
                "step": step,
                "action": "adjust_pose",
                "object": obj,
                "coordinate": pick_xyz,
                "dest_coordinate": dest,
                "destination": dest_name,
                "angle": angle,
                "rpy": [0.0, 0.0, angle],
                "retry": 0,
                "max_retry": 3,
                "status": "pending",
                "reason": "倒放/倾倒翻正后再放置",
                "original_cmd": user_cmd,
                "target_pose_state": "normal",
            }
        )
        place_rpy = [0.0, 0.0, angle]
    step += 1
    steps.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "move",
            "object": obj,
            "coordinate": dest,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": place_rpy,
            "retry": 0,
            "max_retry": 3,
            "status": "pending",
            "reason": "移至放置点",
            "original_cmd": user_cmd,
        }
    )
    step += 1
    steps.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "place",
            "object": obj,
            "coordinate": dest,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": place_rpy,
            "retry": 0,
            "max_retry": 3,
            "status": "pending",
            "reason": "",
            "original_cmd": user_cmd,
            "on_fail": {
                "action": "skip_continue" if skip_on_fail else "retry_place",
                "max_retry": 3,
                "reason": "该件规划/执行失败则跳过，继续下一件"
                if skip_on_fail
                else "失败则重感知-调姿-再放",
            },
        }
    )
    if skip_on_fail:
        for t in steps:
            t["skip_on_fail"] = True
            if t.get("action") != "place":
                t["on_fail"] = {
                    "action": "skip_continue",
                    "reason": "该件规划/执行失败则跳过，继续下一件",
                }
    return steps, pose_state


def build_pack_bin_slots_task(parts_3d, user_cmd):
    """按每件自己的 slot_id / dest_coordinate 依次入格（example_bin 全量）。"""
    if not parts_3d:
        return default_error_task("无可装箱零件")

    # 有 slot 的优先按 slot 排序；否则保持原顺序
    parts = list(parts_3d)
    parts.sort(
        key=lambda p: (
            0 if p.get("slot_id") is not None else 1,
            int(p["slot_id"]) if p.get("slot_id") is not None else 0,
            int(p.get("id") or 0),
        )
    )

    flat = []
    batch = []
    for i, part in enumerate(parts):
        slot_id = part.get("slot_id")
        if part.get("dest_coordinate"):
            dest = list(part["dest_coordinate"])
            dest_name = part.get("destination") or (
                f"料盘格子{slot_id}" if slot_id is not None else "料盘格子"
            )
        elif slot_id is not None:
            dest = slot_xyz(slot_id)
            dest_name = f"料盘格子{slot_id}"
        else:
            dest = list(STORAGE_BOX_XYZ)
            dest_name = STORAGE_BOX_NAME

        sub, pose_state = _one_part_steps(
            part, dest, dest_name, user_cmd, step0=len(flat), skip_on_fail=True
        )
        # 补 slot / object_id 到每一步，方便仿真
        for t in sub:
            if slot_id is not None:
                t["slot_id"] = slot_id
            if part.get("id") is not None:
                t["object_id"] = part.get("id")
            t["pose_state"] = pose_state
        flat.extend(sub)
        batch.append(
            {
                "index": i + 1,
                "object_id": part.get("id"),
                "object": part.get("class"),
                "slot_id": slot_id,
                "pose_state": pose_state,
                "coordinate": list(part.get("xyz") or []),
                "dest_coordinate": dest,
                "steps": [t["action"] for t in sub],
            }
        )

    first = parts[0]
    first_dest_name = (
        f"料盘格子{batch[0].get('slot_id')}"
        if batch[0].get("slot_id") is not None
        else STORAGE_BOX_NAME
    )
    return {
        "task_id": "t_bin_all",
        "action": "pack_bin_slots",
        "object": first.get("class", ""),
        "coordinate": list(first.get("xyz") or [0, 0, 0]),
        "dest_coordinate": list(batch[0]["dest_coordinate"]),
        "destination": first_dest_name,
        "angle": float((first.get("rpy") or [0, 0, 0])[2]),
        "rpy": list(first.get("rpy") or [0, 0, 0]),
        "retry": 0,
        "max_retry": 3,
        "status": "pending",
        "reason": f"共{len(parts)}件，按对应格子依次装箱（含姿态调姿）",
        "original_cmd": user_cmd,
        "job_count": len(parts),
        "batch": batch,
        "tasks": flat,
        "task_sequence_desc": [
            f"{b['index']}.id{b.get('object_id')}→slot{b.get('slot_id')}({b['pose_state']}):{'>'.join(b['steps'])}"
            for b in batch
        ],
    }


def build_pack_region_task(parts_3d, user_cmd, dest_xyz=None, capacity=None):
    capacity = int(capacity or BOX_CAPACITY)
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    region, key = _densest_region(parts_3d)
    region = region[:capacity]
    if not region:
        return default_error_task("最密区域无有效零件")
    flat = []
    batch = []
    for i, part in enumerate(region):
        sub, pose_state = _one_part_steps(part, dest, STORAGE_BOX_NAME, user_cmd, step0=len(flat))
        flat.extend(sub)
        batch.append(
            {
                "index": i + 1,
                "object": part.get("class"),
                "pose_state": pose_state,
                "coordinate": list(part.get("xyz") or []),
                "steps": [t["action"] for t in sub],
            }
        )
    first = region[0]
    return {
        "task_id": "t_pack",
        "action": "pack_region",
        "object": first.get("class", ""),
        "coordinate": list(first.get("xyz") or [0, 0, 0]),
        "dest_coordinate": dest,
        "destination": STORAGE_BOX_NAME,
        "angle": float((first.get("rpy") or [0, 0, 0])[2]),
        "rpy": list(first.get("rpy") or [0, 0, 0]),
        "retry": 0,
        "max_retry": 3,
        "status": "pending",
        "reason": f"零件最多区域{key}共{len(region)}件，依次装箱",
        "original_cmd": user_cmd,
        "region_key": list(key) if key is not None else None,
        "region_count": len(region),
        "batch": batch,
        "tasks": flat,
        "task_sequence_desc": [f"{b['index']}.{b['object']}:{'>'.join(b['steps'])}" for b in batch],
    }


def _part_has_xyz(part) -> bool:
    xyz = part.get("xyz") if isinstance(part, dict) else None
    return isinstance(xyz, (list, tuple)) and len(xyz) >= 2


def find_part_by_index(parts_3d, index):
    if not index:
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    for p in parts_3d or []:
        try:
            if int(p.get("id") or 0) == index:
                return p
        except (TypeError, ValueError):
            continue
    ordered = sorted(parts_3d or [], key=lambda o: int(o.get("id") or 0))
    if 1 <= index <= len(ordered):
        return ordered[index - 1]
    return None


def dest_for_human_slot(parts_3d, human_slot):
    if human_slot is None:
        return None, None
    try:
        human_slot = int(human_slot)
    except (TypeError, ValueError):
        return None, None
    for want in (human_slot, human_slot - 1):
        for p in parts_3d or []:
            try:
                sid = p.get("slot_id")
                if sid is not None and int(sid) == want and p.get("dest_coordinate"):
                    return list(p["dest_coordinate"]), want
            except (TypeError, ValueError):
                continue
    return None, human_slot - 1 if human_slot >= 1 else human_slot


def select_named_parts(parts_3d, names):
    used = set()
    selected = []
    skipped = []
    for name in names or []:
        hit = None
        for p in parts_3d or []:
            pid = p.get("id")
            if pid in used:
                continue
            cls = str(p.get("class") or "")
            if name == cls or (name and name in cls) or (cls and cls in str(name)):
                hit = p
                break
        if hit is None:
            skipped.append({"object": name, "reason": "场景中未找到，跳过继续"})
            continue
        if not _part_has_xyz(hit):
            skipped.append({"object": name, "reason": "无有效坐标，规划失败跳过"})
            continue
        used.add(hit.get("id"))
        selected.append(hit)
    return selected, skipped


def build_pack_parts_task(
    parts_3d,
    user_cmd,
    dest_xyz=None,
    dest_name=None,
    action_name="pack_batch",
    reason="",
):
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    dest_name = dest_name or STORAGE_BOX_NAME
    selected = []
    skipped = []
    for part in parts_3d or []:
        if not _part_has_xyz(part):
            skipped.append(
                {"object": part.get("class"), "reason": "无有效坐标，规划失败跳过"}
            )
            continue
        selected.append(part)
    if not selected:
        return default_error_task("没有可规划零件（已全部跳过）")
    flat = []
    batch = []
    for i, part in enumerate(selected):
        sub, pose_state = _one_part_steps(
            part, dest, dest_name, user_cmd, step0=len(flat), skip_on_fail=True
        )
        for t in sub:
            if part.get("id") is not None:
                t["object_id"] = part.get("id")
        flat.extend(sub)
        batch.append(
            {
                "index": i + 1,
                "object": part.get("class"),
                "object_id": part.get("id"),
                "pose_state": pose_state,
                "coordinate": list(part.get("xyz") or []),
                "steps": [t["action"] for t in sub],
            }
        )
    first = selected[0]
    return {
        "task_id": "t_pack_batch",
        "action": action_name,
        "object": first.get("class", ""),
        "coordinate": list(first.get("xyz") or [0, 0, 0]),
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": float((first.get("rpy") or [0, 0, 0])[2]),
        "rpy": list(first.get("rpy") or [0.0, 0.0, 0.0]),
        "retry": 0,
        "max_retry": 3,
        "status": "pending",
        "reason": reason or f"连续抓放{len(selected)}件，失败跳过继续",
        "original_cmd": user_cmd,
        "job_count": len(selected),
        "skipped": skipped,
        "skip_on_fail": True,
        "batch": batch,
        "tasks": flat,
        "task_sequence_desc": [
            f"{b['index']}.{b['object']}:{'>'.join(b['steps'])}" for b in batch
        ],
        "on_fail": {
            "action": "skip_continue",
            "reason": "该件规划/执行失败则跳过，继续下一件",
        },
    }


def build_transport_task(parts_3d, user_cmd, dest_xyz=None, filled_count=0):
    capacity = BOX_CAPACITY
    need = max(0, capacity - int(filled_count or 0))
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    flat = []
    batch = []
    if need > 0 and parts_3d:
        pack = build_pack_region_task(parts_3d, user_cmd, dest_xyz=dest, capacity=need)
        flat.extend(pack.get("tasks") or [])
        batch = pack.get("batch") or []
    box_xyz = list(STORAGE_BOX_XYZ)
    place_xyz = list(PLACE_AREA_XYZ)
    base = len(flat)
    for i, (action, reason) in enumerate(
        [
            ("perceive", "确认料箱已满"),
            ("pick", "抓取料箱"),
            ("transport", "搬运至零件放置区"),
            ("place", "叠放整齐"),
        ]
    ):
        step = base + i + 1
        flat.append(
            {
                "task_id": f"t{step:03d}",
                "step": step,
                "action": action,
                "object": "料箱",
                "coordinate": box_xyz if action != "transport" else place_xyz,
                "dest_coordinate": place_xyz,
                "destination": "零件放置区",
                "angle": 0.0,
                "rpy": [0.0, 0.0, 0.0],
                "retry": 0,
                "max_retry": 3,
                "status": "pending",
                "reason": reason,
                "original_cmd": user_cmd,
            }
        )
    return {
        "task_id": "t_transport",
        "action": "transport",
        "object": "料箱",
        "coordinate": box_xyz,
        "dest_coordinate": place_xyz,
        "destination": "零件放置区",
        "status": "pending",
        "reason": "未满先补装至满，再搬运叠放" if need > 0 else "箱满直接搬运叠放",
        "original_cmd": user_cmd,
        "box_full": True,
        "pre_pack_count": len(batch),
        "batch": batch,
        "tasks": flat,
        "task_sequence_desc": [t["action"] for t in flat],
    }


def build_sequence(parsed_json):
    obj = parsed_json.get("object", "")
    pick_xyz = list(parsed_json.get("coordinate", [0, 0, 0]))
    dest_name = parsed_json.get("destination")
    dest = parsed_json.get("dest_coordinate")
    rpy = list(parsed_json.get("rpy", [0, 0, 0]))
    angle = float(parsed_json.get("angle", rpy[2] if rpy else 0))
    reason = parsed_json.get("reason", "")
    status = "pending"
    max_retry = int(parsed_json.get("max_retry", 3))
    cmd = parsed_json.get("original_cmd", "")
    pose_state = _normalize_pose(parsed_json.get("pose_state") or "normal")
    parsed_json["pose_state"] = pose_state
    do_place = wants_place(cmd) and dest_name is not None

    perceive = {
        "task_id": "t000",
        "step": 1,
        "action": "perceive",
        "object": obj,
        "coordinate": pick_xyz,
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": angle,
        "rpy": rpy,
        "retry": 0,
        "max_retry": max_retry,
        "status": status,
        "reason": "识别场景目标并定位；判断正常/倒放/倾倒",
        "original_cmd": cmd,
        "pose_state": pose_state,
    }
    pick = {
        "task_id": "t001",
        "step": 2,
        "action": "pick",
        "object": obj,
        "coordinate": pick_xyz,
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": angle,
        "rpy": rpy,
        "retry": 0,
        "max_retry": max_retry,
        "status": status,
        "reason": reason,
        "original_cmd": cmd,
    }
    if not do_place:
        return [perceive, pick]

    dest = list(dest or STORAGE_BOX_XYZ)
    tasks = [perceive, pick]
    step = 2
    place_rpy = rpy
    if _needs_reorient(pose_state):
        step += 1
        tasks.append(
            {
                "task_id": f"t{step:03d}",
                "step": step,
                "action": "adjust_pose",
                "object": obj,
                "coordinate": pick_xyz,
                "dest_coordinate": dest,
                "destination": dest_name,
                "angle": angle,
                "rpy": [0.0, 0.0, angle],
                "retry": 0,
                "max_retry": max_retry,
                "status": "pending",
                "reason": "将倒放/倾倒零件调整为正常姿态后再放置",
                "original_cmd": cmd,
                "target_pose_state": "normal",
            }
        )
        place_rpy = [0.0, 0.0, angle]
    step += 1
    tasks.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "move",
            "object": obj,
            "coordinate": dest,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": place_rpy,
            "retry": 0,
            "max_retry": max_retry,
            "status": "pending",
            "reason": "携带零件移动至放置目标上方",
            "original_cmd": cmd,
        }
    )
    step += 1
    tasks.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "place",
            "object": obj,
            "coordinate": dest,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": place_rpy,
            "retry": 0,
            "max_retry": max_retry,
            "status": "pending",
            "reason": "",
            "original_cmd": cmd,
            "on_fail": {
                "action": "retry_place",
                "max_retry": max_retry,
                "reason": "若摆放失败则重新感知并重试放置",
            },
        }
    )
    return tasks


def intent_do_place(cmd: str, destination=None) -> bool:
    """指令优先；指令为空时再看 destination。"""
    text = (cmd or "").strip()
    if wants_transport(text) or wants_pack_region(text):
        return True
    if text:
        return wants_place(text)
    dest = str(destination or "").strip()
    if dest in ("", "null", "None", "无"):
        return False
    if dest == STORAGE_BOX_NAME or "收纳" in dest or "料箱" in dest or "格子" in dest:
        return True
    return False


def main(llm_raw, parts_3d=None, target_xyz=None, user_cmd=None):
    # Dify 会把 user_cmd 作为独立入参传入
    user_cmd_hint = str(user_cmd or "")
    if isinstance(llm_raw, dict):
        if "text" in llm_raw:
            if not user_cmd_hint:
                user_cmd_hint = str(llm_raw.get("user_cmd", "") or "")
            llm_raw = llm_raw["text"]
    if llm_raw is None or not isinstance(llm_raw, str):
        llm_raw = ""

    parts_3d = parts_3d or []
    # 闭环回传哨兵（预处理 mode=reflect；字段为 JSON 字符串以防 Depth limit）
    if parts_3d and isinstance(parts_3d[0], dict) and parts_3d[0].get("__reflect__"):
        payload = parts_3d[0]

        def _loads(raw, default=None):
            if default is None:
                default = {}
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str) and raw.strip():
                try:
                    return json.loads(raw)
                except Exception:
                    return default
            return default

        last_task = _loads(payload.get("last_task_json") or payload.get("last_task"), {})
        memory = _loads(
            payload.get("memory_json") or payload.get("memory"),
            {
                "finished_parts": [],
                "fail_record": [],
                "current_retry": 0,
                "max_retry": 3,
                "box_full": False,
                "box_capacity": BOX_CAPACITY,
            },
        )
        post_vision = _loads(
            payload.get("post_vision_json") or payload.get("post_vision_objects"), {}
        )

        place_task = last_task
        if last_task.get("action") != "place":
            for t in reversed(last_task.get("tasks") or []):
                if isinstance(t, dict) and t.get("action") == "place":
                    place_task = t
                    break
            else:
                place_task = {
                    "action": "place",
                    "object": last_task.get("object", ""),
                    "coordinate": last_task.get("dest_coordinate") or STORAGE_BOX_XYZ,
                    "dest_coordinate": last_task.get("dest_coordinate") or STORAGE_BOX_XYZ,
                    "rpy": last_task.get("rpy") or [0, 0, 0],
                }

        objs = post_vision.get("objects") if isinstance(post_vision, dict) else []
        target = place_task.get("object", "") or last_task.get("object", "")
        real = next((o for o in (objs or []) if o.get("class") == target), None)
        if real is None:
            check_status, fault_type, check_msg = "need_retry", "part_fall", "零件掉落/未检测到"
        else:
            st = _normalize_pose(real.get("pose_state") or real.get("state") or "normal")
            if _needs_reorient(st):
                check_status, fault_type, check_msg = "need_retry", "wrong_pose", f"姿态异常({st})"
            else:
                pose = real.get("pose_6d") or {}
                dest = list(
                    place_task.get("dest_coordinate")
                    or place_task.get("coordinate")
                    or STORAGE_BOX_XYZ
                )
                if abs(float(pose.get("x", 0)) - float(dest[0])) > 0.015 or abs(
                    float(pose.get("y", 0)) - float(dest[1])
                ) > 0.015:
                    check_status, fault_type, check_msg = "need_retry", "pos_offset", "放置坐标偏移"
                else:
                    check_status, fault_type, check_msg = "success", "", "执行成功"

        curr_r = int(memory.get("current_retry", 0))
        max_r = int(memory.get("max_retry", 3))
        finished = list(memory.get("finished_parts") or [])
        x_off = yaw_off = 0.0
        if check_status == "success":
            if target:
                finished.append(target)
            curr_r = 0
            flow = (
                "transport"
                if len(finished) >= int(memory.get("box_capacity", BOX_CAPACITY))
                else "next_task"
            )
            box_full = flow == "transport"
        elif curr_r >= max_r:
            flow, box_full = "final_fail", False
        else:
            flow, box_full = "retry", False
            curr_r += 1
            if fault_type == "pos_offset":
                x_off = 0.006
            elif fault_type == "wrong_pose":
                yaw_off = 0.12
            elif fault_type == "part_fall":
                x_off = 0.01
            else:
                yaw_off = 0.08

        flat = []
        next_action = flow
        if flow == "retry":
            next_action = "retry_place"
            pick_xyz = list(
                last_task.get("coordinate") or place_task.get("coordinate") or [0.0, 0.0, 0.0]
            )
            pick_xyz = [float(pick_xyz[0]) + x_off, float(pick_xyz[1]), 0.0]
            dest = list(place_task.get("dest_coordinate") or STORAGE_BOX_XYZ)
            yaw = float((last_task.get("rpy") or place_task.get("rpy") or [0, 0, 0])[2]) + yaw_off
            for i, (act, reason) in enumerate(
                [
                    ("perceive", "失败后重新感知"),
                    ("pick", "重新抓取"),
                    ("adjust_pose", "调整姿态"),
                    ("place", "再次放置"),
                ]
            ):
                flat.append(
                    {
                        "task_id": f"t{i + 1:03d}",
                        "step": i + 1,
                        "action": act,
                        "object": target,
                        "coordinate": pick_xyz if act != "place" else dest,
                        "dest_coordinate": dest,
                        "destination": STORAGE_BOX_NAME,
                        "angle": yaw,
                        "rpy": [0.0, 0.0, yaw],
                        "retry": curr_r,
                        "max_retry": max_r,
                        "status": "pending",
                        "reason": reason,
                        "original_cmd": user_cmd_hint,
                    }
                )
        elif flow == "transport":
            next_action = "transport"
            box_xyz = list(STORAGE_BOX_XYZ)
            place_xyz = list(PLACE_AREA_XYZ)
            for i, (act, reason, coord) in enumerate(
                [
                    ("perceive", "确认料箱已满", box_xyz),
                    ("pick", "抓取料箱", box_xyz),
                    ("transport", "搬运至放置区", place_xyz),
                    ("place", "叠放整齐", place_xyz),
                ]
            ):
                flat.append(
                    {
                        "task_id": f"t{i + 1:03d}",
                        "step": i + 1,
                        "action": act,
                        "object": "料箱",
                        "coordinate": coord,
                        "dest_coordinate": place_xyz,
                        "destination": "零件放置区",
                        "angle": 0.0,
                        "rpy": [0.0, 0.0, 0.0],
                        "retry": 0,
                        "max_retry": 3,
                        "status": "pending",
                        "reason": reason,
                        "original_cmd": user_cmd_hint,
                    }
                )

        # 全部扁平字段，禁止再嵌套 check/memory/task 对象
        parsed = {
            "task_id": "t_reflect",
            "action": "reflect",
            "object": target,
            "status": "pending" if flow != "final_fail" else "failed",
            "reason": check_msg,
            "original_cmd": user_cmd_hint,
            "mode": "reflect",
            "flow_tag": flow,
            "next_action": next_action,
            "check_status": check_status,
            "fault_type": fault_type,
            "check_msg": check_msg,
            "x_off": x_off,
            "yaw_off": yaw_off,
            "memory_finished_count": len(finished),
            "memory_current_retry": curr_r,
            "memory_max_retry": max_r,
            "memory_box_full": box_full,
            "coordinate": list(STORAGE_BOX_XYZ),
            "dest_coordinate": list(STORAGE_BOX_XYZ),
            "destination": STORAGE_BOX_NAME,
            "tasks": flat,
            "task_sequence_desc": [f"reflect→{flow}"]
            + [t["action"] for t in flat],
        }
        # final_fail 仍 valid=True，避免下游把业务失败当解析错误改写任务
        return {
            "result": {
                "valid": True,
                "task_json": _shallow_task_for_dify(parsed),
                "error": "",
            }
        }

    for p in parts_3d:
        if "pose_state" not in p:
            p["pose_state"] = _infer_pose(p)
    target_xyz = normalize_xyz(target_xyz or STORAGE_BOX_XYZ)
    allowed_actions = {
        "pick",
        "move",
        "place",
        "idle",
        "perceive",
        "adjust_pose",
        "transport",
        "pack_region",
        "pack_bin_slots",
        "multi_agent_pack",
        "reflect",
    }
    valid = False
    parsed_json = {}
    error_msg = ""

    # 高层意图：不依赖 LLM，直接规划（答疑第③⑤点）
    high_intent = parse_high_intent(
        user_cmd_hint, [p.get("class") for p in parts_3d], parts_3d
    )
    oidx = parse_object_index(user_cmd_hint)
    if oidx is not None and high_intent in ("pick_and_place", "pick_only"):
        target = find_part_by_index(parts_3d, oidx)
        if target is not None:
            human_slot = parse_slot_id(user_cmd_hint)
            dest = None
            dest_name = STORAGE_BOX_NAME
            if human_slot is not None:
                dest, sid = dest_for_human_slot(parts_3d, human_slot)
                dest_name = f"料盘格子{sid if sid is not None else human_slot}"
            elif wants_into_slots(user_cmd_hint) or high_intent == "pick_and_place":
                if target.get("dest_coordinate"):
                    dest = list(target["dest_coordinate"])
                    dest_name = target.get("destination") or f"料盘格子{target.get('slot_id')}"
            if dest is None and high_intent == "pick_and_place":
                dest = list(target_xyz or STORAGE_BOX_XYZ)
                dest_name = STORAGE_BOX_NAME if not wants_into_slots(user_cmd_hint) else dest_name
            if high_intent == "pick_only":
                dest = list(target.get("xyz") or [0, 0, 0])
                dest_name = None
            one = dict(target)
            parsed_json = build_pack_parts_task(
                [one],
                user_cmd_hint,
                dest_xyz=dest,
                dest_name=dest_name or STORAGE_BOX_NAME,
                action_name="pick_and_place" if high_intent == "pick_and_place" else "pick",
                reason=f"按口令生成: {user_cmd_hint}",
            )
            if high_intent == "pick_only":
                parsed_json["action"] = "pick"
                for t in parsed_json.get("tasks") or []:
                    if t.get("action") in ("move", "place"):
                        t["status"] = "skipped"
            return {
                "result": {
                    "valid": True,
                    "task_json": _shallow_task_for_dify(parsed_json),
                    "error": "",
                }
            }
    if high_intent == "pack_n_parts":
        n = parse_part_count(user_cmd_hint) or 1
        ranked = sorted(
            [p for p in parts_3d if _part_has_xyz(p)],
            key=lambda o: (-int(o.get("visible_pixels") or 0), o.get("id") or 0),
        )
        chosen = ranked[:n]
        parsed_json = build_pack_parts_task(
            chosen,
            user_cmd_hint,
            dest_xyz=target_xyz,
            action_name="pack_n_parts",
            reason=f"按口令生成: {user_cmd_hint}；随便拿{n}个，失败跳过继续",
        )
        return {
            "result": {
                "valid": True,
                "task_json": _shallow_task_for_dify(parsed_json),
                "error": "",
            }
        }
    if high_intent == "pack_all_box":
        parsed_json = build_pack_parts_task(
            parts_3d,
            user_cmd_hint,
            dest_xyz=target_xyz,
            action_name="pack_all_box",
            reason=f"按口令生成: {user_cmd_hint}",
        )
        return {
            "result": {
                "valid": True,
                "task_json": _shallow_task_for_dify(parsed_json),
                "error": "",
            }
        }
    if high_intent == "multi_named_pick_place":
        names = extract_named_mentions(
            user_cmd_hint, [p.get("class") for p in parts_3d]
        )
        chosen, skipped = select_named_parts(parts_3d, names)
        parsed_json = build_pack_parts_task(
            chosen,
            user_cmd_hint,
            dest_xyz=target_xyz,
            action_name="multi_named_pick_place",
            reason="多个具体零件连续 pick-place，失败跳过继续",
        )
        parsed_json["skipped"] = (parsed_json.get("skipped") or []) + skipped
        return {
            "result": {
                "valid": True,
                "task_json": _shallow_task_for_dify(parsed_json),
                "error": "",
            }
        }
    if high_intent == "pack_bin_slots":
        ordered = sorted(parts_3d, key=lambda o: int(o.get("id") or 0))
        n = parse_part_count(user_cmd_hint)
        chosen = ordered[:n] if n else ordered
        parsed_json = build_pack_bin_slots_task(chosen, user_cmd_hint)
        parsed_json["reason"] = f"按口令生成: {user_cmd_hint}；共{len(chosen)}件入格，失败跳过继续"
        parsed_json["skip_on_fail"] = True
        return {
            "result": {
                "valid": True,
                "task_json": _shallow_task_for_dify(parsed_json),
                "error": "",
            }
        }
    if high_intent == "multi_agent_pack":
        # 注意：Dify 代码节点返回对象深度上限约 5，禁止 agents 里再嵌整包 task
        sorted_parts = sorted(parts_3d, key=lambda o: o["xyz"][0])
        mid = max(1, len(sorted_parts) // 2) if sorted_parts else 0
        parts_a = sorted_parts[:mid]
        parts_b = sorted_parts[mid:] or sorted_parts[:1]
        pack_a = build_pack_region_task(parts_a, user_cmd_hint, dest_xyz=target_xyz, capacity=BOX_CAPACITY)
        pack_b = build_pack_region_task(
            parts_b, user_cmd_hint, dest_xyz=[target_xyz[0] + 0.35, target_xyz[1], target_xyz[2]], capacity=BOX_CAPACITY
        )
        tr_a = build_transport_task([], user_cmd_hint, filled_count=BOX_CAPACITY)
        tr_b = build_transport_task([], user_cmd_hint, filled_count=BOX_CAPACITY)
        flat = []
        blocks = [
            ("agent_A", pack_a),
            ("agent_C", tr_a),
            ("agent_B", pack_b),
            ("agent_C", tr_b),
        ]
        for agent_id, block in blocks:
            for t in block.get("tasks") or []:
                item = {
                    "task_id": f"t{len(flat) + 1:03d}",
                    "step": len(flat) + 1,
                    "action": t.get("action"),
                    "object": t.get("object"),
                    "coordinate": t.get("coordinate"),
                    "dest_coordinate": t.get("dest_coordinate"),
                    "destination": t.get("destination"),
                    "angle": t.get("angle", 0.0),
                    "rpy": t.get("rpy") or [0.0, 0.0, 0.0],
                    "retry": 0,
                    "max_retry": 3,
                    "status": "pending",
                    "reason": t.get("reason", ""),
                    "original_cmd": user_cmd_hint,
                    "agent_id": agent_id,
                }
                if t.get("pose_state"):
                    item["pose_state"] = t.get("pose_state")
                if t.get("action") == "place":
                    item["on_fail_action"] = "retry_place"
                    item["on_fail_max_retry"] = 3
                flat.append(item)
        parsed_json = {
            "task_id": "t_multi_agent",
            "action": "multi_agent_pack",
            "object": "所有零件",
            "coordinate": list(STORAGE_BOX_XYZ),
            "dest_coordinate": list(PLACE_AREA_XYZ),
            "destination": "零件放置区",
            "status": "pending",
            "reason": "多智能体：A/B分区装箱，C搬运满箱",
            "original_cmd": user_cmd_hint,
            "agent_A_role": "packer",
            "agent_A_parts": int(pack_a.get("region_count") or len(parts_a)),
            "agent_B_role": "packer",
            "agent_B_parts": int(pack_b.get("region_count") or len(parts_b)),
            "agent_C_role": "transporter",
            "collaboration": "A/B并行装箱；演示顺序 A装箱→C搬A→B装箱→C搬B",
            "tasks": flat,
            "task_sequence_desc": ["1.A装箱", "2.C搬A", "3.B装箱", "4.C搬B"],
        }
        return {"result": {"valid": True, "task_json": _shallow_task_for_dify(parsed_json), "error": ""}}
    if high_intent == "pack_densest_region":
        parsed_json = build_pack_region_task(parts_3d, user_cmd_hint, dest_xyz=target_xyz)
        return {
            "result": {
                "valid": parsed_json.get("status") != "error",
                "task_json": _shallow_task_for_dify(parsed_json),
                "error": "" if parsed_json.get("status") != "error" else parsed_json.get("reason", ""),
            }
        }
    if high_intent == "transport_box":
        parsed_json = build_transport_task(parts_3d, user_cmd_hint, dest_xyz=target_xyz)
        return {
            "result": {
                "valid": True,
                "task_json": _shallow_task_for_dify(parsed_json),
                "error": "",
            }
        }

    try:
        json_str = extract_json_text(llm_raw.strip())
        if not json_str:
            raise ValueError("json format error:无json内容")
        parsed_json = json.loads(json_str)
        if not isinstance(parsed_json, dict):
            raise ValueError("json format error:非对象")

        fill_map = {
            "retry": 0,
            "max_retry": 3,
            "task_id": "t001",
            "status": "invalid",
            "action": "pick",
            "object": "",
            "destination": None,
            "reason": "",
            "original_cmd": user_cmd_hint,
        }
        for key, default_val in fill_map.items():
            if key not in parsed_json:
                parsed_json[key] = default_val

        # 始终用工作流入参覆盖 original_cmd（LLM 常漏写）
        parsed_json["original_cmd"] = user_cmd_hint or str(
            parsed_json.get("original_cmd") or ""
        )

        action = str(parsed_json.get("action", "pick")).strip().lower()
        if action in ("pick_and_place", "pickandplace", "pick-place"):
            action = "pick"
            parsed_json["action"] = "pick"

        status_raw = str(parsed_json.get("status", "")).strip()
        object_name = str(parsed_json.get("object", "")).strip()
        if action not in allowed_actions:
            parsed_json["action"] = "pick"
            action = "pick"

        cmd_for_intent = parsed_json["original_cmd"] + " " + object_name
        do_place = intent_do_place(parsed_json["original_cmd"], parsed_json.get("destination"))

        if do_place:
            dest_name, dest_xyz, slot_id = resolve_dest(
                parsed_json.get("destination"),
                target_xyz,
                user_hint=cmd_for_intent,
            )
            if slot_id is None:
                slot_id = parse_slot_id(str(parsed_json.get("destination", "")))
                if slot_id is not None:
                    dest_name, dest_xyz = f"料箱格子{slot_id}", slot_xyz(slot_id)
            if not dest_name:
                dest_name, dest_xyz = STORAGE_BOX_NAME, list(STORAGE_BOX_XYZ)
            parsed_json["destination"] = dest_name
            parsed_json["dest_coordinate"] = dest_xyz
            parsed_json["slot_id"] = slot_id
        else:
            parsed_json["destination"] = None
            parsed_json["dest_coordinate"] = None
            parsed_json["slot_id"] = None
            dest_xyz = None

        if action in ("pick", "move", "place", "perceive"):
            spatial = parse_spatial(object_name) or parse_spatial(
                str(parsed_json.get("reason", ""))
            )
            # 也从 object 字段试空间词——真正空间词通常在指令里，下游 format 会再补

            target_obj = find_object(parts_3d, object_name)
            if target_obj is None and len(parts_3d) > 0:
                # 自动选最大可见
                target_obj = find_max_pixel_obj(parts_3d)
                parsed_json["object"] = target_obj["class"]
                parsed_json["status"] = "invalid"
                parsed_json["reason"] = (
                    f"指令未明确指定零件名称，已匹配最大可见零件{target_obj['class']}"
                )

            if target_obj is None:
                raise ValueError("未找到目标物体")

            parsed_json["object"] = target_obj["class"]
            parsed_json["pose_state"] = _normalize_pose(
                target_obj.get("pose_state") or target_obj.get("state") or _infer_pose(target_obj)
            )
            if action in ("pick", "perceive"):
                parsed_json["coordinate"] = list(target_obj["xyz"])
            else:
                parsed_json["coordinate"] = list(
                    parsed_json.get("dest_coordinate") or target_obj["xyz"]
                )
            parsed_json["angle"] = float(target_obj["rpy"][2])
            parsed_json["rpy"] = list(target_obj["rpy"])

            if action in ("pick", "perceive") and not is_valid_desk_coord(
                parsed_json["coordinate"]
            ):
                raise ValueError(f"物体坐标异常:{parsed_json['coordinate']}")

            if status_raw == "valid":
                valid = True
            elif "已匹配最大可见零件" in parsed_json.get("reason", ""):
                valid = True
            elif object_name and (
                object_name == target_obj["class"] or object_name in target_obj["class"]
            ):
                parsed_json["status"] = "valid"
                valid = True
            else:
                valid = status_raw in ("valid", "invalid")
        else:
            parsed_json["coordinate"] = [0.0, 0.0, 0.0]
            parsed_json["dest_coordinate"] = None
            parsed_json["destination"] = None
            parsed_json["angle"] = 0.0
            parsed_json["rpy"] = [0.0, 0.0, 0.0]
            valid = status_raw == "valid"

        parsed_json["tasks"] = build_sequence(parsed_json)
        pose_state = _normalize_pose(parsed_json.get("pose_state") or "normal")
        parsed_json["pose_state"] = pose_state
        if intent_do_place(parsed_json.get("original_cmd"), parsed_json.get("destination")):
            if _needs_reorient(pose_state):
                parsed_json["task_sequence_desc"] = [
                    "1.perceive 识别并判断姿态",
                    "2.pick 抓取",
                    "3.adjust_pose 调姿",
                    "4.move 移动",
                    "5.place 放入",
                ]
            else:
                parsed_json["task_sequence_desc"] = [
                    "1.perceive 识别定位目标",
                    "2.pick 抓取目标零件",
                    "3.move 移动至放置点",
                    "4.place 放入收纳盒/料箱格子（失败可 retry_place）",
                ]
        else:
            parsed_json["task_sequence_desc"] = [
                "1.perceive 识别定位目标",
                "2.pick 抓取目标零件（仅拿起，不放置）",
            ]
    except json.JSONDecodeError:
        error_msg = "json format error"
        parsed_json = default_error_task(error_msg)
        valid = False
    except Exception as e:
        error_msg = str(e)
        parsed_json = default_error_task(error_msg)
        valid = False

    return {
        "result": {
            "valid": valid,
            "task_json": _shallow_task_for_dify(parsed_json),
            "error": error_msg,
        }
    }
