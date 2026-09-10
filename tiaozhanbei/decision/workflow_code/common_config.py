"""
项目书对齐配置：收纳盒 / 多格料箱 / 任务动作约定。
依据：XH-202607 工业环境下物体感知识别与指令交互型智能体研发
"""
import re

# 仿真同学提供的收纳盒基准点（米）
STORAGE_BOX_NAME = "收纳盒"
STORAGE_BOX_XYZ = [0.30, -0.20, 0.0]  # 对齐组长 dataset_learn scene_layout.storage_box.pos_m

# 多格料箱：相对收纳盒中心的格子偏移（米），模拟「第 N 个格子」
# 布局：从左到右 3 格，再第二排 3 格
BIN_SLOT_OFFSETS = {
    1: [-0.08, 0.00, 0.0],
    2: [0.00, 0.00, 0.0],
    3: [0.08, 0.00, 0.0],
    4: [-0.08, -0.08, 0.0],
    5: [0.00, -0.08, 0.0],
    6: [0.08, -0.08, 0.0],
}

ALLOWED_ACTIONS = {
    "pick",
    "place",
    "move",
    "idle",
    "perceive",
    "retry_place",
    "adjust_pose",
    "transport",
    "stop",
}

# 料箱容量（演示）：装满后触发搬运
BOX_CAPACITY = 6
# 零件放置区（箱满搬运目标）
PLACE_AREA_XYZ = [0.55, -0.35, 0.0]

# 自然语言中的动作关键词
PICK_KEYWORDS = ("抓取", "拿取", "取出", "帮我拿", "给我", "拿一下", "取")
PLACE_KEYWORDS = ("放到", "放入", "放置", "放进", "移到", "移动到", "装箱")
TRANSPORT_KEYWORDS = ("搬运", "搬走", "移送料箱", "搬料箱")
SPATIAL_KEYWORDS = ("左侧", "右边", "右侧", "左边", "中间", "最大", "最近", "最靠前", "最靠后")

# 高层意图
INTENT_PICK_ONLY = "pick_only"
INTENT_PICK_PLACE = "pick_and_place"
INTENT_PACK_REGION = "pack_densest_region"
INTENT_TRANSPORT = "transport_box"
INTENT_MULTI_AGENT = "multi_agent_pack"
INTENT_PACK_N = "pack_n_parts"
INTENT_PACK_ALL_BOX = "pack_all_box"
INTENT_MULTI_NAMED = "multi_named_pick_place"
INTENT_PACK_SLOTS = "pack_bin_slots"
INTENT_REFLECT = "reflect"

CN_COUNT = {
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

# 口语别名（最长优先，避免「螺丝」吃掉「螺丝刀」）
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


def slot_xyz(slot_id: int):
    """返回料箱第 slot_id 格的世界坐标。"""
    base = list(STORAGE_BOX_XYZ)
    offset = BIN_SLOT_OFFSETS.get(int(slot_id), BIN_SLOT_OFFSETS[2])
    return [base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]]


def default_destination():
    return STORAGE_BOX_NAME, list(STORAGE_BOX_XYZ)


def _to_count(raw: str):
    raw = (raw or "").strip()
    if raw.isdigit():
        return int(raw)
    if raw in CN_COUNT:
        return CN_COUNT[raw]
    if raw.startswith("十") and len(raw) == 2 and raw[1] in CN_COUNT:
        return 10 + CN_COUNT[raw[1]]
    return None


def parse_part_count(cmd: str):
    """解析「随便拿 3 个零件」「拿五个螺丝」。格子序号不算件数。"""
    text = cmd or ""
    token = r"(?:\d+|十二|十一|十|[一二两三四五六七八九])"
    for pat in (
        rf"随便拿\s*({token})\s*个",
        rf"任意(?:拿|取|抓)?\s*({token})\s*个",
        rf"前\s*({token})\s*个",
        rf"(?:拿|取|抓)(?:取)?\s*({token})\s*个(?:零件|工件|螺丝|螺钉)?",
        rf"({token})\s*个(?:零件|工件|螺丝|螺钉)",
    ):
        m = re.search(pat, text)
        if not m:
            continue
        n = _to_count(m.group(1))
        if n is not None and 1 <= n <= 50:
            return n
    return None


def wants_pack_all_box(cmd: str) -> bool:
    """所有零件放进收纳盒/收纳箱（单臂连续装箱，失败跳过）。"""
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
    into_box = (
        "收纳盒" in text
        or "收纳箱" in text
        or "放进" in text
        or "放入" in text
        or "放到" in text
    )
    return bool(into_box)


def wants_pack_n(cmd: str) -> bool:
    n = parse_part_count(cmd)
    if n is None:
        return False
    if wants_pack_all_box(cmd) or wants_multi_agent(cmd):
        return False
    return True


def wants_pack_bin_slots(cmd: str) -> bool:
    """12 螺丝入格 / 依次对应格子。已写明「格子N」的单件指令不算。"""
    text = cmd or ""
    if re.search(r"第\s*[零一二三四五六七八九十0-9]+\s*个?\s*格子?", text):
        return False
    if re.search(r"格子\s*[0-9]+", text) and not (
        "依次" in text or "对应" in text or "螺丝" in text or "螺钉" in text
    ):
        return False
    if "对应格子" in text or "入格" in text:
        return True
    if ("依次" in text or "逐个" in text or "一个个" in text) and (
        "格子" in text or "料盘" in text or "料箱" in text
    ):
        return True
    if "格子" in text and ("螺丝" in text or "螺钉" in text or "喇叭头" in text):
        return True
    if re.search(r"1[0-2]\s*个", text) and "格子" in text:
        return True
    return False


def wants_multi_named(cmd: str, class_names=None) -> bool:
    """多个具体零件连续 pick-place（轴承和螺丝放到收纳盒）。"""
    text = cmd or ""
    if wants_pack_all_box(text) or wants_pack_n(text) or wants_multi_agent(text):
        return False
    names = extract_named_mentions(text, class_names)
    if len(names) >= 2:
        return True
    return bool(re.search(r"(和|、|以及|然后|再拿|再取)", text) and len(names) >= 2)


def extract_named_mentions(cmd: str, class_names=None):
    """按指令出现顺序提取零件名；跳过重叠子串。"""
    text = cmd or ""
    if not text:
        return []
    occupied = [False] * len(text)
    found = []

    def _mark(start: int, end: int, name: str):
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


def wants_multi_agent(cmd: str) -> bool:
    """「帮我把所有零件装箱」等多智能体协同（不含「放进收纳盒」）。"""
    text = cmd or ""
    if "多智能体" in text or "多机械臂" in text:
        return True
    if wants_pack_all_box(text):
        return False
    if ("所有" in text or "全部" in text) and "装箱" in text:
        return True
    return False


def wants_transport(cmd: str) -> bool:
    """「帮我搬运一下料箱盒」等。"""
    text = cmd or ""
    if any(k in text for k in TRANSPORT_KEYWORDS):
        return True
    if "搬运" in text and ("料箱" in text or "盒子" in text or "盒" in text):
        return True
    return False


def wants_pack_region(cmd: str) -> bool:
    """「帮我把零件最多的区域装箱」等。"""
    text = cmd or ""
    if wants_multi_agent(text):
        return False
    if "最多" in text and ("区域" in text or "区" in text):
        return True
    if "装箱" in text and ("区域" in text or "最多" in text):
        return True
    if "零件最多" in text:
        return True
    return False


def parse_intent(cmd: str, class_names=None) -> str:
    """高层意图：多智能体 / 全量入盒 / 拿N个 / 多名连续 / 入格 / 搬运 / 区域装箱 / 单件。"""
    if wants_multi_agent(cmd):
        return INTENT_MULTI_AGENT
    if wants_pack_all_box(cmd):
        return INTENT_PACK_ALL_BOX
    if wants_transport(cmd):
        return INTENT_TRANSPORT
    if wants_pack_bin_slots(cmd):
        return INTENT_PACK_SLOTS
    if wants_pack_n(cmd):
        return INTENT_PACK_N
    if wants_multi_named(cmd, class_names):
        return INTENT_MULTI_NAMED
    if wants_pack_region(cmd):
        return INTENT_PACK_REGION
    if wants_place(cmd):
        return INTENT_PICK_PLACE
    return INTENT_PICK_ONLY


def wants_place(cmd: str) -> bool:
    """
    是否要求「拿起并放到目标」。
    搬运 / 区域装箱由 parse_intent 优先处理，此处仍可能为 True。
    """
    text = cmd or ""
    if wants_transport(text):
        return False
    if wants_multi_agent(text):
        return True
    if wants_pack_region(text):
        return True
    if any(k in text for k in PLACE_KEYWORDS):
        return True
    if "收纳盒" in text or "收纳箱" in text or "料箱" in text:
        return True
    if "格子" in text:
        return True
    return False
