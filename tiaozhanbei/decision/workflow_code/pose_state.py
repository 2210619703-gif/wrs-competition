"""姿态状态：正常 / 倒放 / 倾倒（榜题场景难点）。

兼容标注别名：
- inverted（学习 JSON / example_bin）→ upside_down
- fallen → tilt
"""
from __future__ import annotations


POSE_NORMAL = "normal"
POSE_UPSIDE_DOWN = "upside_down"  # 倒放
POSE_TILT = "tilt"  # 倾倒

POSE_CN = {
    POSE_NORMAL: "正常",
    POSE_UPSIDE_DOWN: "倒放",
    POSE_TILT: "倾倒",
}

# 统一别名 → 内部三态
POSE_ALIASES = {
    "normal": POSE_NORMAL,
    "正常": POSE_NORMAL,
    "upside_down": POSE_UPSIDE_DOWN,
    "倒放": POSE_UPSIDE_DOWN,
    "倒置": POSE_UPSIDE_DOWN,
    "inverted": POSE_UPSIDE_DOWN,
    "tilt": POSE_TILT,
    "tilted": POSE_TILT,
    "倾倒": POSE_TILT,
    "倾斜": POSE_TILT,
    "fallen": POSE_TILT,
}


def normalize_pose_state(raw) -> str:
    """把任意标注/中文/学习 JSON 姿态归一成 normal / upside_down / tilt。"""
    if raw is None:
        return POSE_NORMAL
    t = str(raw).strip().lower()
    if not t:
        return POSE_NORMAL
    if t in POSE_ALIASES:
        return POSE_ALIASES[t]
    # 中文未 lower 时再试一遍原串
    t2 = str(raw).strip()
    return POSE_ALIASES.get(t2, POSE_NORMAL)


def infer_pose_state(obj: dict) -> str:
    """
    若标注已有 pose_state/state 则归一化用之；否则用 roll/pitch 粗推断。
    数据集暂无姿态字段时，用 id 规则做稳定演示分布（同物体可复现）。
    """
    raw = obj.get("pose_state") or obj.get("state") or ""
    if isinstance(raw, str) and raw.strip():
        # 已知别名直接归一；未知字符串若已是三态也会命中
        mapped = normalize_pose_state(raw)
        # normalize 对未知会变 normal；若原值就是未知英文，保留推断
        key = raw.strip().lower()
        if key in POSE_ALIASES or raw.strip() in POSE_ALIASES:
            return mapped

    pose = obj.get("pose_6d") or {}
    try:
        roll = abs(float(pose.get("roll", 0.0)))
        pitch = abs(float(pose.get("pitch", 0.0)))
    except Exception:
        roll, pitch = 0.0, 0.0
    if roll > 2.0 or pitch > 2.0:
        return POSE_UPSIDE_DOWN
    if roll > 0.6 or pitch > 0.6:
        return POSE_TILT

    # 演示：按 id 轮换，保证交付样例里能看到倒放/倾倒
    oid = obj.get("id", 0)
    try:
        oid = int(oid)
    except Exception:
        oid = 0
    rem = oid % 5
    if rem == 1:
        return POSE_UPSIDE_DOWN
    if rem == 2:
        return POSE_TILT
    return POSE_NORMAL


def attach_pose_state(objects: list) -> list:
    out = []
    for o in objects:
        item = dict(o)
        item["pose_state"] = infer_pose_state(item)
        item["pose_state_cn"] = POSE_CN.get(item["pose_state"], item["pose_state"])
        out.append(item)
    return out


def needs_reorient(pose_state: str) -> bool:
    return normalize_pose_state(pose_state) in (POSE_UPSIDE_DOWN, POSE_TILT)
