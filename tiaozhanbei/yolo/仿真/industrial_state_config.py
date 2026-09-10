"""零件状态（三态）配置：与规范类别 ID 独立，不修改 §7 类别编号。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

STATE_NAMES: list[str] = ["normal", "inverted", "fallen"]
STATE_TO_ID: dict[str, int] = {n: i for i, n in enumerate(STATE_NAMES)}
ID_TO_STATE: dict[int, str] = {i: n for i, n in enumerate(STATE_NAMES)}

# 与 dataset_learn 已有标注的姿态角分布一致
_FALLEN_ANGLE = math.pi / 2.0
_INVERTED_ANGLE = math.pi

# 外观上「倒放≈正常」的大类：倒放并入正常，只保留 normal/fallen
_SYMMETRIC_PREFIXES: tuple[str, ...] = (
    "垫圈",
    "螺母",
    "铆钉",
    "轴承",
    "滚子轴承",
    "齿轮",
)

# 明确非对称：即使前缀在对称列表，也保留三态
_ASYMMETRIC_KEYWORDS: tuple[str, ...] = (
    "齿条",
    "螺丝刀",
    "扳手",
    "钳子",
    "台钳",
    "刷子",
    "切削",
    "棘轮",
    "砂光",
    "磨削",
    "扭矩",
    "六角头",
    "圆柱头",
    "圆头",
    "沉头",
    "十字",
    "一字",
    "内六角",
    "紧定",
    "石膏板",
)


def is_symmetric_part(class_name: str | None) -> bool:
    """倒放在图像上几乎不可辨的零件 → True。"""
    name = (class_name or "").strip()
    if not name:
        return True
    if name.startswith("机器工具"):
        return False
    for kw in _ASYMMETRIC_KEYWORDS:
        if kw in name:
            return False
    for pref in _SYMMETRIC_PREFIXES:
        if name.startswith(pref):
            return True
    return False


def canonicalize_state(class_name: str | None, state: str | None) -> str:
    """对称件：inverted → normal；别名与非法值 → 规范三态。"""
    aliases = {
        "upside_down": "inverted",
        "inverted": "inverted",
        "倒放": "inverted",
        "倒置": "inverted",
        "tilt": "fallen",
        "tilted": "fallen",
        "tipped": "fallen",
        "lying": "fallen",
        "fallen": "fallen",
        "倾倒": "fallen",
        "侧倒": "fallen",
        "upright": "normal",
        "normal": "normal",
        "正放": "normal",
        "正常": "normal",
    }
    t = str(state or "normal").strip().lower()
    st = aliases.get(t, t)
    if st not in STATE_TO_ID:
        st = "normal"
    if is_symmetric_part(class_name) and st == "inverted":
        return "normal"
    return st


def infer_state_from_pose(pose: dict | None, class_name: str | None = None) -> str:
    """按 max(|roll|, |pitch|) 推断放置状态，再按零件对称性归一。"""
    p = pose or {}
    mx = max(abs(float(p.get("roll", 0.0))), abs(float(p.get("pitch", 0.0))))
    if mx < 0.5:
        st = "normal"
    elif mx < 2.3:
        st = "fallen"
    else:
        st = "inverted"
    return canonicalize_state(class_name, st)


def sample_state(
    rng: np.random.Generator,
    probs: Sequence[float] | None = None,
    class_name: str | None = None,
) -> str:
    """按概率抽样状态。对称件只在 normal/fallen 间抽。"""
    if class_name is not None and is_symmetric_part(class_name):
        # probs 若给了三元，把 inverted 并入 normal
        if probs is not None and len(probs) == 3:
            pn, pi, pf = [float(x) for x in probs]
            p2 = [pn + pi, pf]
        else:
            p2 = [0.45, 0.55]
        s = float(sum(p2)) or 1.0
        p2 = [x / s for x in p2]
        return str(rng.choice(["normal", "fallen"], p=p2))

    p = list(probs) if probs is not None else [0.34, 0.33, 0.33]
    if len(p) != 3:
        raise ValueError("probs 需对应 [normal, inverted, fallen]")
    s = float(sum(p))
    if s <= 0:
        return "normal"
    p = [x / s for x in p]
    return str(rng.choice(STATE_NAMES, p=p))


def sample_placement_euler(
    state: str,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """生成与 state 标签一致的 (roll, pitch, yaw)。

    - normal: 近直立，仅微扰
    - fallen: 一侧约 ±π/2（倾倒）
    - inverted: 一侧约 ±π（倒放）
    """
    yaw = float(rng.uniform(0.0, 2.0 * math.pi))
    st = (state or "normal").strip().lower()
    if st == "fallen":
        tip = float(_FALLEN_ANGLE + rng.uniform(-0.12, 0.12))
        tip *= float(rng.choice([-1.0, 1.0]))
        if rng.random() < 0.5:
            return tip, float(rng.uniform(-0.08, 0.08)), yaw
        return float(rng.uniform(-0.08, 0.08)), tip, yaw
    if st == "inverted":
        tip = float(_INVERTED_ANGLE + rng.uniform(-0.12, 0.12))
        tip *= float(rng.choice([-1.0, 1.0]))
        if rng.random() < 0.5:
            return tip, float(rng.uniform(-0.08, 0.08)), yaw
        return float(rng.uniform(-0.08, 0.08)), tip, yaw
    # normal
    return (
        float(rng.uniform(-0.06, 0.06)),
        float(rng.uniform(-0.06, 0.06)),
        yaw,
    )


def bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
