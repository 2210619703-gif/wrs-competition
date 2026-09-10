"""
YOLO 检测框中文显示。

- 若模型本身是中文细分类（如「轴承-四点接触球轴承」），直接显示类名；
- 若是 8 大类英文，可映射为大类中文（轴承、齿轮…）；
- 状态可显示为 正常/倒放/倾倒。

用法:
  from yolo_label_zh import class_to_zh, draw_box_label_bgr
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 与 industrial_class_config 8class 对应的中文显示名（不修改训练 ID）
CLASS_EN_TO_ZH: dict[str, str] = {
    "bearing": "轴承",
    "gear": "齿轮",
    "machine_tool": "机器工具",
    "nut": "螺母",
    "rivet": "铆钉",
    "roller_bearing": "滚子轴承",
    "screw_bolt": "螺钉螺栓",
    "washer": "垫圈",
}

STATE_EN_TO_ZH: dict[str, str] = {
    "normal": "正常",
    "inverted": "倒放",
    "fallen": "倾倒",
}

_FONT_CACHE: dict[int, object] = {}


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def class_to_zh(name: str) -> str:
    raw = (name or "").strip()
    if _has_cjk(raw):
        return raw
    key = raw.lower()
    return CLASS_EN_TO_ZH.get(key, raw)


def state_to_zh(name: str) -> str:
    key = (name or "").strip().lower()
    return STATE_EN_TO_ZH.get(key, name)


def format_det_label(
    class_name: str,
    conf: float | None = None,
    state: str | None = None,
    state_conf: float | None = None,
    zh: bool = False,
) -> str:
    """拼检测框文字；zh=True 时类名/状态尽量中文（细分类名原样保留）。"""
    c = class_to_zh(class_name) if zh or _has_cjk(class_name or "") else class_name
    if state:
        s = state_to_zh(state) if zh else state
        if state_conf is not None:
            return f"{c}|{s} {state_conf:.2f}"
        return f"{c}|{s}"
    if conf is not None:
        return f"{c} {conf:.2f}"
    return c


def _load_font(size: int = 16):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont

    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    font = None
    for p in candidates:
        if p.is_file():
            font = ImageFont.truetype(str(p), size=size)
            break
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_box_label_bgr(
    img_bgr: np.ndarray,
    xyxy,
    label: str,
    color_bgr=(0, 200, 0),
    thickness: int = 2,
    font_size: int = 16,
    use_pil: bool = True,
) -> np.ndarray:
    """
    在 BGR 图上画框+标签。
    use_pil=True 时用 PIL 画中文（推荐）；False 时退回 cv2.putText（中文会乱码）。
    """
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)

    if not use_pil:
        cv2.putText(
            img_bgr,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_bgr,
            2,
        )
        return img_bgr

    from PIL import Image, ImageDraw

    # PIL 用 RGB；框色转 RGB
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _load_font(font_size)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    label_y = max(0, y1 - th - 4)
    draw.rectangle([x1, label_y, x1 + tw + 4, label_y + th + 2], fill=(255, 255, 255))
    draw.rectangle(
        [x1, label_y, x1 + tw + 4, label_y + th + 2], outline=color_rgb, width=1
    )
    draw.text((x1 + 2, label_y + 1), label, fill=(0, 0, 0), font=font)
    out = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    np.copyto(img_bgr, out)
    return img_bgr
