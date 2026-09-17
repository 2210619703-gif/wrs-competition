# -*- coding: utf-8 -*-
"""把口语变成水果抓放指令。"""
from __future__ import annotations

import re

from fruit_config import FRUIT_EN, FRUIT_ZH, JOG_ZH, ZH_TO_EN, dest_label

_ALL_KEYS = ("所有水果", "全部水果", "所有的水果", "全部", "所有")
_PLACE_KEYS = ("放进", "放到", "放入", "放置", "装进", "装箱")
_PICK_KEYS = ("抓", "拿", "夹", "取")
_WANDER_KEYS = (
    "随便动",
    "自己动",
    "乱动",
    "晃一晃",
    "晃晃",
    "活动一下",
    "随便走走",
    "自己走走",
    "自由活动",
    "动一动",
    "动动",
)
_DIR_PHRASES = (
    ("up", ("往上", "向上", "抬高", "升高", "抬起来", "上一点", "再高点", "高一点", "上移")),
    ("down", ("往下", "向下", "降低", "降下来", "下一点", "低一点", "再低点", "下移")),
    ("left", ("往左边", "向左边", "往左", "向左", "左边", "左一点", "左移")),
    ("right", ("往右边", "向右边", "往右", "向右", "右边", "右一点", "右移")),
    ("forward", ("往前边", "向前边", "往前", "向前", "前边", "前面", "前一点", "前移")),
    ("back", ("往后退", "向后退", "往后", "向后", "后边", "后面", "后一点", "退后", "后退", "后移")),
)
_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_NUM_TOKEN = r"[一二两三四五六七八九十俩\d]+"
_UNIT = r"(?:个|只|颗|枚)?"


def _parse_num(tok: str) -> int | None:
    tok = (tok or "").strip()
    if not tok:
        return None
    if tok.isdigit():
        n = int(tok)
        return n if n > 0 else None
    if tok in _CN_NUM:
        return _CN_NUM[tok]
    return None


def _fruit_mentions(raw: str, compact: str) -> list[str]:
    fruits: list[str] = []
    for zh, en in sorted(ZH_TO_EN.items(), key=lambda kv: len(kv[0]), reverse=True):
        if zh in raw and en not in fruits:
            fruits.append(en)
    for en in FRUIT_EN:
        if en in compact and en not in fruits:
            fruits.append(en)
    return fruits


def _per_fruit_counts(compact: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    names = "|".join(sorted((re.escape(z) for z in ZH_TO_EN), key=len, reverse=True))
    used: list[tuple[int, int]] = []
    items = sorted(ZH_TO_EN.items(), key=lambda kv: len(kv[0]), reverse=True)
    for zh, en in items:
        zh_re = re.escape(zh)
        patterns = (
            rf"({_NUM_TOKEN})\s*{_UNIT}\s*{zh_re}",
            rf"{zh_re}\s*({_NUM_TOKEN})\s*(?:个|只|颗|枚)(?!{names})",
        )
        for pat in patterns:
            for m in re.finditer(pat, compact):
                span = (m.start(), m.end())
                if any(span[0] < u[1] and span[1] > u[0] for u in used):
                    continue
                n = _parse_num(m.group(1))
                if not n:
                    continue
                used.append(span)
                counts[en] = counts.get(en, 0) + n
    return counts


def _bare_count(compact: str, fruits: list[str], counts: dict[str, int]) -> int | None:
    if counts:
        return None
    m = re.search(rf"({_NUM_TOKEN})\s*(?:个|只|颗|枚)", compact)
    if not m:
        return None
    return _parse_num(m.group(1))


def _motion_dirs(compact: str) -> list[str]:
    hits: list[tuple[int, int, str]] = []
    for name, phrases in _DIR_PHRASES:
        for p in sorted(phrases, key=len, reverse=True):
            start = 0
            while True:
                i = compact.find(p, start)
                if i < 0:
                    break
                hits.append((i, i + len(p), name))
                start = i + len(p)
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    used: list[tuple[int, int]] = []
    dirs: list[str] = []
    for a, b, name in hits:
        if any(a < u[1] and b > u[0] for u in used):
            continue
        used.append((a, b))
        dirs.append(name)
    return dirs


def parse_fruit_cmd(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("空指令")
    compact = re.sub(r"\s+", "", raw).lower().replace("ａ", "a").replace("ｂ", "b")

    dest = None
    if re.search(r"a\s*(框|筐|盒|收纳)", compact) or re.search(
        r"(框|筐|盒)a", compact
    ) or re.search(r"(放进|放到|放入|装进)a", compact):
        dest = "A"
    if re.search(r"b\s*(框|筐|盒|收纳)", compact) or re.search(
        r"(框|筐|盒)b", compact
    ) or re.search(r"(放进|放到|放入|装进)b", compact):
        dest = "B"
    # 空位：必须带「放」，避免「往左」被当成放到左边。
    if re.search(r"(放进|放到|放入|放置|装进|放至|放到了).{0,6}(左边|左侧|左面|左空位|左位)", compact) or re.search(
        r"(左边|左侧|左面|左空位|左位).{0,4}(放)", compact
    ) or re.search(r"(放)(到)?(左边|左侧|左面|左空位|左位)", compact):
        dest = "LEFT"
    if re.search(r"(放进|放到|放入|放置|装进|放至|放到了).{0,6}(右边|右侧|右面|右空位|右位)", compact) or re.search(
        r"(右边|右侧|右面|右空位|右位).{0,4}(放)", compact
    ) or re.search(r"(放)(到)?(右边|右侧|右面|右空位|右位)", compact):
        dest = "RIGHT"

    want_all = any(k in raw for k in _ALL_KEYS) or "all" in compact
    fruits = _fruit_mentions(raw, compact)
    counts = {} if want_all else _per_fruit_counts(compact)
    count = None if want_all else _bare_count(compact, fruits, counts)
    if counts:
        for en in counts:
            if en not in fruits:
                fruits.append(en)

    fruit_like = bool(fruits) or dest is not None or "水果" in raw
    dirs = [] if fruit_like else _motion_dirs(compact)
    wander = (not fruit_like) and any(k in compact or k in raw for k in _WANDER_KEYS)
    if wander and not dirs:
        return {
            "raw": raw,
            "action": "wander",
            "fruits": [],
            "all": False,
            "count": None,
            "counts": {},
            "dest": None,
            "dirs": [],
        }
    if dirs:
        return {
            "raw": raw,
            "action": "jog",
            "fruits": [],
            "all": False,
            "count": None,
            "counts": {},
            "dest": None,
            "dirs": dirs,
        }

    has_place = dest is not None or any(w in raw for w in _PLACE_KEYS)
    # 只有明确说「所有/全部/水果」才抓全部。听成「柿子」这种未知词时
    # 不能当成全部，否则会把筐里的苹果也抓走。
    if want_all or (not fruits and count is None and "水果" in raw):
        fruits = []
        want_all = True
        counts = {}
        count = None

    action = "pick_place" if has_place and dest else "pick"
    return {
        "raw": raw,
        "action": action,
        "fruits": fruits,
        "all": bool(want_all),
        "count": count,
        "counts": counts,
        "dest": dest,
        "dirs": [],
    }


def cmd_quota(cmd: dict) -> int | None:
    counts = cmd.get("counts") or {}
    if counts:
        return int(sum(counts.values()))
    n = cmd.get("count")
    return int(n) if n else None


def describe_cmd(cmd: dict) -> str:
    if cmd.get("action") == "wander":
        return "自己动一动"
    if cmd.get("action") == "jog":
        bits = [f"往{JOG_ZH.get(d, d)}" for d in (cmd.get("dirs") or [])]
        return ("".join(bits) + "移动") if bits else "点动"
    dest = cmd.get("dest")
    counts = cmd.get("counts") or {}
    n = cmd.get("count")
    if counts:
        obj = "、".join(f"{c}个{FRUIT_ZH.get(en, en)}" for en, c in counts.items())
    elif n:
        kind = "、".join(FRUIT_ZH.get(f, f) for f in cmd.get("fruits") or []) or "水果"
        obj = f"{n}个{kind}"
    elif cmd.get("all"):
        obj = "全部水果"
    elif not cmd.get("fruits"):
        obj = "没听清的水果"
    else:
        obj = "、".join(FRUIT_ZH.get(f, f) for f in cmd["fruits"])
    if cmd.get("action") == "pick_place" and dest:
        return f"把{obj}放到{dest_label(dest)}"
    return f"抓起{obj}"
