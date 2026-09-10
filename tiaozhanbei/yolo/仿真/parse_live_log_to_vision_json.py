"""
兜底：从 [LIVE] 终端日志文本解析 vision_objects JSON。

用法:
  python parse_live_log_to_vision_json.py terminal.log
  python parse_live_log_to_vision_json.py terminal.log --out vision.json
  type terminal.log | python parse_live_log_to_vision_json.py -
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 示例行:
#   [1] fastener       conf=0.84  pos=(-0.201, 0.367, 0.010) yaw=42.6 R=[+0.74,-0.68,+0.00; +0.68,+0.74,+0.00]  pts=1148
LINE_RE = re.compile(
    r"^\s*\[\d+\]\s+(\S+)\s+conf=([\d.]+)\s+"
    r"pos=\(\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\)"
    r"(?:\s+yaw=([+-]?[\d.]+))?"
    r"(?:\s+R=\[([^\]]+)\])?",
    re.MULTILINE,
)


def _parse_rotate_matrix(raw: str | None) -> list[list[float]] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(";")]
    rows: list[list[float]] = []
    for part in parts:
        nums = [float(x) for x in re.findall(r"[+-]?[\d.]+", part)]
        if len(nums) == 3:
            rows.append([round(n, 2) for n in nums])
    return rows if len(rows) == 2 else None


def parse_live_log(text: str) -> list[dict]:
    items: list[dict] = []
    for m in LINE_RE.finditer(text):
        name, conf, x, y, z, yaw, rot_raw = m.groups()
        obj: dict = {
            "name": name,
            "conf": round(float(conf), 4),
            "coord": [round(float(x), 3), round(float(y), 3)],
            "z": round(float(z), 3),
        }
        if yaw is not None:
            obj["yaw"] = round(float(yaw), 1)
        rot = _parse_rotate_matrix(rot_raw)
        if rot is not None:
            obj["rotate_matrix"] = rot
        items.append(obj)
    return items


def main():
    p = argparse.ArgumentParser(description="LIVE 日志 → vision_objects JSON")
    p.add_argument("log", help="日志文件路径，或 - 表示 stdin")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    if args.log == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    items = parse_live_log(text)
    payload = json.dumps(items, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        print(f"[INFO] parsed {len(items)} objects -> {args.out}", file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()
