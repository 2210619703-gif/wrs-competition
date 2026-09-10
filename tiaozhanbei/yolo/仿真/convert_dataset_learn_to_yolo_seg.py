"""
将 dataset_learn 转为 YOLO-seg 数据集（实例多边形标签）。

优先 annotations.json 的 polygon；不足时从 masks/*.png 提取轮廓；再不行用 bbox 四边形。

用法:
  # 8 大类（默认）
  python convert_dataset_learn_to_yolo_seg.py
  # 细分类（按 annotations.json 的 class 字段，如"轴承-凸轮滚轮"）
  python convert_dataset_learn_to_yolo_seg.py --fine-grained
  python convert_dataset_learn_to_yolo_seg.py --fine-grained --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
from sim_dirs import DIR_DATA

SRC = DIR_DATA

CLASS_NAMES_8 = [
    "bearing", "gear", "machine_tool", "nut",
    "rivet", "roller_bearing", "screw_bolt", "washer",
]

FOLDER_TO_CLASS_8 = {
    "Bearings": 0, "Gears": 1, "Machines & Tools": 2, "Nuts": 3,
    "Rivets": 4, "Roller bearings": 5, "Screws and bolts": 6, "Washers": 7,
}

PREFIX_TO_CLASS_8 = {
    "轴承": 0, "齿轮": 1, "机器工具": 2, "机床": 2,
    "螺母": 3, "铆钉": 4, "滚子轴承": 5, "螺钉螺栓": 6, "垫圈": 7,
}


def class_id_8(obj: dict) -> int | None:
    stl = obj.get("stl_path") or ""
    parts = stl.replace("\\", "/").split("/")
    for folder, cid in FOLDER_TO_CLASS_8.items():
        if folder in parts:
            return cid
    name = obj.get("class") or ""
    for pref, cid in PREFIX_TO_CLASS_8.items():
        if name.startswith(pref):
            return cid
    return None


def scan_fine_classes(samples: list[Path]) -> list[str]:
    """扫描全部 annotations.json，收集细分类名并排序，返回稳定的类别列表。"""
    names: set[str] = set()
    for sample_dir in samples:
        data = json.loads((sample_dir / "annotations.json").read_text(encoding="utf-8"))
        for obj in data.get("objects", []):
            cls = (obj.get("class") or "").strip()
            if cls:
                names.add(cls)
    return sorted(names)


def normalize_poly(pts: np.ndarray, w: int, h: int) -> list[float] | None:
    if pts is None or len(pts) < 3:
        return None
    xs = np.clip(pts[:, 0].astype(np.float64), 0, w - 1) / w
    ys = np.clip(pts[:, 1].astype(np.float64), 0, h - 1) / h
    out = []
    prev = None
    for x, y in zip(xs, ys):
        cur = (round(float(x), 6), round(float(y), 6))
        if prev is not None and cur == prev:
            continue
        out.extend([cur[0], cur[1]])
        prev = cur
    if len(out) < 6:
        return None
    return out


def poly_from_annotation(obj: dict) -> np.ndarray | None:
    poly = obj.get("polygon")
    if not poly or len(poly) < 3:
        return None
    pts = np.array(poly, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    return pts[:, :2]


def poly_from_mask(sample_dir: Path, obj: dict) -> np.ndarray | None:
    rel = obj.get("mask_path")
    if not rel:
        return None
    path = sample_dir / rel
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    bin_m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 16:
        return None
    eps = 0.002 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, max(eps, 1.0), True)
    if len(approx) < 3:
        approx = cnt
    return approx.reshape(-1, 2).astype(np.float64)


def poly_from_bbox(bbox) -> np.ndarray | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def object_to_yolo_line(
    sample_dir: Path, obj: dict, w: int, h: int, stats: dict,
    class_id_fn,
) -> str | None:
    cid = class_id_fn(obj)
    if cid is None:
        stats["skipped_cls"] += 1
        return None

    bbox = obj.get("bbox")
    area = 0.0
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    vis = int(obj.get("visible_pixels") or area)
    if area and area < stats["_min_area"]:
        stats["skipped_tiny"] += 1
        return None
    if vis < stats["_min_visible"]:
        stats["skipped_tiny"] += 1
        return None

    pts = poly_from_annotation(obj)
    src = "poly"
    if pts is None:
        pts = poly_from_mask(sample_dir, obj)
        src = "mask"
    if pts is None:
        pts = poly_from_bbox(bbox)
        src = "bbox"
    if pts is None:
        stats["skipped_nopoly"] += 1
        return None

    if len(pts) > 80:
        cnt = pts.reshape(-1, 1, 2).astype(np.float32)
        eps = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(eps, 1.0), True)
        if len(approx) >= 3:
            pts = approx.reshape(-1, 2).astype(np.float64)

    norm = normalize_poly(pts, w, h)
    if norm is None:
        stats["skipped_nopoly"] += 1
        return None

    stats["instances"] += 1
    stats[f"src_{src}"] += 1
    coords = " ".join(f"{v:.6f}" for v in norm)
    return f"{cid} {coords}"


def load_class_names(path: Path) -> list[str]:
    """从 JSON 列表 / {id: name} / YOLO data.yaml 锁定类别顺序。"""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        names: dict[int, str] = {}
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            key, val = s.split(":", 1)
            if key.isdigit():
                names[int(key)] = val.strip().strip("'\"")
        if not names:
            raise ValueError(f"未从 {path} 解析到 names")
        return [names[i] for i in range(max(names) + 1) if i in names]
    data = json.loads(text)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "names" in data and isinstance(data["names"], dict):
            data = data["names"]
        keys = sorted(int(k) for k in data)
        return [str(data[k] if k in data else data[str(k)]) for k in keys]
    raise ValueError(f"无法解析类别文件: {path}")


def write_yaml(out_dir: Path, class_names: list[str]) -> Path:
    yaml_path = out_dir / "data.yaml"
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        safe = name.replace("'", "''")
        if ":" in safe or safe != name.strip():
            lines.append(f"  {i}: '{safe}'")
        else:
            lines.append(f"  {i}: {safe}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "classes_zh.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return yaml_path


def convert(args):
    src = Path(args.src) if args.src else SRC
    if not src.is_dir():
        raise FileNotFoundError(f"未找到 {src}")

    samples = sorted(
        [p for p in src.iterdir() if p.is_dir() and (p / "annotations.json").exists()],
        key=lambda p: p.name,
    )
    if args.min_id > 0:
        samples = [
            p for p in samples if p.name.isdigit() and int(p.name) >= args.min_id
        ]
    if not samples:
        raise FileNotFoundError(f"{src} 下没有带 annotations.json 的样本目录")

    fine = args.fine_grained
    if fine:
        if args.class_names is not None:
            class_names = load_class_names(args.class_names)
            print(f"[INFO] 锁定类别顺序: {args.class_names} ({len(class_names)} 类)")
        else:
            class_names = scan_fine_classes(samples)
        name_to_id = {n: i for i, n in enumerate(class_names)}
        dst = Path(args.dst) if args.dst else ROOT / "dataset_yolo_learn_seg_fine"
        def class_id_fn(obj):
            cls = (obj.get("class") or "").strip()
            return name_to_id.get(cls)
    else:
        class_names = CLASS_NAMES_8
        dst = Path(args.dst) if args.dst else ROOT / "dataset_yolo_learn_seg"
        class_id_fn = class_id_8

    print(f"[INFO] 模式: {'细分类 (' + str(len(class_names)) + '类)' if fine else '8 大类'}")
    print(f"[INFO] 样本数: {len(samples)}  输出: {dst}")

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(samples) * args.val_ratio)))
    val_set = set(indices[:n_val])

    existing = set()
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        if args.append:
            existing.update(p.stem for p in (dst / "images" / split).glob("*.*"))
        else:
            for f in (dst / "images" / split).glob("*"):
                f.unlink()
            for f in (dst / "labels" / split).glob("*"):
                f.unlink()
    if args.append:
        before = len(samples)
        samples = [p for p in samples if p.name not in existing]
        print(f"[INFO] append: 已有 {len(existing)}，本次新增 {len(samples)}（过滤前 {before}）")

    stats = {
        "train": 0, "val": 0, "instances": 0,
        "skipped_tiny": 0, "skipped_cls": 0, "skipped_nopoly": 0,
        "src_poly": 0, "src_mask": 0, "src_bbox": 0,
        "_min_area": args.min_area, "_min_visible": args.min_visible,
    }

    for i, sample_dir in enumerate(samples):
        split = "val" if i in val_set else "train"
        data = json.loads((sample_dir / "annotations.json").read_text(encoding="utf-8"))
        img_name = data.get("image") or f"{sample_dir.name}.jpg"
        img_src = sample_dir / img_name
        if not img_src.exists():
            cand = list(sample_dir.glob("*.jpg"))
            if not cand:
                print(f"[WARN] 跳过无图片: {sample_dir}")
                continue
            img_src = cand[0]

        w = int(data.get("width") or 1280)
        h = int(data.get("height") or 720)
        lines = []
        for obj in data.get("objects", []):
            line = object_to_yolo_line(sample_dir, obj, w, h, stats, class_id_fn)
            if line:
                lines.append(line)

        stem = sample_dir.name
        shutil.copy2(img_src, dst / "images" / split / f"{stem}.jpg")
        (dst / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats[split] += 1

    yaml_path = write_yaml(dst, class_names)
    print("\n[DONE] YOLO-seg 转换完成")
    print(f"  输出: {dst}")
    print(f"  类别数: {len(class_names)}")
    print(f"  train={stats['train']} val={stats['val']} instances={stats['instances']}")
    print(f"  来源: poly={stats['src_poly']} mask={stats['src_mask']} bbox回退={stats['src_bbox']}")
    print(f"  跳过: tiny={stats['skipped_tiny']} unknown_cls={stats['skipped_cls']} nopoly={stats['skipped_nopoly']}")
    print(f"  yaml: {yaml_path}")
    if fine:
        print(f"\n  类别列表（前10）: {class_names[:10]}")
        print(f"  完整列表见: {yaml_path}")
    print(
        "\n训练:\n"
        + (
            "  python train_annotations_yolo.py --train --model best.pt --epochs 100"
            if fine
            else "  python train_dataset_learn_seg.py --train --model yolo26s-seg.pt --epochs 100"
        )
    )


def parse_args():
    p = argparse.ArgumentParser(description="dataset_learn -> YOLO-seg")
    p.add_argument("--fine-grained", action="store_true", help="细分类模式（按 class 字段，而非 8 大类）")
    p.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="锁定细分类顺序（JSON 列表 / {id:name} / data.yaml），须与 best.pt 的 names 一致",
    )
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--min-area", type=float, default=64.0)
    p.add_argument("--min-visible", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--append", action="store_true", help="不删已有 train/val，只追加新样本")
    p.add_argument("--min-id", type=int, default=0, help="只转换编号 >= min-id 的样本")
    p.add_argument("--src", type=Path, default=None, help="仿真样本根目录，默认 dataset_learn")
    p.add_argument("--dst", type=Path, default=None, help="YOLO 输出目录")
    return p.parse_args()


if __name__ == "__main__":
    convert(parse_args())
