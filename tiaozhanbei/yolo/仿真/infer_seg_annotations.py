"""
YOLO-seg 推理，输出与仿真 dataset_learn/annotations.json 同字段。

物体字段:
  id, class, bbox, polygon, mask_path, pose_6d, state, visible_pixels, stl_path
顶层字段:
  image, width, height, objects, camera
若图片旁已有 annotations.json，会拷贝 camera / seed / scene_layout 等采集元数据。

用法:
  python infer_seg_annotations.py --sample 0001
  python infer_seg_annotations.py --image ../dataset_learn/0001/0001.jpg --pretty --save-masks
  python infer_seg_annotations.py --image 数据样例/0001/0001.jpg --weights-small 权重/yolo26s-seg-tiny19.pt --pretty --save-masks
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from annotations_format import (
    DEFAULT_CAMERA,
    make_annotations,
    make_object,
    make_pose_6d,
)
from industrial_state_config import STATE_NAMES, canonicalize_state, bbox_iou_xyxy

ROOT = Path(__file__).resolve().parent
TB_DIR = ROOT.parent.parent
DATASET_ROOT = TB_DIR / "dataset_learn"
OUTPUTS = ROOT / "outputs"
DEFAULT_WEIGHTS = ROOT / "best.pt"
if not DEFAULT_WEIGHTS.exists():
    DEFAULT_WEIGHTS = ROOT / "权重" / "yolo26s-seg-fine.pt"
DEFAULT_WEIGHTS_SMALL = ROOT / "权重" / "yolo26s-seg-tiny19.pt"
DEFAULT_STATE_HEAD = ROOT / "权重" / "state-head-yolo26.pt"
if not DEFAULT_STATE_HEAD.exists():
    DEFAULT_STATE_HEAD = ROOT / "weights" / "state-head-yolo26.pt"
if not DEFAULT_STATE_HEAD.exists():
    DEFAULT_STATE_HEAD = ROOT / "runs" / "classify" / "state-head-yolo26-v3" / "weights" / "best.pt"
DEFAULT_MODELS_DIR = TB_DIR / "Part_Model"


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-seg → annotations.json 格式")
    p.add_argument("--sample", default="", help="dataset_learn 样本号，如 0001；与 --image 二选一")
    p.add_argument("--image", type=Path, default=None, help="输入图片；不填则用 --sample")
    p.add_argument("--depth", type=Path, default=None, help="深度图 .npy；不指定则尝试同目录 depth_m.npy")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO-seg 权重，默认 best.pt")
    p.add_argument(
        "--weights-small",
        type=Path,
        default=DEFAULT_WEIGHTS_SMALL if DEFAULT_WEIGHTS_SMALL.exists() else None,
        help="19 类近景小件权重；只补大模型没框到的小目标，不覆盖已有框",
    )
    p.add_argument("--no-small", action="store_true", help="只用 124 类全桌，不做小件补检")
    p.add_argument("--small-conf", type=float, default=0.40, help="小件权重置信度")
    p.add_argument("--small-max-area", type=float, default=0.08, help="小件框面积上限（占全图比例）")
    p.add_argument("--small-iou", type=float, default=0.30, help="与大模型框 IoU 超过此值则丢弃小件框")
    p.add_argument("--state-head", type=Path, default=None, help="状态分类头（可选）")
    p.add_argument("--out", type=Path, default=None, help="最终 JSON；默认 outputs/000x.json")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640, help="全图推理尺寸；--slice 时全图用 1280")
    p.add_argument("--slice", action="store_true", help="全图1280 + 重叠切片，提高小件召回")
    p.add_argument("--slice-tile", type=int, default=640)
    p.add_argument("--slice-overlap", type=float, default=0.25)
    p.add_argument("--save-masks", action="store_true", help="写出 masks/*.png，并填 mask_path")
    p.add_argument("--mask-dir", type=Path, default=None)
    p.add_argument(
        "--save-pcd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把每个物体 mask 反投影成点云 ply（默认开，需深度图）",
    )
    p.add_argument("--pcd-dir", type=Path, default=None, help="点云目录，默认 <输出目录>/pointclouds")
    p.add_argument("--pcd-stride", type=int, default=1, help="点云采样步长，1=全像素")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--pretty", action="store_true", help="缩进 2 空格（与仿真 JSON 一致）")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    return p.parse_args()


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def merge_small_dets(big_items, small_dets, width: int, height: int, max_area: float, iou_thr: float):
    """小件只补空缺：框要小，且不能和大模型已有框重叠。"""
    img_area = float(width * height)
    added = []
    for d in small_dets:
        x1, y1, x2, y2 = [float(v) for v in d.bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if img_area > 0 and area / img_area > max_area:
            continue
        if any(bbox_iou_xyxy(d.bbox, b[2]) >= iou_thr for b in big_items):
            continue
        mask_bin = d.mask if d.mask is not None else np.zeros((height, width), dtype=np.uint8)
        added.append((d.cls_name, float(d.conf), [float(v) for v in d.bbox], mask_bin))
    return added


def imwrite_unicode(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"无法编码: {path}")
    buf.tofile(str(path))


def bootstrap_tiaozhanbei() -> None:
    pkg111 = ROOT / "tiaozhanbei111"
    if not pkg111.is_dir():
        return
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if "tiaozhanbei" not in sys.modules:
        pkg = types.ModuleType("tiaozhanbei")
        pkg.__path__ = [str(pkg111)]
        sys.modules["tiaozhanbei"] = pkg
    elif getattr(sys.modules["tiaozhanbei"], "__path__", None) is None:
        sys.modules["tiaozhanbei"].__path__ = [str(pkg111)]


def load_zh_maps() -> tuple[dict[str, str], dict[str, str]]:
    bootstrap_tiaozhanbei()
    try:
        from tiaozhanbei.sim.environment import CATEGORY_ZH, LEAF_DIR_ZH

        return dict(CATEGORY_ZH), dict(LEAF_DIR_ZH)
    except Exception:
        return {}, {}


def model_roots(preferred: Path) -> list[Path]:
    cands = [
        preferred,
        TB_DIR / "Part_Model",
        ROOT / "industrial_models",
        ROOT / "tiaozhanbei111" / "Part_Model",
        Path(r"F:\wrs-main-competition\tiaozhanbei\Part_Model"),
    ]
    out, seen = [], set()
    for p in cands:
        if p is None or not p.is_dir():
            continue
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def class_from_stl_rel(rel: Path, category_zh: dict, leaf_zh: dict) -> str:
    parts = rel.parts
    if not parts:
        return ""
    top = parts[0]
    leaf = parts[-2] if len(parts) >= 2 else Path(parts[-1]).stem
    return f"{category_zh.get(top, top)}-{leaf_zh.get(leaf, leaf)}"


def localize_stl_path(raw: str | None, roots: list[Path]) -> str | None:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    norm = raw.replace("\\", "/")
    rel = None
    for marker in ("Part_Model/", "industrial_models/"):
        if marker in norm:
            rel = norm.split(marker, 1)[1]
            break
    if rel is None:
        rel = Path(norm).name
    for root in roots:
        cand = root / rel
        if cand.is_file():
            return str(cand.resolve())
        hits = list(root.rglob(Path(rel).name))
        if hits:
            return str(hits[0].resolve())
    return None


def build_stl_map(models_dir: Path) -> dict[str, str]:
    category_zh, leaf_zh = load_zh_maps()
    stl_map: dict[str, str] = {}
    for root in model_roots(models_dir):
        for stl in sorted(root.rglob("*.stl")):
            try:
                rel = stl.relative_to(root)
            except ValueError:
                continue
            cls_name = class_from_stl_rel(rel, category_zh, leaf_zh)
            if cls_name and cls_name not in stl_map:
                stl_map[cls_name] = str(stl.resolve())
    return stl_map


def match_sidecar_object(bbox: list[int], cls_name: str, sidecar: dict | None) -> dict | None:
    if not sidecar:
        return None
    best, best_score = None, 0.3
    for obj in sidecar.get("objects") or []:
        gb = obj.get("bbox")
        if not gb or len(gb) != 4:
            continue
        iou = bbox_iou_xyxy(bbox, gb)
        if iou < 0.3:
            continue
        score = iou + (0.2 if (obj.get("class") or "") == cls_name else 0.0)
        if score > best_score:
            best, best_score = obj, score
    return best


def load_sidecar(image: Path) -> dict | None:
    cand = image.parent / "annotations.json"
    if not cand.is_file():
        return None
    try:
        return json.loads(cand.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def mask_to_fullsize(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return (mask > 0.5).astype(np.uint8)
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return (resized > 0.5).astype(np.uint8)


def mask_to_polygon(mask_bin: np.ndarray, epsilon_ratio: float = 0.01) -> list[list[int]]:
    contours, _ = cv2.findContours(
        mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
    if len(approx) < 3:
        approx = cnt
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def polygon_from_xy(xy, width: int, height: int) -> list[list[int]]:
    if xy is None or len(xy) < 3:
        return []
    pts = []
    prev = None
    for p in xy:
        x = int(np.clip(round(float(p[0])), 0, width - 1))
        y = int(np.clip(round(float(p[1])), 0, height - 1))
        cur = [x, y]
        if cur != prev:
            pts.append(cur)
            prev = cur
    return pts if len(pts) >= 3 else []


def bbox_from_xyxy(xyxy, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = int(max(0, min(width - 1, np.floor(x1))))
    y1 = int(max(0, min(height - 1, np.floor(y1))))
    x2 = int(max(0, min(width - 1, np.ceil(x2))))
    y2 = int(max(0, min(height - 1, np.ceil(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def crop_xyxy(img: np.ndarray, xyxy, pad: float = 0.12) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w - 1, x2 + pad * bw))
    y2 = int(min(h - 1, y2 + pad * bh))
    return img[y1:y2, x1:x2].copy()


def predict_state(state_model, crop_bgr: np.ndarray) -> str:
    results = state_model.predict(crop_bgr, verbose=False)
    r0 = results[0]
    probs = getattr(r0, "probs", None)
    if probs is None:
        return "normal"
    top1 = int(probs.top1)
    names = r0.names or {i: n for i, n in enumerate(STATE_NAMES)}
    name = names.get(top1, STATE_NAMES[top1] if top1 < len(STATE_NAMES) else "normal")
    return str(name)


def state_to_rp(state: str | None) -> tuple[float, float]:
    st = (state or "normal").lower()
    if st == "fallen":
        return float(np.pi / 2.0), 0.0
    if st == "inverted":
        return float(np.pi), 0.0
    return 0.0, 0.0


def estimate_yaw_from_mask(mask_bin: np.ndarray) -> float:
    """图像平面主轴角（无深度时的回退）。"""
    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 10:
        return 0.0
    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    return float(np.radians(rect[2]))


def camera_arrays(camera: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    R = np.asarray(camera.get("rotmat") or DEFAULT_CAMERA["rotmat"], dtype=np.float64)
    t = np.asarray(camera.get("position") or DEFAULT_CAMERA["position"], dtype=np.float64)
    intr = camera.get("intrinsics") or DEFAULT_CAMERA["intrinsics"]
    return R, t.reshape(3), intr


def pixels_depth_to_world(
    us: np.ndarray,
    vs: np.ndarray,
    zs: np.ndarray,
    camera: dict,
) -> np.ndarray:
    """像素 + 深度 → 桌面世界系。约定与 SimDataCamera.depth_to_pointcloud 一致。"""
    R, t, intr = camera_arrays(camera)
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    x_cam = (us - cx) * zs / fx
    y_cam = zs
    z_cam = -(vs - cy) * zs / fy
    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    return pts_cam @ R.T + t


def mask_to_world_points(
    mask_bin: np.ndarray,
    depth: np.ndarray,
    camera: dict,
    bbox: list[int] | None = None,
) -> np.ndarray | None:
    if mask_bin is not None and mask_bin.any():
        ys, xs = np.where(mask_bin > 0)
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        ys, xs = np.mgrid[y1 : y2 + 1, x1 : x2 + 1]
        ys, xs = ys.ravel(), xs.ravel()
    else:
        return None
    if len(xs) < 8:
        return None
    step = max(1, len(xs) // 2500)
    xs = xs[::step].astype(np.float64)
    ys = ys[::step].astype(np.float64)
    h, w = depth.shape[:2]
    xi = np.clip(xs.astype(int), 0, w - 1)
    yi = np.clip(ys.astype(int), 0, h - 1)
    zs = depth[yi, xi].astype(np.float64)
    near = float(camera.get("near") or 0.01)
    far = float(camera.get("far") or 5.0)
    valid = np.isfinite(zs) & (zs > near) & (zs < far * 0.98)
    if int(valid.sum()) < 8:
        return None
    return pixels_depth_to_world(xs[valid], ys[valid], zs[valid], camera)


def mask_to_world_cloud(
    mask_bin: np.ndarray,
    depth: np.ndarray,
    camera: dict,
    img_bgr: np.ndarray | None = None,
    bbox: list[int] | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """mask 内像素 → 世界系点云 + RGB。比位姿估计用的点更密。"""
    if mask_bin is not None and mask_bin.any():
        ys, xs = np.where(mask_bin > 0)
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        ys, xs = np.mgrid[y1 : y2 + 1, x1 : x2 + 1]
        ys, xs = ys.ravel(), xs.ravel()
    else:
        return None, None
    stride = max(1, int(stride))
    xs = xs[::stride]
    ys = ys[::stride]
    if len(xs) < 8:
        return None, None
    h, w = depth.shape[:2]
    xi = np.clip(xs.astype(int), 0, w - 1)
    yi = np.clip(ys.astype(int), 0, h - 1)
    zs = depth[yi, xi].astype(np.float64)
    near = float(camera.get("near") or 0.01)
    far = float(camera.get("far") or 5.0)
    valid = np.isfinite(zs) & (zs > near) & (zs < far * 0.98)
    if int(valid.sum()) < 8:
        return None, None
    pts = pixels_depth_to_world(
        xs[valid].astype(np.float64), ys[valid].astype(np.float64), zs[valid], camera
    )
    if img_bgr is not None:
        colors = img_bgr[yi[valid], xi[valid]][:, ::-1].astype(np.uint8)
    else:
        colors = np.full((len(pts), 3), 180, dtype=np.uint8)
    return pts, colors


def save_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(len(points))
    if colors is None or len(colors) != n:
        colors = np.full((n, 3), 180, dtype=np.uint8)
    colors = colors.astype(np.uint8)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for p, c in zip(points, colors):
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def wrap_yaw(yaw: float) -> float:
    y = float(yaw) % (2.0 * np.pi)
    return y if y >= 0 else y + 2.0 * np.pi


def ang_diff(a: float, b: float) -> float:
    d = abs(wrap_yaw(a) - wrap_yaw(b))
    return min(d, 2.0 * np.pi - d)


def yaw_from_world_xy(pts: np.ndarray) -> float:
    """桌面点云 XY 主轴 → yaw（弧度，[0, 2π)）。"""
    xy = pts[:, :2] - np.median(pts[:, :2], axis=0)
    if xy.shape[0] < 8 or float(np.linalg.norm(xy.std(axis=0))) < 1e-6:
        return 0.0
    cov = (xy.T @ xy) / max(xy.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    d = evecs[:, int(np.argmax(evals))]
    if d[0] < 0:
        d = -d
    return wrap_yaw(float(np.arctan2(d[1], d[0])))


def pixel_to_desk_xy(u: float, v: float, camera: dict) -> np.ndarray | None:
    """像素射线与桌面 z=0 相交（细长件 yaw 比端点深度更稳）。"""
    R, t, intr = camera_arrays(camera)
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    dirc = np.array([(u - cx) / fx, 1.0, -(v - cy) / fy], dtype=np.float64)
    dirw = R @ dirc
    if abs(float(dirw[2])) < 1e-8:
        return None
    s = -float(t[2]) / float(dirw[2])
    if s <= 0:
        return None
    return t + s * dirw


def yaw_from_mask_long_axis(
    mask_bin: np.ndarray,
    depth: np.ndarray,
    camera: dict,
    pts_world: np.ndarray,
) -> float:
    """图像长轴两端点落到桌面，得到世界 yaw。"""
    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 12:
        return yaw_from_world_xy(pts_world)
    img = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    c = img.mean(axis=0)
    cov = ((img - c).T @ (img - c)) / max(len(img) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    d = evecs[:, int(np.argmax(evals))]
    proj = (img - c) @ d
    i0, i1 = int(np.argmin(proj)), int(np.argmax(proj))
    p0, p1 = img[i0], img[i1]
    w0 = pixel_to_desk_xy(float(p0[0]), float(p0[1]), camera)
    w1 = pixel_to_desk_xy(float(p1[0]), float(p1[1]), camera)
    if w0 is None or w1 is None:
        return yaw_from_world_xy(pts_world)
    vec = w1[:2] - w0[:2]
    if float(np.linalg.norm(vec)) < 1e-6:
        return yaw_from_world_xy(pts_world)
    yaw = wrap_yaw(float(np.arctan2(vec[1], vec[0])))
    pca = yaw_from_world_xy(pts_world)
    if ang_diff(yaw + np.pi, pca) < ang_diff(yaw, pca):
        yaw = wrap_yaw(yaw + np.pi)
    return yaw


def inlier_world_points(pts: np.ndarray, band_m: float = 0.04) -> np.ndarray:
    """去掉深度飞点，只留中位高度附近的点。"""
    z = pts[:, 2]
    med = float(np.median(z))
    keep = np.abs(z - med) < band_m
    if int(keep.sum()) < 16:
        return pts
    return pts[keep]


def refine_state_by_geometry(
    class_name: str,
    pts_world: np.ndarray | None,
    cls_state: str,
) -> str:
    """用点云直立程度纠正状态头：工具趴桌=normal；螺栓横躺=fallen。螺母/垫圈保持正常。"""
    if pts_world is None or len(pts_world) < 16:
        return cls_state
    pts = inlier_world_points(pts_world)
    centered = pts - np.median(pts, axis=0)
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    upright = abs(float(axis[2]))
    z_span = float(np.ptp(pts[:, 2]))
    xy_span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    name = class_name or ""
    if name.startswith(("螺母", "垫圈", "轴承", "滚子轴承", "齿轮", "铆钉")):
        return canonicalize_state(class_name, "normal")
    is_fastener = name.startswith("螺钉") or ("螺栓" in name)
    is_tool = name.startswith("机器工具")
    geo = None
    if is_fastener:
        if upright > 0.55 or z_span > 0.7 * max(xy_span, 1e-6):
            geo = "normal"
        elif upright < 0.42 and xy_span > z_span * 1.15:
            geo = "fallen"
    elif is_tool and (z_span < 0.06 or z_span < 0.55 * max(xy_span, 1e-6)):
        geo = "normal"
    if geo is None:
        return cls_state
    return canonicalize_state(class_name, geo)


def estimate_coordinate_pose(
    mask_bin: np.ndarray,
    depth: np.ndarray | None,
    camera: dict,
    bbox: list[int] | None,
    state: str,
) -> tuple[float, float, float, float, float, float, np.ndarray | None]:
    """bbox/mask + 相机标定 → coordinate/pose_6d。"""
    roll, pitch = state_to_rp(state)
    yaw = estimate_yaw_from_mask(mask_bin) if mask_bin is not None and mask_bin.any() else 0.0
    x = y = z = 0.0
    pts = None
    if depth is not None:
        pts = mask_to_world_points(mask_bin, depth, camera, bbox)
        if pts is not None:
            xyz = np.median(pts, axis=0)
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            if abs(z) < 0.08:
                z = 0.0
            pts_use = inlier_world_points(pts)
            aspect = 1.0
            if bbox is not None:
                bw = max(1.0, float(bbox[2] - bbox[0]))
                bh = max(1.0, float(bbox[3] - bbox[1]))
                aspect = max(bw, bh) / min(bw, bh)
            if aspect >= 1.8 and mask_bin is not None and mask_bin.any():
                yaw = yaw_from_mask_long_axis(mask_bin, depth, camera, pts_use)
            else:
                yaw = yaw_from_world_xy(pts_use)
    return x, y, z, roll, pitch, yaw, pts


def resolve_intrinsics(args, sidecar: dict | None, width: int, height: int) -> dict:
    cam = (sidecar or {}).get("camera") or {}
    src = cam.get("intrinsics") or DEFAULT_CAMERA["intrinsics"]
    return {
        "fx": args.fx if args.fx is not None else src.get("fx", DEFAULT_CAMERA["intrinsics"]["fx"]),
        "fy": args.fy if args.fy is not None else src.get("fy", DEFAULT_CAMERA["intrinsics"]["fy"]),
        "cx": args.cx if args.cx is not None else src.get("cx", width / 2.0 - 0.5),
        "cy": args.cy if args.cy is not None else src.get("cy", height / 2.0 - 0.5),
    }


def resolve_camera(sidecar: dict | None, intr: dict) -> dict:
    if sidecar and sidecar.get("camera"):
        cam = dict(sidecar["camera"])
        cam.setdefault("intrinsics", intr)
        return cam
    cam = json.loads(json.dumps(DEFAULT_CAMERA))
    cam["intrinsics"] = intr
    return cam


def resolve_sample_id(image: Path, sample: str) -> str:
    if sample:
        return str(sample).zfill(4) if str(sample).isdigit() else str(sample)
    if image.parent.name.isdigit():
        return image.parent.name.zfill(4)
    stem = image.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits.zfill(4) if digits else stem


def resolve_image(args) -> Path:
    if args.image is not None:
        return args.image
    if not args.sample:
        raise SystemExit("请指定 --sample 0001 或 --image 路径")
    sid = str(args.sample).zfill(4) if str(args.sample).isdigit() else str(args.sample)
    for name in (f"{sid}.jpg", f"{sid}.png", f"{sid}_labeled.png"):
        cand = DATASET_ROOT / sid / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"找不到样本图：{DATASET_ROOT / sid / f'{sid}.jpg'}")


def main():
    args = parse_args()
    args.image = resolve_image(args)
    if not args.image.exists():
        raise FileNotFoundError(f"图片不存在: {args.image}")
    if not args.weights.exists():
        raise FileNotFoundError(f"权重不存在: {args.weights}")

    from ultralytics import YOLO

    img_bgr = imread_unicode(args.image)
    if img_bgr is None:
        raise RuntimeError(f"无法读图: {args.image}")
    height, width = img_bgr.shape[:2]

    sidecar = load_sidecar(args.image)
    if args.depth is None:
        auto_depth = args.image.parent / "depth_m.npy"
        if auto_depth.is_file():
            args.depth = auto_depth

    intr = resolve_intrinsics(args, sidecar, width, height)
    camera = resolve_camera(sidecar, intr)

    depth = None
    if args.depth is not None:
        if args.depth.exists():
            depth = np.load(str(args.depth))
        else:
            print(f"[WARN] 深度图不存在，pose_6d 的 xyz 置 0: {args.depth}", file=sys.stderr)
    if args.save_pcd and depth is None:
        print("[WARN] 无深度图，跳过点云导出。提供 --depth 或同目录 depth_m.npy", file=sys.stderr)

    roots = model_roots(args.models_dir)
    stl_map = build_stl_map(args.models_dir)
    if stl_map:
        print(f"[INFO] STL 查表 {len(stl_map)} 类，根目录 {len(roots)} 个")

    print(f"[INFO] 加载 seg 模型: {args.weights}")
    model = YOLO(str(args.weights))

    state_model = None
    state_head = args.state_head
    if state_head is None and DEFAULT_STATE_HEAD.exists():
        state_head = DEFAULT_STATE_HEAD
    if state_head is not None:
        if state_head.exists():
            print(f"[INFO] 加载状态头: {state_head}")
            state_model = YOLO(str(state_head))
        else:
            print(f"[WARN] 状态头不存在，state=normal: {state_head}", file=sys.stderr)

    sid = resolve_sample_id(args.image, args.sample)
    sample_out = OUTPUTS / sid
    out_path = args.out if args.out is not None else OUTPUTS / f"{sid}.json"
    mask_dir = args.mask_dir if args.mask_dir is not None else sample_out / "masks"
    pcd_dir = args.pcd_dir if args.pcd_dir is not None else sample_out / "pointclouds"
    if args.out is None:
        args.pretty = True

    if args.slice:
        from sliced_predict import sliced_predict

        print("[INFO] 切片推理：全图 1280 + 重叠 tile")
        slice_dets = sliced_predict(
            model,
            img_bgr,
            conf=args.conf,
            iou=args.iou,
            imgsz_full=1280,
            tile=args.slice_tile,
            overlap=args.slice_overlap,
            nms_iou=args.iou,
            with_masks=True,
        )
        det_items = []
        for d in slice_dets:
            mask_bin = d.mask if d.mask is not None else np.zeros((height, width), dtype=np.uint8)
            xyxy = [float(v) for v in d.bbox]
            det_items.append((d.cls_name, float(d.conf), xyxy, mask_bin))
    else:
        results = model.predict(
            img_bgr, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False
        )[0]
        names = results.names or {}
        boxes = results.boxes
        masks = results.masks
        det_items = []
        n_det = 0 if boxes is None else len(boxes)
        for i in range(n_det):
            box = boxes[i]
            cls_name = names.get(int(box.cls), str(int(box.cls)))
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            mask_bin = np.zeros((height, width), dtype=np.uint8)
            if masks is not None and i < len(masks.data):
                mask_bin = mask_to_fullsize(masks.data[i].cpu().numpy(), width, height)
            det_items.append((cls_name, float(box.conf), xyxy, mask_bin))

    use_small = (not args.no_small) and args.weights_small is not None
    if use_small:
        if not args.weights_small.exists():
            raise FileNotFoundError(f"小件权重不存在: {args.weights_small}")
        from sliced_predict import tile_only_predict

        print(f"[INFO] 加载小件权重: {args.weights_small}")
        small_model = YOLO(str(args.weights_small))
        small_dets = tile_only_predict(
            small_model,
            img_bgr,
            conf=args.small_conf,
            iou=args.iou,
            tile=args.slice_tile,
            overlap=args.slice_overlap,
            nms_iou=args.iou,
            with_masks=True,
        )
        extra = merge_small_dets(
            det_items, small_dets, width, height, args.small_max_area, args.small_iou
        )
        print(
            f"[INFO] 小件补检：切片 {len(small_dets)} 个，"
            f"采纳 {len(extra)} 个（不覆盖大模型已有框）"
        )
        det_items.extend(extra)

    objects = []
    for i, (cls_name, score, xyxy, mask_bin) in enumerate(det_items):
        bbox = bbox_from_xyxy(xyxy, width, height)

        polygon: list[list[int]] = []
        if mask_bin is not None and mask_bin.any():
            polygon = mask_to_polygon(mask_bin)
        if len(polygon) < 3 and bbox is not None:
            x1, y1, x2, y2 = bbox
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        visible_pixels = int(mask_bin.sum()) if mask_bin.any() else max(
            0, (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        )

        mask_path = ""
        if args.save_masks:
            mask_file = mask_dir / f"{args.image.stem}_{i + 1:02d}.png"
            imwrite_unicode(mask_file, (mask_bin * 255).astype(np.uint8))
            mask_path = f"masks/{mask_file.name}"

        state = "normal"
        if state_model is not None:
            crop = crop_xyxy(img_bgr, xyxy)
            if crop.size > 0:
                state = canonicalize_state(cls_name, predict_state(state_model, crop))

        x, y, z, roll, pitch, yaw, pts_w = estimate_coordinate_pose(
            mask_bin, depth, camera, bbox, state
        )
        state = refine_state_by_geometry(cls_name, pts_w, state)
        roll, pitch = state_to_rp(state)

        stl_path = stl_map.get(cls_name)
        hit = match_sidecar_object(bbox, cls_name, sidecar)
        if hit:
            loc = localize_stl_path(hit.get("stl_path"), roots)
            if loc:
                stl_path = loc

        pcd_rel = ""
        if args.save_pcd and depth is not None:
            cloud, colors = mask_to_world_cloud(
                mask_bin, depth, camera, img_bgr, bbox, stride=args.pcd_stride
            )
            if cloud is not None:
                pcd_file = pcd_dir / f"{args.image.stem}_{i + 1:02d}.ply"
                save_pointcloud_ply(pcd_file, cloud, colors)
                pcd_rel = f"pointclouds/{pcd_file.name}"

        objects.append(
            make_object(
                obj_id=i + 1,
                class_name=cls_name,
                bbox=bbox,
                polygon=polygon,
                mask_path=mask_path,
                pose_6d=make_pose_6d(x, y, z, roll, pitch, yaw),
                state=state,
                visible_pixels=visible_pixels,
                stl_path=stl_path,
                score=float(score),
                pointcloud_path=pcd_rel or None,
            )
        )

    output = make_annotations(
        image=args.image.name,
        width=width,
        height=height,
        objects=objects,
        camera=camera,
        sidecar=sidecar,
    )
    payload = json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    print(f"[DONE] 检测到 {len(objects)} 个物体 → {out_path}")
    for o in objects:
        pose = o["pose_6d"]
        print(
            f"  [{o['id']}] {o['class']:24s} state={o['state']:9s} "
            f"bbox={o['bbox']} pixels={o['visible_pixels']:6d} "
            f"coord=({pose['x']:.3f},{pose['y']:.3f}) "
            f"xyz=({pose['x']:.3f},{pose['y']:.3f},{pose['z']:.3f}) yaw={pose['yaw']:.3f} "
            f"stl={'ok' if o.get('stl_path') else 'null'} "
            f"pcd={o.get('pointcloud_path') or '-'}"
        )


if __name__ == "__main__":
    main()
