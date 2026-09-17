# -*- coding: utf-8 -*-
"""眼在手上：D405 + YOLO26s-seg（只香蕉/苹果/橙子）→ 世界系位姿。

世界点 = T_base_flange(q) @ T_flange_cam @ p_camera
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np

from fruit_config import (
    DEMO_DIR,
    FRUIT_CLASS_IDS,
    FRUIT_EN,
    FRUIT_RADIUS_BOUNDS_M,
    FRUIT_ZH,
    HANDEYE_JSON,
    MAX_DEPTH_M,
    MIN_POINTS_PER_OBJECT,
    REPO_ROOT,
    SURFACE_TO_CENTER_K,
    VISION_DIR,
    WORKSPACE_BOUNDS_M,
    WORLD_TRIM_M,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
    YOLO_WEIGHTS,
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_CN_FONTS: dict[int, object] = {}
_YOLO = None
_YOLO_PATH = None
_YOLO_LOCK = threading.Lock()
_YOLO_WARMED = False
_YOLO_WARM_THREAD = None


def load_yolo(weights=None):
    """进程内只加载一次权重。每次 detect 再 `YOLO()` 会把 0.1s 推理拖成几秒。"""
    global _YOLO, _YOLO_PATH
    weights = os.path.abspath(str(weights or YOLO_WEIGHTS))
    with _YOLO_LOCK:
        if _YOLO is None or _YOLO_PATH != weights:
            from ultralytics import YOLO

            if not os.path.isfile(weights):
                raise FruitVisionError(f"找不到水果分割权重：{weights}")
            t0 = time.perf_counter()
            _YOLO = YOLO(weights)
            _YOLO_PATH = weights
            print(f"[vision] YOLO 加载 {os.path.basename(weights)} {time.perf_counter() - t0:.2f}s")
        return _YOLO


def _yolo_predict(model, source, *, conf, iou, imgsz):
    """和 ``fruit_yolo_realtime.live_yolo_preview`` 同一套调用。

    实时窗口用 ``conf=min(阈值, 0.08)`` 做 NMS，画面上橙子能到 0.6+；
    抓拍若直接 ``conf=0.35``，同一张图经常 0 框。
    """
    floor = 0.08
    color = np.ascontiguousarray(source)
    with _YOLO_LOCK:
        return model.predict(
            color,
            conf=min(float(conf), floor),
            iou=float(iou),
            imgsz=int(imgsz),
            classes=list(FRUIT_CLASS_IDS),
            verbose=False,
        )[0]


def warmup_yolo(logger=None):
    """只加载权重，不要拿黑图先跑一遍。YOLO26 会把第一次推理的输入形状缓存下来。"""
    global _YOLO_WARMED
    log = logger or print
    if _YOLO_WARMED:
        return load_yolo()
    try:
        t0 = time.perf_counter()
        model = load_yolo()
        _YOLO_WARMED = True
        log(f"[vision] YOLO 权重已就绪 {time.perf_counter() - t0:.2f}s  imgsz={YOLO_IMGSZ}")
        return model
    except Exception as e:
        log(f"[vision] YOLO 加载失败：{e}")
        return None


def warmup_yolo_async(logger=None):
    global _YOLO_WARM_THREAD
    if _YOLO_WARMED:
        return None
    if _YOLO_WARM_THREAD is not None and _YOLO_WARM_THREAD.is_alive():
        return _YOLO_WARM_THREAD
    t = threading.Thread(
        target=warmup_yolo,
        kwargs={"logger": logger},
        daemon=True,
        name="yolo-warmup",
    )
    _YOLO_WARM_THREAD = t
    t.start()
    if logger:
        logger("[vision] 正在后台加载 YOLO，和去观察位重叠")
    return t


def new_sample_id(prefix: str = "fruit") -> str:
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"


def _cn_font(size: int = 18):
    if size in _CN_FONTS:
        return _CN_FONTS[size]
    from PIL import ImageFont

    font = None
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if os.path.isfile(path):
            font = ImageFont.truetype(path, size=size)
            break
    if font is None:
        font = ImageFont.load_default()
    _CN_FONTS[size] = font
    return font


def put_labels_cn(img_bgr: np.ndarray, items, *, size: int = 18) -> np.ndarray:
    """cv2.putText 没有中文字形，UTF-8 会被画成问号。这里用系统字体画。

    items: (x, y, text, color_bgr)
    """
    import cv2
    from PIL import Image, ImageDraw

    if not items:
        return img_bgr
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _cn_font(size)
    for x, y, text, color_bgr in items:
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text((int(x), int(y)), str(text), fill=color_rgb, font=font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class FruitVisionError(RuntimeError):
    pass


def load_handeye(path: str | None = None) -> np.ndarray:
    path = path or HANDEYE_JSON
    if not os.path.isfile(path):
        raise FruitVisionError(
            f"还没有眼在手上标定文件：{path}\n"
            "标定完成后再把 4x4 的 affine_mat（法兰→相机）存成这个 JSON。"
            "格式示例：{{\"mode\": \"eye_in_hand\", \"affine_mat\": [[...],[...],[...],[...]]}}"
        )
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    mat = np.asarray(data.get("affine_mat"), dtype=float)
    if mat.shape != (4, 4):
        raise FruitVisionError(f"{path} 的 affine_mat 不是 4x4")
    if np.allclose(mat, np.eye(4)):
        raise FruitVisionError(
            f"{path} 里还是单位阵，不能当标定用。请换成现场标定得到的法兰→相机矩阵。"
        )
    return mat


def load_handeye_raw(path: str | None = None) -> np.ndarray:
    """标定脚本用：文件不存在或单位阵都返回一个 4x4，不报错。"""
    path = path or HANDEYE_JSON
    if not os.path.isfile(path):
        return np.eye(4)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    mat = np.asarray(data.get("affine_mat"), dtype=float)
    if mat.shape != (4, 4):
        raise FruitVisionError(f"{path} 的 affine_mat 不是 4x4")
    return mat


def save_handeye(mat, path: str | None = None) -> str:
    path = path or HANDEYE_JSON
    mat = np.asarray(mat, dtype=float)
    if mat.shape != (4, 4):
        raise FruitVisionError(f"affine_mat 必须是 4x4，收到 {mat.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "eye_in_hand",
                "affine_mat": mat.tolist(),
                "note": "T_flange_cam 法兰→相机",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return os.path.abspath(path)


def describe_handeye(mat) -> str:
    mat = np.asarray(mat, dtype=float)
    pos = mat[:3, 3]
    return (
        f"法兰→相机 平移 [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]m"
        + ("  （还是单位阵，未标定）" if np.allclose(mat, np.eye(4)) else "")
    )


_FK_ROBOT = None


def flange_pose(q) -> tuple[np.ndarray, np.ndarray]:
    """当前关节 → 法兰位姿。用 WRS 仿真臂 FK，和规划器同一套运动学。"""
    global _FK_ROBOT
    from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

    if _FK_ROBOT is None:
        _FK_ROBOT = PantheraHTSglArm(enable_cc=False)
    q = np.asarray(q, dtype=float).reshape(-1)
    _FK_ROBOT.fk(q, update=True)
    pos = np.asarray(_FK_ROBOT.manipulator.gl_flange_pos, dtype=float).reshape(3)
    rot = np.asarray(_FK_ROBOT.manipulator.gl_flange_rotmat, dtype=float).reshape(3, 3)
    return pos, rot


def flange_homomat(q) -> np.ndarray:
    from wrs.basis.robot_math import homomat_from_posrot

    pos, rot = flange_pose(q)
    return homomat_from_posrot(pos, rot)


def joints_deg(q) -> tuple[float, ...]:
    q = np.asarray(q, dtype=float).reshape(-1)
    return tuple(round(float(np.degrees(v)), 2) for v in q)


def format_conf_deg(q) -> str:
    return "(" + ", ".join(f"{v:.2f}" for v in joints_deg(q)) + ")"


def describe_arm_state(q, *, r2cam=None) -> str:
    """关节角 + 法兰世界坐标（可选再算相机位姿）。"""
    from wrs.basis.robot_math import rotmat_to_euler

    q = np.asarray(q, dtype=float).reshape(-1)
    pos, rot = flange_pose(q)
    rpy = np.degrees(rotmat_to_euler(rot))
    lines = [
        f"关节(°) {format_conf_deg(q)}",
        f"可粘贴  LOOK_CONF_DEG = {format_conf_deg(q)}",
        f"法兰 xyz [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m",
        f"法兰 rpy [{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}] °",
    ]
    if r2cam is not None:
        from wrs.basis.robot_math import homomat_from_posrot

        w2c = homomat_from_posrot(pos, rot) @ np.asarray(r2cam, dtype=float)
        cpos = w2c[:3, 3]
        crpy = np.degrees(rotmat_to_euler(w2c[:3, :3]))
        lines.append(f"相机 xyz [{cpos[0]:+.4f}, {cpos[1]:+.4f}, {cpos[2]:+.4f}] m")
        lines.append(f"相机 rpy [{crpy[0]:+.2f}, {crpy[1]:+.2f}, {crpy[2]:+.2f}] °")
    return "\n".join(lines)


def camera_homomat(q, r2cam: np.ndarray) -> np.ndarray:
    """相机在世界系的位姿。第三列就是视线方向（相机 +Z 朝外看）。"""
    from wrs.basis.robot_math import homomat_from_posrot

    pos, rot = flange_pose(q)
    w2r = homomat_from_posrot(pos, rot)
    return w2r @ np.asarray(r2cam, dtype=float)


def world_from_camera(q, p_cam: np.ndarray, r2cam: np.ndarray) -> np.ndarray:
    """相机系点 → 世界系。p_cam Nx3。"""
    from wrs.basis.robot_math import transform_points_by_homomat

    w2c = camera_homomat(q, r2cam)
    pts = np.asarray(p_cam, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return transform_points_by_homomat(w2c, pts)


def silhouette_radius(world: np.ndarray, view: np.ndarray) -> float:
    """从垂直于视线的轮廓宽度反推件的半径。

    取两个横向跨度里小的那个的一半：球形件两个方向一样；香蕉这种长条，
    小的那个跨度才接近它在视线方向上的厚度。
    """
    v = np.asarray(view, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12)
    u = np.cross(v, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(v, [1.0, 0.0, 0.0])
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(v, u)
    spans = []
    for ax in (u, w):
        d = world @ ax
        spans.append(float(np.percentile(d, 95) - np.percentile(d, 5)))
    lo, hi = FRUIT_RADIUS_BOUNDS_M
    return float(np.clip(min(spans) / 2.0, lo, hi))


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter
    )
    return inter / max(area, 1e-9)


def _nms_objects(objects: list, iou: float = 0.45) -> list:
    ordered = sorted(objects, key=lambda o: float(o.get("score") or 0), reverse=True)
    keep = []
    for o in ordered:
        bb = o.get("bbox") or [0, 0, 0, 0]
        if any(_bbox_iou(bb, k.get("bbox") or [0, 0, 0, 0]) >= iou for k in keep):
            continue
        keep.append(o)
    for i, o in enumerate(keep, 1):
        o["id"] = i
    return keep


def _vivid_hues(color_bgr, mask):
    import cv2

    if mask is None or not np.any(mask):
        return None
    hsv = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2HSV)
    pix = hsv[mask]
    if len(pix) < 30:
        return None
    s = pix[:, 1].astype(float)
    v = pix[:, 2].astype(float)
    vivid = (s >= 60) & (v >= 50)
    if int(vivid.sum()) < 20:
        return None
    return pix[vivid, 0].astype(float)


def _color_vote(color_bgr, mask) -> str | None:
    """红/橙像素投票。中间色相（8–12）不算任何一边，避免红苹果被中位数判成橙。"""
    h = _vivid_hues(color_bgr, mask)
    if h is None:
        return None
    n = float(len(h))
    n_red = float(np.sum((h <= 8.0) | (h >= 160.0)))
    n_orange = float(np.sum((h >= 12.0) & (h <= 22.0)))
    n_green = float(np.sum((h >= 35.0) & (h <= 90.0)))
    if n_red >= 0.28 * n and n_red >= n_orange * 1.15:
        return "apple"
    if n_green >= 0.35 * n and n_green >= n_orange:
        return "apple"
    if n_orange >= 0.40 * n and n_orange >= n_red * 1.8:
        return "orange"
    return None


def _reclass_fruit(en: str, color_bgr, mask, bbox, score: float = 1.0) -> str:
    if en == "banana":
        x1, y1, x2, y2 = bbox
        ar = max(x2 - x1, y2 - y1) / max(1, min(x2 - x1, y2 - y1))
        if ar >= 1.55:
            return en
    vote = _color_vote(color_bgr, mask)
    if vote is None:
        return en
    # 只纠正低置信误检：橙子被 YOLO 标成 0.1 的苹果。
    # 0.45+ 的苹果框信 YOLO，否则暖光下红苹果中位色相会落进橙区间。
    if en == "apple" and vote == "orange" and float(score) < 0.28:
        return "orange"
    if en == "orange" and vote == "apple":
        return "apple"
    return en


def in_workspace(pts: np.ndarray) -> np.ndarray:
    b = WORKSPACE_BOUNDS_M
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return (
        (x >= b["x"][0])
        & (x <= b["x"][1])
        & (y >= b["y"][0])
        & (y <= b["y"][1])
        & (z >= b["z"][0])
        & (z <= b["z"][1])
    )


def detect_fruits(color, points_cam, q, *, weights=None, conf=None, iou=None, imgsz=None, r2cam=None):
    """返回 annotations 风格 dict。"""
    weights = weights or YOLO_WEIGHTS
    if not os.path.isfile(weights):
        raise FruitVisionError(f"找不到水果分割权重：{weights}")
    t = _YOLO_WARM_THREAD
    if t is not None and t.is_alive():
        t.join(timeout=60)
    r2cam = load_handeye() if r2cam is None else np.asarray(r2cam, dtype=float)
    view = camera_homomat(q, r2cam)[:3, 2]
    model = load_yolo(weights)
    imgsz = int(imgsz if imgsz is not None else YOLO_IMGSZ)
    t0 = time.perf_counter()
    result = _yolo_predict(
        model,
        color,
        conf=conf if conf is not None else YOLO_CONF,
        iou=iou if iou is not None else YOLO_IOU,
        imgsz=imgsz,
    )
    print(f"[vision] YOLO {time.perf_counter() - t0:.3f}s  imgsz={imgsz}")
    names = result.names or {}
    boxes = result.boxes
    masks = result.masks
    h, w = color.shape[:2]
    n_raw = 0 if boxes is None else len(boxes)
    luma = float(np.mean(color)) if color.size else 0.0
    if n_raw == 0:
        print(
            f"[vision] YOLO 一个框都没有  luma={luma:.1f}  "
            f"{int(color.shape[1])}x{int(color.shape[0])}"
        )
    else:
        hits = []
        for i in range(n_raw):
            en = str(names.get(int(boxes[i].cls), int(boxes[i].cls)))
            hits.append(f"{en}:{float(boxes[i].conf):.2f}")
        print(f"[vision] YOLO 原始框 {n_raw} 个  {', '.join(hits)}")
    objects = []
    n = 0 if boxes is None else len(boxes)
    for i in range(n):
        en = str(names.get(int(boxes[i].cls), int(boxes[i].cls)))
        if en not in FRUIT_EN:
            continue
        score = float(boxes[i].conf)
        x1, y1, x2, y2 = (float(v) for v in boxes[i].xyxy[0].cpu().numpy())
        bbox = [
            int(np.clip(np.floor(x1), 0, w - 1)),
            int(np.clip(np.floor(y1), 0, h - 1)),
            int(np.clip(np.ceil(x2), 0, w - 1)),
            int(np.clip(np.ceil(y2), 0, h - 1)),
        ]
        if masks is not None and i < len(masks.data):
            import cv2

            m = masks.data[i].cpu().numpy().astype(np.float32)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = m > 0.5
        else:
            mask = np.zeros((h, w), dtype=bool)
            mask[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1] = True
        yolo_en = en
        en = _reclass_fruit(en, color, mask, bbox, score)
        if en != yolo_en:
            print(
                f"[vision] 颜色纠偏 {FRUIT_ZH.get(yolo_en, yolo_en)}→"
                f"{FRUIT_ZH.get(en, en)}  YOLO={score:.2f}"
            )
        min_keep = float(conf if conf is not None else YOLO_CONF)
        # 低分苹果被颜色改成橙子时才放行，其它低分框丢掉。
        if en != yolo_en and en == "orange" and yolo_en == "apple":
            min_keep = 0.08
        if score < min_keep:
            continue
        cam_pts = points_cam[mask]
        cam_pts = cam_pts[np.linalg.norm(cam_pts, axis=1) > 1e-6]
        cam_pts = cam_pts[cam_pts[:, 2] < MAX_DEPTH_M]
        if len(cam_pts) < MIN_POINTS_PER_OBJECT:
            print(f"[vision] 跳过 {en}：有效点 {len(cam_pts)}")
            continue
        world = world_from_camera(q, cam_pts, r2cam)
        keep = in_workspace(world)
        world = world[keep]
        if len(world) < MIN_POINTS_PER_OBJECT:
            print(f"[vision] 跳过 {en}：工作区内点 {len(world)}")
            continue
        surf = np.median(world, axis=0)
        ys = world[:, 1] - surf[1]
        xs = world[:, 0] - surf[0]
        yaw = float(np.arctan2(np.sum(xs * ys), np.sum(xs * xs) - np.sum(ys * ys) + 1e-9)) / 2.0
        # 点云只有朝着相机那半边表面，中位数落在那半边上，不是水果中心。
        # 这台相机在 LOOK 位是斜着往前看的（偏离竖直 60° 左右），所以这个偏差
        # 大头在水平方向：不补就每次都抓在水果靠基座那一侧。沿视线往前推回中心。
        radius = silhouette_radius(world, view)
        shift = SURFACE_TO_CENTER_K * radius * view
        center = surf + np.array([shift[0], shift[1], 0.0])
        # 高度不跟着推：顶面是直接看得见的量，规划器从顶面往下扎就行。
        # 95 分位容易被深度飞点抬飞（同一桌橙子顶面比苹果高 4cm），
        # 80 / 20 分位稳一些；规划器还会再用中位数 + 半径封顶。
        z_top = float(np.percentile(world[:, 2], 80))
        z_low = float(np.percentile(world[:, 2], 20))
        center += np.asarray(WORLD_TRIM_M, dtype=float)
        z_top += float(WORLD_TRIM_M[2])
        z_low += float(WORLD_TRIM_M[2])
        print(
            f"[vision] {en} 半径≈{radius * 100:.1f}cm，"
            f"视线补偿 {np.linalg.norm(shift[:2]) * 100:.1f}cm → "
            f"({center[0]:.3f}, {center[1]:.3f})"
        )
        poly = []
        try:
            import cv2

            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                poly = cnts[0].reshape(-1, 2).astype(int).tolist()
        except Exception:
            poly = []
        objects.append(
            {
                "id": len(objects) + 1,
                "class": FRUIT_ZH.get(en, en),
                "yolo_class": en,
                "score": score,
                "bbox": bbox,
                "polygon": poly,
                "state": "normal",
                "z_top": z_top,
                "z_low": z_low,
                "radius": radius,
                "pose_6d": {
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": yaw,
                },
            }
        )
    objects = _nms_objects(objects)
    return {
        "width": int(w),
        "height": int(h),
        "objects": objects,
        "camera": "eye_in_hand",
    }


def draw_detect(color, ann: dict):
    """在**这一张**抓拍上画框，不要叠到后续实时流。"""
    import cv2

    img = np.asarray(color).copy()
    labels = []
    for obj in ann.get("objects") or []:
        poly = obj.get("polygon") or []
        if len(poly) >= 3:
            cv2.polylines(img, [np.asarray(poly, dtype=np.int32)], True, (0, 255, 0), 2)
        p = obj.get("pose_6d") or {}
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        labels.append(
            (
                int(bbox[0]),
                max(4, int(bbox[1]) - 22),
                f"{obj.get('class')} {obj.get('score', 0):.2f} "
                f"({p.get('x', 0):+.3f},{p.get('y', 0):+.3f},{p.get('z', 0):+.3f})",
                (0, 255, 255),
            )
        )
    return put_labels_cn(img, labels)


def save_annotations(ann: dict, sample_id: str | None = None, frame=None, preview=None) -> str:
    sample_id = sample_id or new_sample_id()
    os.makedirs(os.path.join(VISION_DIR, sample_id), exist_ok=True)
    path = os.path.join(VISION_DIR, sample_id, "annotations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    img = preview
    if img is None and frame is not None:
        try:
            img = draw_detect(frame.color if hasattr(frame, "color") else frame, ann)
        except Exception as e:
            print(f"[vision] 预览失败：{e}")
            img = None
    if img is not None:
        try:
            import cv2

            prev = os.path.join(VISION_DIR, sample_id, "detect_preview.jpg")
            cv2.imwrite(prev, img)
            print(f"[vision] 预览 {prev}")
        except Exception as e:
            print(f"[vision] 预览失败：{e}")
    print(f"[vision] annotations -> {path}  n={len(ann.get('objects') or [])}")
    return path


def capture_and_detect(
    q,
    *,
    sample_id=None,
    resolution=None,
    serial=None,
    frames=None,
    grab=None,
):
    from fruit_config import CAMERA_RESOLUTION, REAL_DIR, SNAPSHOT_FRAMES

    sample_id = sample_id or new_sample_id()
    frames = SNAPSHOT_FRAMES if frames is None else int(frames)
    r2cam = load_handeye()
    if grab is not None:
        frame = grab()
    else:
        if REAL_DIR not in sys.path:
            sys.path.insert(0, REAL_DIR)
        from d405_camera import D405Camera

        with D405Camera(resolution or CAMERA_RESOLUTION, serial, logger=print) as cam:
            frame = cam.capture_median(max(1, frames)) if hasattr(cam, "capture_median") else cam.capture()
    print(f"[vision] 新抓拍 {sample_id}  {int(frame.color.shape[1])}x{int(frame.color.shape[0])}")
    ann = detect_fruits(frame.color, frame.points, q, r2cam=r2cam)
    preview = draw_detect(frame.color, ann)
    save_annotations(ann, sample_id=sample_id, preview=preview)
    return ann, frame, preview
