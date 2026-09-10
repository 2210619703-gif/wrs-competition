# -*- coding: utf-8 -*-
"""YOLO + D405 + 手眼矩阵 → 仿真格式 ``annotations.json``。

这一层做的事和 ``competition/yolo_utils.detect_blocks_world`` 是同一套思路：

    mask（彩色图）→ 取出对应的相机系 3D 点 → 手眼变到世界系
    → 中位数当中心、XY 主轴当 yaw → 写成下游能读的 JSON

区别是输出格式对齐 ``tiaozhanbei/dataset_learn/*/annotations.json``，
这样仿真规划器可以直接把它当 ``--dataset-root`` 用，不需要改仿真侧代码。

单独用（先存一帧，再离线反复调）::

    python tiaozhanbei/real/d405_camera.py --save-dir tiaozhanbei/real/outputs/frame01
    python tiaozhanbei/real/vision_to_annotations.py --frame-dir tiaozhanbei/real/outputs/frame01

直接连相机::

    python tiaozhanbei/real/vision_to_annotations.py --sample-id real01
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from real_config import REAL_DIR
from vision_config import (
    CAMERA_RESOLUTION,
    CAMERA_SERIAL,
    DESK_Z,
    MIN_POINTS_PER_OBJECT,
    STATE_FALLEN_FLATNESS,
    STATE_MIN_Z_SPAN_M,
    WORKSPACE_BOUNDS_M,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
    MIN_OBJECT_SEP_M,
    YOLO_RECALL_CONF,
    YOLO_RECALL_IMGSZ,
    YOLO_RECALL_MAX_IOU,
    YOLO_RECALL_MIN_CENTER_PX,
    YOLO_VOTE_FRAMES,
    YOLO_VOTE_IOU,
    YOLO_VOTE_MIN,
    ClassMap,
    VisionConfigError,
    default_weights,
    describe_handeye,
    load_handeye,
    points_in_workspace,
    transform_points,
    valid_points,
)

VISION_DIR = os.path.join(REAL_DIR, "outputs", "vision")
DEFAULT_SAMPLE_ID = "real01"


class VisionError(RuntimeError):
    """识别或坐标转换失败。"""


# ---------------------------------------------------------------------------
# 几何估计
# ---------------------------------------------------------------------------
def inlier_points(pts: np.ndarray, band_m: float = 0.04) -> np.ndarray:
    """去掉深度飞点，只留中位高度附近的一层。"""
    if len(pts) < 16:
        return pts
    z = pts[:, 2]
    keep = np.abs(z - float(np.median(z))) < band_m
    return pts[keep] if int(keep.sum()) >= 16 else pts


def yaw_from_world_xy(pts: np.ndarray) -> float:
    """世界系 XY 主轴 → yaw（弧度）。

    细长件（螺丝刀、扳手、螺栓）的朝向靠这个定；近似圆形件算出来接近随机，
    但那类零件本身就没有明确朝向，不影响抓取。
    """
    xy = pts[:, :2] - np.median(pts[:, :2], axis=0)
    if len(xy) < 8 or float(np.linalg.norm(xy.std(axis=0))) < 1e-6:
        return 0.0
    cov = (xy.T @ xy) / max(len(xy) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    d = evecs[:, int(np.argmax(evals))]
    if d[0] < 0:
        d = -d
    return float(np.arctan2(d[1], d[0]))


# 钳子/扳手/螺丝刀这类件平时就是平躺在桌上。几何上「扁」不等于侧翻：
# 旧逻辑把扁的点云标成 fallen，仿真再 roll=90°，看起来就像侧翻 90°。
_FLAT_TOOL_MARKERS = (
    "机器工具",
    "钳子",
    "扳手",
    "螺丝刀",
    "锤子",
    "夹具",
    "剪刀",
)
_ALWAYS_FLAT_MARKERS = ("螺母", "垫圈", "轴承", "滚子轴承", "齿轮", "铆钉")


def estimate_state(zh_class: str, pts: np.ndarray) -> str:
    """按世界系点云的形状判断 normal / fallen / inverted。

    真机这边没有状态分类头（只有分割权重），所以完全靠几何：
    螺母、垫圈、轴承、齿轮、手工工具平躺都是 normal；
    只有螺丝/销钉这类「立着才是正放」的紧固件，横躺才标 fallen。

    判不出 inverted：单目俯视看不出螺丝是正插还是倒插。
    """
    name = zh_class or ""
    if name.startswith(_ALWAYS_FLAT_MARKERS) or name.startswith(_FLAT_TOOL_MARKERS):
        return "normal"
    if any(key in name for key in _FLAT_TOOL_MARKERS):
        return "normal"
    if len(pts) < 16:
        return "normal"
    p = inlier_points(pts)
    z_span = float(np.ptp(p[:, 2]))
    xy_span = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
    if z_span < STATE_MIN_Z_SPAN_M:
        return "normal"
    if z_span < STATE_FALLEN_FLATNESS * max(xy_span, 1e-6):
        return "fallen"
    return "normal"


def state_to_rp(state: str) -> tuple[float, float]:
    """状态 → roll/pitch，规则和 tiaozhanbei/yolo 那边保持一致。"""
    if state == "fallen":
        return float(np.pi / 2.0), 0.0
    if state == "inverted":
        return float(np.pi), 0.0
    return 0.0, 0.0


def world_points_for_mask(mask_u8: np.ndarray, points_hw3: np.ndarray, w2c_mat) -> np.ndarray:
    """mask 覆盖的像素 → 世界系有效点（已过滤无效深度和工作区外）。"""
    ys, xs = np.nonzero(mask_u8)
    if not len(xs):
        return np.empty((0, 3), dtype=float)
    cam_pts = valid_points(points_hw3[ys, xs])
    if not len(cam_pts):
        return cam_pts
    return points_in_workspace(transform_points(w2c_mat, cam_pts))


def pose_from_world_points(zh_class: str, world: np.ndarray) -> dict:
    """世界系点云 → ``pose_6d`` + ``state``。

    抽出来单独一个函数，是为了能用合成数据精确校验：这段算错了，机械臂就会
    伸到错的地方，而看日志是看不出来的（数字都很"合理"）。
    """
    pts = inlier_points(world)
    cx_m, cy_m = (float(v) for v in np.median(pts[:, :2], axis=0))
    top_z = float(np.quantile(pts[:, 2], 0.90))
    state = estimate_state(zh_class, world)
    roll, pitch = state_to_rp(state)
    return {
        "pose_6d": {
            "x": cx_m,
            "y": cy_m,
            # 仿真会把零件坐到桌面 z=0，这里的 z 只作为核对用
            "z": float(max(0.0, top_z - DESK_Z)),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw_from_world_xy(pts),
        },
        "state": state,
    }


def mask_to_polygon(mask_u8: np.ndarray, epsilon_ratio: float = 0.01) -> list[list[int]]:
    import cv2

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * cv2.arcLength(cnt, True), True)
    if len(approx) < 3:
        approx = cnt
    return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(
        0.0, by2 - by1
    )
    union = area - inter
    return inter / union if union > 0.0 else 0.0


def vote_detections(per_frame, *, iou_thr: float, min_votes: int) -> list[dict]:
    """跨帧按类名 + IoU 聚类，票数够的留下最高分那条。"""
    clusters: list[list[dict]] = []
    for dets in per_frame:
        for det in dets:
            best_i, best_iou = -1, float(iou_thr)
            for ci, cl in enumerate(clusters):
                ref = max(cl, key=lambda d: d["score"])
                if ref["en_class"] != det["en_class"]:
                    continue
                iou = box_iou(ref["bbox_xyxy"], det["bbox_xyxy"])
                if iou > best_iou:
                    best_i, best_iou = ci, iou
            if best_i >= 0:
                clusters[best_i].append(det)
            else:
                clusters.append([det])
    winners = []
    for cl in clusters:
        if len(cl) < int(min_votes):
            continue
        best = dict(max(cl, key=lambda d: d["score"]))
        best["votes"] = len(cl)
        winners.append(best)
    return winners


def _predict_raw(model, color, *, names, conf, iou, imgsz) -> list[dict]:
    import cv2

    result = model.predict(color, conf=conf, iou=iou, imgsz=int(imgsz), verbose=False)[0]
    boxes = result.boxes
    masks = result.masks
    h, w = color.shape[:2]
    dets: list[dict] = []
    n_det = 0 if boxes is None else len(boxes)
    for i in range(n_det):
        box = boxes[i]
        en_class = str(names.get(int(box.cls), int(box.cls)))
        score = float(box.conf)
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
        bbox_xyxy = [x1, y1, x2, y2]
        bbox = [
            int(np.clip(np.floor(x1), 0, w - 1)),
            int(np.clip(np.floor(y1), 0, h - 1)),
            int(np.clip(np.ceil(x2), 0, w - 1)),
            int(np.clip(np.ceil(y2), 0, h - 1)),
        ]
        if masks is not None and i < len(masks.data):
            m = masks.data[i].cpu().numpy().astype(np.float32)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_u8 = (m > 0.5).astype(np.uint8)
        else:
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            mask_u8[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1] = 1
        dets.append(
            {
                "en_class": en_class,
                "score": score,
                "bbox": bbox,
                "bbox_xyxy": bbox_xyxy,
                "mask": mask_u8,
            }
        )
    return dets


def _objects_from_dets(dets, frame, w2c_mat, cmap, skipped) -> list[dict]:
    h, w = frame.color.shape[:2]
    points = frame.points.reshape(h, w, 3)
    objects: list[dict] = []
    for det in dets:
        en_class = det["en_class"]
        part = cmap.resolve(en_class)
        if part is None:
            skipped.append(f"{en_class}(类名表里没有或 STL 缺失)")
            continue
        mask_u8 = det["mask"]
        if mask_u8.shape[:2] != (h, w):
            import cv2

            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_u8 = (mask_u8 > 0).astype(np.uint8)
        if not mask_u8.any():
            skipped.append(f"{en_class}(mask 为空)")
            continue
        world = world_points_for_mask(mask_u8, points, w2c_mat)
        if len(world) < MIN_POINTS_PER_OBJECT:
            skipped.append(
                f"{en_class}(工作区内有效点只有 {len(world)}，"
                "多半是深度无效或手眼没标准)"
            )
            continue
        pose = pose_from_world_points(part["zh"], world)
        objects.append(
            {
                "id": len(objects) + 1,
                "class": part["zh"],
                "bbox": det["bbox"],
                "polygon": mask_to_polygon(mask_u8),
                "mask_path": "",
                "pose_6d": pose["pose_6d"],
                "visible_pixels": int(mask_u8.sum()),
                "stl_path": part["stl_path"],
                "state": pose["state"],
                "score": float(det["score"]),
                "yolo_class": en_class,
                "world_points": int(len(world)),
                "stl_is_approx": bool(part["approx"]),
                "votes": int(det.get("votes", 1)),
            }
        )
    return objects


def _bbox_center(box) -> tuple[float, float]:
    return (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))


def _center_px(a, b) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(math.hypot(ax - bx, ay - by))


def nms_world_objects(objects, skipped, *, sep_m: float = MIN_OBJECT_SEP_M) -> list[dict]:
    """世界系中心太近的只留置信度高的，避免仿真把两件叠在同一点。"""
    ordered = sorted(objects, key=lambda o: float(o.get("score") or 0.0), reverse=True)
    kept: list[dict] = []
    for obj in ordered:
        p = obj["pose_6d"]
        xy = (float(p["x"]), float(p["y"]))
        clash = None
        for other in kept:
            q = other["pose_6d"]
            d = math.hypot(xy[0] - float(q["x"]), xy[1] - float(q["y"]))
            if d < sep_m:
                clash = other
                break
        if clash is not None:
            skipped.append(
                f"{obj['yolo_class']}(和 {clash['yolo_class']} 世界距 "
                f"{math.hypot(xy[0] - float(clash['pose_6d']['x']), xy[1] - float(clash['pose_6d']['y']))*100:.1f}cm，"
                "当成叠在同一件上丢掉)"
            )
            continue
        kept.append(obj)
    for i, obj in enumerate(kept, 1):
        obj["id"] = i
    return kept


def _merge_recall(model, frame, names, iou, existing, vote_iou, logger) -> list[dict]:
    """投票/主阈值漏掉的黑件，用更大输入再补一轮。"""
    recall = _predict_raw(
        model,
        frame.color,
        names=names,
        conf=YOLO_RECALL_CONF,
        iou=iou,
        imgsz=YOLO_RECALL_IMGSZ,
    )
    added = []
    out = list(existing)
    for det in recall:
        too_close = False
        for w in out:
            if box_iou(det["bbox_xyxy"], w["bbox_xyxy"]) >= YOLO_RECALL_MAX_IOU:
                too_close = True
                break
            if _center_px(det["bbox_xyxy"], w["bbox_xyxy"]) < YOLO_RECALL_MIN_CENTER_PX:
                too_close = True
                break
        if too_close:
            continue
        det = dict(det)
        det["votes"] = 1
        out.append(det)
        added.append(f"{det['en_class']} {det['score']:.2f}")
    if added:
        logger(
            f"[vision] 补召回 {len(added)} 个（imgsz={YOLO_RECALL_IMGSZ} "
            f"conf>={YOLO_RECALL_CONF}）：{'、'.join(added)}"
        )
    return out


def detect_to_annotations(
    frame,
    w2c_mat,
    *,
    weights: str | None = None,
    class_map: ClassMap | None = None,
    conf: float = YOLO_CONF,
    iou: float = YOLO_IOU,
    imgsz: int = YOLO_IMGSZ,
    vote_frames=None,
    vote_min: int = YOLO_VOTE_MIN,
    vote_iou: float = YOLO_VOTE_IOU,
    logger=print,
) -> dict:
    """一帧（或一叠帧投票）→ 仿真格式 annotations dict。"""
    from ultralytics import YOLO

    weights = weights or default_weights()
    cmap = class_map or ClassMap()
    logger(f"[vision] 权重 {os.path.basename(weights)}；{cmap.describe()}")

    model = YOLO(str(weights))
    if getattr(model, "task", None) != "segment":
        logger(
            f"[vision] 注意：权重任务是 {getattr(model, 'task', None)!r}，不是 segment。"
            "没有 mask 时只能用 bbox 取点，位置会偏。"
        )
    names = model.names or {}
    skipped: list[str] = []
    stack = [frame]
    if vote_frames:
        stack = list(vote_frames)
    if len(stack) <= 1:
        dets = _predict_raw(
            model, frame.color, names=names, conf=conf, iou=iou, imgsz=imgsz
        )
        logger(f"[vision] YOLO 检出 {len(dets)} 个候选（conf>={conf} imgsz={imgsz}）")
        dets = _merge_recall(model, frame, names, iou, dets, vote_iou, logger)
        objects = nms_world_objects(
            _objects_from_dets(dets, frame, w2c_mat, cmap, skipped), skipped
        )
    else:
        per_frame = []
        for i, snap in enumerate(stack, 1):
            dets = _predict_raw(
                model, snap.color, names=names, conf=conf, iou=iou, imgsz=imgsz
            )
            per_frame.append(dets)
            logger(
                f"[vision] 投票帧 {i}/{len(stack)}：{len(dets)} 个 "
                + (
                    "、".join(f"{d['en_class']} {d['score']:.2f}" for d in dets)
                    or "空"
                )
            )
        need = min(int(vote_min), len(stack))
        winners = vote_detections(per_frame, iou_thr=vote_iou, min_votes=need)
        tallies = {}
        for dets in per_frame:
            seen = {d["en_class"] for d in dets}
            for name in seen:
                tallies[name] = tallies.get(name, 0) + 1
        tally_txt = "、".join(
            f"{k}×{v}/{len(stack)}" for k, v in sorted(tallies.items())
        ) or "无"
        logger(
            f"[vision] 多帧投票 {len(stack)} 帧，门槛 ≥{need}：{tally_txt} "
            f"→ 采用 {len(winners)} 个"
        )
        winners = _merge_recall(
            model, frame, names, iou, winners, vote_iou, logger
        )
        objects = nms_world_objects(
            _objects_from_dets(winners, frame, w2c_mat, cmap, skipped), skipped
        )

    for msg in skipped:
        logger(f"[vision] 跳过 {msg}")

    h, w = frame.color.shape[:2]
    ann = {
        "image": "color.png",
        "width": int(w),
        "height": int(h),
        "objects": objects,
        "camera": {
            "position": np.asarray(w2c_mat, dtype=float)[:3, 3].tolist(),
            "rotmat": np.asarray(w2c_mat, dtype=float)[:3, :3].tolist(),
            "intrinsics": dict(frame.intrinsics),
        },
        "source": "real_d405_yolo_seg",
        "weights": os.path.basename(str(weights)),
    }
    logger(f"[vision] 可用零件 {len(objects)} 个（跳过 {len(skipped)} 个）")
    return ann


def save_annotations(ann: dict, out_root: str, sample_id: str, frame=None) -> str:
    """写成 ``<out_root>/<sample_id>/annotations.json``。

    这个目录结构就是仿真的 ``--dataset-root`` + ``--sample-id``，
    存一份彩色图方便事后核对识别对不对。
    """
    sample_dir = os.path.join(os.path.abspath(out_root), str(sample_id))
    os.makedirs(sample_dir, exist_ok=True)
    ann_path = os.path.join(sample_dir, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    if frame is not None:
        import cv2

        cv2.imwrite(os.path.join(sample_dir, "color.png"), frame.color)
        np.save(os.path.join(sample_dir, "depth_m.npy"), frame.depth_m.astype(np.float32))
    return ann_path


def save_preview(ann: dict, frame, out_path: str) -> str:
    """画出 mask 轮廓 + 世界坐标，用来肉眼确认识别和坐标是不是靠谱。"""
    import cv2

    img = frame.color.copy()
    for obj in ann["objects"]:
        poly = obj.get("polygon") or []
        if len(poly) >= 3:
            cv2.polylines(img, [np.asarray(poly, dtype=np.int32)], True, (0, 255, 0), 2)
        x1, y1, _x2, _y2 = obj["bbox"]
        p = obj["pose_6d"]
        cv2.putText(
            img,
            f"{obj['yolo_class']} {obj.get('score', 0):.2f}",
            (x1, max(14, y1 - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"({p['x']:+.3f},{p['y']:+.3f}) yaw={np.degrees(p['yaw']):+.0f} {obj['state']}",
            (x1, max(28, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, img)
    return os.path.abspath(out_path)


def print_objects(ann: dict, logger=print) -> None:
    if not ann["objects"]:
        logger("[vision] 没有可用零件。常见原因：手眼没标准（点全被工作区滤掉）、"
               "深度无效、或权重认不出这些件。")
        return
    logger("[vision] 识别结果（世界坐标，米）：")
    for obj in ann["objects"]:
        p = obj["pose_6d"]
        flag = " 形状近似" if obj.get("stl_is_approx") else ""
        logger(
            f"  [{obj['id']}] {obj['class']:<22s} {obj['yolo_class']:<28s} "
            f"conf={obj.get('score', 0):.2f} "
            f"xy=({p['x']:+.4f},{p['y']:+.4f}) z={p['z']:.3f} "
            f"yaw={np.degrees(p['yaw']):+7.1f}° {obj['state']:<7s} "
            f"pts={obj['world_points']}{flag}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D405 + YOLO → 仿真格式 annotations.json")
    p.add_argument("--frame-dir", default="", help="用存下的帧，不连相机")
    p.add_argument("--resolution", default=CAMERA_RESOLUTION, choices=("mid", "high"))
    p.add_argument("--serial", default=CAMERA_SERIAL)
    p.add_argument(
        "--frames",
        type=int,
        default=YOLO_VOTE_FRAMES,
        help="曝光稳住后连拍帧数（深度中位数 + YOLO 投票）",
    )
    p.add_argument("--weights", default="", help="YOLO 权重；默认用本目录的")
    p.add_argument("--handeye", default="", help="手眼 JSON；默认 handeye_d405.json")
    p.add_argument("--conf", type=float, default=YOLO_CONF)
    p.add_argument("--iou", type=float, default=YOLO_IOU)
    p.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    p.add_argument("--out-root", default=VISION_DIR, help="输出根目录（当仿真的 --dataset-root）")
    p.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    p.add_argument("--no-preview", action="store_true", help="不写预览图")
    return p


def capture_frame(args, logger=print):
    """连相机时返回 (中位深度帧, 投票用帧列表)；离线只有单帧。"""
    from d405_camera import D405Camera, frames_to_median, load_frame

    if args.frame_dir:
        logger(f"[vision] 离线帧 {args.frame_dir}")
        frame = load_frame(args.frame_dir)
        return frame, None
    with D405Camera(args.resolution, args.serial, logger=logger) as cam:
        stack = cam.capture_stack(args.frames)
        return frames_to_median(stack), stack


def main() -> int:
    args = build_parser().parse_args()

    from d405_camera import CameraError

    try:
        w2c = load_handeye(args.handeye or None)
        print(f"[vision] 手眼 {describe_handeye(w2c)}")
        frame, vote_frames = capture_frame(args)
        ann = detect_to_annotations(
            frame,
            w2c,
            weights=args.weights or None,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            vote_frames=vote_frames,
        )
    except (VisionConfigError, VisionError, CameraError) as e:
        print(f"[vision] 失败：{e}")
        return 1

    print_objects(ann)
    ann_path = save_annotations(ann, args.out_root, args.sample_id, frame=frame)
    print(f"[vision] annotations -> {ann_path}")
    if not args.no_preview:
        prev = save_preview(
            ann, frame, os.path.join(os.path.dirname(ann_path), "detect_preview.jpg")
        )
        print(f"[vision] 预览图 -> {prev}")
    print(
        "[vision] 交给仿真规划：\n"
        f"  python tiaozhanbei/real/run_task_real.py --task <任务.json> "
        f"--dataset-root {os.path.abspath(args.out_root)} --sample-id {args.sample_id}"
    )
    return 0 if ann["objects"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
