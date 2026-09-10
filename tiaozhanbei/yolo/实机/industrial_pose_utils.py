"""检测框 + 深度图 → 3D 位置与 yaw 估计。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PoseEstimate:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[int]
    position_world: np.ndarray | None
    rotmat: np.ndarray | None = None
    yaw_deg: float | None = None
    n_points: int = 0

    def as_dict(self):
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "position_world": None if self.position_world is None else self.position_world.tolist(),
            "rotmat": None if self.rotmat is None else self.rotmat.tolist(),
            "yaw_deg": self.yaw_deg,
            "n_points": self.n_points,
        }


def intrinsics_from_fov(width: int, height: int, fov_deg: float = 45.0) -> np.ndarray:
    h_fov = np.deg2rad(fov_deg)
    aspect = width / height
    v_fov = 2 * np.arctan(np.tan(h_fov / 2) / aspect)
    fx = (width / 2) / np.tan(h_fov / 2)
    fy = (height / 2) / np.tan(v_fov / 2)
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def cam_cv_to_world(point_cam_cv: np.ndarray, vcam) -> np.ndarray:
    """OpenCV 相机坐标 (X右 Y下 Z前) → 世界坐标。"""
    p_p3d = np.array([point_cam_cv[0], point_cam_cv[2], -point_cam_cv[1]], dtype=np.float64)
    return vcam.cam_rotmat @ p_p3d + vcam.cam_pos


def depth_roi_points_world(
    bbox_xyxy,
    depth: np.ndarray,
    vcam,
    min_points: int = 20,
) -> tuple[np.ndarray | None, int]:
    """从 bbox 区域深度反投影得到世界坐标点云。"""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, 0

    roi = depth[y1:y2, x1:x2]
    valid = roi[(roi > 0.01) & (roi < getattr(vcam, "depth_far", 3.0))]
    if valid.size < min_points:
        return None, int(valid.size)

    K = vcam.intrinsics
    us, vs = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    us = us.reshape(-1)
    vs = vs.reshape(-1)
    zs = roi.reshape(-1)
    mask = (zs > 0.01) & (zs < getattr(vcam, "depth_far", 3.0))
    us, vs, zs = us[mask], vs[mask], zs[mask]
    if zs.size < min_points:
        return None, int(zs.size)

    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts_world = []
    for x, y, z in zip(xs, ys, zs):
        pts_world.append(cam_cv_to_world(np.array([x, y, z]), vcam))
    return np.asarray(pts_world, dtype=np.float64), int(len(pts_world))


def depth_roi_points_camera(
    bbox_xyxy,
    depth: np.ndarray,
    K: np.ndarray,
    depth_far: float = 3.0,
    min_points: int = 20,
) -> tuple[np.ndarray | None, int]:
    """仅相机坐标系点云（用于无 vcam 的离线图片）。"""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, 0

    roi = depth[y1:y2, x1:x2]
    us, vs = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    us = us.reshape(-1)
    vs = vs.reshape(-1)
    zs = roi.reshape(-1)
    mask = (zs > 0.01) & (zs < depth_far)
    us, vs, zs = us[mask], vs[mask], zs[mask]
    if zs.size < min_points:
        return None, int(zs.size)

    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack([xs, ys, zs], axis=1)
    return pts, int(len(pts))


def estimate_position_from_points(points: np.ndarray) -> np.ndarray:
    """使用中位数，降低深度噪声引起的抖动。"""
    return np.median(points, axis=0)


def subsample_points(points: np.ndarray, max_points: int = 800) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
    return points[idx]


def align_rotmat_continuous(reference: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    """消除 PCA 的 180° 翻转，使相邻帧旋转连续。"""
    aligned = rotmat.copy()
    for i in range(3):
        if np.dot(aligned[:, i], reference[:, i]) < 0:
            aligned[:, i] *= -1
    if np.linalg.det(aligned) < 0:
        aligned[:, 2] *= -1
    return aligned


def smooth_rotmat(prev: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    new = align_rotmat_continuous(prev, new)
    blended = (1.0 - alpha) * prev + alpha * new
    u, _, vt = np.linalg.svd(blended)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def filter_point_outliers(points: np.ndarray, percentile: float = 88.0, margin: float = 0.02) -> np.ndarray:
    """剔除深度离群点，稳定 PCA 与位置估计。"""
    if points.shape[0] < 30:
        return points
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    thresh = float(np.percentile(dist, percentile)) + margin
    kept = points[dist <= thresh]
    return kept if kept.shape[0] >= 20 else points


def deduplicate_poses(
    poses: list[PoseEstimate],
    dist_thresh: float = 0.06,
    iou_thresh: float = 0.30,
    max_poses: int | None = None,
) -> list[PoseEstimate]:
    """3D 距离 + bbox IoU 联合去重，只保留高置信度目标。"""
    valid = [p for p in poses if p.position_world is not None and p.rotmat is not None]
    order = sorted(range(len(valid)), key=lambda i: -valid[i].confidence)
    keep_idx: list[int] = []
    for i in order:
        pose = valid[i]
        duplicate = False
        for kept in keep_idx:
            other = valid[kept]
            dist = float(np.linalg.norm(pose.position_world - other.position_world))
            iou = bbox_iou(pose.bbox_xyxy, other.bbox_xyxy)
            if dist < dist_thresh or iou > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            keep_idx.append(i)
    result = [valid[i] for i in keep_idx]
    if max_poses is not None and len(result) > max_poses:
        result = sorted(result, key=lambda p: -p.confidence)[:max_poses]
    return result


def bbox_center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def deduplicate_boxes_global(boxes, classes, confs, iou_thresh: float = 0.5):
    """跨类别去重：重叠框只保留置信度最高者。"""
    order = np.argsort(-confs)
    keep = []
    for idx in order:
        if all(bbox_iou(boxes[idx], boxes[k]) <= iou_thresh for k in keep):
            keep.append(idx)
    return keep


def deduplicate_boxes_by_center(boxes, confs, center_dist: float = 40.0):
    """检测框中心过近时只保留高置信度框（解决 IoU 偏低但仍重复的情况）。"""
    order = np.argsort(-confs)
    keep: list[int] = []
    for idx in order:
        cx, cy = bbox_center(boxes[idx])
        too_close = False
        for kept in keep:
            kx, ky = bbox_center(boxes[kept])
            if (cx - kx) ** 2 + (cy - ky) ** 2 <= center_dist ** 2:
                too_close = True
                break
        if not too_close:
            keep.append(idx)
    return keep


def deduplicate_boxes_multi_stage(
    boxes,
    classes,
    confs,
    iou_thresh: float = 0.65,
    center_px: float = 32.0,
) -> list[int]:
    """IoU + 2D 中心距离联合去重，减少同一物体的重复框。"""
    keep = deduplicate_boxes_global(boxes, classes, confs, iou_thresh=iou_thresh)
    order = sorted(keep, key=lambda i: -confs[i])
    final: list[int] = []
    for idx in order:
        cx, cy = bbox_center(boxes[idx])
        duplicate = False
        for kept in final:
            kcx, kcy = bbox_center(boxes[kept])
            if (cx - kcx) ** 2 + (cy - kcy) ** 2 < center_px ** 2:
                duplicate = True
                break
        if not duplicate:
            final.append(idx)
    return final


def deduplicate_boxes(boxes, classes, confs, iou_thresh: float = 0.5):
    """保留同类高 IoU 框中 conf 最高者。"""
    order = np.argsort(-confs)
    keep = []
    for idx in order:
        ok = True
        for kept in keep:
            if classes[idx] == classes[kept] and bbox_iou(boxes[idx], boxes[kept]) > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(idx)
    return keep


def angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def yaw_error_ambiguous_deg(est_yaw: float, gt_yaw: float) -> float:
    """yaw 存在 180° 对称歧义时，取较小误差（适用于垫圈/螺母等近似对称件）。"""
    e1 = angle_diff_deg(est_yaw, gt_yaw)
    e2 = angle_diff_deg(est_yaw, gt_yaw + 180.0)
    return min(e1, e2)


def estimate_rotmat_tabletop_yaw(points_world: np.ndarray, bbox=None) -> np.ndarray | None:
    """桌面场景：Z 轴固定竖直向上，仅估计绕 Z 的 yaw。"""
    points_world = subsample_points(points_world)
    if points_world.shape[0] < 20:
        return None

    center = np.median(points_world, axis=0)
    xy = points_world[:, :2] - center[:2]
    if np.allclose(xy.std(axis=0), 0):
        return None

    cov = (xy.T @ xy) / max(xy.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    direction = evecs[:, int(np.argmax(evals))]
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        alt = np.array([-direction[1], direction[0]])
        if bw >= bh:
            if abs(direction[0]) < abs(alt[0]):
                direction = alt
        else:
            if abs(direction[1]) < abs(alt[1]):
                direction = alt
        direction = direction / (np.linalg.norm(direction) + 1e-8)

    if direction[0] < 0:
        direction = -direction

    x_axis = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return None
    y_axis /= y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    return np.column_stack([x_axis, y_axis, z_axis])


def estimate_rotmat_6d_tabletop(points_world: np.ndarray) -> np.ndarray | None:
    """
    由点云 PCA 估计 6D 旋转矩阵（桌面场景）。
    列向量分别为物体 X/Y/Z 轴在世界系下的方向；Z 轴近似朝上。
    """
    points_world = subsample_points(points_world)
    if points_world.shape[0] < 20:
        return None

    center = np.median(points_world, axis=0)
    centered = points_world - center
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    _, evecs = np.linalg.eigh(cov)

    z_axis = evecs[:, 0]
    if z_axis[2] < 0:
        z_axis = -z_axis

    x_axis = evecs[:, 2]
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return None
    y_axis = y_axis / y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    rotmat = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(rotmat) < 0:
        rotmat[:, 0] *= -1
    return rotmat


@dataclass
class _PoseTrack:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[int]
    position_world: np.ndarray
    rotmat: np.ndarray
    yaw_deg: float | None
    n_points: int
    missed: int = 0


class PoseSmoother:
    """跨帧 EMA 平滑位置与旋转，抑制坐标轴抖动。"""

    def __init__(
        self,
        pos_alpha: float = 0.22,
        rot_alpha: float = 0.18,
        match_dist: float = 0.08,
        match_iou: float = 0.20,
        max_missed: int = 2,
        max_tracks: int = 12,
        dedup_dist: float = 0.06,
    ):
        self.pos_alpha = pos_alpha
        self.rot_alpha = rot_alpha
        self.match_dist = match_dist
        self.match_iou = match_iou
        self.max_missed = max_missed
        self.max_tracks = max_tracks
        self.dedup_dist = dedup_dist
        self._tracks: list[_PoseTrack] = []
        self._next_id = 0

    def reset(self):
        self._tracks.clear()
        self._next_id = 0

    def _match_cost(self, track: _PoseTrack, pose: PoseEstimate) -> float:
        if pose.position_world is None:
            return float("inf")
        dist = float(np.linalg.norm(track.position_world - pose.position_world))
        iou = bbox_iou(track.bbox_xyxy, pose.bbox_xyxy)
        if dist > self.match_dist and iou < self.match_iou:
            return float("inf")
        return dist - 0.08 * iou

    def _dedup_tracks(self, tracks: list[_PoseTrack]) -> list[_PoseTrack]:
        order = sorted(range(len(tracks)), key=lambda i: -tracks[i].confidence)
        keep: list[int] = []
        for i in order:
            duplicate = False
            for k in keep:
                dist = float(np.linalg.norm(tracks[i].position_world - tracks[k].position_world))
                iou = bbox_iou(tracks[i].bbox_xyxy, tracks[k].bbox_xyxy)
                if dist < self.dedup_dist or iou > 0.30:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(i)
        kept = [tracks[i] for i in keep]
        if len(kept) > self.max_tracks:
            kept = sorted(kept, key=lambda t: -t.confidence)[: self.max_tracks]
        return kept

    def update(self, poses: list[PoseEstimate]) -> list[PoseEstimate]:
        poses = deduplicate_poses(poses, dist_thresh=self.dedup_dist, max_poses=self.max_tracks)
        valid_poses = [p for p in poses if p.position_world is not None and p.rotmat is not None]
        unmatched = set(range(len(valid_poses)))
        updated_tracks: list[_PoseTrack] = []

        for track in self._tracks:
            best_idx = None
            best_cost = float("inf")
            for idx in unmatched:
                cost = self._match_cost(track, valid_poses[idx])
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            if best_idx is None:
                track.missed += 1
                if track.missed <= self.max_missed:
                    updated_tracks.append(track)
                continue

            pose = valid_poses[best_idx]
            unmatched.remove(best_idx)
            pos = (1.0 - self.pos_alpha) * track.position_world + self.pos_alpha * pose.position_world
            rot = smooth_rotmat(track.rotmat, pose.rotmat, self.rot_alpha)
            yaw = estimate_yaw_deg_from_rotmat(rot)
            updated_tracks.append(
                _PoseTrack(
                    track_id=track.track_id,
                    class_id=pose.class_id,
                    class_name=pose.class_name,
                    confidence=max(track.confidence, pose.confidence),
                    bbox_xyxy=pose.bbox_xyxy,
                    position_world=pos,
                    rotmat=rot,
                    yaw_deg=yaw,
                    n_points=pose.n_points,
                    missed=0,
                )
            )

        for idx in unmatched:
            pose = valid_poses[idx]
            updated_tracks.append(
                _PoseTrack(
                    track_id=self._next_id,
                    class_id=pose.class_id,
                    class_name=pose.class_name,
                    confidence=pose.confidence,
                    bbox_xyxy=pose.bbox_xyxy,
                    position_world=pose.position_world.copy(),
                    rotmat=pose.rotmat.copy(),
                    yaw_deg=pose.yaw_deg,
                    n_points=pose.n_points,
                    missed=0,
                )
            )
            self._next_id += 1

        self._tracks = self._dedup_tracks(updated_tracks)
        return [
            PoseEstimate(
                class_id=t.class_id,
                class_name=t.class_name,
                confidence=t.confidence,
                bbox_xyxy=t.bbox_xyxy,
                position_world=t.position_world,
                rotmat=t.rotmat,
                yaw_deg=t.yaw_deg,
                n_points=t.n_points,
            )
            for t in self._tracks
        ]


def estimate_yaw_deg_from_rotmat(rotmat: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(rotmat[1, 0], rotmat[0, 0])))


def estimate_yaw_deg_from_points_world(points_world: np.ndarray) -> float | None:
    rotmat = estimate_rotmat_6d_tabletop(points_world)
    if rotmat is None:
        return None
    return estimate_yaw_deg_from_rotmat(rotmat)


def rotation_error_deg(rot_est: np.ndarray, rot_gt: np.ndarray) -> float:
    rot_diff = rot_est @ rot_gt.T
    cos_angle = np.clip((np.trace(rot_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def world_to_pixel(point_world: np.ndarray, vcam) -> tuple[int, int] | None:
    p_local = vcam.cam_rotmat.T @ (point_world - vcam.cam_pos)
    x_cv = p_local[0]
    y_cv = -p_local[2]
    z_cv = p_local[1]
    if z_cv <= 1e-4:
        return None
    K = vcam.intrinsics
    u = K[0, 0] * x_cv / z_cv + K[0, 2]
    v = K[1, 1] * y_cv / z_cv + K[1, 2]
    return int(round(u)), int(round(v))


def draw_axis_2d(img, origin, end, color, thickness=2):
    if origin is None or end is None:
        return
    cv2.arrowedLine(img, origin, end, color, thickness, tipLength=0.25)


def draw_pose_axes_2d(img_bgr, pose: PoseEstimate, vcam, axis_len: float = 0.04):
    if pose.position_world is None or pose.rotmat is None:
        return
    origin = world_to_pixel(pose.position_world, vcam)
    if origin is None:
        return
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X红 Y绿 Z蓝
    for i, color in enumerate(colors):
        axis_end = pose.position_world + pose.rotmat[:, i] * axis_len
        end = world_to_pixel(axis_end, vcam)
        draw_axis_2d(img_bgr, origin, end, color, thickness=2)


def estimate_yaw_deg_from_points_world_legacy(points_world: np.ndarray) -> float | None:
    """桌面场景：在 XY 平面上 PCA 求主方向 yaw。"""
    if points_world.shape[0] < 10:
        return None
    xy = points_world[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    if np.allclose(xy.std(axis=0), 0):
        return None
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    direction = vt[0]
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


def estimate_poses_from_yolo(
    results,
    depth: np.ndarray,
    vcam,
    class_names: dict[int, str],
    conf_thresh: float = 0.25,
    dedup_iou: float = 0.45,
    global_dedup: bool = True,
    smoother: PoseSmoother | None = None,
    pose_dedup_dist: float = 0.06,
    max_poses: int = 12,
    center_dist: float = 40.0,
) -> list[PoseEstimate]:
    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    boxes = boxes_obj.xyxy.cpu().numpy()
    classes = boxes_obj.cls.cpu().numpy().astype(int)
    confs = boxes_obj.conf.cpu().numpy()

    keep_fn = deduplicate_boxes_global if global_dedup else deduplicate_boxes
    keep = keep_fn(boxes, classes, confs, iou_thresh=dedup_iou)
    if center_dist > 0 and keep:
        sub_keep = deduplicate_boxes_by_center(boxes[keep], confs[keep], center_dist=center_dist)
        keep = [keep[i] for i in sub_keep]
    poses: list[PoseEstimate] = []

    for idx in keep:
        if confs[idx] < conf_thresh:
            continue
        bbox = [int(v) for v in boxes[idx]]
        cls_id = int(classes[idx])
        cls_name = class_names.get(cls_id, str(cls_id))

        points, n_pts = depth_roi_points_world(bbox, depth, vcam)
        if points is None:
            continue

        points = filter_point_outliers(points)
        points = subsample_points(points)
        pos = estimate_position_from_points(points)
        rotmat = estimate_rotmat_tabletop_yaw(points, bbox=bbox)
        if rotmat is None:
            continue
        yaw = estimate_yaw_deg_from_rotmat(rotmat)
        poses.append(
            PoseEstimate(
                class_id=cls_id,
                class_name=cls_name,
                confidence=float(confs[idx]),
                bbox_xyxy=bbox,
                position_world=pos,
                rotmat=rotmat,
                yaw_deg=yaw,
                n_points=n_pts,
            )
        )

    poses = deduplicate_poses(poses, dist_thresh=pose_dedup_dist, max_poses=max_poses)
    if smoother is not None:
        return smoother.update(poses)
    return poses


def load_meta_annotations(meta_path: Path) -> tuple[dict | None, list[dict]]:
    if not meta_path.exists():
        return None, []
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return None, data
    return data.get("camera"), data.get("objects", [])


def apply_camera_from_meta(vcam, camera: dict):
    vcam.cam_pos = np.array(camera["cam_pos"], dtype=np.float64)
    if "cam_rotmat" in camera:
        vcam.cam_rotmat = np.array(camera["cam_rotmat"], dtype=np.float64)


def load_meta_annotations_legacy(meta_path: Path) -> list[dict]:
    _, objects = load_meta_annotations(meta_path)
    return objects


def match_pose_to_gt(pose: PoseEstimate, gt_items: list[dict], matched_gt: set[int]) -> tuple[int | None, float]:
    best_idx, best_iou = None, 0.0
    for i, gt in enumerate(gt_items):
        if i in matched_gt:
            continue
        if gt.get("class_id") != pose.class_id:
            continue
        iou = bbox_iou(pose.bbox_xyxy, gt["bbox_xyxy"])
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    if best_idx is None or best_iou < 0.1:
        return None, 0.0
    return best_idx, best_iou


def rotation_matrix_to_yaw_deg(rotmat: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(rotmat[1, 0], rotmat[0, 0])))


def draw_poses_on_image(
    img_bgr: np.ndarray,
    poses: list[PoseEstimate],
    vcam=None,
    axis_len: float = 0.04,
) -> np.ndarray:
    out = img_bgr.copy()
    valid = [p for p in poses if p.position_world is not None and p.rotmat is not None]
    for idx, pose in enumerate(valid):
        x1, y1, x2, y2 = pose.bbox_xyxy
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{pose.class_name} {pose.confidence:.2f}"
        if pose.yaw_deg is not None:
            label += f" yaw={pose.yaw_deg:.0f}"
        label_y = max(y1 - 6 - 14 * (idx % 2), 14)
        cv2.putText(out, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if vcam is not None:
            draw_pose_axes_2d(out, pose, vcam, axis_len=axis_len)
    return out


def print_pose_report(poses: list[PoseEstimate], prefix: str = ""):
    if not poses:
        print(f"{prefix}未估计到任何位姿")
        return
    print(f"{prefix}6D 位姿估计 ({len(poses)} 个):")
    for i, pose in enumerate(poses, 1):
        if pose.position_world is None:
            print(f"  [{i}] {pose.class_name:14s} conf={pose.confidence:.2f}  深度点不足({pose.n_points})")
            continue
        p = pose.position_world
        yaw_txt = f" yaw={pose.yaw_deg:.1f}" if pose.yaw_deg is not None else ""
        rot_txt = ""
        if pose.rotmat is not None:
            rot_txt = (
                f" R=[{pose.rotmat[0,0]:+.2f},{pose.rotmat[0,1]:+.2f},{pose.rotmat[0,2]:+.2f}; "
                f"{pose.rotmat[1,0]:+.2f},{pose.rotmat[1,1]:+.2f},{pose.rotmat[1,2]:+.2f}]"
            )
        print(
            f"  [{i}] {pose.class_name:14s} conf={pose.confidence:.2f}  "
            f"pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}){yaw_txt}{rot_txt}  pts={pose.n_points}"
        )


def pose_to_vision_object(pose: PoseEstimate) -> dict | None:
    """单目标 → 工作流 vision_objects 元素（内部世界坐标格式）。"""
    if pose.position_world is None or pose.rotmat is None:
        return None
    p = pose.position_world
    obj = {
        "name": pose.class_name,
        "conf": round(float(pose.confidence), 4),
        "coord": [round(float(p[0]), 3), round(float(p[1]), 3)],
        "z": round(float(p[2]), 3),
    }
    if pose.yaw_deg is not None:
        obj["yaw"] = round(float(pose.yaw_deg), 1)
    obj["rotate_matrix"] = [
        [round(float(pose.rotmat[0, j]), 2) for j in range(3)],
        [round(float(pose.rotmat[1, j]), 2) for j in range(3)],
    ]
    return obj


def poses_to_vision_objects(poses: list[PoseEstimate]) -> list[dict]:
    """位姿列表 → vision_objects JSON 数组（无物体时 []）。"""
    items: list[dict] = []
    for pose in poses:
        obj = pose_to_vision_object(pose)
        if obj is not None:
            items.append(obj)
    return items


# 下游「三维测试」格式：中文名 + 像素三维包围盒 + state
# 示例字段: name/category/position{x_min..z_max}/confidence/state
CLASS_CN_META: dict[str, tuple[str, str, float]] = {
    # eng_name -> (中文名, 类别, 默认半高 mm，用于估 z_min/z_max)
    "bearing": ("轴承", "零件", 12.0),
    "gear": ("齿轮", "零件", 15.0),
    "machine_tool": ("机床工具", "工具", 35.0),
    "nut": ("螺母", "紧固件", 8.0),
    "rivet": ("铆钉", "紧固件", 10.0),
    "roller_bearing": ("滚子轴承", "零件", 14.0),
    "screw_bolt": ("螺丝螺栓", "紧固件", 12.0),
    "washer": ("垫圈", "紧固件", 4.0),
    "round_part": ("圆形零件", "零件", 12.0),
    "fastener": ("紧固件", "紧固件", 10.0),
}


def pose_to_downstream3d_object(
    pose: PoseEstimate,
    *,
    depth: np.ndarray | None = None,
    state: str = "未拾取",
    z_unit: str = "mm",
) -> dict | None:
    """
    转为下游三维测试输入格式。

    position.x/y_* : 图像像素 bbox
    position.z_*   : 优先用深度图 ROI（相机深度，转 mm）；否则用世界 Z + 类别默认高度
    """
    if pose.bbox_xyxy is None or len(pose.bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in pose.bbox_xyxy]
    cn_name, category, half_h_mm = CLASS_CN_META.get(
        pose.class_name, (pose.class_name, "零件", 15.0)
    )

    z_min_mm: float
    z_max_mm: float
    if depth is not None:
        h, w = depth.shape[:2]
        xa, xb = max(0, x1), min(w, x2)
        ya, yb = max(0, y1), min(h, y2)
        roi = depth[ya:yb, xa:xb] if xb > xa and yb > ya else None
        if roi is not None and roi.size > 0:
            valid = roi[(roi > 0.01) & (roi < 5.0)]
            if valid.size >= 10:
                # 相机深度（米）→ mm；取分位数更稳
                z_min_mm = float(np.percentile(valid, 10) * 1000.0)
                z_max_mm = float(np.percentile(valid, 90) * 1000.0)
            else:
                z_min_mm, z_max_mm = 0.0, half_h_mm * 2
        else:
            z_min_mm, z_max_mm = 0.0, half_h_mm * 2
    elif pose.position_world is not None:
        z_c = float(pose.position_world[2]) * 1000.0
        z_min_mm = max(0.0, z_c - half_h_mm)
        z_max_mm = z_c + half_h_mm
    else:
        z_min_mm, z_max_mm = 0.0, half_h_mm * 2

    if z_unit != "mm":
        # 保留扩展；当前下游样例数值量级接近 mm
        pass

    return {
        "name": cn_name,
        "category": category,
        "position": {
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2,
            "z_min": int(round(z_min_mm)),
            "z_max": int(round(z_max_mm)),
        },
        "confidence": round(float(pose.confidence), 2),
        "state": state,
    }


def poses_to_downstream3d(
    poses: list[PoseEstimate],
    *,
    depth: np.ndarray | None = None,
    state: str = "未拾取",
) -> list[dict]:
    items: list[dict] = []
    for pose in poses:
        obj = pose_to_downstream3d_object(pose, depth=depth, state=state)
        if obj is not None:
            items.append(obj)
    return items


def vision_objects_json(
    poses: list[PoseEstimate],
    *,
    indent: int | None = None,
    fmt: str = "internal",
    depth: np.ndarray | None = None,
) -> str:
    """
    导出 JSON 字符串。
    fmt=internal  : 世界坐标 vision_objects（原格式）
    fmt=downstream3d : 下游三维测试框格式
    """
    if fmt == "downstream3d":
        data = poses_to_downstream3d(poses, depth=depth)
    else:
        data = poses_to_vision_objects(poses)
    return json.dumps(data, ensure_ascii=False, indent=indent)


def print_pose_errors(poses: list[PoseEstimate], gt_items: list[dict]):
    matched_gt: set[int] = set()
    pos_errors = []
    yaw_errors = []
    rot_errors = []
    print("[POSE-EVAL] 与标注真值对比:")
    for pose in poses:
        gt_idx, iou = match_pose_to_gt(pose, gt_items, matched_gt)
        if gt_idx is None or pose.position_world is None:
            print(f"  - {pose.class_name}: 无匹配真值")
            continue
        matched_gt.add(gt_idx)
        gt = gt_items[gt_idx]
        gt_pos = np.array(gt["position_world"], dtype=np.float64)
        pos_err = float(np.linalg.norm(pose.position_world - gt_pos) * 100)
        pos_errors.append(pos_err)
        gt_rot = np.array(gt["rotation_world"], dtype=np.float64)
        gt_yaw = rotation_matrix_to_yaw_deg(gt_rot)
        yaw_err = None
        rot_err = None
        if pose.rotmat is not None:
            rot_err = rotation_error_deg(pose.rotmat, gt_rot)
            rot_errors.append(rot_err)
        if pose.yaw_deg is not None:
            yaw_err = angle_diff_deg(pose.yaw_deg, gt_yaw)
            yaw_errors.append(yaw_err)
        err_txt = f", rot_err={rot_err:.1f}deg" if rot_err is not None else ""
        yaw_txt = f", yaw_err={yaw_err:.1f}deg" if yaw_err is not None else ""
        print(
            f"  - {pose.class_name}: pos_err={pos_err:.1f}cm, iou={iou:.2f}{yaw_txt}{err_txt} "
            f"(gt=({gt_pos[0]:.3f},{gt_pos[1]:.3f},{gt_pos[2]:.3f}))"
        )
    if pos_errors:
        print(f"[POSE-EVAL] 平均位置误差: {np.mean(pos_errors):.1f} cm")
    if yaw_errors:
        print(f"[POSE-EVAL] 平均 yaw 误差: {np.mean(yaw_errors):.1f} deg")
    if rot_errors:
        print(f"[POSE-EVAL] 平均 6D 旋转误差: {np.mean(rot_errors):.1f} deg")
