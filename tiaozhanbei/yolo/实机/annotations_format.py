"""
仿真 dataset_learn/annotations.json 的字段约定。

YOLO 推理输出必须与采集脚本 capture_learn_format 对齐：
  顶层: image, width, height, objects, camera
  物体: id, class, bbox, polygon, mask_path, pose_6d, state, visible_pixels, stl_path

采集侧额外字段（seed / desk_stripe / part_paths / scene_layout 等）
在推理时若旁路 annotations.json 存在则原样拷贝，否则省略。
"""

from __future__ import annotations

from typing import Any

# 与 SimDataCamera(1280x720, fov=45°) 采集默认一致
DEFAULT_CAMERA: dict[str, Any] = {
    "position": [0.65, -0.95, 0.75],
    "rotmat": [
        [0.8253072500228882, -0.4787442088127136, -0.29945263266563416],
        [0.5646839141845703, 0.6997030973434448, 0.4376615285873413],
        [0.0, -0.5303013324737549, 0.8478091955184937],
    ],
    "intrinsics": {
        "fx": 1545.0966799187809,
        "fy": 1545.096596981165,
        "cx": 639.5,
        "cy": 359.5,
    },
    "near": 0.01,
    "far": 5.0,
}

OBJECT_KEYS = (
    "id",
    "class",
    "bbox",
    "polygon",
    "mask_path",
    "pose_6d",
    "state",
    "visible_pixels",
    "stl_path",
)

POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")

# 从已有 annotations.json 原样带回的采集元数据（推理本身算不出来）
SIDECAR_KEYS = (
    "camera",
    "seed",
    "desk_stripe",
    "part_paths",
    "scene_layout",
    "n_parts",
    "capture_mode",
    "angle_index",
    "yaw_deg",
)


def empty_pose_6d() -> dict[str, float]:
    return {k: 0.0 for k in POSE_KEYS}


def make_pose_6d(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> dict[str, float]:
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def make_object(
    obj_id: int,
    class_name: str,
    bbox: list[int] | None,
    polygon: list[list[int]] | None = None,
    mask_path: str = "",
    pose_6d: dict | None = None,
    state: str = "normal",
    visible_pixels: int = 0,
    stl_path: str | None = None,
    score: float | None = None,
    pointcloud_path: str | None = None,
) -> dict[str, Any]:
    """按仿真字段顺序构造单个 object。score 为 YOLO 置信度，仿真 GT 可无此字段。"""
    pose = empty_pose_6d()
    if pose_6d:
        for k in POSE_KEYS:
            if pose_6d.get(k) is not None:
                pose[k] = float(pose_6d[k])
    obj = {
        "id": int(obj_id),
        "class": str(class_name),
        "bbox": bbox,
        "polygon": polygon or [],
        "mask_path": mask_path or "",
        "coordinate": [pose["x"], pose["y"]],
        "pose_6d": pose,
        "state": state or "normal",
        "visible_pixels": int(visible_pixels),
        "stl_path": stl_path or None,
    }
    if score is not None:
        obj["score"] = round(float(score), 4)
    if pointcloud_path:
        obj["pointcloud_path"] = pointcloud_path
    return obj


def make_annotations(
    image: str,
    width: int,
    height: int,
    objects: list[dict],
    camera: dict | None = None,
    sidecar: dict | None = None,
) -> dict[str, Any]:
    """按仿真 annotations.json 顶层顺序组装。"""
    out: dict[str, Any] = {
        "image": image,
        "width": int(width),
        "height": int(height),
        "objects": objects,
        "camera": camera or DEFAULT_CAMERA,
    }
    if sidecar:
        for key in SIDECAR_KEYS:
            if key == "camera":
                continue
            if key in sidecar:
                out[key] = sidecar[key]
        if sidecar.get("camera"):
            out["camera"] = sidecar["camera"]
        box = (sidecar.get("scene_layout") or {}).get("storage_box")
        if box and "target_container" not in out:
            out["target_container"] = box
    return out
