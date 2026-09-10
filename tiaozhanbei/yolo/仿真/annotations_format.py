"""infer_seg_annotations.py 用的 annotations.json 结构。"""
from __future__ import annotations


DEFAULT_CAMERA = {
    "position": [0.65, -0.95, 0.75],
    "rotmat": [
        [0.8253072500228882, -0.4787442088127136, -0.29945263266563416],
        [0.5646839141845703, 0.6997030973434448, 0.4376615285873413],
        [0.0, -0.5303013324737549, 0.8478091955184937],
    ],
    "intrinsics": {"fx": 1545.0966799187809, "fy": 1545.096596981165, "cx": 639.5, "cy": 359.5},
    "near": 0.01,
    "far": 5.0,
}


def make_pose_6d(x, y, z, roll, pitch, yaw) -> dict:
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def make_object(
    obj_id,
    class_name,
    bbox,
    polygon,
    mask_path,
    pose_6d,
    state,
    visible_pixels,
    stl_path,
    score=None,
    pointcloud_path=None,
) -> dict:
    item = {
        "id": int(obj_id),
        "class": class_name,
        "bbox": [int(v) for v in bbox],
        "polygon": polygon or [],
        "mask_path": mask_path or "",
        "pose_6d": pose_6d,
        "visible_pixels": int(visible_pixels),
        "stl_path": stl_path,
        "state": state or "normal",
    }
    if score is not None:
        item["score"] = float(score)
    if pointcloud_path:
        item["pointcloud_path"] = pointcloud_path
    return item


def make_annotations(image, width, height, objects, camera, sidecar=None) -> dict:
    out = {
        "image": image,
        "width": int(width),
        "height": int(height),
        "objects": objects,
        "camera": camera or DEFAULT_CAMERA,
        "source": "yolo_seg",
    }
    if sidecar:
        for key in ("seed", "desk_stripe", "part_paths", "scene_layout", "n_parts", "manifest_round"):
            if key in sidecar:
                out[key] = sidecar[key]
    return out
