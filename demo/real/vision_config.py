# -*- coding: utf-8 -*-
"""视觉侧配置：D405 相机、YOLO 权重、工作区范围、手眼矩阵、类名映射。

相机是**固定俯视**（eye-to-hand）：相机架在桌子上方不动，不跟着机械臂走。
所以手眼标定的产物就是一个固定的 ``w2c_mat``（世界 ← 相机），每帧直接用::

    pcd_world = w2c_mat @ pcd_camera

这比眼在手上简单：不需要每帧读关节角再乘 TCP 位姿，标一次就固定了。
代价是相机一旦被碰动，必须重标。

``real_config.py`` 管机械臂，本文件管相机和识别，两边不互相依赖。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from real_config import REAL_DIR, TB_DIR

# ---------------------------------------------------------------------------
# 相机
# ---------------------------------------------------------------------------
CAMERA_RESOLUTION = "mid"        # RealSenseD405 的 'mid'(848x480) / 'high'(1280x720)
CAMERA_SERIAL: str | None = None  # 多台相机时填序列号，单台留 None 自动选
CAMERA_WARMUP_FRAMES = 20        # 开机暗帧，30fps 大约 0.7s
CAMERA_AE_SETTLE_WINDOW = 8      # 亮度窗口：连续这么多帧变化够小才算曝光稳
CAMERA_AE_SETTLE_LUMA = 3.0      # 窗口内平均亮度 max-min 小于它（0~255）
CAMERA_AE_SETTLE_MAX_FRAMES = 45 # 曝光最多再等这么多帧，超时就用当前画面

# ---------------------------------------------------------------------------
# 手眼标定产物（固定俯视：单个 world ← camera 齐次矩阵）
# ---------------------------------------------------------------------------
HANDEYE_JSON = os.path.join(REAL_DIR, "handeye_d405.json")

# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
# 真实零件权重放在本目录；排在前面的优先。
WEIGHT_CANDIDATES = ("real_multi.pt", "best(3).pt", "best.pt")
YOLO_CONF = 0.35
YOLO_IOU = 0.50
YOLO_IMGSZ = 960                 # 与实时预览一致；默认 640 会漏掉近景小件
YOLO_VOTE_FRAMES = 5             # 曝光稳住后再对这么多彩图分别推理
YOLO_VOTE_MIN = 2                # 同一物体至少在这么多帧出现才算数（压掉闪一下的误检）
YOLO_VOTE_IOU = 0.30             # 跨帧认为是同一物体的 bbox IoU 门槛
# 扳手这类黑件在 960 上经常 <0.35。投票漏掉的，再用更大输入补一轮。
YOLO_RECALL_IMGSZ = 1280
YOLO_RECALL_CONF = 0.20
# 补召回框若和已有框太近，多半是把螺丝刀又认成扳手，不要当第二件。
YOLO_RECALL_MAX_IOU = 0.12
YOLO_RECALL_MIN_CENTER_PX = 55
# 世界系中心近于这个距离视为同一堆，只留分高的。仿真叠在一起规划必炸。
MIN_OBJECT_SEP_M = 0.06
CLASS_MAP_JSON = os.path.join(REAL_DIR, "yolo_class_map.json")

# ---------------------------------------------------------------------------
# 工作区过滤（世界坐标，米）
# ---------------------------------------------------------------------------
# 桌面在 z=0，机器人底座在原点。落在这个盒子外面的点一律丢掉：
# 相机会拍到桌子边缘、地面、机械臂本体，不滤会把这些当成零件的一部分。
WORKSPACE_BOUNDS_M = {
    "x": (0.10, 0.62),
    "y": (-0.40, 0.40),
    "z": (-0.02, 0.30),
}
MIN_POINTS_PER_OBJECT = 40       # 有效点少于它就认为这个检测不可用
DESK_Z = 0.0                     # 桌面高度；仿真里零件都摆在 z=0
MAX_DEPTH_M = 1.5                # 超过这个距离的点当无效（D405 近距相机）

# 姿态判定阈值：用世界系点云的高度跨度 / 水平跨度比值区分正放与倒下
STATE_FALLEN_FLATNESS = 0.55     # z 跨度 < 0.55 * xy 跨度 → 视为倒下
STATE_MIN_Z_SPAN_M = 0.008       # z 跨度太小说明是薄片件，不判倒下


class VisionConfigError(RuntimeError):
    """权重缺失、手眼未标定、类名表缺失等。"""


# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------
def default_weights() -> str:
    """返回本目录里的默认权重路径。"""
    for name in WEIGHT_CANDIDATES:
        cand = os.path.join(REAL_DIR, name)
        if os.path.isfile(cand):
            return cand
    raise VisionConfigError(
        "在 tiaozhanbei/real 下找不到 YOLO 权重，试过："
        + "、".join(WEIGHT_CANDIDATES)
        + "\n请把真实零件权重放进本目录，或用 --weights 指定。"
    )


# ---------------------------------------------------------------------------
# 手眼矩阵：沿用 competition/manual_calib_er4ia.py 的 {"affine_mat": 4x4} 格式
# ---------------------------------------------------------------------------
def load_handeye(path: str | None = None) -> np.ndarray:
    path = path or HANDEYE_JSON
    if not os.path.isfile(path):
        raise VisionConfigError(
            f"手眼标定文件不存在：{path}\n"
            "请先跑 python tiaozhanbei/real/hand_eye_calib.py 标定固定俯视相机。"
        )
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    mat = np.asarray(data.get("affine_mat"), dtype=float)
    if mat.shape != (4, 4):
        raise VisionConfigError(f"{path} 里的 affine_mat 不是 4x4，而是 {mat.shape}")
    return mat


def save_handeye(mat, path: str | None = None) -> str:
    path = path or HANDEYE_JSON
    mat = np.asarray(mat, dtype=float)
    if mat.shape != (4, 4):
        raise VisionConfigError(f"affine_mat 必须是 4x4，收到 {mat.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"affine_mat": mat.tolist()}, f, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


# 固定俯视相机的光轴应该大致朝下，偏离超过这个角度基本可以断定标定文件不对
HANDEYE_TILT_WARN_DEG = 60.0


def describe_handeye(mat) -> str:
    mat = np.asarray(mat, dtype=float)
    pos = mat[:3, 3]
    z_axis = mat[:3, 2]
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(-z_axis[2])))))
    text = (
        f"相机位置 [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]m  "
        f"光轴偏离垂直向下 {tilt_deg:.1f}°"
    )
    if tilt_deg > HANDEYE_TILT_WARN_DEG:
        # 最常见的原因：把眼在手上的标定文件（competition 那份）当成这里的用了
        text += "  ← 这不像固定俯视相机，标定文件可能不对"
    return text


# ---------------------------------------------------------------------------
# 类名映射：YOLO 英文类 → 仿真中文类 + STL 绝对路径
# ---------------------------------------------------------------------------
class ClassMap:
    """由 ``build_class_map.py`` 生成的 ``yolo_class_map.json``。

    仿真靠 ``stl_path`` 加载碰撞网格来搜抓取位姿，所以这张表必须能给出一个
    真实存在的 STL。表里每类只挑了叶目录下的第一个 STL 当代表形状——同一类
    零件尺寸差别大的时候，抓取宽度会有偏差，这点在 README 里有说明。
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or CLASS_MAP_JSON
        if not os.path.isfile(self.path):
            raise VisionConfigError(
                f"类名映射表不存在：{self.path}\n"
                "请先跑 python tiaozhanbei/real/build_class_map.py 生成。"
            )
        with open(self.path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        self.classes: dict[str, dict] = data.get("classes") or {}
        self.unmapped: list[str] = data.get("unmapped") or []
        root = data.get("part_model_root") or os.path.join(TB_DIR, "Part_Model")
        self.part_root = os.path.abspath(root)
        if not self.classes:
            raise VisionConfigError(f"{self.path} 里 classes 为空，请重新生成")

    def resolve(self, en_class: str) -> dict | None:
        """英文类名 → ``{"zh", "stl_path", "approx"}``；表里没有或 STL 丢了返回 None。"""
        entry = self.classes.get(str(en_class))
        if not entry:
            return None
        stl_path = os.path.join(self.part_root, *entry["stl_dir"].split("/"), entry["stl"])
        if not os.path.isfile(stl_path):
            return None
        return {
            "zh": entry["zh"],
            "stl_path": os.path.abspath(stl_path),
            "approx": bool(entry.get("approx")),
        }

    def describe(self) -> str:
        approx = sum(1 for v in self.classes.values() if v.get("approx"))
        return (
            f"类名映射[{os.path.basename(self.path)}] {len(self.classes)} 类"
            + (f"（{approx} 类形状近似）" if approx else "")
            + (f"，未映射 {len(self.unmapped)} 类" if self.unmapped else "")
        )


# ---------------------------------------------------------------------------
# 点云工具（和 competition/yolo_utils.py 同一套命名与约定）
# ---------------------------------------------------------------------------
def transform_points(mat, points) -> np.ndarray:
    """把点从相机系变到世界系：``pcd_world = mat @ pcd_camera``。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    mat = np.asarray(mat, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"变换矩阵必须是 4x4，收到 {mat.shape}")
    homo = np.ones((len(pts), 4), dtype=float)
    homo[:, :3] = pts
    return (mat @ homo.T).T[:, :3]


def valid_points(points) -> np.ndarray:
    """丢掉 NaN、原点（D405 无效深度返回 0）和过远的点。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(pts):
        return pts
    dist = np.linalg.norm(pts, axis=1)
    keep = np.isfinite(pts).all(axis=1) & (dist > 1e-6) & (dist < MAX_DEPTH_M)
    return pts[keep]


def points_in_workspace(points_world, bounds=None) -> np.ndarray:
    bounds = bounds or WORKSPACE_BOUNDS_M
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    if not len(pts):
        return pts
    keep = np.ones(len(pts), dtype=bool)
    for axis, name in enumerate(("x", "y", "z")):
        lo, hi = bounds[name]
        keep &= (pts[:, axis] >= lo) & (pts[:, axis] <= hi)
    return pts[keep]
