"""
基于 WRS 点云配准的物体位姿估计：观测深度点云 ↔ STL 模板 ICP。

默认流程：
  1) 粗位姿：点云中位数位置 + 桌面 PCA yaw（初值）
  2) 精配准：registration_icp_ptpt(模板 → 观测)，可选 180° yaw 歧义试探
  3) 失败则回退粗位姿

模板坐标系与 tiaozhanbei111.environment.load_part_model 一致：
  mm→m 缩放、XY 中心归零、底面贴 z=0（不含摆放 yaw）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_WRS = _ROOT / "wrs-main"
if _WRS.is_dir() and str(_WRS) not in sys.path:
    sys.path.insert(0, str(_WRS))

_TEMPLATE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _subsample(points: np.ndarray, n: int) -> np.ndarray:
    if points is None or len(points) == 0:
        return points
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[idx]


def load_canonical_template_points(
    stl_path: str | Path,
    scale: float = 0.001,
    n_points: int = 900,
) -> np.ndarray:
    """加载 STL → 与仿真摆放一致的规范局部点云（未施加 yaw、未平移到桌面 xy）。"""
    stl_path = Path(stl_path)
    key = (str(stl_path.resolve()), int(n_points))
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]

    # 直接用 wrs.basis.trimesh，避免 import data_adapter 拉起 Panda3D
    import wrs.basis.trimesh as trm

    mesh = trm.load(str(stl_path))
    mesh.apply_scale(np.array([scale, scale, scale], dtype=float))
    bounds = mesh.bounds
    xy_center = (bounds[0, :2] + bounds[1, :2]) / 2.0
    mesh.apply_translation(np.array([-xy_center[0], -xy_center[1], 0.0]))
    z_min = mesh.bounds[0, 2]
    mesh.apply_translation(np.array([0.0, 0.0, -z_min]))

    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 20:
        raise RuntimeError(f"模板点过少: {stl_path}")
    pts = _subsample(pts, n_points)
    _TEMPLATE_CACHE[key] = pts
    return pts


def rotmat_yaw(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def homomat_from_pos_rot(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def refine_pose_icp(
    obs_world: np.ndarray,
    template_local: np.ndarray,
    init_pos: np.ndarray,
    init_rot: np.ndarray,
    maxcorrdist: float = 0.012,
    try_yaw_flip: bool = True,
    max_obs_points: int = 700,
) -> tuple[np.ndarray | None, float]:
    """
    ICP：模板(src) → 观测(tgt)。返回 (4x4 世界位姿, rmse)；失败则 (None, inf)。
    """
    from wrs.vision.depth_camera.util_functions import registration_icp_ptpt

    obs = _subsample(np.asarray(obs_world, dtype=np.float64), max_obs_points)
    tpl = np.asarray(template_local, dtype=np.float64)
    if obs is None or len(obs) < 30 or len(tpl) < 30:
        return None, float("inf")

    candidates = [init_rot]
    if try_yaw_flip:
        candidates.append(init_rot @ rotmat_yaw(np.pi))

    best_T, best_rmse = None, float("inf")
    for rot in candidates:
        init = homomat_from_pos_rot(init_pos, rot)
        try:
            rmse, T = registration_icp_ptpt(
                tpl, obs, inithomomat=init, maxcorrdist=maxcorrdist, toggledebug=False
            )
        except Exception:
            continue
        rmse = float(rmse) if rmse is not None else float("inf")
        if not np.isfinite(rmse):
            continue
        if rmse < best_rmse:
            best_rmse, best_T = rmse, np.asarray(T, dtype=np.float64)

    # 合理 RMSE 阈值（米）：过大说明没对齐上
    if best_T is None or best_rmse > 0.025:
        return None, best_rmse
    return best_T, best_rmse


def refine_pose_icp_global(
    obs_world: np.ndarray,
    template_local: np.ndarray,
    voxel: float = 0.004,
    max_obs_points: int = 900,
) -> tuple[np.ndarray | None, float]:
    """较慢：FGR + ICP（registration_ptpt）。"""
    from wrs.vision.depth_camera.util_functions import registration_ptpt

    obs = _subsample(np.asarray(obs_world, dtype=np.float64), max_obs_points)
    tpl = np.asarray(template_local, dtype=np.float64)
    if obs is None or len(obs) < 40 or len(tpl) < 40:
        return None, float("inf")
    try:
        rmse, T = registration_ptpt(tpl, obs, downsampling_voxelsize=voxel, toggledebug=False)
        rmse = float(rmse) if rmse is not None else float("inf")
        if not np.isfinite(rmse) or rmse > 0.03:
            return None, rmse
        return np.asarray(T, dtype=np.float64), rmse
    except Exception:
        return None, float("inf")
