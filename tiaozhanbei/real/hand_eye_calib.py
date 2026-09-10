# -*- coding: utf-8 -*-
"""固定俯视 D405 的手眼标定（eye-to-hand）。

做法沿用 ``competition/manual_calib_er4ia.py``：在仿真场景里画出桌面和机械臂，
把相机实时点云按当前标定矩阵变到世界系一起显示，然后用键盘手动推，直到点云里
的桌面和机械臂和仿真模型重合。产物是一个 4x4 齐次矩阵，格式与那边一致::

    {"affine_mat": [[...4x4...]]}

和眼在手上的区别：相机固定不动，标定矩阵**就是** ``w2c_mat``（世界 ← 相机），
不需要每帧再乘 TCP 位姿，也不需要连机械臂。相机被碰动过就得重标。

点云画成相机的真实颜色（不是单色），桌沿、零件、机械臂本体都能分清，
比纯绿点云好对齐。加 ``--connect`` 会持续读真机关节角同步到仿真臂上，
手推机械臂时仿真跟着动，可以直接拿臂本体当对齐参照——这两点抄的是
``manual_calib_piper_outhand_camera.py``。

    python tiaozhanbei/real/hand_eye_calib.py
    python tiaozhanbei/real/hand_eye_calib.py --connect

按键（和 competition 那份一致，另加几个）：

===========  ====================================================
w / s        相机沿世界 +X / -X 平移
a / d        相机沿世界 +Y / -Y 平移
q / e        相机沿世界 +Z / -Z 平移
z / x        绕世界 X 轴转（绕相机自身位置，不改位置）
c / v        绕世界 Y 轴转
b / n        绕世界 Z 轴转
f            自动贴桌面：拟合点云主平面 → 摆平并压到 z=0
1 / 2        平移步长 减半 / 加倍
3 / 4        旋转步长 减半 / 加倍
p            保存到 handeye_d405.json
r            回到进入时的矩阵
0            仿真臂摆回配置 / --conf-deg 那组默认角
[ / ]        选中上一 / 下一关节
- / =        选中关节 -1° / +1°（5 / 6 改步长）
8            打印当前仿真关节角（可抄回 real_config.CALIB_REST_CONF_DEG）
9            恢复跟随真机关节角（仅 --connect）
===========  ====================================================

先用 ``f`` 把桌面摆平压到 z=0（高度和倾斜最难用眼睛调），再用 wasd 平移对齐
机械臂底座，最后 ``b/n`` 转一下偏航，然后 ``p`` 保存。

每次微调都会自动覆盖写 ``handeye_d405.json``（``--no-autosave`` 可关掉），
所以中途关窗口也不会丢进度；``r`` 只回到**进入时**的矩阵，不是回到上次保存。

注意：本文件是 ``tiaozhanbei/real`` 里唯一需要 WRS / Panda3D 的脚本
（要画机械臂当对齐参照）。真机执行链路本身不依赖它。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from real_config import CALIB_REST_CONF, CALIB_REST_CONF_DEG, REPO_ROOT, SIM_JAW_OPEN_MAX
from vision_config import (
    CAMERA_RESOLUTION,
    CAMERA_SERIAL,
    HANDEYE_JSON,
    MAX_DEPTH_M,
    WORKSPACE_BOUNDS_M,
    describe_handeye,
    load_handeye,
    points_in_workspace,
    save_handeye,
    transform_points,
    valid_points,
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MOVE_STEP_M = 0.002
ROT_STEP_RAD = np.radians(1.0)
PCD_STRIDE = 4          # 点云抽稀，全量画会卡
DESK_BAND_M = 0.05      # 拟合桌面时，取 z 直方图峰值上下这个范围内的点
RBT_SYNC_S = 0.2        # --connect 时同步真机关节角的周期（秒）
JOINT_STEP_DEG = 1.0    # 标定窗口里逐轴微调的默认步长


def open_sim_gripper(robot, jaw_width: float) -> float:
    """把仿真夹爪张到 ``jaw_width``（米），返回实际生效的开口。

    ``PantheraHTSglArm`` 建出来的 ``jaw_range`` 只有 ``[0, 0.012]``，直接设更大的
    开口会报 out of range，所以和仿真脚本一样先把量程放开。
    """
    ee = robot.end_effector
    jaw_width = float(np.clip(jaw_width, 0.0, SIM_JAW_OPEN_MAX))
    jaw_max = max(jaw_width, 0.012)
    ee.jaw_range = np.array([0.0, jaw_max], dtype=float)
    ee.close_bias = 0.0
    ee._command_max = jaw_max
    if hasattr(ee, "jlc") and len(ee.jlc.jnts) > 1:
        ee.jlc.jnts[0].motion_range = np.array([0.0, jaw_max / 2.0], dtype=float)
        ee.jlc.jnts[1].motion_range = np.array([0.0, jaw_max], dtype=float)
    if hasattr(ee, "_calibrate_opening_direction"):
        ee._calibrate_opening_direction()
    return jaw_width


def _fmt_conf_deg(conf) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in np.degrees(conf)) + "]°"


def _goto_sim_conf(robot, conf, jaw_width: float, log=print):
    """把仿真臂摆到指定关节角。某个轴超出模型限位时夹到边界再试一次。"""
    conf = np.asarray(conf, dtype=float).reshape(-1)
    try:
        robot.goto_given_conf(conf, ee_values=jaw_width)
        return conf
    except Exception as e:
        clipped = conf.copy()
        jnts = getattr(getattr(robot, "arm", None), "jnts", None) or []
        for i, jnt in enumerate(jnts[: len(clipped)]):
            lo, hi = getattr(jnt, "motion_range", (None, None))
            if lo is None:
                continue
            if clipped[i] < lo:
                clipped[i] = float(lo)
            elif clipped[i] > hi:
                clipped[i] = float(hi)
        if np.allclose(clipped, conf):
            log(f"[calib] 仿真臂摆位失败：{e}")
            return None
        try:
            robot.goto_given_conf(clipped, ee_values=jaw_width)
            log(f"[calib] 部分关节超出模型限位，已夹到 {_fmt_conf_deg(clipped)}")
            return clipped
        except Exception:
            log(f"[calib] 仿真臂摆位失败：{e}")
            return None


def rotmat_from_axangle(axis, angle: float) -> np.ndarray:
    """罗德里格斯公式；避免为了一个函数去 import WRS 的数学库。"""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    kx, ky, kz = axis / n
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=float)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def fit_desk_plane(points_world: np.ndarray):
    """在世界系点云里找桌面主平面，返回 ``(法向, 平面上一点)``。

    先按 z 直方图取最密的一层当桌面候选（桌面是画面里最大的平面），
    再对这层点做最小二乘平面拟合，比直接对全部点拟合稳得多。
    """
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    if len(pts) < 200:
        return None, None
    hist, edges = np.histogram(pts[:, 2], bins=80)
    peak = int(np.argmax(hist))
    z_peak = 0.5 * (edges[peak] + edges[peak + 1])
    band = pts[np.abs(pts[:, 2] - z_peak) < DESK_BAND_M]
    if len(band) < 200:
        return None, None
    centroid = band.mean(axis=0)
    # 协方差最小奇异向量就是平面法向
    _u, _s, vh = np.linalg.svd(band - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, centroid


class FixedOverheadCalib:
    """把 w2c 矩阵和 Panda3D 交互绑在一起。"""

    def __init__(
        self,
        base,
        robot,
        camera,
        init_mat,
        *,
        out_path: str,
        logger=print,
        joint_source=None,
        jaw_width: float = 0.0,
        autosave: bool = True,
        rest_conf=None,
    ) -> None:
        self.base = base
        self.robot = robot
        self.camera = camera
        self.mat = np.asarray(init_mat, dtype=float).copy()
        self.mat0 = self.mat.copy()
        self.out_path = out_path
        self.log = logger
        # 有真机时持续读关节角，对齐时看到的是实际姿态而不是冻住的 HOME
        self.joint_source = joint_source
        self.jaw_width = float(jaw_width)
        self.autosave = bool(autosave)
        self.rest_conf = np.asarray(
            rest_conf if rest_conf is not None else CALIB_REST_CONF, dtype=float
        )
        self.move_step = MOVE_STEP_M
        self.rot_step = ROT_STEP_RAD
        self.joint_step = np.radians(JOINT_STEP_DEG)
        self._joint_idx = 0
        self._node_pcd = None
        self._node_rbt = None
        self._pcd_cam = None
        self._pcd_rgb = None
        self._last_conf = None
        self._sync_paused = False
        self._dirty = True

        self._bind_keys()
        self._redraw_robot()  # 先把臂画出来，不用等第一帧点云
        base.taskMgr.doMethodLater(0.5, self._task_grab, "grab frame")
        base.taskMgr.add(self._task_adjust, "adjust calib")
        if self.joint_source is not None:
            base.taskMgr.doMethodLater(RBT_SYNC_S, self._task_sync_rbt, "sync rbt")

    # -- 按键 -------------------------------------------------------------
    def _bind_keys(self) -> None:
        self._held: dict[str, bool] = {}
        held_keys = "wsadqezxcvbn"
        for key in held_keys:
            self._held[key] = False
            self.base.inputmgr.keymap.setdefault(key, False)
            self.base.inputmgr.accept(key, self._set_key, [key, True])
            self.base.inputmgr.accept(f"{key}-up", self._set_key, [key, False])
        # 一次性动作用 accept，不需要按住
        self.base.inputmgr.accept("f", self._seat_to_desk)
        self.base.inputmgr.accept("p", self._save)
        self.base.inputmgr.accept("r", self._reset)
        self.base.inputmgr.accept("1", self._scale_move, [0.5])
        self.base.inputmgr.accept("2", self._scale_move, [2.0])
        self.base.inputmgr.accept("3", self._scale_rot, [0.5])
        self.base.inputmgr.accept("4", self._scale_rot, [2.0])
        self.base.inputmgr.accept("0", self._snap_zero)
        self.base.inputmgr.accept("9", self._resume_sync)
        self.base.inputmgr.accept("8", self._print_sim_conf)
        self.base.inputmgr.accept("5", self._scale_joint, [0.5])
        self.base.inputmgr.accept("6", self._scale_joint, [2.0])
        for key in ("[", "bracketleft"):
            self.base.inputmgr.accept(key, self._select_joint, [-1])
        for key in ("]", "bracketright"):
            self.base.inputmgr.accept(key, self._select_joint, [1])
        for key in ("-", "minus"):
            self.base.inputmgr.accept(key, self._nudge_joint, [-1])
        for key in ("=", "equal"):
            self.base.inputmgr.accept(key, self._nudge_joint, [1])

    def _set_key(self, key: str, down: bool) -> None:
        self._held[key] = down

    def _scale_move(self, factor: float) -> None:
        self.move_step = float(np.clip(self.move_step * factor, 1e-4, 0.05))
        self.log(f"[calib] 平移步长 {self.move_step * 1000:.2f}mm")

    def _scale_rot(self, factor: float) -> None:
        self.rot_step = float(np.clip(self.rot_step * factor, np.radians(0.05), np.radians(15)))
        self.log(f"[calib] 旋转步长 {np.degrees(self.rot_step):.2f}°")

    def _reset(self) -> None:
        self.mat = self.mat0.copy()
        self._dirty = True
        # 复位也落盘，保证文件里的矩阵始终等于画面上看到的
        if self.autosave:
            save_handeye(self.mat, self.out_path)
        self.log("[calib] 已回到进入时的矩阵（文件同步回退）")

    def _current_sim_conf(self):
        if self._last_conf is not None:
            return np.asarray(self._last_conf, dtype=float)
        getter = getattr(self.robot, "get_jnt_values", None)
        if callable(getter):
            return np.asarray(getter(), dtype=float)
        return np.asarray(self.rest_conf, dtype=float)

    def _apply_sim_conf(self, conf, *, note: str) -> None:
        applied = _goto_sim_conf(self.robot, conf, self.jaw_width, self.log)
        if applied is None:
            return
        self._last_conf = np.asarray(applied, dtype=float)
        self._sync_paused = True
        self._redraw_robot()
        extra = "（已暂停跟随真机，按 9 恢复）" if self.joint_source else ""
        self.log(f"[calib] {note} {_fmt_conf_deg(applied)}{extra}")

    def _snap_zero(self) -> None:
        """仿真臂摆回配置 / --conf-deg 那组角。不驱动真机。"""
        self._apply_sim_conf(self.rest_conf, note="仿真臂已摆回默认角")

    def _print_sim_conf(self) -> None:
        conf = self._current_sim_conf()
        deg = tuple(round(float(v), 2) for v in np.degrees(conf))
        self.log(f"[calib] 当前仿真关节角 {_fmt_conf_deg(conf)}")
        self.log(f"[calib] 可抄到 real_config.CALIB_REST_CONF_DEG = {deg}")

    def _select_joint(self, delta: int) -> None:
        n = max(1, int(getattr(self.robot, "ndof", 6) or 6))
        self._joint_idx = (self._joint_idx + int(delta)) % n
        conf = self._current_sim_conf()
        ang = float(np.degrees(conf[self._joint_idx])) if self._joint_idx < len(conf) else 0.0
        self.log(
            f"[calib] 选中 J{self._joint_idx + 1}  "
            f"当前 {ang:.2f}°  步长 {np.degrees(self.joint_step):.2f}°"
        )

    def _nudge_joint(self, sign: int) -> None:
        conf = self._current_sim_conf()
        if self._joint_idx >= len(conf):
            return
        conf = conf.copy()
        conf[self._joint_idx] += sign * self.joint_step
        self._apply_sim_conf(
            conf, note=f"J{self._joint_idx + 1} {'+' if sign > 0 else '-'}{np.degrees(self.joint_step):.2f}° →"
        )

    def _scale_joint(self, factor: float) -> None:
        self.joint_step = float(np.clip(self.joint_step * factor, np.radians(0.05), np.radians(15)))
        self.log(f"[calib] 关节步长 {np.degrees(self.joint_step):.2f}°")

    def _resume_sync(self) -> None:
        if self.joint_source is None:
            self.log("[calib] 没接真机，没有关节角可跟随")
            return
        self._sync_paused = False
        self._last_conf = None
        self.log("[calib] 恢复跟随真机关节角")

    def _save(self) -> None:
        path = save_handeye(self.mat, self.out_path)
        self.log(f"[calib] 已保存 -> {path}\n[calib] {describe_handeye(self.mat)}")

    # -- 调整 -------------------------------------------------------------
    def _translate(self, axis_world, amount: float) -> None:
        self.mat[:3, 3] = self.mat[:3, 3] + np.asarray(axis_world, dtype=float) * amount

    def _rotate(self, axis_world, angle: float) -> None:
        """绕世界轴旋转，但保持相机位置不动（只改朝向）。

        绕世界原点转会把相机甩得很远，手调时完全没法用。
        """
        dR = rotmat_from_axangle(axis_world, angle)
        self.mat[:3, :3] = dR @ self.mat[:3, :3]

    def _seat_to_desk(self) -> None:
        """把点云主平面摆平并压到 z=0。"""
        if self._pcd_cam is None:
            self.log("[calib] 还没拿到点云")
            return
        world = transform_points(self.mat, valid_points(self._pcd_cam))
        normal, centroid = fit_desk_plane(world)
        if normal is None:
            self.log("[calib] 点云太少或没有明显平面，贴桌面失败")
            return
        # 让平面法向转到 +Z：绕 normal×z 轴转两者夹角
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(normal, z)
        angle = float(np.arccos(np.clip(float(np.dot(normal, z)), -1.0, 1.0)))
        if np.linalg.norm(axis) > 1e-9 and abs(angle) > 1e-6:
            dR = rotmat_from_axangle(axis, angle)
            # 绕桌面质心转，避免把相机位置甩飞
            self.mat[:3, :3] = dR @ self.mat[:3, :3]
            self.mat[:3, 3] = centroid + dR @ (self.mat[:3, 3] - centroid)
        # 摆平之后高度变了，重新量一次再压到 z=0
        world = transform_points(self.mat, valid_points(self._pcd_cam))
        _n2, c2 = fit_desk_plane(world)
        dz = float(c2[2] if c2 is not None else centroid[2])
        self.mat[2, 3] -= dz
        self._dirty = True
        if self.autosave:
            save_handeye(self.mat, self.out_path)
        self.log(
            f"[calib] 贴桌面：摆平 {np.degrees(angle):.2f}°，下压 {dz * 1000:+.1f}mm。"
            "角度没降到 0 就再按几次 f，然后用 wasd 对齐机械臂底座。"
        )

    def _task_adjust(self, task):
        moved = False
        axes = {
            "w": ((1, 0, 0), 1), "s": ((1, 0, 0), -1),
            "a": ((0, 1, 0), 1), "d": ((0, 1, 0), -1),
            "q": ((0, 0, 1), 1), "e": ((0, 0, 1), -1),
        }
        for key, (axis, sign) in axes.items():
            if self._held.get(key):
                self._translate(axis, sign * self.move_step)
                moved = True
        rots = {
            "z": ((1, 0, 0), 1), "x": ((1, 0, 0), -1),
            "c": ((0, 1, 0), 1), "v": ((0, 1, 0), -1),
            "b": ((0, 0, 1), 1), "n": ((0, 0, 1), -1),
        }
        for key, (axis, sign) in rots.items():
            if self._held.get(key):
                self._rotate(axis, sign * self.rot_step)
                moved = True
        if moved:
            self._dirty = True
            # Piper 那份每次微调都落盘，中途崩了也不丢进度
            if self.autosave:
                save_handeye(self.mat, self.out_path)
        return task.again

    def _task_sync_rbt(self, task):
        """把真机当前关节角同步到仿真臂。"""
        if self._sync_paused:
            return task.again
        try:
            conf = np.asarray(self.joint_source(), dtype=float)
        except Exception as e:
            self.log(f"[calib] 读真机关节角失败：{e}")
            return task.again
        # 手调标定时机械臂基本不动，重建整套网格很贵，位形没变就别重画
        if self._last_conf is not None and np.allclose(conf, self._last_conf, atol=1e-4):
            return task.again
        try:
            self.robot.goto_given_conf(conf, ee_values=self.jaw_width)
        except Exception as e:
            self.log(f"[calib] 同步仿真臂失败：{e}")
            return task.again
        self._last_conf = conf
        # 位形变了要重画网格。必须就地摘掉再挂回来：这个任务比取帧任务跑得勤，
        # 只置脏标志等取帧去重画的话，机械臂大部分时间是不显示的。
        self._redraw_robot()
        return task.again

    # -- 显示 -------------------------------------------------------------
    def _task_grab(self, task):
        try:
            frame = self.camera.capture()
            self._pcd_cam = frame.points.reshape(-1, 3)[:: PCD_STRIDE]
            # BGR → RGB 归一化，画成彩色点云；纯绿点云看不出桌沿和零件
            self._pcd_rgb = (
                frame.color.reshape(-1, 3)[:: PCD_STRIDE][:, ::-1].astype(float) / 255.0
            )
            self._dirty = True
        except Exception as e:
            self.log(f"[calib] 取帧失败：{e}")
        if self._dirty:
            self._redraw()
            self._dirty = False
        return task.again

    def _redraw_robot(self) -> None:
        if self._node_rbt is not None:
            self._node_rbt.detach()
        self._node_rbt = self.robot.gen_meshmodel(alpha=0.55)
        self._node_rbt.attach_to(self.base)

    def _keep_mask(self, pcd_cam: np.ndarray) -> np.ndarray:
        """相机系点 → 布尔掩码：有效深度 且 变到世界系后落在显示范围内。

        显示范围比工作区放宽，便于看到桌面边缘当对齐参照。
        """
        pts = np.asarray(pcd_cam, dtype=float).reshape(-1, 3)
        dist = np.linalg.norm(pts, axis=1)
        keep = np.isfinite(pts).all(axis=1) & (dist > 1e-6) & (dist < MAX_DEPTH_M)
        if not keep.any():
            return keep
        world = transform_points(self.mat, pts)
        shown_bounds = {
            "x": (WORKSPACE_BOUNDS_M["x"][0] - 0.35, WORKSPACE_BOUNDS_M["x"][1] + 0.35),
            "y": (WORKSPACE_BOUNDS_M["y"][0] - 0.35, WORKSPACE_BOUNDS_M["y"][1] + 0.35),
            "z": (-0.15, 0.60),
        }
        for axis, name in enumerate(("x", "y", "z")):
            lo, hi = shown_bounds[name]
            keep &= (world[:, axis] >= lo) & (world[:, axis] <= hi)
        return keep

    def _redraw(self) -> None:
        import wrs.modeling.geometric_model as mgm

        if self._node_pcd is not None:
            self._node_pcd.detach()
            self._node_pcd = None
        if self._node_rbt is None:
            self._redraw_robot()
        if self._pcd_cam is None:
            return
        # 用掩码而不是逐步过滤，才能让颜色和点一一对应
        keep = self._keep_mask(self._pcd_cam)
        if not keep.any():
            return
        shown = transform_points(self.mat, self._pcd_cam)[keep]
        rgba = np.array([0.25, 0.85, 0.35, 1.0])
        if self._pcd_rgb is not None and len(self._pcd_rgb) == len(keep):
            rgb = self._pcd_rgb[keep]
            rgba = np.hstack((rgb, np.ones((len(rgb), 1))))
        self._node_pcd = mgm.gen_pointcloud(shown, rgba=rgba)
        mgm.gen_frame(self.mat[:3, 3], self.mat[:3, :3], ax_length=0.08).attach_to(self._node_pcd)
        self._node_pcd.attach_to(self.base)


def initial_guess(height_m: float = 0.60, x_m: float = 0.35) -> np.ndarray:
    """没有旧标定文件时的初值：相机悬在桌面上方、镜头朝下。

    RealSense 光学系是 +X 右、+Y 下、+Z 朝前（朝被拍物体）。相机朝下俯视时，
    相机 +Z 对应世界 -Z，相机 +Y 对应世界 +X（画面下方朝机器人前方）。
    这个初值只是让点云一开始就大致落在桌面附近，剩下的靠 f / wasd 调。
    """
    mat = np.eye(4)
    mat[:3, :3] = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    mat[:3, 3] = np.array([x_m, 0.0, height_m], dtype=float)
    return mat


def main() -> int:
    p = argparse.ArgumentParser(description="固定俯视 D405 手眼标定")
    p.add_argument("--out", default=HANDEYE_JSON)
    p.add_argument("--resolution", default=CAMERA_RESOLUTION, choices=("mid", "high"))
    p.add_argument("--serial", default=CAMERA_SERIAL)
    p.add_argument("--frame-dir", default="", help="用存下的帧代替实时相机（离线试手感）")
    p.add_argument("--height", type=float, default=0.60, help="初值：相机大致高度（米）")
    p.add_argument("--x", type=float, default=0.35, help="初值：相机大致 X（米）")
    p.add_argument(
        "--connect",
        action="store_true",
        help="连真机读当前关节角来显示实际位形；默认用 CALIB_REST_CONF_DEG / --conf-deg",
    )
    p.add_argument(
        "--conf-deg",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=None,
        help="仿真臂关节角（度）。不写则用 real_config.CALIB_REST_CONF_DEG",
    )
    p.add_argument(
        "--jaw-width",
        type=float,
        default=SIM_JAW_OPEN_MAX,
        help=f"仿真夹爪显示开口（米），默认张到最大 {SIM_JAW_OPEN_MAX}",
    )
    p.add_argument(
        "--no-autosave",
        action="store_true",
        help="关掉每次微调自动落盘，只在按 p 时保存",
    )
    args = p.parse_args()

    from d405_camera import CameraError, D405Camera, load_frame

    import wrs.visualization.panda.world as wd
    from tiaozhanbei.sim.environment import attach_desk_stripe, gen_desk
    from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm

    if os.path.isfile(args.out):
        init_mat = load_handeye(args.out)
        print(f"[calib] 已加载现有标定 {args.out}")
        print(f"[calib] {describe_handeye(init_mat)}")
    else:
        init_mat = initial_guess(args.height, args.x)
        print(f"[calib] 没有 {args.out}，用俯视初值起步（相机高 {args.height:.2f}m）")

    # 静态帧模式：不连相机，只是试试按键和显示
    class _StaticCam:
        def __init__(self, frame):
            self._frame = frame

        def capture(self):
            return self._frame

    if args.frame_dir:
        cam = _StaticCam(load_frame(args.frame_dir))
        print(f"[calib] 离线模式：用 {args.frame_dir} 里的帧")
        closer = None
    else:
        try:
            cam = D405Camera(args.resolution, args.serial)
            cam.open()
        except CameraError as e:
            print(f"[calib] 相机打不开：{e}")
            print("[calib] 想先离线试按键，可以加 --frame-dir <存过的帧目录>")
            return 1
        closer = cam

    base = wd.World(cam_pos=np.array([1.6, -1.2, 1.0]), lookat_pos=np.array([0.3, 0.0, 0.05]))
    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)
    attach_desk_stripe(base, style="vert_gray", seed=0)

    import wrs.modeling.geometric_model as mgm

    mgm.gen_frame(ax_length=0.20).attach_to(base)

    robot = PantheraHTSglArm(enable_cc=False)
    rest_conf = (
        np.radians(np.asarray(args.conf_deg, dtype=float))
        if args.conf_deg is not None
        else np.asarray(CALIB_REST_CONF, dtype=float)
    )
    arm = None
    joint_source = None
    if args.connect:
        from real_arm import RealArm, RealArmError

        try:
            arm = RealArm(dry_run=False, port=None)
            arm.connect()
            joint_source = arm.joint_values
            print(f"[calib] 真机当前位形(°) {np.degrees(arm.joint_values()).round(1).tolist()}")
            print("[calib] 已接真机：手推机械臂时仿真臂会跟着动，可以拿它当对齐参照")
        except (RealArmError, RuntimeError) as e:
            print(f"[calib] 连真机失败，退回配置里的仿真角：{e}")
            arm = None
    # 夹爪张开显示：合爪时两指并在一起，很难看出末端到底在哪，对齐时没有参照
    jaw_width = open_sim_gripper(robot, args.jaw_width)
    if joint_source is not None:
        robot.goto_given_conf(np.asarray(joint_source(), dtype=float), ee_values=jaw_width)
    else:
        applied = _goto_sim_conf(robot, rest_conf, jaw_width)
        src = "--conf-deg" if args.conf_deg is not None else "real_config.CALIB_REST_CONF_DEG"
        print(f"[calib] 仿真臂 {_fmt_conf_deg(applied if applied is not None else rest_conf)}（来自 {src}）")
        print("[calib] 改角度：改配置 / --conf-deg / 窗口里 [ ] 选轴、- = 微调，8 打印当前角")
    try:
        from real_config import GripperMap, load_real_config

        gmap = GripperMap.from_config(load_real_config())
        print(
            f"[calib] 夹爪开口 {jaw_width * 1000:.0f}mm"
            f"（真机约 {np.degrees(gmap.width_to_angle(jaw_width)):.0f}°）"
        )
    except Exception as e:  # 读不到 robot.cfg 不该拦住标定
        print(f"[calib] 夹爪开口 {jaw_width * 1000:.0f}mm（换算真机角度失败：{e}）")

    print(
        "[calib] 操作：f 自动贴桌面 → wasd/qe 平移 → zxcvbn 旋转 → p 保存\n"
        "[calib] 1/2 调平移步长，3/4 调旋转步长，[ ] 选关节，- = 调关节，5/6 关节步长，8 打印关节角，0 回到默认角，9 跟随真机。"
        "先让点云里的桌面和仿真桌面重合，再对齐机械臂底座。"
    )
    if not args.no_autosave:
        print(f"[calib] 每次微调都会自动写入 {os.path.abspath(args.out)}（p 仍可手动存）")
    FixedOverheadCalib(
        base,
        robot,
        cam,
        init_mat,
        out_path=args.out,
        joint_source=joint_source,
        jaw_width=jaw_width,
        autosave=not args.no_autosave,
        rest_conf=rest_conf,
    )
    try:
        base.run()
    finally:
        if closer is not None:
            closer.close()
        if arm is not None:
            try:
                arm.close()
            except Exception as e:
                print(f"[calib] 断开真机失败：{e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
