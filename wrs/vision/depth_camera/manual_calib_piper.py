#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/16 16:18
# @Author : ZhangXi
# -*- coding: utf-8 -*-
__VERSION__ = '0.0.3'

import os
from pathlib import Path
import json
from abc import ABC, abstractmethod
from direct.task.TaskManagerGlobal import taskMgr
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.robot_interface as ri
import numpy as np


def py2json_data_formatter(data):
    """将Python数据格式化为JSON兼容格式。仅支持np.ndarray, str, int, float, dict, list, Path。"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (str, float, int, dict)):
        return data
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    """通过JSON转储数据。"""
    path = str(path)
    if reminder and os.path.exists(path):
        option = input(f"文件 {path} 已存在。确定要覆盖写入吗？ (y/n): ")
        print(option)
        option_up = option.upper()
        if option_up not in ["Y", "YES"]:
            return False
    with open(path, "w", encoding='utf-8') as f:
        json.dump(py2json_data_formatter(data), f, ensure_ascii=False, indent=4)
    return True


class ManualCalibrationBase(ABC):
    """手动标定基类，定义了手动标定的核心流程和交互逻辑。"""

    def __init__(self, rbt_s: ri.RobotInterface, rbt_x, sensor_hdl, init_calib_mat: rm.np.ndarray = None,
                 component_name="arm", move_resolution=.001, rotation_resolution=rm.np.radians(5)):
        """
        初始化手动标定。
        :param rbt_s: 仿真机器人模型
        :param rbt_x: 真实机器人控制器
        :param sensor_hdl: 传感器（相机）控制器
        :param init_calib_mat: 初始标定矩阵。若为None，则默认为单位矩阵
        :param component_name: 相机安装的组件名称
        :param move_resolution: 手动平移调整的步长
        :param rotation_resolution: 手动旋转调整的步长
        """
        self._rbt_s = rbt_s
        self._rbt_x = rbt_x
        self._sensor_hdl = sensor_hdl
        self._init_calib_mat = np.eye(4) if init_calib_mat is None else init_calib_mat
        self._component_name = component_name

        # 用于存储机器人和点云的绘图节点
        self._plot_node_rbt = None
        self._plot_node_pcd = None
        self._pcd = None

        # 键盘映射
        self._key = {}
        self.map_key()
        self.move_resolution = move_resolution
        self.rotation_resolution = rotation_resolution

        # 添加异步任务
        taskMgr.doMethodLater(2, self.sync_rbt, "sync_rbt_task")
        taskMgr.add(self.adjust, "manual_adjust_task")
        taskMgr.doMethodLater(5, self.sync_pcd, "sync_pcd_task")

    @abstractmethod
    def get_pcd(self) -> np.ndarray:
        """获取点云数据的抽象方法。"""
        pass

    @abstractmethod
    def get_rbt_jnt_val(self) -> np.ndarray:
        """获取机器人关节角度的抽象方法。"""
        pass

    @abstractmethod
    def align_pcd(self, pcd) -> np.ndarray:
        """
        根据标定矩阵对齐点云的抽象方法。
        此处实现眼在手(Eye-in-Hand)或眼在手外(Eye-to-Hand)的坐标变换。
        """
        pass

    def move_adjust(self, dir, dir_global, key_name=None):
        """通过平移修正标定矩阵。"""
        if self._rbt_x is None: return
        rbt_pose = self._rbt_x.get_pose()
        w2r_mat = rm.homomat_from_posrot(*rbt_pose)
        r2w_mat = np.linalg.inv(w2r_mat)
        dir_in_robot_frame = r2w_mat[:3, :3].dot(dir_global)
        self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_in_robot_frame * self.move_resolution

    def rotate_adjust(self, dir, dir_global, key_name=None):
        """通过旋转修正标定矩阵。"""
        if self._rbt_x is None: return
        rbt_pose = self._rbt_x.get_pose()
        w2r_mat = rm.homomat_from_posrot(*rbt_pose)
        R_w_r = w2r_mat[:3, :3]
        delta_R_world = rm.rotmat_from_axangle(dir_global, self.rotation_resolution)
        R_r_w = R_w_r.T
        delta_R_robot = R_r_w @ delta_R_world @ R_w_r
        current_R_r_c = self._init_calib_mat[:3, :3]
        self._init_calib_mat[:3, :3] = np.dot(delta_R_robot, current_R_r_c)

    def map_key(self, x='w', x_='s', y='a', y_='d', z='q', z_='e', x_cw='z', x_ccw='x', y_cw='c', y_ccw='v', z_cw='b',
                z_ccw='n'):
        """映射键盘按键以进行手动调整。"""

        def add_key(keys: str or list):
            assert isinstance(keys, str) or isinstance(keys, list)
            if isinstance(keys, str):
                keys = [keys]

            def set_keys(base, k, v):
                base.inputmgr.keymap[k] = v

            for key in keys:
                if key in base.inputmgr.keymap: continue
                base.inputmgr.keymap[key] = False
                base.inputmgr.accept(key, set_keys, [base, key, True])
                base.inputmgr.accept(key + '-up', set_keys, [base, key, False])

        add_key([x, x_, y, y_, z, z_, x_cw, x_ccw, y_cw, y_ccw, z_cw, z_ccw])
        self._key.update({'x': x, 'x_': x_, 'y': y, 'y_': y_, 'z': z, 'z_': z_,
                          'x_cw': x_cw, 'x_ccw': x_ccw, 'y_cw': y_cw, 'y_ccw': y_ccw,
                          'z_cw': z_cw, 'z_ccw': z_ccw})

    def sync_pcd(self, task):
        """同步获取并更新点云。"""
        if self._sensor_hdl:
            self._pcd = self.get_pcd()
        self.plot()
        self.save()
        return task.again

    def sync_rbt(self, task):
        """同步真实机器人和仿真机器人的状态。"""
        # 仅当真实机器人控制器存在时才进行同步
        if self._rbt_x and self._rbt_x_target:
            rbt_jnt_val = self.get_rbt_jnt_val()
            self._rbt_s.fk(rbt_jnt_val, update=True)

            target_arm_jnt_val = self._rbt_x_target.get_joint_values()
            self._rbt_s_target.fk(target_arm_jnt_val, update=True)

        self.plot()
        return task.again

    def save(self):
        """保存手动标定结果。"""
        dump_json({'affine_mat': self._init_calib_mat.tolist()}, "mutual_look_calibration_result.json", reminder=False)

    def plot(self, task=None):
        """绘制机器人模型和点云。"""
        # 清理旧的绘制节点
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
        if hasattr(self, '_plot_node_rbt_target') and self._plot_node_rbt_target is not None:
            self._plot_node_rbt_target.detach()

        # 绘制机器人
        self._plot_node_rbt = self._rbt_s.gen_meshmodel(alpha=1)
        self._plot_node_rbt.attach_to(base)

        # 如果是双臂标定，也绘制目标臂
        if hasattr(self, '_rbt_s_target') and self._rbt_s_target is not None:
            self._plot_node_rbt_target = self._rbt_s_target.gen_meshmodel(alpha=1)
            self._plot_node_rbt_target.attach_to(base)

        # 绘制点云
        pcd = self._pcd
        if pcd is not None:
            pcd_points = pcd[:, :3]
            pcd_colors = pcd[:, 3:6] if pcd.shape[1] >= 6 else np.array([1, 1, 1])
            pcd_color_rgba = np.c_[pcd_colors, np.ones(len(pcd_colors))] if pcd_colors.ndim > 1 else np.array(
                [1, 1, 1, 1])

            pcd_aligned = self.align_pcd(pcd_points)
            self._plot_node_pcd = mgm.gen_pointcloud(pcd_aligned, rgba=pcd_color_rgba)
            mgm.gen_frame(self._init_calib_mat[:3, 3], self._init_calib_mat[:3, :3]).attach_to(self._plot_node_pcd)
            self._plot_node_pcd.attach_to(base)

        if task is not None:
            return task.again

    def adjust(self, task):
        """检查键盘输入并调整标定矩阵。"""
        was_adjusted = False
        keymap = base.inputmgr.keymap
        # 平移调整
        if keymap.get(self._key['x']): self.move_adjust(self._init_calib_mat[:3, 0],
                                                        np.array([1, 0, 0])); was_adjusted = True
        if keymap.get(self._key['x_']): self.move_adjust(-self._init_calib_mat[:3, 0],
                                                         np.array([-1, 0, 0])); was_adjusted = True
        if keymap.get(self._key['y']): self.move_adjust(self._init_calib_mat[:3, 1],
                                                        np.array([0, 1, 0])); was_adjusted = True
        if keymap.get(self._key['y_']): self.move_adjust(-self._init_calib_mat[:3, 1],
                                                         np.array([0, -1, 0])); was_adjusted = True
        if keymap.get(self._key['z']): self.move_adjust(self._init_calib_mat[:3, 2],
                                                        np.array([0, 0, 1])); was_adjusted = True
        if keymap.get(self._key['z_']): self.move_adjust(-self._init_calib_mat[:3, 2],
                                                         np.array([0, 0, -1])); was_adjusted = True
        # 旋转调整
        if keymap.get(self._key['x_cw']): self.rotate_adjust(self._init_calib_mat[:3, 0],
                                                             np.array([1, 0, 0])); was_adjusted = True
        if keymap.get(self._key['x_ccw']): self.rotate_adjust(-self._init_calib_mat[:3, 0],
                                                              np.array([-1, 0, 0])); was_adjusted = True
        if keymap.get(self._key['y_cw']): self.rotate_adjust(self._init_calib_mat[:3, 1],
                                                             np.array([0, 1, 0])); was_adjusted = True
        if keymap.get(self._key['y_ccw']): self.rotate_adjust(-self._init_calib_mat[:3, 1],
                                                              np.array([0, -1, 0])); was_adjusted = True
        if keymap.get(self._key['z_cw']): self.rotate_adjust(self._init_calib_mat[:3, 2],
                                                             np.array([0, 0, 1])); was_adjusted = True
        if keymap.get(self._key['z_ccw']): self.rotate_adjust(-self._init_calib_mat[:3, 2],
                                                              np.array([0, 0, -1])); was_adjusted = True

        if was_adjusted: self.plot()
        return task.again


class PiperMutualCalib(ManualCalibrationBase):
    """
    针对Piper双臂“互看”场景的专用手动标定类。
    - rbt_s/rbt_x: 指代搭载相机的机械臂（左臂）。
    - rbt_s_target/rbt_x_target: 指代被观察的机械臂（右臂）。
    """

    def __init__(self, rbt_s_cam, rbt_x_cam, rbt_s_target, rbt_x_target, **kwargs):
        super().__init__(rbt_s=rbt_s_cam, rbt_x=rbt_x_cam, **kwargs)
        self._rbt_s_target = rbt_s_target
        self._rbt_x_target = rbt_x_target
        self._plot_node_rbt_target = None

    def get_pcd(self):
        """从相机获取点云和颜色数据。"""
        pcd, pcd_color, _, _ = self._sensor_hdl.get_pcd_texture_depth()
        return np.hstack((pcd, pcd_color))

    def get_rbt_jnt_val(self):
        """获取搭载相机机械臂的关节角度。"""
        return self._rbt_x.get_joint_values()

    def align_pcd(self, pcd):
        """将点云从相机坐标系转换到世界坐标系。"""
        flange_to_camera_mat = self._init_calib_mat
        if self._rbt_x is None:  # 仿真模式下使用仿真模型姿态
            base_to_flange_mat = self._rbt_s.get_gl_tcp()
        else:  # 实体模式下使用真实手臂姿态
            base_to_flange_pose = self._rbt_x.get_pose()
            base_to_flange_mat = rm.homomat_from_posrot(*base_to_flange_pose)

        world_to_base_mat = self._rbt_s.get_pose()
        world_to_camera_mat = world_to_base_mat.dot(base_to_flange_mat).dot(flange_to_camera_mat)
        return rm.transform_points_by_homomat(world_to_camera_mat, points=pcd)

    def plot(self, task=None):
        """重写此方法以同时绘制两个机械臂和点云。"""
        # --- 清理旧节点 ---
        if self._plot_node_rbt is not None: self._plot_node_rbt.detach()
        if self._plot_node_rbt_target is not None: self._plot_node_rbt_target.detach()
        if self._plot_node_pcd is not None: self._plot_node_pcd.detach()

        # --- 绘制搭载相机的机械臂 (左臂)，设为半透明以作区分 ---
        self._plot_node_rbt = self._rbt_s.gen_meshmodel(alpha=0.5)
        self._plot_node_rbt.attach_to(base)

        # --- 绘制被观察的机械臂 (右臂)，不透明，作为对齐目标 ---
        self._plot_node_rbt_target = self._rbt_s_target.gen_meshmodel(alpha=1)
        self._plot_node_rbt_target.attach_to(base)

        # --- 绘制点云 ---
        if self._pcd is not None:
            pcd_points = self._pcd[:, :3]
            pcd_colors = self._pcd[:, 3:6] if self._pcd.shape[1] >= 6 else np.array([0.5, 0.5, 0.5])
            pcd_color_rgba = np.c_[pcd_colors, np.ones(len(pcd_colors))] if pcd_colors.ndim > 1 else np.array(
                [0.5, 0.5, 0.5, 1])

            pcd_aligned = self.align_pcd(pcd_points)
            self._plot_node_pcd = mgm.gen_pointcloud(pcd_aligned, rgba=pcd_color_rgba)
            self._plot_node_pcd.attach_to(base)

        if task is not None: return task.again


def load_calibration_matrix_from_json(filepath):
    """从JSON文件中加载标定矩阵。"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        mat_list = data.get('affine_mat')
        if mat_list is None: raise ValueError(f"JSON 文件 '{filepath}' 中没有找到 'affine_mat' 字段。")
        return np.array(mat_list)
    except FileNotFoundError:
        print(f"警告: 找不到标定文件 '{filepath}'。将使用默认的单位矩阵。")
        return np.eye(4)
    except Exception as e:
        print(f"加载标定文件时出错: {e}。将使用默认的单位矩阵。")
        return np.eye(4)


if __name__ == "__main__":
    import wrs.visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405
    from wrs.robot_sim.manipulators.piper.piper import Piper
    import platform
    # 确保从正确的路径导入 PiperArmController
    from wrs.robot_con.piper.piper import PiperArmController

    # --- 1. 初始化仿真环境 ---
    base = wd.World(cam_pos=[0, 2, 0], lookat_pos=[0, 0, 0.5], lens_type=2)
    mgm.gen_frame(ax_length=0.2).attach_to(base)

    # --- 2. 初始化真实机器人控制器 ---
    # ==================== 關鍵修改：模仿 piper.py 的連接邏輯 ====================
    # 自动根据系统判断 CAN 通道
    if platform.system() == "Windows":
        # 在 Windows 上，Piper SDK 通常使用数字索引作为 CAN 名称
        can_left = "0"
        can_right = "1"
        print("[INFO] 检测到 Windows 系统，使用 CAN 名称 '0' 和 '1'")
    else:
        # 在 Linux 上，通常使用 'canX' 格式
        can_left = "can0"
        can_right = "can1"
        print("[INFO] 检测到 Linux 系统，使用 CAN 名称 'can0' 和 'can1'")

    # 使用 try-except 结构，即使硬體連接失敗，程式也能以純仿真模式運行
    try:
        rbtx_left = PiperArmController(can_name=can_left, has_gripper=True)
        print(f"[INFO] 左臂 PiperArmController 已连接 ({can_left})")
    except Exception as e:
        print(f"[WARN] 左臂 CAN 接口不可用 ({can_left})，将不进行实体同步: {e}")
        rbtx_left = None

    try:
        rbtx_right = PiperArmController(can_name=can_right, has_gripper=True)
        print(f"[INFO] 右臂 PiperArmController 已连接 ({can_right})")
    except Exception as e:
        print(f"[WARN] 右臂 CAN 接口不可用 ({can_right})，将不进行实体同步: {e}")
        rbtx_right = None

    # --- 3. 初始化仿真机器人模型 ---
    rbt_left = Piper(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0), pos=np.array([0, 0, 0]), name='piper_left')
    rbt_right = Piper(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0), pos=np.array([0, -0.6, 0]), name='piper_right')

    # --- 4. 初始化相机 ---
    cam_id = '243322074546'
    try:
        rs_pipe = RealSenseD405(device=cam_id)
        print("相机预热中...")
        rs_pipe.get_pcd_texture_depth()
        rs_pipe.get_pcd_texture_depth()
        print("相机准备就绪。")
    except Exception as e:
        print(f"[ERROR] 无法连接到相机 ID {cam_id}: {e}")
        rs_pipe = None

    # --- 5. 加载初始标定矩阵 ---
    init_mat_path = "manual_calibration_realman.json"
    init_mat = load_calibration_matrix_from_json(init_mat_path)
    print("初始标定矩阵加载成功。")

    # --- 6. 实例化标定管理器 (右看左) ---
    calibration_manager = PiperMutualCalib(
        rbt_s_cam=rbt_right,  # 右臂作为相机搭载臂
        rbt_x_cam=rbtx_right,  # 实体右臂控制器
        rbt_s_target=rbt_left,  # 左臂作为被观察对象
        rbt_x_target=rbtx_left,  # 实体左臂控制器
        sensor_hdl=rs_pipe,
        init_calib_mat=init_mat,
        move_resolution=0.001,
        rotation_resolution=np.radians(1)
    )

    if rbtx_left is None or rbtx_right is None:
        print("\n[重要提示] 未能連接到一個或多個真實機械臂。")
        print("程式將在純仿真模式下運行，無法與實體機械臂同步。")
        print("您仍然可以操作點雲，但模型將不會移動。")

    print("\n标定程序启动。请使用键盘进行调整：")
    print("  - W/S: 沿全局X轴平移")
    print("  - A/D: 沿全局Y轴平移")
    print("  - Q/E: 沿全局Z轴平移")
    print("  - Z/X: 绕全局X轴旋转")
    print("  - C/V: 绕全局Y轴旋转")
    print("  - B/N: 绕全局Z轴旋转")
    print("标定结果将自动保存到 mutual_look_calibration_result.json 文件。")

    base.run()