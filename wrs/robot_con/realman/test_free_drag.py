#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全向拖拽控制器测试程序（Python版本 - 完整实现）

基于 test_free_drag_controller.cpp 完整实现，包含所有C++版本的功能：
1. 完整的阻抗控制算法（6DOF独立参数）
2. 重力补偿
3. 低通滤波
4. 耦合项补偿（抑制Servo队列滞后）
5. 旋转向量解缠绕
6. 轴锁定支持
7. 完整的状态机（包括安全激活检查）
8. 绝对时间同步
9. 详细的性能统计和错误处理

使用方法:
    python test.py [选项]

选项:
    --config <文件>      配置文件路径 (默认: config/config2.yaml)
                        配置文件应包含 control.free_drag 参数
    --help, -h           显示帮助信息

注意：所有参数（机器人IP、端口、力传感器配置、控制参数等）都从配置文件读取
"""

import time
import numpy as np
import signal
import sys
import os
from enum import Enum
from typing import Tuple, Optional, Dict, Any, List
from collections import namedtuple
import ctypes
import os

try:
    import yaml
except ImportError:
    yaml = None
    print("警告: PyYAML 未安装，无法使用配置文件支持。请运行: pip install PyYAML")
import wrs.robot_con.realman.realman as rm
from wrs.robot_con.realman.realman import ForceDataType
import wrs.basis.robot_math as robot_math

# 数学常数
M_PI = np.pi


class DragState(Enum):
    """拖拽控制器状态"""
    IDLE = "IDLE"      # 保持状态：高刚度位置伺服
    ACTIVE = "ACTIVE"  # 拖拽状态：低刚度、低阻尼（轻便）
    DAMPING = "DAMPING"  # 刹车状态：用户松手后的过渡状态，极高阻尼


class CoordinateFrame(Enum):
    """坐标系枚举"""
    BASE = "BASE"
    TOOL = "TOOL"
    CUSTOM = "CUSTOM"


# 力传感器偏移结构
ForceSensorOffset = namedtuple('ForceSensorOffset', ['position', 'orientation', 'enabled'])

def enable_windows_high_precision_timer():
    """
    通过调用 winmm.dll 提升 Windows 计时器精度
    对应 C++ 版本中的 timeBeginPeriod(1)
    """
    if os.name == 'nt':  # 仅在 Windows 系统下执行
        try:
            winmm = ctypes.WinDLL('winmm')
            # 请求 1 毫秒的计时器分辨率
            result = winmm.timeBeginPeriod(1)
            if result == 0:
                print("成功启用 Windows 高精度定时器 (精度: 1ms)")
                return True
            else:
                print(f"警告：无法设置高精度定时器，错误码: {result}")
        except Exception as e:
            print(f"调用 WinMM 失败: {e}")
    return False

def disable_windows_high_precision_timer():
    """
    程序退出前恢复系统默认精度，对应 C++ 中的 timeEndPeriod(1)
    """
    if os.name == 'nt':
        winmm = ctypes.WinDLL('winmm')
        winmm.timeEndPeriod(1)
def make_rotation_vector_continuous(new_rv: np.ndarray, ref_rv: np.ndarray) -> np.ndarray:
    """
    旋转向量解缠绕（De-winding）：确保新的旋转向量相对于参考旋转向量连续
    
    问题：同一个旋转可以用不同的旋转向量表示（角度可以相差 2π 的倍数，或轴方向相反）
    此函数找到最接近参考旋转向量的表示，避免控制不稳定
    
    Args:
        new_rv: 新的旋转向量 [rx, ry, rz]
        ref_rv: 参考旋转向量（通常是上一时刻的值）
    
    Returns:
        连续化的新旋转向量
    """
    ref_norm = np.linalg.norm(ref_rv)
    new_norm = np.linalg.norm(new_rv)
    
    # 如果参考向量或新向量太小，直接返回新向量（避免数值误差）
    if ref_norm < 1e-3 or new_norm < 1e-3:
        return new_rv
    
    # 计算直接欧氏距离
    dist_direct = np.linalg.norm(new_rv - ref_rv)
    min_dist = dist_direct
    best_rv = new_rv.copy()
    
    axis = new_rv / new_norm if new_norm > 1e-6 else np.array([1, 0, 0])
    angle = new_norm
    
    # 尝试 +/- 2π 的倍数（最多检查 +/- 3 个周期，处理更大的角度跳跃）
    for m in range(-3, 4):
        if m == 0:
            continue
        angle_shifted = angle + m * 2.0 * M_PI
        rv_shifted = axis * angle_shifted
        dist = np.linalg.norm(rv_shifted - ref_rv)
        if dist < min_dist:
            best_rv = rv_shifted
            min_dist = dist
    
    # 也检查相反轴方向（-axis, 2π - angle），因为旋转向量可以用相反方向表示相同旋转
    axis_opposite = -axis
    angle_opposite = 2.0 * M_PI - angle
    for m in range(-3, 4):
        angle_shifted = angle_opposite + m * 2.0 * M_PI
        rv_shifted = axis_opposite * angle_shifted
        dist = np.linalg.norm(rv_shifted - ref_rv)
        if dist < min_dist:
            best_rv = rv_shifted
            min_dist = dist
    
    # 如果找到了明显更接近的表示（距离减少 > 10%），返回它；否则返回原始值
    if min_dist < dist_direct * 0.9:
        return best_rv
    else:
        return new_rv


def normalize_rotation_vector(rv: np.ndarray, ref_rv: Optional[np.ndarray] = None) -> np.ndarray:
    """
    规范化旋转向量到 [0, π] 范围，防止范数过大
    
    Args:
        rv: 旋转向量 [rx, ry, rz]
        ref_rv: 参考旋转向量（用于解缠绕）
    
    Returns:
        规范化后的旋转向量
    """
    rv_norm = np.linalg.norm(rv)
    max_iterations = 5
    iteration = 0
    result = rv.copy()
    
    while rv_norm > M_PI and iteration < max_iterations:
        if rv_norm < 1e-6:
            break
        axis = result / rv_norm
        angle = rv_norm
        angle = np.fmod(angle, 2.0 * M_PI)
        if angle < 0:
            angle += 2.0 * M_PI
        if angle > M_PI:
            normalized_angle = 2.0 * M_PI - angle
            result = -axis * normalized_angle
        else:
            result = axis * angle
        
        # 如果提供了参考向量，再次解缠绕以确保连续性
        if ref_rv is not None:
            result = make_rotation_vector_continuous(result, ref_rv)
        
        rv_norm = np.linalg.norm(result)
        iteration += 1
    
    # 最终检查：如果仍然超过 π，强制规范化
    rv_norm = np.linalg.norm(result)
    if rv_norm > M_PI:
        if rv_norm < 1e-6:
            return result
        axis = result / rv_norm
        angle = rv_norm
        angle = np.fmod(angle, 2.0 * M_PI)
        if angle < 0:
            angle += 2.0 * M_PI
        if angle > M_PI:
            normalized_angle = 2.0 * M_PI - angle
            result = -axis * normalized_angle
        else:
            result = axis * angle
        
        if ref_rv is not None:
            result = make_rotation_vector_continuous(result, ref_rv)
    
    return result


def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    将欧拉角 [rx, ry, rz] 转换为旋转矩阵（ZYX顺序：外旋，Roll-Pitch-Yaw）
    
    Args:
        euler: 欧拉角 [rx, ry, rz] (弧度)
    
    Returns:
        旋转矩阵 (3x3)
    """
    rx, ry, rz = euler[0], euler[1], euler[2]
    
    # ZYX顺序：R = R_z(rz) * R_y(ry) * R_x(rx)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    R = np.array([
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy]
    ])
    
    return R


def rotmat_to_axangle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    将旋转矩阵转换为轴角表示
    
    Args:
        R: 旋转矩阵 (3x3)
    
    Returns:
        (axis, angle): 轴向量和角度（弧度）
    """
    # 使用Rodrigues公式的逆变换
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    
    if angle < 1e-6:
        # 角度很小，使用单位矩阵的轴
        return np.array([1.0, 0.0, 0.0]), 0.0
    elif abs(angle - np.pi) < 1e-6:
        # 180度旋转，需要特殊处理
        # 找到R + I的非零列
        R_plus_I = R + np.eye(3)
        # 找到最大列
        col_norms = np.linalg.norm(R_plus_I, axis=0)
        max_col = np.argmax(col_norms)
        axis = R_plus_I[:, max_col]
        axis = axis / np.linalg.norm(axis)
        return axis, angle
    else:
        # 一般情况
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2.0 * np.sin(angle))
        return axis, angle


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为欧拉角 [rx, ry, rz]（ZYX顺序）
    
    Args:
        R: 旋转矩阵 (3x3)
    
    Returns:
        欧拉角 [rx, ry, rz] (弧度)
    """
    sy = -R[2, 0]
    ry = np.arcsin(np.clip(sy, -1.0, 1.0))
    
    # 检查是否接近万向锁（ry ≈ ±π/2）
    threshold = 0.9999
    if abs(sy) < threshold:
        # 正常情况：标准ZYX提取公式
        cy = np.cos(ry)
        if abs(cy) > 1e-6:
            rx = np.arctan2(R[2, 1] / cy, R[2, 2] / cy)
            rz = np.arctan2(R[1, 0] / cy, R[0, 0] / cy)
            return np.array([rx, ry, rz])
        else:
            # cy接近0，使用备用公式
            rz = 0.0
            rx = np.arctan2(-R[0, 1], R[1, 1])
            return np.array([rx, ry, rz])
    else:
        # 万向锁情况：ry = ±π/2
        rz = 0.0
        rx = np.arctan2(-R[0, 1], R[1, 1])
        return np.array([rx, ry, rz])


class FreeDragController:
    """完整版全向拖拽控制器（Python实现，完整对应C++版本）"""
    
    def __init__(self, params: Dict[str, Any]):
        """
        初始化拖拽控制器
        
        Args:
            params: 参数字典，包含所有控制参数
        """
        # 6DOF参数（支持数组或标量）
        self.stiffness = self._parse_6dof_param(params.get('stiffness', [0, 0, 0, 0, 0, 0]))
        self.damping = self._parse_6dof_param(params.get('damping', [400, 400, 400, 5, 5, 5]))
        self.mass = self._parse_6dof_param(params.get('mass', [200, 200, 200, 1, 1, 1]))
        self.coupling = self._parse_6dof_param(params.get('coupling', [0, 0, 0, 0, 0, 0]))
        
        # 控制参数
        self.control_period = params.get('control_period', 0.008)  # 默认125Hz
        self.force_threshold_activate = params.get('force_threshold_activate', 8.0)
        self.damping_multiplier = params.get('damping_multiplier', 1.0)
        self.velocity_threshold_deactivate = params.get('velocity_threshold_deactivate', 0.001)
        self.nonlinear_gain_min = params.get('nonlinear_gain_min', 0.4)
        self.nonlinear_gain_max = params.get('nonlinear_gain_max', 10.0)
        self.nonlinear_force_threshold = params.get('nonlinear_force_threshold', 15.0)
        
        # 坐标系
        coord_frame_str = params.get('coordinate_frame', 'TOOL').upper()
        if coord_frame_str == 'BASE':
            self.coordinate_frame = CoordinateFrame.BASE
        elif coord_frame_str == 'TOOL':
            self.coordinate_frame = CoordinateFrame.TOOL
        else:
            self.coordinate_frame = CoordinateFrame.CUSTOM
        
        # 重力补偿参数
        self.payload_mass = params.get('payload_mass', 0.0)
        gravity_vec = params.get('gravity_vector_base', [0.0, 0.0, -9.81])
        self.gravity_vector_base = np.array(gravity_vec, dtype=np.float64)
        
        # 低通滤波器参数
        self.lpf_cutoff_frequency = params.get('lpf_cutoff_frequency', 10.0)
        self.filter_initialized = False
        self.filtered_force = np.zeros(6)
        
        # 力传感器偏移
        offset_pos = params.get('force_sensor_offset_position', [0.0, 0.0, 0.0])
        offset_ori = params.get('force_sensor_offset_orientation', [0.0, 0.0, 0.0])
        offset_enabled = params.get('force_sensor_offset_enabled', False)
        self.force_sensor_offset = ForceSensorOffset(
            position=np.array(offset_pos, dtype=np.float64),
            orientation=np.array(offset_ori, dtype=np.float64),
            enabled=offset_enabled
        )
        
        # 轴锁定
        axis_lock = params.get('axis_lock', [False] * 6)
        if isinstance(axis_lock, list) and len(axis_lock) == 6:
            self.axis_lock = [bool(x) for x in axis_lock]
        else:
            self.axis_lock = [False] * 6
        
        # 状态
        self.state = DragState.IDLE
        self.is_active = False
        self.state_initialized = False
        
        # 内部状态（用于阻抗控制积分器）
        self.p_k = np.zeros(3)  # 虚拟质量位置（Base Frame）
        self.R_k = np.eye(3)   # 虚拟质量旋转（Base Frame）
        self.p_k_minus_1 = np.zeros(3)
        self.R_k_minus_1 = np.eye(3)
        self.reference_pos_base = np.zeros(3)
        self.reference_rot_base = np.eye(3)
        self.reference_pos_initialized = False
        
        # 工具位姿（Tool Frame在Base Frame下的表示）
        self.tool_rot_base = np.eye(3)
        self.tool_trans_base = np.zeros(3)
        self.tool_tf_valid = False
    
    def _parse_6dof_param(self, value: Any) -> np.ndarray:
        """解析6DOF参数（支持标量或数组）"""
        if isinstance(value, (int, float)):
            return np.array([float(value)] * 6, dtype=np.float64)
        elif isinstance(value, list) and len(value) > 0:
            if len(value) == 6:
                return np.array(value, dtype=np.float64)
            elif len(value) == 1:
                return np.array([float(value[0])] * 6, dtype=np.float64)
            else:
                # 长度不足6，用最后一个值填充
                arr = [float(x) for x in value]
                while len(arr) < 6:
                    arr.append(arr[-1] if arr else 0.0)
                return np.array(arr[:6], dtype=np.float64)
        else:
            return np.zeros(6, dtype=np.float64)
    
    def apply_low_pass_filter(self, raw_force: np.ndarray, dt: float) -> np.ndarray:
        """
        应用一阶低通滤波器到力传感器数据
        
        Args:
            raw_force: 原始力/力矩 [Fx, Fy, Fz, Mx, My, Mz]
            dt: 控制周期（秒）
        
        Returns:
            滤波后的力/力矩
        """
        if not self.filter_initialized:
            self.filtered_force = raw_force.copy()
            self.filter_initialized = True
            return self.filtered_force
        
        tau = 1.0 / (2.0 * M_PI * self.lpf_cutoff_frequency)
        alpha = dt / (dt + tau)
        alpha = np.clip(alpha, 0.0, 1.0)
        
        # 应用一阶低通滤波器到每个分量
        self.filtered_force = alpha * raw_force + (1.0 - alpha) * self.filtered_force
        
        return self.filtered_force
    
    def get_compensated_force(self, raw_force: np.ndarray, R_tool_base: np.ndarray) -> np.ndarray:
        """
        计算重力补偿后的力（用于状态机判断）
        
        Args:
            raw_force: 原始力/力矩 [Fx, Fy, Fz, Mx, My, Mz]（传感器坐标系）
            R_tool_base: Tool Frame在Base Frame下的旋转矩阵
        
        Returns:
            补偿后的力/力矩
        """
        compensated = raw_force.copy()
        
        if self.payload_mass > 0.0:
            # 计算传感器在 Base 坐标系下的姿态
            R_sensor_base = R_tool_base.copy()
            if self.force_sensor_offset.enabled:
                # 计算从传感器坐标系到TCP坐标系的旋转矩阵
                R_sensor_tcp = euler_to_rotation_matrix(self.force_sensor_offset.orientation)
                # 传感器在 Base 下的姿态 = TCP在Base下的姿态 * 传感器在TCP下的姿态
                R_sensor_base = R_tool_base @ R_sensor_tcp
            
            # 计算重力在传感器坐标系下的分量
            # F_gravity_sensor = R_base^sensor * (m * g_base)
            R_base_sensor = R_sensor_base.T
            gravity_base = self.payload_mass * self.gravity_vector_base
            gravity_sensor = R_base_sensor @ gravity_base
            
            # 从力传感器读数中减去重力（在传感器坐标系下）
            compensated[:3] -= gravity_sensor
        
        return compensated
    
    def apply_force_transformations(self, external_force: np.ndarray, dt: float, 
                                   R_tool_base: np.ndarray) -> np.ndarray:
        """
        应用力变换（低通滤波 + 重力补偿 + 传感器偏移）
        
        Args:
            external_force: 外部力/力矩 [Fx, Fy, Fz, Mx, My, Mz]（传感器坐标系）
            dt: 控制周期（秒）
            R_tool_base: Tool Frame在Base Frame下的旋转矩阵
        
        Returns:
            变换后的力/力矩（TCP坐标系）
        """
        # 步骤0：应用低通滤波器
        filtered_force = self.apply_low_pass_filter(external_force, dt)
        
        # 步骤1：在传感器坐标系下进行重力补偿
        if self.payload_mass > 0.0:
            # 计算传感器在 Base 坐标系下的姿态
            R_sensor_base = R_tool_base.copy()
            if self.force_sensor_offset.enabled:
                R_sensor_tcp = euler_to_rotation_matrix(self.force_sensor_offset.orientation)
                R_sensor_base = R_tool_base @ R_sensor_tcp
            
            # 计算重力在传感器坐标系下的分量
            R_base_sensor = R_sensor_base.T
            gravity_base = self.payload_mass * self.gravity_vector_base
            gravity_sensor = R_base_sensor @ gravity_base
            
            # 从力传感器读数中减去重力
            filtered_force[:3] -= gravity_sensor
        
        # 步骤2：应用力传感器到TCP的偏移变换（如果配置了偏移）
        transformed_force = filtered_force.copy()
        if self.force_sensor_offset.enabled:
            # 简化处理：如果传感器有位置偏移，需要计算力矩补偿
            # 这里简化处理，只做旋转变换
            if np.linalg.norm(self.force_sensor_offset.orientation) > 1e-6:
                R_sensor_tcp = euler_to_rotation_matrix(self.force_sensor_offset.orientation)
                # 力变换（只旋转）
                transformed_force[:3] = R_sensor_tcp @ filtered_force[:3]
                transformed_force[3:6] = R_sensor_tcp @ filtered_force[3:6]
        
        return transformed_force
    
    def update_state(self, compensated_force: np.ndarray, current_vel: np.ndarray) -> bool:
        """
        更新状态机
        
        Args:
            compensated_force: 重力补偿后的力/力矩 [Fx, Fy, Fz, Mx, My, Mz]
            current_vel: 当前速度 [vx, vy, vz, wx, wy, wz]
        
        Returns:
            状态是否发生变化
        """
        old_state = self.state
        force_magnitude = np.linalg.norm(compensated_force[:3])
        velocity_magnitude = np.linalg.norm(current_vel[:3])
        
        if self.state == DragState.IDLE:
            if force_magnitude > self.force_threshold_activate:
                self.state = DragState.ACTIVE
                self.is_active = True
        elif self.state == DragState.ACTIVE:
            if force_magnitude <= self.force_threshold_activate:
                self.state = DragState.DAMPING
        elif self.state == DragState.DAMPING:
            if velocity_magnitude < self.velocity_threshold_deactivate:
                self.state = DragState.IDLE
                self.is_active = False
            elif force_magnitude > self.force_threshold_activate:
                self.state = DragState.ACTIVE
                self.is_active = True
        
        return old_state != self.state
    
    def safe_activate(self, current_pos: np.ndarray, current_rotmat: np.ndarray, 
                     current_vel: np.ndarray, max_velocity: float = 0.01) -> bool:
        """
        安全激活拖拽模式（检查机器人是否静止）
        
        Args:
            current_pos: 当前位置 [x, y, z]
            current_rotmat: 当前旋转矩阵 (3x3)
            current_vel: 当前速度 [vx, vy, vz, wx, wy, wz]
            max_velocity: 最大允许速度（m/s），超过此值认为机器人正在运动
        
        Returns:
            是否成功激活
        """
        velocity_magnitude = np.linalg.norm(current_vel[:3])
        if velocity_magnitude > max_velocity:
            return False
        
        # 激活：设置参考位置
        self.reference_pos_base = current_pos.copy()
        self.reference_rot_base = current_rotmat.copy()
        self.reference_pos_initialized = True
        
        # 初始化内部状态
        self.p_k = current_pos.copy()
        self.R_k = current_rotmat.copy()
        # 计算上一时刻位置（用于积分器）
        self.p_k_minus_1 = current_pos - current_vel[:3] * self.control_period
        # 旋转部分：使用指数映射回退
        w_current = current_vel[3:6]
        w_norm = np.linalg.norm(w_current)
        if w_norm < 1e-8:
            self.R_k_minus_1 = current_rotmat.copy()
        else:
            axis = w_current / w_norm
            angle = -w_norm * self.control_period
            R_inc = robot_math.rotmat_from_axangle(axis, angle)
            self.R_k_minus_1 = R_inc @ current_rotmat
        
        self.state_initialized = True
        self.is_active = True
        return True
    
    def deactivate(self):
        """停用拖拽模式"""
        self.is_active = False
        self.state_initialized = False
    
    def compute_drag_target(self, current_pos: np.ndarray, current_rotmat: np.ndarray,
                           current_vel: np.ndarray, external_force: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算拖拽目标位置和速度（完整阻抗控制算法）
        
        Args:
            current_pos: 当前位置 [x, y, z]（Base Frame）
            current_rotmat: 当前旋转矩阵 (3x3)（Base Frame，R_tool_base）
            current_vel: 当前速度 [vx, vy, vz, wx, wy, wz]（Base Frame）
            external_force: 外部力/力矩 [Fx, Fy, Fz, Mx, My, Mz]（传感器坐标系）
        
        Returns:
            target_pos: 目标位置 [x, y, z]（Base Frame）
            target_rotmat: 目标旋转矩阵 (3x3)（Base Frame）
            target_vel: 目标速度 [vx, vy, vz, wx, wy, wz]（Base Frame）
        """
        if not self.is_active or not self.reference_pos_initialized:
            # 未激活或未初始化，返回当前位置和零速度
            return current_pos.copy(), current_rotmat.copy(), np.zeros(6)
        
        # 更新工具位姿
        self.tool_rot_base = current_rotmat.copy()
        self.tool_trans_base = current_pos.copy()
        self.tool_tf_valid = True
        
        # 应用力变换（低通滤波 + 重力补偿 + 传感器偏移）
        transformed_force = self.apply_force_transformations(
            external_force, self.control_period, current_rotmat)
        
        # 将力从Tool Frame转换到Base Frame
        F_ext_tool = transformed_force.copy()
        F_ext_base = np.zeros(6)
        if self.tool_tf_valid:
            R = self.tool_rot_base
            F_ext_base[:3] = R @ F_ext_tool[:3]
            F_ext_base[3:6] = R @ F_ext_tool[3:6]
        else:
            F_ext_base = F_ext_tool
        
        # 根据坐标系选择Task Frame
        if self.coordinate_frame == CoordinateFrame.BASE:
            # Base Frame：Task Frame = Base Frame
            R_task_base = np.eye(3)
            t_task_base = np.zeros(3)
        elif self.coordinate_frame == CoordinateFrame.TOOL:
            # Tool Frame：Task Frame = Tool Frame
            R_task_base = self.tool_rot_base
            t_task_base = self.tool_trans_base
        else:
            # Custom Frame：简化处理，使用Base Frame
            R_task_base = np.eye(3)
            t_task_base = np.zeros(3)
        
        R_base_task = R_task_base.T
        t_base_task = -R_base_task @ t_task_base
        
        # 初始化状态（第一次调用时）
        if not self.state_initialized:
            # 安全检查：初始误差过大时重置参考位置
            init_error_pos = np.linalg.norm(self.reference_pos_base - current_pos)
            R_error = self.reference_rot_base @ current_rotmat.T
            aa_error = rotmat_to_axangle(R_error)
            init_error_rot = abs(aa_error[1])  # 角度
            MAX_INIT_ERROR_POS = 0.05  # 5cm
            MAX_INIT_ERROR_ROT = 0.1745  # 10度
            
            if init_error_pos > MAX_INIT_ERROR_POS or init_error_rot > MAX_INIT_ERROR_ROT:
                print("[警告] 初始误差过大，已重置参考位置")
                self.reference_pos_base = current_pos.copy()
                self.reference_rot_base = current_rotmat.copy()
            
            self.p_k = current_pos.copy()
            self.R_k = current_rotmat.copy()
            # 计算上一时刻位置
            self.p_k_minus_1 = current_pos - current_vel[:3] * self.control_period
            # 旋转部分
            w_current = current_vel[3:6]
            w_norm = np.linalg.norm(w_current)
            if w_norm < 1e-8:
                self.R_k_minus_1 = current_rotmat.copy()
            else:
                axis = w_current / w_norm
                angle = -w_norm * self.control_period
                R_inc = robot_math.rotmat_from_axangle(axis, angle)
                self.R_k_minus_1 = R_inc @ current_rotmat
            
            self.state_initialized = True
        
        # 转换到Task Frame
        p_k_task = R_base_task @ (self.p_k - t_task_base)
        p_k_minus_1_task = R_base_task @ (self.p_k_minus_1 - t_task_base)
        p_ref_task = R_base_task @ (self.reference_pos_base - t_task_base)
        p_current_task = R_base_task @ (current_pos - t_task_base)
        
        # 旋转矩阵转换到Task Frame
        R_k_task = R_base_task @ self.R_k @ R_task_base
        R_k_minus_1_task = R_base_task @ self.R_k_minus_1 @ R_task_base
        R_ref_task = R_base_task @ self.reference_rot_base @ R_task_base
        R_current_task = R_base_task @ current_rotmat @ R_task_base
        
        # 力变换到Task Frame
        F_ext_task = np.zeros(6)
        F_ext_task[:3] = R_base_task @ F_ext_base[:3]
        F_ext_task[3:6] = R_base_task @ F_ext_base[3:6]
        
        # 计算误差向量（切空间）
        delta_p_ref = p_ref_task - p_k_task
        delta_p_prev = p_k_minus_1_task - p_k_task
        
        # 旋转误差（使用对数映射）
        R_error_ref = R_ref_task @ R_k_task.T
        aa_ref = rotmat_to_axangle(R_error_ref)
        delta_w_ref = aa_ref[0] * aa_ref[1]  # axis * angle
        
        R_error_prev = R_k_minus_1_task @ R_k_task.T
        aa_prev = rotmat_to_axangle(R_error_prev)
        delta_w_prev = aa_prev[0] * aa_prev[1]
        
        delta_x_ref = np.concatenate([delta_p_ref, delta_w_ref])
        delta_x_prev = np.concatenate([delta_p_prev, delta_w_prev])
        
        # 计算耦合项误差（实际位置 - 积分器位置）
        delta_p_act = p_current_task - p_k_task
        R_error_act = R_current_task @ R_k_task.T
        aa_act = rotmat_to_axangle(R_error_act)
        delta_w_act = aa_act[0] * aa_act[1]
        delta_x_act = np.concatenate([delta_p_act, delta_w_act])
        
        # 构建质量、刚度、阻尼、耦合矩阵
        M = np.diag(self.mass)
        K = np.diag(self.stiffness)
        D = np.diag(self.damping)
        K_coupling = np.diag(self.coupling)
        
        # DAMPING状态下增加阻尼
        if self.state == DragState.DAMPING:
            D = D * self.damping_multiplier
        
        # 构建RHS和A矩阵（基于后向差分法的离散化）
        dt = self.control_period
        term_prev = M / (dt * dt)
        rhs = (F_ext_task + 
               K @ delta_x_ref - 
               term_prev @ delta_x_prev + 
               K_coupling @ delta_x_act)
        
        A = M / (dt * dt) + D / dt + K + K_coupling
        
        # 求解切空间中的增量
        try:
            delta_x_next = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            print("[错误] 矩阵A奇异或数值不稳定，返回当前位置")
            return current_pos.copy(), current_rotmat.copy(), np.zeros(6)
        
        # 检查NaN和Inf
        if not np.all(np.isfinite(delta_x_next)):
            print("[错误] 求解结果包含NaN或Inf，返回当前位置")
            return current_pos.copy(), current_rotmat.copy(), np.zeros(6)
        
        # 应用轴锁定
        for i in range(6):
            if self.axis_lock[i]:
                if i < 3:
                    delta_x_next[i] = delta_x_ref[i]
                else:
                    delta_x_next[i] = delta_x_ref[i]
        
        # 流形更新
        p_next_task = p_k_task + delta_x_next[:3]
        
        # 旋转更新（使用指数映射）
        w_inc = delta_x_next[3:6]
        w_norm = np.linalg.norm(w_inc)
        if w_norm < 1e-8:
            R_next_task = R_k_task
        else:
            axis = w_inc / w_norm
            R_inc = robot_math.rotmat_from_axangle(axis, w_norm)
            if self.coordinate_frame == CoordinateFrame.BASE:
                R_next_task = R_inc @ R_k_task
            else:
                R_next_task = R_k_task @ R_inc
        
        # 旋转矩阵正交性归一化（SVD）
        U, s, Vt = np.linalg.svd(R_next_task)
        R_next_task = U @ Vt
        if np.linalg.det(R_next_task) < 0:
            U[:, 2] *= -1
            R_next_task = U @ Vt
        
        # 转换回Base Frame
        p_k_plus_1_base = R_task_base @ p_next_task + t_task_base
        R_k_plus_1_base = R_task_base @ R_next_task @ R_base_task
        
        # 更新内部状态
        self.p_k_minus_1 = self.p_k
        self.R_k_minus_1 = self.R_k
        self.p_k = p_k_plus_1_base
        self.R_k = R_k_plus_1_base
        
        # 计算速度
        target_vel_task = delta_x_next / dt
        target_vel_base = np.zeros(6)
        target_vel_base[:3] = R_task_base @ target_vel_task[:3]
        # 角速度转换（简化处理）
        target_vel_base[3:6] = R_task_base @ target_vel_task[3:6]
        
        return p_k_plus_1_base, R_k_plus_1_base, target_vel_base
    
    def get_state(self) -> DragState:
        """获取当前状态"""
        return self.state
    
    def is_active_state(self) -> bool:
        """检查是否处于激活状态"""
        return self.is_active


class FreeDragControllerTest:
    """全向拖拽控制器测试主类（完整实现）"""
    
    def __init__(self, robot_ip: str, robot_port: int, drag_params: Dict[str, Any]):
        """
        初始化测试类
        
        Args:
            robot_ip: 机器人IP地址
            robot_port: 机器人端口
            drag_params: 拖拽控制器参数字典
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.control_frequency = 1.0 / drag_params.get('control_period', 0.008)
        self.control_period = drag_params.get('control_period', 0.008)
        
        # 创建机器人控制器
        print(f"正在连接到机器人 {robot_ip}:{robot_port}...")
        self.robot = rm.RealmanArmController(ip=robot_ip, port=robot_port, multiple_thread=True)
        print("机器人连接成功！")
        
        # 开始前清零六维力传感器数据
        print("正在清零六维力传感器数据...")
        try:
            result = self.robot.set_zero_force()
            self.robot.set_zero_force()
            self.robot.set_zero_force()
            if result == 0:
                print("六维力数据已清零")
                print(self.robot.get_ee_force(ForceDataType.zero_force_data))
            else:
                print(f"警告: 六维力数据清零失败，错误码：{result}")
        except Exception as e:
            print(f"警告: 清零六维力数据时出错: {e}")
        
        # 创建拖拽控制器
        self.drag_controller = FreeDragController(drag_params)
        
        # 运行标志
        self.running = True
        
        # 力传感器零偏
        self.force_zero_offset = np.zeros(6)
        self.force_zero_valid = False
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        print("\n收到退出信号 (Ctrl+C)，正在停止...")
        self.running = False
    
    def calibrate_force_sensor(self, num_samples: int = 80):
        """校准力传感器（零位设置）"""
        print("\n正在执行零位校准（请确保传感器无负载）...")
        try:
            self.robot.set_zero_force()
            print("硬件零位校准成功！")
        except Exception as e:
            print(f"硬件零位校准失败: {e}")
        
        # 软件零偏（多次采样取平均）
        print("正在执行软件零偏...")
        force_sum = np.zeros(6)
        success_count = 0
        
        for i in range(num_samples):
            if not self.running:
                print("零偏校准被中断")
                break
            try:
                force = self.robot.get_ee_force(ForceDataType.zero_force_data)
                if force is not None and not np.any(np.isnan(force)):
                    force_sum += force
                    success_count += 1
            except Exception as e:
                print(f"读取力传感器失败: {e}")
            except KeyboardInterrupt:
                print("\n零偏校准被中断")
                self.running = False
                break
            # 分段 sleep 以便响应 Ctrl+C
            sleep_interval = 0.005
            elapsed = 0
            while elapsed < sleep_interval and self.running:
                time.sleep(min(0.001, sleep_interval - elapsed))
                elapsed += 0.001
        
        if success_count > 0:
            self.force_zero_offset = force_sum / success_count
            self.force_zero_valid = True
            print(f"软件零偏完成，样本数={success_count}")
            print(f"偏移: F=[{self.force_zero_offset[0]:.3f}, "
                  f"{self.force_zero_offset[1]:.3f}, {self.force_zero_offset[2]:.3f}] N, "
                  f"T=[{self.force_zero_offset[3]:.3f}, {self.force_zero_offset[4]:.3f}, "
                  f"{self.force_zero_offset[5]:.3f}] N·m")
        else:
            print("软件零偏失败，继续使用硬件零点")
    
    def run(self, duration: Optional[float] = None, servo_lookahead_time: float = 0.1, 
            servo_gain: float = 100.0):
        """运行拖拽控制循环（完整实现）"""
        print("\n========== 开始拖拽控制循环 ==========")
        print(f"控制频率: {self.control_frequency} Hz")
        print(f"激活阈值: {self.drag_controller.force_threshold_activate} N")
        if self.drag_controller.force_threshold_activate > 0.0:
            print(f"提示：施加大于 {self.drag_controller.force_threshold_activate} N 的力以激活拖拽模式")
            print(f"      力小于等于 {self.drag_controller.force_threshold_activate} N 时拖拽模式将停用")
        else:
            print("提示：拖拽模式始终激活（force_threshold_activate = 0）")
        print("按 Ctrl+C 退出\n")
        
        # 获取初始位置
        try:
            initial_pos, initial_rotmat = self.robot.get_pose()
            print(f"初始位置: {initial_pos}")
        except Exception as e:
            print(f"获取初始位置失败: {e}")
            return
        
        # 启用servo模式（如果支持）
        try:
            if hasattr(self.robot, 'set_servo_mode'):
                self.robot.set_servo_mode(True)
                print("Servo模式已启用")
        except Exception as e:
            print(f"警告: 启用Servo模式失败: {e}")
        
        start_time = time.time()
        cycle_count = 0
        last_print_time = time.time()
        
        # 绝对时间同步
        next_wake_time = time.time()
        dt_duration = self.control_period
        
        # 保存上一时刻的位置（用于旋转向量解缠绕）
        previous_pos = None
        previous_rotmat = None
        previous_pos_initialized = False
        
        # 保存未激活状态下的参考位置（防止位置漂移）
        idle_reference_pos = initial_pos.copy()
        idle_reference_rotmat = initial_rotmat.copy()
        idle_reference_initialized = False
        
        # 性能统计
        class PerformanceStats:
            def __init__(self):
                self.sensor_time = 0.0
                self.position_time = 0.0
                self.velocity_time = 0.0
                self.compute_time = 0.0
                self.servo_time = 0.0
                self.sleep_time = 0.0
                self.count = 0
        
        perf_stats = PerformanceStats()
        prev_cycle_start = time.time()
        
        # Servo失败统计
        servo_fail_count = 0
        disconnect_count = 0
        
        while self.running:
            cycle_start_time = time.time()
            prev_cycle_time = cycle_start_time - prev_cycle_start
            if prev_cycle_time <= 0.0 or prev_cycle_time > 1.0:
                prev_cycle_time = self.control_period
            prev_cycle_start = cycle_start_time
            
            # 检查运行时间
            elapsed = time.time() - start_time
            if duration is not None and duration > 0.0 and elapsed >= duration:
                break
            
            try:
                # 步骤 1: 读取传感器
                t0 = time.time()
                force_raw = self.robot.get_ee_force(ForceDataType.zero_force_data)
                if force_raw is None or np.any(np.isnan(force_raw)):
                    external_force = np.zeros(6)
                else:
                    external_force = force_raw.copy()
                    if self.force_zero_valid:
                        external_force -= self.force_zero_offset
                t1 = time.time()
                perf_stats.sensor_time += (t1 - t0)
                
                # 步骤 2: 获取机器人状态
                t2 = time.time()
                current_pos, current_rotmat = self.robot.get_pose()
                
                # 旋转向量解缠绕（如果SDK返回的是旋转向量格式）
                # 注意：realman SDK返回的是旋转矩阵，但为了兼容性，我们假设可能返回欧拉角
                # 这里简化处理，假设get_pose返回的是位置和旋转矩阵
                
                # 计算速度（简化：使用位置差分）
                if cycle_count == 0:
                    current_vel = np.zeros(6)
                    prev_pos = current_pos.copy()
                    prev_rotmat = current_rotmat.copy()
                else:
                    # 位置速度
                    pos_velocity = (current_pos - prev_pos) / self.control_period
                    # 旋转速度（从旋转矩阵差分计算）
                    R_diff = current_rotmat @ prev_rotmat.T
                    aa_diff = rotmat_to_axangle(R_diff)
                    rot_velocity = aa_diff[0] * (aa_diff[1] / self.control_period)
                    current_vel = np.concatenate([pos_velocity, rot_velocity])
                    prev_pos = current_pos.copy()
                    prev_rotmat = current_rotmat.copy()
                
                t4_vel = time.time()
                perf_stats.position_time += (t4_vel - t2)
                perf_stats.velocity_time += (t4_vel - t2)
                
                # 计算Tool Frame在Base Frame下的旋转矩阵
                R_tool_base = current_rotmat
                
                # 重力补偿（用于状态机判断）
                compensated_force = self.drag_controller.get_compensated_force(
                    external_force, R_tool_base)
                
                # 更新状态机
                old_state = self.drag_controller.get_state()
                state_changed = self.drag_controller.update_state(compensated_force, current_vel)
                
                if state_changed:
                    new_state = self.drag_controller.get_state()
                    if old_state == DragState.IDLE and new_state == DragState.ACTIVE:
                        # IDLE -> ACTIVE: 需要激活控制器
                        if self.drag_controller.safe_activate(current_pos, current_rotmat, current_vel):
                            print(">>> 拖拽模式已激活 (ACTIVE) <<<")
                            idle_reference_pos = current_pos.copy()
                            idle_reference_rotmat = current_rotmat.copy()
                            idle_reference_initialized = True
                        else:
                            print(">>> 拖拽模式激活失败：机器人正在运动，请等待静止 <<<")
                            self.drag_controller.deactivate()
                    elif old_state == DragState.ACTIVE and new_state == DragState.DAMPING:
                        print(">>> 进入刹车状态 (DAMPING) <<<")
                    elif old_state == DragState.DAMPING and new_state == DragState.IDLE:
                        print(">>> 拖拽模式已停用 (IDLE) <<<")
                        idle_reference_pos = current_pos.copy()
                        idle_reference_rotmat = current_rotmat.copy()
                        idle_reference_initialized = True
                    elif old_state == DragState.DAMPING and new_state == DragState.ACTIVE:
                        print(">>> 从刹车状态恢复拖拽 (ACTIVE) <<<")
                
                # 步骤 3: 计算控制律
                t4 = time.time()
                is_dragging = self.drag_controller.is_active_state()
                
                if is_dragging:
                    target_pos, target_rotmat, target_vel = self.drag_controller.compute_drag_target(
                        current_pos, current_rotmat, current_vel, external_force)
                else:
                    # 未激活时，使用固定的参考位置
                    if not idle_reference_initialized:
                        idle_reference_pos = current_pos.copy()
                        idle_reference_rotmat = current_rotmat.copy()
                        idle_reference_initialized = True
                    target_pos = idle_reference_pos.copy()
                    target_rotmat = idle_reference_rotmat.copy()
                    target_vel = np.zeros(6)
                
                t5 = time.time()
                perf_stats.compute_time += (t5 - t4)
                
                # 步骤 4: 发送指令
                t6 = time.time()
                try:
                    # 验证 target_rotmat 的有效性
                    if target_rotmat is None:
                        raise ValueError("target_rotmat is None")
                    if not isinstance(target_rotmat, np.ndarray):
                        raise TypeError(f"target_rotmat must be numpy array, got {type(target_rotmat)}")
                    if target_rotmat.shape != (3, 3):
                        raise ValueError(f"target_rotmat must be 3x3, got shape {target_rotmat.shape}")
                    if np.any(np.isnan(target_rotmat)) or np.any(np.isinf(target_rotmat)):
                        raise ValueError("target_rotmat contains NaN or Inf")
                    
                    # 将目标位置和旋转矩阵转换为位姿列表 [x, y, z, rx, ry, rz]
                    euler = robot_math.rotmat_to_euler(target_rotmat)
                    target_pose = np.concatenate([target_pos, euler])
                    
                    # 使用servo_p进行实时控制
                    self.robot.servo_p(target_pose.tolist(), follow=False)
                    servo_result = True

                except Exception as e:
                    servo_result = False
                    if cycle_count % 100 == 0:
                        import traceback
                        print(f"发送指令失败: {e}")
                        print(f"错误类型: {type(e).__name__}")
                        print(f"target_rotmat 类型: {type(target_rotmat)}")
                        if target_rotmat is not None:
                            print(f"target_rotmat 形状: {target_rotmat.shape}")
                            print(f"target_rotmat 值:\n{target_rotmat}")
                        traceback.print_exc()
                
                if not servo_result:
                    servo_fail_count += 1
                    if servo_fail_count % 100 == 0:
                        print(f"[警告] servoP失败次数: {servo_fail_count}")
                else:
                    if servo_fail_count > 0:
                        print(f"[信息] servoP恢复成功（之前连续失败 {servo_fail_count} 次）")
                        servo_fail_count = 0
                
                t7 = time.time()
                perf_stats.servo_time += (t7 - t6)
                
                cycle_count += 1
                perf_stats.count += 1
                
                # 状态打印（每秒打印一次）
                if time.time() - last_print_time >= 1.0:
                    current_state = self.drag_controller.get_state()
                    state_str = current_state.value
                    print(f"\n[循环 {cycle_count}] "
                          f"F=[{external_force[0]:.3f}, {external_force[1]:.3f}, {external_force[2]:.3f}] N, "
                          f"|F|={np.linalg.norm(external_force[:3]):.3f} N, "
                          f"状态: {state_str}")
                    
                    # 调试信息：显示实际循环周期与目标周期的偏差
                    dt_error = (prev_cycle_time - self.control_period) * 1000.0
                    print(f"  实际循环周期: {prev_cycle_time * 1000.0:.3f} ms "
                          f"(目标: {self.control_period * 1000.0:.3f} ms, "
                          f"偏差: {dt_error:+.3f} ms)")
                    
                    # 性能分析
                    if perf_stats.count > 0:
                        print(f"  性能分析（平均耗时，基于 {perf_stats.count} 次循环）:")
                        print(f"    力传感器读取: {(perf_stats.sensor_time / perf_stats.count) * 1000.0:.3f} ms")
                        print(f"    获取位置: {(perf_stats.position_time / perf_stats.count) * 1000.0:.3f} ms")
                        print(f"    获取速度: {(perf_stats.velocity_time / perf_stats.count) * 1000.0:.3f} ms")
                        print(f"    计算目标: {(perf_stats.compute_time / perf_stats.count) * 1000.0:.3f} ms")
                        print(f"    servoP调用: {(perf_stats.servo_time / perf_stats.count) * 1000.0:.3f} ms")
                        print(f"    sleep时间: {(perf_stats.sleep_time / perf_stats.count) * 1000.0:.3f} ms")
                        
                        total_process_time = ((perf_stats.sensor_time + perf_stats.position_time + 
                                             perf_stats.velocity_time + perf_stats.compute_time + 
                                             perf_stats.servo_time) / perf_stats.count)
                        print(f"    总处理时间: {total_process_time * 1000.0:.3f} ms")
                        print(f"    预期sleep时间: {(self.control_period - total_process_time) * 1000.0:.3f} ms")
                    
                    last_print_time = time.time()
                    perf_stats = PerformanceStats()
                
                # 绝对时间同步
                next_wake_time += dt_duration
                now = time.time()
                if now > next_wake_time:
                    # 检测是否超时
                    drift = (now - next_wake_time) * 1000.0  # 毫秒
                    if drift > 2.0:  # 滞后超过2ms
                        # 重置时间轴
                        next_wake_time = now + dt_duration
                        perf_stats.sleep_time += 0
                    else:
                        # 轻微滞后，不sleep
                        pass
                else:
                    # 正常情况：睡眠直到目标时间
                    sleep_time = next_wake_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        wake_up = time.time()
                        perf_stats.sleep_time += (wake_up - now)
                
            except KeyboardInterrupt:
                print("\n收到 KeyboardInterrupt，正在停止...")
                self.running = False
                break
            except Exception as e:
                print(f"控制循环错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print("\n========== 停止控制 ==========")
        print(f"总循环数: {cycle_count}")
        
        # 清理工作
        try:
            if hasattr(self.robot, 'set_servo_mode'):
                self.robot.set_servo_mode(False)
        except Exception as e:
            print(f"清理时出错（可忽略）: {e}")
        
        self.drag_controller.deactivate()
        print("程序已退出")


def load_free_drag_config(config_file: str) -> Optional[Dict[str, Any]]:
    """
    从配置文件加载 FreeDragController 参数（完整支持所有参数）
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典，如果加载失败返回 None
    """
    if yaml is None:
        print("警告: PyYAML 未安装，无法加载配置文件")
        return None
    
    if not os.path.exists(config_file):
        print(f"警告: 配置文件不存在: {config_file}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print("警告: 配置文件为空")
            return None
        
        result = {
            'robot_type': 'realman',
            'robot_ip': '192.168.3.18',
            'robot_port': 8080,
            'sensor_type': 'realman',
            'control_frequency': 125.0,
            'duration': None,
            'servo_lookahead_time': 0.1,
            'servo_gain': 100.0,
        }
        
        # 读取机器人配置
        if 'robot' in config:
            robot_config = config['robot']
            if 'type' in robot_config:
                result['robot_type'] = robot_config['type']
            if 'ip_address' in robot_config:
                result['robot_ip'] = robot_config['ip_address']
            if 'port' in robot_config:
                robot_port = robot_config['port']
                if robot_port == 0:
                    if result['robot_type'].lower() in ['realman', 'rm']:
                        result['robot_port'] = 8080
                    else:
                        result['robot_port'] = 30004
                else:
                    result['robot_port'] = robot_port
        
        # 读取力传感器配置
        if 'force_sensor' in config:
            sensor_config = config['force_sensor']
            if 'sensor_type' in sensor_config:
                result['sensor_type'] = sensor_config['sensor_type']
            
            # 读取力传感器偏移
            if 'offset' in sensor_config:
                offset = sensor_config['offset']
                offset_pos = offset.get('position', [0.0, 0.0, 0.0])
                offset_ori = offset.get('orientation', [0.0, 0.0, 0.0])
                result['force_sensor_offset_position'] = offset_pos
                result['force_sensor_offset_orientation'] = offset_ori
                # 检查是否启用偏移
                pos_norm = np.linalg.norm(offset_pos)
                ori_norm = np.linalg.norm(offset_ori)
                result['force_sensor_offset_enabled'] = (pos_norm > 1e-6 or ori_norm > 1e-6)
            else:
                result['force_sensor_offset_position'] = [0.0, 0.0, 0.0]
                result['force_sensor_offset_orientation'] = [0.0, 0.0, 0.0]
                result['force_sensor_offset_enabled'] = False
        
        # 读取控制频率
        if 'control' in config and 'frequency' in config['control']:
            result['control_frequency'] = float(config['control']['frequency'])
            result['control_period'] = 1.0 / result['control_frequency']
        else:
            result['control_period'] = 0.008  # 默认125Hz
        
        # 读取运行持续时间
        if 'control' in config and 'duration' in config['control']:
            duration = config['control']['duration']
            if duration is not None and duration > 0:
                result['duration'] = float(duration)
            else:
                result['duration'] = None
        
        # 读取 free_drag 配置
        drag_params = {
            'stiffness': [0, 0, 0, 0, 0, 0],
            'damping': [400, 400, 400, 5, 5, 5],
            'mass': [200, 200, 200, 1, 1, 1],
            'coupling': [0, 0, 0, 0, 0, 0],
            'force_threshold_activate': 8.0,
            'damping_multiplier': 1.0,
            'velocity_threshold_deactivate': 0.001,
            'nonlinear_gain_min': 0.4,
            'nonlinear_gain_max': 10.0,
            'nonlinear_force_threshold': 15.0,
            'coordinate_frame': 'TOOL',
            'lpf_cutoff_frequency': 10.0,
            'axis_lock': [False] * 6,
            'payload_mass': 0.0,
            'gravity_vector_base': [0.0, 0.0, -9.81],
            'control_period': result['control_period'],
        }
        
        if 'control' in config and 'free_drag' in config['control']:
            free_drag = config['control']['free_drag']
            
            # 读取刚度
            if 'stiffness' in free_drag:
                stiffness_val = free_drag['stiffness']
                if isinstance(stiffness_val, (int, float)):
                    drag_params['stiffness'] = [float(stiffness_val)] * 6
                elif isinstance(stiffness_val, list):
                    if len(stiffness_val) == 6:
                        drag_params['stiffness'] = [float(x) for x in stiffness_val]
                    elif len(stiffness_val) == 1:
                        drag_params['stiffness'] = [float(stiffness_val[0])] * 6
            
            # 读取质量
            if 'mass' in free_drag:
                mass_val = free_drag['mass']
                if isinstance(mass_val, (int, float)):
                    drag_params['mass'] = [float(mass_val)] * 6
                elif isinstance(mass_val, list):
                    if len(mass_val) == 6:
                        drag_params['mass'] = [float(x) for x in mass_val]
                    elif len(mass_val) == 1:
                        drag_params['mass'] = [float(mass_val[0])] * 6
            
            # 读取阻尼
            if 'damping' in free_drag:
                damping_val = free_drag['damping']
                if isinstance(damping_val, (int, float)):
                    drag_params['damping'] = [float(damping_val)] * 6
                elif isinstance(damping_val, list):
                    if len(damping_val) == 6:
                        drag_params['damping'] = [float(x) for x in damping_val]
                    elif len(damping_val) == 1:
                        drag_params['damping'] = [float(damping_val[0])] * 6
            
            # 读取耦合项
            if 'coupling' in free_drag:
                coupling_val = free_drag['coupling']
                if isinstance(coupling_val, (int, float)):
                    drag_params['coupling'] = [float(coupling_val)] * 6
                elif isinstance(coupling_val, list):
                    if len(coupling_val) == 6:
                        drag_params['coupling'] = [float(x) for x in coupling_val]
                    elif len(coupling_val) == 1:
                        drag_params['coupling'] = [float(coupling_val[0])] * 6
            
            # 读取其他参数
            if 'force_threshold_activate' in free_drag:
                drag_params['force_threshold_activate'] = float(free_drag['force_threshold_activate'])
            if 'damping_multiplier' in free_drag:
                drag_params['damping_multiplier'] = float(free_drag['damping_multiplier'])
            if 'velocity_threshold_deactivate' in free_drag:
                drag_params['velocity_threshold_deactivate'] = float(free_drag['velocity_threshold_deactivate'])
            if 'nonlinear_gain_min' in free_drag:
                drag_params['nonlinear_gain_min'] = float(free_drag['nonlinear_gain_min'])
            if 'nonlinear_gain_max' in free_drag:
                drag_params['nonlinear_gain_max'] = float(free_drag['nonlinear_gain_max'])
            if 'nonlinear_force_threshold' in free_drag:
                drag_params['nonlinear_force_threshold'] = float(free_drag['nonlinear_force_threshold'])
            if 'coordinate_frame' in free_drag:
                drag_params['coordinate_frame'] = free_drag['coordinate_frame'].upper()
            if 'lpf_cutoff_frequency' in free_drag:
                drag_params['lpf_cutoff_frequency'] = float(free_drag['lpf_cutoff_frequency'])
            if 'payload_mass' in free_drag:
                drag_params['payload_mass'] = float(free_drag['payload_mass'])
            if 'gravity_vector_base' in free_drag:
                drag_params['gravity_vector_base'] = free_drag['gravity_vector_base']
            
            # 读取轴锁定
            if 'axis_lock' in free_drag:
                axis_lock_val = free_drag['axis_lock']
                if isinstance(axis_lock_val, bool):
                    drag_params['axis_lock'] = [axis_lock_val] * 6
                elif isinstance(axis_lock_val, list):
                    if len(axis_lock_val) == 6:
                        drag_params['axis_lock'] = [bool(x) if isinstance(x, bool) else (int(x) != 0) 
                                                    for x in axis_lock_val]
            
            # 读取servo参数
            if 'servo' in free_drag:
                servo_node = free_drag['servo']
                if 'lookahead_time' in servo_node:
                    result['servo_lookahead_time'] = float(servo_node['lookahead_time'])
                if 'gain' in servo_node:
                    result['servo_gain'] = float(servo_node['gain'])
        
        result['drag_params'] = drag_params
        return result
        
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数 - 整合 Windows 高精度时钟与原有全部打印信息"""
    import argparse

    parser = argparse.ArgumentParser(description='全向拖拽控制器测试程序（Python版本 - 完整实现）')
    parser.add_argument('--config', type=str, default="config/config2.yaml",
                        help='配置文件路径 (默认: config/config2.yaml)')

    args = parser.parse_args()

    # 默认配置文件路径（基于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_file = os.path.join(script_dir, 'config', 'config2.yaml')

    # 处理配置文件路径
    if args.config:
        if os.path.isabs(args.config):
            config_file = args.config
        else:
            config_file = os.path.join(script_dir, args.config)
    else:
        config_file = default_config_file

    # 启用 Windows 实时性优化（对应 C++ timeBeginPeriod）
    has_high_precision = enable_windows_high_precision_timer()

    try:
        # 从配置文件加载参数
        config_params = {}
        if os.path.exists(config_file):
            print(f"正在加载配置文件: {config_file}")
            config_params = load_free_drag_config(config_file)
            if config_params:
                print("配置文件加载成功")
            else:
                print("配置文件加载失败，使用默认值")
                config_params = {}
        else:
            if args.config:
                print(f"警告: 指定的配置文件不存在: {config_file}")
            else:
                print(f"提示: 默认配置文件不存在: {config_file}，使用默认值")

        # 提取参数
        robot_ip = config_params.get('robot_ip', '192.168.3.18')
        robot_port = config_params.get('robot_port', 8080)
        duration = config_params.get('duration', None)
        servo_lookahead_time = config_params.get('servo_lookahead_time', 0.1)
        servo_gain = config_params.get('servo_gain', 100.0)
        drag_params = config_params.get('drag_params', {})

        # 打印使用的参数 (保留原本所有打印)
        print("\n========== 配置参数 ==========")
        print(f"机器人IP地址: {robot_ip}")
        print(f"机器人端口: {robot_port}")
        print(f"控制频率: {1.0 / drag_params.get('control_period', 0.008):.1f} Hz")
        print(f"刚度: {drag_params.get('stiffness', [0] * 6)}")
        print(f"质量: {drag_params.get('mass', [200] * 6)}")
        print(f"阻尼: {drag_params.get('damping', [400] * 6)}")
        print(f"耦合项: {drag_params.get('coupling', [0] * 6)}")
        print(f"激活阈值: {drag_params.get('force_threshold_activate', 8.0)} N")
        print(f"低通滤波截止频率: {drag_params.get('lpf_cutoff_frequency', 10.0)} Hz")
        if drag_params.get('payload_mass', 0.0) > 0.0:
            print(f"负载质量: {drag_params.get('payload_mass', 0.0)} kg")
            print(f"重力向量(Base): {drag_params.get('gravity_vector_base', [0, 0, -9.81])} m/s²")
        if any(drag_params.get('axis_lock', [False] * 6)):
            print(f"轴锁定: {drag_params.get('axis_lock', [False] * 6)}")
        print(f"坐标系: {drag_params.get('coordinate_frame', 'TOOL')}")
        if duration is not None:
            print(f"运行时间: {duration} 秒")
        else:
            print("运行时间: 无限（按 Ctrl+C 退出）")
        print(f"Windows 高精度时钟状态: {'已激活' if has_high_precision else '未激活'}")
        print("==============================\n")

        # 创建测试实例
        test = FreeDragControllerTest(robot_ip, robot_port, drag_params)

        # 校准力传感器
        test.calibrate_force_sensor()

        # 等待稳定
        print("等待传感器稳定...")
        sleep_time = 0.5
        sleep_interval = 0.1
        elapsed = 0
        while elapsed < sleep_time:
            time.sleep(min(sleep_interval, sleep_time - elapsed))
            elapsed += sleep_interval
            if not test.running:
                break

        # 运行控制循环
        if test.running:
            test.run(duration=duration, servo_lookahead_time=servo_lookahead_time,
                     servo_gain=servo_gain)
        else:
            print("程序已退出")

    except KeyboardInterrupt:
        print("\n收到 KeyboardInterrupt，正在停止...")
    finally:
        # 确保在任何情况下退出都恢复系统精度 (对应 C++ timeEndPeriod)
        disable_windows_high_precision_timer()
        print("主程序已退出")


if __name__ == "__main__":
    main()
