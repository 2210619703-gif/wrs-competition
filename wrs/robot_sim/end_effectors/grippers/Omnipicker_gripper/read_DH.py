#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/5/24 11:15
# @Author : ZhangXi
import xml.etree.ElementTree as ET
import numpy as np
from math import sin, cos, atan2, sqrt

def rpy_to_rot_matrix(roll, pitch, yaw):
    # 根据rpy转换成旋转矩阵（XYZ顺序）
    Rx = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])
    Ry = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])
    Rz = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = []
    for joint in root.findall('joint'):
        name = joint.attrib['name']
        origin = joint.find('origin')
        axis = joint.find('axis')

        if origin is not None:
            xyz = origin.attrib.get('xyz', '0 0 0')
            rpy = origin.attrib.get('rpy', '0 0 0')
            xyz = np.array(list(map(float, xyz.split())))
            rpy = np.array(list(map(float, rpy.split())))
        else:
            xyz = np.zeros(3)
            rpy = np.zeros(3)

        if axis is not None:
            axis_xyz = np.array(list(map(float, axis.attrib.get('xyz', '0 0 1').split())))
        else:
            axis_xyz = np.array([0, 0, 1])

        parent_link = joint.find('parent').attrib['link']
        child_link = joint.find('child').attrib['link']

        joints.append({
            'name': name,
            'parent': parent_link,
            'child': child_link,
            'xyz': xyz,
            'rpy': rpy,
            'axis': axis_xyz
        })

    return joints

def compute_dh_parameters(joints):
    dh_params = []

    # 遍历关节计算DH参数
    # DH参数间需要知道相邻joint之间的相对变换，先简单用joint的origin表示
    for i in range(len(joints)):
        joint = joints[i]

        # 这里简化处理：直接用joint的origin的z轴平移作为d
        # x轴平移作为a
        # rpy中pitch近似作为alpha
        # theta为变量，初始0

        d = joint['xyz'][2]   # 沿z轴平移
        a = joint['xyz'][0]   # 沿x轴平移
        alpha = joint['rpy'][1]  # pitch角，绕x轴旋转
        theta = 0  # 关节变量，默认0

        dh_params.append({
            'joint_name': joint['name'],
            'd': d,
            'a': a,
            'alpha': alpha,
            'theta': theta
        })

    return dh_params

def print_dh_params(dh_params):
    print(f"{'Joint Name':20s} {'d (m)':>10s} {'a (m)':>10s} {'alpha (rad)':>12s} {'theta (rad)':>12s}")
    for param in dh_params:
        print(f"{param['joint_name']:20s} {param['d']:10.5f} {param['a']:10.5f} {param['alpha']:12.5f} {param['theta']:12.5f}")

if __name__ == "__main__":
    urdf_path = "robot_urdf.xacro"  # 请替换成你的urdf文件路径
    joints = parse_urdf(urdf_path)
    dh_params = compute_dh_parameters(joints)
    print_dh_params(dh_params)
