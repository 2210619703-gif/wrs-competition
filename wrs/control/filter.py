#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/6/7 18:11
# @Author : ZhangXi
import numpy as np
from collections import deque


class Pose6DFilter:
    def __init__(self, alpha=0.2):
        """
        alpha: 滤波系数，0 < alpha < 1
        alpha 越小，滤波越强，响应越慢；alpha 越大，响应越快，滤波越弱
        """
        self.alpha = alpha
        self.prev_filtered = None

    def reset(self):
        self.prev_filtered = None

    def update(self, pose6d):
        """
        输入6维姿态：[x, y, z, roll, pitch, yaw]
        输出滤波后的6维姿态
        """
        pose6d = np.array(pose6d)
        if pose6d.shape != (3,):
            raise ValueError("输入必须是6维姿态向量")

        if self.prev_filtered is None:
            # 第一次直接赋值
            filtered = pose6d
        else:
            # 低通滤波公式：
            # y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
            filtered = self.alpha * pose6d + (1 - self.alpha) * self.prev_filtered

        self.prev_filtered = filtered
        return filtered
