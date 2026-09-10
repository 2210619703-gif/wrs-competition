#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/6/16 15:00
# @Author : ZhangXi
import time
import numpy as np
import pickle
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as gm
from wrs import mgm


def load_pointcloud_data(filename):
    """
    加载保存的点云数据。
    每一帧数据是一个包含 'pcd', 'pcd_color', 'depth_img', 'color_img' 的字典。
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"[INFO] ✅ 成功加载 {len(data)} 帧数据。")
        return data
    except Exception as e:
        print(f"[ERROR] 无法读取文件 {filename}: {e}")
        return []


def visualize_pointcloud_frames(frames, interval_sec=1.0):
    """
    可视化点云帧。

    参数:
        frames (list of dict): 每一帧包含 'pcd' 和 'pcd_color' 的字典
        interval_sec (float): 每帧显示时间（秒）
    """
    print(f"[INFO] 开始可视化 {len(frames)} 帧点云...")
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    onscreen = []

    def update_view(task):
        nonlocal index
        if index >= len(frames):
            print("[INFO] 所有帧已显示完毕。")
            base.taskMgr.remove("visualize")
            return task.done

        # 清除上一帧内容
        for obj in onscreen:
            obj.detach()
        onscreen.clear()

        # 当前帧点云
        frame = frames[index]
        pcd = frame['pcd']
        pcd_color = frame['pcd_color']

        # 如果没有alpha通道则补上
        if pcd_color.shape[1] == 3:
            pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
        else:
            pcd_color_rgba = pcd_color

        cloud_model = mgm.gen_pointcloud(pcd, pcd_color_rgba)
        cloud_model.attach_to(base)
        onscreen.append(cloud_model)

        print(f"[INFO] 正在显示第 {index + 1}/{len(frames)} 帧")
        index += 1
        return task.again

    index = 0
    base.taskMgr.doMethodLater(interval_sec, update_view, "visualize")
    base.run()


if __name__ == "__main__":
    # 修改为你自己的 pkl 文件路径
    filename = "pointcloud_20250922_161935.pkl"
    frames = load_pointcloud_data(filename)
    if frames:
        visualize_pointcloud_frames(frames, interval_sec=2)
