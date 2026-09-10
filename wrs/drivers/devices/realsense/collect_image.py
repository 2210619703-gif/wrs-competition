#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/6/16 14:42
# @Author : ZhangXi
import numpy as np
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as gm
from realsense_d400s import RealSenseD405
from wrs import wd, rm, mgm
import cv2
import pickle
import threading
from queue import Queue
import signal
import sys
import datetime

# ✅ 全局变量
saved_frames_data = []
separator = {'separator': True}
rs_pipe = None
base = None
exit_called = False

def save_pointcloud_data(all_data, filename=None):
    if not all_data:
        print("[INFO] 没有帧需要保存。")
        return

    real_frames = [d for d in all_data if isinstance(d, dict) and 'pcd' in d]

    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pointcloud_{timestamp}.pkl"

    try:
        print(f"[INFO] Saving {len(real_frames)} frames to '{filename}'...")
        with open(filename, 'wb') as f:
            pickle.dump(real_frames, f)
        print(f"[INFO] ✅ Saved {len(real_frames)} frames to '{filename}'")
    except Exception as e:
        print(f"[ERROR] Failed to save point cloud data: {e}")
    print("[INFO] ✅ 释放已保存数据...")
    all_data.clear()


def capture_frames(rs_pipe, frame_queue):
    while True:
        pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
        frame_queue.put({
            'pcd': pcd,
            'pcd_color': pcd_color,
            'depth_img': depth_img,
            'color_img': color_img,
        })

def update(onscreen, frame_queue, key_map, task):
    if len(onscreen) > 0:
        for ele in onscreen:
            ele.detach()

    try:
        frame = frame_queue.get_nowait()  # 非阻塞方式
    except:
        return task.cont  # 当前没帧就跳过

    pcd = frame['pcd']
    pcd_color = frame['pcd_color']
    depth_img = frame['depth_img']
    color_img = frame['color_img']

    pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
    onscreen.append(mgm.gen_pointcloud(pcd, pcd_color_rgba))
    onscreen[-1].attach_to(base)

    if key_map.get('s', False):
        print("[INFO] 's' pressed: Storing point cloud in memory...")
        saved_frames_data.append({
            'pcd': pcd.copy(),
            'pcd_color': pcd_color_rgba.copy(),
            'depth_img': depth_img.copy(),
            'color_img': color_img.copy(),
        })
        print(f"[INFO] Total frames stored: {len(saved_frames_data)}")

    return task.cont



def setup_key_listener():
    key_map = {}

    def on_key_press(key):
        key_map[key] = True

    def on_key_release(key):
        key_map[key] = False

    base.accept('s', on_key_press, ['s'])
    base.accept('s-up', on_key_release, ['s'])

    return key_map

def exit_handler(sig=None, frame=None):
    global exit_called
    if exit_called:
        return
    exit_called = True
    print("\n[INFO] 程序即将退出，正在保存点云数据...")
    save_pointcloud_data(saved_frames_data)
    sys.exit(0)

if __name__ == "__main__":
    print("[INFO] Starting the application...")

    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405
    frame_queue = Queue(maxsize=10)
    frame_thread = threading.Thread(target=capture_frames, args=(rs_pipe, frame_queue), daemon=True)
    frame_thread.start()

    onscreen = []
    key_map = setup_key_listener()

    base.taskMgr.doMethodLater(0.2, update, "update",
                               extraArgs=[onscreen, frame_queue, key_map],
                               appendTask=True)

    try:
        base.run()
    finally:
        print("[INFO] Application exiting, saving data...")
        save_pointcloud_data(saved_frames_data)
