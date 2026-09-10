#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/6/18 11:40
# @Author : ZhangXi
import pickle
import cv2
import os


def export_color_images_from_pkl(pkl_path, save_dir="exported_images"):
    # 检查文件是否存在且不为空
    if not os.path.exists(pkl_path):
        print(f"[ERROR] 文件不存在: {pkl_path}")
        return

    file_size = os.path.getsize(pkl_path)
    if file_size == 0:
        print(f"[ERROR] 文件为空: {pkl_path}")
        return

    print(f"[INFO] 文件大小: {file_size} 字节")

    os.makedirs(save_dir, exist_ok=True)

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"[INFO] 读取 {len(data)} 帧数据")

        for i, frame in enumerate(data):
            if 'color_img' in frame:
                color_img = frame['color_img']
                filename = os.path.join(save_dir, f"color_image_{i:03d}.png")
                cv2.imwrite(filename, color_img)
                print(f"[INFO] 已保存: {filename}")
            else:
                print(f"[WARNING] 第{i}帧中没有 color_img")

    except EOFError:
        print(f"[ERROR] Pickle文件损坏或格式不正确: {pkl_path}")
    except Exception as e:
        print(f"[ERROR] 读取文件时发生错误: {str(e)}")


if __name__ == "__main__":
    pkl_file_path = "pointcloud_20250922_161935.pkl"
    export_color_images_from_pkl(pkl_file_path)