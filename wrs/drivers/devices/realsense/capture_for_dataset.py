#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import cv2
import os
import sys

sys.path.insert(0, r'F:\wrs-main-competition')

from realsense_d400s import RealSenseD405

def main():
    SAVE_DIR = r"F:\wrs-main-competition\wrs\drivers\devices\realsense\block_images\images\train"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 单色积木快捷键映射（o=orange）
    color_keys = {
        ord('r'): 'red',
        ord('g'): 'green',
        ord('b'): 'blue',
        ord('y'): 'yellow',
        ord('o'): 'orange',
        ord('p'): 'purple',
    }

    print("初始化相机...")
    cam = RealSenseD405()
    print("相机就绪。")
    print("操作说明：")
    print("  单色积木: 按 r/g/b/y/o 直接保存（自动添加颜色标签）")
    print("  多色组合: 按 s 保存（文件名含 multi，后续用 LabelMe 手动标注）")
    print("  按 q 退出程序")
    print(f"保存目录: {SAVE_DIR}")
    print("-" * 60)

    while True:
        _, _, _, color_img = cam.get_pcd_texture_depth()
        if color_img is None:
            print("获取图像失败")
            continue

        cv2.imshow("Save: r/g/b/y/o (single) | s (multi) | q to quit", color_img)

        key = cv2.waitKey(50) & 0xFF
        if key in color_keys:
            color_name = color_keys[key]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{color_name}_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, color_img)
            print(f"已保存单色照片: {filename}")
        elif key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, color_img)
            print(f"已保存多色组合照片: {filename}")
        elif key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()