import sys
sys.path.insert(0, r'F:\wrs-main-competition')  # 修改为你的 wrs 路径

import cv2
from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405, find_devices

def main():
    # 1. 查找已连接的 RealSense 设备
    serials, ctx = find_devices()
    if not serials:
        print("未找到 RealSense 设备，请检查相机连接。")
        return

    # 2. 使用第一个找到的设备创建相机实例
    print(f"使用设备: {serials[0]}")
    camera = RealSenseD405(device=serials[0])  # 可指定分辨率，默认 high

    print("按 'q' 键退出，按 's' 键保存当前彩色图像。")

    while True:
        # 3. 获取彩色图像（BGR 格式）
        color_img = camera.get_color_img()
        if color_img is None:
            print("获取图像失败")
            break

        # 显示图像
        cv2.imshow("RealSense Color", color_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("capture.png", color_img)
            print("已保存 capture.png")

    # 4. 释放资源
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()