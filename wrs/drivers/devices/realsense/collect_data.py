import numpy as np
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as gm
from realsense_d400s import RealSenseD405
from wrs import wd, rm, mgm
import cv2
import pickle
import signal
import datetime

saved_frames_data = []
separator = {'separator': True}  # 用于分隔每帧
rs_pipe = None  # 全局用于信号中断调用
base = None  # Panda3D base
exit_called = False  # 防止重复保存


def save_pointcloud_data(all_data, filename=None):
    if not all_data:
        print("[INFO] 没有帧需要保存。")
        return

    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pointcloud_{timestamp}.pkl"

    try:
        print(f"[INFO] Saving {len(all_data)} frames to '{filename}'...")
        with open(filename, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"[INFO] ✅ Saved {len(all_data)} frames to '{filename}'")
    except Exception as e:
        print(f"[ERROR] Failed to save point cloud data: {e}")
    print("[INFO] ✅ 释放已保存数据...")
    all_data.clear()


def update(onscreen, key_map, task):
    if len(onscreen) > 0:
        for ele in onscreen:
            ele.detach()

    try:
        pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
    except RuntimeError:
        if key_map.get('s', False):
            print("[ERROR] Failed to retrieve frame when trying to save.")
        return task.cont

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
        save_pointcloud_data(saved_frames_data)
        print(f"[INFO] Frame Saved Successfully")

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
    # save_pointcloud_data(saved_frames_data)
    sys.exit(0)


if __name__ == "__main__":
    print("[INFO] Starting the application...")

    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405()

    # frame_queue = Queue(maxsize=10)
    # frame_thread = threading.Thread(target=capture_frames, args=(rs_pipe, frame_queue), daemon=True)
    # frame_thread.start()

    onscreen = []
    key_map = setup_key_listener()

    base.taskMgr.doMethodLater(0.2, update, "update",
                               extraArgs=[onscreen, key_map],
                               appendTask=True)

    try:
        base.run()
    finally:
        print("[INFO] Application exiting, saving data...")
        # save_pointcloud_data(saved_frames_data)
