#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO + 点云 6D 位姿 + 可抓取判定（平滑版）
- 颜色识别（YOLO 或仿真给定）
- 粗略位姿估计（面心 + yaw）
- 顶面可抓取判定（时序平滑，放宽条件）
- 结构化输出供决策层使用
- 支持仿真/实机切换
"""

import sys
import time
import random
from pathlib import Path

# ========== 配置 ==========
SIMULATION_MODE = False   # True: 仿真模式; False: 实机模式（需相机和模型）

# 实机模式配置 — train-6: 7 类 (red/green/blue/yellow/purple/brown/white)
MODEL_PATH = Path(__file__).resolve().parents[4] / "runs/detect/train-6/weights/best.pt"
CUBE_SIZE = 0.05
ICP_THRESHOLD = 0.05

YOLO_IMGSZ = 640
YOLO_CONF = 0.5
YOLO_EVERY_N_FRAMES = 2
YOLO_MAX_DET = 16
YOLO_DEVICE = "cpu"

MIN_POINTS_FOR_POSE = 30
MIN_POINTS_FOR_GRASP = 20          # 降低点数要求
MAX_TILT_ANGLE_DEG = 45.0          # 放宽倾斜角度（原35°）
FLATNESS_THRESHOLD = 0.025         # 平整度放宽到2.5cm

SMOOTH_ALPHA_POS = 1.0
SMOOTH_ALPHA_YAW = 0.5             # 增强 yaw 平滑（原0.9）

WORKSPACE_ROI = {
    'x_min': -0.15, 'x_max': 0.25,
    'y_min': -0.20, 'y_max': 0.25,
    'z_min': 0.20, 'z_max': 0.80,
}
NOISE_FILTER = {'z_min': 0.05, 'z_max': 1.50}
PCD_UPDATE_INTERVAL = 0.2          # 提高点云显示流畅度（原0.8）
AXIS_UPDATE_INTERVAL = 0.3         # 坐标系更新间隔（秒），减少抖动

_CLASS_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]

# ---------- 全局输出数据 ----------
_output_data = []
_graspable_history = {}   # 用于抓取判定的时序平滑 {block_name: [bool, bool, bool]}

def get_output_data():
    return _output_data.copy()


# ========== 仿真数据生成器 ==========
class SimulationDataGenerator:
    def __init__(self):
        self.blocks_info = [
            {"color": "blue",  "position": (0.05, 0.02, 0.48), "yaw_deg": 5.0,  "graspable": True},
            {"color": "green", "position": (-0.04, 0.04, 0.48), "yaw_deg": -8.0, "graspable": True},
            {"color": "red",   "position": (0.20, -0.05, 0.52), "yaw_deg": -90.0, "graspable": False},
        ]
        self.frame_idx = 0

    def get_frame(self):
        self.frame_idx += 1
        blocks = []
        for info in self.blocks_info:
            pos = info["position"]
            perturb = (random.uniform(-0.002, 0.002), random.uniform(-0.002, 0.002), random.uniform(-0.001, 0.001))
            position = (pos[0]+perturb[0], pos[1]+perturb[1], pos[2]+perturb[2])
            yaw_deg = info["yaw_deg"] + random.uniform(-1.0, 1.0)
            blocks.append({
                'name': info["color"],
                'cls_id': 0,
                'bbox': (0,0,0,0),
                'face_center': np.array(position),
                'rot_mat': None,
                'euler_deg': (0, 0, yaw_deg),
                'points': np.empty((0,3)),
                'graspable': info["graspable"],
            })
        return blocks


# ========== 实机模式所有函数 ==========
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from realsense_d400s import RealSenseD405

def _yolo_use_half():
    try:
        import torch
        return YOLO_DEVICE != "cpu" and torch.cuda.is_available()
    except ImportError:
        return False

class PoseSmoother:
    def __init__(self, alpha_pos=0.3, alpha_yaw=0.3):
        self.alpha_pos = alpha_pos
        self.alpha_yaw = alpha_yaw
        self.pos_prev = None
        self.yaw_prev = None
        self.max_yaw_change = np.radians(15.0)  # 每帧最大角度变化15°
    def reset(self):
        self.pos_prev = None
        self.yaw_prev = None
    def smooth(self, pos_new, yaw_new):
        if self.pos_prev is None:
            self.pos_prev = pos_new.copy()
            self.yaw_prev = yaw_new
            return pos_new, yaw_new
        # 位置平滑
        pos_smooth = self.alpha_pos * pos_new + (1 - self.alpha_pos) * self.pos_prev
        # 角度平滑，并限制变化率
        delta = yaw_new - self.yaw_prev
        if delta > np.pi:
            delta -= 2*np.pi
        elif delta < -np.pi:
            delta += 2*np.pi
        # 限制单步变化不超过 max_yaw_change
        delta = np.clip(delta, -self.max_yaw_change, self.max_yaw_change)
        yaw_smooth = self.yaw_prev + self.alpha_yaw * delta
        self.pos_prev = pos_smooth
        self.yaw_prev = yaw_smooth
        return pos_smooth, yaw_smooth

def filter_roi(verts, colors, roi):
    if verts is None or len(verts) == 0:
        return verts, colors
    mask = ((verts[:, 0] >= roi['x_min']) & (verts[:, 0] <= roi['x_max']) &
            (verts[:, 1] >= roi['y_min']) & (verts[:, 1] <= roi['y_max']) &
            (verts[:, 2] >= roi['z_min']) & (verts[:, 2] <= roi['z_max']))
    return verts[mask], colors[mask] if colors is not None else None

def compute_yaw_from_points(points):
    if len(points) < 10:
        return 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 增大下采样体素，提高稳定性
    pcd = pcd.voxel_down_sample(0.008)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)
    pts = np.asarray(pcd.points)
    if len(pts) < 10:
        return 0.0
    xy = pts[:, :2] - np.mean(pts[:, :2], axis=0)
    cov = np.cov(xy.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    main_dir = eig_vecs[:, np.argmax(eig_vals)]
    yaw = np.arctan2(main_dir[1], main_dir[0])
    return yaw

def extract_block_points(verts_orig, h, w, x1, y1, x2, y2):
    if verts_orig is None or len(verts_orig) != h * w:
        return np.empty((0, 3))
    roi = verts_orig.reshape(h, w, 3)[y1:y2, x1:x2].reshape(-1, 3)
    mask = (roi[:, 2] > NOISE_FILTER['z_min']) & (roi[:, 2] < NOISE_FILTER['z_max'])
    return roi[mask]

def compute_face_center_from_depth(depth_img, cx, cy, intr_mat):
    h, w = depth_img.shape
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    if cx_i < 0 or cx_i >= w or cy_i < 0 or cy_i >= h:
        return None
    z_mm = depth_img[cy_i, cx_i]
    if z_mm <= 0 or z_mm > 1500:
        return None
    z = z_mm / 1000.0
    fx, fy = intr_mat[0,0], intr_mat[1,1]
    cx0, cy0 = intr_mat[0,2], intr_mat[1,2]
    X = (cx - cx0) * z / fx
    Y = (cy - cy0) * z / fy
    return np.array([X, Y, z])

def is_graspable(points, block_name):
    """带时序平滑的抓取判定"""
    if points is None or len(points) < MIN_POINTS_FOR_GRASP:
        current = False
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.008)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)
        pts = np.asarray(pcd.points)
        if len(pts) < MIN_POINTS_FOR_GRASP:
            current = False
        else:
            try:
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=150)
                [a, b, c, d] = plane_model
                normal = np.array([a, b, c])
                normal = normal / np.linalg.norm(normal)
                vertical = np.array([0, 0, 1])
                cos_angle = np.dot(normal, vertical)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                tilt_deg = np.degrees(np.arccos(cos_angle))
                if tilt_deg > MAX_TILT_ANGLE_DEG:
                    current = False
                else:
                    distances = np.abs(np.dot(pts, normal[:3]) + d) / np.linalg.norm(normal[:3])
                    flatness = np.std(distances)
                    current = (flatness <= FLATNESS_THRESHOLD)
            except Exception:
                current = False
    # 时序平滑（最近3帧，至少2帧为True）
    if block_name not in _graspable_history:
        _graspable_history[block_name] = []
    _graspable_history[block_name].append(current)
    if len(_graspable_history[block_name]) > 3:
        _graspable_history[block_name].pop(0)
    smooth_result = sum(_graspable_history[block_name]) >= 2
    return smooth_result

def build_blocks_from_detections(boxes, classes, names, verts_orig, h, w, depth_img, intr_mat, scale_x, scale_y):
    blocks = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
        cls_id = int(classes[i])
        cls_name = names[cls_id]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx_depth = cx * scale_x
        cy_depth = cy * scale_y
        face_center = compute_face_center_from_depth(depth_img, cx_depth, cy_depth, intr_mat)
        block_points = extract_block_points(verts_orig, h, w, x1, y1, x2, y2)
        block = {
            'name': cls_name,
            'cls_id': cls_id,
            'bbox': (x1, y1, x2, y2),
            'face_center': face_center,
            'rot_mat': None,
            'euler_deg': None,
            'points': block_points,
            'graspable': False,
        }
        if face_center is not None and len(block_points) >= MIN_POINTS_FOR_POSE:
            yaw = compute_yaw_from_points(block_points)
            z_axis = np.array([0, 0, 1])
            x_axis = np.array([np.cos(yaw), np.sin(yaw), 0])
            y_axis = np.cross(z_axis, x_axis)
            rot_mat = np.column_stack((x_axis, y_axis, z_axis))
            block['rot_mat'] = rot_mat
            block['euler_deg'] = (0, 0, np.degrees(yaw))
            # 使用带平滑的抓取判定
            block['graspable'] = is_graspable(block_points, cls_name)
        blocks.append(block)
    return blocks

# 可视化函数（英文显示）
def draw_face_center_sphere(vis, center, radius=0.003, color=(1,0,0)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    vis.add_geometry(sphere, reset_bounding_box=False)
    return sphere

def draw_3d_axis(vis, center, rot_mat, scale=0.04):
    center = np.asarray(center, dtype=np.float64)
    rot = np.asarray(rot_mat, dtype=np.float64)
    colors = [[1,0,0], [0,1,0], [0,0,1]]
    combined = o3d.geometry.TriangleMesh()
    for i, col in enumerate(colors):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=scale*0.015, cone_radius=scale*0.035,
            cylinder_height=scale*0.7, cone_height=scale*0.3, resolution=8)
        target_dir = rot[:, i]
        z_axis = np.array([0,0,1])
        if np.allclose(target_dir, z_axis):
            R = np.eye(3)
        elif np.allclose(target_dir, -z_axis):
            R = np.diag([1,-1,-1])
        else:
            v = np.cross(z_axis, target_dir)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, target_dir)
            vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            R = np.eye(3) + vx + vx @ vx * ((1-c)/(s**2))
        arrow.rotate(R, center=(0,0,0))
        arrow.translate(center)
        arrow.paint_uniform_color(col)
        combined += arrow
    vis.add_geometry(combined, reset_bounding_box=False)
    return combined

def project_cam_to_pixel(points_3d, intr_mat):
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    z = pts[:, 2]
    valid = z > 1e-6
    u = intr_mat[0,0] * pts[:,0] / z + intr_mat[0,2]
    v = intr_mat[1,1] * pts[:,1] / z + intr_mat[1,2]
    return np.stack([u, v], axis=1), valid

def draw_yolo_view(img, blocks, fps_text, intr_mat):
    vis_img = img.copy()
    for block in blocks:
        x1, y1, x2, y2 = block['bbox']
        color = _CLASS_COLORS[block['cls_id'] % len(_CLASS_COLORS)]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = block['name']
        if block['face_center'] is not None:
            c = block['face_center']
            grasp_mark = " [Graspable]" if block['graspable'] else " [Not Graspable]"
            label = f"{block['name']}{grasp_mark} ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})"
            pts_2d, valid = project_cam_to_pixel(c, intr_mat)
            if valid[0]:
                cx, cy = int(round(pts_2d[0,0])), int(round(pts_2d[0,1]))
                cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis_img, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(vis_img, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis_img

def clear_geometries(vis, geoms):
    for g in geoms:
        vis.remove_geometry(g, reset_bounding_box=False)
    geoms.clear()

def update_face_center_spheres(vis, spheres, blocks):
    clear_geometries(vis, spheres)
    for block in blocks:
        if block['face_center'] is not None:
            spheres.append(draw_face_center_sphere(vis, block['face_center']))

def update_pointcloud_axes(vis, axes, blocks):
    clear_geometries(vis, axes)
    for block in blocks:
        if block['face_center'] is not None and block['rot_mat'] is not None:
            axes.append(draw_3d_axis(vis, block['face_center'], block['rot_mat'], scale=0.04))

def update_live_pointcloud(vis, live_pcd, verts, pcd_colors, live_pcd_added, last_update, now):
    if verts is None or len(verts) == 0:
        return live_pcd_added, last_update
    if now - last_update < PCD_UPDATE_INTERVAL:
        return live_pcd_added, last_update
    live_pcd.points = o3d.utility.Vector3dVector(verts)
    if pcd_colors is not None:
        live_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    else:
        live_pcd.paint_uniform_color([0, 1, 0])
    if not live_pcd_added:
        vis.add_geometry(live_pcd, reset_bounding_box=True)
        return True, now
    vis.update_geometry(live_pcd)
    return live_pcd_added, now

def create_standard_cube(size=0.05, density=12):
    s = size / 2
    g = np.linspace(-s, s, density)
    xx, yy, zz = np.meshgrid(g, g, g)
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([1, 0, 0])
    return pcd

def run_yolo(model, color_img, half):
    return model.predict(color_img, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                         max_det=YOLO_MAX_DET, device=YOLO_DEVICE,
                         half=half, verbose=False)

# ---------- 实机主程序 ----------
def run_real_mode():
    global _output_data
    print("Loading YOLO model...")
    _use_half = _yolo_use_half()
    model = YOLO(str(MODEL_PATH))
    print("Initializing RealSense camera...")
    cam = RealSenseD405(resoultion='mid')
    intr_mat = cam.intr_mat

    print("Starting point cloud window...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud + Face Centers + Axes", width=800, height=600)
    live_pcd = o3d.geometry.PointCloud()
    live_pcd_added = False
    face_spheres = []
    axes_geoms = []
    icp_geometries = []

    cached_boxes = np.empty((0, 4))
    cached_classes = np.empty((0,))
    cached_names = {}
    cached_blocks = []
    frame_idx = 0
    icp_trigger = False
    last_print_time = 0
    t_prev = time.perf_counter()
    fps_smooth = 0.0
    last_pcd_update = 0.0
    last_axis_update = 0.0      # 坐标系上次更新时间
    smoothers = [PoseSmoother(SMOOTH_ALPHA_POS, SMOOTH_ALPHA_YAW) for _ in range(8)]

    print("=" * 60)
    print("Real mode running... Press [Q] to exit | [SPACE] to trigger ICP")
    print("Face center = depth at box center | Point cloud: red spheres + RGB axes")
    print("Graspability smoothed (3 frames), tilt <= {} deg, flatness <= {:.1f}cm".format(MAX_TILT_ANGLE_DEG, FLATNESS_THRESHOLD*100))
    print("Axis update interval: {}s, yaw smoothing alpha={}".format(AXIS_UPDATE_INTERVAL, SMOOTH_ALPHA_YAW))
    print("=" * 60)

    while True:
        verts_raw, pcd_colors_raw, depth_img, color_img = cam.get_pcd_texture_depth()
        if color_img is None or depth_img is None:
            continue

        frame_idx += 1
        now = time.perf_counter()
        h, w = color_img.shape[:2]
        verts_orig = verts_raw

        # 点云滤波
        verts = verts_raw.copy() if verts_raw is not None else None
        pcd_colors = pcd_colors_raw.copy() if pcd_colors_raw is not None else None
        if verts is not None and len(verts) > 0:
            valid = (verts[:, 2] > NOISE_FILTER['z_min']) & (verts[:, 2] < NOISE_FILTER['z_max'])
            verts = verts[valid]
            pcd_colors = pcd_colors[valid] if pcd_colors is not None else None
            verts, pcd_colors = filter_roi(verts, pcd_colors, WORKSPACE_ROI)

        live_pcd_added, last_pcd_update = update_live_pointcloud(
            vis, live_pcd, verts, pcd_colors, live_pcd_added, last_pcd_update, now)

        # YOLO 检测
        run_detect = (frame_idx % YOLO_EVERY_N_FRAMES == 0) or len(cached_boxes) == 0
        if run_detect:
            results = run_yolo(model, color_img, _use_half)
            r0 = results[0]
            cached_names = r0.names
            if r0.boxes is not None and len(r0.boxes):
                cached_boxes = r0.boxes.xyxy.cpu().numpy()
                cached_classes = r0.boxes.cls.cpu().numpy()
            else:
                cached_boxes = np.empty((0, 4))
                cached_classes = np.empty((0,))
                for sm in smoothers:
                    sm.reset()

        depth_h, depth_w = depth_img.shape
        scale_x = depth_w / w
        scale_y = depth_h / h

        if scale_x != 1.0 or scale_y != 1.0:
            boxes_scaled = cached_boxes.copy()
            boxes_scaled[:, [0,2]] = boxes_scaled[:, [0,2]] * scale_x
            boxes_scaled[:, [1,3]] = boxes_scaled[:, [1,3]] * scale_y
            boxes_use = boxes_scaled.astype(int)
        else:
            boxes_use = cached_boxes.astype(int)

        raw_blocks = build_blocks_from_detections(
            boxes_use, cached_classes, cached_names, verts_orig, h, w, depth_img, intr_mat, scale_x, scale_y)

        # 平滑
        for idx, block in enumerate(raw_blocks):
            if block['face_center'] is not None and block['rot_mat'] is not None and idx < len(smoothers):
                center_raw = block['face_center']
                yaw_raw = np.arctan2(block['rot_mat'][1,0], block['rot_mat'][0,0])
                center_smooth, yaw_smooth = smoothers[idx].smooth(center_raw, yaw_raw)
                block['face_center'] = center_smooth
                x_axis = np.array([np.cos(yaw_smooth), np.sin(yaw_smooth), 0])
                z_axis = np.array([0,0,1])
                y_axis = np.cross(z_axis, x_axis)
                rot_mat_smooth = np.column_stack((x_axis, y_axis, z_axis))
                block['rot_mat'] = rot_mat_smooth
                block['euler_deg'] = (0, 0, np.degrees(yaw_smooth))
        cached_blocks = raw_blocks

        # 更新全局输出数据
        _output_data = []
        for b in cached_blocks:
            if b['face_center'] is not None:
                data = {
                    'color': b['name'],
                    'position': tuple(b['face_center'].tolist()),
                    'yaw_deg': b['euler_deg'][2] if b['euler_deg'] else 0.0,
                    'graspable': b['graspable'],
                    'confidence': 1.0
                }
                _output_data.append(data)

        # 更新点云中的红色小球（每帧更新，开销小）
        update_face_center_spheres(vis, face_spheres, cached_blocks)

        # 坐标系降频更新（减少抖动）
        if now - last_axis_update >= AXIS_UPDATE_INTERVAL:
            update_pointcloud_axes(vis, axes_geoms, cached_blocks)
            last_axis_update = now

        # 帧率
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
        fps_text = f"FPS: {fps_smooth:.1f}"

        yolo_vis = draw_yolo_view(color_img, cached_blocks, fps_text, intr_mat)
        cv2.imshow("YOLO + Face Center Tracking", yolo_vis)

        # 终端输出（每3秒，英文）
        if time.time() - last_print_time > 3.0 and cached_blocks:
            print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            for b in cached_blocks:
                if b['face_center'] is not None:
                    c = b['face_center']
                    euler = b['euler_deg'] if b['euler_deg'] else (0,0,0)
                    grasp_str = "Graspable" if b['graspable'] else "Not Graspable"
                    print(f"[{b['name']}] FaceCenter: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})  yaw={euler[2]:.1f}°  {grasp_str}")
                else:
                    print(f"[{b['name']}] FaceCenter extraction failed")
            last_print_time = time.time()

        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            icp_trigger = True
            print("\n[ICP] Triggered...")

        if icp_trigger and verts is not None and len(verts) > 0:
            for geom in icp_geometries:
                vis.remove_geometry(geom, reset_bounding_box=False)
            icp_geometries.clear()
            icp_target = o3d.geometry.PointCloud()
            icp_target.points = o3d.utility.Vector3dVector(verts)
            if pcd_colors is not None:
                icp_target.colors = o3d.utility.Vector3dVector(pcd_colors)
            target_down = icp_target.voxel_down_sample(0.005)
            source_pcd = create_standard_cube(CUBE_SIZE)
            init_trans = np.eye(4)
            init_trans[:3, 3] = np.mean(verts, axis=0)
            source_moved = source_pcd.transform(init_trans)
            source_down = source_moved.voxel_down_sample(0.005)
            reg = o3d.pipelines.registration.registration_icp(
                source_down, target_down, ICP_THRESHOLD, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
            print("ICP transformation:\n", reg.transformation)
            icp_result = source_moved.transform(reg.transformation)
            icp_result.paint_uniform_color([1, 0, 0])
            vis.add_geometry(icp_target, reset_bounding_box=False)
            vis.add_geometry(icp_result, reset_bounding_box=False)
            icp_geometries.extend([icp_target, icp_result])
            icp_trigger = False
            vis.poll_events()
            vis.update_renderer()

    try:
        cam.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("Real mode exited.")

# ========== 主入口 ==========
if __name__ == "__main__":
    if SIMULATION_MODE:
        print("=" * 60)
        print("Simulation mode running... Outputting simulated data.")
        sim_gen = SimulationDataGenerator()
        try:
            while True:
                blocks = sim_gen.get_frame()
                _output_data = []
                for b in blocks:
                    data = {
                        'color': b['name'],
                        'position': tuple(b['face_center'].tolist()),
                        'yaw_deg': b['euler_deg'][2],
                        'graspable': b['graspable'],
                        'confidence': 1.0
                    }
                    _output_data.append(data)
                print(f"\r--- {time.strftime('%H:%M:%S')} ---", end='')
                for d in _output_data:
                    grasp_str = "Graspable" if d['graspable'] else "Not Graspable"
                    print(f"\n[{d['color']}] Position: {d['position']}, yaw={d['yaw_deg']:.1f}°, {grasp_str}", end='')
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nSimulation mode exited.")
    else:
        run_real_mode()