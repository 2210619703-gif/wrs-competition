#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panda3D 仿真数据采集相机。

该模块不修改 WRS 原有的 panda3d_utils.py，而是在其虚拟相机思想基础上，
额外提供 RGB、深度图、点云和语义标注导出能力。
"""
from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager

import cv2
import numpy as np
from panda3d.core import GraphicsOutput, PerspectiveLens, Texture
from PIL import Image, ImageDraw, ImageFont

from wrs import rm


class SimDataCamera:
    """用于离屏采集仿真数据的虚拟相机。"""

    def __init__(
        self,
        base,
        cam_pos=np.array([0.75, -0.85, 0.75]),
        lookat_pos=np.array([0.0, 0.0, 0.02]),
        resolution=np.array([640, 480]),
        fov=45.0,
        near=0.01,
        far=5.0,
        name="sim_data_camera",
    ):
        self.base = base
        self.resolution = np.asarray(resolution, dtype=int)
        self.width = int(self.resolution[0])
        self.height = int(self.resolution[1])
        self.near = float(near)
        self.far = float(far)

        self.color_tex = Texture(f"{name}_color")
        self.depth_tex = Texture(f"{name}_depth")
        self.depth_tex.setFormat(Texture.FDepthComponent)

        self.buffer = base.win.makeTextureBuffer(
            f"{name}_buffer",
            self.width,
            self.height,
            self.color_tex,
            True,
        )
        self.buffer.setClearColor((0, 0, 0, 1))
        self.buffer.addRenderTexture(
            self.depth_tex,
            GraphicsOutput.RTMCopyRam,
            GraphicsOutput.RTPDepth,
        )

        lens = PerspectiveLens()
        lens.setFov(float(fov))
        lens.setNearFar(self.near, self.far)
        lens.setAspectRatio(self.width / self.height)

        self.cam_np = base.makeCamera(self.buffer, camName=name)
        self.cam_np.node().setLens(lens)
        self.cam_np.setPos(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.cam_np.lookAt(float(lookat_pos[0]), float(lookat_pos[1]), float(lookat_pos[2]))

    @property
    def cam_pos(self):
        """相机在世界坐标系下的位置。"""
        return np.array([*self.cam_np.getPos()], dtype=float)

    @property
    def cam_rotmat(self):
        """相机在世界坐标系下的旋转矩阵。"""
        return np.array(self.cam_np.getMat().getUpper3(), dtype=float).T

    @property
    def intrinsics(self):
        """根据相机 FOV 估计针孔相机内参。"""
        fov_x, fov_y = self.cam_np.node().getLens().getFov()
        fx = self.width / (2.0 * np.tan(np.deg2rad(fov_x) / 2.0))
        fy = self.height / (2.0 * np.tan(np.deg2rad(fov_y) / 2.0))
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    def render(self):
        """强制渲染几帧，确保纹理数据已经从 GPU 拷贝回内存。"""
        for _ in range(3):
            self.base.graphicsEngine.renderFrame()

    def get_rgb(self):
        """返回 RGB 图像，shape=(H, W, 3)，dtype=uint8。"""
        self.render()
        data = self.color_tex.getRamImageAs("RGB")
        rgb = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        return np.flipud(rgb).copy()

    def get_depth(self):
        """返回米制深度图，shape=(H, W)，无效或远平面位置会接近 far。"""
        self.render()
        raw = self.depth_tex.getRamImage()
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size != self.width * self.height:
            # 某些驱动会以 8-bit 深度纹理返回，退化为 0~1 归一化深度。
            arr_u8 = np.frombuffer(raw, dtype=np.uint8)
            arr = arr_u8[: self.width * self.height].astype(np.float32) / 255.0
        z_buffer = arr.reshape((self.height, self.width))
        z_buffer = np.flipud(z_buffer).copy()
        z_ndc = z_buffer * 2.0 - 1.0
        depth = (2.0 * self.near * self.far) / (
            self.far + self.near - z_ndc * (self.far - self.near)
        )
        return depth.astype(np.float32)

    def depth_to_pointcloud(self, depth, rgb=None, max_depth=None, stride=1):
        """由深度图反投影生成世界坐标点云。"""
        if max_depth is None:
            max_depth = self.far * 0.98
        intr = self.intrinsics
        us, vs = np.meshgrid(np.arange(self.width), np.arange(self.height))
        us = us[::stride, ::stride]
        vs = vs[::stride, ::stride]
        z = depth[::stride, ::stride]
        valid = np.isfinite(z) & (z > self.near) & (z < max_depth)

        x_cam = (us[valid] - intr["cx"]) * z[valid] / intr["fx"]
        y_cam = z[valid]
        z_cam = -(vs[valid] - intr["cy"]) * z[valid] / intr["fy"]
        points_cam = np.column_stack((x_cam, y_cam, z_cam))
        points_world = points_cam @ self.cam_rotmat.T + self.cam_pos

        colors = None
        if rgb is not None:
            colors = rgb[::stride, ::stride][valid]
        return points_world.astype(np.float32), colors

    @contextmanager
    def _semantic_colors(self, part_infos):
        """临时把每个零件改成唯一纯色，用于渲染语义分割图。"""
        states = []
        for idx, info in enumerate(part_infos, start=1):
            model = info["model"]
            pdndp = model.pdndp
            old_color = pdndp.getColor()
            states.append((pdndp, old_color))
            color = id_to_color(idx)
            pdndp.setColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)
            pdndp.setLightOff()
            pdndp.setTextureOff()
        try:
            yield
        finally:
            for pdndp, old_color in states:
                pdndp.setColor(old_color)
                pdndp.clearLight()
                pdndp.clearTexture()

    def get_segmentation(self, part_infos):
        """返回语义分割 RGB 图、单通道 ID mask，以及自动标注列表。

        ID mask 约定：像素值 0=背景，1/2/3...=对应物体 id。
        """
        with self._semantic_colors(part_infos):
            seg_rgb = self.get_rgb()

        id_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        annotations = []
        for idx, info in enumerate(part_infos, start=1):
            color = np.array(id_to_color(idx), dtype=np.uint8)
            # 精确匹配分割颜色；bbox 由 mask 像素的最小外接矩形得到。
            obj_mask = np.all(seg_rgb == color, axis=2)
            id_mask[obj_mask] = idx
            ys, xs = np.where(obj_mask)
            if len(xs) == 0:
                bbox = None
                area = 0
            else:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                area = int(obj_mask.sum())
            annotations.append(
                {
                    "id": idx,
                    "name": info["name"],
                    "stl_path": info["stl_path"],
                    "mask_color": color.tolist(),
                    "bbox_xywh": bbox,
                    "visible_pixels": area,
                }
            )
        return seg_rgb, id_mask, annotations

    def capture_all(self, part_infos, output_dir, pointcloud_stride=2):
        """采集并保存 RGB、深度、点云、分割图、ID mask 和标注 JSON。"""
        os.makedirs(output_dir, exist_ok=True)

        rgb = self.get_rgb()
        depth = self.get_depth()
        points, colors = self.depth_to_pointcloud(depth, rgb=rgb, stride=pointcloud_stride)
        seg_rgb, id_mask, annotations = self.get_segmentation(part_infos)

        cv2.imwrite(os.path.join(output_dir, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        labeled_rgb = draw_labeled_rgb(rgb, annotations)
        cv2.imwrite(os.path.join(output_dir, "rgb_labeled.png"), cv2.cvtColor(labeled_rgb, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(output_dir, "depth_m.npy"), depth)
        save_depth_preview(depth, os.path.join(output_dir, "depth_preview.png"), self.far)
        cv2.imwrite(os.path.join(output_dir, "segmentation.png"), cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))
        # mask.png：单通道 ID 图（肉眼可能偏暗）；mask.npy：原始 uint8 数组，便于程序读取。
        # mask_preview.png：把 id 映射成彩色，方便肉眼检查。
        cv2.imwrite(os.path.join(output_dir, "mask.png"), id_mask)
        np.save(os.path.join(output_dir, "mask.npy"), id_mask)
        save_mask_preview(id_mask, os.path.join(output_dir, "mask_preview.png"))
        save_pointcloud_ply(os.path.join(output_dir, "pointcloud.ply"), points, colors)

        meta = {
            "camera": {
                "position": self.cam_pos.tolist(),
                "rotmat": self.cam_rotmat.tolist(),
                "intrinsics": self.intrinsics,
                "resolution": [self.width, self.height],
                "near": self.near,
                "far": self.far,
            },
            "annotations": annotations,
            "mask_files": {
                "png": "mask.png",
                "npy": "mask.npy",
                "preview": "mask_preview.png",
                "meaning": "pixel=0 background; pixel=id object instance",
            },
        }
        with open(os.path.join(output_dir, "annotations.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    def capture_learn_format(self, part_infos, output_dir, sample_id="0001", pointcloud_stride=2):
        """按学习标注格式导出：单张图片 + 每物体独立 mask + 带 6D 位姿的 JSON。

        目录结构（相对 output_dir）：
            {sample_id}.jpg
            annotations.json
            masks/{sample_id}_{id:02d}_{class}.png

        annotations.json 字段与常见抓取/检测学习格式对齐：
            image / width / height / objects[
                id, class, bbox[x1,y1,x2,y2], polygon, mask_path, pose_6d
            ]
        """
        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        rgb = self.get_rgb()
        depth = self.get_depth()
        points, colors = self.depth_to_pointcloud(depth, rgb=rgb, stride=pointcloud_stride)
        seg_rgb, id_mask, _ = self.get_segmentation(part_infos)

        image_name = f"{sample_id}.jpg"
        save_image_unicode(
            os.path.join(output_dir, image_name),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        # 额外保留一份便于人工检查的标注预览（不进学习 JSON）。
        preview_anns = []
        objects = []
        for idx, info in enumerate(part_infos, start=1):
            obj_mask = (id_mask == idx).astype(np.uint8)
            ys, xs = np.where(obj_mask > 0)
            class_name = info["name"]
            # 文件名只用 ASCII（id），类别中文放在 JSON 的 class 字段，
            # 避免 Windows 下 cv2.imwrite 对中文路径静默失败。
            mask_rel = f"masks/{sample_id}_{idx:02d}.png"
            mask_abs = os.path.join(output_dir, mask_rel)
            # 二值 mask：前景 255，背景 0；用 imencode+tofile 兼容 Unicode 路径。
            save_image_unicode(mask_abs, (obj_mask * 255).astype(np.uint8))

            if len(xs) == 0:
                bbox = None
                polygon = []
                visible_pixels = 0
            else:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                bbox = [x1, y1, x2, y2]
                polygon = mask_to_polygon(obj_mask)
                visible_pixels = int(obj_mask.sum())

            pose_6d = pose_6d_from_part_info(info)
            objects.append(
                {
                    "id": idx,
                    "class": class_name,
                    "bbox": bbox,
                    "polygon": polygon,
                    "mask_path": mask_rel.replace("\\", "/"),
                    "pose_6d": pose_6d,
                    "visible_pixels": visible_pixels,
                    "stl_path": info.get("stl_path"),
                }
            )
            preview_anns.append(
                {
                    "id": idx,
                    "name": class_name,
                    "bbox_xywh": None
                    if bbox is None
                    else [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1],
                    "mask_color": list(id_to_color(idx)),
                    "visible_pixels": visible_pixels,
                }
            )

        labeled_rgb = draw_labeled_rgb(rgb, preview_anns)
        save_image_unicode(
            os.path.join(output_dir, f"{sample_id}_labeled.png"),
            cv2.cvtColor(labeled_rgb, cv2.COLOR_RGB2BGR),
        )
        np.save(os.path.join(output_dir, "depth_m.npy"), depth)
        save_depth_preview(depth, os.path.join(output_dir, "depth_preview.png"), self.far)
        save_image_unicode(
            os.path.join(output_dir, "segmentation.png"),
            cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR),
        )
        save_pointcloud_ply(os.path.join(output_dir, "pointcloud.ply"), points, colors)

        meta = {
            "image": image_name,
            "width": self.width,
            "height": self.height,
            "objects": objects,
            "camera": {
                "position": self.cam_pos.tolist(),
                "rotmat": self.cam_rotmat.tolist(),
                "intrinsics": self.intrinsics,
                "near": self.near,
                "far": self.far,
            },
        }
        with open(os.path.join(output_dir, "annotations.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta


def sanitize_filename(name):
    """去掉 Windows 非法文件名字符，保证 mask 文件名可写。"""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(name)).strip()
    return cleaned or "object"


def save_image_unicode(path, image_bgr_or_gray):
    """保存图片；兼容 Windows 下含中文的路径（cv2.imwrite 对此常静默失败）。"""
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, buf = cv2.imencode(ext, image_bgr_or_gray)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    buf.tofile(path)


def mask_to_polygon(binary_mask, epsilon_ratio=0.01):
    """从二值 mask 提取外轮廓多边形（像素坐标，顺时针/逆时针均可）。"""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    if len(approx) < 3:
        approx = contour
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def pose_6d_from_part_info(info):
    """从场景摆放位姿得到学习格式 pose_6d（世界系，米 / 弧度）。"""
    pos = np.asarray(info["pos"], dtype=float)
    rotmat = np.asarray(info["rotmat"], dtype=float)
    roll, pitch, yaw = rm.rotmat_to_euler(rotmat)
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def id_to_color(idx):
    """将物体 ID 编码为唯一且远离黑色的 RGB 颜色。

    旧实现用 (idx, 0, 0)，物体 1 的颜色是 (1,0,0)，几乎等于背景黑，
    容差匹配时会把大片背景算进 mask，导致 2D 框撑满整张图。
    """
    # 用互质步长打散通道，并抬高亮度，避免接近 (0,0,0)。
    r = (37 * idx + 40) % 200 + 40
    g = (67 * idx + 60) % 200 + 40
    b = (97 * idx + 80) % 200 + 40
    return (int(r), int(g), int(b))


def save_depth_preview(depth, path, far):
    """保存便于查看的 8-bit 深度预览图。"""
    depth_vis = np.clip(depth / far, 0.0, 1.0)
    depth_vis = (255 * (1.0 - depth_vis)).astype(np.uint8)
    cv2.imwrite(path, depth_vis)


def save_mask_preview(id_mask, path):
    """把单通道 ID mask 转成彩色预览图，方便肉眼检查。

    背景保持黑色；每个物体 id 使用与 segmentation 相同的 id_to_color 配色。
    """
    h, w = id_mask.shape
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in np.unique(id_mask):
        if idx == 0:
            continue
        preview[id_mask == idx] = id_to_color(int(idx))
    cv2.imwrite(path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def draw_labeled_rgb(rgb, annotations):
    """在 RGB 图上绘制 bbox 和物体名称，生成便于人工查看的标注预览图。"""
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    font = load_chinese_font(size=11)#调整字体大小

    for ann in annotations:
        bbox = ann.get("bbox_xywh")
        if bbox is None or ann.get("visible_pixels", 0) <= 0:
            continue
        x, y, w, h = [int(v) for v in bbox]
        color = tuple(int(v) for v in ann.get("mask_color", [255, 0, 0]))
        label = ann["name"]

        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y - text_h - 4)
        draw.rectangle([x, label_y, x + text_w + 4, label_y + text_h + 2], fill=(255, 255, 255))
        draw.rectangle([x, label_y, x + text_w + 4, label_y + text_h + 2], outline=color, width=1)
        draw.text((x + 2, label_y + 1), label, fill=(0, 0, 0), font=font)

    return np.asarray(image)


def load_chinese_font(size=16):
    """优先加载 Windows 常见中文字体，保证中文标签能正常显示。"""
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_pointcloud_ply(path, points, colors=None):
    """保存 ASCII PLY 点云。"""
    if colors is None:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    colors = colors.astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
