import json
import os
import cv2
from pathlib import Path

# ========== 配置路径 ==========
JSON_DIR = r"F:\wrs-main-competition\wrs\drivers\devices\realsense\block_images\labels\train"
IMAGE_DIR = r"F:\wrs-main-competition\wrs\drivers\devices\realsense\block_images\images\train"
OUTPUT_LABEL_DIR = r"F:\wrs-main-competition\wrs\drivers\devices\realsense\block_images\labels\train"
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 类别名到ID的映射（请与 data.yaml 中的顺序一致）
class_names = ['red', 'green', 'blue', 'yellow', 'orange']
class_to_id = {name: idx for idx, name in enumerate(class_names)}

def convert_polygon_to_bbox(points):
    """将多边形点集转换为 YOLO 格式的边界框（归一化）"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    # 归一化（图像尺寸需要在调用时传入）
    return xmin, ymin, xmax, ymax

def main():
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个 JSON 文件。")

    for json_file in json_files:
        json_path = os.path.join(JSON_DIR, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取图片文件名（从 imagePath 字段）
        image_filename = data.get('imagePath')
        if not image_filename:
            print(f"警告: {json_file} 缺少 imagePath，跳过")
            continue

        # 对应的图片完整路径
        img_path = os.path.join(IMAGE_DIR, image_filename)
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在 {img_path}，跳过")
            continue

        # 读取图片尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，跳过")
            continue
        h, w = img.shape[:2]

        shapes = data.get('shapes', [])
        if not shapes:
            # 没有标注，跳过（不生成空的txt文件）
            continue

        txt_filename = os.path.splitext(json_file)[0] + '.txt'
        txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_filename)

        with open(txt_path, 'w') as out_f:
            for shape in shapes:
                label = shape.get('label')
                if label not in class_to_id:
                    print(f"警告: 未知标签 '{label}' 在 {json_file}，跳过该标注")
                    continue
                class_id = class_to_id[label]
                points = shape.get('points', [])
                if len(points) < 3:
                    continue
                # 计算多边形的最小外接矩形
                xmin, ymin, xmax, ymax = convert_polygon_to_bbox(points)
                # 归一化
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                # 写入
                out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"已生成: {txt_path}")

    print("转换完成。")

if __name__ == '__main__':
    main()