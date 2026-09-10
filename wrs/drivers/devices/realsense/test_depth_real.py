import cv2
import numpy as np
from realsense_d400s import RealSenseD405
from ultralytics import YOLO

MODEL_PATH = r"F:\wrs-main-competition\runs\detect\train-6\weights\best.pt"
model = YOLO(MODEL_PATH)
cam = RealSenseD405(resoultion='mid')

while True:
    verts, colors, depth, color = cam.get_pcd_texture_depth()
    if color is None or depth is None:
        continue
    # 确保深度图与彩色图尺寸一致（如果分辨率不同，必须缩放）
    if depth.shape[:2] != color.shape[:2]:
        print(f"警告: 深度图尺寸 {depth.shape} vs 彩色图尺寸 {color.shape}")
        depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    results = model(color, conf=0.5)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if 0 <= cx < depth.shape[1] and 0 <= cy < depth.shape[0]:
            z = depth[cy, cx]
            print(f"框中心深度: {z} mm (分辨率: {depth.shape})")
            # 在深度图上画点
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.circle(depth_norm, (cx, cy), 3, (255,0,0), -1)
            cv2.imshow("Depth", depth_norm)
        cv2.rectangle(color, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("Color", color)
    if cv2.waitKey(1) == ord('q'):
        break
cam.stop()
cv2.destroyAllWindows()
