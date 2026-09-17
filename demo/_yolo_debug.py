# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2
from fruit_vision import refine_fruit_class

img = cv2.imread(r"f:\wrs-main-competition\demo\outputs\vision\fruit_20260917_155442_250\detect_preview.jpg")
m = YOLO(r"f:\wrs-main-competition\demo\yolo26s-seg.pt")
r = m.predict(img, conf=0.08, imgsz=640, classes=[46, 47, 49], verbose=False)[0]
h, w = img.shape[:2]
for i, b in enumerate(r.boxes or []):
    en = r.names[int(b.cls)]
    mm = r.masks.data[i].cpu().numpy()
    if mm.shape[:2] != (h, w):
        mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = mm > 0.5
    print(en, f"{float(b.conf):.2f}", "->", refine_fruit_class(en, img, mask))
