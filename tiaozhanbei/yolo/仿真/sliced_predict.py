"""重叠切片推理：全图 imgsz=1280 + 局部 tile，NMS 合并。用于小零件补召回。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Det:
    cls_id: int
    cls_name: str
    conf: float
    bbox: list[int]
    mask: np.ndarray | None = field(default=None, repr=False)


def _windows(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    tile = min(tile, h, w)
    step = max(1, int(tile * (1.0 - overlap)))
    ys = list(range(0, max(1, h - tile + 1), step))
    xs = list(range(0, max(1, w - tile + 1), step))
    if ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    if xs[-1] != w - tile:
        xs.append(max(0, w - tile))
    out = []
    for y in ys:
        for x in xs:
            out.append((x, y, x + tile, y + tile))
    # unique
    seen = set()
    uniq = []
    for w0 in out:
        if w0 not in seen:
            seen.add(w0)
            uniq.append(w0)
    return uniq


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def nms_classwise(dets: list[Det], iou_thr: float) -> list[Det]:
    by: dict[int, list[Det]] = {}
    for d in dets:
        by.setdefault(d.cls_id, []).append(d)
    kept: list[Det] = []
    for group in by.values():
        group = sorted(group, key=lambda x: -x.conf)
        used = [False] * len(group)
        for i, a in enumerate(group):
            if used[i]:
                continue
            kept.append(a)
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if _iou(a.bbox, group[j].bbox) >= iou_thr:
                    used[j] = True
    return sorted(kept, key=lambda x: -x.conf)


def _predict_crop(model, crop_bgr, conf, iou, imgsz, names) -> list[tuple]:
    r = model.predict(crop_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    boxes = r.boxes
    masks = r.masks
    out = []
    n = 0 if boxes is None else len(boxes)
    for j in range(n):
        box = boxes[j]
        xyxy = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
        mask = None
        if masks is not None and j < len(masks.data):
            m = masks.data[j].cpu().numpy()
            if m.shape[:2] != crop_bgr.shape[:2]:
                m = cv2.resize(m.astype(np.float32), (crop_bgr.shape[1], crop_bgr.shape[0]))
            mask = (m > 0.5).astype(np.uint8)
        cid = int(box.cls)
        out.append((cid, names.get(cid, str(cid)), float(box.conf), xyxy, mask))
    return out


def sliced_predict(
    model,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz_full: int = 1280,
    tile: int = 640,
    overlap: float = 0.25,
    nms_iou: float = 0.5,
    with_masks: bool = False,
) -> list[Det]:
    """全图 1280 + 重叠切片，按类 NMS。"""
    h, w = img_bgr.shape[:2]
    names = model.names or {}
    full: list[Det] = []
    tiles: list[Det] = []

    for cid, cname, sc, xyxy, mask in _predict_crop(
        model, img_bgr, conf, iou, imgsz_full, names
    ):
        bbox = [int(round(v)) for v in xyxy]
        bbox[0], bbox[1] = max(0, bbox[0]), max(0, bbox[1])
        bbox[2], bbox[3] = min(w - 1, bbox[2]), min(h - 1, bbox[3])
        mfull = None
        if with_masks and mask is not None:
            if mask.shape[:2] != (h, w):
                mfull = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mfull = mask
        full.append(Det(cid, cname, sc, bbox, mfull))

    for x1, y1, x2, y2 in _windows(h, w, tile, overlap):
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        for cid, cname, sc, xyxy, mask in _predict_crop(
            model, crop, conf, iou, tile, names
        ):
            bbox = [
                int(round(xyxy[0] + x1)),
                int(round(xyxy[1] + y1)),
                int(round(xyxy[2] + x1)),
                int(round(xyxy[3] + y1)),
            ]
            bbox[0], bbox[1] = max(0, bbox[0]), max(0, bbox[1])
            bbox[2], bbox[3] = min(w - 1, bbox[2]), min(h - 1, bbox[3])
            # 与全图同框重叠则丢掉切片，避免高分错框把全图正确框挤掉
            if any(d.cls_id == cid and _iou(bbox, d.bbox) >= nms_iou for d in full):
                continue
            mfull = None
            if with_masks and mask is not None:
                mfull = np.zeros((h, w), dtype=np.uint8)
                mh, mw = mask.shape[:2]
                mfull[y1 : y1 + mh, x1 : x1 + mw] = mask
            tiles.append(Det(cid, cname, sc, bbox, mfull))

    return nms_classwise(full + tiles, nms_iou)


def tile_only_predict(
    model,
    img_bgr: np.ndarray,
    conf: float = 0.4,
    iou: float = 0.5,
    tile: int = 640,
    overlap: float = 0.25,
    nms_iou: float = 0.5,
    with_masks: bool = True,
) -> list[Det]:
    """只跑重叠切片，不跑全图。给近景小件专用权重用。"""
    h, w = img_bgr.shape[:2]
    names = model.names or {}
    tiles: list[Det] = []
    for x1, y1, x2, y2 in _windows(h, w, tile, overlap):
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        for cid, cname, sc, xyxy, mask in _predict_crop(
            model, crop, conf, iou, tile, names
        ):
            bbox = [
                int(round(xyxy[0] + x1)),
                int(round(xyxy[1] + y1)),
                int(round(xyxy[2] + x1)),
                int(round(xyxy[3] + y1)),
            ]
            bbox[0], bbox[1] = max(0, bbox[0]), max(0, bbox[1])
            bbox[2], bbox[3] = min(w - 1, bbox[2]), min(h - 1, bbox[3])
            mfull = None
            if with_masks and mask is not None:
                mfull = np.zeros((h, w), dtype=np.uint8)
                mh, mw = mask.shape[:2]
                mfull[y1 : y1 + mh, x1 : x1 + mw] = mask
            tiles.append(Det(cid, cname, sc, bbox, mfull))
    return nms_classwise(tiles, nms_iou)
