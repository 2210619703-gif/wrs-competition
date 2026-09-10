# 模板区专用 Segment 数据集

与通用 `block_images` 分开，训练时用 `--name train_template`，
**不会覆盖** `runs/segment/.../train/weights/best.pt`。

## 目录

```text
template_seg/
  images/train|val   ← 只放模板区固定视角照片
  labels/train|val   ← LabelMe JSON 存这里（Change Save Dir）
  data.yaml
```

## 标注（Seg 多边形）

```powershell
cd F:\wrs-main-competition
conda activate yolov8
labelme wrs/drivers/devices/realsense/block_images/template_seg/images/train
```

1. Open Dir → `template_seg/images/train`
2. **Change Save Dir** → `template_seg/labels/train`
3. Create Polygons，标签：`red/green/blue/yellow/orange`
4. val 同理

## 转换 + 训练（新权重名）

```powershell
python wrs/drivers/devices/realsense/block_images/labelme_json_to_seg.py --root wrs/drivers/devices/realsense/block_images/template_seg --inplace

python train_blocks_seg.py --data wrs/drivers/devices/realsense/block_images/template_seg/data.yaml --name train_template
```

产出：`runs/segment/runs/segment/train_template/weights/best.pt`  
（原 `train/weights/best.pt` 不动）

## 接入配置

训完后把 `er4ia_competition_config.py` 里：

```python
YOLO_TEMPLATE_WEIGHT_PATH = .../train_template/weights/best.pt
```
