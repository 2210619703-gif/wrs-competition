# 分层感知流水线

标准 YOLO detect 只有 `class / bbox / score`，不够仿真与抓取。本项目拆成四层。

## 1. YOLO-seg（感知主干）

- **训**：`train_annotations_yolo.py`，标签来自 `annotations.json` 的 `class` + `polygon`
- **权重**：`best.pt`（YOLO26s-seg，124 类中文细分类；同内容也可叫 `权重/yolo26s-seg-fine.pt`）
- **小件补检**：`权重/yolo26s-seg-tiny19.pt`（19 类近景螺丝/铆钉/垫圈/滑动轴承）。只在切片上跑，框面积须小于全图 8%，且与大模型框 IoU < 0.3 才采纳，**不覆盖**已有框
- **出**：`class`、`bbox`、`score`、`polygon`、`mask`、`visible_pixels`

## 2. 状态头（pose_state）

- **训**：`export_dataset_learn_state_cls.py` → `train_state_head.py`
- **权重**：`权重/state-head-yolo26.pt`
- **出**：`state` ∈ {normal, inverted, fallen}
- 也可把类别扩成 `螺丝刀_fallen`，本包采用「检测后再分类」，不改 124 类 id

## 3. 相机标定 + 几何（坐标）

- **不训练**
- 输入：mask（优先）或 bbox + `depth_m.npy` + `camera.intrinsics/position/rotmat`
- 反投影约定与 `SimDataCamera.depth_to_pointcloud` 相同（Panda 相机系 → 桌面世界系）
- **出**：`coordinate = [x, y]`，同时写入 `pose_6d.x y z`（桌面件 z 近 0 则置 0）

## 4. 朝向

- **state 几何纠正**：工具趴桌→normal；螺栓横躺→fallen（覆盖状态头误判）
- **yaw**：圆件用世界 XY 主轴；细长件用图像长轴落到桌面
- **roll / pitch**：由纠正后的 `state` 映射  
- **stl_path**：扫描 `industrial_models`，并可用旁路 annotations 按 IoU 对齐到具体 STL
  - normal → (0, 0)  
  - fallen → (π/2, 0)  
  - inverted → (π, 0)  
- 早期不必上完整 6D 位姿网络

## 代码填的字段

- `id`：检测顺序
- `stl_path`：`industrial_models` 按中文类名查表
- `target_container`：`scene_layout.storage_box`（固定收纳盒）
- `scene_layout` / `camera` / `seed`：采集 JSON 原样带回
- `score`：YOLO `conf`

## 决策微调数据

`annotations_to_vision.py` 把上述 JSON 转成 `VISION_API.md` 中的 `vision_objects` 数组，可供规划 / 大模型微调。
