# 感知模块 → 决策 接口说明

视觉识别模块的输出参数名为 **`vision_objects`**，类型为 **JSON 字符串**（内容是 JSON 数组）。

大模型 / 工作流同学读取该 JSON，即可获知当前场景中各工业零件的类别、位置与朝向，用于任务规划、语言推理或机器人抓取决策。

---

## 1. 数据格式

### 有物体时

```json
[
  {
    "name": "gear",
    "conf": 0.967,
    "coord": [0.389, 0.023],
    "z": 0.305,
    "yaw": 38.5,
    "rotate_matrix": [[0.78, -0.62, 0.0], [0.62, 0.78, 0.0]]
  }
]
```

### 无物体时

```json
[]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 零件类别（见下表） |
| `conf` | float | YOLO 检测置信度，范围 0~1 |
| `coord` | [float, float] | 世界坐标 **X、Y**，单位 **米** |
| `z` | float | 世界坐标 **Z**，单位 **米**（桌面约 0） |
| `yaw` | float | 绕 Z 轴旋转角，单位 **度** |
| `rotate_matrix` | 2×3 float | 旋转矩阵前两行（桌面 yaw，Z 轴朝上） |

### 坐标系

- **WRS 仿真世界坐标系**
- 单位：**米（m）**
- Z 轴向上，桌面高度 Z ≈ 0
- 完整 3D 位置为 `(coord[0], coord[1], z)`

---

## 2. 类别（8class，当前使用）

| name | 含义 | STL 文件夹 |
|------|------|------------|
| `bearing` | 轴承 | Bearings |
| `gear` | 齿轮 | Gears |
| `machine_tool` | 机床/工具 | Machines & Tools |
| `nut` | 螺母 | Nuts |
| `rivet` | 铆钉 | Rivets |
| `roller_bearing` | 滚子轴承 | Roller bearings |
| `screw_bolt` | 螺丝螺栓 | Screws and bolts |
| `washer` | 垫圈 | Washers |

共 **8 类**，对应 **131** 个 STL 模型。若需合并为 4 类，使用 `--class-set merged4`。

---

## 3. 示例文件

项目内已提供两份样例（可直接用于联调）：

| 文件 | 说明 |
|------|------|
| `samples/vision_objects_example.json` | 检测到 4 个物体 |
| `samples/vision_objects_empty.json` | 空场景 `[]` |

---

## 4. 如何生成 JSON

环境：`conda activate wrs`，在项目根目录 `F:\tiaozhanbei` 执行。

### 单张图片导出（推荐联调）

需要成对的 RGB 图片 + 深度图（dataset 内已有）：

```powershell
python export_vision_objects.py ^
  --image dataset/images/val/000000.png ^
  --class-set 8class ^
  --weights runs/detect/train-2/weights/best.pt ^
  --out vision.json
```

（省略 `--class-set` 与 `--weights` 时默认即为 8 类 + `train-2` 权重。）

### 仿真实时更新

```powershell
python test_industrial_detection.py ^
  --mode sim --pose --class-set 8class ^
  --weights runs/detect/train-2/weights/best.pt ^
  --vision-json-out vision.json
```

仿真运行时约每 30 帧覆盖写入 `vision.json`。

### 指定外部图片

```powershell
python test_industrial_detection.py ^
  --mode image --pose ^
  --image dataset/images/val/000000.png ^
  --class-set 8class ^
  --vision-json-out vision.json
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--conf` | 0.38 | 置信度阈值，越高漏检越多 |
| `--pretty` | 关 | 多行缩进 JSON（调试用） |
| `--out` | stdout | 输出文件路径 |

---

## 5. 下游「三维测试」格式（推荐对接）

三维测试运行框需要的输入与上文内部世界坐标格式不同，请使用 **`downstream3d`**：

```json
[
  {
    "name": "机床工具",
    "category": "工具",
    "position": {
      "x_min": 120, "y_min": 180, "x_max": 280, "y_max": 360,
      "z_min": 520, "z_max": 680
    },
    "confidence": 0.91,
    "state": "未拾取"
  }
]
```

| 字段 | 说明 |
|------|------|
| `name` | 中文零件名 |
| `category` | 类别：工具 / 零件 / 紧固件 |
| `position.x/y_*` | 图像像素包围盒 |
| `position.z_*` | 深度范围（mm，来自深度图 ROI） |
| `confidence` | 检测置信度 |
| `state` | 默认 `未拾取`（拾取状态需控制侧维护） |

样例：`samples/vision_objects_downstream3d_example.json`

```powershell
# 仿真导出（默认已是 downstream3d）
python test_industrial_detection.py --mode sim --pose --class-set 8class --weights runs/detect/train-2/weights/best.pt --vision-json-out vision.json

# 单图导出
python export_vision_objects.py --image dataset/images/val/000000.png --format downstream3d --pretty --out vision.json
```

说明：当前检测类别是工业零件（bearing/gear/…），**没有单独的「盒子」类**；若下游指令需要盒子，需控制侧另建容器目标，或我们后续加盒子检测。

---

## 6. 对接建议

1. **读取方式**：大模型侧读取 `vision.json`，或接收 JSON 字符串参数。
2. **空场景**：务必处理 `[]`。
3. **多物体**：数组元素独立；`state` 由控制/规划侧更新。
4. **模型权重**：`runs/detect/train-2/weights/best.pt`（8 类）。

---

## 7. 代码位置（视觉组内部参考）

| 文件 | 作用 |
|------|------|
| `industrial_pose_utils.py` | 位姿估计 + `vision_objects_json()` 格式转换 |
| `export_vision_objects.py` | 单张图导出 JSON 命令行工具 |
| `test_industrial_detection.py` | 仿真/图片检测，`--vision-json-out` 写文件 |

格式定义函数：`industrial_pose_utils.vision_objects_json()`
