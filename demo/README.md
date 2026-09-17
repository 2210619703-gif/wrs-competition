# 水果真机演示（demo）

用 fafu / Panthera-HT 抓香蕉、苹果、橙子，放到 A / B 收纳筐。

和 `tiaozhanbei/real` 工业件链路的差别：

- 相机是**眼在手上**，不是桌面固定俯视
- YOLO 用官方 `yolo26s-seg.pt`，只保留香蕉 / 苹果 / 橙子
- **不用**工业件仿真场景、grasp pickle、dataset_learn
- 规划仍走 **WRS 仿真臂 IK**（顶抓 + 关节插值），下发仍走 `tiaozhanbei/real` 的 `servo_j`
- 放置用现场示教的 A / B 筐关节角，不建仿真收纳盒

默认 **dry-run**，不勾「允许真机执行」、不加 `--execute` 不会动抓取。

---

## 目录

| 文件 | 作用 |
|------|------|
| `fruit_config.py` | 唯一配置：示教位、RUN_MODE、速度、YOLO、工作区 |
| `fruit_demo_ui.py` | 主界面（模仿 `real_demo_ui.py`） |
| `fruit_real.py` | 命令行：观察位 / 零位 / A·B 框 / 识别 / 抓放 |
| `fruit_vision.py` | D405 + YOLO-seg + 眼在手上 → 世界坐标 |
| `fruit_sim_planner.py` | WRS Panthera-HT IK 规划，写出 `outputs/traj.json` |
| `fruit_cmd.py` | 口令解析 |
| `fruit_stt.py` | 谷歌语音（zh-CN，要外网） |
| `fruit_yolo_realtime.py` | **D405 live 窗口**（对齐 `d405_camera.py --live / --yolo`） |
| `fruit_seg_demo.py` | 只测分割：图片 / 笔记本摄像头，不动臂 |
| `yolo26s-seg.pt` | 官方 COCO 分割权重（没有会自动下） |
| `hand_eye_calib.py` | **眼在手上标定**：点云对齐仿真臂/桌面，写出 `handeye_eye_in_hand.json` |
| `read_joints.py` | **读真机关节角 + 法兰世界坐标**，抄到 LOOK / A / B |
| `find_look_pose.py` | **搜 LOOK 位**：相机尽量俯视桌面、看全水果区、夹爪盲区甩到区外 |
| `handeye_eye_in_hand.json` | **标定完成后生成**，仓库里默认没有 |
| `real/` | 真机执行层（从 tiaozhanbei/real 拷出：相机、`servo_j`、轨迹适配） |

产物在 `outputs/`：`traj.json`、`plan.json`、`vision/`。

---

## 1. 环境

仓库根目录，和真机工业件同一套 Python / SDK：

```bash
pip install ultralytics opencv-python numpy pyrealsense2
pip install SpeechRecognition sounddevice
pip install pillow
```

机械臂 SDK 在 `fafu_arm_sdk-main/fafu_robot_python`。不在默认位置时设置 `FAFU_SDK_DIR`。

语音要能访问 Google；打字下指令可以不装语音包。

---

## 2. 上机前必须做的三件事

### 2.1 填三个示教位

打开 `fruit_config.py`，把现场 J1~J6 **角度（度）** 填进去。读数（只读、不动臂）：

```bash
python demo/read_joints.py
```

手推到观察位 / A 筐 / B 筐各读一次。输出里有可粘贴的：

```python
LOOK_CONF_DEG  = (..., ..., ..., ..., ..., ...)
BIN_A_CONF_DEG = (..., ..., ..., ..., ..., ...)
BIN_B_CONF_DEG = (..., ..., ..., ..., ..., ...)
```

界面里连上臂后也可以点「读关节角 / 法兰坐标」。还是 `None` 时，去该位的模式会直接报错。

### 2.2 眼在手上标定

相机装在臂上。把臂摆到能看见桌面的姿势，然后：

```bash
python demo/hand_eye_calib.py
```

会连 D405 + 真机，把点云变到世界系画在仿真臂/桌面上。键盘 `f` 贴桌面，`wasd/qe` 平移，`zxcvbn` 旋转，对上之后 `p` 保存到：

`demo/handeye_eye_in_hand.json`

（法兰 → 相机 4×4，`mode: eye_in_hand`）。**不要**用 `tiaozhanbei/real/hand_eye_calib.py`，那是固定俯视。

世界坐标：`T_base_cam = FK(当前关节的法兰) @ affine_mat`。

没有这份文件，或里面仍是单位阵，识别会失败，不会拿假矩阵去抓。按键说明见脚本开头注释。

### 2.3 夹爪标定（建议）

和工业件共用 `tiaozhanbei/real/gripper_calib.json`：

```bash
python tiaozhanbei/real/calibrate_gripper.py --execute
```

---

## 3. RUN_MODE

改 `fruit_config.py` 里的 `RUN_MODE`，或命令行 `--mode`。

| 模式 | 做什么 |
|------|--------|
| `look` | 只到观察位（张开夹爪） |
| `zero` | 只到电机零位 |
| `bin_a` / `bin_b` | 只到 A / B 筐，方便核对示教 |
| `vision` | 观察位拍照 + YOLO，写出 annotations，不抓 |
| `once` | 识别 → 规划 → 按口令抓放一次 |
| `interactive` | 打开界面（默认） |

其它开关也在配置文件：

| 项 | 默认 | 含义 |
|----|------|------|
| `EXECUTE_ON_REAL` | `False` | 是否真下发抓取 |
| `USE_SERVO_J` | `True` | 用 servo_j |
| `DEFAULT_MOVE_SPEED` | ≤8 | 首次建议保持低速 |
| `WORKSPACE_BOUNDS_M` | 见配置 | 世界系工作区，按桌面改 |

---

## 4. 怎么跑

仓库根目录。

### 界面（日常）

```bash
python demo/fruit_demo_ui.py
```

左侧录音（谷歌）+ 观察位 / 零位 / A框 / B框 + 日志；右侧 D405 直播。启动本程序的终端也可以打字回车，`q` 退出。

不勾「允许真机执行」只规划、打印。勾了还会再确认一次。界面有急停。

没有机械臂时：

```bash
python demo/fruit_demo_ui.py --no-connect
```

### 命令行

```bash
python demo/fruit_real.py --mode look
python demo/fruit_real.py --mode zero
python demo/fruit_real.py --mode bin_a
python demo/fruit_real.py --mode vision
python demo/fruit_real.py --mode once --cmd "把苹果放进A框"
python demo/fruit_real.py --mode once --cmd "把苹果放进A框" --execute
```

`--mode look` 到位后会直接开实时 YOLO 窗口（就是下面那个），**关窗（`q` / `Esc`）才回静置位**，
窗口里按 `j` 打印关节角——对相机、核观察位、抄示教位一个窗口全办了。用的是同一条臂连接，
不会再抢一次串口。相机开不起来自动退回「按回车继续」。开关是 `LOOK_SHOW_LIVE`，
临时不要就 `--no-live`。

### D405 实时窗口（对齐工业件 live）

和 `python tiaozhanbei/real/d405_camera.py --live --yolo` 同一套 OpenCV 窗口，相机走同一份 `D405Camera`：

```bash
python demo/fruit_yolo_realtime.py --live
python demo/fruit_yolo_realtime.py --yolo
```

`--live` 是彩色+深度并排；`--yolo` 在彩色上画香蕉/苹果/橙子（不写也默认开 YOLO）。`q` 退出，`[` `]` 调阈值，`-` `=` 调输入尺寸，`s` 存图。绿色过阈值，橙色分数不够。不连机械臂、不用手眼。

### 只测 YOLO（图片 / 笔记本摄像头）

```bash
python demo/fruit_seg_demo.py --image 某张图.jpg
python demo/fruit_seg_demo.py --webcam
```

---

## 5. 可以说哪些指令

语音或打字：

- `帮我把苹果放进A框`
- `把香蕉放到B筐`
- `把橙子放进A收纳筐`
- `所有水果放进A框`
- `抓苹果`（只抓起来，不去筐）

水果：苹果 / 香蕉 / 橙子（橘子、桔子当橙子）。筐：A / B。

---

## 6. 一条指令实际在干什么

```
到观察位
  → D405 拍一帧（深度对齐彩色）
  → YOLO26s-seg 只留 banana / apple / orange 的 mask
  → 眼在手上变到世界系
  → 口令选出目标水果和 A/B 筐
  → WRS IK 先试竖直顶抓、够不着就斜着：接近 → 抓取 → 抬起 → 插值到示教筐
  → traj_adapter 裁限位、抽帧、夹爪事件
  → servo_j 下发（或 dry-run 打印）
  → 回到观察位
```

抓取是开环的：按拍照那一瞬的位置规划。桌面动过要重新说一句。

转场（去下一个水果、夹着水果去料框）会先把 TCP 竖直抬到 `carry_z_m()`，在高处平移，
再下来。放到框时这个高度至少是 `BIN_HEIGHT_M + FRUIT_BELOW_TCP_M + PLACE_CLEAR_M`
（12cm 框 + 5cm 下垂 + 6cm 余量，并且不低于 `TRANSIT_Z_M`）。抬不上去
（IK 无解或解出来是另一个臂型）就退回直连。

**画面里有多个同类水果时**（比如三个橙子都要进 A 框）：一个一个做，每个都是完整的
接近 → 合爪 → 抬起 → 去筐 → 松爪，做完直接去下一个，中间不回观察位。顺序按 YOLO
识别置信度从高到低，**和水果在桌上的位置无关**。某个失败不中断整批，跳过做下一个。

`RECAPTURE_EACH_FRUIT` 决定拍几次（`--recapture` / `--no-recapture` 可临时覆盖）：

| | 拍照 | 适合 |
|------|------|------|
| `False`（默认） | 只拍一次，全部规划成一批 | 水果摆得开、互不干扰，快 |
| `True` | 每抓完一个回观察位重拍重新规划 | 摆得密、怕抓第一个时碰动别的 |

`False` 时后面几个用的是第一帧的旧坐标，碰动了就会抓偏。`True` 慢一轮视觉的时间，
但坐标永远是最新的；某个水果连着抓不走会被记下跳过，最多转 `MAX_PICK_ROUNDS` 轮。

---

## 7. 建议顺序

1. `python tiaozhanbei/real/check_offline.py`
2. `python demo/read_joints.py` 填三个示教位，`--mode look / bin_a / bin_b` 空载核对
3. `python demo/hand_eye_calib.py`，确认 `handeye_eye_in_hand.json` 不是单位阵
4. `python demo/find_look_pose.py` 挑个俯视的 LOOK 位（标定是眼在手上的，换位不用重标）
5. `--mode vision`，看 `outputs/vision/fruit01/detect_preview.jpg` 框和坐标
6. 不 `--execute` 跑 `--mode once`，看 `outputs/plan.json`
7. 低速 `--execute`，手放在急停上

---

## 8. 常见问题

| 现象 | 处理 |
|------|------|
| 还没填 LOOK / A / B | 去该位会报错，先改 `fruit_config.py` |
| 找不到手眼文件 / 单位阵 | 标定后再写入 `handeye_eye_in_hand.json` |
| 谷歌转写失败 / 未接入外网 | Python 不走 Windows 系统代理。`fruit_config.STT_PROXY` 填 Clash「混合代理端口」，你这台是 `http://127.0.0.1:7897`。Clash 要开着，规则别把 google 走 DIRECT |
| 检出杯子、人 | 代码已按 COCO 46/47/49 过滤；确认用的是本目录 `yolo26s-seg.pt` |
| 抓偏 / 抓空 | 先看预览图坐标；工作区 `WORKSPACE_BOUNDS_M` 是否把件滤掉；开环下桌面不能动 |
| 每次都偏同一个方向 | 标定不一定有问题。点云只有朝相机那半边，识别坐标天生偏向相机侧，已按 `SURFACE_TO_CENTER_K` 沿视线补回中心；还差就调它，剩下的固定偏差填 `WORLD_TRIM_M` |
| 半空就合爪 | 相机只看得见件朝上那面，抓取点要从顶面往下扎。调大 `GRASP_DEPTH_FRAC` / `GRASP_EXTRA_SINK_M` |
| 合爪只夹到橙子上 1/3 | 深度飞点把顶面抬高了。规划器会用中位数+半径封顶，再按件高 62% 下探；还浅就加大 `GRASP_EXTRA_SINK_M["orange"]` |
| 合爪时怼桌面 | 调小 `GRASP_DEPTH_FRAC` / `GRASP_EXTRA_SINK_M`；`MIN_GRASP_Z_M` 是 TCP 的兜底下限 |
| IK 够不着 | 规划器会自动从竖直顶抓退到斜抓（`GRASP_TILT_DEG`），再退就是缩短接近段；日志里会写用的是哪种 |
| 转场蹭到别的水果 | 调高 `TRANSIT_Z_M`；太高会 IK 无解退回直连，看日志 |
| 放进 A 框时水果撞框沿 | 框高写在 `BIN_HEIGHT_M`（现在 12cm）。转场 TCP = 框高 + `FRUIT_BELOW_TCP_M` + `PLACE_CLEAR_M`。还蹭就加大 `PLACE_CLEAR_M` 或 `TRANSIT_Z_M` |
| 规划时报「夹不住」 | 件比夹爪最大开口（`SIM_JAW_OPEN_MAX` = 5.5cm）还宽，换小一点的水果，或确认 `gripper_calib.json` 标定对不对 |
| 和工业 `real/` 混用权重 | 水果用本目录 `yolo26s-seg.pt`，不要用 `tiaozhanbei/real/real_multi.pt` |
