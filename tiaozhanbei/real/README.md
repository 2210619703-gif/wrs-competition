# 执行模块（真机）

驱动 fafu 机械臂完成决策模块下发的抓放任务。**所有和真机有关的文件都在本目录**，
整个目录可以直接拷到机械臂那台工控机上用。

底层是 `fafu_arm_sdk-main/fafu_robot_python` 的 `FafuRobotController`，
本目录只做适配和安全约束，不修改 SDK。

## 为什么能直接复用仿真规划

仿真里的 `PantheraHTSglArm` 和 fafu 真机是同一条 Panthera-HT 六轴：
六个关节的原点偏移和轴向在 URDF 里逐项一致，关节顺序、单位（弧度）、零位也一致。
所以仿真规划出来的 `jv_list` 可以直接对应真机 J1~J6，不需要重做 IK。

但有四处必须适配，`traj_adapter.py` 负责：

| 差异 | 仿真 | 真机 | 处理 |
| --- | --- | --- | --- |
| 关节范围 | 比真机宽（尤其腕部 J5/J6） | `robot.cfg` 软限位更紧 | 逐点裁剪；越界超过 8° 直接判失败 |
| 腕部滚转写法 | 不知道夹爪 180° 对称，常给出 152° 这种真机够不到的角度 | J6 只有 [−95°, 90.5°] | 按单调段折算到最短等价行程（见下） |
| 轨迹密度 | 逐帧动画，且 RRT 相邻帧可能差 25° 以上 | 需要均匀小步 | 先按角度变化抽帧，再把大跨度插值拆小 |
| 夹爪 | 平移开口，单位米 | M7 旋转关节，0~105° | 只在合爪/松爪两个时刻动作，角度由标定表换算 |

**腕部最短行程折算**是能不能落地、以及 J6 会不会空转的关键。
二指夹爪绕腕转 180° 后两根手指互换，夹持效果完全一样，所以 152° 和 −28°
是同一个位形，但只有后者真机够得到、也只有后者不用空转半圈。
适配器按单调段把净转角换成等价最短值（`WRIST_SHORTEST_PATH`，默认开），
抓取平台仍是平台、端点仍 180° 等价，只把多余的整/半圈摘掉。
实测同一条抓螺丝刀的轨迹，J6 行程从 303°（老解缠还会折返到 604°）降到 57°。
关掉这个开关会退回逐点解缠：J6 软限位跨度只有 185.5°，两个分支重叠区 5.5°，
逐点挑会在窄边界上来回翻 180°，表现就是腕部一直扭。

夹爪那条差异需要人工标定，务必先做。

> 注意：`--joint-ranges` 只收紧 WRS 的 `motion_range`，而 IK 走的是 TracIK、
> 按 URDF 限位解算，并不会因此拒绝超范围解。所以**真正的限位防线是本模块的适配层**，
> 不要指望规划阶段已经帮你挡住了。

## 目录

| 文件 | 作用 |
| --- | --- |
| `real_config.py` | 唯一配置入口：定位 SDK、解析 `robot.cfg` 软限位、夹爪映射、下发安全上限 |
| `export_sim_traj.py` | 调 `tiaozhanbei/sim` 的两个规划器（标准抓放 / EC 入格），导出关节轨迹 JSON |
| `traj_adapter.py` | 仿真轨迹 → 真机动作脚本（限位裁剪 / 抽帧 / 夹爪事件） |
| `real_arm.py` | `FafuRobotController` 封装 + `execute_plan` 执行器 |
| `run_task_real.py` | 命令行主入口，串起上面全部步骤 |
| `real_demo_ui.py` | 演示界面：D405 实时画面 + 语音指令 → 智能体 → 真机执行 |
| `calibrate_gripper.py` | 标定夹爪角度 ↔ 实际开口，生成 `gripper_calib.json` |
| `check_offline.py` | 不接硬件的自检（`--vision` 连视觉侧一起查） |
| `outputs/` | 运行产物：`traj.json`（仿真轨迹）、`plan.json`（真机动作脚本） |

视觉侧（固定俯视 D405）：

| 文件 | 作用 |
| --- | --- |
| `vision_config.py` | 相机参数、工作区范围、手眼矩阵读写、YOLO 类名 → 仿真类名/STL 映射 |
| `d405_camera.py` | D405 采集封装，深度对齐到彩色帧；可存帧离线复现 |
| `hand_eye_calib.py` | 固定俯视手动标定，产出 `handeye_d405.json`；彩色点云 + `--connect` 同步真机位形 |
| `manual_calib_piper_outhand_camera.py` | Piper 臂的原版手动标定脚本，仅作参考，不接本目录硬件 |
| `vision_to_annotations.py` | YOLO mask + 点云 + 手眼 → 仿真格式 `annotations.json` |
| `run_vision_real.py` | 视觉全链路入口：识别 → 规划 → 适配 → dry-run/执行 |
| `build_class_map.py` | 生成 `yolo_class_map.json`（换权重或改 STL 库时才需要重跑） |
| `real_multi.pt` | 真实多零件分割权重，113 类（默认） |
| `best(3).pt` | 上一版真实零件分割权重 |

## 上手顺序

**1. 离线自检**（不接硬件，任何机器都能跑）

```bash
python tiaozhanbei/real/check_offline.py
```

确认能找到 SDK、`robot.cfg` 解析正常、`fafu_motor` 扩展和当前解释器版本匹配。
SDK 不在默认位置时用环境变量指定：`set FAFU_SDK_DIR=D:\path\to\fafu_robot_python`。

**2. dry-run**（跑真实规划，但不碰机械臂）

```bash
python tiaozhanbei/real/run_task_real.py --task tiaozhanbei/tasks/delivery_agent_auto.json
```

会打出完整动作清单：每段路径多少个航点、终点关节角、夹爪什么时候合什么时候松，
以及所有限位裁剪告警；然后用一个「虚拟臂」把整个执行流程走一遍，
把每一条 `move_j` 和夹爪指令都打印出来。产物落在 `outputs/`。

多零件任务同样支持，一次 dry-run 会依次列出每个目标：

```bash
python tiaozhanbei/real/run_task_real.py --task tiaozhanbei/tasks/multi_pick_place/delivery_agent_auto_batch.json
```

螺丝入格（EC）走同一个入口，只是换个场景。两条链路导出的轨迹格式相同，
真机侧不用区分，`plan` 里的「规划来源」字段会标明是哪个仿真规划的：

```bash
python tiaozhanbei/real/run_task_real.py --scene ec
python tiaozhanbei/real/run_task_real.py --scene ec --task tiaozhanbei/sim/ec/tasks/test03.json
```

EC 规划比标准抓放慢很多（12 个 job 约 12 分钟），跑完一次后建议用
`--traj` 复用导出的轨迹，不必每次重新规划。

**3. 标定夹爪**（接硬件，只动夹爪）

```bash
python tiaozhanbei/real/calibrate_gripper.py --execute
```

夹爪会依次停在若干角度，用卡尺量开口按毫米输入，生成 `gripper_calib.json`。
之后主入口会自动用这张表，不用再传参数。

**4. 真机执行**（接硬件，手放在急停上）

```bash
python tiaozhanbei/real/run_task_real.py --task ... --execute --speed 8
```

复用上一步已经导出的轨迹、跳过重新规划：

```bash
python tiaozhanbei/real/run_task_real.py --traj tiaozhanbei/real/outputs/traj.json --execute
```

## 视觉到真机（固定俯视 D405）

前面的流程里零件位置来自数据集标注。接了相机之后，这一步换成现场识别：

```
D405 拍一帧（深度对齐到彩色帧）
  → YOLO-seg 出 mask + 类别
  → mask 覆盖的像素取相机系 3D 点
  → 手眼矩阵变到世界系，按工作区盒子滤掉桌沿/地面/机械臂本体
  → 中位数当中心、XY 主轴当 yaw、高度跨度判姿态
  → annotations.json（和 dataset_learn 同格式）
  → 后面完全走原来那条链：仿真规划 → 限位适配 → 下发
```

关键点：输出格式故意做成和 `tiaozhanbei/dataset_learn/*/annotations.json` 一样，
所以**仿真侧一行代码都不用改**，把输出目录当 `--dataset-root` 传进去就行。

### 为什么固定俯视比眼在手上简单

`competition/manual_calib_er4ia.py` 那套是眼在手上（相机装在 TCP 上），
每帧都要读机械臂位姿再算 `w2c = w2r · r2c`。相机固定在桌子上方不动的话，
标定产物**就是**那个固定的 `w2c`（世界 ← 相机），每帧直接用，
不需要连机械臂、也不会被关节角误差污染。

代价是相机一旦被碰动、或者挪了位置，必须重新标定。

### 1. 标定手眼

```bash
python tiaozhanbei/real/hand_eye_calib.py
```

会打开一个仿真场景：桌面（上表面 z=0）、机械臂（摆在仿真 HOME 位形）、
以及按当前矩阵变换过的实时点云。目标是让点云里的桌面和机械臂跟模型重合。
点云用相机的真实颜色画，桌沿和零件比单色好认。

按键沿用 `manual_calib_er4ia.py` 的布局：`wasd`/`qe` 平移，`zxcvbn` 绕世界轴旋转，
`1234` 调步长，`p` 保存，`r` 复位。另外加了一个 **`f` 自动贴桌面**：
拟合点云主平面，摆平并压到 z=0。高度和倾斜最难用眼睛调准，先按 `f`，
再用 `wasd` 平移对齐机械臂底座、`b`/`n` 转偏航，最后 `p` 保存。

每次微调都会自动覆盖写 `handeye_d405.json`，中途关窗口不会丢进度；
只想按 `p` 才存就加 `--no-autosave`。注意 `r` 是回到**进入时**的矩阵。

```bash
python tiaozhanbei/real/hand_eye_calib.py --connect
```

加 `--connect` 会连真机并持续同步关节角：手推机械臂时仿真臂跟着动，
可以直接拿臂本体当对齐参照，比只看桌面准。这套交互抄自
`manual_calib_piper_outhand_camera.py`（那份是 Piper 臂的原版，留作参考）。

> `hand_eye_calib.py` 是本目录里唯一需要 WRS / Panda3D 的脚本（要画机械臂当参照）。
> 真机执行链路本身不依赖它。

### 2. 存一帧，先离线把识别调通

相机就一个，边调边占着不方便。存下来的帧可以反复用：

```bash
python tiaozhanbei/real/d405_camera.py --save-dir tiaozhanbei/real/outputs/frame01
python tiaozhanbei/real/vision_to_annotations.py --frame-dir tiaozhanbei/real/outputs/frame01
```

会打印每个零件的世界坐标、yaw、姿态、点数，并生成 `detect_preview.jpg`
（画着 mask 轮廓和坐标）。**先看这张图**：框对不对、坐标像不像，再往下走。

### 3. 视觉全链路

平时**不用敲开关**，默认行为全部写在 `real_config.py` 的「运行模式」一节：

```bash
python tiaozhanbei/real/run_vision_real.py
```

要改行为就改配置里那几个常量，不用记命令行：

| 配置项 | 作用 | 默认 |
| --- | --- | --- |
| `RUN_MODE` | `vision` 只识别 / `interactive` 循环下指令 / `once` 跑完就退 / `zero` 送到电机零位 | `interactive` |
| `EXECUTE_ON_REAL` | `True` 真下发，`False` 只 dry-run 打印 | `False` |
| `CONTINUOUS_PATH` | `True` 整段路径连续下发，`False` 逐点阻塞（慢但每步确认到位） | `True` |
| `PICK_ONLY` | `True` 只抓起来悬停不放，仿真不建收纳盒 | `True` |
| `PARK_BEFORE_VISION` | 拍照前退到避让位，避免臂被 YOLO 当成零件 | `True` |
| `START_FROM_PARK` | 避让位到了就从那里接抓取，不再绕回仿真 HOME | `True` |
| `HOME_ON_EXIT` | 退出时把臂送回 `EXIT_HOME_CONF` 静置位（异常/急停退出会跳过） | `True` |
| `USE_SERVO_J` | 用 `servo_j` 在线流式喂帧（动作最连贯） | `True` |
| `JAW_OPEN_ON_ARRIVE` | 到 HOME / 避让位后张开夹爪 | `True` |
| `WRIST_SHORTEST_PATH` | J6 按夹爪 180° 对称折到最短行程，避免空转 | `True` |
| `MOVE_J_TOLERANCE_RAD` | `move_j` 到位判定带，太小会一直等到超时 | `1.5°` |
| `PICK_TARGETS` | `once` 模式抓什么；空列表 = 抓全部 | `[]` |
| `DEFAULT_MOVE_SPEED` | 下发速度 `(0,100]` | `10` |
| `RELEASE_MODE_ON_CLOSE` | 断开时的刹车档：`stop` / `brake` / `hold` | `brake` |
| `DEST_XYZ` / `DEST_NAME` | 放置点坐标和名字（`PICK_ONLY=True` 时不生效） | `(0.30,-0.20,0)` |

命令行只用来临时覆盖某一次，每个开关都有 `--xxx` / `--no-xxx`：

```bash
python tiaozhanbei/real/run_vision_real.py --mode vision       # 这次只识别
python tiaozhanbei/real/run_vision_real.py --no-execute        # 这次只 dry-run
python tiaozhanbei/real/run_vision_real.py --no-continuous     # 这次逐点确认
python tiaozhanbei/real/run_vision_real.py --mode once --pick 螺丝刀 --speed 8
```

启动时会打印一行当前生效的模式和开关，照着核对就行。
`--reuse-vision` 复用上次的识别结果，不重新拍照。

### 交互模式：识别一次，反复下指令

```powershell
# RUN_MODE = "interactive" 时直接跑，不用带参数
python tiaozhanbei/real/run_vision_real.py
```

`RUN_MODE = "interactive"`（或命令行 `--mode interactive`）把「拍照识别」和
「下指令」拆开：脚本只在启动时拍一次照，之后停在 `[real] 抓什么>` 提示符上
循环收指令。配置里 `EXECUTE_ON_REAL = True` 就是敲完回车直接下发。
提示符支持：

| 输入 | 含义 |
| --- | --- |
| `螺丝刀` | 抓这个（中文片段 / YOLO 英文类名 / `id:3` 都行） |
| `螺丝刀 扳手` | 一条指令依次抓多个 |
| `all` | 抓识别到的全部 |
| `look` | 退避让位重新拍照识别（桌面动过必须用这个） |
| `list` | 列出当前识别到的零件 |
| `q` | 退出 |

注意每条指令都是按**启动那次拍照**的位置开环规划的。桌面动过就先 `look`，
否则会抓空或撞件。`EXECUTE_ON_REAL = False` 时每条指令走的是 dry-run，只打印 plan。

整个交互会话共用同一条真机连接（`ArmSession`），每条指令之间不断开——
断开会让电机松掉，臂当场沉下去。只在 `q` 退出时才断一次。

### 动作连贯 vs 逐点确认

默认每个航点单独发一次阻塞 `move_j`，到位才发下一个，所以动作一卡一卡；
遇到 `not settled within 20.0s` 还会干等满超时。好处是每步都能确认到位，
第一次上电看得清，出问题也容易定位。用 `CONTINUOUS_PATH = False` 切回这个。

`CONTINUOUS_PATH = True`（默认）改成把整段路径交给 SDK 的 `move_jntspace_path`：
它先按 50ms 插值成密集帧，再逐帧刷位置通道（`0x8090`），动作是连续的。
走完之后会补一次阻塞 `move_j` 确认到终点——不然合爪可能发生在半路上。

旧的 `fafu_motor.pyd` 缺 `set_many_pos_vel_acc`，但流式走的是
`set_many_pos_vel_tqe`，不受影响；真缺了会自动退回逐点模式并打印提示。

`PICK_ONLY = True` 把任务的 `destination` 置空，仿真据此完全不建收纳盒
（日志里不会再出现 `[scene] 收纳盒 …`），盒子也不参与碰撞检测。
轨迹在抓起并抬到悬停高度后结束，夹爪保持闭合。等真实盒位量出来了，
把 `DEST_XYZ` 填对再把 `PICK_ONLY` 改成 `False`。

### 演示界面（三个模块串在一起）

命令行那套要分三步跑。想一口气演示「说一句话，机械臂就动」用这个：

```bash
python tiaozhanbei/real/real_demo_ui.py
```

布局和仿真那套 `decision/agent_demo_ui.py` 一样：左边录音按钮 + 音量条 + 终端输出，
右边 D405 实时画面，底部是进度条。区别是右边不贴 Panda3D 窗口，直接播相机。

启动就开始直播，**不做识别**。你说完话点「结束并识别」，那一刻才抓一帧
（连拍取深度中位数）跑 YOLO，然后走：智能体出任务 JSON → 仿真规划器导出轨迹
→ 限位适配 → 执行。整个过程画面继续直播，能看着机械臂动。

**抓帧前会先把机械臂退到避让位。** 相机是固定俯视，机械臂留在画面里会被 YOLO
当成零件（实测夹爪被认成 `pliers`、置信度 0.9），仿真 HOME 也在画面内所以顶不了
这个用。避让位是 `real_config.PARK_CONF`，现场换了相机位置就要重新量一次：
把臂收到画面外，用 `smoke_move.py` 读回关节角填进去。

位形的走法是：开机在 HOME → 点「移到避让位」或下第一条指令时走到避让位 →
每轮识别前都回避让位 → 执行完停在避让位（不回 HOME）→ 关窗时才送回 HOME。
执行中途关窗只急停，不会自动走位。
`PARK_BEFORE_VISION = False`（或 `--no-park`）可以整个关掉避让位逻辑。

识别前回避让位会**张开夹爪**（`JAW_OPEN_ON_ARRIVE`），方便 YOLO。
抓完再回避让位 / 退出回静置位**不会张爪**，否则刚夹住的件会掉下去。合着的夹爪在相机里
会被 YOLO 认成 pliers，而且下一次抓取本来也要先张开。张爪发生在**到位之后**，
不是路上——否则手里还夹着东西时会半路松手。

`q` 退出（含 EOF / Ctrl-C）时臂会先回**静置位** `EXIT_HOME_CONF` 再断开
（`HOME_ON_EXIT`）：避让位是悬在半空的，收回来更站得住。
异常和急停退出会跳过这一步——那时候再让臂走位不安全。

静置位和 `SIM_HOME_CONF` 是两个不同的东西，别混：

| 常量 | 角色 | 改它的影响 |
|---|---|---|
| `SIM_HOME_CONF` | 仿真轨迹的 HOME 锚点。`hand_eye_calib.py` 仍拿它当仿真位形。视觉抓取默认**不再**先回到这里（`START_FROM_PARK`） | 改了就和仿真脱钩 |
| `EXIT_HOME_CONF` | 退出/断电前的静置位，纯粹为了「站得住」 | 只影响退出动作，随现场调 |

### 断开时的刹车档

`RELEASE_MODE_ON_CLOSE` 三档：`stop` 完全松开（能手推，会沉）、
`brake` 短路阻尼（不耗电，姿态大致保持）、`hold` 电机主动顶住（最稳，耗电发热）。

`hold` 档**没走 SDK 的 `close_connection(joint_release="hold")`**，而是
`RealArm._true_hold_and_close()` 自己实现的。原因是 SDK 那条路径调
`set_motor_mode(mid, 0x0A)`，而 C++ 里它先**无条件** `stop()` 切断 PWM
（后面还跟一个超时 200ms 的 `read_motor_state`），再补一帧
`build_pos_vel_tqe_int16(pos, 0, tqe=0)` —— 字面量 `0` 在协议里是
「最大力矩 = 0」，等于没力气。结果电机停在 `0x0A` 却顶不住重力，
实测比 `brake` 还容易下坠。

自己的实现改走公共 API `set_pos_vel_tqe`，它会把 `tqe_raw=0` 翻译成
`NAN_INT16`（用电机默认最大力矩），关串口前发最后一批保持帧，
夹爪也一起顶住（用抓取那套 `effort`），所以夹着的件不会松手。
下面任一条不满足就自动回退到 SDK 标准流程并打印原因：

- 扩展没有 `set_pos_vel_tqe`；
- 读不到电机状态；
- **有关节停在软限位外** —— `set_pos_vel_tqe` 会裁剪目标位置，
  补帧反而会让它猛地弹回限位内，这正是 SDK 宁可绕过公共 API 的原因。

### 三种下发方式

| | `servo_j`（默认） | `move_jntspace_path` | `move_j` |
| --- | --- | --- | --- |
| 谁出中间点 | **我们自己**按 100Hz 喂 | SDK 先 TOPPRA 插密再逐帧发 | 电机 / Python 循环 |
| 动作 | 最连贯 | 连贯 | 一卡一顿 |
| 看门狗 | 有，断流约 150ms 自动刹车 | 无 | 无 |
| 开关 | `USE_SERVO_J = True` | `CONTINUOUS_PATH = True` | 两个都关 |

`servo_j` 现在同时管点到点（避让位 / HOME）和抓取轨迹。点到点不再按
`MAX_STEP_RAD` 拆成「1/2、2/2」两段 `move_j`——servo 本身就是密集小步，
一次插到底，也就没有每段等到位的那段停顿。

几个实现上必须这么做的点：

- **节奏用绝对时刻推进**，不是每轮 `sleep(dt)`。sleep 有误差，累加起来帧间距会越飘，
  动作反而更抖，严重时超过看门狗时限被刹车。落后了就直接追下一帧，不补睡。
- **Windows 定时粒度要抬到 1ms**（`timeBeginPeriod(1)`）。默认约 15ms 撑不住 100Hz。
- **帧间距必须小于 `SERVO_MAX_STEP_RAD`**。超了 `servo_j` 会把每帧裁掉一截，
  结果是走不到位还一直在追。代码里按 `0.9 × max_step` 留了余量。
- **喂完帧不等于到位**：位置环还在收敛，所以会继续喂终点顶住，直到进入
  `SERVO_SETTLE_TOL_RAD` 或超时告警。
- **会话一定成对开关**。中途抛异常也会 `servo_end`——否则看门狗武装着，臂会被刹住。
  `servo_end("hold")` 把电机留在位置模式 `0x0A`，后面还能直接接 `move_j`。
- 会话期间不做慢活（打印、开夹爪、读文件），那些都在会话外面。

速度换算和 `move_j` 对齐：`speed=100` 对应 `SERVO_VEL_AT_FULL_SPEED`（180°/s），
所以同一个 `speed` 在两种模式下快慢差不多。注意 `MAX_MOVE_SPEED = 40` 会先把
`speed` 压到 40，实际上限是 72°/s。

`--no-servo` 或 `USE_SERVO_J = False` 退回原来的 `move_j` 路径。
SDK 缺 `servo_start` / `servo_j` / `servo_end` 时也会自动退回并打印提示。

### move_j 为什么会等很久

`move_j(block=True)` 要轮询到每个关节都进入到位带才返回。SDK 默认带宽是 `0.1°`，
但位置模式没有积分项，带重力负载时静差常有零点几度到一度，判不到位就一直轮询到
`MOVE_J_TIMEOUT`（20 秒）才返回——日志里那句
`not settled within 20.0s` 就是这么来的，实际上臂早就停住了。

所以 `MOVE_J_TOLERANCE_RAD` 放宽到 `1.5°`：到位就走，不再干等超时。
另一半时间花在速度上：`speed` 线性映射到平均角速度 `(speed/100) × 0.5 turns/s`，
`speed=5` 只有约 `9°/s`，一步 25° 就要将近 3 秒。要快就调
`DEFAULT_MOVE_SPEED`，不过别一次调太猛。

### 断开时的刹车

SDK 的 `close_connection` 默认 `joint_release="stop"`：切断 PWM，关节被重力
带着往下沉。臂停在避让位或半空时，这一下看起来就是「摔下去」，夹着东西还会松手。

`RealArm.close()` 改成按 `real_config.RELEASE_MODE_ON_CLOSE` 走（夹爪一直是 brake）：

| 档位 | 行为 | 什么时候用 |
| --- | --- | --- |
| `stop` | 完全松开，能用手推臂，但会沉 | 要手动摆位形时 |
| `brake` | 短路阻尼，不耗电但阻止运动，姿态大致保持 | 默认，日常用 |
| `hold` | 电机主动顶住，最稳 | 夹着东西不能掉；但持续耗电发热，别长时间挂着 |

配置里改 `RELEASE_MODE_ON_CLOSE` 是全局默认，命令行 `--release-mode {stop,brake,hold}`
临时覆盖一次（`run_vision_real.py` 和 `real_demo_ui.py` 都认）。
写了三档之外的值会在 import 时直接报错。

**但光靠刹车档救不回「挪完就掉」。** 电机不下垂靠的是位置模式（`0x0A`）通电顶住，
而这个状态是「连着」才有的：`close_connection` 自己不发释放指令，可一旦关掉串口、
进程退出，调试板就没有主机再刷位置帧，重力立刻把臂带下去。
把档位设成 `hold` 也没用——那条 `set_motor_mode(0x0A)` 在 `move_j` 之后本来就是
no-op（电机早就在 `0x0A`），掉的是**断开本身**。

所以真正的修法是**别中途断开**：`run_vision_real.py` 用一个 `ArmSession`
把避让 → 识别 → 执行全程串在同一条连接上，只在收尾时断一次。
以前是「连上 → 挪到避让位 → 断开 → 去识别」，断开那一下臂就沉了。

收尾那次断开仍然会松到刹车档能保持的程度。手上有件、或臂悬在半空时先扶一下，
或者把臂先送到一个机械上站得住的位形再退出。

上一次识别的轮廓会叠在实时画面上（可关）。这只是给人看的：真正下发用的坐标
只来自抓帧那一刻，不是画面上的框。

安全上和命令行一致，**默认 dry-run**：不勾「允许真机执行」绝不下发指令，勾了
还要再确认一次。避让位也算下发，所以 dry-run 下机械臂不会动——这时画面里如果
有臂，识别结果就会多出个「钳子」。界面上有急停按钮，关窗时若正在执行会先急停。

没有相机也能把界面和链路走通（识别那步会因为缺手眼标定停下，其余照常）：

```bash
python tiaozhanbei/real/d405_camera.py --save-dir tiaozhanbei/real/outputs/frame01
python tiaozhanbei/real/real_demo_ui.py --frame-dir tiaozhanbei/real/outputs/frame01
```

Dify 地址和 Key 在文件顶部，和决策界面同一套；连不上会自动回退本地规则，
日志里的 `[JSON_SOURCE]` 会写明这次 JSON 是远程还是本地出的。

### 视觉侧的已知限制

这几条会直接影响抓取成败，先知道比事后查好：

1. **开环，执行阶段不看相机**。轨迹是按拍照那一刻的零件位置规划的，
   桌面动过就得重新跑一次识别，否则会抓空或撞件。
2. **STL 只是代表形状**。类名映射表给每类挑了叶目录下的第一个 STL 当碰撞网格，
   仿真按它搜抓取位姿。同类零件尺寸差得多（比如 M4 和 M12 螺栓）时抓取宽度会偏。
   权重里还有 `nut`、`screw` 这种泛类，形状只能算近似，识别到会单独提示。
3. **判不出 `inverted`**。单目俯视看不出螺丝是正插还是倒插，只能靠高度跨度区分
   立着（normal）和横躺（fallen）。要区分倒置得加侧视相机或状态分类头，
   现在的做法是宁可报 normal 也不瞎猜。
4. **圆形件的 yaw 没有意义**。螺母、垫圈、轴承的 XY 主轴基本是噪声，
   但这类零件本身没有明确朝向，不影响抓取。
5. **工作区盒子要按现场改**。`vision_config.WORKSPACE_BOUNDS_M` 决定哪些点算有效。
   调窄了零件会被整个滤掉（日志里报「工作区内有效点只有 N」），
   调宽了会把桌沿和机械臂本体当成零件的一部分。

### 视觉侧自检

```bash
python tiaozhanbei/real/check_offline.py --vision
```

不接相机也能跑完：查权重、类名映射表（113 类是否都能解析到真实存在的 STL）、
手眼文件、`pyrealsense2`，并用合成深度图验证「深度 → 相机系 → 世界系 → 位姿」
这条坐标链——已知位置的台阶和长条零件，还原出来的 xy 误差、yaw、高度都要对得上。
这段算错了机械臂会伸到错的地方，而光看日志是看不出来的（数字都很「合理」）。

## 安全设计

- **默认 dry-run**。不加 `--execute` 绝不下发任何指令，连串口都不打开。
- **默认低速**。`speed=10`（约 18°/s），命令行传再大也会被 `MAX_MOVE_SPEED` 压住。
- **两层限位**。适配阶段按 `robot.cfg` 裁剪，下发阶段 `move_j` 再裁一次。
- **不留瞬移**。相邻航点单关节超过 25° 会被插值拆成小步；HOME 到首个航点、
  目标与目标之间的空档、以及收尾回 HOME，都会自动拆成小步走。
  其中「目标之间的过渡」是直线关节插补、仿真没做碰撞检查，会单独报警，
  EC 多件入格时一定要先空载跑一遍确认。
- **逐点阻塞**。`follow_path` 一个航点一个航点走，每步确认到位，不做流式 servo。
- **异常即急停**。`with RealArm(...)` 退出时若带着异常，先 `emergency_stop` 再断开。
- **单目标失败不中断**。多零件任务里某个目标失败会跳过继续下一个，和仿真行为一致；
  要一失败就停加 `--stop-on-fail`。

调保守或调快都改 `real_config.py` 顶部那组常量，不要散在各处改。

## 首次上真机的注意事项

1. **确认零位**。仿真和 SDK 都假设关节零位一致。上电后先 `check_offline.py`，
   再手动确认机械臂物理零位和 URDF 一致，否则整条轨迹会整体偏移。
2. **仿真 HOME 不是 SDK HOME**。SDK 的 `go_home()` 是全零位形，仿真轨迹从
   `[0, 0.30, 0.30, 0, 0.20, 0]` rad 起步。本模块用的是后者（`RealArm.go_sim_home`）。
3. **先空场地跑一遍**。桌面上不放零件，只看机械臂轨迹形状对不对。
4. **注意越界告警**。仿真的关节范围比真机宽，被裁剪的位形在真机上到不了规划的位置，
   抓取可能偏。越界超过 8° 会直接判失败拒绝下发；告警多的话回仿真换个抓取方案，
   别在真机上硬试。
5. **夹爪对称假设**。腕部等价角度改写建立在「二指夹爪转 180° 等效」之上。
   如果换了非对称末端（比如带偏置的吸盘或三指爪），要把
   `traj_adapter._wrap_options` 里的半圈选项去掉。

## 与其他模块的关系

```
decision/  自然语言 → 任务 JSON          real/  D405 + YOLO → annotations.json
              ↓                                        ↓
              └────────────────┬───────────────────────┘
                               ↓
sim/       规划 + 仿真动画（--export-traj 导出关节轨迹）
           ├─ run_agent_pick_place_side_sim.py   标准抓放
           └─ ec/run_ec_bin_sim.py               螺丝入格
                               ↓
real/      限位适配 → fafu 真机执行      ← 本目录
```

零件位置有两个来源：数据集标注（`--dataset-root tiaozhanbei/dataset_learn`）
或现场识别（`run_vision_real.py`）。两者产出同一种 `annotations.json`，
所以仿真侧不区分。

本目录只往外调这两个仿真脚本的 `--export-traj`，不反向依赖决策模块。
两个脚本都支持 `--joint-ranges`（按真机软限位规划），导出格式一致。

和 `tiaozhanbei/yolo/` 的区别：那边是感知模块（仿真图用 `yolo/仿真`，真实 RGB-D 用 `yolo/实机`）。
本目录是真机机械臂执行，输入是任务 JSON 和与仿真同结构的 `annotations.json`。
