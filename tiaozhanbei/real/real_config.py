# -*- coding: utf-8 -*-
"""真机执行模块的唯一配置入口。

这里只做四件事：

1. 找到 fafu SDK（``fafu_arm_sdk-main/fafu_robot_python``）并挂到 ``sys.path``；
2. 解析 ``robot.cfg``，把软限位统一换算成弧度；
3. 提供仿真夹爪开口（米）↔ 真机夹爪角度（弧度）的映射；
4. 集中放下发到硬件的安全上限。

本文件不 import SDK，也不碰串口，可以在没有硬件、没有 ``fafu_motor.pyd``
的机器上直接跑，方便离线自检。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

REAL_DIR = os.path.dirname(os.path.abspath(__file__))
TB_DIR = os.path.abspath(os.path.join(REAL_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(TB_DIR, os.pardir))

# 仿真 / TracIK 用的本体 URDF。运动学原点与 WRS PantheraHT 一致，
# 限位按厂家文件；末端 tool_link 只作法兰外 165mm 标记，IK 仍解到 link6，
# 夹爪 TCP 由 PantheraGripper 叠加，避免和 tool_link 重复计算。
FAFU_BASE_URDF = os.path.join(REPO_ROOT, "fafu_arm_sdk-main", "fafu_baseV1.urdf")

# 仿真侧常量，必须与 tiaozhanbei/sim/run_agent_pick_place_side_sim.py 保持一致。
# 这里复制而不是 import，是为了让真机模块不依赖 WRS / Panda3D。
SIM_HOME_CONF = (0.0, 0.30, 0.30, 0.0, 0.20, 0.0)   # rad，仿真轨迹的起始位形
SIM_JAW_OPEN_MAX = 0.055                            # m，仿真夹爪最大开口
SIM_JAW_CLOSE_MIN = 0.0005                          # m

# 识别用避让位。D405 是固定俯视，机械臂停在画面里会被 YOLO 当成零件
# （实测夹爪被认成 pliers、置信度 0.9），所以抓帧前必须先退到这里。
# 仿真 HOME 也在画面内，顶不了这个用。数值是现场把臂收到底座一侧、
# 确认完全出画后用 smoke_move.py 读回来的关节角。
PARK_CONF_DEG = (0.86, 0.0, 49.72, -46.8, 0.54, -0.43)
PARK_CONF = tuple(math.radians(v) for v in PARK_CONF_DEG)   # rad
PARK_SETTLE_S = 0.8                      # 退到避让位后等相机刷掉带机械臂的旧帧

# 放置松爪位（关节角，现场示教），当作料框区用，免去量盒子尺寸。
# 只有智能体输出「放置」或 run_vision_real 接下一项任务要松爪时才来这里。
# 纯抓取只回避让位，不走这个位。
# 数值是 2026-09-04 现场读回来的：[ -53.96, 131.15, 109.22, -59.00, 6.52, -10.55]°
PLACE_CONF_DEG = (-53.96, 131.15, 109.22, -59.00, 6.52, -10.55)
PLACE_CONF = tuple(math.radians(v) for v in PLACE_CONF_DEG)  # rad

# 退出前的静置位（HOME_ON_EXIT 用这个，不是 SIM_HOME_CONF）。
# 两者刻意分开：SIM_HOME_CONF 是**规划轨迹的起点锚点** —— execute_plan 的
# go_home_first 先回到它，仿真出的轨迹第一个航点才接得上，hand_eye_calib.py
# 也拿它当仿真位形。改那个会和仿真脱钩，所以静置位单独放这里。
# 数值是现场收臂后用 smoke_move.py 读回来的关节角。
EXIT_HOME_CONF_DEG = (0.79, 0.07, 0.22, -5.69, 5.58, -0.40)
EXIT_HOME_CONF = tuple(math.radians(v) for v in EXIT_HOME_CONF_DEG)   # rad

# 电机零位（全 0 rad）。和仿真 HOME、退出静置位都不是一回事：
# SDK 的 go_home()、现场「臂已经在零位」指的都是这个。
# J2 软限位下沿正好是 0°，贴着限位但合法。
ZERO_CONF = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# hand_eye_calib.py 仿真臂的默认关节角（度）。改这 6 个数就会改标定画面里的臂。
# 命令行 ``--conf-deg`` 可临时覆盖；标定窗口里也能用 [ ] - = 逐轴调。
# 下发 go_zero() 仍走 ZERO_CONF（全 0），不要把这两套混用。
CALIB_REST_CONF_DEG = (0.04, 0.00, -0.47, -1.44, 0.00, -0.11)
CALIB_REST_CONF = tuple(math.radians(v) for v in CALIB_REST_CONF_DEG)

# ---------------------------------------------------------------------------
# 运行模式（改这里就不用在命令行敲一长串开关）
# ---------------------------------------------------------------------------
# run_vision_real.py 的默认行为全部取自本节。命令行仍然可以临时覆盖，
# 每个开关都有 --xxx / --no-xxx 两种写法，命令行优先级最高。
#
# RUN_MODE 四选一：
#   "vision"      只识别：退避让位 → 拍照 → 出 annotations + 预览图，就停。
#                 等价于以前的 --no-plan。想先看框准不准时用这个。
#   "interactive" 识别一次，然后停在提示符循环收指令（敲零件名就抓）。
#                 桌面没动就一直用同一帧规划，动过了敲 look 重拍。
#   "once"        一次性跑完：识别 → 规划 → 按 PICK_TARGETS 抓完就退出。
#   "zero"        只把真机送到电机零位（全 0°）然后断开，不识别不规划。
#                 上电后姿态乱了、或要对齐标定模型时用。
RUN_MODE = "interactive"
RUN_MODES = ("vision", "interactive", "once", "zero")

# 真的给机械臂下发指令。False = dry-run，只打印会发什么，不碰硬件。
# 第一次换场景 / 换标定，建议先 False 跑一轮看 plan。
EXECUTE_ON_REAL = True

# 整段路径连续下发（动作连贯）。False = 逐点阻塞，慢且一卡一卡，
# 但每个航点都确认到位，调试时更容易定位问题。
CONTINUOUS_PATH = True

# 只抓起来悬停，不放进收纳盒。现场盒子位置还没量出来时用 True。
# 改成 False 的话记得同时把 DEST_XYZ 填对，否则会往错的地方放。
PICK_ONLY = True

# 拍照前把臂退到避让位。臂停在画面里会被 YOLO 当成零件（实测夹爪被认成
# pliers、置信度 0.9），除非臂本来就不在相机视野里，别关。
PARK_BEFORE_VISION = True

# 视觉链路已经停在避让位后，抓取从当前位置接上，不再绕回 SIM_HOME_CONF。
# 仿真轨迹仍从 HOME 起、到 HOME 止；执行时会：
#   1) 跳过 approach 开头只为离开 HOME 的航点，从当前（避让位）接到剩余路径；
#   2) carry 若以 HOME 为终点且本轮结束后要回避让位，抬起后就转去避让位。
# False = 老行为（先回 HOME 再走规划）。--home-first 也能单次打开。
START_FROM_PARK = True
# 从抓取位算起，单关节至少走出这么多才算「抬起来了」，然后才切去避让位。
# 太小会擦着桌面就走；太大又会把回 HOME 的后半段带上。
CARRY_LIFT_MIN_RAD = math.radians(18.0)

# 退出时把臂送回 EXIT_HOME_CONF 静置位。避让位是悬在半空的，断电前收回来更站得住。
# 异常/急停退出不会走这一步（那时候动臂不安全）。
HOME_ON_EXIT = True

# 到达 HOME / 避让位后是否张开夹爪。只影响调用方没显式传 open_jaw 的场合。
# 识别前回避让位会张开（合着的夹爪会被 YOLO 认成 pliers）。
# 抓完再回避让位必须传 open_jaw=False，否则刚夹住的件会松手掉下去。
# 敲 q 退出回静置位则张开（任务结束，件放在静置位再松）。
JAW_OPEN_ON_ARRIVE = True
# 只给 run_vision_real.py 用：上一项还夹着件时，终端输入了下一项任务，
# 先到示教放置位（料框）停这么久再松爪，然后才抓下一个。
# 智能体演示界面不走这段；抓取指令也不会自动去放置位。
# 0 = 到位立刻松。
PARK_RELEASE_DELAY_S = 3.0

# RUN_MODE = "once" 时抓什么。空列表 = 抓识别到的全部。
# 支持中文片段 / YOLO 英文类名 / "id:3"，例：["螺丝刀", "扳手"]
PICK_TARGETS: list[str] = []

# 放置点世界坐标（米）和名字。PICK_ONLY = True 时这两个不起作用。
DEST_XYZ = (0.30, -0.20, 0.0)
DEST_NAME = "收纳盒"

# 断开连接时怎么处理关节（"刹车档"）。SDK 自己的默认是 "stop"，会直接切 PWM，
# 关节被重力带着往下沉：臂停在避让位或半空时就是「摔一下」，夹着东西还会松手。
# 所以这里默认 "brake"。三档的取舍：
#   "stop"   完全松开。能用手推臂，但会沉。要手动摆位形时才用。
#   "brake"  短路阻尼。不耗电，但阻止运动，姿态能大致保持住。默认。
#   "hold"   电机主动顶住。最稳，但持续耗电发热，别长时间挂着。
RELEASE_MODE_ON_CLOSE = "hold"
RELEASE_MODES = ("stop", "brake", "hold")

# "hold" 档在关串口前自己补的那一帧的力矩上限（raw int16）。
#
# 为什么要自己补：SDK 的 close_connection(joint_release="hold") 走
# set_motor_mode(mid, 0x0A)，而 C++ 里这条路径先无条件 stop() 切断 PWM，
# 再补一帧 build_pos_vel_tqe_int16(pos, 0, tqe=0)。字面量 0 在协议里是
# "最大力矩 = 0"，也就是没力气 —— 电机停在 0x0A 却顶不住重力，比 brake
# 还差（brake 至少有三相短路阻尼）。所以 RealArm.close() 绕开它，
# 直接走公共 API set_pos_vel_tqe 补最后一帧。
#
# 0 = 用电机默认最大力矩（公共 API 会把 0 翻译成 NAN_INT16=0x8000，
#     语义是"不额外限制"，和 robot.cfg 的 max_torque_raw=0 一致）。
# 非 0 = 显式上限。别为了省电填很小的值，顶不住重力就白做了。
HOLD_TORQUE_RAW = 0

if RUN_MODE not in RUN_MODES:
    # 配置写错了要在 import 时就炸，不然会带着意料之外的行为去动真机
    raise ValueError(f"RUN_MODE 只能是 {RUN_MODES} 之一，现在是 {RUN_MODE!r}")
if RELEASE_MODE_ON_CLOSE not in RELEASE_MODES:
    raise ValueError(
        f"RELEASE_MODE_ON_CLOSE 只能是 {RELEASE_MODES} 之一，"
        f"现在是 {RELEASE_MODE_ON_CLOSE!r}"
    )

# ---------------------------------------------------------------------------
# 下发安全上限（改这里就能整体调保守 / 调快）
# ---------------------------------------------------------------------------
DEFAULT_MOVE_SPEED = 5          # move_j 的 speed，(0,100]；10 ≈ 18°/s，首次上电建议别调高
MAX_MOVE_SPEED = 40              # 允许的最大 speed，命令行传再大也会被压到这个值
MAX_STEP_RAD = math.radians(25)  # 相邻航点单关节最大跨度，超了就插值拆成小步
WAYPOINT_MIN_DELTA_RAD = math.radians(1.0)  # 抽帧阈值：所有关节变化都小于它就丢掉这帧
# 仿真放宽过关节范围，落到真机要裁剪。裁掉一点点无所谓，裁多了说明真机根本到不了
# 规划的位姿（抓取会偏、避障也没验证过），这时候必须拦下来而不是硬发。
CLAMP_WARN_RAD = math.radians(1.0)
CLAMP_FAIL_RAD = math.radians(8.0)
MOVE_J_TIMEOUT = 20.0            # 单个航点的阻塞超时（秒）
# move_j(block=True) 的到位判定带。SDK 默认 0.1°，对带重力负载的位置环太苛刻：
# 位置模式没有积分项，静差通常有零点几度到一度，判不到位就一直轮询到
# MOVE_J_TIMEOUT 才返回——这就是「一步要等 20 秒」的原因。放宽到实际能到的量级，
# 到位就走，不再干等超时。真要严格到位再调小。
MOVE_J_TOLERANCE_RAD = math.radians(1.5)
# 连续下发（stream）：把整段路径交给 SDK 的 move_jntspace_path，它会先插值成
# 密集帧再逐帧刷位置通道，动作是连贯的。逐点阻塞 move_j 则是「到位才发下一点」，
# 稳但一卡一卡，而且每点最多等 MOVE_J_TIMEOUT 秒。
STREAM_CONTROL_FREQ = 0.05       # 流式下发的帧周期（秒）
STREAM_SETTLE_TIMEOUT = 8.0      # 流式走完后，补一次阻塞 move_j 确认到位的超时（秒）

# ---------------------------------------------------------------------------
# servo_j 在线流式（点到点和轨迹都走它，动作最连贯）
# ---------------------------------------------------------------------------
# move_j 是「给一个目标、自己走完」。旧 fafu_motor.pyd 缺 set_many_pos_vel_acc，
# move_j 退化成 style=linear：目标只发一次，然后循环里反复重发同一帧等到位，
# 中间没有平滑曲线，所以又慢又抖。
#
# servo_j 换成我们自己按固定频率喂中间点：每帧都是一小步，速度由帧间距决定，
# 电机只是跟随，不会被反复重新触发。代价是必须守住节奏——
# 断流超过 SERVO_WATCHDOG_MS 固件就会刹车（这也是它的安全网）。
USE_SERVO_J = True

SERVO_RATE_HZ = 100.0            # 喂帧频率。Windows 默认定时粒度约 15ms，
                                 # real_arm 会调 timeBeginPeriod(1) 才撑得住 100Hz
# 固件看门狗：这么久没收到新帧就自动刹车。进程崩了、循环卡住都靠它兜。
# SDK 提示低于 30ms 会误触发；Windows 上留点余量，别贴着帧周期设。
SERVO_WATCHDOG_MS = 150
# 识别 / 规划时主线程会抢走 GIL，150ms 内容易断流。固件一刹车就坠落，
# 下一帧再把避让位打回去就是「猛地抬起」。空闲和走路都关掉看门狗，
# 只靠续帧顶住；进程崩了臂会慢慢沉，总比现场甩一下好。
SERVO_IDLE_WATCHDOG_MS = 0
# 保持目标与实测偏离超过这个值时，按 HOLD_RETURN_VEL 慢慢收回，禁止瞬移。
HOLD_DRIFT_TOL_RAD = math.radians(2.0)
HOLD_RETURN_VEL_RAD_S = math.radians(8.0)
# speed=100 时的关节角速度。和 move_j 的换算对齐（0.5 turns/s = 180°/s），
# 这样同一个 speed 值在两种模式下快慢差不多。
SERVO_VEL_AT_FULL_SPEED = math.radians(180.0)
SERVO_ACC_RAD_S2 = math.radians(12.0)  # 关节加减速。9°/s 大约 0.75s 起停，机座少抖
# 加加速度：把梯形速度的尖角抹成 S 曲线。越大起停越干脆，机座越容易晃。
SERVO_JERK_RAD_S3 = math.radians(40.0)
SERVO_MAX_VEL_RAD_S = 1.0        # 写进每帧的速度上限（rad/s），绝对安全帽
SERVO_MAX_STEP_RAD = math.radians(2.0)   # 单帧最大跳变，超了 SDK 会裁并计数
SERVO_MAX_LAG_RAD = math.radians(12.0)   # 跟随误差告警阈值（只计数，不停）
# 轻微前瞻，把 100Hz 微台阶抹平。太大路径会发飘、到位变钝。
SERVO_LOOKAHEAD_S = 0.04
# False = 位置通道（0x8090），固件位置环跟随，不用调增益，最稳。
# True = MIT 阻抗通道，手感更软但要调 kp/kd。
SERVO_USE_MIT = False
SERVO_SETTLE_TOL_RAD = math.radians(1.0)  # 喂完帧后确认到位的判定带
SERVO_SETTLE_TIMEOUT = 3.0       # 到位确认最多再等这么久（秒）
# 会话结束后电机留在什么模式。"hold" = 位置模式 0x0A，和 move_j 结束后一致，
# 后面还能直接接 move_j / 再开一次 servo，不用重新 enable。
SERVO_FINISH_MODE = "hold"
# 腕部滚转（J6）按夹爪 180° 对称折算到最短行程。
#
# 仿真不知道两指夹爪转 180° 是同一个夹持位形，会让 J6 从 0° 一路摇到 152°
# 再摇回来。实测一条 pick 路径 J6 行程 604°，占掉整段动作绝大部分时间
# （9°/s 下光 J6 就要 60 多秒），而等价的最短走法只要 56°。
#
# True  = 按单调段折算净转角，形状不变（平台仍是平台，抓取时腕部保持不动），
#         只把多余的整/半圈摘掉。代价：腕部不再逐点走仿真验证过的中间姿态，
#         夹着长条件（螺丝刀这类）搬运时，物体在空间中的朝向变化和仿真不同，
#         第一次跑新场景建议空载看一遍。
# False = 退回逐点解缠的老做法。注意 J6 软限位跨度 185.5°、等价分支间隔 180°，
#         两个分支都合法的重叠区只有 5.5°，逐点挑会在这条窄边界上反复翻转，
#         表现就是 J6 来回扭 180°。
WRIST_SHORTEST_PATH = True

GRIPPER_VEL = 0.15               # turns/s
GRIPPER_ACC = 0.5                # turns/s^2

# ---- 力控抓取 ----
# 合爪目标是夹爪软限位的闭合端，碰到零件（力矩或堵转）就停，不是停在规划宽度。
# 规划宽度只作对照：STL 常比实际抓点粗，按它停的话夹爪还张着一大截。
#
# 真正决定"夹多紧"的是下面两个，不是 GRASP_FORCE_THRESHOLD（那只是"夹到了"的
# 判定线，调它只影响什么时候认定成功，不影响出力大小）。
#
# 接触之后再往里过合 GRASP_OVERDRIVE，让位置环顶住，出力由 GRASP_EFFORT_RAW 限幅。
# 软物体/易碎件调小（2~4°），硬质金属件可以到 8~10°。
GRASP_OVERDRIVE_DEG = 6.0
GRASP_OVERDRIVE_RAD = math.radians(GRASP_OVERDRIVE_DEG)

# 固件侧力矩上限（raw int16），走 set_pos_vel_tqe 通道 —— 这才是"力控"本体。
# None = 用 robot.cfg 里的 gripper_max_torque_raw（当前 300）。
# M4438_30 上 1 raw ≈ 0.0053 Nm，300 ≈ 1.6 Nm。夹不紧就往上加，
# 但夹爪会一直堵转顶着，加太高会发热，别长时间夹着不放。
GRASP_EFFORT_RAW: int | None = None

# Python 侧"夹到了"的判定阈值（raw int16，比较的是实测 |torque|）。
# 必须落在（空载合爪的摩擦力矩, 力矩上限）之间：
#   太低  → 空载合爪的摩擦就触发，每次都报"抓到了"，实际手里是空的；
#   ≥上限 → 实测力矩永远到不了这个值，力控退化成"等堵转/超时才停"。
# 原来是 500，比上限 300 还高，所以这条判定一直是死的。
# 标定方法：空手跑一次合爪，看日志里的 peak，取它的 1.5~2 倍。
GRASP_FORCE_THRESHOLD = 220

if GRASP_OVERDRIVE_RAD < 0:
    raise ValueError(f"GRASP_OVERDRIVE_DEG 不能是负数，现在是 {GRASP_OVERDRIVE_DEG}")
if GRASP_FORCE_THRESHOLD <= 0:
    raise ValueError(
        f"GRASP_FORCE_THRESHOLD 必须是正数，现在是 {GRASP_FORCE_THRESHOLD}"
    )
if GRASP_EFFORT_RAW is not None and GRASP_FORCE_THRESHOLD >= GRASP_EFFORT_RAW:
    # 阈值 ≥ 上限时实测力矩永远够不着，力控会静默退化成等堵转，
    # 表面上还是"抓到了"，所以这里直接炸掉而不是打个告警
    raise ValueError(
        f"GRASP_FORCE_THRESHOLD({GRASP_FORCE_THRESHOLD}) 必须小于 "
        f"GRASP_EFFORT_RAW({GRASP_EFFORT_RAW})，否则力矩判定永远触发不了"
    )

GRIPPER_CALIB_FILE = os.path.join(REAL_DIR, "gripper_calib.json")


class RealConfigError(RuntimeError):
    """SDK 缺失、robot.cfg 解析失败等配置类错误。"""


# ---------------------------------------------------------------------------
# SDK 定位
# ---------------------------------------------------------------------------
def _sdk_candidates() -> list[str]:
    env = os.environ.get("FAFU_SDK_DIR", "").strip()
    cands = [env] if env else []
    cands += [
        os.path.join(REPO_ROOT, "fafu_arm_sdk-main", "fafu_robot_python"),
        os.path.join(REPO_ROOT, "fafu_arm_sdk", "fafu_robot_python"),
        os.path.join(os.path.dirname(REPO_ROOT), "fafu_arm_sdk-main", "fafu_robot_python"),
    ]
    return [c for c in cands if c]


def find_sdk_dir(required: bool = True) -> str | None:
    """返回 ``fafu_robot_python`` 目录；可用环境变量 ``FAFU_SDK_DIR`` 覆盖。"""
    for cand in _sdk_candidates():
        if os.path.isfile(os.path.join(cand, "fafu_robot_controller.py")):
            return os.path.abspath(cand)
    if required:
        raise RealConfigError(
            "找不到 fafu SDK（fafu_robot_python）。已尝试：\n  "
            + "\n  ".join(_sdk_candidates())
            + "\n请设置环境变量 FAFU_SDK_DIR 指向该目录。"
        )
    return None


def ensure_sdk_on_path() -> str:
    """把 SDK 目录插到 ``sys.path`` 最前，返回该目录。"""
    import sys

    sdk_dir = find_sdk_dir(required=True)
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    return sdk_dir


# ---------------------------------------------------------------------------
# robot.cfg 解析（自己解析，避免依赖 fafu_motor.pyd）
# ---------------------------------------------------------------------------
def _strip_comment(line: str) -> str:
    for mark in ("#", ";"):
        idx = line.find(mark)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def parse_robot_cfg(cfg_path: str) -> dict:
    """把 ``robot.cfg`` 读成 ``{key: 原始字符串}``。"""
    if not os.path.isfile(cfg_path):
        raise RealConfigError(f"robot.cfg 不存在：{cfg_path}")
    out: dict[str, str] = {}
    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            out[key.strip().lower()] = val.strip()
    return out


def _pos_to_rad(value: float, pos_unit: str) -> float:
    unit = (pos_unit or "degrees").lower()
    if unit.startswith("deg"):
        return math.radians(value)
    if unit.startswith("turn"):
        return value * 2.0 * math.pi
    return float(value)


@dataclass
class RealRobotConfig:
    """真机侧的静态参数，全部已换算成弧度。"""

    sdk_dir: str
    cfg_path: str
    port: str = "auto"
    baudrate: int = 4_000_000
    motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    gripper_id: int = 7
    pos_unit: str = "degrees"
    # 6 个手臂关节的软限位（rad），顺序与 joint_ids 一致
    joint_limits: tuple[tuple[float, float], ...] = ()
    gripper_limits: tuple[float, float] = (0.0, math.radians(105.0))
    gripper_max_torque_raw: int = 300

    @property
    def joint_ids(self) -> tuple[int, ...]:
        return tuple(i for i in self.motor_ids if i != self.gripper_id)

    @property
    def num_joints(self) -> int:
        return len(self.joint_ids)

    def limits_deg(self) -> list[tuple[float, float]]:
        return [(math.degrees(lo), math.degrees(hi)) for lo, hi in self.joint_limits]

    def clamp(self, q) -> tuple[list[float], list[int]]:
        """把一组关节角裁到软限位内，返回 ``(裁剪后, 被裁的关节下标)``。"""
        out: list[float] = []
        hit: list[int] = []
        for i, v in enumerate(q):
            v = float(v)
            if i < len(self.joint_limits):
                lo, hi = self.joint_limits[i]
                if v < lo or v > hi:
                    hit.append(i)
                    v = min(max(v, lo), hi)
            out.append(v)
        return out, hit


def load_real_config(cfg_path: str | None = None, sdk_dir: str | None = None) -> RealRobotConfig:
    """读取 SDK 目录下的 ``robot.cfg``，组装成 :class:`RealRobotConfig`。"""
    sdk_dir = os.path.abspath(sdk_dir) if sdk_dir else find_sdk_dir(required=True)
    cfg_path = os.path.abspath(cfg_path) if cfg_path else os.path.join(sdk_dir, "robot.cfg")
    raw = parse_robot_cfg(cfg_path)

    pos_unit = raw.get("pos_unit", "degrees")
    motor_ids = tuple(
        int(x) for x in raw.get("motor_ids", "1,2,3,4,5,6,7").replace(",", " ").split()
    )
    if not motor_ids:
        raise RealConfigError(f"{cfg_path} 里 motor_ids 为空")
    gripper_id = motor_ids[-1]

    limits: dict[int, tuple[float, float]] = {}
    for mid in motor_ids:
        val = raw.get(f"limits.{mid}")
        if not val:
            continue
        parts = [p for p in val.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise RealConfigError(f"{cfg_path} 里 limits.{mid} 格式不对：{val!r}")
        lo, hi = (_pos_to_rad(float(p), pos_unit) for p in parts)
        limits[mid] = (min(lo, hi), max(lo, hi))

    joint_ids = [i for i in motor_ids if i != gripper_id]
    missing = [i for i in joint_ids if i not in limits]
    if missing:
        raise RealConfigError(f"{cfg_path} 缺少关节 {missing} 的 limits.* 配置")

    try:
        baudrate = int(raw.get("baudrate", "4000000"))
    except ValueError:
        baudrate = 4_000_000
    try:
        torque_raw = int(raw.get("gripper_max_torque_raw", "300"))
    except ValueError:
        torque_raw = 300

    return RealRobotConfig(
        sdk_dir=sdk_dir,
        cfg_path=cfg_path,
        port=raw.get("port", "auto"),
        baudrate=baudrate,
        motor_ids=motor_ids,
        gripper_id=gripper_id,
        pos_unit=pos_unit,
        joint_limits=tuple(limits[i] for i in joint_ids),
        gripper_limits=limits.get(gripper_id, (0.0, math.radians(105.0))),
        gripper_max_torque_raw=torque_raw,
    )


# ---------------------------------------------------------------------------
# 夹爪映射：仿真是平移开口（米），真机 M7 是旋转关节（弧度）
# ---------------------------------------------------------------------------
@dataclass
class GripperMap:
    """开口宽度 → 夹爪角度。

    默认按线性插值：``0 m`` 对应软限位下端（合），``SIM_JAW_OPEN_MAX`` 对应上端（开）。
    真机上机构基本不会是理想线性，跑一次 ``calibrate_gripper.py`` 会生成
    ``gripper_calib.json``，里面是若干 ``[开口(m), 角度(deg)]`` 采样点，
    存在时优先用这张表做分段线性插值。
    """

    angle_min: float
    angle_max: float
    width_max: float = SIM_JAW_OPEN_MAX
    table: list[tuple[float, float]] = field(default_factory=list)  # [(width_m, angle_rad)]
    source: str = "linear"

    @classmethod
    def from_config(cls, cfg: RealRobotConfig, calib_path: str | None = None) -> "GripperMap":
        lo, hi = cfg.gripper_limits
        gm = cls(angle_min=lo, angle_max=hi)
        path = calib_path or GRIPPER_CALIB_FILE
        if os.path.isfile(path):
            gm.load_table(path)
        return gm

    def load_table(self, path: str) -> None:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        pts = data.get("points") if isinstance(data, dict) else data
        table: list[tuple[float, float]] = []
        for item in pts or []:
            width_m = float(item[0] if isinstance(item, (list, tuple)) else item["width_m"])
            if isinstance(item, (list, tuple)):
                angle = math.radians(float(item[1]))
            elif "angle_deg" in item:
                angle = math.radians(float(item["angle_deg"]))
            else:
                angle = float(item["angle_rad"])
            table.append((width_m, angle))
        if len(table) < 2:
            raise RealConfigError(f"{path} 至少需要两个标定点")
        self.table = sorted(table, key=lambda p: p[0])
        self.width_max = self.table[-1][0]
        self.source = os.path.basename(path)

    def width_to_angle(self, width_m: float) -> float:
        """开口（米）→ 夹爪角度（弧度），结果已裁到软限位内。"""
        w = max(0.0, float(width_m))
        if self.table:
            pts = self.table
            if w <= pts[0][0]:
                angle = pts[0][1]
            elif w >= pts[-1][0]:
                angle = pts[-1][1]
            else:
                angle = pts[-1][1]
                for (w0, a0), (w1, a1) in zip(pts, pts[1:]):
                    if w0 <= w <= w1:
                        t = 0.0 if w1 == w0 else (w - w0) / (w1 - w0)
                        angle = a0 + t * (a1 - a0)
                        break
        else:
            span = max(self.width_max, 1e-6)
            t = min(w / span, 1.0)
            angle = self.angle_min + t * (self.angle_max - self.angle_min)
        return min(max(angle, self.angle_min), self.angle_max)

    def describe(self) -> str:
        return (
            f"夹爪映射[{self.source}] "
            f"开口 0~{self.width_max * 1000:.0f}mm → "
            f"角度 {math.degrees(self.angle_min):.1f}~{math.degrees(self.angle_max):.1f}°"
        )
