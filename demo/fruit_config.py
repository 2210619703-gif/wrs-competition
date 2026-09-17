# -*- coding: utf-8 -*-
"""水果真机演示的配置。改这里就不用记一长串命令行。

机械臂走 ``demo/real``（从 ``tiaozhanbei/real`` 拷出的相机 / ``servo_j`` / 轨迹适配），
下发仍是 fafu / Panthera-HT。相机是**眼在手上**。标定：

    python demo/hand_eye_calib.py

产物 ``handeye_eye_in_hand.json``（``affine_mat`` = 法兰 → 相机 T_flange_cam）。
没有这份文件就不能做识别抓取。世界 ← 相机 = FK(当前关节) @ T_flange_cam。

三个示教位请现场读关节角（度）填进来::

    python demo/read_joints.py

把打印出来的 6 个数填到下面对应元组。还是 None 时，去该位的 RUN_MODE 会直接报错。
左右空位同样示教，填 ``SLOT_LEFT_CONF_DEG`` / ``SLOT_RIGHT_CONF_DEG``。
"""
from __future__ import annotations

import math
import os
import sys

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DEMO_DIR, os.pardir))
# 真机执行层：从 tiaozhanbei/real 拷来的独立副本，不再依赖 tiaozhanbei 目录。
REAL_DIR = os.path.join(DEMO_DIR, "real")
if REAL_DIR not in sys.path:
    sys.path.insert(0, REAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from real_config import (  # noqa: E402
    DEFAULT_MOVE_SPEED as REAL_DEFAULT_MOVE_SPEED,
    MAX_MOVE_SPEED,
    RELEASE_MODE_ON_CLOSE,
    RELEASE_MODES,
    SIM_JAW_CLOSE_MIN,
    SIM_JAW_OPEN_MAX,
    ZERO_CONF,
)

# ---------------------------------------------------------------------------
# 现场示教关节角（度）——测完填这里
# 顺序 J1~J6，和 demo/read_joints.py 示教读数一致。
# ---------------------------------------------------------------------------
# A 收纳筐：夹着水果送到这里再松爪
BIN_A_CONF_DEG = (-37.51, 97.96, 81.72, -54.11, -4.54, 5.08)
# B 收纳筐
BIN_B_CONF_DEG = (None, None, None, None, None, None)
# 桌面左边 / 右边空位（不是筐）。夹爪移到空位上方，python demo/read_joints.py
# 把打印的 6 个数填进来。还是 None 时，「把橙子放到左边」会提示还没示教。
SLOT_LEFT_CONF_DEG = (None, None, None, None, None, None)
SLOT_RIGHT_CONF_DEG = (None, None, None, None, None, None)
# 观察位：眼在手上拍桌面的位形（YOLO 在这里拍）。
# python demo/find_look_pose.py 搜出来的：视线偏离竖直 10°，相机高 35cm，
# 目标区全覆盖，夹爪盲区正好甩到区边界外。相机越接近俯视，识别坐标的水平
# 偏差越小（这个位 ≈4mm，原来那个 61° 斜视的位 ≈21mm 而且只看得到 45%）。
# 想退回去：(-5.69, 0.00, 75.17, -103.39, -3.53, 3.60)
LOOK_CONF_DEG = (9.33, 71.74, 78.36, -85.76, 0.41, 13.11)

# ---------------------------------------------------------------------------
# RUN_MODE
# ---------------------------------------------------------------------------
# look        只去观察位（张开夹爪），方便对相机
# zero        只去电机零位
# bin_a       只去 A 收纳筐
# bin_b       只去 B 收纳筐
# slot_left   只去左边空位（示教核对）
# slot_right  只去右边空位
# vision      去观察位 → 拍照 → YOLO 水果 → 写出 annotations，不动抓取
# once        观察位识别 → 按 DEFAULT_CMD 或 --cmd 抓放一次就退
# interactive 打开 UI / 命令行循环（界面入口默认这个）
RUN_MODE = "bin_a"
RUN_MODES = (
    "look",
    "zero",
    "bin_a",
    "bin_b",
    "slot_left",
    "slot_right",
    "vision",
    "once",
    "interactive",
)
# look 模式到位后直接开 fruit_yolo_realtime 那个实时 YOLO 窗口，
# 关窗（q / Esc）才回静置位。窗口里按 j 打印关节角，用的是同一条臂连接。
# 相机开不起来会自动退回「按回车继续」。--live / --no-live 可临时覆盖。
LOOK_SHOW_LIVE = True

# 真的下发。False = dry-run 只打印。第一次填完示教位建议先 False。
EXECUTE_ON_REAL = True
# 必须走 servo_j
USE_SERVO_J = True
DEFAULT_MOVE_SPEED = min(int(REAL_DEFAULT_MOVE_SPEED), 8)
HOME_ON_EXIT = True
JAW_OPEN_ON_LOOK = True

# once 模式没传 --cmd 时用这句
DEFAULT_CMD = "抓取橙子"

# 谷歌语音走 Clash 混合端口。Python 不会自动用 Windows 系统代理，
# 必须在这里写上。Clash 信息页里的「混合代理端口」是多少就填多少
# （你这台是 7897；老版本常见 7890）。空字符串 = 不走代理。
STT_PROXY = "http://127.0.0.1:7897"
# 中文语音播报。用本机 Windows 语音，不走谷歌。False 就只打日志。
TTS_ENABLED = True

# ---------------------------------------------------------------------------
# 眼在手上 + YOLO
# ---------------------------------------------------------------------------
HANDEYE_JSON = os.path.join(DEMO_DIR, "handeye_eye_in_hand.json")
YOLO_WEIGHTS = os.path.join(DEMO_DIR, "yolo26s-seg.pt")
FRUIT_CLASS_IDS = (46, 47, 49)  # COCO banana / apple / orange
FRUIT_EN = ("banana", "apple", "orange")
FRUIT_ZH = {"banana": "香蕉", "apple": "苹果", "orange": "橙子"}
ZH_TO_EN = {
    "香蕉": "banana",
    "苹果": "apple",
    "橙子": "orange",
    "橘子": "orange",
    "桔子": "orange",
    "橙": "orange",
    # 谷歌常把「橙子」听成这些
    "柿子": "orange",
    "陈子": "orange",
    "成子": "orange",
    "城子": "orange",
    "秤子": "orange",
}

YOLO_CONF = 0.35
YOLO_IOU = 0.50
# 推理输入边长。现场橙子在 640 能稳定检出；不要为了快降到 480。
YOLO_IMGSZ = 640
CAMERA_RESOLUTION = "mid"
# 连拍取深度中位数。彩色用最后一帧。现场要 5 帧给深度/YOLO 投票。
SNAPSHOT_FRAMES = 5
LOOK_SETTLE_S = 0.8

# 语音点动：每次沿世界系挪这么多。你站在基座后面看桌子：
# 前=+X 远离基座，左=+Y，上=+Z。
JOG_STEP_M = 0.04
JOG_Z_MIN_M = 0.08
WANDER_STEPS = 6
JOG_AXIS = {
    "forward": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}
JOG_ZH = {
    "forward": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "up": "上",
    "down": "下",
}
# 点动范围比水果工作区略宽，观察位 / A 框附近也能挪。最低高度防扎桌。
JOG_BOUNDS_M = {
    "x": (-0.05, 0.85),
    "y": (-0.60, 0.60),
    "z": (0.08, 0.55),
}

# 工作区（世界系，米）。按现场桌面改。
WORKSPACE_BOUNDS_M = {
    "x": (0.05, 0.70),
    "y": (-0.45, 0.45),
    "z": (-0.05, 0.45),
}
MIN_POINTS_PER_OBJECT = 40
MAX_DEPTH_M = 1.2

# 一条指令里有多个同类水果时怎么做：
#   False：拍一次，把看到的全部规划成一批顺序做完。快，但坐标是那一帧的，
#          抓第一个时碰动了第二个，第二个还按旧坐标去抓。
#   True ：每抓完一个就回观察位重拍重新规划。慢一些，碰动了也不怕。
# 顺序都是 YOLO 的置信度从高到低，和水果在桌上的位置无关。
RECAPTURE_EACH_FRUIT = False
# 逐个重拍时的轮数上限，防止某个水果一直抓不走把流程卡死。
MAX_PICK_ROUNDS = 10
# 逐个重拍时，这一轮没抓走的水果下一轮就跳过；离上次目标这么近算同一个。
SKIP_RETRY_RADIUS_M = 0.04

# 深度相机只拍得到水果朝着它那半边表面，点云中位数落在那半边上，比真中心近
# 大半个半径。相机在 LOOK 位是斜着往前看的，所以这个偏差大头在水平方向，
# 表现为每次都抓在水果靠相机那一侧。识别时沿视线把中心往前推 K×半径。
# 球面上可见半球的深度中位数大约是 0.7R，所以 K 默认 0.7。
# 还是偏近就调大（上限 1.0 = 推到整个半径），过头了就调小。
SURFACE_TO_CENTER_K = 0.7
# 半径是从轮廓宽度现算的，这里只是防噪声算飞的上下限。
FRUIT_RADIUS_BOUNDS_M = (0.015, 0.05)
# 上面都补完还剩的固定偏差，直接在这里平移（世界系 x/y/z，米）。
# 量法：把夹爪尖顶到水果中心，python demo/read_joints.py 读 TCP 世界坐标，
# 减去日志里同一个水果的识别坐标，差值填进来。
WORLD_TRIM_M = (0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# 仿真规划器（WRS PantheraHT IK，不是工业件那套 grasp pickle）
# ---------------------------------------------------------------------------
APPROACH_Z_M = 0.10          # 沿接近轴后退多少（竖直抓就是正上方，斜抓是斜后上方）
# 抓起后相对抓取点再竖直抬这么高。放到框时还会加高到够得过框沿。
LIFT_Z_M = 0.08
# 件离基座近时「抓取点 + 10cm」常常没有 IK 解，但同一点近几厘米就有。
# 姿态（竖直/斜）都试遍还不行，规划器才按这个步长缩短接近段，
# 缩到这个下限还无解才算这个水果抓不了。
APPROACH_Z_STEP_M = 0.02
MIN_APPROACH_Z_M = 0.03
# 从水果顶面往下扎多少。固定值是下限；球形件还会按量到的高度再扎
# GRASP_DEPTH_FRAC 那一截，取两者较大的。还浅就加大 frac / extra。
GRASP_DEPTH_M = {
    "banana": 0.015,
    "apple": 0.040,
    "orange": 0.038,
}
# 按件高从顶面往下的比例。0.5 = 赤道。苹果和橙子都略过赤道，避免合在上 1/3。
GRASP_DEPTH_FRAC = {
    "banana": 0.35,
    "apple": 0.66,
    "orange": 0.62,
}
# 按比例算完后再往下扎这么多。觉得还在上 1/3 合爪就加。
GRASP_EXTRA_SINK_M = {
    "banana": 0.0,
    "apple": 0.014,
    "orange": 0.012,
}
# TCP 最低不许低于这个高度，防止标定有偏差时直接怼进桌面。桌面标定后约 z=0。
MIN_GRASP_Z_M = 0.012
# 也不许扎到可见点云下沿以下，免得真穿到桌面里。
MIN_GRASP_ABOVE_BOTTOM_M = 0.008
JAW_CLOSE_M = {
    "banana": 0.028,
    "apple": 0.045,
    "orange": 0.048,
}
IK_YAW_OFFSETS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)
# 夹爪偏离竖直的角度，按顺序试：先试竖直顶抓，不行就越来越斜着下手。
# 斜着抓时两根手指仍然水平（绕手指开合轴倾），件靠基座太近时往往只有斜的有解。
GRASP_TILT_DEG = (0.0, 20.0, 35.0, 45.0)
PATH_STEPS = 24              # 关节线性插值点数（仿真规划器输出给 servo_j）
# A/B 料框高度。夹着水果横移时，TCP 必须高于「框沿 + 水果下垂 + 余量」，
# 否则橙子/苹果底部会蹭到框。框更高就改大。
BIN_HEIGHT_M = 0.12
# 合爪后水果还垂在 TCP 下面这么多（大约半径）。宁大勿小。
FRUIT_BELOW_TCP_M = 0.05
# 越过框沿后再留的空，防标定误差和摆放歪一点。
PLACE_CLEAR_M = 0.06
# 松爪以后再竖直抬这么多才横移离开，避免夹爪/件扫到框沿。
PLACE_EXIT_EXTRA_M = 0.08
# 桌面世界高度。手眼如果把桌面标到不是 0，填实际值，框沿高度会跟着加。
TABLE_Z_M = 0.0
# 转场 TCP 世界高度下限；真正用的是 max(这个值, 框沿+下垂+余量)。
TRANSIT_Z_M = 0.28


def carry_z_m() -> float:
    """夹着水果横移时 TCP 的世界高度。"""
    need = float(TABLE_Z_M) + float(BIN_HEIGHT_M) + float(FRUIT_BELOW_TCP_M) + float(PLACE_CLEAR_M)
    return max(float(TRANSIT_Z_M), need)


def exit_bin_z_m() -> float:
    """松爪后离开 A/B 框时的 TCP 高度，比夹着走的时候再高一截。"""
    return carry_z_m() + float(PLACE_EXIT_EXTRA_M)

OUTPUT_DIR = os.path.join(DEMO_DIR, "outputs")
VISION_DIR = os.path.join(OUTPUT_DIR, "vision")

if RUN_MODE not in RUN_MODES:
    raise ValueError(f"RUN_MODE 只能是 {RUN_MODES}，现在是 {RUN_MODE!r}")


class FruitConfigError(RuntimeError):
    pass


def _deg_to_rad(name: str, deg) -> tuple[float, ...]:
    if deg is None or any(v is None for v in deg):
        raise FruitConfigError(
            f"还没填 {name}。打开 demo/fruit_config.py，把现场读到的 "
            "J1~J6 关节角（度）写进对应的 *_CONF_DEG。"
        )
    if len(deg) != 6:
        raise FruitConfigError(f"{name} 必须是 6 个数，现在是 {deg!r}")
    return tuple(math.radians(float(v)) for v in deg)


DEST_ZH = {
    "A": "A框",
    "B": "B框",
    "LEFT": "左边",
    "RIGHT": "右边",
}


def dest_label(which) -> str:
    key = str(which or "").strip().upper()
    if key in ("LEFT", "L", "SLOT_LEFT", "左", "左边", "左侧"):
        return DEST_ZH["LEFT"]
    if key in ("RIGHT", "R", "SLOT_RIGHT", "右", "右边", "右侧"):
        return DEST_ZH["RIGHT"]
    if key in ("A", "BIN_A", "框A", "A框", "A筐"):
        return DEST_ZH["A"]
    if key in ("B", "BIN_B", "框B", "B框", "B筐"):
        return DEST_ZH["B"]
    return DEST_ZH.get(key, str(which or ""))


def look_conf():
    return _deg_to_rad("LOOK_CONF_DEG（观察位）", LOOK_CONF_DEG)


def bin_a_conf():
    return _deg_to_rad("BIN_A_CONF_DEG（A 收纳筐）", BIN_A_CONF_DEG)


def bin_b_conf():
    return _deg_to_rad("BIN_B_CONF_DEG（B 收纳筐）", BIN_B_CONF_DEG)


def slot_left_conf():
    return _deg_to_rad("SLOT_LEFT_CONF_DEG（左边空位）", SLOT_LEFT_CONF_DEG)


def slot_right_conf():
    return _deg_to_rad("SLOT_RIGHT_CONF_DEG（右边空位）", SLOT_RIGHT_CONF_DEG)


def bin_conf(which: str):
    key = str(which or "").strip().upper()
    if key in ("A", "BIN_A", "框A", "A框", "A筐"):
        return bin_a_conf(), "A"
    if key in ("B", "BIN_B", "框B", "B框", "B筐"):
        return bin_b_conf(), "B"
    if key in ("LEFT", "L", "SLOT_LEFT", "左", "左边", "左侧"):
        return slot_left_conf(), "LEFT"
    if key in ("RIGHT", "R", "SLOT_RIGHT", "右", "右边", "右侧"):
        return slot_right_conf(), "RIGHT"
    raise FruitConfigError(f"放置目标只能是 A / B / 左边 / 右边，收到 {which!r}")


def conf_is_filled(deg) -> bool:
    return deg is not None and all(v is not None for v in deg)
