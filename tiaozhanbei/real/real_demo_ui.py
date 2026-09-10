# -*- coding: utf-8 -*-
"""真机演示界面：D405 实时画面 + 语音指令 → 智能体 → 机械臂执行。

把三个模块串成一条链，界面布局和仿真那套 (``decision/agent_demo_ui.py``) 一致：

    左侧  录音按钮 + 音量条 + 终端输出
    右侧  D405 实时画面（识别框叠在上面）
    底部  录音结束 → 转写 → YOLO识别 → 智能体规划 → 轨迹规划 → 机械臂执行

和仿真界面的区别是右侧不贴 Panda3D 窗口，直接播相机。流程::

    启动先回避让位（终端有进度）→ 到位后再弹出界面
      → 相机开始直播（不识别）
      → 你说一句话，或在启动界面的那个终端里打字回车
      → 结束录音 / 回车那一刻抓一帧
      → YOLO + 手眼 → annotations.json
      → 决策智能体（Dify，失败回退本地规则）→ 任务 JSON
      → 仿真规划器导出关节轨迹 → 限位适配
      → dry-run 打印，或真机执行（画面继续直播）

**默认 dry-run**，不勾「允许真机执行」绝不下发指令。执行期间画面只是记录，
不参与闭环：轨迹按抓帧那一刻的零件位置开环规划，桌面动过要重新下指令。

用法::

    python tiaozhanbei/real/real_demo_ui.py

启动后可在这个终端里打字回车下指令（和界面录音同一条链路），``q`` 退出。

没有相机时可以拿存下来的帧顶替，界面其余部分照常走::

    python tiaozhanbei/real/d405_camera.py --save-dir tiaozhanbei/real/outputs/frame01
    python tiaozhanbei/real/real_demo_ui.py --frame-dir tiaozhanbei/real/outputs/frame01
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import numpy as np

_REAL_DIR = Path(__file__).resolve().parent
_TB_DIR = _REAL_DIR.parent
_DECISION_DIR = _TB_DIR / "decision"
for _p in (_REAL_DIR, _DECISION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ========== Dify（与决策界面保持一致，改这里就够）==========
DIFY_API_URL = os.getenv("DIFY_API_URL", "http://localhost:8080/v1/workflows/run")
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "app-pcJzT4Jht70CXyFoWGRp8ts9")
os.environ["DIFY_API_URL"] = DIFY_API_URL
os.environ["DIFY_API_KEY"] = DIFY_API_KEY
# ==========================================================

try:
    from PIL import Image, ImageTk
except ImportError:  # 没有 Pillow 就只能显示文字，其余流程不受影响
    Image = None
    ImageTk = None

from d405_camera import RESOLUTIONS  # noqa: E402
from export_sim_traj import OUTPUT_DIR, ExportError, export_trajectory  # noqa: E402
from real_arm import RealArm, RealArmError, execute_plan, release_at_place  # noqa: E402
from real_config import (  # noqa: E402
    DEFAULT_MOVE_SPEED,
    MAX_MOVE_SPEED,
    PARK_CONF,
    RELEASE_MODE_ON_CLOSE,
    RELEASE_MODES,
    START_FROM_PARK,
    USE_SERVO_J,
    GripperMap,
    RealConfigError,
    load_real_config,
)
from run_vision_real import NO_DEST, select_targets  # noqa: E402
from traj_adapter import adapt, format_plan, load_sim_traj, save_plan  # noqa: E402
from vision_config import (  # noqa: E402
    CAMERA_RESOLUTION,
    CAMERA_SERIAL,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
    VisionConfigError,
    describe_handeye,
    load_handeye,
)
from vision_to_annotations import (  # noqa: E402
    VISION_DIR,
    VisionError,
    detect_to_annotations,
    save_annotations,
    save_preview,
)

from agent.cn_stt import CnSttError, transcribe_pcm16  # noqa: E402
from agent.demo_voice_agent import (  # noqa: E402
    audio_level_stats,
    rehydrate_tasks,
    run_agent,
)
from agent.run_batch import API_KEY, API_URL  # noqa: E402

SAMPLE_ID = "real01"                      # 识别结果写在 outputs/vision/<SAMPLE_ID>/
DEFAULT_DEST_XYZ = (0.30, -0.20, 0.0)     # 收纳盒世界坐标，和决策模块对齐
DEFAULT_DESTINATION = "收纳盒"
SNAPSHOT_FRAMES = 5                       # 曝光稳住后连拍帧数（深度中位数 + YOLO 投票）
PREVIEW_FPS = 25
PARK_SETTLE_S = 0.8                       # 退到避让位后等直播刷掉带机械臂的旧帧

REFERENCE_VOICE = "把螺丝刀放到收纳盒 / 随便拿2个零件放进收纳盒 / 把所有零件放进收纳盒"

_NULL_DEST = frozenset({"", "null", "none", "nil", "无"})
_PLACE_WORDS = ("放到", "放入", "放置", "放进", "移到", "移动到", "装箱", "收纳盒", "料框", "料箱")
_PICK_WORDS = ("夹取", "抓取", "拿取", "取出", "抓起", "拿起", "夹起", "帮我拿", "拿一下", "给我拿")
_PLACE_ACTIONS = frozenset({
    "place",
    "pick_place",
    "multi_pick_place",
    "pack_region",
    "pack_n_parts",
    "pack_all_box",
    "multi_named_pick_place",
    "pack_batch",
    "multi_agent_pack",
})
_PICK_ACTIONS = frozenset({
    "pick",
    "pick_place",
    "multi_pick",
    "multi_pick_place",
    "pack_region",
    "pack_n_parts",
    "pack_all_box",
    "multi_named_pick_place",
    "pack_batch",
})


def _dest_is_null(destination) -> bool:
    if destination is None:
        return True
    return str(destination).strip().lower() in _NULL_DEST


def cmd_intent(cmd: str, *, holding: bool = False) -> str | None:
    """从口语判断 pick / place / pick_place；看不出就返回 None，再交给智能体 JSON。"""
    text = (cmd or "").strip()
    if not text:
        return None
    has_place = any(w in text for w in _PLACE_WORDS)
    has_pick = any(w in text for w in _PICK_WORDS)
    if has_pick and has_place:
        return "pick_place"
    if has_pick:
        return "pick"
    if has_place:
        return "place" if holding else "pick_place"
    return None


def agent_intent(task: dict, cmd: str = "") -> str:
    """按智能体 JSON 判断意图。destination=null 且没有 place 步就是只抓。"""
    task = task or {}
    text = (cmd or task.get("original_cmd") or "").strip()
    action = str(task.get("action") or "").lower()
    steps = [str(s.get("action") or "").lower() for s in (task.get("tasks") or [])]
    dest = task.get("destination")
    place = (
        action in _PLACE_ACTIONS
        or "place" in steps
        or not _dest_is_null(dest)
        or any(w in text for w in _PLACE_WORDS)
    )
    pick = (
        action in _PICK_ACTIONS
        or "pick" in steps
        or any(w in text for w in _PICK_WORDS)
    )
    if place and pick:
        return "pick_place"
    if place:
        return "place"
    return "pick"


def _agent_pick_names(task: dict) -> list[str]:
    picks: list[str] = []
    obj_name = str(task.get("object") or "").strip()
    if obj_name and obj_name not in ("所有零件", "全部"):
        picks.append(obj_name)
    for step in task.get("tasks") or []:
        if not isinstance(step, dict):
            continue
        name = str(step.get("object") or "").strip()
        if name and name not in picks and name not in ("所有零件", "全部"):
            picks.append(name)
    for item in task.get("batch") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("object") or "").strip()
        if name and name not in picks and name not in ("所有零件", "全部"):
            picks.append(name)
    return picks


def _vision_geom(obj: dict) -> dict:
    yaw = float(obj["pose_6d"]["yaw"])
    return {
        "object": obj["class"],
        "coordinate": [float(obj["pose_6d"]["x"]), float(obj["pose_6d"]["y"]), 0.0],
        "angle": yaw,
        "rpy": [0.0, 0.0, yaw],
        "pose_state": obj.get("state", "normal"),
        "destination": NO_DEST,
        "dest_coordinate": [0.0, 0.0, 0.0],
    }


def _apply_vision_geom(item: dict, obj: dict) -> None:
    item.update(_vision_geom(obj))


def _match_target(name: str, targets: list[dict], index: int) -> dict:
    key = str(name or "").strip()
    if key:
        for obj in targets:
            cls = str(obj.get("class") or "")
            yolo = str(obj.get("yolo_class") or "")
            if key == cls or key == yolo or key in cls or cls.endswith(key):
                return obj
    if 0 <= index < len(targets):
        return targets[index]
    return targets[0]


def patch_agent_task_for_planner(task: dict, targets: list[dict], cmd: str) -> dict:
    """智能体 JSON 当底板，几何改成同一帧 YOLO、规划器格式（贴桌 z=0、不建盒）。"""
    if not targets:
        raise RuntimeError("没有选中任何零件")
    out = copy.deepcopy(task) if isinstance(task, dict) else {}
    geom0 = _vision_geom(targets[0])
    out["original_cmd"] = out.get("original_cmd") or cmd
    out["status"] = out.get("status") or "valid"
    out["retry"] = out.get("retry", 0)
    out["max_retry"] = out.get("max_retry", 3)
    out.update(geom0)

    if len(targets) == 1:
        out["action"] = "pick"
        out.setdefault("task_id", "real001")
        steps = [s for s in (out.get("tasks") or []) if isinstance(s, dict)]
        kept = []
        for step in steps:
            if str(step.get("action") or "").lower() in ("move", "place"):
                continue
            _apply_vision_geom(step, _match_target(step.get("object"), targets, 0))
            kept.append(step)
        if not any(str(s.get("action") or "").lower() == "pick" for s in kept):
            common = {k: geom0[k] for k in geom0}
            kept = [
                {"task_id": "t000", "step": 1, "action": "perceive", **common},
                {"task_id": "t001", "step": 2, "action": "pick", **common},
            ]
        out["tasks"] = kept
        out.pop("batch", None)
        return out

    out["action"] = "multi_pick"
    out["object"] = "所有零件"
    out.setdefault("task_id", "real_batch")
    old_batch = [b for b in (out.get("batch") or []) if isinstance(b, dict)]
    new_batch = []
    used = set()
    for i, obj in enumerate(targets):
        item = None
        for bi, b in enumerate(old_batch):
            if bi in used:
                continue
            if _match_target(b.get("object"), targets, i) is obj:
                item = dict(b)
                used.add(bi)
                break
        if item is None:
            item = dict(old_batch[i]) if i < len(old_batch) else {}
        _apply_vision_geom(item, obj)
        item["steps"] = ["perceive", "pick"]
        item.setdefault("reason", f"视觉识别 {obj.get('yolo_class') or obj.get('class')}")
        new_batch.append(item)
    out["batch"] = new_batch
    if isinstance(out.get("tasks"), list):
        kept = []
        for i, step in enumerate(out["tasks"]):
            if not isinstance(step, dict):
                continue
            if str(step.get("action") or "").lower() in ("move", "place"):
                continue
            _apply_vision_geom(step, _match_target(step.get("object"), targets, i))
            kept.append(step)
        out["tasks"] = kept
    return out

# 深色低对比，和决策界面同一套配色
BG = "#101722"
PANEL = "#121C29"
INK = "#D7E2EC"
MUTED = "#8EA0B5"
ACCENT = "#1F6F8B"
ACCENT_HOVER = "#245F77"
OK = "#2A9D8F"
DANGER = "#A8553A"
BORDER = "#263447"


class CameraWorker(threading.Thread):
    """后台持续取流，UI 只读最新一帧。

    相机管道只归这个线程管：识别要用的「稳定帧」也在这里 ``capture_median``，
    避免 UI 线程和它抢同一个 pipeline。
    """

    def __init__(self, args, logger) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.log = logger
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._ready = threading.Event()
        self._error: str = ""
        self._snap_req = threading.Event()
        self._snap_done = threading.Event()
        self._snap_out: tuple[str, object] | None = None

    # -- 对外 ---------------------------------------------------------------
    @property
    def error(self) -> str:
        return self._error

    def wait_ready(self, timeout: float = 20.0) -> bool:
        return self._ready.wait(timeout)

    def latest(self):
        with self._lock:
            return self._latest

    def grab_stable(self, timeout: float = 20.0):
        """要一帧深度中位数帧，给 YOLO 用。"""
        self._snap_out = None
        self._snap_done.clear()
        self._snap_req.set()
        if not self._snap_done.wait(timeout):
            raise RuntimeError("抓帧超时，相机可能已断开")
        kind, payload = self._snap_out or ("err", RuntimeError("抓帧失败"))
        if kind != "ok":
            raise payload if isinstance(payload, Exception) else RuntimeError(str(payload))
        return payload

    def stop(self) -> None:
        self._stop_evt.set()

    # -- 线程体 -------------------------------------------------------------
    def run(self) -> None:
        try:
            if self.args.frame_dir:
                self._run_offline()
            else:
                self._run_camera()
        except Exception as e:
            self._error = str(e)
            self._ready.set()
            self.log(f"[cam] 取流结束：{e}")

    def _run_offline(self) -> None:
        from d405_camera import load_frame

        frame = load_frame(self.args.frame_dir)
        self.log(f"[cam] 离线帧回放 {self.args.frame_dir}（没有连相机）")
        with self._lock:
            self._latest = frame
        self._ready.set()
        while not self._stop_evt.is_set():
            if self._snap_req.is_set():
                self._snap_out = ("ok", frame)
                self._snap_req.clear()
                self._snap_done.set()
            time.sleep(0.05)

    def _run_camera(self) -> None:
        from d405_camera import D405Camera

        with D405Camera(self.args.resolution, self.args.serial, logger=self.log) as cam:
            while not self._stop_evt.is_set():
                if self._snap_req.is_set():
                    try:
                        self._snap_out = ("ok", cam.capture_stack(self.args.frames))
                    except Exception as e:
                        self._snap_out = ("err", e)
                    self._snap_req.clear()
                    self._snap_done.set()
                    continue
                try:
                    frame = cam.capture(timeout_ms=2000)
                except Exception as e:
                    self._error = str(e)
                    self.log(f"[cam] 取帧失败：{e}")
                    break
                with self._lock:
                    self._latest = frame
                self._ready.set()


def startup_go_park(args) -> tuple:
    """先连电机并回避让位；连不上或没上电就不打开界面。

    连接会一直交给界面，识别/规划期间靠续帧顶住，避免反复断连导致坠落。
    """
    try:
        cfg = load_real_config()
    except RealConfigError as e:
        print(f"[real] 配置错误：{e}")
        return None, False, None
    print("[real] 启动：先确认电机在线，再退到避让位")
    arm = RealArm(
        cfg,
        dry_run=False,
        speed=max(1, min(int(args.speed), MAX_MOVE_SPEED)),
        port=args.port,
        logger=print,
        release_mode=args.release_mode,
        use_servo=args.servo,
    )
    try:
        arm.connect()
        if getattr(args, "no_park", False):
            print("[real] --no-park：电机已连接，跳过回避让位")
        else:
            arm.go_park()
            time.sleep(PARK_SETTLE_S)
            print("[real] 已在避让位，打开界面（连接保持，不关串口）")
        arm.hold_here("避让位")
        return cfg, not getattr(args, "no_park", False), arm
    except Exception as e:
        try:
            arm.close()
        except Exception:
            pass
        print(f"[real] 启动失败：{e}")
        print("[real] 电机没上电、串口被占用或急停时，不打开界面")
        return None, False, None


class RealDemoApp(tk.Tk):
    def __init__(self, args, cfg=None, parked: bool = False, arm=None) -> None:
        super().__init__()
        self.args = args
        self.title("工业指令智能体 · 真机")
        self._view_size = RESOLUTIONS.get(getattr(args, "resolution", CAMERA_RESOLUTION), (848, 480))
        vw, vh = self._view_size
        # 右侧画布按相机分辨率铺，不再被拉高留黑边
        self.geometry(f"{max(1040, vw + 470)}x{max(650, vh + 220)}")
        self.minsize(max(1040, vw + 380), max(650, vh + 180))
        self.configure(bg=BG)

        self._busy = False
        self._closing = False
        self._recording = False
        self._rec_chunks: queue.Queue = queue.Queue()
        self._rec_stream = None
        self._rec_started_at = 0.0
        self._rec_live_peak = 0
        self._rec_device_name = ""
        self._timer_job: str | None = None
        self._ui_events: queue.Queue = queue.Queue()
        self._preview_photo = None
        self._last_ann: dict | None = None
        self._arm: RealArm | None = arm
        self._parked = bool(parked)  # 动过真机就要在退出时送回零位
        self._holding = False  # 抓取成功后夹着件；放置或下一项抓取前才松
        self._agent_intent = "pick"
        self._stage_names = [
            "录音结束",
            "正在转写",
            "YOLO识别",
            "智能体规划",
            "轨迹规划",
            "机械臂执行",
        ]
        self._stage_index = -1
        self._stage_failed = False

        self.cfg = cfg
        self.w2c = None

        self._setup_style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.after(50, self._drain_ui_events)
        self.after(100, self._draw_progress)
        self.after(150, self._boot)

    # ------------------------------------------------------------------ 样式
    def _setup_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=INK, font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Card.TLabel", background=PANEL, foreground=INK)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure(
            "Title.TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 18, "bold")
        )
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Microsoft YaHei UI", 10))
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#0F1720",
            background=ACCENT,
            bordercolor=BORDER,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#FFFFFF",
            padding=(14, 8),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#2F4054")])
        style.configure(
            "Danger.TButton",
            background=DANGER,
            foreground="#FFFFFF",
            padding=(14, 8),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("Danger.TButton", background=[("active", "#A34B1F")])
        style.configure(
            "Card.TCheckbutton", background=PANEL, foreground=INK, focuscolor=PANEL
        )
        style.map("Card.TCheckbutton", background=[("active", PANEL)])
        style.configure("Card.TSpinbox", fieldbackground="#0F1720", foreground=INK)

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=(14, 6))
        ttk.Label(header, text="工业指令智能体 · 真机", style="Title.TLabel").pack(side="left")
        self.status = ttk.Label(header, text="", style="Sub.TLabel")
        self.status.pack(side="right", fill="x", expand=True, padx=(16, 0))

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=18, pady=(6, 10))
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left_card = tk.Frame(main, width=380, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        left_card.grid_propagate(False)
        left_card.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        left = ttk.Frame(left_card, style="Card.TFrame")
        left.pack(fill="both", expand=True, padx=14, pady=14)

        self.mic_btn = ttk.Button(left, text="开始录音", style="Accent.TButton", command=self.on_mic_toggle)
        self.mic_btn.pack(fill="x")
        self.rec_label = ttk.Label(left, text="说完后再次点击按钮结束识别", style="Muted.TLabel")
        self.rec_label.pack(anchor="w", pady=(10, 4))
        self.mic_meter = ttk.Progressbar(left, maximum=100, length=240, mode="determinate")
        self.mic_meter.pack(fill="x", pady=(0, 4))
        self.mic_level_label = ttk.Label(left, text="音量 0", style="Muted.TLabel")
        self.mic_level_label.pack(anchor="w", pady=(0, 6))
        ttk.Label(left, text="可参考语音", style="Muted.TLabel").pack(anchor="w", pady=(0, 2))
        ttk.Label(
            left, text=REFERENCE_VOICE, style="Card.TLabel", justify="left", wraplength=330
        ).pack(anchor="w", fill="x", pady=(0, 4))
        ttk.Label(
            left,
            text="也可在启动本界面的终端里打字，回车发送；q 退出",
            style="Muted.TLabel",
            justify="left",
            wraplength=330,
        ).pack(anchor="w", fill="x", pady=(0, 10))

        ctrl = ttk.Frame(left, style="Card.TFrame")
        ctrl.pack(fill="x", pady=(0, 10))
        self.execute_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl,
            text="允许真机执行（不勾只 dry-run）",
            variable=self.execute_var,
            style="Card.TCheckbutton",
        ).pack(anchor="w")
        speed_row = ttk.Frame(ctrl, style="Card.TFrame")
        speed_row.pack(fill="x", pady=(6, 0))
        ttk.Label(speed_row, text="速度", style="Muted.TLabel").pack(side="left")
        self.speed_var = tk.IntVar(value=min(self.args.speed, MAX_MOVE_SPEED))
        ttk.Spinbox(
            speed_row,
            from_=1,
            to=MAX_MOVE_SPEED,
            width=5,
            textvariable=self.speed_var,
            style="Card.TSpinbox",
        ).pack(side="left", padx=(8, 0))
        self.boxes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            speed_row, text="画面显示识别框", variable=self.boxes_var, style="Card.TCheckbutton"
        ).pack(side="right")
        ttk.Button(ctrl, text="急停", style="Danger.TButton", command=self.on_estop).pack(
            fill="x", pady=(8, 0)
        )

        ttk.Label(left, text="终端输出", style="Muted.TLabel").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            left,
            width=52,
            height=22,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            bg="#0F1720",
            fg=INK,
            insertbackground=INK,
            relief="flat",
            borderwidth=0,
        )
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        try:
            self.log.vbar.configure(
                bg="#172233",
                activebackground="#22324A",
                troughcolor="#0F1720",
                highlightbackground=BORDER,
                relief="flat",
                borderwidth=0,
                width=12,
            )
        except Exception:
            pass
        self.log.bind("<MouseWheel>", self._on_log_mousewheel)

        right_card = tk.Frame(main, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        right_card.grid(row=0, column=1, sticky="nw")
        right = ttk.Frame(right_card, style="Card.TFrame")
        right.pack(padx=14, pady=14)
        ttk.Label(right, text="现场画面", style="Muted.TLabel").pack(anchor="w")
        vw, vh = self._view_size
        self.view = tk.Canvas(
            right, width=vw, height=vh, bg="#111827", highlightthickness=0, relief="flat"
        )
        self.view.pack(pady=(8, 0))
        self.view.configure(width=vw, height=vh)

        bottom_card = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        bottom_card.pack(fill="x", padx=18, pady=(0, 16))
        bottom = ttk.Frame(bottom_card, style="Card.TFrame")
        bottom.pack(fill="x", padx=12, pady=5)
        self.stage_canvas = tk.Canvas(bottom, height=58, bg=PANEL, highlightthickness=0, xscrollincrement=20)
        self.stage_canvas.pack(fill="x", expand=True)
        self.stage_canvas.bind("<MouseWheel>", self._on_stage_mousewheel)
        self.stage_canvas.bind("<Configure>", lambda _e: self._draw_progress())

    # -------------------------------------------------------------- UI 小工具
    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _append_log(self, text: str) -> None:
        self.log.insert("end", str(text) + "\n")
        self.log.see("end")

    def log_async(self, text: str) -> None:
        """给后台线程用的日志入口，顺带回显到外部终端。"""
        print(text, flush=True)
        self._post_ui(lambda t=text: self._append_log(t))

    def _post_ui(self, callback) -> None:
        if not self._closing:
            self._ui_events.put(callback)

    def _drain_ui_events(self) -> None:
        if self._closing:
            return
        while True:
            try:
                callback = self._ui_events.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except tk.TclError:
                return
            except Exception as e:
                try:
                    self._append_log(f"[UI] 后台事件处理失败：{e}")
                except Exception:
                    pass
        self.after(50, self._drain_ui_events)

    def _on_log_mousewheel(self, event) -> str:
        self.log.yview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def _on_stage_mousewheel(self, event) -> str:
        self.stage_canvas.xview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def _set_stage(self, index: int, failed: bool = False) -> None:
        self._stage_index = index
        self._stage_failed = failed
        self._draw_progress()

    def _draw_progress(self) -> None:
        canvas = getattr(self, "stage_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 820)
        pad_x = 70
        usable = max(1, width - pad_x * 2)
        y = 18
        step = usable / max(1, len(self._stage_names) - 1)
        for i, name in enumerate(self._stage_names):
            x = pad_x + i * step
            if i > 0:
                prev_x = pad_x + (i - 1) * step
                canvas.create_line(
                    prev_x + 13, y, x - 13, y, fill=OK if i <= self._stage_index else BORDER, width=4
                )
            active = i <= self._stage_index
            bad = self._stage_failed and i == self._stage_index
            fill = DANGER if bad else (OK if active else "#172233")
            outline = DANGER if bad else (OK if active else BORDER)
            canvas.create_oval(x - 14, y - 14, x + 14, y + 14, fill=fill, outline=outline, width=2)
            canvas.create_text(
                x, y, text=str(i + 1), fill="#FFFFFF" if active else MUTED,
                font=("Microsoft YaHei UI", 10, "bold"),
            )
            canvas.create_text(x, y + 25, text=name, fill=INK, font=("Microsoft YaHei UI", 9))
        canvas.configure(scrollregion=(0, 0, width, 58))

    def _show_view_message(self, text: str) -> None:
        self.view.delete("all")
        self.view.create_text(
            max(20, self.view.winfo_width() // 2),
            max(20, self.view.winfo_height() // 2),
            text=text,
            fill=INK,
            font=("Microsoft YaHei UI", 15, "bold"),
        )

    def _ask_confirm(self, title: str, text: str) -> bool:
        """在 UI 线程弹确认框，后台线程同步等结果。"""
        done = threading.Event()
        box = {"ok": False}

        def ask() -> None:
            box["ok"] = bool(messagebox.askyesno(title, text))
            done.set()

        self._post_ui(ask)
        done.wait(120)
        return box["ok"]

    # ------------------------------------------------------------------ 启动
    def _boot(self) -> None:
        self._append_log(f"[API] {API_URL}")
        self._append_log(f"[API] key={(API_KEY or '')[:12]}…")
        self._show_view_message("相机启动中…")

        if self.cfg is None:
            try:
                self.cfg = load_real_config()
            except RealConfigError as e:
                self._append_log(f"[real] 配置错误：{e}")
                self._set_status("真机配置不可用，只能看画面")
        if self.cfg is not None:
            self._append_log(f"[real] robot.cfg = {self.cfg.cfg_path}")
            self._append_log(
                "[real] 软限位(°) "
                + " ".join(f"J{i + 1}[{lo:.0f},{hi:.0f}]" for i, (lo, hi) in enumerate(self.cfg.limits_deg()))
            )
            if self._parked:
                self._append_log("[real] 启动时已在避让位")
                self._set_status("已在避让位，可以下指令")

        try:
            self.w2c = load_handeye(self.args.handeye or None)
            self._append_log(f"[vision] 手眼 {describe_handeye(self.w2c)}")
        except VisionConfigError as e:
            self._append_log(f"[vision] 手眼标定不可用：{e}")

        self.cam = CameraWorker(self.args, self.log_async)
        self.cam.start()
        self.after(200, self._tick_preview)
        self._start_console_input()

    # ------------------------------------------------------------------ 画面
    def _tick_preview(self) -> None:
        if self._closing:
            return
        try:
            self._render_preview()
        except Exception as e:
            self._show_view_message(f"画面异常：{e}")
        self.after(int(1000 / PREVIEW_FPS), self._tick_preview)

    def _render_preview(self) -> None:
        frame = self.cam.latest()
        if frame is None:
            self._show_view_message(self.cam.error or "相机启动中…")
            return
        if Image is None or ImageTk is None:
            self._show_view_message("需要 Pillow 才能显示画面：pip install pillow")
            return

        img = frame.color
        if self.boxes_var.get() and self._last_ann:
            img = self._overlay_boxes(img, self._last_ann)

        src_w, src_h = int(img.shape[1]), int(img.shape[0])
        if (src_w, src_h) != self._view_size:
            self._view_size = (src_w, src_h)
            self.view.configure(width=src_w, height=src_h)
        pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        if (pil.width, pil.height) != self._view_size:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            pil = pil.resize(self._view_size, resample)
        self._preview_photo = ImageTk.PhotoImage(pil)
        self.view.delete("all")
        self.view.create_image(0, 0, image=self._preview_photo, anchor="nw")

    def _overlay_boxes(self, color: np.ndarray, ann: dict) -> np.ndarray:
        """把上一次识别的轮廓画到实时画面上。

        相机固定俯视、场景静止，所以这些框在后续帧里依然对得上；真正下发用的
        坐标仍然只来自抓帧那一刻，不是这里画的东西。
        """
        if int(ann.get("width") or 0) != color.shape[1] or int(ann.get("height") or 0) != color.shape[0]:
            return color
        try:
            import cv2
        except Exception:
            return color
        img = color.copy()
        for obj in ann.get("objects") or []:
            poly = obj.get("polygon") or []
            if len(poly) >= 3:
                cv2.polylines(img, [np.asarray(poly, dtype=np.int32)], True, (0, 255, 0), 2)
            bbox = obj.get("bbox") or [0, 0, 0, 0]
            p = obj.get("pose_6d") or {}
            cv2.putText(
                img,
                f"[{obj.get('id')}] {obj.get('yolo_class', '')} {obj.get('score', 0):.2f}",
                (int(bbox[0]), max(14, int(bbox[1]) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"({p.get('x', 0):+.3f},{p.get('y', 0):+.3f}) {obj.get('state', '')}",
                (int(bbox[0]), max(28, int(bbox[1]) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA,
            )
        return img

    # ------------------------------------------------------------------ 录音
    def on_mic_toggle(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._busy:
            messagebox.showinfo("提示", "正在跑任务，请稍等")
            return
        try:
            import sounddevice as sd
        except ImportError as e:
            messagebox.showerror("缺少依赖", f"麦克风需要：pip install sounddevice\n{e}")
            return

        self._rec_chunks = queue.Queue()
        self._rec_live_peak = 0
        self.mic_meter.configure(value=0)
        self.mic_level_label.configure(text="音量 0")
        try:
            info = sd.query_devices(kind="input")
            self._rec_device_name = str(info.get("name") or "默认输入")
        except Exception:
            self._rec_device_name = "默认输入"

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            self._rec_chunks.put(indata.copy())
            try:
                peak, _rms = audio_level_stats(indata)
                self._rec_live_peak = max(int(self._rec_live_peak * 0.75), peak)
            except Exception:
                pass

        try:
            self._rec_stream = sd.InputStream(
                samplerate=16000, channels=1, dtype="int16", callback=callback
            )
            self._rec_stream.start()
        except Exception as e:
            messagebox.showerror("麦克风失败", str(e))
            return

        self._recording = True
        self._rec_started_at = time.time()
        self._set_stage(-1)
        self.mic_btn.configure(text="结束并识别", style="Danger.TButton")
        self._set_status("录音中…说完点「结束并识别」")
        self.log.delete("1.0", "end")
        self._append_log(f"[MIC] 设备={self._rec_device_name}")
        self._tick_rec_timer()

    def _tick_rec_timer(self) -> None:
        if not self._recording:
            return
        sec = int(time.time() - self._rec_started_at)
        self.mic_meter.configure(value=min(100, int(self._rec_live_peak * 100 / 3000)))
        self.mic_level_label.configure(text=f"音量 {self._rec_live_peak}")
        self.rec_label.configure(text=f"录音中 {sec:02d}s")
        self._timer_job = self.after(200, self._tick_rec_timer)

    def _stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        if self._timer_job:
            self.after_cancel(self._timer_job)
            self._timer_job = None
        self.rec_label.configure(text="")
        self.mic_btn.configure(text="开始录音", style="Accent.TButton")
        self.mic_meter.configure(value=0)
        self.mic_level_label.configure(text="音量 0")

        stream = self._rec_stream
        self._rec_stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        chunks = []
        while not self._rec_chunks.empty():
            chunks.append(self._rec_chunks.get())
        if not chunks:
            messagebox.showwarning("提示", "没有录到声音，请重试")
            self._set_status("录音为空")
            return

        self._busy = True
        self._set_status("识别中…")
        self._set_stage(0)
        self._append_log("[MIC] 录音结束，正在转写…")
        threading.Thread(target=self._worker_from_mic, args=(chunks,), daemon=True).start()

    def _worker_from_mic(self, chunks: list) -> None:
        try:
            audio = np.concatenate(chunks, axis=0)
            peak, rms = audio_level_stats(audio)
            self.log_async(f"[MIC] 音量 peak={peak} rms={rms:.1f}")
            if peak < 200 or rms < 30:
                raise RuntimeError("麦克风录到的声音太小，请检查输入设备/音量后重试")
            pcm = np.squeeze(audio).astype("int16").tobytes()
            if len(pcm) < 16000:
                raise RuntimeError("录音太短，请多说一会儿再点结束")

            self._post_ui(lambda: self._set_stage(1))
            try:
                text = transcribe_pcm16(pcm, 16000)
            except CnSttError as e:
                raise RuntimeError(str(e)) from e

            text = text.strip()
            self.log_async(f"[STT] {text}（FunASR 国内本地）")
            self._pipeline(text)
        except Exception as e:
            self._post_ui(lambda msg=str(e): self._on_fail(msg))

    # ------------------------------------------------------------------ 外部终端打字
    def _start_console_input(self) -> None:
        """在启动 UI 的那个终端里收指令，和语音走同一条 _pipeline。"""
        if not getattr(sys.stdin, "isatty", lambda: False)():
            self._append_log("[TERM] 当前没有交互终端，只能用界面录音下指令")
            return
        t = threading.Thread(target=self._console_loop, daemon=True, name="console-cmd")
        t.start()
        self._append_log("[TERM] 启动本界面的终端可以打字下指令，回车发送，q 退出")

    def _console_prompt(self) -> None:
        if self._closing:
            return
        print("\n[real] 指令> ", end="", flush=True)

    def _console_loop(self) -> None:
        time.sleep(0.3)
        print(
            "\n[real] 终端指令已就绪。在这里打字回车，和界面录音走同一条链路。"
            " q 退出界面。",
            flush=True,
        )
        self._console_prompt()
        while not self._closing:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            text = line.strip()
            if not text:
                self._console_prompt()
                continue
            if text.lower() in {"q", "quit", "exit"}:
                print("[real] 收到退出，正在关界面…", flush=True)
                self._post_ui(self.on_close)
                break
            if not self._submit_text_cmd(text):
                self._console_prompt()

    def _submit_text_cmd(self, cmd: str) -> bool:
        if self._closing:
            return False
        if self._recording:
            print("[real] 正在录音，先在界面点结束，或等录音完再打字", flush=True)
            return False
        if self._busy:
            print("[real] 上一条还在跑，等完成后再输入", flush=True)
            return False
        self._busy = True

        def start() -> None:
            self._set_stage(-1)
            self.log.delete("1.0", "end")
            self._set_status("终端指令处理中…")
            self._append_log(f"[TERM] {cmd}")

        self._post_ui(start)
        threading.Thread(target=self._worker_from_text, args=(cmd,), daemon=True).start()
        return True

    def _worker_from_text(self, cmd: str) -> None:
        try:
            self._pipeline(cmd)
        except Exception as e:
            self._post_ui(lambda msg=str(e): self._on_fail(msg))

    # -------------------------------------------------------------- 主链路
    def _pipeline(self, cmd: str) -> None:
        """避让 → 识别 → 智能体 → 轨迹 → 执行。全程在后台线程，画面照常直播。"""
        try:
            guess = cmd_intent(cmd, holding=self._holding)

            if guess == "place" and self._holding:
                if self._last_ann is not None:
                    self._step_agent(cmd, self._last_ann)
                    intent = self._agent_intent or guess
                else:
                    intent = "place"
                if intent == "place":
                    self._step_place_only()
                    return

            if self._holding and guess != "place":
                self.log_async("[real] 手上还夹着上一件，先放到料框再执行新的抓取")
                self._step_place_only()

            self._step_park(open_jaw=not self._holding)
            ann, ann_path = self._step_vision()
            if not (ann.get("objects") or []):
                raise RuntimeError("没识别到可用零件，不往下走")

            task_path = self._step_agent(cmd, ann)
            intent = self._agent_intent or "pick"
            if intent == "place":
                self._step_place_only()
                return
            plan = self._step_plan(task_path)
            self._step_execute(plan, place_after=(intent == "pick_place"))
        except Exception as e:
            self._post_ui(lambda msg=str(e): self._on_fail(msg))
        else:
            self._post_ui(self._on_done)

    def _make_arm(self, *, dry_run: bool) -> RealArm:
        return RealArm(
            self.cfg,
            dry_run=dry_run,
            speed=int(self.speed_var.get()),
            port=self.args.port,
            logger=self.log_async,
            release_mode=self.args.release_mode,
            use_servo=self.args.servo,
        )

    def _ensure_live_arm(self) -> RealArm:
        """整段任务共用一条连接。反复 close/connect 会 motor_reset，臂坠一下再弹回。"""
        if self.cfg is None:
            raise RuntimeError("真机配置没加载成功，无法连接机械臂")
        live = self._arm
        if live is not None and live.is_connected:
            live.speed = int(self.speed_var.get())
            live.log = self.log_async
            live.hold_here("待命")
            return live
        if live is not None:
            try:
                live.close()
            except Exception:
                pass
            self._arm = None
        arm = self._make_arm(dry_run=False)
        arm.connect()
        arm.hold_here("刚连上")
        self._arm = arm
        return arm

    def _step_park(self, *, open_jaw: bool = True) -> None:
        """抓帧前把机械臂退出画面。

        每一轮都要做：执行完虽然停在避让位，但中间的轨迹会带着臂穿过画面，
        而且相机是固定俯视，臂只要入画就会被认成零件。
        手上还夹着件时必须 ``open_jaw=False``，否则会在避让位松手。
        """
        if self.args.no_park:
            return
        if not bool(self.execute_var.get()):
            self.log_async("[real] dry-run：真机不动；机械臂若在画面里会被误认成零件")
            return
        if self.cfg is None:
            raise RuntimeError("真机配置没加载成功，无法移动到避让位")

        self._post_ui(lambda: self._set_status("机械臂退到避让位…"))
        arm = self._ensure_live_arm()
        arm.go_park(open_jaw=open_jaw)
        arm.hold_here("避让位")
        self._parked = True
        time.sleep(PARK_SETTLE_S)   # 等直播吐出几帧不含机械臂的画面

    def _step_vision(self) -> tuple[dict, str]:
        self._post_ui(lambda: self._set_stage(2))
        self._post_ui(lambda: self._set_status("现场识别中…"))
        if self.w2c is None:
            raise RuntimeError("没有手眼标定，无法把像素换成世界坐标（先跑 hand_eye_calib.py）")

        payload = self.cam.grab_stable()
        vote_frames = None
        if isinstance(payload, (list, tuple)) and payload and hasattr(payload[0], "color"):
            from d405_camera import frames_to_median  # noqa: PLC0415

            vote_frames = list(payload)
            frame = frames_to_median(vote_frames)
            self.log_async(f"[vision] 已抓取 {len(vote_frames)} 帧，开始多帧投票")
        else:
            frame = payload
            self.log_async("[vision] 已抓取稳定帧，开始 YOLO 识别")
        ann = detect_to_annotations(
            frame,
            self.w2c,
            weights=self.args.weights or None,
            conf=self.args.conf,
            iou=self.args.iou,
            imgsz=self.args.imgsz,
            vote_frames=vote_frames,
            logger=self.log_async,
        )
        for obj in ann.get("objects") or []:
            p = obj["pose_6d"]
            self.log_async(
                f"  [{obj['id']}] {obj['class']} conf={obj.get('score', 0):.2f} "
                f"xy=({p['x']:+.4f},{p['y']:+.4f}) yaw={math.degrees(p['yaw']):+.1f}° {obj['state']}"
            )
        ann_path = save_annotations(ann, self.args.vision_root, self.args.sample_id, frame=frame)
        self.log_async(f"[vision] annotations -> {ann_path}")
        try:
            preview = save_preview(
                ann, frame, os.path.join(os.path.dirname(ann_path), "detect_preview.jpg")
            )
            self.log_async(f"[vision] 预览图 -> {preview}")
        except Exception as e:
            self.log_async(f"[vision] 预览图写入失败：{e}")

        self._last_ann = ann
        return ann, ann_path

    def _step_agent(self, cmd: str, ann: dict) -> str:
        self._post_ui(lambda: self._set_stage(3))
        self._post_ui(lambda: self._set_status("智能体规划中…"))
        vision = {
            "objects": ann.get("objects") or [],
            "target_container": {
                "name": self.args.destination,
                "xyz": [float(v) for v in self.args.dest],
            },
            "scene_layout": {
                "storage_box": {"pos_m": [float(v) for v in self.args.dest]},
            },
        }
        task = run_agent(cmd, vision, sample_id=self.args.sample_id, allow_fallback=True)
        task = rehydrate_tasks(task)

        source = str(task.get("source") or "")
        if source.startswith("dify_api"):
            label = "远程 Dify 智能体"
        elif source:
            label = "本地 fallback 智能体"
        else:
            label = "未知来源"
        self.log_async(f"[JSON_SOURCE] {label}")

        # 智能体 JSON 当底板；几何用同一帧 YOLO 覆写（贴桌 z=0、不建盒）。
        # 放不放到示教料框，只看意图，规划器仍走只抓。
        guess = cmd_intent(cmd, holding=self._holding)
        self._agent_intent = guess or agent_intent(task, cmd)
        self.log_async(
            f"[agent] 意图={self._agent_intent}"
            "（抓取=夹到避让位；放置=每件抓完去示教料框放下，再抓下一件）"
        )
        if self._agent_intent == "place":
            return ""

        picks = _agent_pick_names(task)
        obj_name = str(task.get("object") or "").strip()
        want_all = (not picks) or ("所有" in cmd) or obj_name in ("所有零件", "全部")
        targets = select_targets(ann, picks, want_all)
        patched = patch_agent_task_for_planner(task, targets, cmd)
        self.log_async(
            "[agent] 已按本帧视觉改写智能体几何："
            + "、".join(str(o.get("class")) for o in targets)
        )
        payload = {
            "scene": "standard",
            "scene_source": "real_d405",
            "agent_source": label,
            "user_cmd": cmd,
            "sample_id": self.args.sample_id,
            "task": patched,
        }
        task_path = os.path.abspath(self.args.task_out)
        os.makedirs(os.path.dirname(task_path), exist_ok=True)
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.log_async(f"[agent] 任务 JSON -> {task_path}")
        self.log_async(json.dumps(payload["task"], ensure_ascii=False, indent=2))
        return task_path

    def _step_plan(self, task_path: str) -> dict:
        if self.cfg is None:
            raise RuntimeError("真机配置没加载成功，无法做限位适配")
        self._post_ui(lambda: self._set_stage(4))
        self._post_ui(lambda: self._set_status("轨迹规划中…（这一步比较慢）"))

        traj_path = export_trajectory(
            task_path,
            self.args.traj_out,
            scene="standard",
            sample_id=self.args.sample_id,
            dataset_root=self.args.vision_root,
            joint_ranges_deg=None if self.args.relaxed_limits else self.cfg.limits_deg(),
            logger=self.log_async,
        )
        traj = load_sim_traj(traj_path)
        gmap = GripperMap.from_config(self.cfg)
        plan = adapt(traj, self.cfg, gmap, min_delta=math.radians(self.args.min_delta_deg))
        plan_path = save_plan(plan, self.args.plan_out)
        self.log_async(format_plan(plan))
        self.log_async(f"[real] plan 已保存：{plan_path}")
        if plan.get("errors"):
            raise RuntimeError("plan 有致命问题，未下发；先解决上面的 [ERROR]")
        return plan

    def _step_place_only(self) -> None:
        """智能体只要放置：送到示教料框立刻松爪，不再规划抓取。"""
        self._post_ui(lambda: self._set_stage(5))
        execute = bool(self.execute_var.get())
        if execute:
            if not self._ask_confirm(
                "真机执行确认",
                "即将送到示教放置位（料框）并松爪。\n确认现场安全后再继续。",
            ):
                self.log_async("[real] 已取消放置")
                return
        self._post_ui(lambda: self._set_status("放置中…" if execute else "dry-run 放置…"))
        self.log_async("[real] 放置 → 示教料框松爪（不停顿）")
        if not execute:
            self._holding = False
            self.log_async("[real] dry-run：不到放置位。勾选「允许真机执行」再来一次。")
            return
        arm = self._ensure_live_arm()
        try:
            release_at_place(arm, delay_s=0.0, park_after=not self.args.no_park)
            arm.hold_here("放置结束")
        except (RealArmError, RuntimeError) as e:
            raise RuntimeError(f"放置中止：{e}") from e
        self._holding = False
        if not self.args.no_park:
            self._parked = True

    def _step_execute(self, plan: dict, *, place_after: bool = False) -> None:
        self._post_ui(lambda: self._set_stage(5))
        execute = bool(self.execute_var.get())

        if execute:
            n_seg = len(plan.get("segments") or [])
            if place_after:
                tip = f"即将抓取 {n_seg} 个目标并送到示教料框松爪，速度 {self.speed_var.get()}。"
            else:
                tip = f"即将抓取 {n_seg} 个目标并夹到避让位（不放置），速度 {self.speed_var.get()}。"
            if not self._ask_confirm(
                "真机执行确认",
                tip + "\n确认现场安全、手放在急停上后再继续。",
            ):
                self.log_async("[real] 已取消真机执行，改为 dry-run")
                execute = False

        self._post_ui(lambda e=execute: self._set_status("真机执行中…" if e else "dry-run 中…"))
        if execute:
            arm = self._ensure_live_arm()
        else:
            arm = self._make_arm(dry_run=True)
        try:
            summary = execute_plan(
                plan,
                arm,
                go_home_first=self.args.home_first,
                go_home_after=False,
                park_after=not self.args.no_park,
                place_after=place_after,
                release_after=place_after,
                stop_on_fail=self.args.stop_on_fail,
            )
            if execute:
                arm.hold_here("任务结束")
        except (RealArmError, RuntimeError) as e:
            raise RuntimeError(f"执行中止：{e}") from e
        if execute and not self.args.no_park:
            self._parked = True
        if execute and summary["ok"] == summary["total"] and summary["total"] > 0:
            self._holding = not place_after

        if not execute:
            self.log_async("[real] 以上是 dry-run，没有下发任何指令。勾选「允许真机执行」再来一次。")
        if summary["ok"] != summary["total"]:
            raise RuntimeError(f"部分目标失败：成功 {summary['ok']}/{summary['total']}")

    # ------------------------------------------------------------------ 收尾
    def _on_done(self) -> None:
        self._busy = False
        self._set_status("完成")
        self._console_prompt()

    def _on_fail(self, err: str) -> None:
        self._busy = False
        self._append_log(f"[FAIL] {err}")
        self._set_stage(max(0, self._stage_index), failed=True)
        self._set_status("失败，可重新下指令")
        self._console_prompt()

    def on_estop(self) -> None:
        arm = self._arm
        if arm is None:
            self._append_log("[real] 当前没有正在执行的动作")
            return
        try:
            arm.estop()
            self._append_log("[real] 已发送急停")
        except Exception as e:
            self._append_log(f"[real] 急停失败：{e}")

    def on_close(self) -> None:
        self._closing = True
        if self._recording:
            try:
                self._stop_recording()
            except Exception:
                pass
        arm = self._arm
        self._arm = None
        if arm is not None and arm.is_connected:
            if self._busy:
                # 有动作在跑，先急停；急停之后位形不可信，不能再自动走位
                try:
                    arm.estop()
                except Exception:
                    pass
                try:
                    arm.close()
                except Exception:
                    pass
            else:
                self._set_status("退出回零位…")
                self.update_idletasks()
                try:
                    arm.log = print
                    arm.go_zero(open_jaw=True)
                except Exception as e:
                    print(f"[real] 退出回零位失败：{e}")
                try:
                    arm.close()
                except Exception:
                    pass
        elif self.cfg is not None and (
            self._parked or bool(self.execute_var.get())
        ):
            self._set_status("退出回零位…")
            self.update_idletasks()
            try:
                with RealArm(
                    self.cfg,
                    dry_run=False,
                    speed=int(self.speed_var.get()),
                    port=self.args.port,
                    logger=print,
                    release_mode=self.args.release_mode,
                    use_servo=self.args.servo,
                ) as home_arm:
                    home_arm.go_zero(open_jaw=True)
            except Exception as e:
                print(f"[real] 退出回零位失败：{e}")

        cam = getattr(self, "cam", None)
        if cam is not None:
            cam.stop()
            cam.join(timeout=2)
        self.destroy()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D405 实时画面 + 语音指令 → 智能体 → fafu 真机")

    vis = p.add_argument_group("视觉")
    vis.add_argument("--frame-dir", default="", help="用存下的帧回放，不连相机（离线联调）")
    vis.add_argument("--resolution", default=CAMERA_RESOLUTION, choices=("mid", "high"))
    vis.add_argument("--serial", default=CAMERA_SERIAL)
    vis.add_argument("--frames", type=int, default=SNAPSHOT_FRAMES, help="抓帧时连拍取中位数的帧数")
    vis.add_argument("--weights", default="", help="YOLO 权重；默认用本目录的")
    vis.add_argument("--handeye", default="", help="手眼 JSON；默认 handeye_d405.json")
    vis.add_argument("--conf", type=float, default=YOLO_CONF)
    vis.add_argument("--iou", type=float, default=YOLO_IOU)
    vis.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    vis.add_argument("--sample-id", default=SAMPLE_ID)
    vis.add_argument("--vision-root", default=VISION_DIR)

    tsk = p.add_argument_group("任务")
    tsk.add_argument(
        "--dest", type=float, nargs=3, default=list(DEFAULT_DEST_XYZ),
        metavar=("X", "Y", "Z"), help="放置点世界坐标（米）",
    )
    tsk.add_argument("--destination", default=DEFAULT_DESTINATION)

    out = p.add_argument_group("产物")
    out.add_argument("--task-out", default=os.path.join(OUTPUT_DIR, "task_from_agent.json"))
    out.add_argument("--traj-out", default=os.path.join(OUTPUT_DIR, "traj_agent.json"))
    out.add_argument("--plan-out", default=os.path.join(OUTPUT_DIR, "plan_agent.json"))

    run = p.add_argument_group("执行")
    run.add_argument("--speed", type=int, default=DEFAULT_MOVE_SPEED)
    run.add_argument("--port", default=None, help="串口，默认按 robot.cfg")
    run.add_argument("--min-delta-deg", type=float, default=1.0)
    run.add_argument("--relaxed-limits", action="store_true",
                     help="用仿真放宽后的关节范围规划（结果通常不能下发真机）")
    run.add_argument(
        "--home-first", dest="home_first", action=argparse.BooleanOptionalAction,
        default=not START_FROM_PARK,
        help="抓取前先回仿真 HOME。默认从避让位直接接",
    )
    run.add_argument(
        "--no-park",
        action="store_true",
        help="不使用避让位（机械臂会留在画面里，YOLO 会把它当成零件）",
    )
    run.add_argument("--stop-on-fail", action="store_true")
    run.add_argument(
        "--servo", dest="servo", action=argparse.BooleanOptionalAction,
        default=USE_SERVO_J,
        help="用 servo_j 在线流式喂帧（动作最连贯）；--no-servo 退回 move_j",
    )
    run.add_argument(
        "--release-mode", choices=RELEASE_MODES, default=RELEASE_MODE_ON_CLOSE,
        help="断开时的刹车档：stop=完全松开(会沉) / brake=阻尼保持(不耗电) / "
             f"hold=主动顶住(耗电发热)；默认 {RELEASE_MODE_ON_CLOSE}",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg, parked, arm = startup_go_park(args)
    if cfg is None:
        return 1
    app = RealDemoApp(args, cfg=cfg, parked=parked, arm=arm)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
