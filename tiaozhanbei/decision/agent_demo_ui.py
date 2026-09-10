"""
可视化 Demo（决策模块 / 整项目主入口）。

  文字 / 快捷按钮 / 开始·结束录音 → 调 Dify → 显示任务 → 写出 JSON

用法：
  python agent_demo_ui.py

本机 Dify：只改下面 DIFY_API_URL / DIFY_API_KEY。
浏览器打开的 /workflow/xxx 是编辑页，不要填进来。
API 固定是 /v1/workflows/run；Key 在该工作流「访问 API」里复制。
"""

from __future__ import annotations

import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from PIL import Image, ImageTk
except ImportError:  # 右侧预览降级为文字，不影响录音/仿真主流程
    Image = None
    ImageTk = None

# ========== Dify（本界面入口以这里为准）==========
# 网页上的「Web 应用 URL」是 http://localhost/workflow/xxx ，不要贴进来。
# 本机 Docker 把 8080 映射到容器 80，API 必须带 :8080
DIFY_API_URL = "http://localhost:8080/v1/workflows/run"
# 在该工作流页面：发布 → 访问 API → 创建密钥，把 app- 开头的一整行贴到这里
DIFY_API_KEY = "app-pcJzT4Jht70CXyFoWGRp8ts9"
os.environ["DIFY_API_URL"] = DIFY_API_URL
os.environ["DIFY_API_KEY"] = DIFY_API_KEY
# ================================================

from agent.cn_stt import CnSttError, transcribe_pcm16  # noqa: E402
from agent.demo_voice_agent import (  # noqa: E402
    DEFAULT_OUT,
    audio_level_stats,
    load_simple_samples,
    load_vision_by_sample,
    rehydrate_tasks,
    run_agent,
    speech_to_text_from_file,
)
from agent.run_batch import API_KEY, API_URL  # noqa: E402

ROOT = Path(__file__).resolve().parent
EC_SAMPLE_ID = "0502"  # 原 ec/0001 喇叭头螺丝+料盘，已并入 dataset_learn
EC_SAMPLE_DIR = ROOT.parent / "dataset_learn" / EC_SAMPLE_ID
EC_TASK_TEMPLATE = ROOT.parent / "sim" / "ec" / "tasks" / "test03.json"
EC_TASK_OUT = ROOT.parent / "sim" / "ec" / "tasks" / "delivery_ec_auto.json"
EC_SCRIPT = ROOT.parent / "sim" / "ec" / "run_ec_bin_sim.py"
EC_CACHED_TRAJ = ROOT.parent / "sim" / "ec" / "traj" / "ec_all12_traj.json"
STD_SCRIPT = ROOT.parent / "sim" / "run_agent_pick_place_side_sim.py"
YOLO_DIR = ROOT.parent / "yolo" / "仿真"
YOLO_WEIGHTS = YOLO_DIR / "best.pt"
YOLO_INFER_SCRIPT = YOLO_DIR / "infer_seg_annotations.py"
YOLO_OUTPUTS = YOLO_DIR / "outputs"
YOLO_SIM_DATASET = YOLO_DIR / "sim_dataset"
DATASET_LEARN = ROOT.parent / "dataset_learn"
FORCE_START_SAMPLE = "0501"  # 标准场景样本号；留空则从高成功率样本池随机
FORCE_START_SCENE = "ec_bin"  # "ec_bin" 只用 EC 场景/仿真；改回 "" 即显示标准样本
USE_HIGH_SUCCESS_SAMPLE_POOL = True  # False 则从全部 dataset_learn 样本随机
USE_LIVE_START_SCENE = True  # True: UI 启动后右侧直接贴 Panda3D HOME 场景窗口
USE_SIM_WORKER = True  # True: 标准仿真优先投递到常驻 worker，失败再回退单次启动
HIGH_SUCCESS_SAMPLE_IDS = (
    "0501",  # 固定成功：三件全部可连续 pick-place
    "0002",  # 已验证：随机两件连续 pick-place 稳定
    "0001",  # 用户指定加入：螺丝刀场景
)
HIGH_SUCCESS_HINTS = {
    "0501": "把所有零件放进收纳筐",
    "0002": "随便拿取两个零件放进收纳筐",
    "0001": "把螺丝刀放进收纳筐",
}

QUICK_TASKS = [
    ("拿取并放到收纳盒", "拿取轴承-凸轮滚轮放到收纳盒"),
    ("只拿取", "给我一个轴承-凸轮滚轮"),
    ("放到第3格", "取出轴承-凸轮滚轮放到料箱的第3个格子中"),
    ("螺丝对应入格", "把桌面上的喇叭头螺丝依次放进料盘对应格子"),
    ("最多区域装箱", "帮我把零件最多的区域装箱"),
    ("搬运料箱", "帮我搬运一下料箱盒"),
    ("所有零件装箱", "帮我把所有零件装箱"),
]

# 深色低对比：整体接近右侧显示区，减少板块间跳色
BG = "#101722"
PANEL = "#121C29"
INK = "#D7E2EC"
MUTED = "#8EA0B5"
ACCENT = "#1F6F8B"
ACCENT_HOVER = "#245F77"
OK = "#2A9D8F"
DANGER = "#A8553A"
BORDER = "#263447"


class AgentDemoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("工业指令智能体 · Demo")
        self.geometry("1360x780")
        self.minsize(1040, 650)
        self.configure(bg=BG)
        self._wait_home_scene_before_show = bool(USE_LIVE_START_SCENE and os.name == "nt")
        if self._wait_home_scene_before_show:
            self.withdraw()

        self._busy = False
        self._recording = False
        self._rec_chunks: queue.Queue = queue.Queue()
        self._rec_stream = None
        self._rec_started_at = 0.0
        self._rec_rate = 16000
        self._rec_device_name = ""
        self._rec_live_peak = 0
        self._rec_live_rms = 0.0
        self._last_delivery: dict | None = None
        self._timer_job: str | None = None
        self._sim_proc: subprocess.Popen | None = None
        self._sim_worker_proc: subprocess.Popen | None = None
        self._sim_worker_dir = ROOT / "tasks" / "sim_worker_requests"
        self._sim_worker_ready_file = ROOT / "tasks" / "sim_worker_ready.flag"
        self._sim_worker_ready = False
        self._worker_sim_active = False
        self._preview_proc: subprocess.Popen | None = None
        self._preview_hwnd: int | None = None
        self._preview_ready_file: Path | None = None
        self._preview_kind: str | None = None
        self._sim_queue: list[tuple[Path, str, str, str | None, int, int]] = []
        self._panda_hwnd: int | None = None
        self._panda_geometry: tuple[int, int, int, int] | None = None
        self._panda_deferred = False
        self._panda_ready_to_show = False
        self._panda_reposition_job: str | None = None
        self._sim_start_file: Path | None = None
        self._closing = False
        self._ui_events: queue.Queue = queue.Queue()
        self._scene_photo = None
        self._stage_names = ["录音结束", "正在转写", "YOLO识别", "仿真规划", "启动仿真"]
        self._stage_index = -1
        self._stage_failed = False
        self._sample_ids = self._load_sample_ids()
        self._reliable_sample_ids = self._load_reliable_sample_ids()
        self._startup_sample_pool = (
            self._reliable_sample_ids
            if USE_HIGH_SUCCESS_SAMPLE_POOL
            else self._sample_ids
        )
        self._current_sample_id = (
            FORCE_START_SAMPLE
            if FORCE_START_SAMPLE and FORCE_START_SAMPLE in self._sample_ids
            else random.choice(self._startup_sample_pool)
            if self._startup_sample_pool
            else "0001"
        )
        self._samples = self._load_samples_safe()
        self._class_to_samples = self._build_class_index(self._samples)
        self._known_classes = self._load_known_classes()

        self._setup_style()
        self._build()
        self.bind("<Configure>", self._on_root_configure)
        self._set_status(f"API 已就绪 · {API_KEY[:12]}… · {API_URL}")
        self.after(50, self._drain_ui_events)
        self.after(100, self._draw_progress)
        self.after(120, self._start_sim_worker)
        self.after(150, self._show_random_start_scene)
        self.after(400, self._tick_panda_follow)

    def _load_sample_ids(self) -> list[str]:
        ids: list[str] = []
        dataset_root = ROOT.parent / "dataset_learn"
        if dataset_root.exists():
            ids = [
                p.name
                for p in dataset_root.iterdir()
                if p.is_dir() and (p / "annotations.json").exists()
            ]
        if not ids:
            try:
                ids = [str(item.get("sample_id")) for item in load_simple_samples()]
            except Exception:
                ids = []
        ids = sorted({sid for sid in ids if sid})
        return ids or ["0001"]

    def _load_reliable_sample_ids(self) -> list[str]:
        existing = set(self._sample_ids)
        ids = [sid for sid in HIGH_SUCCESS_SAMPLE_IDS if sid in existing]
        return ids or self._sample_ids

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=INK, font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabelframe", background=PANEL, foreground=INK)
        style.configure("TLabelframe.Label", background=PANEL, foreground=MUTED, font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Card.TLabel", background=PANEL, foreground=INK)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Title.TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Microsoft YaHei UI", 10))
        style.configure("TEntry", fieldbackground="#0F1720", foreground=INK)
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
            "Ghost.TButton",
            background=PANEL,
            foreground=INK,
            padding=(12, 8),
        )
        style.map("Ghost.TButton", background=[("active", BORDER)])
        style.configure(
            "Danger.TButton",
            background=DANGER,
            foreground="#FFFFFF",
            padding=(14, 8),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("Danger.TButton", background=[("active", "#A34B1F")])
        style.configure(
            "Chip.TButton",
            background="#172233",
            foreground=INK,
            padding=(10, 10),
            font=("Microsoft YaHei UI", 9),
        )
        style.map("Chip.TButton", background=[("active", "#1D2B3F")])

    def _build(self) -> None:
        self.cmd_var = tk.StringVar(value="")
        self.sample_var = tk.StringVar(value="auto")
        self.sample_preview_var = tk.StringVar(value="自动选场景：等待语音指令")

        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=(14, 6))
        ttk.Label(header, text="工业指令智能体", style="Title.TLabel").pack(side="left")
        self.status = ttk.Label(header, text="", style="Sub.TLabel")
        self.status.pack(side="right", fill="x", expand=True, padx=(16, 0))

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=18, pady=(6, 10))
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left_card = tk.Frame(
            main,
            width=360,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        left_card.grid_propagate(False)
        left_card.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        left = ttk.Frame(left_card, style="Card.TFrame")
        left.pack(fill="both", expand=True, padx=14, pady=14)

        self.mic_btn = ttk.Button(
            left,
            text="开始录音",
            style="Accent.TButton",
            command=self.on_mic_toggle,
        )
        self.mic_btn.pack(fill="x")
        self.rec_label = ttk.Label(left, text="说完后再次点击按钮结束识别", style="Muted.TLabel")
        self.rec_label.pack(anchor="w", pady=(10, 4))
        self.mic_meter = ttk.Progressbar(left, maximum=100, length=240, mode="determinate")
        self.mic_meter.pack(fill="x", pady=(0, 4))
        self.mic_level_label = ttk.Label(left, text="音量 0", style="Muted.TLabel")
        self.mic_level_label.pack(anchor="w", pady=(0, 6))
        ttk.Label(left, text="可参考语音", style="Muted.TLabel").pack(anchor="w", pady=(0, 2))
        self.reference_voice_label = ttk.Label(
            left,
            text="场景加载后显示",
            style="Card.TLabel",
            justify="left",
            wraplength=320,
        )
        self.reference_voice_label.pack(anchor="w", fill="x", pady=(0, 12))

        ttk.Label(left, text="终端输出", style="Muted.TLabel").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            left,
            width=52,
            height=28,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            bg="#0F1720",
            fg="#D7E2EC",
            insertbackground="#D7E2EC",
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
        right_card.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(right_card, style="Card.TFrame")
        right.pack(fill="both", expand=True, padx=14, pady=14)
        ttk.Label(right, text="识别场景", style="Muted.TLabel").pack(anchor="w")
        self.scene_canvas = tk.Canvas(
            right,
            bg="#111827",
            highlightthickness=0,
            relief="flat",
        )
        self.scene_canvas.pack(fill="both", expand=True, pady=(8, 8))
        self.scene_canvas.bind("<Configure>", lambda _e: self._redraw_scene_image())

        bottom_card = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        bottom_card.pack(fill="x", padx=18, pady=(0, 16))
        bottom = ttk.Frame(bottom_card, style="Card.TFrame")
        bottom.pack(fill="x", padx=12, pady=5)
        self.stage_canvas = tk.Canvas(
            bottom,
            height=58,
            bg=PANEL,
            highlightthickness=0,
            xscrollincrement=20,
        )
        self.stage_canvas.pack(fill="x", expand=True)
        self.stage_canvas.bind("<MouseWheel>", self._on_stage_mousewheel)
        self.stage_canvas.bind("<Configure>", lambda _e: self._draw_progress())

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

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

    def _load_samples_safe(self) -> list[dict]:
        try:
            return load_simple_samples()
        except Exception as e:
            self._post_ui(lambda msg=str(e): self._append_log(f"[DATASET] simple_dataset 读取失败：{msg}"))
            return []

    def _load_known_classes(self) -> set[str]:
        classes = set(self._class_to_samples.keys())
        dataset_root = ROOT.parent / "dataset_learn"
        if not dataset_root.exists():
            return classes
        for ann_path in dataset_root.glob("*/annotations.json"):
            try:
                ann = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for obj in ann.get("objects") or []:
                cls = str(obj.get("class") or "").strip()
                if cls:
                    classes.add(cls)
        return classes

    def _build_class_index(self, samples: list[dict]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for item in samples:
            sid = str(item.get("sample_id") or "").strip()
            if not sid:
                continue
            objects = ((item.get("vision_objects") or {}).get("objects") or [])
            for obj in objects:
                cls = str(obj.get("class") or "").strip()
                if not cls:
                    continue
                bucket = index.setdefault(cls, [])
                if sid not in bucket:
                    bucket.append(sid)
        return index

    def _compact_text(self, text: str) -> str:
        return re.sub(r"[\s_\-·/\\（）()，。,.、：:；;]+", "", str(text or ""))

    def _class_tokens(self, cls: str) -> list[str]:
        parts = [p for p in re.split(r"[-_/\\（）()·\s]+", cls) if p]
        compact = self._compact_text(cls)
        tokens = [compact, *parts]
        # 类名后半段通常是用户会说的短名，如“机器工具-螺丝刀”里的“螺丝刀”。
        if "-" in cls:
            tokens.append(cls.split("-")[-1])
        seen: set[str] = set()
        out: list[str] = []
        for token in tokens:
            token = self._compact_text(token)
            if len(token) >= 2 and token not in seen:
                seen.add(token)
                out.append(token)
        return out

    def _select_sample_for_cmd(self, cmd: str) -> tuple[str, str, str]:
        text = self._compact_text(cmd)
        best: tuple[int, str, str] | None = None
        for cls, sids in self._class_to_samples.items():
            score = 0
            compact_cls = self._compact_text(cls)
            if compact_cls and compact_cls in text:
                score = 1000 + len(compact_cls)
            else:
                for token in self._class_tokens(cls):
                    if token in text:
                        score = max(score, 100 + len(token))
            if score <= 0:
                continue
            sid = sorted(sids, key=lambda x: int(x) if x.isdigit() else 999999)[0]
            if best is None or score > best[0]:
                best = (score, sid, cls)
        if best is not None:
            return best[1], best[2], "按零件名自动匹配"
        fallback = self._sample_ids[0] if self._sample_ids else "0001"
        return fallback, "", "未识别到明确零件名，使用默认场景"

    def _object_tokens_from_classes(self, classes: list[str] | set[str]) -> set[str]:
        tokens: set[str] = set()
        for cls in classes:
            for token in self._class_tokens(cls):
                if token:
                    tokens.add(token)
        return tokens

    def _missing_object_tokens(self, cmd: str, ann: dict) -> list[str]:
        text = self._compact_text(cmd)
        if not text:
            return []

        present_classes = {
            str(obj.get("class") or "").strip()
            for obj in (ann.get("objects") or [])
            if str(obj.get("class") or "").strip()
        }
        present_tokens = self._object_tokens_from_classes(present_classes)

        mentioned_tokens: set[str] = set()
        for cls in self._known_classes:
            for token in self._class_tokens(cls):
                if token and token in text:
                    mentioned_tokens.add(token)

        missing = sorted(
            {token for token in mentioned_tokens if token not in present_tokens},
            key=lambda item: (-len(item), item),
        )
        return missing

    def _set_stage(self, index: int, failed: bool = False) -> None:
        self._stage_index = index
        self._stage_failed = failed
        self._draw_progress()

    def _draw_progress(self) -> None:
        if not hasattr(self, "stage_canvas"):
            return
        canvas = self.stage_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 760)
        pad_x = 70
        usable = max(1, width - pad_x * 2)
        y = 18
        step = usable / max(1, len(self._stage_names) - 1)
        for i, name in enumerate(self._stage_names):
            x = pad_x + i * step
            if i > 0:
                prev_x = pad_x + (i - 1) * step
                line_color = OK if i <= self._stage_index else BORDER
                canvas.create_line(prev_x + 13, y, x - 13, y, fill=line_color, width=4)
            active = i <= self._stage_index
            fill = DANGER if self._stage_failed and i == self._stage_index else (OK if active else "#172233")
            outline = DANGER if self._stage_failed and i == self._stage_index else (OK if active else BORDER)
            text_color = "#FFFFFF" if active else MUTED
            canvas.create_oval(x - 14, y - 14, x + 14, y + 14, fill=fill, outline=outline, width=2)
            canvas.create_text(x, y, text=str(i + 1), fill=text_color, font=("Microsoft YaHei UI", 10, "bold"))
            canvas.create_text(x, y + 25, text=name, fill=INK, font=("Microsoft YaHei UI", 9))
        canvas.configure(scrollregion=(0, 0, width, 58))

    def _on_stage_mousewheel(self, event) -> str:
        self.stage_canvas.xview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def _on_log_mousewheel(self, event) -> str:
        self.log.yview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def _scene_image_candidates(self, sid: str) -> list[Path]:
        sample_dir = ROOT.parent / "dataset_learn" / sid
        return [
            sample_dir / "sim_home_preview.jpg",
            sample_dir / "sim_home_preview.png",
            sample_dir / "sim_preview.jpg",
            sample_dir / "sim_preview.png",
            sample_dir / f"{sid}.png",
            sample_dir / f"{sid}.jpg",
            sample_dir / "segmentation.png",
            sample_dir / "depth_preview.png",
        ]

    def _show_scene_for_sample(self, sid: str, title: str = "") -> None:
        for path in self._scene_image_candidates(sid):
            if path.exists():
                self._show_image_path(path)
                self._append_log(f"[SCENE] 右侧显示 {path}")
                return
        self._scene_image_path = None
        self.scene_canvas.delete("all")
        self.scene_canvas.create_text(
            max(20, self.scene_canvas.winfo_width() // 2),
            max(20, self.scene_canvas.winfo_height() // 2),
            text=title or f"样本 {sid} 暂无预览图",
            fill="#D7E2EC",
            font=("Microsoft YaHei UI", 16, "bold"),
        )

    def _show_raw_for_sample(self, sid: str) -> None:
        self._show_scene_for_sample(sid, f"样本 {sid} 暂无原始预览图")

    def _home_preview_path(self, sid: str) -> Path:
        return ROOT.parent / "dataset_learn" / str(sid) / "sim_home_preview.jpg"

    def _ensure_home_preview_for_sample(self, sid: str) -> None:
        out_path = self._home_preview_path(sid)
        if out_path.exists():
            self._show_raw_for_sample(sid)
            return

        self._show_scene_message(f"正在生成仿真 HOME 初始场景…\nsample={sid}")
        self._append_log(f"[SCENE] 生成仿真 HOME 初始图 sample={sid}")

        def worker() -> None:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(STD_SCRIPT),
                        "--preview-only",
                        "--sample-id",
                        str(sid),
                        "--dataset-root",
                        str((ROOT.parent / "dataset_learn").resolve()),
                        "--preview-out",
                        str(out_path.resolve()),
                    ],
                    cwd=str(ROOT.parent.parent),
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=90,
                )
                if proc.stdout:
                    for line in proc.stdout.splitlines():
                        if "[preview]" in line:
                            self._post_ui(lambda t=line: self._append_log(f"[SCENE] {t}"))
                if proc.returncode != 0:
                    self._post_ui(lambda: self._append_log("[SCENE] 仿真 HOME 初始图生成失败，回退数据集图片"))
            except Exception as e:
                self._post_ui(lambda msg=str(e): self._append_log(f"[SCENE] 仿真 HOME 初始图生成异常，回退数据集图片：{msg}"))
            self._post_ui(lambda s=sid: self._show_raw_for_sample(s))

        threading.Thread(target=worker, daemon=True).start()

    def _show_random_start_scene(self) -> None:
        if FORCE_START_SCENE == "ec_bin":
            self._show_ec_start_scene()
            return
        sid = self._current_sample_id
        self._set_status(f"随机场景已就绪 · sample={sid}")
        pool_name = "高成功率池" if USE_HIGH_SUCCESS_SAMPLE_POOL else "全部数据集"
        self._append_log(f"[SCENE] 启动随机场景 sample={sid} pool={pool_name}")
        self._set_reference_voice_for_sample(sid)
        if USE_LIVE_START_SCENE and os.name == "nt":
            self._launch_live_start_scene(sid)
        else:
            self._show_scene_message("等待 Panda HOME 初始场景…")

    def _launch_live_start_scene(self, sid: str) -> None:
        if self._live_start_scene_alive(f"std:{sid}"):
            return
        self._stop_live_start_scene()
        if not STD_SCRIPT.exists():
            self._show_scene_message("Panda HOME 初始场景不可用")
            return
        preview_dataset_root = DATASET_LEARN
        cmd = [
            sys.executable,
            str(STD_SCRIPT),
            "--preview-window-only",
            "--sample-id",
            sid,
            "--dataset-root",
            str(preview_dataset_root.resolve()),
        ]
        ready_file = ROOT / "tasks" / f"home_preview_ready_{int(time.time() * 1000)}_{os.getpid()}.flag"
        try:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.unlink(missing_ok=True)
        except Exception:
            pass
        cmd.extend(["--ready-file", str(ready_file.resolve())])
        sim_env = os.environ.copy()
        sim_env["PYTHONIOENCODING"] = "utf-8"
        sim_env["PYTHONUTF8"] = "1"
        try:
            self._preview_proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT.parent.parent),
                env=sim_env,
                stdout=None,
                stderr=None,
            )
        except Exception as e:
            self._append_log(f"[SCENE] Panda HOME 场景启动失败：{e}")
            self._show_scene_message("Panda HOME 初始场景启动失败")
            if self._wait_home_scene_before_show:
                self.deiconify()
            return
        self._preview_ready_file = ready_file
        self._preview_kind = f"std:{sid}"
        self.after(200, lambda p=self._preview_proc, r=ready_file: self._poll_live_start_ready(p, r))

    def _show_ec_start_scene(self) -> None:
        if USE_LIVE_START_SCENE and os.name == "nt":
            self._launch_ec_live_start_scene()
            return
        ec_dir = EC_SAMPLE_DIR
        for path in (
            ec_dir / "0001.png",
            ec_dir / "0001.jpg",
            ec_dir / "segmentation.png",
            ec_dir / "depth_preview.png",
        ):
            if path.exists():
                self._set_status(f"EC 场景已就绪 · sample={EC_SAMPLE_ID}")
                self._append_log(f"[SCENE] 启动 EC 场景 sample={EC_SAMPLE_ID} 图={path}")
                if hasattr(self, "reference_voice_label"):
                    self.reference_voice_label.configure(text="把前三个螺丝放到对应格子里")
                self._show_image_path(path)
                return
        self._show_scene_message("EC 场景暂无预览图")

    def _live_start_scene_alive(self, kind: str | None = None) -> bool:
        proc = self._preview_proc
        if proc is None or proc.poll() is not None:
            return False
        return kind is None or self._preview_kind == kind

    def _launch_ec_live_start_scene(self) -> None:
        if self._live_start_scene_alive("ec"):
            return
        self._stop_live_start_scene()
        if not EC_SCRIPT.exists():
            self._show_scene_message("EC Panda HOME 初始场景不可用")
            if self._wait_home_scene_before_show:
                self.deiconify()
            return
        ec_dir = EC_SAMPLE_DIR
        ready_file = ROOT / "tasks" / f"ec_home_preview_ready_{int(time.time() * 1000)}_{os.getpid()}.flag"
        try:
            ready_file.parent.mkdir(parents=True, exist_ok=True)
            ready_file.unlink(missing_ok=True)
        except Exception:
            pass
        cmd = [
            sys.executable,
            str(EC_SCRIPT),
            "--preview-window-only",
            "--sample-dir",
            str(ec_dir.resolve()),
            "--ready-file",
            str(ready_file.resolve()),
        ]
        sim_env = os.environ.copy()
        sim_env["PYTHONIOENCODING"] = "utf-8"
        sim_env["PYTHONUTF8"] = "1"
        try:
            self._preview_proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT.parent.parent),
                env=sim_env,
                stdout=None,
                stderr=None,
            )
        except Exception as e:
            self._append_log(f"[SCENE] EC Panda HOME 场景启动失败：{e}")
            self._show_scene_message("EC Panda HOME 初始场景启动失败")
            if self._wait_home_scene_before_show:
                self.deiconify()
            return
        self._preview_ready_file = ready_file
        self._preview_kind = "ec"
        self._set_status("EC Panda HOME 初始场景启动中…")
        if hasattr(self, "reference_voice_label"):
            self.reference_voice_label.configure(text="把前三个螺丝放到对应格子里")
        self.after(200, lambda p=self._preview_proc, r=ready_file: self._poll_live_start_ready(p, r))

    def _show_image_path(self, path: Path) -> None:
        self._scene_image_path = path
        self._redraw_scene_image()

    def _show_scene_message(self, text: str) -> None:
        self._scene_image_path = None
        self.scene_canvas.delete("all")
        self.scene_canvas.create_text(
            max(20, self.scene_canvas.winfo_width() // 2),
            max(20, self.scene_canvas.winfo_height() // 2),
            text=text,
            fill="#D7E2EC",
            font=("Microsoft YaHei UI", 16, "bold"),
        )

    def _clear_scene_display(self) -> None:
        self._scene_image_path = None
        self._scene_photo = None
        self.scene_canvas.delete("all")

    def _redraw_scene_image(self) -> None:
        if Image is None or ImageTk is None:
            return
        path = getattr(self, "_scene_image_path", None)
        if not path:
            return
        canvas = self.scene_canvas
        w = max(80, canvas.winfo_width())
        h = max(80, canvas.winfo_height())
        try:
            img = Image.open(path).convert("RGB")
            scale = max(w / img.width, h / img.height)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            img = img.resize(new_size, resample)
            left = max(0, (img.width - w) // 2)
            top = max(0, (img.height - h) // 2)
            img = img.crop((left, top, left + w, top + h))
            self._scene_photo = ImageTk.PhotoImage(img)
        except Exception as e:
            canvas.delete("all")
            canvas.create_text(w // 2, h // 2, text=f"场景图读取失败：{e}", fill="#D7E2EC")
            return
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, image=self._scene_photo, anchor="center")

    def _sample_summary(self, objects: list[dict]) -> str:
        counts: dict[str, int] = {}
        for obj in objects:
            cls = str(obj.get("class") or "未知物体")
            counts[cls] = counts.get(cls, 0) + 1
        parts = [f"{name}×{count}" for name, count in list(counts.items())[:6]]
        more = "" if len(counts) <= 6 else f" 等 {len(counts)} 类"
        return "，".join(parts) + more if parts else "无物体"

    def _objects_for_sample(self, sid: str) -> list[dict]:
        ann_path = ROOT.parent / "dataset_learn" / str(sid) / "annotations.json"
        if not ann_path.exists():
            return []
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return []
        return [obj for obj in ann.get("objects") or [] if obj.get("class")]

    def _reference_voice_for_sample(self, sid: str) -> str:
        sid = str(sid)
        if sid in HIGH_SUCCESS_HINTS:
            return HIGH_SUCCESS_HINTS[sid]
        objects = self._objects_for_sample(sid)
        if len(objects) >= 3:
            return "把所有零件放进收纳筐"
        if len(objects) >= 2:
            return "随便拿取两个零件放进收纳筐"
        if objects:
            cls = str(objects[0].get("class") or "零件")
            name = cls.split("-")[-1] if "-" in cls else cls
            return f"拿取{name}放进收纳筐"
        return "随便拿取一个零件放进收纳筐"

    def _set_reference_voice_for_sample(self, sid: str) -> None:
        if hasattr(self, "reference_voice_label"):
            self.reference_voice_label.configure(text=self._reference_voice_for_sample(sid))

    def _sample_detail_lines(self, objects: list[dict]) -> list[str]:
        lines = []
        for obj in objects[:20]:
            pose = obj.get("pose_6d") or {}
            xy = ""
            if "x" in pose and "y" in pose:
                xy = f" @ ({float(pose.get('x', 0.0)):.3f}, {float(pose.get('y', 0.0)):.3f})"
            state = obj.get("state") or obj.get("pose_state") or "normal"
            lines.append(
                f"#{obj.get('id', '?')}  {obj.get('class', '未知物体')}  "
                f"state={state}{xy}"
            )
        if len(objects) > 20:
            lines.append(f"... 还有 {len(objects) - 20} 个物体未显示")
        return lines

    def _refresh_sample_preview(self) -> None:
        sid = self.sample_var.get().strip() or "0001"
        try:
            vision, sid = load_vision_by_sample(sid)
            objects = vision.get("objects") or []
            self.sample_preview_var.set(
                f"样本 {sid}：{len(objects)} 个物体｜{self._sample_summary(objects)}"
            )
        except Exception as e:
            self.sample_preview_var.set(f"样本 {sid} 读取失败：{e}")

    def on_preview_sample(self) -> None:
        sid = self.sample_var.get().strip() or "0001"
        try:
            vision, sid = load_vision_by_sample(sid)
            objects = vision.get("objects") or []
        except Exception as e:
            messagebox.showerror("样本预览失败", str(e))
            return
        summary = self._sample_summary(objects)
        details = "\n".join(self._sample_detail_lines(objects)) or "无物体"
        messagebox.showinfo(
            "样本预览",
            f"样本：{sid}\n物体数：{len(objects)}\n摘要：{summary}\n\n{details}",
        )

    def _is_ec_command(self, cmd: str) -> bool:
        text = (cmd or "").replace(" ", "")
        has_screw = any(k in text for k in ("喇叭头", "螺丝", "螺钉", "螺栓"))
        has_slot = any(k in text for k in ("对应格子", "格子", "料盘", "入格", "格"))
        return has_screw and has_slot

    def _parse_count_token(self, token: str) -> int | None:
        token = str(token or "").strip()
        if not token:
            return None
        if token.isdigit():
            return int(token)
        digits = {
            "零": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if token in digits:
            return digits[token]
        if "十" in token:
            left, right = token.split("十", 1)
            tens = 1 if left == "" else digits.get(left)
            ones = 0 if right == "" else digits.get(right)
            if tens is None or ones is None:
                return None
            return tens * 10 + ones
        return None

    def _ec_slot_dest(self, template_batch: list[dict], slot_id: int):
        for item in template_batch:
            if int(item.get("slot_id", -1)) == int(slot_id):
                return item.get("dest_coordinate")
        return None

    def _clone_ec_job(
        self,
        src: dict,
        index: int,
        slot_no: int,
        template_batch: list[dict],
    ) -> dict:
        job = json.loads(json.dumps(src, ensure_ascii=False))
        slot_id = int(slot_no) - 1
        if slot_id < 0:
            raise ValueError("格子编号必须从 1 开始")
        job["index"] = int(index)
        job["slot_id"] = slot_id
        job["destination"] = f"料盘格子{slot_id}"
        dest = self._ec_slot_dest(template_batch, slot_id)
        if dest is not None:
            job["dest_coordinate"] = dest
        return job

    def _select_ec_batch_by_cmd(self, cmd: str, template_batch: list[dict]) -> tuple[list[dict], str]:
        text = (cmd or "").replace(" ", "")
        num_pat = r"\d+|[零一二两三四五六七八九十]+"
        by_object = {int(item.get("object_id")): item for item in template_batch}

        pairs = []
        pair_pat = rf"第({num_pat})个?螺(?:丝|钉|栓).*?第({num_pat})个?格"
        for m in re.finditer(pair_pat, text):
            obj_no = self._parse_count_token(m.group(1))
            slot_no = self._parse_count_token(m.group(2))
            if obj_no is not None and slot_no is not None:
                pairs.append((obj_no, slot_no))

        if pairs:
            selected = []
            for idx, (obj_no, slot_no) in enumerate(pairs, 1):
                if obj_no not in by_object:
                    raise ValueError(f"EC 模板里没有第 {obj_no} 个螺丝")
                selected.append(
                    self._clone_ec_job(by_object[obj_no], idx, slot_no, template_batch)
                )
            return selected, "指定螺丝到指定格子"

        obj_pat = rf"第({num_pat})个?螺(?:丝|钉|栓)"
        obj_match = re.search(obj_pat, text)
        if obj_match:
            obj_no = self._parse_count_token(obj_match.group(1))
            if obj_no not in by_object:
                raise ValueError(f"EC 模板里没有第 {obj_no} 个螺丝")
            return [
                self._clone_ec_job(by_object[obj_no], 1, obj_no, template_batch)
            ], f"第 {obj_no} 个螺丝放入对应格子"

        count_patterns = (
            rf"前({num_pat})个",
            rf"(?:只要|演示|展示|跑|执行)({num_pat})个",
        )
        for pat in count_patterns:
            m = re.search(pat, text)
            if not m:
                continue
            n = self._parse_count_token(m.group(1))
            if n is None:
                continue
            n = max(1, min(int(n), len(template_batch)))
            selected = [
                self._clone_ec_job(item, idx + 1, int(item.get("slot_id", idx)) + 1, template_batch)
                for idx, item in enumerate(template_batch[:n])
            ]
            return selected, f"前 {n} 个螺丝"

        selected = [
            self._clone_ec_job(item, idx + 1, int(item.get("slot_id", idx)) + 1, template_batch)
            for idx, item in enumerate(template_batch)
        ]
        return selected, "完整 EC batch"

    def _build_ec_task(self, cmd: str) -> dict:
        if not EC_TASK_TEMPLATE.exists():
            raise FileNotFoundError(f"找不到 EC 任务模板：{EC_TASK_TEMPLATE}")
        task = json.loads(EC_TASK_TEMPLATE.read_text(encoding="utf-8-sig"))
        template_batch = task.get("batch") or []
        if not template_batch:
            raise ValueError(f"EC 任务模板没有 batch：{EC_TASK_TEMPLATE}")
        selected_batch, select_reason = self._select_ec_batch_by_cmd(cmd, template_batch)
        task["batch"] = selected_batch
        task["job_count"] = len(selected_batch)
        task["original_cmd"] = cmd
        task["source"] = f"ec_template:test03:{select_reason}"
        task["reason"] = f"按口令生成：{select_reason}"
        if selected_batch:
            first = selected_batch[0]
            task["coordinate"] = first.get("coordinate")
            task["dest_coordinate"] = first.get("dest_coordinate")
            task["destination"] = first.get("destination")
            task["object"] = first.get("object") or task.get("object")
        return task

    def _load_ec_annotations(self) -> tuple[dict, Path]:
        ann_path = EC_SAMPLE_DIR / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"找不到 EC 视觉标注：{ann_path}")
        return json.loads(ann_path.read_text(encoding="utf-8")), ann_path

    def _finish_ec_delivery(self, cmd: str, task: dict, *, ann_path: Path | None = None) -> None:
        task = rehydrate_tasks(task)
        if not task.get("batch"):
            raw = task.get("batch_json")
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    task["batch"] = parsed
        agent_source = self._task_source_label(task)
        if str(task.get("source") or "").startswith("ec_template"):
            agent_source = "本地 EC 规则生成"
        print(f"[JSON_SOURCE] {agent_source}", flush=True)
        EC_TASK_OUT.parent.mkdir(parents=True, exist_ok=True)
        EC_TASK_OUT.write_text(
            json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        delivery = {
            "scene": "ec_bin",
            "sample_id": EC_SAMPLE_ID,
            "agent_source": agent_source,
            "user_cmd": cmd,
            "task": task,
        }
        if ann_path is not None:
            delivery["yolo_annotations"] = str(ann_path.resolve())
        self._post_ui(lambda: self._on_ok(delivery, EC_TASK_OUT))

    def _run_ec_with_agent(self, cmd: str) -> None:
        self._post_ui(lambda: self._append_log(
            f"[CURRENT_SCENE] sample={EC_SAMPLE_ID} reason=EC料盘场景，任务走智能体"
        ))
        self._post_ui(self._show_ec_start_scene)
        self._post_ui(lambda: self._set_stage(2))
        ann, ann_path = self._load_ec_annotations()
        missing_tokens = self._missing_object_tokens(cmd, ann)
        if missing_tokens:
            self._post_ui(lambda m=missing_tokens: self._on_missing_object(m))
            return
        vision = self._vision_from_ann(ann)
        self._post_ui(lambda: self._set_stage(3))
        self._post_ui(lambda: self._set_status("智能体规划中…"))
        try:
            task = run_agent(cmd, vision, sample_id="ec0001", allow_fallback=False)
        except Exception as e:
            self._post_ui(lambda msg=str(e): self._append_log(
                f"[WARN] Dify 失败，回退本地 EC 规则：{msg}"
            ))
            task = self._build_ec_task(cmd)
        self._finish_ec_delivery(cmd, task, ann_path=ann_path)

    def _storage_from_ann(self, ann: dict) -> dict:
        storage = (ann.get("scene_layout") or {}).get("storage_box") or {}
        xyz = list(storage.get("pos_m") or [0.3, -0.2, 0.0])
        while len(xyz) < 3:
            xyz.append(0.0)
        return {"name": "收纳盒", "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])]}

    def _vision_from_ann(self, ann: dict) -> dict:
        return {
            "objects": ann.get("objects") or [],
            "target_container": self._storage_from_ann(ann),
            "scene_layout": ann.get("scene_layout") or {},
        }

    def _task_source_label(self, task: dict) -> str:
        source = str((task or {}).get("source") or "")
        if source.startswith("dify_api"):
            return "远程 Dify 智能体"
        if source == "local_feasibility_precheck":
            return "本地可行性预筛"
        if source == "local_fallback" or source.startswith("local"):
            return "本地 fallback 智能体"
        return source or "未知来源"

    def _parse_random_scene_count(self, cmd: str, total: int) -> int | None:
        text = self._compact_text(cmd)
        dest_words = (
            "收纳盒",
            "收纳箱",
            "收纳筐",
            "收纳框",
            "料框",
            "料筐",
            "料箱",
            "收纳",
            "筐",
            "箱",
            "盒",
        )
        part_words = (
            "零件",
            "零建",
            "零键",
            "连接",
            "物体",
            "工件",
            "部件",
            "配件",
        )
        all_parts = ("所有" in text or "全部" in text) and any(
            key in text for key in part_words
        )
        if all_parts and any(word in text for word in dest_words):
            return max(1, total)

        if not any(key in text for key in ("随便", "随机", "任意")):
            return None
        if not any(key in text for key in part_words):
            return None

        m = re.search(r"(\d+)(?:个|件)?", text)
        if m:
            return max(1, min(int(m.group(1)), total))

        zh_digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "俩": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        for word, value in zh_digits.items():
            if f"{word}个" in text or f"{word}件" in text or any(
                f"{word}{part}" in text for part in part_words
            ):
                return max(1, min(value, total))
        return 1

    def _task_from_ann_object(
        self,
        obj: dict,
        index: int,
        cmd: str,
        dest_xyz: list[float],
    ) -> dict:
        cls = str(obj.get("class") or "")
        pose = obj.get("pose_6d") or {}
        coord = [
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("z", 0.0)),
        ]
        yaw = float(pose.get("yaw", 0.0))
        rpy = [
            float(pose.get("roll", 0.0)),
            float(pose.get("pitch", 0.0)),
            yaw,
        ]
        actions = ("perceive", "pick", "move", "place")
        steps = []
        for step_no, action in enumerate(actions, start=1):
            step_coord = dest_xyz if action in {"move", "place"} else coord
            steps.append(
                {
                    "task_id": f"t{step_no:03d}",
                    "step": step_no,
                    "action": action,
                    "object": cls,
                    "coordinate": step_coord,
                    "dest_coordinate": dest_xyz,
                    "destination": "收纳盒",
                    "angle": yaw,
                    "rpy": rpy,
                    "retry": 0,
                    "max_retry": 3,
                    "status": "pending",
                    "reason": "随机候选预筛",
                    "original_cmd": cmd,
                    "pose_state": obj.get("state") or obj.get("pose_state") or "normal",
                }
            )
        return {
            "task_id": f"t_random_feasible_{index:02d}",
            "action": "pick_place",
            "object": cls,
            "coordinate": coord,
            "dest_coordinate": dest_xyz,
            "destination": "收纳盒",
            "angle": yaw,
            "rpy": rpy,
            "retry": 0,
            "max_retry": 3,
            "status": "pending",
            "reason": "随机候选预筛通过后执行",
            "original_cmd": cmd,
            "pose_state": obj.get("state") or obj.get("pose_state") or "normal",
            "tasks": steps,
            "task_sequence_desc": [
                "1.perceive 识别定位目标",
                "2.pick 抓取目标零件",
                "3.move 移动至放置点",
                "4.place 放入收纳盒",
            ],
        }

    def _precheck_random_feasible_task(
        self,
        cmd: str,
        ann: dict,
        sid: str,
        dataset_root: Path,
        agent_source: str,
    ) -> dict | None:
        objects = [obj for obj in (ann.get("objects") or []) if obj.get("class")]
        requested = self._parse_random_scene_count(cmd, len(objects))
        if requested is None or requested <= 0:
            return None

        dest_xyz = self._storage_from_ann(ann)["xyz"]
        candidates = objects[:]
        random.shuffle(candidates)
        out_dir = DEFAULT_OUT.parent / "feasibility_checks"
        out_dir.mkdir(parents=True, exist_ok=True)
        ok_tasks: list[dict] = []

        self._post_ui(lambda n=requested: self._append_log(f"[PLAN_CHECK] 多件任务预筛：目标 {n} 个，先找可规划成功的零件"))
        if ann.get("purpose") == "guaranteed_all_parts_pick_place_demo" and requested <= len(objects):
            # 0501 当初跑通的顺序：小螺丝先占料框正中，倒下件最后放。
            # 按 annotations.json 原文顺序会让塑料螺丝先占正中，第 3 件被分到无 IK 的右边点。
            pack_rank = {
                "螺钉螺栓-石膏板圆头螺丝": 0,
                "轴承-组合轴承": 1,
                "螺钉螺栓-塑料圆柱头螺丝": 2,
            }
            packed = sorted(
                objects,
                key=lambda o: pack_rank.get(str(o.get("class") or "").strip(), 99),
            )
            for idx, obj in enumerate(packed[:requested], start=1):
                candidate = self._task_from_ann_object(obj, idx, cmd, dest_xyz)
                candidate["source"] = agent_source or "local_fallback"
                candidate["postprocess"] = "local_feasibility_precheck"
                ok_tasks.append(candidate)
                cls = str(obj.get("class") or "")
                self._post_ui(lambda c=cls, k=len(ok_tasks): self._append_log(f"[PLAN_CHECK] 0501固定成功集，直接加入 {k}/{requested}: {c}"))
        else:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            for idx, obj in enumerate(candidates, start=1):
                candidate = self._task_from_ann_object(obj, idx, cmd, dest_xyz)
                candidate["source"] = agent_source or "local_fallback"
                candidate["postprocess"] = "local_feasibility_precheck"
                path = out_dir / f"{sid}_candidate_{idx:02d}.json"
                path.write_text(
                    json.dumps({"task": candidate}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                cls = str(obj.get("class") or "")
                try:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            str(STD_SCRIPT),
                            "--task",
                            str(path.resolve()),
                            "--sample-id",
                            sid,
                            "--dataset-root",
                            str(dataset_root.resolve()),
                            "--no-anime",
                        ],
                        cwd=str(ROOT.parent.parent),
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    self._post_ui(lambda c=cls: self._append_log(f"[PLAN_CHECK] 预筛超时，跳过: {c}"))
                    continue
                if proc.returncode == 0:
                    ok_tasks.append(candidate)
                    self._post_ui(lambda c=cls, k=len(ok_tasks): self._append_log(f"[PLAN_CHECK] 可规划成功 {k}/{requested}: {c}"))
                    if len(ok_tasks) >= requested:
                        break
                else:
                    self._post_ui(lambda c=cls: self._append_log(f"[PLAN_CHECK] 跳过不可规划零件: {c}"))

        if len(ok_tasks) < requested:
            self.after(
                0,
                lambda a=len(ok_tasks), n=requested: self._append_log(
                    f"[PLAN_CHECK] 目标 {n} 个中仅 {a} 个可规划成功，不满足数量要求"
                ),
            )
            return {
                "__precheck_failed__": True,
                "requested": requested,
                "available": len(ok_tasks),
            }
        if len(ok_tasks) == 1:
            return ok_tasks[0]

        flat = []
        batch = []
        for batch_index, task in enumerate(ok_tasks, start=1):
            for step in task.get("tasks") or []:
                item = dict(step)
                item["batch_index"] = batch_index
                item["step"] = len(flat) + 1
                item["task_id"] = f"t{len(flat) + 1:03d}"
                flat.append(item)
            batch.append(
                {
                    "index": batch_index,
                    "object": task.get("object"),
                    "coordinate": task.get("coordinate"),
                    "dest_coordinate": task.get("dest_coordinate"),
                    "destination": task.get("destination"),
                    "steps": [step.get("action") for step in task.get("tasks") or []],
                }
            )
        first = ok_tasks[0]
        return {
            "task_id": "t_random_feasible_multi",
            "action": "multi_pick_place",
            "object": "、".join(str(task.get("object") or "") for task in ok_tasks),
            "coordinate": first.get("coordinate"),
            "dest_coordinate": first.get("dest_coordinate"),
            "destination": first.get("destination"),
            "status": "pending",
            "reason": f"随机任务已预筛，选择 {len(ok_tasks)} 个可规划成功零件",
            "source": agent_source or "local_fallback",
            "postprocess": "local_feasibility_precheck",
            "original_cmd": cmd,
            "batch": batch,
            "tasks": flat,
            "task_sequence_desc": [
                f"{item['index']}.{item['object']}:{'>'.join(item['steps'])}"
                for item in batch
            ],
        }

    def _gt_ann_for_sample(self, sid: str) -> dict | None:
        path = DATASET_LEARN / sid / "annotations.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _pose_from_obj(self, obj: dict) -> tuple[list[float], list[float], float, str]:
        pose = obj.get("pose_6d") or {}
        coord = [
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("z", 0.0)),
        ]
        yaw = float(pose.get("yaw", 0.0))
        rpy = [
            float(pose.get("roll", 0.0)),
            float(pose.get("pitch", 0.0)),
            yaw,
        ]
        state = str(obj.get("state") or obj.get("pose_state") or "normal")
        return coord, rpy, yaw, state

    def _match_gt_object(
        self,
        gt_ann: dict,
        cls: str,
        hint_xy=None,
        used_ids: set[int] | None = None,
    ) -> dict | None:
        hits = []
        for obj in gt_ann.get("objects") or []:
            if str(obj.get("class") or "") != cls:
                continue
            oid = int(obj.get("id") or 0)
            if used_ids and oid in used_ids:
                continue
            hits.append(obj)
        if not hits:
            return None
        if hint_xy is not None and len(hits) > 1:
            try:
                hx, hy = float(hint_xy[0]), float(hint_xy[1])
            except (TypeError, ValueError, IndexError):
                hx = hy = 0.0

            def _dist(obj: dict) -> float:
                pose = obj.get("pose_6d") or {}
                dx = float(pose.get("x", 0.0)) - hx
                dy = float(pose.get("y", 0.0)) - hy
                return dx * dx + dy * dy

            hits.sort(key=_dist)
        return hits[0]

    def _write_gt_pick_pose(self, item: dict, obj: dict) -> None:
        coord, rpy, yaw, state = self._pose_from_obj(obj)
        item["coordinate"] = coord
        item["angle"] = yaw
        item["rpy"] = rpy
        item["pose_state"] = state

    def _align_task_to_gt(self, task: dict, gt_ann: dict) -> dict:
        """把决策给出的抓取坐标对齐到 dataset_learn 真值，便于仿真按标注摆件。"""
        if not task or not gt_ann:
            return task

        def snap(item: dict, consume: bool, used: set[int]) -> None:
            cls = str(item.get("object") or "")
            if not cls or cls.startswith("料箱"):
                return
            obj = self._match_gt_object(gt_ann, cls, item.get("coordinate"), used)
            if obj is None:
                return
            if consume:
                used.add(int(obj.get("id") or 0))
            self._write_gt_pick_pose(item, obj)

        used: set[int] = set()
        snap(task, consume=False, used=used)
        for step in task.get("tasks") or []:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action") or "").lower()
            if action in {"move", "place"}:
                continue
            snap(step, consume=False, used=used)
        used = set()
        for item in self._task_batch_items(task):
            snap(item, consume=True, used=used)
        return task

    def _run_yolo_scene(self, sample: str) -> tuple[dict, str, Path, Path]:
        sid = sample.zfill(4) if sample.isdigit() else sample
        curated_ann_path = ROOT.parent / "dataset_learn" / sid / "annotations.json"
        if curated_ann_path.exists():
            try:
                curated_ann = json.loads(curated_ann_path.read_text(encoding="utf-8"))
            except Exception:
                curated_ann = None
            if isinstance(curated_ann, dict) and curated_ann.get("purpose") == "guaranteed_all_parts_pick_place_demo":
                ann_path = YOLO_OUTPUTS / f"{sid}.json"
                ann_path.parent.mkdir(parents=True, exist_ok=True)
                ann_path.write_text(json.dumps(curated_ann, ensure_ascii=False, indent=2), encoding="utf-8")
                sim_sample_dir = YOLO_SIM_DATASET / sid
                sim_sample_dir.mkdir(parents=True, exist_ok=True)
                sim_ann_path = sim_sample_dir / "annotations.json"
                sim_ann_path.write_text(json.dumps(curated_ann, ensure_ascii=False, indent=2), encoding="utf-8")
                self._post_ui(lambda: self._set_status("使用固定成功识别场景…"))
                self._post_ui(lambda p=sim_ann_path: self._append_log(
                    f"[YOLO] sample={sid} 使用固定成功 annotations，仿真仍读 {DATASET_LEARN}"
                ))
                return curated_ann, sid, ann_path, DATASET_LEARN

        if not YOLO_INFER_SCRIPT.exists():
            raise FileNotFoundError(f"找不到 YOLO 推理脚本：{YOLO_INFER_SCRIPT}")
        if not YOLO_WEIGHTS.exists():
            raise FileNotFoundError(f"找不到 YOLO 权重：{YOLO_WEIGHTS}")

        self._post_ui(lambda: self._set_status("YOLO 识别场景中…"))
        self._post_ui(lambda: self._append_log(f"[YOLO] sample={sid} weights={YOLO_WEIGHTS.name}"))

        cmd = [
            sys.executable,
            str(YOLO_INFER_SCRIPT),
            "--sample",
            sid,
            "--weights",
            str(YOLO_WEIGHTS),
            "--save-masks",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(YOLO_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        output = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if output:
            self._post_ui(lambda text=output: self._append_log(text))
        if err:
            self._post_ui(lambda text=err: self._append_log(text))
        if proc.returncode != 0:
            raise RuntimeError(f"YOLO 识别失败，退出码 {proc.returncode}")

        ann_path = YOLO_OUTPUTS / f"{sid}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"YOLO 未生成最终 JSON：{ann_path}")
        ann = json.loads(ann_path.read_text(encoding="utf-8"))

        sim_sample_dir = YOLO_SIM_DATASET / sid
        sim_sample_dir.mkdir(parents=True, exist_ok=True)
        sim_ann_path = sim_sample_dir / "annotations.json"
        sim_ann_path.write_text(
            json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._post_ui(lambda: self._append_log(
            f"[YOLO] 识别已写入 {ann_path}；仿真摆件仍用 {DATASET_LEARN / sid}"
        ))
        return ann, sid, ann_path, DATASET_LEARN

    def _run_cmd(self, cmd: str) -> None:
        if self._recording:
            messagebox.showinfo("提示", "请先结束录音")
            return
        self.cmd_var.set(cmd)
        self.on_run()

    def on_run(self) -> None:
        self._run_text_command(self.cmd_var.get().strip())

    def _run_text_command(self, cmd: str) -> None:
        if self._recording:
            messagebox.showinfo("提示", "请先结束录音")
            return
        if self._busy:
            messagebox.showinfo("提示", "正在跑上一条，请稍等")
            return
        if not cmd:
            messagebox.showwarning("提示", "请先输入指令")
            return
        self._busy = True
        self._set_status("基于当前随机场景识别…")
        self._set_stage(2)
        self.log.delete("1.0", "end")
        self._append_log(f"[CMD] {cmd}")
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    # ---------- 录音：开始 / 结束 ----------
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
            messagebox.showerror(
                "缺少依赖",
                f"麦克风需要：pip install -r requirements.txt\n{e}",
            )
            return

        self._rec_chunks = queue.Queue()
        self._rec_rate = 16000
        self._rec_live_peak = 0
        self._rec_live_rms = 0.0
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
                peak, rms = audio_level_stats(indata)
                self._rec_live_peak = max(int(self._rec_live_peak * 0.75), peak)
                self._rec_live_rms = rms
            except Exception:
                pass

        try:
            # 与改采样率之前相同：16k int16，这套之前能识别
            self._rec_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="int16",
                callback=callback,
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
        self._append_log("[MIC] 录音已开始，说完请点「结束并识别」")
        self._tick_rec_timer()

    def _tick_rec_timer(self) -> None:
        if not self._recording:
            return
        sec = int(time.time() - self._rec_started_at)
        meter_value = min(100, int(self._rec_live_peak * 100 / 3000))
        self.mic_meter.configure(value=meter_value)
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
        threading.Thread(target=self._worker_mic_chunks, args=(chunks,), daemon=True).start()

    def _worker_mic_chunks(self, chunks: list) -> None:
        try:
            import numpy as np

            audio = np.concatenate(chunks, axis=0)
            peak, rms = audio_level_stats(audio)
            self._post_ui(lambda p=peak, r=rms: self._append_log(f"[MIC] 音量 peak={p} rms={r:.1f}"))
            if peak < 200 or rms < 30:
                raise RuntimeError(
                    "麦克风录到的声音太小，请检查 Windows 输入设备/输入音量，或靠近麦克风再试"
                )
            pcm = np.squeeze(audio).astype("int16").tobytes()
            if len(pcm) < 16000:  # <0.5s @16k mono int16
                raise RuntimeError("录音太短，请多说一会儿再点结束")

            self._post_ui(lambda: self._set_stage(1))
            try:
                text = transcribe_pcm16(pcm, 16000)
            except CnSttError as e:
                raise RuntimeError(str(e)) from e

            text = text.strip()
            self._post_ui(lambda t=text: self.cmd_var.set(t))
            self._post_ui(lambda t=text: self._append_log(f"[STT] {t}"))
            self._worker(text)
        except Exception as e:
            err = str(e)
            self._post_ui(lambda msg=err: self._on_fail(msg))

    def on_audio(self) -> None:
        if self._recording:
            messagebox.showinfo("提示", "请先结束录音")
            return
        path = filedialog.askopenfilename(
            title="选择录音",
            filetypes=[
                ("音频", "*.m4a;*.wav;*.mp3;*.aac"),
                ("全部", "*.*"),
            ],
        )
        if not path:
            return
        if self._busy:
            messagebox.showinfo("提示", "正在跑上一条，请稍等")
            return
        self._busy = True
        self._set_status("语音转文字中…")
        self.log.delete("1.0", "end")
        threading.Thread(target=self._worker_audio, args=(path,), daemon=True).start()

    def _worker_audio(self, path: str) -> None:
        try:
            self._post_ui(lambda: self._set_stage(1))
            text = speech_to_text_from_file(Path(path))
            self._post_ui(lambda: self.cmd_var.set(text))
            self._post_ui(lambda: self._append_log(f"[STT] {text}"))
            self._worker(text)
        except Exception as e:
            err = str(e)
            self._post_ui(lambda msg=err: self._on_fail(msg))

    def _worker(self, cmd: str, sample: str | None = None) -> None:
        try:
            if FORCE_START_SCENE == "ec_bin" or self._is_ec_command(cmd):
                self._run_ec_with_agent(cmd)
                return

            sid, reason = (
                (sample, "指定场景")
                if sample
                else (self._current_sample_id, "启动时随机场景")
            )
            sid = str(sid or "0001").zfill(4) if str(sid or "").isdigit() else str(sid or "0001")
            self._post_ui(lambda s=sid, r=reason: self._append_log(
                f"[CURRENT_SCENE] sample={s} reason={r}"
            ))
            self._post_ui(lambda: self._show_scene_message("YOLO 识别场景中…"))
            self._post_ui(lambda: self._set_stage(2))
            ann, sid, ann_path, dataset_root = self._run_yolo_scene(sid)
            missing_tokens = self._missing_object_tokens(cmd, ann)
            if missing_tokens:
                self._post_ui(lambda m=missing_tokens: self._on_missing_object(m))
                return
            vision = self._vision_from_ann(ann)
            self._post_ui(lambda: self._set_stage(3))
            self._post_ui(lambda: self._set_status("智能体规划中…"))
            task = run_agent(cmd, vision, sample_id=sid, allow_fallback=True)
            task = rehydrate_tasks(task)
            gt_ann = self._gt_ann_for_sample(sid)
            if gt_ann:
                task = self._align_task_to_gt(task, gt_ann)
                self._post_ui(lambda: self._append_log(
                    "[SIM] 识别只给决策；仿真按 dataset_learn 真值摆件并对齐抓取坐标"
                ))
            task_source = str(task.get("source") or "")
            precheck_ann = ann
            if gt_ann:
                yolo_cls = {
                    str(obj.get("class") or "").strip()
                    for obj in (ann.get("objects") or [])
                    if str(obj.get("class") or "").strip()
                }
                precheck_ann = dict(gt_ann)
                precheck_ann["objects"] = [
                    obj
                    for obj in (gt_ann.get("objects") or [])
                    if str(obj.get("class") or "").strip() in yolo_cls
                ]
            feasible_task = self._precheck_random_feasible_task(
                cmd, precheck_ann, sid, dataset_root, task_source
            )
            if feasible_task is not None:
                if feasible_task.get("__precheck_failed__"):
                    self.after(
                        0,
                        lambda info=feasible_task: self._on_precheck_not_enough(
                            int(info.get("requested") or 0),
                            int(info.get("available") or 0),
                        ),
                    )
                    return
                task = feasible_task
            agent_source = self._task_source_label(task)
            print(f"[JSON_SOURCE] {agent_source}", flush=True)
            delivery = {
                "scene": "standard",
                "scene_source": "yolo",
                "sample_id": sid,
                "scene_reason": reason,
                "agent_source": agent_source,
                "user_cmd": cmd,
                "yolo_annotations": str(ann_path.resolve()),
                "dataset_root": str(dataset_root.resolve()),
                "task": task,
            }
            out_path = DEFAULT_OUT
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._post_ui(lambda: self._on_ok(delivery, out_path))
        except Exception as e:
            err = str(e)
            self._post_ui(lambda msg=err: self._on_fail(msg))

    def _split_multi_pick_place_tasks(self, delivery: dict) -> list[Path]:
        task = delivery.get("task") or {}
        batch_items = self._task_batch_items(task)
        if task.get("action") not in {"multi_pick_place", "multi_agent_pack"} and not batch_items:
            return []

        grouped: list[tuple[str, list[dict]]] = []
        if batch_items:
            for item in batch_items:
                obj = str(item.get("object") or "")
                if not obj or obj.startswith("料箱"):
                    continue
                grouped.append((obj, self._steps_from_batch_item(task, item)))
        else:
            for step in task.get("tasks") or []:
                obj = str(step.get("object") or "")
                action = str(step.get("action") or "").lower()
                if not obj or action not in {"perceive", "pick", "adjust_pose", "move", "place"}:
                    continue
                if obj.startswith("料箱"):
                    continue
                if not grouped or grouped[-1][0] != obj:
                    grouped.append((obj, []))
                grouped[-1][1].append(step)

        paths: list[Path] = []
        out_dir = DEFAULT_OUT.parent / "multi_pick_place"
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, (obj, steps) in enumerate(grouped, start=1):
            pick_step = next(
                (step for step in steps if str(step.get("action") or "").lower() == "pick"),
                steps[0] if steps else {},
            )
            if not pick_step:
                continue
            subtask = dict(pick_step)
            subtask["task_id"] = f"{task.get('task_id', 'multi_pick_place')}_{index:02d}"
            subtask["action"] = "pick_place"
            subtask["object"] = obj
            subtask["tasks"] = [dict(step) for step in steps]
            subtask["task_sequence_desc"] = [
                f"{index}.{obj}:{'>'.join(str(step.get('action') or '') for step in steps)}"
            ]
            sub_delivery = dict(delivery)
            sub_delivery["task"] = subtask
            sub_delivery["multi_pick_place_index"] = index
            sub_delivery["multi_pick_place_total"] = len(grouped)
            path = out_dir / f"delivery_agent_auto_part_{index:02d}.json"
            path.write_text(json.dumps(sub_delivery, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(path)
        return paths

    def _task_batch_items(self, task: dict) -> list[dict]:
        batch = task.get("batch")
        if isinstance(batch, list) and batch:
            return [item for item in batch if isinstance(item, dict)]
        batch_json = task.get("batch_json")
        if isinstance(batch_json, str) and batch_json.strip():
            try:
                parsed = json.loads(batch_json)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        return []

    def _steps_from_batch_item(self, task: dict, item: dict) -> list[dict]:
        obj = str(item.get("object") or task.get("object") or "")
        coord = item.get("coordinate") or task.get("coordinate") or [0.0, 0.0, 0.0]
        dest = item.get("dest_coordinate") or task.get("dest_coordinate") or [0.3, -0.2, 0.0]
        destination = item.get("destination") or task.get("destination") or "收纳盒"
        pose_state = item.get("pose_state") or task.get("pose_state") or "normal"
        actions = item.get("steps") or ["perceive", "pick", "move", "place"]
        steps = []
        for idx, action in enumerate(actions, start=1):
            step_coord = dest if str(action).lower() in {"move", "place"} else coord
            steps.append(
                {
                    "task_id": f"t{idx:03d}",
                    "step": idx,
                    "action": action,
                    "object": obj,
                    "coordinate": step_coord,
                    "dest_coordinate": dest,
                    "destination": destination,
                    "angle": item.get("angle", task.get("angle", 0.0)),
                    "rpy": item.get("rpy", task.get("rpy", [0.0, 0.0, item.get("angle", 0.0)])),
                    "retry": 0,
                    "max_retry": 3,
                    "status": "pending",
                    "reason": item.get("reason", "batch 子任务"),
                    "original_cmd": task.get("original_cmd"),
                    "pose_state": pose_state,
                }
            )
        return steps

    def _launch_multi_pick_place_sequence(
        self,
        delivery: dict,
        sample_id: str,
        dataset_root: str | None,
    ) -> bool:
        paths = self._split_multi_pick_place_tasks(delivery)
        if len(paths) < 2:
            return False

        total = len(paths)
        self._sim_queue.clear()
        batch_path = DEFAULT_OUT.parent / "multi_pick_place" / "delivery_agent_auto_batch.json"
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_log(
            f"[SIM] 单进程连续 pick-place：共 {total} 个目标，将在同一个仿真窗口内依次抓取放置"
            "（仿真界面：连续抓放，不回 HOME）"
        )
        self._launch_sim(batch_path, sample_id, scene="standard", dataset_root=dataset_root)
        return True

    def _launch_next_queued_sim(self) -> None:
        if not self._sim_queue:
            return
        path, sample_id, scene, dataset_root, index, total = self._sim_queue.pop(0)
        self._append_log(f"[SIM] 连续 pick-place：开始第 {index}/{total} 个")
        self._launch_sim(path, sample_id, scene=scene, dataset_root=dataset_root)

    def _restore_scene_after_sim_failure(self) -> None:
        self._show_random_start_scene()
        self._set_status("仿真失败，已恢复识别场景")

    def _on_precheck_not_enough(self, requested: int, available: int) -> None:
        self._busy = False
        self._append_log(
            f"[PLAN_CHECK] 不启动仿真：你要求 {requested} 个，但当前场景只有 {available} 个可规划成功"
        )
        self._append_log("当前场景无法满足该数量要求，请重新下发指令或更换场景")
        self._set_stage(3, failed=True)
        self._show_random_start_scene()
        self._set_status("可规划数量不足，已恢复识别场景")

    def _on_ok(self, delivery: dict, out_path: Path) -> None:
        self._busy = False
        self._last_delivery = delivery
        scene = delivery.get("scene") or "standard"
        self._append_log(json.dumps(delivery, ensure_ascii=False, indent=2))
        saved = str(out_path.resolve())
        self._set_status(f"已保存：{saved}")
        if scene == "standard" and self._launch_multi_pick_place_sequence(
            delivery,
            str(delivery.get("sample_id") or "0001"),
            delivery.get("dataset_root"),
        ):
            return
        self._launch_sim(
            out_path,
            str(delivery.get("sample_id") or "0001"),
            scene=scene,
            dataset_root=delivery.get("dataset_root"),
        )

    def _start_sim_worker(self) -> None:
        if not USE_SIM_WORKER or FORCE_START_SCENE == "ec_bin" or not STD_SCRIPT.exists():
            return
        proc = self._sim_worker_proc
        if proc is not None and proc.poll() is None:
            return
        try:
            self._sim_worker_dir.mkdir(parents=True, exist_ok=True)
            self._sim_worker_ready_file.parent.mkdir(parents=True, exist_ok=True)
            self._sim_worker_ready_file.unlink(missing_ok=True)
            for stale in self._sim_worker_dir.glob("*"):
                if stale.suffix in {".json", ".tmp"}:
                    stale.unlink(missing_ok=True)
        except Exception:
            pass
        cmd = [
            sys.executable,
            str(STD_SCRIPT),
            "--worker-dir",
            str(self._sim_worker_dir.resolve()),
            "--worker-ready-file",
            str(self._sim_worker_ready_file.resolve()),
        ]
        sim_env = os.environ.copy()
        sim_env["PYTHONIOENCODING"] = "utf-8"
        sim_env["PYTHONUTF8"] = "1"
        try:
            self._sim_worker_proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT.parent.parent),
                env=sim_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            self._append_log(f"[SIM_WORKER] 启动失败，后续回退单次启动：{e}")
            self._sim_worker_proc = None
            self._sim_worker_ready = False
            return
        self._sim_worker_ready = False
        threading.Thread(target=self._pump_worker_output, args=(self._sim_worker_proc,), daemon=True).start()
        self.after(200, self._poll_worker_ready)

    def _poll_worker_ready(self, remaining: int = 240) -> None:
        proc = self._sim_worker_proc
        if proc is None or proc.poll() is not None:
            self._sim_worker_ready = False
            return
        if self._sim_worker_ready_file.exists():
            self._sim_worker_ready = True
            return
        if remaining <= 0:
            self._sim_worker_ready = False
            self._append_log("[SIM_WORKER] 等待 ready 超时，后续回退单次启动")
            return
        self.after(250, lambda n=remaining - 1: self._poll_worker_ready(n))

    def _worker_available(self) -> bool:
        proc = self._sim_worker_proc
        return bool(
            USE_SIM_WORKER
            and self._sim_worker_ready
            and proc is not None
            and proc.poll() is None
            and not self._worker_sim_active
        )

    def _dispatch_sim_to_worker(
        self,
        task_path: Path,
        sample_id: str,
        dataset_root: str | None,
        start_file: Path | None,
        panda_geometry: tuple[int, int, int, int],
    ) -> bool:
        proc = self._sim_worker_proc
        if not self._worker_available() or proc is None:
            return False
        req = {
            "task": str(task_path.resolve()),
            "sample_id": sample_id,
            "dataset_root": dataset_root,
            "wait_start_file": str(start_file.resolve()) if start_file is not None else "",
            "no_rrt": False,
            "sim_ui_batch": True,
        }
        req_path = self._sim_worker_dir / f"request_{int(time.time() * 1000)}.json"
        tmp_path = req_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(req_path)
        except Exception as e:
            self._append_log(f"[SIM_WORKER] 任务投递失败，回退单次启动：{e}")
            return False
        self._worker_sim_active = True
        self._sim_proc = proc
        self._append_log("[SIM_WORKER] 已投递任务到常驻仿真 worker")
        self._set_status("常驻仿真 worker 已接收任务，规划中…")
        self.after(
            500,
            lambda p=proc, g=panda_geometry: self._poll_panda_window_when_ready(p, g, True),
        )
        return True

    def _launch_sim(
        self,
        task_path: Path,
        sample_id: str,
        scene: str = "standard",
        dataset_root: str | None = None,
    ) -> None:
        if self._sim_proc is self._sim_worker_proc and not self._worker_sim_active:
            self._sim_proc = None
        if self._sim_proc is not None and self._sim_proc.poll() is None:
            messagebox.showinfo("仿真已在运行", "当前已有仿真窗口在运行，请关闭后再启动新的任务。")
            return

        sim_script = EC_SCRIPT if scene == "ec_bin" else STD_SCRIPT
        if not sim_script.exists():
            messagebox.showerror("仿真启动失败", f"找不到仿真脚本：\n{sim_script}")
            return

        cmd = [
            sys.executable,
            str(sim_script),
            "--task",
            str(task_path.resolve()),
            "--auto-play",
        ]
        start_file: Path | None = None
        if os.name == "nt":
            start_file = ROOT / "tasks" / f"ui_start_{int(time.time() * 1000)}_{os.getpid()}.go"
            try:
                start_file.parent.mkdir(parents=True, exist_ok=True)
                if start_file.exists():
                    start_file.unlink()
            except Exception:
                start_file = None
            if start_file is not None:
                cmd.extend(["--wait-start-file", str(start_file.resolve())])
        if scene == "ec_bin":
            cmd.extend(["--sample-dir", str(EC_SAMPLE_DIR.resolve())])
            if EC_CACHED_TRAJ.is_file():
                cmd.extend(["--load-traj", str(EC_CACHED_TRAJ.resolve())])
        if scene != "ec_bin":
            cmd.extend(["--sample-id", sample_id])
            if dataset_root:
                cmd.extend(["--dataset-root", dataset_root])
            cmd.append("--sim-ui-batch")
        sim_env = os.environ.copy()
        sim_env["PYTHONIOENCODING"] = "utf-8"
        sim_env["PYTHONUTF8"] = "1"
        panda_geometry = self._scene_canvas_geometry()
        self._panda_hwnd = None
        self._panda_geometry = panda_geometry
        self._panda_deferred = False
        self._panda_ready_to_show = False
        self._sim_start_file = start_file
        self._set_stage(3)
        self._set_status(
            "仿真播放中…"
            if scene == "ec_bin" and EC_CACHED_TRAJ.is_file()
            else "仿真规划中…"
        )
        if scene == "standard" and self._dispatch_sim_to_worker(
            task_path,
            sample_id,
            dataset_root,
            start_file,
            panda_geometry,
        ):
            return
        try:
            self._sim_proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT.parent.parent),
                env=sim_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            messagebox.showerror("仿真启动失败", str(e))
            self._set_status("仿真启动失败")
            return

        self._append_log("[SIM] 已启动仿真动画：")
        self._append_log("[SIM] " + " ".join(cmd))
        self._set_status(
            "仿真播放中，等待动画开始…"
            if scene == "ec_bin" and EC_CACHED_TRAJ.is_file()
            else "仿真规划中，等待动画开始…"
        )
        threading.Thread(target=self._pump_sim_output, args=(self._sim_proc,), daemon=True).start()
        self.after(
            500,
            lambda p=self._sim_proc, g=panda_geometry: self._poll_panda_window_when_ready(p, g, True),
        )

    def _scene_canvas_geometry(self) -> tuple[int, int, int, int]:
        self.update_idletasks()
        margin = 10
        x = int(self.scene_canvas.winfo_rootx()) - margin
        y = int(self.scene_canvas.winfo_rooty()) - margin
        w = max(320, int(self.scene_canvas.winfo_width()) + margin * 2)
        h = max(240, int(self.scene_canvas.winfo_height()) + margin * 2)
        return x, y, w, h

    def _on_root_configure(self, event) -> None:
        if event.widget is not self:
            return
        if self._panda_hwnd is None and self._preview_hwnd is None:
            return
        if self._panda_deferred:
            return
        if self._panda_reposition_job is not None:
            self.after_cancel(self._panda_reposition_job)
        self._panda_reposition_job = self.after(120, self._reposition_panda_window)

    def _reposition_panda_window(self) -> None:
        self._panda_reposition_job = None
        if os.name != "nt":
            return
        try:
            import ctypes
        except Exception:
            return
        x, y, w, h = self._scene_canvas_geometry()
        proc = self._sim_proc
        hwnd = self._panda_hwnd
        if proc is not None and hwnd is not None and proc.poll() is None:
            self._apply_borderless_window(ctypes.windll.user32, int(hwnd), x, y, w, h)
        preview_proc = self._preview_proc
        preview_hwnd = self._preview_hwnd
        if preview_proc is not None and preview_hwnd is not None and preview_proc.poll() is None:
            self._apply_borderless_window(ctypes.windll.user32, int(preview_hwnd), x, y, w, h)

    def _window_rect(self, hwnd: int) -> tuple[int, int, int, int] | None:
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return None
        user32 = ctypes.windll.user32
        handle = wintypes.HWND(int(hwnd))
        if not user32.IsWindow(handle):
            return None
        rect = wintypes.RECT()
        if not user32.GetWindowRect(handle, ctypes.byref(rect)):
            return None
        return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)

    def _tick_panda_follow(self) -> None:
        try:
            self._sync_panda_windows_to_canvas()
        except Exception:
            pass
        if not self._closing:
            self.after(350, self._tick_panda_follow)

    def _sync_panda_windows_to_canvas(self) -> None:
        """Keep every managed Panda window glued to the scene canvas.

        Configure events alone miss maximize/fullscreen and lose track of a
        window whose handle was re-created, so re-assert the rect periodically.
        """
        if os.name != "nt" or self._closing:
            return
        if self.state() == "withdrawn":
            return
        targets: list[int] = []
        preview_proc = self._preview_proc
        if preview_proc is not None and preview_proc.poll() is None:
            hwnd = self._preview_hwnd
            if not hwnd or self._window_rect(int(hwnd)) is None:
                hwnd = self._find_panda_window_by_pid(int(preview_proc.pid))
                self._preview_hwnd = hwnd or None
            if hwnd:
                targets.append(int(hwnd))
        sim_proc = self._sim_proc
        if (
            not self._panda_deferred
            and sim_proc is not None
            and sim_proc.poll() is None
            and self._panda_hwnd
        ):
            targets.append(int(self._panda_hwnd))
        if not targets:
            return
        x, y, w, h = self._scene_canvas_geometry()
        try:
            import ctypes
        except Exception:
            return
        user32 = ctypes.windll.user32
        for hwnd in targets:
            current = self._window_rect(hwnd)
            if current is not None and current == (x, y, w, h):
                continue
            self._apply_borderless_window(user32, hwnd, x, y, w, h)

    def _show_deferred_panda_window(self) -> None:
        self._panda_ready_to_show = True
        if not self._panda_deferred or self._panda_hwnd is None:
            return
        proc = self._sim_proc
        if proc is None or proc.poll() is not None or os.name != "nt":
            return
        try:
            import ctypes
        except Exception:
            return
        self._panda_deferred = False
        geometry = self._scene_canvas_geometry()
        self._panda_geometry = geometry
        x, y, w, h = geometry
        user32 = ctypes.windll.user32
        hwnd = int(self._panda_hwnd)
        try:
            user32.ShowWindow(hwnd, 5)
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass
        self._apply_borderless_window(user32, hwnd, x, y, w, h)
        self._repeat_panda_reposition(10)
        self._append_log("[SIM] 动画准备就绪，Panda 窗口已贴到右侧画布")
        self.after(80, self._release_sim_start_gate)
        self.after(160, self._stop_live_start_scene)

    def _poll_preview_window_when_ready(
        self,
        proc: subprocess.Popen,
        geometry: tuple[int, int, int, int] | None,
        remaining: int = 120,
    ) -> None:
        if os.name != "nt" or geometry is None or remaining <= 0:
            return
        if proc.poll() is not None:
            return
        hwnd = self._find_panda_window_by_pid(int(proc.pid))
        if not hwnd:
            self.after(
                500,
                lambda p=proc, g=geometry, n=remaining - 1: self._poll_preview_window_when_ready(p, g, n),
            )
            return
        self._preview_hwnd = hwnd
        try:
            import ctypes
        except Exception:
            return
        x, y, w, h = self._scene_canvas_geometry()
        self._clear_scene_display()
        self._apply_borderless_window(ctypes.windll.user32, int(hwnd), x, y, w, h)

    def _poll_live_start_ready(
        self,
        proc: subprocess.Popen,
        ready_file: Path,
        remaining: int = 240,
    ) -> None:
        if proc.poll() is not None:
            if self._wait_home_scene_before_show:
                self.deiconify()
            self._append_log("[SCENE] Panda HOME 初始场景进程已退出")
            return
        if ready_file.exists():
            if self._wait_home_scene_before_show:
                self.deiconify()
                self.update_idletasks()
            self._set_status("Panda HOME 初始场景已就绪，可以开始语音输入")
            self.after(
                100,
                lambda p=proc: self._poll_preview_window_when_ready(p, self._scene_canvas_geometry()),
            )
            return
        if remaining <= 0:
            if self._wait_home_scene_before_show:
                self.deiconify()
            self._append_log("[SCENE] 等待 Panda HOME 初始场景超时，请查看外部终端输出")
            return
        self.after(250, lambda p=proc, r=ready_file, n=remaining - 1: self._poll_live_start_ready(p, r, n))

    def _pump_preview_output(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                text = line.rstrip()
                if text:
                    self._post_ui(lambda t=text: self._append_log(f"[SCENE] {t}"))
                time.sleep(0.02)
        finally:
            if proc is self._preview_proc:
                self._preview_proc = None
                self._preview_hwnd = None
                self._preview_kind = None

    def _stop_live_start_scene(self) -> None:
        proc = self._preview_proc
        ready_file = self._preview_ready_file
        self._preview_proc = None
        self._preview_hwnd = None
        self._preview_ready_file = None
        self._preview_kind = None
        if ready_file is not None:
            try:
                ready_file.unlink(missing_ok=True)
            except Exception:
                pass
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _release_sim_start_gate(self) -> None:
        start_file = self._sim_start_file
        proc = self._sim_proc
        if start_file is None or proc is None or proc.poll() is not None:
            return
        try:
            start_file.parent.mkdir(parents=True, exist_ok=True)
            start_file.write_text("start\n", encoding="utf-8")
            self._append_log("[SIM] UI 贴合完成，开始播放动画")
        except Exception as e:
            self._append_log(f"[SIM] 播放启动信号写入失败：{e}")

    def _hide_deferred_panda_window(self, hwnd: int) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes
        except Exception:
            return
        self._panda_deferred = True
        geometry = self._panda_geometry or self._scene_canvas_geometry()
        _x, _y, w, h = geometry
        user32 = ctypes.windll.user32
        try:
            # 规划中只挪到屏外，不要 ShowWindow(SW_HIDE)：Panda 正在加载网格时藏窗会卡死
            user32.SetWindowPos(int(hwnd), 0, -32000, -32000, int(w), int(h), 0x0004)
        except Exception:
            pass

    def _repeat_panda_reposition(self, remaining: int) -> None:
        if remaining <= 0:
            return
        proc = self._sim_proc
        hwnd = self._panda_hwnd
        if proc is None or hwnd is None or proc.poll() is not None or self._panda_deferred:
            return
        self._reposition_panda_window()
        self.after(250, lambda n=remaining - 1: self._repeat_panda_reposition(n))

    def _find_panda_window_by_pid(self, target_pid: int) -> int:
        if os.name != "nt":
            return 0
        try:
            import ctypes
            from ctypes import wintypes
        except Exception as e:
            self._append_log(f"[SIM] Win32 窗口定位不可用：{e}")
            return 0

        user32 = ctypes.windll.user32
        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        found = ctypes.c_void_p(0)

        def callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if int(pid.value) != int(target_pid):
                return True
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, 256)
            if "WRS Robot Planning" in title.value or "Panda" in title.value:
                found.value = int(hwnd)
                return False
            return True

        user32.EnumWindows(enum_proc(callback), 0)
        return int(found.value or 0)

    def _poll_panda_window_when_ready(
        self,
        proc: subprocess.Popen,
        geometry: tuple[int, int, int, int] | None,
        defer_show: bool = False,
        remaining: int = 240,
    ) -> None:
        if os.name != "nt" or geometry is None:
            self._release_sim_start_gate()
            return
        if remaining <= 0:
            self._append_log("[SIM] 未能定位 Panda 窗口，直接释放动画启动信号")
            self._release_sim_start_gate()
            return
        if proc.poll() is not None:
            return
        hwnd = self._find_panda_window_by_pid(int(proc.pid))
        if not hwnd:
            self.after(
                500,
                lambda p=proc, g=geometry, d=defer_show, n=remaining - 1: self._poll_panda_window_when_ready(p, g, d, n),
            )
            return

        self._panda_hwnd = hwnd
        self._panda_geometry = geometry
        if defer_show:
            self._panda_deferred = True
            if self._panda_ready_to_show:
                self._show_deferred_panda_window()
            else:
                self._hide_deferred_panda_window(hwnd)
            return

        self._repeat_panda_reposition(12)
        self._append_log("[SIM] Panda 窗口已移动到右侧画布")

    def _place_panda_window_when_ready(
        self,
        proc: subprocess.Popen,
        geometry: tuple[int, int, int, int] | None,
        defer_show: bool = False,
    ) -> None:
        self._post_ui(lambda p=proc, g=geometry, d=defer_show: self._poll_panda_window_when_ready(p, g, d))
        return
        if os.name != "nt":
            return
        if geometry is None:
            return
        try:
            import ctypes
            from ctypes import wintypes
        except Exception as e:
            self._post_ui(lambda msg=str(e): self._append_log(f"[SIM] Win32 窗口定位不可用：{msg}"))
            return

        user32 = ctypes.windll.user32
        target_pid = int(proc.pid)

        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        hwnd_found = ctypes.c_void_p(0)

        def find_window() -> int:
            hwnd_found.value = 0

            def callback(hwnd, _lparam):
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if int(pid.value) != target_pid:
                    return True
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, 256)
                if "WRS Robot Planning" in title.value or "Panda" in title.value:
                    hwnd_found.value = int(hwnd)
                    return False
                return True

            user32.EnumWindows(EnumWindowsProc(callback), 0)
            return int(hwnd_found.value or 0)

        for _ in range(240):
            if proc.poll() is not None:
                return
            hwnd = find_window()
            if hwnd:
                self._panda_hwnd = hwnd
                self._panda_geometry = geometry
                x, y, w, h = geometry
                if defer_show:
                    if self._panda_ready_to_show:
                        self._panda_deferred = True
                        self._post_ui(self._show_deferred_panda_window)
                        return
                    self._panda_deferred = True
                    self._post_ui(lambda h=hwnd: self._hide_deferred_panda_window(h))
                    return
                self._post_ui(lambda: self._repeat_panda_reposition(12))
                self._post_ui(lambda: self._append_log("[SIM] Panda 窗口已移动到右侧画布"))
                return
            time.sleep(0.5)

    def _apply_borderless_window(self, user32, hwnd: int, x: int, y: int, w: int, h: int) -> None:
        import ctypes
        from ctypes import wintypes

        GWL_STYLE = -16
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        WS_SYSMENU = 0x00080000
        SWP_NOZORDER = 0x0004
        SWP_FRAMECHANGED = 0x0020
        SWP_SHOWWINDOW = 0x0040

        try:
            if not user32.IsWindow(wintypes.HWND(int(hwnd))):
                return
        except Exception:
            return

        if hasattr(user32, "GetWindowLongPtrW") and hasattr(user32, "SetWindowLongPtrW"):
            get_long = user32.GetWindowLongPtrW
            set_long = user32.SetWindowLongPtrW
            long_ptr = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
        else:
            get_long = user32.GetWindowLongW
            set_long = user32.SetWindowLongW
            long_ptr = ctypes.c_long

        get_long.argtypes = [wintypes.HWND, ctypes.c_int]
        get_long.restype = long_ptr
        set_long.argtypes = [wintypes.HWND, ctypes.c_int, long_ptr]
        set_long.restype = long_ptr
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL

        hwnd = wintypes.HWND(int(hwnd))
        style = int(get_long(hwnd, GWL_STYLE))
        style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
        set_long(hwnd, GWL_STYLE, long_ptr(style))
        user32.SetWindowPos(
            hwnd,
            wintypes.HWND(0),
            int(x),
            int(y),
            int(w),
            int(h),
            SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
        )

    def _finish_worker_task(self, code: int) -> None:
        has_next = bool(self._sim_queue)
        self._worker_sim_active = False
        self._sim_proc = None
        self._panda_hwnd = None
        self._panda_geometry = None
        self._panda_deferred = False
        self._panda_ready_to_show = False
        start_file = self._sim_start_file
        self._sim_start_file = None
        if start_file is not None:
            try:
                start_file.unlink(missing_ok=True)
            except Exception:
                pass
        self._append_log(f"[SIM_WORKER] 当前任务结束 code={code}")
        if code != 0:
            if has_next:
                self._append_log("[SIM] 当前目标规划失败，跳过该目标并继续下一个")
            else:
                self._restore_scene_after_sim_failure()
        self._launch_next_queued_sim()

    def _pump_worker_output(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                text = line.rstrip()
                if not text:
                    continue
                if text.startswith("[worker] ready"):
                    self._sim_worker_ready = True
                    self._post_ui(lambda: self._append_log("[SIM_WORKER] 常驻仿真 worker 已就绪"))
                elif text.startswith("[worker] task-finished"):
                    code = 0 if "code=0" in text else 1
                    self._post_ui(lambda c=code: self._finish_worker_task(c))
                elif text.startswith("[worker]"):
                    self._post_ui(lambda t=text: self._append_log(f"[SIM_WORKER] {t}"))
                else:
                    self._post_ui(lambda t=text: self._append_log(f"[SIM] {t}"))
                    if "[sim] 动画就绪" in text:
                        self._post_ui(lambda: self._set_stage(4))
                        self._post_ui(lambda: self._set_status("仿真动画播放中…"))
                        self._post_ui(self._show_deferred_panda_window)
                time.sleep(0.02)
        finally:
            if proc is self._sim_worker_proc:
                self._sim_worker_proc = None
                self._sim_worker_ready = False
                if self._worker_sim_active:
                    self._post_ui(lambda: self._finish_worker_task(1))
                self._post_ui(lambda: self._append_log("[SIM_WORKER] 常驻仿真 worker 已退出，后续回退单次启动"))

    def _pump_sim_output(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                text = line.rstrip()
                if text:
                    self._post_ui(lambda t=text: self._append_log(f"[SIM] {t}"))
                    if self._sim_start_file is not None:
                        should_show_panda = "[sim] 动画就绪" in text
                    else:
                        should_show_panda = "[sim] 操作:" in text or "[preview] saved=" in text
                    if should_show_panda:
                        self._post_ui(lambda: self._set_stage(4))
                        self._post_ui(lambda: self._set_status("仿真动画播放中…"))
                        self._post_ui(self._show_deferred_panda_window)
                    time.sleep(0.02)
        finally:
            code = proc.wait()
            has_next = bool(self._sim_queue)
            self._panda_hwnd = None
            self._panda_geometry = None
            self._panda_deferred = False
            self._panda_ready_to_show = False
            start_file = self._sim_start_file
            self._sim_start_file = None
            if start_file is not None:
                try:
                    start_file.unlink(missing_ok=True)
                except Exception:
                    pass
            self._post_ui(lambda c=code: self._append_log(f"[SIM] 进程结束 exit={c}"))
            if code != 0:
                if has_next:
                    self._post_ui(lambda: self._append_log("[SIM] 当前目标规划失败，跳过该目标并继续下一个"))
                else:
                    self._post_ui(self._restore_scene_after_sim_failure)
            self._post_ui(self._launch_next_queued_sim)

    def _on_fail(self, err: str) -> None:
        self._busy = False
        self._append_log(f"[FAIL] {err}")
        self._set_stage(max(0, self._stage_index), failed=True)
        self._set_status("失败：请确认 Dify 已开，或穿透地址 / API Key 正确")
        messagebox.showerror("调用失败", err)

    def _on_missing_object(self, missing_tokens: list[str]) -> None:
        self._busy = False
        if missing_tokens:
            self._append_log(f"[CHECK] 未在当前场景中找到：{', '.join(missing_tokens)}")
        self._append_log("该零件不存在于该场景中，请重新下发指令")
        self._set_stage(2, failed=True)
        self._set_status("该零件不存在于该场景中，请重新下发指令")

    def on_save(self) -> None:
        if not self._last_delivery:
            messagebox.showinfo("提示", "还没有结果可保存")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="delivery_agent_auto.json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        Path(path).write_text(
            json.dumps(self._last_delivery, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._set_status(f"已保存：{path}")

    def on_open_dir(self) -> None:
        import os

        out_dir = DEFAULT_OUT.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(out_dir))

    def destroy(self) -> None:
        self._closing = True
        self._sim_queue.clear()
        self._panda_deferred = False
        self._panda_geometry = None
        self._panda_ready_to_show = False
        if self._recording:
            try:
                self._stop_recording()
            except Exception:
                pass
        if self._panda_reposition_job is not None:
            try:
                self.after_cancel(self._panda_reposition_job)
            except Exception:
                pass
            self._panda_reposition_job = None
        self._stop_live_start_scene()
        proc = self._sim_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        worker_proc = self._sim_worker_proc
        if worker_proc is not None and worker_proc is not proc and worker_proc.poll() is None:
            try:
                worker_proc.terminate()
                worker_proc.wait(timeout=2)
            except Exception:
                try:
                    worker_proc.kill()
                except Exception:
                    pass
        super().destroy()


def main() -> None:
    app = AgentDemoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
