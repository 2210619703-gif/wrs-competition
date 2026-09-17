# -*- coding: utf-8 -*-
"""水果真机演示界面：眼在手上 D405 + 谷歌语音 + 仿真规划器 + servo_j。

布局模仿 ``tiaozhanbei/real/real_demo_ui.py``：

    左侧  录音 / 口令 / 到观察位·零位·A框·B框 / 日志
    右侧  D405 画面
    底部  转写 → YOLO → 仿真规划 → 执行

用法::

    python demo/fruit_demo_ui.py

示教位、是否下发、servo_j 都在 ``fruit_config.py``。
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import numpy as np

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from fruit_cmd import describe_cmd, parse_fruit_cmd
from fruit_config import (
    CAMERA_RESOLUTION,
    DEFAULT_MOVE_SPEED,
    EXECUTE_ON_REAL,
    HOME_ON_EXIT,
    MAX_MOVE_SPEED,
    REAL_DIR,
    RELEASE_MODE_ON_CLOSE,
    RELEASE_MODES,
    REPO_ROOT,
    SNAPSHOT_FRAMES,
    USE_SERVO_J,
    BIN_A_CONF_DEG,
    BIN_B_CONF_DEG,
    LOOK_CONF_DEG,
    HANDEYE_JSON,
    conf_is_filled,
)
from fruit_real import connect_arm, go_named, run_task
from fruit_vision import describe_arm_state, load_handeye_raw
from fruit_stt import GoogleSttError, transcribe_pcm16
from fruit_tts import shutdown as tts_shutdown
from fruit_tts import speak

if REAL_DIR not in sys.path:
    sys.path.insert(0, REAL_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from d405_camera import RESOLUTIONS, CameraFrame, D405Camera  # noqa: E402
from real_arm import RealArm, RealArmError  # noqa: E402
from real_config import RealConfigError, load_real_config  # noqa: E402

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

BG = "#101722"
PANEL = "#121C29"
INK = "#D7E2EC"
MUTED = "#8EA0B5"
ACCENT = "#1F6F8B"
ACCENT_HOVER = "#245F77"
DANGER = "#A8553A"
BORDER = "#263447"
OK = "#2A9D8F"

REFERENCE_VOICE = "把苹果放到A框 / 把橙子放到左边 / 往左 往上 / 随便动动"


def audio_level_stats(audio) -> tuple[int, float]:
    x = np.squeeze(np.asarray(audio)).astype("float32")
    if x.size == 0:
        return 0, 0.0
    return int(np.max(np.abs(x))), float(np.sqrt(np.mean(x * x)))


class CameraWorker(threading.Thread):
    def __init__(self, args, logger) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.log = logger
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._ready = threading.Event()
        self._error = ""
        self._snap_req = threading.Event()
        self._snap_done = threading.Event()
        self._snap_out = None

    def wait_ready(self, timeout: float = 20.0) -> bool:
        return self._ready.wait(timeout)

    def latest(self):
        with self._lock:
            return self._latest

    def grab_stable(self, timeout: float = 20.0):
        self._snap_out = None
        self._snap_done.clear()
        self._snap_req.set()
        if not self._snap_done.wait(timeout):
            raise RuntimeError("抓帧超时")
        kind, payload = self._snap_out or ("err", RuntimeError("抓帧失败"))
        if kind != "ok":
            raise payload if isinstance(payload, Exception) else RuntimeError(str(payload))
        return payload

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            with D405Camera(self.args.resolution, self.args.serial, logger=self.log) as cam:
                while not self._stop.is_set():
                    if self._snap_req.is_set():
                        try:
                            snap = cam.capture_median(self.args.frames)
                            # RealSense 缓冲下一帧会复用，必须拷走，否则 YOLO 吃到空图。
                            self._snap_out = (
                                "ok",
                                CameraFrame(
                                    color=np.ascontiguousarray(snap.color.copy()),
                                    depth_m=np.ascontiguousarray(snap.depth_m.copy()),
                                    points=np.ascontiguousarray(snap.points.copy()),
                                    intrinsics=dict(snap.intrinsics),
                                ),
                            )
                        except Exception as e:
                            self._snap_out = ("err", e)
                        self._snap_req.clear()
                        self._snap_done.set()
                        continue
                    try:
                        frame = cam.capture(timeout_ms=2000)
                    except Exception as e:
                        self._error = str(e)
                        self.log(f"[cam] {e}")
                        break
                    with self._lock:
                        self._latest = CameraFrame(
                            color=np.ascontiguousarray(frame.color.copy()),
                            depth_m=np.ascontiguousarray(frame.depth_m.copy()),
                            points=np.ascontiguousarray(frame.points.copy()),
                            intrinsics=dict(frame.intrinsics),
                        )
                    self._ready.set()
        except Exception as e:
            self._error = str(e)
            self._ready.set()
            self.log(f"[cam] {e}")


class FruitDemoApp(tk.Tk):
    def __init__(self, args, cfg=None, arm=None) -> None:
        super().__init__()
        self.args = args
        self.cfg = cfg
        self._arm = arm
        self.title("真机")
        self._view_size = RESOLUTIONS.get(args.resolution, (848, 480))
        vw, vh = self._view_size
        self.geometry(f"{max(1040, vw + 470)}x{max(650, vh + 220)}")
        self.minsize(max(1040, vw + 380), max(650, vh + 180))
        self.configure(bg=BG)

        self._busy = False
        self._closing = False
        self._recording = False
        self._rec_chunks: queue.Queue = queue.Queue()
        self._rec_stream = None
        self._ui_events: queue.Queue = queue.Queue()
        self._preview_photo = None
        self._detect_photo = None
        self._stage_names = ["录音结束", "谷歌转写", "YOLO识别", "仿真规划", "机械臂执行"]
        self._stage_index = -1
        self._stage_failed = False
        self.cam = CameraWorker(args, self._cam_log)

        self._setup_style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self._drain_ui)
        self.after(100, self._draw_progress)
        self.after(150, self._boot)

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
        style.configure("Title.TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED)
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#0F1720",
            background=ACCENT,
            bordercolor=BORDER,
        )
        style.configure("Accent.TButton", background=ACCENT, foreground="#FFFFFF", padding=(12, 7))
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])
        style.configure("Danger.TButton", background=DANGER, foreground="#FFFFFF", padding=(12, 7))
        style.configure("Card.TCheckbutton", background=PANEL, foreground=INK, focuscolor=PANEL)

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=(14, 6))
        ttk.Label(header, text="真机", style="Title.TLabel").pack(side="left")
        self.status = ttk.Label(header, text="", style="Sub.TLabel")
        self.status.pack(side="right", fill="x", expand=True, padx=(16, 0))

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=18, pady=(6, 10))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left_card = tk.Frame(main, width=380, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        left_card.grid_propagate(False)
        left_card.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        left = ttk.Frame(left_card, style="Card.TFrame")
        left.pack(fill="both", expand=True, padx=14, pady=14)

        self.mic_btn = ttk.Button(left, text="开始录音（谷歌）", style="Accent.TButton", command=self.on_mic)
        self.mic_btn.pack(fill="x")
        self.rec_label = ttk.Label(left, text="说完再点一次结束", style="Muted.TLabel")
        self.rec_label.pack(anchor="w", pady=(8, 4))
        self.mic_meter = ttk.Progressbar(left, maximum=100, mode="determinate")
        self.mic_meter.pack(fill="x")
        ttk.Label(left, text=REFERENCE_VOICE, style="Muted.TLabel", wraplength=330, justify="left").pack(
            anchor="w", pady=(8, 8)
        )

        pose = ttk.Frame(left, style="Card.TFrame")
        pose.pack(fill="x", pady=(0, 4))
        for text, mode in (("观察位", "look"), ("零位", "zero"), ("A框", "bin_a"), ("B框", "bin_b")):
            ttk.Button(pose, text=text, command=lambda m=mode: self.on_go(m)).pack(
                side="left", expand=True, fill="x", padx=2
            )
        slots = ttk.Frame(left, style="Card.TFrame")
        slots.pack(fill="x", pady=(0, 8))
        for text, mode in (("左边空位", "slot_left"), ("右边空位", "slot_right")):
            ttk.Button(slots, text=text, command=lambda m=mode: self.on_go(m)).pack(
                side="left", expand=True, fill="x", padx=2
            )
        ttk.Button(left, text="读关节角 / 法兰坐标", command=self.on_read_joints).pack(
            fill="x", pady=(0, 8)
        )

        ctrl = ttk.Frame(left, style="Card.TFrame")
        ctrl.pack(fill="x", pady=(0, 8))
        self.execute_var = tk.BooleanVar(value=bool(EXECUTE_ON_REAL))
        ttk.Checkbutton(
            ctrl, text="允许真机执行（不勾只 dry-run）", variable=self.execute_var, style="Card.TCheckbutton"
        ).pack(anchor="w")
        row = ttk.Frame(ctrl, style="Card.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="速度", style="Muted.TLabel").pack(side="left")
        self.speed_var = tk.IntVar(value=min(self.args.speed, MAX_MOVE_SPEED))
        tk.Spinbox(
            row,
            from_=1,
            to=MAX_MOVE_SPEED,
            width=4,
            textvariable=self.speed_var,
            font=("SimHei", 18, "bold"),
            fg="#F8FAFC",
            bg="#0B1220",
            insertbackground="#F8FAFC",
            buttonbackground=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            relief="flat",
            justify="center",
            bd=0,
        ).pack(side="left", padx=8)
        ttk.Button(ctrl, text="急停", style="Danger.TButton", command=self.on_estop).pack(fill="x", pady=(8, 0))

        ttk.Label(left, text="终端输出", style="Muted.TLabel").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            left, width=52, height=18, wrap="word", font=("Microsoft YaHei UI", 9),
            bg="#0F1720", fg=INK, insertbackground=INK, relief="flat",
        )
        self.log.pack(fill="both", expand=True, pady=(6, 0))

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        self.canvas = tk.Canvas(right, bg="#0A1018", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.hint = ttk.Label(right, text="相机画面", style="Muted.TLabel")
        self.hint.pack(anchor="w", pady=(6, 0))

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=18, pady=(0, 14))
        self.stage_vars = []
        for name in self._stage_names:
            cell = ttk.Frame(bottom)
            cell.pack(side="left", expand=True, fill="x", padx=4)
            bar = ttk.Progressbar(cell, maximum=100, mode="determinate")
            bar.pack(fill="x")
            lab = ttk.Label(cell, text=name, style="Muted.TLabel")
            lab.pack()
            self.stage_vars.append((bar, lab))

    def _boot(self) -> None:
        try:
            from fruit_stt import apply_stt_proxy

            apply_stt_proxy()
        except Exception:
            pass
        self.cam.start()
        self.after(200, self._tick_preview)
        self._start_console()
        if self._arm is not None and conf_is_filled(LOOK_CONF_DEG) and self.execute_var.get():
            try:
                go_named(self._arm, "look")
            except Exception as e:
                self._append_log(f"[real] 去观察位失败：{e}")
        self._prompt_voice()

    def _prompt_voice(self) -> None:
        self._append_log("请输入语音")
        self._set_status("请输入语音")

    def _cam_log(self, text: str) -> None:
        msg = str(text)
        print(msg, flush=True)
        if any(k in msg for k in ("失败", "打不开", "超时", "占用", "ERROR", "Error")):
            self.log_async(msg)

    def _start_console(self) -> None:
        if not getattr(sys.stdin, "isatty", lambda: False)():
            return
        threading.Thread(target=self._console_loop, daemon=True).start()

    def _console_loop(self) -> None:
        time.sleep(0.4)
        print("\n[fruit] 指令> ", end="", flush=True)
        while not self._closing:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            text = line.strip()
            if not text:
                print("[fruit] 指令> ", end="", flush=True)
                continue
            if text.lower() in {"q", "quit", "exit"}:
                self._post(self.on_close)
                break
            self._submit(text)
            print("[fruit] 指令> ", end="", flush=True)

    def _submit(self, cmd: str) -> None:
        if self._busy or self._recording:
            print("[fruit] 正忙，等一下", flush=True)
            return
        self._busy = True
        self._post(lambda: self._append_log(f"[TERM] {cmd}"))
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd: str) -> None:
        try:
            self._pipeline(cmd)
        except Exception as e:
            self._post(lambda m=str(e): self._fail(m))

    def _pipeline(self, cmd: str) -> None:
        parsed = parse_fruit_cmd(cmd)
        if self._arm is None or self.cfg is None:
            raise RuntimeError("机械臂未连接")
        execute = bool(self.execute_var.get())
        self._arm.speed = int(self.speed_var.get())
        if parsed.get("action") in ("jog", "wander"):
            self._post(self._clear_detect_photo)
            self._post(lambda: self._set_stage(3))
            self._post(lambda: self._set_status(describe_cmd(parsed)))
            run_task(self._arm, self.cfg, cmd, execute=execute)
            self._post(lambda: self._set_stage(4))
            self._post(self._done)
            return
        self._post(self._clear_detect_photo)
        self._post(lambda: self._set_stage(2))
        self._post(lambda: self._set_status("新抓拍识别 + 仿真规划…"))

        def on_preview(img):
            photo = np.asarray(img).copy()
            self._post(lambda p=photo: self._set_detect_photo(p))

        summary = run_task(
            self._arm,
            self.cfg,
            cmd,
            execute=execute,
            grab=self.cam.grab_stable,
            on_preview=on_preview,
        )
        if summary.get("preview") is not None:
            photo = np.asarray(summary["preview"]).copy()
            self._post(lambda p=photo: self._set_detect_photo(p))
        self._post(lambda: self._set_stage(4))
        self._post(self._done)

    def on_go(self, mode: str) -> None:
        if self._busy:
            messagebox.showinfo("提示", "任务还在跑")
            return
        if self._arm is None:
            messagebox.showerror("未连接", "机械臂没连上")
            return
        if not bool(self.execute_var.get()):
            self._append_log(f"[real] dry-run：不会去 {mode}。勾选允许真机执行。")
            return

        labels = {
            "look": "观察位",
            "zero": "零位",
            "bin_a": "A框",
            "bin_b": "B框",
            "slot_left": "左边空位",
            "slot_right": "右边空位",
        }
        zh = labels.get(mode, mode)

        def work():
            try:
                self._post(self._clear_detect_photo)
                speak(f"正在去{zh}")
                go_named(self._arm, mode)
                self.log_async(f"[real] 已到 {mode}")
                speak(f"已到{zh}")
            except Exception as e:
                self.log_async(f"[real] {e}")
                speak("去指定位置失败")

        threading.Thread(target=work, daemon=True).start()

    def on_read_joints(self) -> None:
        if self._arm is None:
            messagebox.showerror("未连接", "机械臂没连上，无法读关节角")
            return
        try:
            q = self._arm.joint_values()
            r2cam = None
            if os.path.isfile(HANDEYE_JSON):
                mat = load_handeye_raw()
                if not np.allclose(mat, np.eye(4)):
                    r2cam = mat
            text = describe_arm_state(q, r2cam=r2cam)
            self._append_log("[real] " + text.replace("\n", "\n[real] "))
        except Exception as e:
            self._append_log(f"[real] 读关节失败：{e}")
            messagebox.showerror("读关节失败", str(e))

    def on_mic(self) -> None:
        if self._recording:
            self._stop_rec()
        else:
            self._start_rec()

    def _start_rec(self) -> None:
        if self._busy:
            messagebox.showinfo("提示", "正在跑任务")
            return
        try:
            import sounddevice as sd
        except ImportError as e:
            messagebox.showerror("缺少依赖", f"pip install sounddevice SpeechRecognition\n{e}")
            return
        self._rec_chunks = queue.Queue()

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            self._rec_chunks.put(indata.copy())

        try:
            self._rec_stream = sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=callback)
            self._rec_stream.start()
        except Exception as e:
            messagebox.showerror("麦克风失败", str(e))
            return
        self._recording = True
        self._rec_t0 = time.time()
        self.mic_btn.configure(text="结束并识别", style="Danger.TButton")
        self._tick_rec()

    def _tick_rec(self) -> None:
        if not self._recording:
            return
        self.rec_label.configure(text=f"录音中 {int(time.time() - self._rec_t0):02d}s")
        self.after(200, self._tick_rec)

    def _stop_rec(self) -> None:
        self._recording = False
        self.mic_btn.configure(text="开始录音（谷歌）", style="Accent.TButton")
        if self._rec_stream is not None:
            try:
                self._rec_stream.stop()
                self._rec_stream.close()
            except Exception:
                pass
            self._rec_stream = None
        chunks = []
        while not self._rec_chunks.empty():
            chunks.append(self._rec_chunks.get())
        if not chunks:
            messagebox.showwarning("提示", "没有录到声音")
            return
        self._busy = True
        self._set_stage(0)
        threading.Thread(target=self._worker_mic, args=(chunks,), daemon=True).start()

    def _worker_mic(self, chunks) -> None:
        try:
            audio = np.concatenate(chunks, axis=0)
            peak, rms = audio_level_stats(audio)
            self.log_async(f"[MIC] peak={peak} rms={rms:.1f}")
            pcm = np.squeeze(audio).astype("int16").tobytes()
            self._post(lambda: self._set_stage(1))
            text = transcribe_pcm16(pcm, 16000)
            self.log_async(f"[STT] {text}（Google zh-CN）")
            self._pipeline(text)
        except GoogleSttError as e:
            self._post(lambda m=str(e): self._fail(m))
        except Exception as e:
            self._post(lambda m=str(e): self._fail(m))

    def _set_detect_photo(self, img) -> None:
        self._detect_photo = None if img is None else np.asarray(img).copy()

    def _clear_detect_photo(self) -> None:
        self._detect_photo = None

    def _tick_preview(self) -> None:
        if self._closing:
            return
        img = self._detect_photo
        if img is None:
            frame = self.cam.latest()
            if frame is not None:
                img = frame.color
        if img is not None and Image is not None:
            rgb = np.asarray(img)[:, :, ::-1]
            pil = Image.fromarray(rgb)
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            pil.thumbnail((cw, ch))
            self._preview_photo = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._preview_photo)
            if self._detect_photo is not None:
                self.canvas.create_text(
                    12,
                    12,
                    text="本次检测（新抓拍）",
                    fill="#F5E6A3",
                    anchor="nw",
                    font=("Microsoft YaHei", 11),
                )
        self.after(40, self._tick_preview)

    def _set_stage(self, i: int, failed: bool = False) -> None:
        self._stage_index = i
        self._stage_failed = failed
        self._draw_progress()

    def _draw_progress(self) -> None:
        for i, (bar, _lab) in enumerate(self.stage_vars):
            if failed := (self._stage_failed and i == self._stage_index):
                bar["value"] = 100
            elif i < self._stage_index:
                bar["value"] = 100
            elif i == self._stage_index:
                bar["value"] = 60
            else:
                bar["value"] = 0

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def log_async(self, text: str) -> None:
        self._post(lambda t=text: self._append_log(t))

    def _post(self, fn) -> None:
        self._ui_events.put(fn)

    def _drain_ui(self) -> None:
        while True:
            try:
                fn = self._ui_events.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                pass
        if not self._closing:
            self.after(50, self._drain_ui)

    def _done(self) -> None:
        self._busy = False
        self._prompt_voice()
        speak("任务完成")

    def _fail(self, err: str) -> None:
        self._busy = False
        self._append_log(f"[FAIL] {err}")
        self._set_stage(max(0, self._stage_index), failed=True)
        self._set_status("失败")
        speak("任务失败")
        self._prompt_voice()

    def on_estop(self) -> None:
        if self._arm is None:
            return
        try:
            self._arm.estop()
            self._append_log("[real] 已急停")
            speak("已急停", important=True)
        except Exception as e:
            self._append_log(f"[real] 急停失败：{e}")

    def on_close(self) -> None:
        self._closing = True
        tts_shutdown()
        self.cam.stop()
        arm = self._arm
        self._arm = None
        if arm is not None:
            try:
                if HOME_ON_EXIT:
                    from real_config import EXIT_HOME_CONF

                    arm.go_conf(EXIT_HOME_CONF, label="静置位", open_jaw=True)
            except Exception:
                try:
                    arm.estop()
                except Exception:
                    pass
            try:
                arm.close()
            except Exception:
                pass
        self.destroy()


def parse_args():
    p = argparse.ArgumentParser(description="水果真机 UI")
    p.add_argument("--resolution", default=CAMERA_RESOLUTION, choices=tuple(RESOLUTIONS))
    p.add_argument("--serial", default=None)
    p.add_argument("--frames", type=int, default=SNAPSHOT_FRAMES)
    p.add_argument("--speed", type=int, default=DEFAULT_MOVE_SPEED)
    p.add_argument("--port", default=None)
    p.add_argument("--release-mode", default=RELEASE_MODE_ON_CLOSE, choices=RELEASE_MODES)
    p.add_argument("--no-connect", action="store_true", help="不连机械臂，只看相机")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = None
    arm = None
    if not args.no_connect:
        try:
            cfg = load_real_config()
            arm = connect_arm(
                cfg,
                execute=True,
                speed=args.speed,
                port=args.port,
                logger=print,
            )
            print("[fruit] 机械臂已连接（界面里不勾执行就不会动抓取）")
        except (RealConfigError, RealArmError, Exception) as e:
            print(f"[fruit] 机械臂未连接：{e}")
            print("[fruit] 仍打开界面；可加 --no-connect 跳过")
            cfg, arm = None, None
    app = FruitDemoApp(args, cfg=cfg, arm=arm)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
