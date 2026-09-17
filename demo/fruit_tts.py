# -*- coding: utf-8 -*-
"""中文语音播报。走本机 Windows SAPI，不走谷歌。

只保留「当前步骤」那一句：新步骤到来就打断上一句，避免机械臂已经在动
还在说「正在规划」。pyttsx3 在 Windows 上 ``runAndWait`` 容易卡住，所以
只用 SAPI / PowerShell。
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time

from fruit_config import TTS_ENABLED, dest_label

_SVS_ASYNC = 1
_SVS_PURGE = 2
_SPRS_SPEAKING = 2

_QUEUE: queue.Queue = queue.Queue(maxsize=8)
_WORKER: threading.Thread | None = None
_LOCK = threading.Lock()
_STOP = threading.Event()
_READY = threading.Event()
_SEQ = 0


def _pick_voice(spk) -> str:
    keys = ("huihui", "yaoyao", "kangkang", "chinese", "zh-cn", "zh_cn", "中文")
    try:
        voices = spk.GetVoices()
        for i in range(int(voices.Count)):
            v = voices.Item(i)
            try:
                desc = str(v.GetDescription())
            except Exception:
                desc = ""
            if any(k in desc.lower() or k in desc for k in keys):
                spk.Voice = v
                return desc
    except Exception:
        pass
    return ""


def _latest_seq() -> int:
    with _LOCK:
        return _SEQ


def _drain_queue() -> None:
    while True:
        try:
            _QUEUE.get_nowait()
            _QUEUE.task_done()
        except queue.Empty:
            return


def _purge_spk(spk) -> None:
    if spk is None:
        return
    try:
        spk.Speak("", _SVS_PURGE)
    except Exception:
        pass


def _speak_powershell(text: str) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try { $s.SelectVoiceByHints(2, 30, 0, [Globalization.CultureInfo]'zh-CN') } catch {};"
        "$s.Speak([Console]::In.ReadToEnd())"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=40,
        check=False,
    )


def _speak_sapi(spk, text: str, seq: int) -> None:
    """异步说，被更新的步骤打断就停。"""
    spk.Speak(str(text), _SVS_ASYNC | _SVS_PURGE)
    deadline = time.time() + 0.5
    speaking = False
    while time.time() < deadline:
        if seq != _latest_seq():
            _purge_spk(spk)
            return
        try:
            if int(spk.Status.RunningState) == _SPRS_SPEAKING:
                speaking = True
                break
        except Exception:
            break
        time.sleep(0.02)
    if not speaking:
        return
    while True:
        if seq != _latest_seq():
            _purge_spk(spk)
            return
        try:
            if int(spk.Status.RunningState) != _SPRS_SPEAKING:
                return
        except Exception:
            return
        time.sleep(0.04)


def _worker_loop() -> None:
    spk = None
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        spk = win32com.client.Dispatch("SAPI.SpVoice")
        name = _pick_voice(spk)
        try:
            spk.Rate = 3
        except Exception:
            pass
        print(f"[tts] 引擎 SAPI  {name or '默认音色'}")
    except Exception as e:
        print(f"[tts] SAPI 不可用，改用 PowerShell：{e}")
        spk = None
    _READY.set()

    while not _STOP.is_set():
        try:
            item = _QUEUE.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        seq, text = item
        text = str(text).strip()
        if not text or seq != _latest_seq():
            _QUEUE.task_done()
            continue
        try:
            if spk is not None:
                _speak_sapi(spk, text, seq)
            else:
                _speak_powershell(text)
        except Exception as e:
            print(f"[tts] 播报失败：{e}  改用 PowerShell")
            try:
                if seq == _latest_seq():
                    _speak_powershell(text)
            except Exception as e2:
                print(f"[tts] PowerShell 也失败：{e2}")
        _QUEUE.task_done()


def _ensure_worker() -> None:
    global _WORKER
    with _LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _STOP.clear()
        _READY.clear()
        _WORKER = threading.Thread(target=_worker_loop, daemon=True, name="fruit-tts")
        _WORKER.start()
    _READY.wait(timeout=5.0)


def speak(text: str, *, important: bool = False) -> None:
    """播报当前步骤。新句子会清掉还没说的，并打断正在说的上一句。

    important 仍给急停用，效果相同。
    """
    global _SEQ
    text = " ".join(str(text or "").split())
    if not text:
        return
    print(f"[tts] {text}")
    if not TTS_ENABLED:
        return
    _ensure_worker()
    with _LOCK:
        _SEQ += 1
        seq = _SEQ
        _drain_queue()
    try:
        _QUEUE.put_nowait((seq, text))
    except queue.Full:
        print("[tts] 队列满，丢掉这一句")


def shutdown() -> None:
    global _SEQ
    with _LOCK:
        _SEQ += 1
        _drain_queue()
    _STOP.set()
    try:
        _QUEUE.put_nowait(None)
    except queue.Full:
        pass


def announce_action(targets: list[dict], dest: str | None, *, execute: bool) -> str:
    names = [str(o.get("class") or "水果") for o in targets] or ["水果"]
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    parts = [f"{n}个{k}" if n > 1 else k for k, n in counts.items()]
    obj = "、".join(parts)
    if dest:
        line = f"现在把{obj}放到{dest_label(dest)}"
    else:
        line = f"现在抓取{obj}"
    if not execute:
        speak("这是预演，机械臂不会动")
        speak(line)
    # 真机等手臂真正开始这一段时再报，避免「开始执行」堵在运动后面。
    return line


def announce_segment(seg: dict, dest: str | None) -> None:
    name = str(seg.get("object") or "水果")
    if dest:
        speak(f"现在把{name}放到{dest_label(dest)}")
    else:
        speak(f"现在抓取{name}")
