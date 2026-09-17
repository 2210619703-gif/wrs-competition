# -*- coding: utf-8 -*-
"""谷歌语音转写（speech_recognition + Google Web Speech，中文 zh-CN）。

Google 接口在国内要代理。Python 自己的 urllib **不会**去读 Windows 系统代理，
Clash 开了「系统代理」也不够，必须把混合端口写进 ``fruit_config.STT_PROXY``。

    pip install SpeechRecognition
"""
from __future__ import annotations

import io
import os
import socket
import wave
from pathlib import Path
from urllib.parse import urlparse

from fruit_config import STT_PROXY


class GoogleSttError(RuntimeError):
    pass


_PROXY_READY = False


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def _proxy_url() -> str:
    raw = (os.environ.get("FRUIT_STT_PROXY") or STT_PROXY or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    return raw


def apply_stt_proxy() -> str:
    """给 urllib / speech_recognition 装上 Clash 代理。进程内只装一次。"""
    global _PROXY_READY
    url = _proxy_url()
    if _PROXY_READY:
        return url
    if url:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ[key] = url
        # 本机回环不要绕进代理死循环
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        extra = "127.0.0.1,localhost"
        os.environ["NO_PROXY"] = extra if not no_proxy else f"{no_proxy},{extra}"
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
        try:
            import urllib.request

            handler = urllib.request.ProxyHandler({"http": url, "https": url})
            urllib.request.install_opener(urllib.request.build_opener(handler))
        except Exception:
            pass
    _PROXY_READY = True
    return url


def _proxy_listening(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 80)
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _google_reachable() -> str:
    """空字符串表示通了，否则是给用户看的原因。"""
    import urllib.error
    import urllib.request

    url = apply_stt_proxy()
    if url and not _proxy_listening(url):
        parsed = urlparse(url)
        return (
            f"代理 {parsed.hostname}:{parsed.port} 没在听。"
            f"打开 Clash，看「混合代理端口」是不是这个数，改 fruit_config.STT_PROXY"
        )
    try:
        urllib.request.urlopen(
            "https://www.google.com/generate_204", timeout=5
        )
        return ""
    except Exception as e:
        hint = f"走代理 {url}" if url else "没配代理"
        return (
            f"谷歌语音连不上（{hint}）：{e}\n"
            "Clash 要开着，规则别把 google 走 DIRECT；端口写在 fruit_config.STT_PROXY"
        )


def transcribe_pcm16(pcm: bytes, sample_rate: int = 16000) -> str:
    if not pcm:
        raise GoogleSttError("没有录到声音")
    apply_stt_proxy()
    try:
        import speech_recognition as sr
    except ImportError as e:
        raise GoogleSttError(
            "谷歌语音需要：pip install SpeechRecognition\n"
            "并保证能经 Clash 访问 Google。"
        ) from e
    unreachable = _google_reachable()
    if unreachable:
        raise GoogleSttError(unreachable)
    recognizer = sr.Recognizer()
    audio = sr.AudioData(pcm, int(sample_rate), 2)
    try:
        text = recognizer.recognize_google(audio, language="zh-CN")
    except sr.UnknownValueError as e:
        raise GoogleSttError("没听清，请重试") from e
    except sr.RequestError as e:
        raise GoogleSttError(
            f"谷歌语音接口失败（需要外网 / Clash）：{e}\n"
            f"当前代理 {apply_stt_proxy() or '未设置'}，改 fruit_config.STT_PROXY"
        ) from e
    text = str(text or "").strip(" \t\r\n。.?？,，!！")
    if not text:
        raise GoogleSttError("没听清，请重试")
    return text


def transcribe_wav(path: str | Path) -> str:
    apply_stt_proxy()
    try:
        import speech_recognition as sr
    except ImportError as e:
        raise GoogleSttError("谷歌语音需要：pip install SpeechRecognition") from e
    unreachable = _google_reachable()
    if unreachable:
        raise GoogleSttError(unreachable)
    recognizer = sr.Recognizer()
    with sr.AudioFile(str(path)) as src:
        audio = recognizer.record(src)
    try:
        text = recognizer.recognize_google(audio, language="zh-CN")
    except sr.UnknownValueError as e:
        raise GoogleSttError("没听清，请重试") from e
    except sr.RequestError as e:
        raise GoogleSttError(f"谷歌语音接口失败（需要外网 / Clash）：{e}") from e
    return str(text or "").strip()
