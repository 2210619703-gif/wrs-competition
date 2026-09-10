# -*- coding: utf-8 -*-
"""国内语音转写：阿里 FunASR（魔搭 paraformer-zh），本地推理，不走 Google / Clash。

第一次会从 modelscope.cn 拉模型，之后离线可用。
"""
from __future__ import annotations

import re
import tempfile
import wave
from pathlib import Path

_MODEL = None
_SENSEVOICE_TAGS = re.compile(r"<\|[^|]*\|>")
_SPACE = re.compile(r"\s+")


class CnSttError(RuntimeError):
    """转写失败或还没装 FunASR。"""


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from funasr import AutoModel
    except ImportError as e:
        raise CnSttError(
            "国内语音识别需要 FunASR：pip install -U funasr modelscope kaldi-native-fbank\n"
            "装好后第一次转写会从魔搭下载中文模型，不走 Clash。"
        ) from e
    try:
        import kaldi_native_fbank  # noqa: F401
    except ImportError:
        try:
            import torchaudio  # noqa: F401
        except ImportError as e:
            raise CnSttError(
                "FunASR 还缺特征提取后端。已装 FunASR 的话再装：pip install kaldi-native-fbank"
            ) from e
    _MODEL = AutoModel(
        model="paraformer-zh",
        disable_update=True,
        disable_pbar=True,
        device="cpu",
    )
    return _MODEL


def _clean_text(raw: str) -> str:
    text = _SENSEVOICE_TAGS.sub("", raw or "")
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = rich_transcription_postprocess(text)
    except Exception:
        pass
    return _SPACE.sub("", text).strip(" \t\r\n。.?？,，!！")


def _text_from_result(res) -> str:
    if not res:
        return ""
    item = res[0] if isinstance(res, (list, tuple)) else res
    if isinstance(item, dict):
        raw = item.get("text") or ""
    else:
        raw = str(item)
    return _clean_text(raw)


def transcribe_wav(path: str | Path) -> str:
    """识别一段 wav，返回中文指令。"""
    model = _load_model()
    res = model.generate(input=str(path), cache={})
    text = _text_from_result(res)
    if not text:
        raise CnSttError("没听清，请重试")
    return text


def transcribe_pcm16(pcm: bytes, sample_rate: int = 16000) -> str:
    """识别 16-bit PCM（单声道）。"""
    if not pcm:
        raise CnSttError("没有录到声音")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm)
        return transcribe_wav(wav_path)
    finally:
        Path(wav_path).unlink(missing_ok=True)
