"""
决策模块入口：文字 / 语音指令 → 任务序列 JSON。

优先调用本机 Dify（http://localhost:8080/v1/workflows/run）；
失败或未配置时走 offline/build_tasks_local 本地规划。
视觉默认读 simple_dataset.json；螺丝组可用 --sample 0502 或 ec0001（dataset_learn/0502）。
输出默认：../tasks/delivery_agent_auto.json。

用法见 使用说明.md。示例：
  python agent/demo_voice_agent.py --cmd "拿取轴承-凸轮滚轮放到收纳盒" --sample 0001
  python agent/demo_voice_agent.py --mic --sample 0001
  python agent/demo_voice_agent.py --showcase --sample 0001
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

_DECISION = Path(__file__).resolve().parent.parent
if str(_DECISION) not in sys.path:
    sys.path.insert(0, str(_DECISION))

from agent.run_batch import (  # noqa: E402
    API_KEY,
    API_URL,
    call_workflow,
    extract_task_from_outputs,
)

try:
    from agent.run_batch import local_fallback as _local_fallback
except ImportError:
    _local_fallback = None

ROOT = _DECISION
# 与仿真共用 tiaozhanbei/dataset_learn
RAW_DATASET = ROOT.parent / "dataset_learn"
SIMPLE_DATASET = ROOT / "simple_dataset.json"
EC_DATASET = ROOT / "simple_dataset_ec.json"
DEFAULT_OUT = ROOT.parent / "tasks" / "delivery_agent_auto.json"

# 对齐项目书 / 答疑：拿取放置、区域装箱、搬运、多智能体
SHOWCASE_CMDS = [
    "拿取轴承-凸轮滚轮放到收纳盒",  # perceive→pick→move→place
    "给我一个轴承-凸轮滚轮",  # 只拿取
    "随便拿3个零件放到收纳盒",
    "把轴承和螺丝放到收纳盒",
    "把所有零件放进收纳盒",
    "帮我把零件最多的区域装箱",
    "帮我搬运一下料箱盒",
    "帮我把所有零件装箱",
    "把12个螺丝依次放到对应格子",
]


def load_simple_samples() -> list[dict]:
    if not SIMPLE_DATASET.exists():
        raise SystemExit(
            f"找不到 {SIMPLE_DATASET.name}，请先运行：python offline/extract_dataset.py"
        )
    return json.loads(SIMPLE_DATASET.read_text(encoding="utf-8"))


def load_vision_by_sample(sample_id: str) -> tuple[dict, str]:
    """优先 simple_dataset；螺丝组 ec0001；否则读原始 annotations.json。"""
    sid = sample_id.zfill(4) if sample_id.isdigit() else sample_id
    if sid.lower() in ("ec", "ec0001", "ec_0001", "0001ec", "ec-0001", "ec/0001"):
        sid = "0502"

    if EC_DATASET.exists():
        for item in json.loads(EC_DATASET.read_text(encoding="utf-8")):
            if str(item.get("sample_id")) in (sid, "ec0001", "0502"):
                return item.get("vision_objects") or {}, "0502"

    for item in load_simple_samples():
        if str(item.get("sample_id")) == sid:
            vo = item.get("vision_objects") or {}
            return vo, sid

    ann_path = RAW_DATASET / sid / "annotations.json"
    if ann_path.exists():
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        storage = (ann.get("scene_layout") or {}).get("storage_box") or {}
        xyz = list(storage.get("pos_m") or [0.3, -0.2, 0.0])
        while len(xyz) < 3:
            xyz.append(0.0)
        vo = {
            "objects": ann.get("objects") or [],
            "target_container": {
                "xyz": xyz[:3],
                "name": "料盘" if sid == "0502" else "收纳盒",
            },
            "scene_layout": ann.get("scene_layout") or {},
        }
        return vo, sid

    raise SystemExit(f"找不到样本 {sid}（不在 simple_dataset，也没有原始 annotations）")


def load_vision_from_path(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "vision_objects" in data:
        return data["vision_objects"]
    if "objects" in data:
        return data
    raise SystemExit(f"无法从 {path} 解析视觉 JSON")


def rehydrate_tasks(task: dict) -> dict:
    """Dify 深度限制会把完整步骤放进 tasks_json；优先还原更长的那份。"""
    raw = task.get("tasks_json")
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            parsed = None
    existing = task.get("tasks") if isinstance(task.get("tasks"), list) else []
    n = int(task.get("tasks_count") or 0)
    if isinstance(parsed, list) and parsed:
        if (not existing) or (n and len(existing) < n) or (len(parsed) > len(existing)):
            task = dict(task)
            task["tasks"] = parsed
    return task


def audio_level_stats(audio) -> tuple[int, float]:
    """Return peak/rms for int16 microphone buffers."""
    import numpy as np

    x = np.squeeze(np.asarray(audio)).astype("float32")
    if x.size == 0:
        return 0, 0.0
    peak = int(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    return peak, rms


def speech_to_text_from_mic(seconds: float = 5.0) -> str:
    try:
        import numpy as np
        import sounddevice as sd
        from agent.cn_stt import CnSttError, transcribe_pcm16
    except ImportError as e:
        raise SystemExit(
            "麦克风模式需要：pip install -r requirements.txt\n"
            f"缺少依赖：{e}"
        ) from e

    rate = 16000
    print(f"[MIC] 请说指令（约 {seconds:.0f} 秒）…")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    peak, rms = audio_level_stats(audio)
    print(f"[MIC] 音量 peak={peak} rms={rms:.1f}")
    if peak < 200 or rms < 30:
        raise SystemExit(
            "麦克风录到的声音太小，请检查 Windows 输入设备/输入音量，或靠近麦克风再试"
        )
    pcm = np.squeeze(audio).astype("int16").tobytes()
    try:
        return transcribe_pcm16(pcm, rate)
    except CnSttError as e:
        raise SystemExit(str(e)) from e


def speech_to_text_from_file(audio_path: Path) -> str:
    from agent.cn_stt import CnSttError, transcribe_wav

    if not audio_path.exists():
        raise SystemExit(f"找不到音频文件：{audio_path}")

    path = audio_path
    if path.suffix.lower() != ".wav":
        path = _to_wav(audio_path)
    try:
        return transcribe_wav(path)
    except CnSttError as e:
        raise SystemExit(str(e)) from e


def _to_wav(src: Path) -> Path:
    """
    m4a/mp3 等：优先用 imageio-ffmpeg 自带的 ffmpeg 转成 wav；
    若已是 soundfile 可读格式则直接读。
    """
    out = src.with_suffix(".stt_tmp.wav")
    suffix = src.suffix.lower()

    # m4a/aac/mp3/ogg 走 ffmpeg（soundfile 通常不认 m4a）
    if suffix in {".m4a", ".aac", ".mp3", ".ogg", ".wma", ".flac"}:
        _ffmpeg_to_wav(src, out)
        return out

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise SystemExit(
            f"非 wav 需要：pip install -r requirements.txt\n{e}"
        ) from e

    try:
        data, rate = sf.read(str(src), always_2d=True)
    except Exception:
        _ffmpeg_to_wav(src, out)
        return out

    mono = data.mean(axis=1)
    mono_i16 = (np.clip(mono, -1.0, 1.0) * 32767).astype("int16")
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(mono_i16.tobytes())
    return out


def _ffmpeg_to_wav(src: Path, out: Path) -> None:
    import subprocess

    ffmpeg = None
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    if not ffmpeg:
        raise SystemExit(
            "当前是 .m4a 等格式，需要先转成 wav，或安装转换库后重试：\n"
            "  pip install imageio-ffmpeg\n"
            "然后重新执行刚才的 python agent/demo_voice_agent.py --audio ...\n"
            "也可手机/在线工具把录音另存为 .wav 再传入。"
        )

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="ignore")[:400]
        raise SystemExit(f"音频转换失败：{err}") from e


def resolve_user_cmd(args: argparse.Namespace) -> str:
    if args.mic:
        text = speech_to_text_from_mic(args.seconds)
        print(f"[STT] {text}")
        return text
    if args.audio:
        text = speech_to_text_from_file(args.audio)
        print(f"[STT] {text}")
        return text
    if args.cmd:
        return args.cmd.strip()

    print("请输入指令（或输入 mic 开麦，直接回车用默认抓取指令）")
    typed = input("指令> ").strip()
    if typed.lower() == "mic":
        text = speech_to_text_from_mic(args.seconds)
        print(f"[STT] {text}")
        return text
    return typed or "抓取指定工业零件"


def run_agent(
    user_cmd: str,
    vision: dict,
    *,
    sample_id: str,
    allow_fallback: bool,
) -> dict:
    print(f"[API] {API_URL}")
    print(f"[API] key={API_KEY[:12]}...")
    print(f"[SAMPLE] {sample_id}")
    print(f"[CMD] {user_cmd}")
    try:
        resp = call_workflow(user_cmd, vision)
        data = resp.get("data") or {}
        task, branch = extract_task_from_outputs(
            data.get("outputs") or {},
            status=data.get("status"),
            error=data.get("error"),
        )
        task = rehydrate_tasks(task)
        task["source"] = f"dify_api:{branch}"
        return task
    except Exception as e:
        if not allow_fallback or _local_fallback is None:
            raise
        print(f"[WARN] Dify 失败，本地兜底：{e}")
        task = _local_fallback(user_cmd, vision, sample_id)
        return rehydrate_tasks(task)


def summarize(task: dict) -> None:
    actions = [t.get("action") for t in (task.get("tasks") or [])] or task.get(
        "actions_flat"
    )
    print("[OK] object=", task.get("object"))
    print("[OK] action=", task.get("action"))
    print("[OK] actions=", actions)
    print("[OK] dest=", task.get("dest_coordinate") or task.get("destination"))
    print("[OK] source=", task.get("source"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="指令(文字/语音) → Dify API → 仿真 JSON（组长自动化要求）"
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--cmd", type=str, help="文字指令")
    g.add_argument("--mic", action="store_true", help="麦克风说指令")
    g.add_argument("--audio", type=Path, help="从音频转写指令")
    g.add_argument(
        "--showcase",
        action="store_true",
        help="跑一组典型指令，写入多条交付（给组长演示）",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="0001",
        help="组长数据集样本号，默认 0001",
    )
    parser.add_argument("--vision", type=Path, default=None, help="覆盖：自定义视觉 JSON")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Dify 失败不要本地兜底",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.vision:
        vision = load_vision_from_path(args.vision)
        sample_id = args.sample
    else:
        vision, sample_id = load_vision_by_sample(args.sample)

    allow_fallback = not args.no_fallback

    if args.showcase:
        delivery = []
        for i, cmd in enumerate(SHOWCASE_CMDS, 1):
            print(f"\n======== showcase {i}/{len(SHOWCASE_CMDS)} ========")
            task = run_agent(
                cmd, vision, sample_id=sample_id, allow_fallback=allow_fallback
            )
            summarize(task)
            delivery.append(
                {
                    "sample_id": f"{sample_id}_sc{i:02d}",
                    "user_cmd": cmd,
                    "task": task,
                }
            )
        args.output.write_text(
            json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[OK] showcase 共 {len(delivery)} 条 → {args.output.resolve()}")
        print("把这个 JSON 交给仿真即可。")
        return 0

    user_cmd = resolve_user_cmd(args)
    task = run_agent(
        user_cmd, vision, sample_id=sample_id, allow_fallback=allow_fallback
    )
    summarize(task)

    delivery = {
        "sample_id": sample_id,
        "user_cmd": user_cmd,
        "task": task,
    }
    args.output.write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] wrote= {args.output.resolve()}")
    print("把这个 JSON 交给仿真即可（Python 已调 API，无需网页复制）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
