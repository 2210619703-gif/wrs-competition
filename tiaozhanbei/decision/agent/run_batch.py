"""批量调用决策工作流；Dify 失败时用本地规划兜底。

环境变量 DIFY_API_URL / DIFY_API_KEY 优先；否则读决策模块根目录 dify_api_key.txt，
默认 URL 为 http://localhost:8080/v1/workflows/run。
全量交付也可直接：python offline/build_tasks_local.py
详见 使用说明.md。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests

_PKG = Path(__file__).resolve().parent.parent  # 决策模块根目录
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

try:
    from offline.build_tasks_local import build_pick_task, make_natural_cmd, to_parts
except ImportError:
    build_pick_task = None
    make_natural_cmd = None
    to_parts = None

# ========== Dify 配置 ==========
# 必须写到 /v1/workflows/run；不要只写到 /v1
# 终端 / UI 若已设 DIFY_API_URL / DIFY_API_KEY，会覆盖默认值
# 默认走本机 Docker Dify（8080）；也可用 dify_api_key.txt
def _load_api_key() -> str:
    env = (os.getenv("DIFY_API_KEY") or "").strip()
    if env:
        return env
    key_file = _PKG / "dify_api_key.txt"
    if key_file.is_file():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("app-"):
                return line
    return ""


API_URL = os.getenv("DIFY_API_URL", "http://localhost:8080/v1/workflows/run")
API_KEY = _load_api_key()
TIME_OUT = 180
# ==================================

DATASET_FILE = str(_PKG / "simple_dataset.json")
FULL_OUTPUT_FILE = str(_PKG / "batch_output_result.json")
DELIVERY_FILE = str(_PKG / "delivery_for_simulation.json")


def _as_vision_input(vision_objects):
    """Dify 段落输入需要字符串；若已是 str 则不要再 dumps（否则双重转义会 500）。"""
    if vision_objects is None:
        return ""
    if isinstance(vision_objects, str):
        return vision_objects
    return json.dumps(vision_objects, ensure_ascii=False)


def call_workflow(user_cmd, vision_objects):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": {
            "user_cmd": user_cmd,
            "vision_objects": _as_vision_input(vision_objects),
        },
        "response_mode": "blocking",
        "user": "local_test_user_001",
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIME_OUT)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Dify 服务连接失败：{e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"请求失败，状态码 {resp.status_code}，返回：{resp.text[:800]}")
    return resp.json()


def _coerce_task_dict(value):
    """结束节点有时把 object 序列化成 JSON 字符串。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def extract_task_from_outputs(outputs, *, status=None, error=None):
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError(f"outputs 不是 dict：{type(outputs)}")

    for key in ("task_data", "pick_task", "fail_task"):
        if key not in outputs:
            continue
        task = _coerce_task_dict(outputs[key])
        if task is not None:
            return task, key

    hint = ""
    if not outputs:
        hint = (
            f"；status={status!r} error={error!r}。"
            "常见原因：① Dify 应用未发布 ② 结束节点未勾选输出变量 "
            "task_data/fail_task ③ API Key 绑错应用"
        )
    raise KeyError(
        f"未知输出格式，可用字段：{list(outputs.keys())}{hint}"
    )

def is_valid_3d_coord(coord):
    return (
        isinstance(coord, list)
        and len(coord) == 3
        and all(isinstance(v, (int, float)) for v in coord)
    )


def is_task_failed(task_result):
    if not isinstance(task_result, dict):
        return True
    if task_result.get("status") in ("failed", "error"):
        return True
    if not task_result.get("object"):
        return True
    if not is_valid_3d_coord(task_result.get("coordinate")):
        return True
    reason = str(task_result.get("reason", ""))
    if "json format error" in reason:
        return True
    return False


def local_fallback(user_cmd, vision_objects, sample_id):
    if build_pick_task is None:
        raise RuntimeError(
            "缺少 offline/build_tasks_local.py，无法本地兜底。请向同学要完整包，或保证 Dify 能连上。"
        )
    row = build_pick_task(user_cmd, vision_objects, sample_id)
    task = row["task"]
    task["source"] = "local_fallback"
    return task


def resolve_user_cmd(item, vision_objects, *, smart_cmd: bool) -> str:
    """
    数据集里默认全是「抓取指定工业零件」→ 智能体会选最大件。
    smart_cmd=True 时按场景零件生成多样指令（指定名称/放置/装箱/搬运），
    再用 Python 调 Dify，行为才像智能体。
    """
    raw = (item.get("user_cmd") or "").strip()
    if not smart_cmd:
        return raw or "抓取指定工业零件"
    # 已是具体指令则保留；笼统抓取指令则改写
    if raw and raw not in ("抓取指定工业零件", "抓取零件", "拿取零件"):
        return raw
    if make_natural_cmd is None or to_parts is None:
        return raw or "抓取指定工业零件"
    parts = to_parts(vision_objects)
    return make_natural_cmd(item.get("sample_id", "0001"), parts, raw)


def batch_run(*, limit: int | None = None, smart_cmd: bool = True):
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if limit is not None and limit > 0:
        samples = samples[:limit]

    all_result = []
    delivery = []
    fallback_count = 0
    total = len(samples)

    for index, item in enumerate(samples):
        current_num = index + 1
        sample_id = item.get("sample_id", f"{current_num:04d}")
        vision_objects = item.get("vision_objects")

        if vision_objects is None and "objects" in item:
            vision_objects = {
                "objects": item["objects"],
                "target_container": {"xyz": [0.0, 0.0, 0.0]},
            }

        user_cmd = resolve_user_cmd(item, vision_objects, smart_cmd=smart_cmd)
        print(f"\n======== 第 {current_num}/{total} 组 ({sample_id}) ========")
        print(f"指令: {user_cmd}")

        try:
            workflow_return = call_workflow(user_cmd, vision_objects)
            data = workflow_return.get("data") or {}
            outputs = data.get("outputs") or {}
            task_result, output_key = extract_task_from_outputs(
                outputs, status=data.get("status"), error=data.get("error")
            )
            used_fallback = False

            if is_task_failed(task_result):
                task_result = local_fallback(user_cmd, vision_objects, sample_id)
                used_fallback = True
                fallback_count += 1
                output_key = "local_fallback"

            # 补全 tasks（Dify 深度限制时常空；勿用被截成单件的短 tasks 覆盖）
            raw_json = task_result.get("tasks_json")
            existing = task_result.get("tasks") if isinstance(task_result.get("tasks"), list) else []
            n = int(task_result.get("tasks_count") or 0)
            if raw_json:
                try:
                    parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, list) and parsed:
                    if (not existing) or (n and len(existing) < n) or (len(parsed) > len(existing)):
                        task_result = dict(task_result)
                        task_result["tasks"] = parsed

            record = {
                "样本序号": current_num,
                "sample_id": sample_id,
                "user_cmd": user_cmd,
                "input_vision_objects": vision_objects,
                "parsed_task": task_result,
                "output_branch": output_key,
                "used_local_fallback": used_fallback,
                "full_dify_response": workflow_return,
            }
            all_result.append(record)
            delivery.append(
                {
                    "sample_id": sample_id,
                    "user_cmd": user_cmd,
                    "task": task_result,
                }
            )

            actions = [t.get("action") for t in (task_result.get("tasks") or [])] or (
                task_result.get("actions_flat")
            )
            print(
                f"[OK] object={task_result.get('object')} | "
                f"action={task_result.get('action')} | "
                f"steps={actions} | branch={output_key}"
            )
        except Exception as e:
            print(f"[FAIL] Dify 失败，使用本地兜底：{e}")
            task_result = local_fallback(user_cmd, vision_objects, sample_id)
            fallback_count += 1
            all_result.append(
                {
                    "样本序号": current_num,
                    "sample_id": sample_id,
                    "user_cmd": user_cmd,
                    "input_vision_objects": vision_objects,
                    "parsed_task": task_result,
                    "output_branch": "local_fallback",
                    "used_local_fallback": True,
                    "error": str(e),
                }
            )
            delivery.append(
                {
                    "sample_id": sample_id,
                    "user_cmd": user_cmd,
                    "task": task_result,
                }
            )

    with open(FULL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_result, f, ensure_ascii=False, indent=2)

    with open(DELIVERY_FILE, "w", encoding="utf-8") as f:
        json.dump(delivery, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 批量完成：{len(delivery)}/{total} 条（本地兜底 {fallback_count} 条）")
    print(f"   完整结果：{FULL_OUTPUT_FILE}")
    print(f"   仿真交付：{DELIVERY_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python 调 Dify 智能体批量出任务 JSON（多样指令，不是只抓最大件）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只跑前 N 条（0=全部 500）",
    )
    parser.add_argument(
        "--raw-cmd",
        action="store_true",
        help="不用智能指令改写，沿用数据集原文（几乎全是抓取指定工业零件）",
    )
    args = parser.parse_args()
    batch_run(limit=args.limit or None, smart_cmd=not args.raw_cmd)
