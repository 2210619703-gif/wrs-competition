"""
第③点闭环桥接：执行后视觉回传 → 校验 → 决策 → 修正任务序列
仿真调用方式（推荐，不改 Dify 图也能用）：
  vision_objects JSON 顶层带：
  {
    "mode": "reflect",
    "last_task": { ... 上一步 place 任务或整包 task ... },
    "memory": { ... 可选 ... },
    "post_vision_objects": { "objects": [ ... ] },
    "objects": []
  }
"""
from __future__ import annotations

import json

from closed_loop import decide_next, init_memory, post_vision_check
from common_config import STORAGE_BOX_NAME, STORAGE_BOX_XYZ
from task_sequence import build_recovery_sequence, build_transport_sequence


def _as_dict(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        return json.loads(text)
    return {}


def extract_place_task(last_task: dict) -> dict:
    """从整包 task 里取出最近一次 place 子步骤，供校验。"""
    last_task = last_task or {}
    if last_task.get("action") == "place":
        return last_task
    tasks = last_task.get("tasks") or []
    for t in reversed(tasks):
        if isinstance(t, dict) and t.get("action") == "place":
            return t
    # 兜底：用整包当放置目标
    return {
        "action": "place",
        "object": last_task.get("object", ""),
        "coordinate": last_task.get("dest_coordinate")
        or last_task.get("coordinate")
        or STORAGE_BOX_XYZ,
        "dest_coordinate": last_task.get("dest_coordinate") or STORAGE_BOX_XYZ,
        "destination": last_task.get("destination") or STORAGE_BOX_NAME,
        "rpy": last_task.get("rpy") or [0, 0, 0],
        "original_cmd": last_task.get("original_cmd") or "",
    }


def run_reflect(
    *,
    last_task: dict,
    post_vision_objects,
    memory: dict | None = None,
    user_cmd: str = "",
) -> dict:
    """
    返回：
      flow_tag: next_task | retry | transport | final_fail
      check / decision / memory
      task: 下一步可执行任务（retry 时为 recovery；transport 时为搬运；成功可为空或 next hint）
    """
    memory = memory or init_memory()
    last_task = _as_dict(last_task)
    place_task = extract_place_task(last_task)
    check = post_vision_check(place_task, post_vision_objects, memory)
    decision = decide_next(check, memory, place_task)
    flow = decision["flow_tag"]
    mem = decision["updated_memory"]

    next_task = None
    if flow == "retry":
        pick_xyz = list(
            last_task.get("coordinate")
            or place_task.get("coordinate")
            or [0.0, 0.0, 0.0]
        )
        pick_rpy = list(last_task.get("rpy") or place_task.get("rpy") or [0, 0, 0])
        dest = list(
            place_task.get("dest_coordinate")
            or last_task.get("dest_coordinate")
            or STORAGE_BOX_XYZ
        )
        next_task = build_recovery_sequence(
            object_name=place_task.get("object") or last_task.get("object") or "",
            pick_xyz=pick_xyz,
            pick_rpy=pick_rpy,
            dest_xyz=dest,
            user_cmd=user_cmd or last_task.get("original_cmd") or "",
            fault_type=check.get("fault_type") or "wrong_pose",
            x_off=decision.get("x_off", 0.0),
            yaw_off=decision.get("yaw_off", 0.0),
        )
        next_task["flow_tag"] = "retry"
        next_task["from_check"] = check
    elif flow == "transport":
        next_task = build_transport_sequence(
            user_cmd=user_cmd or "帮我搬运一下料箱盒"
        )
        next_task["flow_tag"] = "transport"
        next_task["from_check"] = check
    elif flow == "next_task":
        next_task = {
            "task_id": "t_next",
            "action": "next_task",
            "object": "",
            "status": "pending",
            "reason": "本步成功，继续下一子任务/下一件",
            "original_cmd": user_cmd or last_task.get("original_cmd") or "",
            "tasks": [],
            "flow_tag": "next_task",
            "from_check": check,
        }
    else:
        next_task = {
            "task_id": "t_fail",
            "action": "stop",
            "object": place_task.get("object", ""),
            "status": "failed",
            "reason": f"超过重试上限: {check.get('msg', '')}",
            "original_cmd": user_cmd or "",
            "tasks": [],
            "flow_tag": "final_fail",
            "from_check": check,
        }

    return {
        "task_id": "t_reflect",
        "action": "reflect",
        "object": place_task.get("object", ""),
        "status": "pending" if flow != "final_fail" else "failed",
        "reason": check.get("msg", ""),
        "original_cmd": user_cmd or last_task.get("original_cmd") or "",
        "mode": "reflect",
        "flow_tag": flow,
        "check": check,
        "decision": {
            "flow_tag": flow,
            "x_off": decision.get("x_off", 0.0),
            "yaw_off": decision.get("yaw_off", 0.0),
        },
        "memory": mem,
        "updated_memory": mem,
        "task": next_task,
        "tasks": next_task.get("tasks") or [],
        "task_sequence_desc": next_task.get("task_sequence_desc")
        or [f"reflect → {flow}"],
    }


def is_reflect_payload(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if str(data.get("mode") or "").lower() == "reflect":
        return True
    if data.get("post_vision_objects") and data.get("last_task"):
        return True
    return False
