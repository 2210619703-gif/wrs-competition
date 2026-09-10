"""
第③点闭环：记忆 / 后置视觉校验 / 决策重规划
对齐搭档 yml：初始化记忆、后置视觉校验、决策规划中枢
"""
from __future__ import annotations

BOX_CAPACITY = 6  # 演示用：装满 6 件即搬运（仿真可改）


def init_memory(max_retry: int = 3, box_capacity: int = BOX_CAPACITY) -> dict:
    return {
        "finished_parts": [],
        "fail_record": [],
        "current_retry": 0,
        "max_retry": max_retry,
        "box_full": False,
        "box_capacity": box_capacity,
    }


def post_vision_check(task_data: dict, post_vision_objects: dict | str, memory: dict | None = None) -> dict:
    """
    执行一步后：用新视觉与预期比对。
    返回 check_status: success | need_retry
    """
    import json

    if isinstance(post_vision_objects, str):
        new_vision = json.loads(post_vision_objects) if post_vision_objects.strip() else {}
    else:
        new_vision = post_vision_objects or {}

    target_name = task_data.get("object", "")
    coord = list(task_data.get("coordinate") or [0, 0, 0])
    rpy = list(task_data.get("rpy") or [0, 0, 0])
    expect_x, expect_y = float(coord[0]), float(coord[1])
    expect_yaw = float(rpy[2] if len(rpy) > 2 else 0.0)

    # 放置任务：期望点在 destination
    action = str(task_data.get("action") or "")
    if action in ("place", "move") and task_data.get("dest_coordinate"):
        dest = list(task_data["dest_coordinate"])
        expect_x, expect_y = float(dest[0]), float(dest[1])

    real_obj = None
    for obj in new_vision.get("objects", []):
        if obj.get("class") == target_name:
            real_obj = obj
            break

    if real_obj is None:
        return {
            "check_status": "need_retry",
            "fault_type": "part_fall",
            "msg": "零件掉落/未检测到，需重新抓取",
        }

    real_state = real_obj.get("pose_state") or real_obj.get("state") or "normal"
    # 兼容中文标签 + example_bin（inverted/fallen）
    _aliases = {
        "normal": "normal",
        "正常": "normal",
        "upside_down": "upside_down",
        "倒放": "upside_down",
        "倒置": "upside_down",
        "inverted": "upside_down",
        "tilt": "tilt",
        "tilted": "tilt",
        "倾倒": "tilt",
        "倾斜": "tilt",
        "fallen": "tilt",
    }
    rs = str(real_state).strip()
    real_state = _aliases.get(rs) or _aliases.get(rs.lower()) or "normal"

    pose = real_obj.get("pose_6d") or {}
    real_x = float(pose.get("x", 0))
    real_y = float(pose.get("y", 0))
    real_yaw = float(pose.get("yaw", 0))

    if real_state in ("tilt", "upside_down"):
        return {
            "check_status": "need_retry",
            "fault_type": "wrong_pose",
            "msg": f"零件姿态异常({real_state})，需调整后重放",
        }

    x_err = abs(expect_x - real_x)
    y_err = abs(expect_y - real_y)
    yaw_err = abs(expect_yaw - real_yaw)
    if yaw_err > 3.14159265:
        yaw_err = abs(yaw_err - 2 * 3.14159265)

    if x_err > 0.015 or y_err > 0.015:
        return {
            "check_status": "need_retry",
            "fault_type": "pos_offset",
            "msg": "放置坐标偏移，需修正后重试",
        }
    if yaw_err > 0.15:
        return {
            "check_status": "need_retry",
            "fault_type": "angle_offset",
            "msg": "放置角度歪斜，需调整姿态",
        }

    return {"check_status": "success", "fault_type": "", "msg": "执行成功"}


def decide_next(check_res: dict, memory: dict, task_data: dict) -> dict:
    """
    决策中枢：success→下一件；need_retry→修正偏移并重试；超限→final_fail。
    box 装满 → box_full=True（上层输出搬运任务）。
    """
    memory = memory or init_memory()
    max_r = int(memory.get("max_retry", 3))
    curr_r = int(memory.get("current_retry", 0))
    capacity = int(memory.get("box_capacity", BOX_CAPACITY))
    status = check_res.get("check_status", "")
    fault = check_res.get("fault_type", "")

    new_memory = {
        "finished_parts": list(memory.get("finished_parts") or []),
        "fail_record": list(memory.get("fail_record") or []),
        "current_retry": curr_r,
        "max_retry": max_r,
        "box_full": bool(memory.get("box_full", False)),
        "box_capacity": capacity,
    }

    x_off = 0.0
    yaw_off = 0.0

    if status == "success":
        obj = task_data.get("object", "")
        if obj:
            new_memory["finished_parts"].append(obj)
        new_memory["current_retry"] = 0
        if len(new_memory["finished_parts"]) >= capacity:
            new_memory["box_full"] = True
            flow_tag = "transport"
        else:
            flow_tag = "next_task"
    elif status == "need_retry":
        if curr_r >= max_r:
            flow_tag = "final_fail"
        else:
            new_memory["current_retry"] = curr_r + 1
            new_memory["fail_record"].append(
                {"part": task_data.get("object", ""), "fault": fault}
            )
            flow_tag = "retry"
            if fault == "pos_offset":
                x_off, yaw_off = 0.006, 0.0
            elif fault == "wrong_pose":
                x_off, yaw_off = 0.0, 0.12
            elif fault == "part_fall":
                x_off, yaw_off = 0.01, 0.0
            elif fault == "angle_offset":
                x_off, yaw_off = 0.0, 0.08
    else:
        flow_tag = "final_fail"

    return {
        "flow_tag": flow_tag,
        "x_off": x_off,
        "yaw_off": yaw_off,
        "updated_memory": new_memory,
        "new_memory": new_memory,  # Dify 输出变量名
        "check_res": check_res,
    }
