import json
import re

STORAGE_BOX_NAME = "收纳盒"
STORAGE_BOX_XYZ = [0.30, -0.20, 0.0]
PLACE_KEYWORDS = ("放到", "放入", "放置", "放进", "移到", "移动到", "装箱")
TRANSPORT_KEYWORDS = ("搬运", "搬走", "移送料箱", "搬料箱")
HIGH_ACTIONS = (
    "pack_region",
    "transport",
    "multi_agent_pack",
    "reflect",
    "pack_bin_slots",
    "pack_n_parts",
    "pack_all_box",
    "multi_named_pick_place",
    "pack_batch",
)


def shallow_task_for_dify(task: dict) -> dict:
    """把深层 tasks/batch 收成 JSON 字符串，避免 Dify Depth limit 把步骤吞掉或报错。"""
    if not isinstance(task, dict):
        return task
    out = dict(task)
    tasks = out.get("tasks")
    if isinstance(tasks, list) and tasks:
        out["tasks_json"] = json.dumps(tasks, ensure_ascii=False)
        out["actions_flat"] = [t.get("action") for t in tasks if isinstance(t, dict)]
        out["tasks_count"] = len(tasks)
        out["tasks"] = []
    batch = out.get("batch")
    if batch is not None:
        out["batch_json"] = json.dumps(batch, ensure_ascii=False)
        out.pop("batch", None)
    agents = out.get("agents")
    if agents is not None:
        out["agents_json"] = json.dumps(agents, ensure_ascii=False)
        out.pop("agents", None)
    return out


def wants_transport(cmd: str) -> bool:
    text = cmd or ""
    if any(k in text for k in TRANSPORT_KEYWORDS):
        return True
    if "搬运" in text and ("料箱" in text or "盒子" in text or "盒" in text):
        return True
    return False


def wants_pack_region(cmd: str) -> bool:
    text = cmd or ""
    if "最多" in text and ("区域" in text or "区" in text):
        return True
    if "装箱" in text and ("区域" in text or "最多" in text):
        return True
    return False


def wants_place(cmd: str) -> bool:
    text = cmd or ""
    if wants_transport(text):
        return False
    if any(k in text for k in PLACE_KEYWORDS):
        return True
    if "收纳盒" in text or "料箱" in text or "格子" in text:
        return True
    return False


def intent_do_place(cmd: str, destination=None) -> bool:
    text = (cmd or "").strip()
    if wants_transport(text) or wants_pack_region(text):
        return True
    if text:
        return wants_place(text)
    dest = str(destination or "").strip()
    if dest in ("", "null", "None", "无"):
        return False
    if dest == STORAGE_BOX_NAME or "收纳" in dest or "料箱" in dest or "格子" in dest:
        return True
    return False


def ensure_sequence(task: dict, user_cmd: str = "") -> dict:
    cmd = (user_cmd or task.get("original_cmd", "") or "").strip()
    if cmd:
        task["original_cmd"] = cmd

    action = str(task.get("action") or "")
    tasks = task.get("tasks")
    # reflect / 高层任务：即使 tasks=[]（next_task、final_fail）也原样保留
    if action in HIGH_ACTIONS:
        task["original_cmd"] = cmd
        if not isinstance(tasks, list):
            task["tasks"] = []
        else:
            for t in tasks:
                if isinstance(t, dict):
                    t["original_cmd"] = cmd
        return task
    if isinstance(tasks, list) and tasks:
        actions = [t.get("action") for t in tasks if isinstance(t, dict)]
        if "transport" in actions or "adjust_pose" in actions:
            for t in tasks:
                if isinstance(t, dict):
                    t["original_cmd"] = cmd
            task["original_cmd"] = cmd
            return task
        if "place" in actions and len(tasks) >= 4:
            if not task.get("destination") and intent_do_place(cmd, task.get("destination")):
                task["destination"] = STORAGE_BOX_NAME
                task["dest_coordinate"] = list(STORAGE_BOX_XYZ)
            task["original_cmd"] = cmd
            for t in tasks:
                if isinstance(t, dict):
                    t["original_cmd"] = cmd
            return task
        if (not intent_do_place(cmd, task.get("destination"))) and actions == [
            "perceive",
            "pick",
        ]:
            task["destination"] = None
            task["dest_coordinate"] = None
            task["original_cmd"] = cmd
            for t in tasks:
                if isinstance(t, dict):
                    t["original_cmd"] = cmd
                    t["destination"] = None
                    t["dest_coordinate"] = None
            return task

    do_place = intent_do_place(cmd, task.get("destination"))
    raw_dest = task.get("destination")
    pose_state = task.get("pose_state") or task.get("state") or "normal"
    _pose_aliases = {
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
    ps = str(pose_state).strip()
    pose_state = _pose_aliases.get(ps) or _pose_aliases.get(ps.lower()) or "normal"
    task["pose_state"] = pose_state

    obj = task.get("object", "")
    pick_xyz = list(task.get("coordinate", [0.0, 0.0, 0.0]))
    rpy = list(task.get("rpy") or [0.0, 0.0, float(task.get("angle", 0.0))])
    angle = round(float(task.get("angle", rpy[2] if len(rpy) > 2 else 0.0)), 6)
    max_retry = int(task.get("max_retry", 3))
    reason = task.get("reason", "")

    if do_place:
        dest_name = raw_dest or STORAGE_BOX_NAME
        if dest_name in ("", "null", "None"):
            dest_name = STORAGE_BOX_NAME
            dest = list(STORAGE_BOX_XYZ)
        else:
            dest = list(task.get("dest_coordinate") or STORAGE_BOX_XYZ)
            if dest_name == STORAGE_BOX_NAME:
                dest = list(STORAGE_BOX_XYZ)
    else:
        dest_name = None
        dest = None

    perceive = {
        "task_id": "t000",
        "step": 1,
        "action": "perceive",
        "object": obj,
        "coordinate": pick_xyz,
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": angle,
        "rpy": rpy,
        "retry": 0,
        "max_retry": max_retry,
        "status": "pending",
        "reason": "识别场景目标并定位；判断正常/倒放/倾倒",
        "original_cmd": cmd,
        "pose_state": pose_state,
    }
    pick = {
        "task_id": "t001",
        "step": 2,
        "action": "pick",
        "object": obj,
        "coordinate": pick_xyz,
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": angle,
        "rpy": rpy,
        "retry": 0,
        "max_retry": max_retry,
        "status": "pending",
        "reason": reason,
        "original_cmd": cmd,
    }

    task["destination"] = dest_name
    task["dest_coordinate"] = dest
    task["action"] = task.get("action") or "pick"
    task["original_cmd"] = cmd
    task["pose_state"] = pose_state

    if not do_place:
        task["slot_id"] = None
        task["tasks"] = [perceive, pick]
        task["task_sequence_desc"] = [
            "1.perceive 识别定位目标",
            "2.pick 抓取目标零件（仅拿起，不放置）",
        ]
        return task

    seq = [perceive, pick]
    step = 2
    place_rpy = list(rpy)
    if pose_state in ("upside_down", "tilt"):
        step += 1
        seq.append(
            {
                "task_id": f"t{step:03d}",
                "step": step,
                "action": "adjust_pose",
                "object": obj,
                "coordinate": pick_xyz,
                "dest_coordinate": dest,
                "destination": dest_name,
                "angle": angle,
                "rpy": [0.0, 0.0, angle],
                "retry": 0,
                "max_retry": max_retry,
                "status": "pending",
                "reason": "将倒放/倾倒零件调整为正常姿态后再放置",
                "original_cmd": cmd,
                "target_pose_state": "normal",
            }
        )
        place_rpy = [0.0, 0.0, angle]

    step += 1
    seq.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "move",
            "object": obj,
            "coordinate": dest,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": place_rpy,
            "retry": 0,
            "max_retry": max_retry,
            "status": "pending",
            "reason": "携带零件移动至放置目标上方",
            "original_cmd": cmd,
        }
    )
    step += 1
    seq.append(
        {
            "task_id": f"t{step:03d}",
            "step": step,
            "action": "place",
            "object": obj,
            "coordinate": dest,
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": place_rpy,
            "retry": 0,
            "max_retry": max_retry,
            "status": "pending",
            "reason": "",
            "original_cmd": cmd,
            "on_fail": {
                "action": "retry_place",
                "max_retry": max_retry,
                "reason": "若摆放失败则重新感知并重试放置",
            },
        }
    )
    task["tasks"] = seq
    if pose_state in ("upside_down", "tilt"):
        task["task_sequence_desc"] = [
            "1.perceive 识别并判断姿态",
            "2.pick 抓取",
            "3.adjust_pose 调姿",
            "4.move",
            "5.place",
        ]
    else:
        task["task_sequence_desc"] = [
            "1.perceive 识别定位目标",
            "2.pick 抓取目标零件",
            "3.move 移动至放置点",
            "4.place 放入收纳盒/料箱格子（失败可 retry_place）",
        ]
    return task


def main(result_dict: dict, user_cmd: str = "") -> dict:
    if not isinstance(result_dict, dict):
        result_dict = {}

    raw_task = result_dict.get("task_json", {})
    error_msg = result_dict.get("error", "")
    new_task = raw_task.copy() if isinstance(raw_task, dict) else {}

    raw_status = new_task.get("status", "")
    raw_reason = new_task.get("reason", "")
    auto_keywords = [
        "匹配最大可见",
        "自动选择",
        "未明确指定零件",
        "未指定具体零件",
        "未指定目标零件",
        "默认选择",
        "选取",
        "空间关系",
        "最多区域",
        "区域装箱",
        "搬运",
    ]
    is_auto_capture = raw_status == "invalid" and any(
        word in raw_reason for word in auto_keywords
    )
    if is_auto_capture or new_task.get("action") in HIGH_ACTIONS:
        # reflect 的 final_fail 需保留 failed；其它高层任务保持 pending
        if new_task.get("action") == "reflect" and new_task.get("flow_tag") == "final_fail":
            new_task["status"] = "failed"
        else:
            new_task["status"] = "pending" if new_task.get("status") != "failed" else "failed"
        if new_task.get("action") not in HIGH_ACTIONS:
            new_task["action"] = "pick"

    cmd = (user_cmd or new_task.get("original_cmd") or "").strip()
    new_task["original_cmd"] = cmd

    if new_task.get("action") not in HIGH_ACTIONS:
        if intent_do_place(cmd, new_task.get("destination")):
            dest = str(new_task.get("destination", "") or "").strip()
            if dest in ("", "null", "None"):
                new_task["destination"] = STORAGE_BOX_NAME
                new_task["dest_coordinate"] = list(STORAGE_BOX_XYZ)
            elif new_task.get("destination") == STORAGE_BOX_NAME:
                new_task["dest_coordinate"] = list(STORAGE_BOX_XYZ)
        elif not wants_transport(cmd) and not wants_pack_region(cmd):
            new_task["destination"] = None
            new_task["dest_coordinate"] = None
            new_task["slot_id"] = None

    new_task = ensure_sequence(new_task, user_cmd=cmd)
    new_task = shallow_task_for_dify(new_task)

    return {
        "valid_flag": bool(result_dict.get("valid", False)),
        "task_data": new_task,
        "error_msg": error_msg,
    }
