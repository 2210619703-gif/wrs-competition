STORAGE_BOX_NAME = "收纳盒"
STORAGE_BOX_XYZ = [0.30, -0.20, 0.0]
PLACE_KEYWORDS = ("放到", "放入", "放置", "放进", "移到", "移动到", "装箱")
TRANSPORT_KEYWORDS = ("搬运", "搬走", "移送料箱", "搬料箱")


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
    if "零件最多" in text:
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


def needs_reorient(pose_state: str) -> bool:
    # 兼容 example_bin: inverted / fallen
    t = str(pose_state or "").strip()
    aliases = {
        "upside_down",
        "tilt",
        "倒放",
        "倾倒",
        "倒置",
        "inverted",
        "fallen",
        "tilted",
        "倾斜",
    }
    return t in aliases or t.lower() in aliases


def _normalize_pose(raw) -> str:
    aliases = {
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
    if raw is None:
        return "normal"
    t = str(raw).strip()
    if t in aliases:
        return aliases[t]
    return aliases.get(t.lower(), "normal")


def ensure_sequence(task_data, user_cmd, status="pending", reason=""):
    obj = task_data.get("object", "")
    pick_xyz = list(task_data.get("coordinate", [0.0, 0.0, 0.0]))
    rpy = list(task_data.get("rpy") or [0.0, 0.0, float(task_data.get("angle", 0.0))])
    angle = round(float(task_data.get("angle", rpy[2] if len(rpy) > 2 else 0.0)), 6)
    max_retry = int(task_data.get("max_retry", 3))
    slot_id = task_data.get("slot_id")
    cmd = (user_cmd or task_data.get("original_cmd") or "").strip()
    pose_state = _normalize_pose(task_data.get("pose_state") or task_data.get("state") or "normal")
    action = str(task_data.get("action") or "")

    # 高层任务（区域装箱 / 搬运 / reflect）已由上游拼好，直接保留
    # reflect 的 next_task / final_fail 可能 tasks=[]，也必须原样保留，不能再拼 pick
    existing = task_data.get("tasks")
    high_actions = (
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
    if action in high_actions:
        out = dict(task_data)
        out["original_cmd"] = cmd
        out["status"] = status if status != "pending" else out.get("status", status)
        if reason and not out.get("reason"):
            out["reason"] = reason
        if not isinstance(out.get("tasks"), list):
            out["tasks"] = []
        return out
    if isinstance(existing, list) and existing:
        acts = [t.get("action") for t in existing if isinstance(t, dict)]
        if "transport" in acts or "adjust_pose" in acts:
            out = dict(task_data)
            out["original_cmd"] = cmd
            out["status"] = status if status != "pending" else out.get("status", status)
            if reason and not out.get("reason"):
                out["reason"] = reason
            return out
        if len(existing) >= 4 and "place" in acts:
            out = dict(task_data)
            out["original_cmd"] = cmd
            return out

    do_place = intent_do_place(cmd, task_data.get("destination"))
    raw_dest = task_data.get("destination")

    if do_place:
        dest_name = raw_dest or STORAGE_BOX_NAME
        if dest_name in ("", "null", "None"):
            dest_name = STORAGE_BOX_NAME
        dest = list(task_data.get("dest_coordinate") or STORAGE_BOX_XYZ)
        if dest_name == STORAGE_BOX_NAME:
            dest = list(STORAGE_BOX_XYZ)
    else:
        dest_name = None
        dest = None
        slot_id = None

    perceive = {
        "task_id": "t000",
        "step": 1,
        "action": "perceive",
        "object": obj,
        "coordinate": pick_xyz,
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": angle,
        "rpy": [round(float(v), 6) for v in rpy],
        "retry": 0,
        "max_retry": max_retry,
        "status": status,
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
        "rpy": [round(float(v), 6) for v in rpy],
        "retry": 0,
        "max_retry": max_retry,
        "status": status,
        "reason": reason,
        "original_cmd": cmd,
    }

    out = dict(pick)
    out["destination"] = dest_name
    out["dest_coordinate"] = dest
    out["slot_id"] = slot_id
    out["original_cmd"] = cmd
    out["pose_state"] = pose_state
    out["coordinate"] = pick_xyz
    out["angle"] = angle
    out["rpy"] = [round(float(v), 6) for v in rpy]
    out["max_retry"] = max_retry
    out["retry"] = 0
    out["object"] = obj
    out["task_id"] = task_data.get("task_id") or "t001"

    if not do_place:
        out["tasks"] = [perceive, pick]
        out["task_sequence_desc"] = [
            "1.perceive 识别定位目标",
            "2.pick 抓取目标零件（仅拿起，不放置）",
        ]
        return out

    step = 2
    tasks = [perceive, pick]
    place_rpy = [round(float(v), 6) for v in rpy]

    if needs_reorient(pose_state):
        step += 1
        tasks.append(
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
    tasks.append(
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
    tasks.append(
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
    out["tasks"] = tasks
    if needs_reorient(pose_state):
        out["task_sequence_desc"] = [
            "1.perceive 识别并判断姿态",
            "2.pick 抓取异常姿态零件",
            "3.adjust_pose 翻正为正常姿态",
            "4.move 移至放置点",
            "5.place 正常姿态放入",
        ]
    else:
        out["task_sequence_desc"] = [
            "1.perceive 识别定位目标",
            "2.pick 抓取目标零件",
            "3.move 移动至放置点",
            "4.place 放入收纳盒/料箱格子（失败可 retry_place）",
        ]
    return out


def main(task_data: dict, error_msg: str, user_cmd: str) -> dict:
    if not isinstance(task_data, dict):
        task_data = {}

    raw_status = task_data.get("status", "")
    raw_reason = task_data.get("reason", "")
    auto_key_words = [
        "匹配最大可见",
        "自动选择",
        "未明确指定零件",
        "选取",
        "默认选择",
        "空间关系",
        "最多区域",
        "区域装箱",
        "搬运",
    ]
    auto_pick_ok = raw_status == "invalid" and any(
        word in raw_reason for word in auto_key_words
    )
    if task_data.get("action") in (
        "pack_region",
        "transport",
        "multi_agent_pack",
        "reflect",
        "pack_bin_slots",
        "pack_n_parts",
        "pack_all_box",
        "multi_named_pick_place",
        "pack_batch",
    ):
        auto_pick_ok = True
    real_error = (
        error_msg != "" or raw_status == "error" or "json format error" in raw_reason
    )

    if real_error and not auto_pick_ok:
        fail = ensure_sequence(task_data, user_cmd, status="failed", reason=error_msg or raw_reason)
        fail["task_id"] = "error"
        fail["action"] = "stop"
        fail["tasks"] = []
        return {"fail_task": fail}

    # reflect：按上游 status（含 final_fail→failed）透传，不要一律 pending
    keep_status = "pending"
    if task_data.get("action") == "reflect" and task_data.get("flow_tag") == "final_fail":
        keep_status = "failed"
    elif task_data.get("status") == "failed":
        keep_status = "failed"

    out = ensure_sequence(
        task_data,
        user_cmd,
        status=keep_status,
        reason=raw_reason,
    )
    return {"fail_task": out}
