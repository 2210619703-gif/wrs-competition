"""
未连接智能体时：本地生成完整任务序列 JSON（不依赖 Dify / LLM）。

用法：python offline/build_tasks_local.py
或双击 offline/一键生成任务JSON.bat
输出写到决策模块根目录：delivery_for_simulation.json、delivery_scenario_35.json
详见 使用说明.md「未连接智能体时」。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 决策模块根目录
sys.path.insert(0, str(ROOT / "workflow_code"))

from common_config import (  # noqa: E402
    INTENT_MULTI_AGENT,
    INTENT_MULTI_NAMED,
    INTENT_PACK_ALL_BOX,
    INTENT_PACK_N,
    INTENT_PACK_REGION,
    INTENT_PACK_SLOTS,
    INTENT_PICK_PLACE,
    INTENT_TRANSPORT,
    STORAGE_BOX_NAME,
    STORAGE_BOX_XYZ,
    extract_named_mentions,
    parse_intent,
    parse_part_count,
)
from multi_agent import build_multi_agent_plan  # noqa: E402
from nl_parse import choose_target, resolve_destination_from_cmd, select_densest_region  # noqa: E402
from pose_state import attach_pose_state, infer_pose_state  # noqa: E402
from task_sequence import (  # noqa: E402
    build_pack_region_sequence,
    build_parts_batch_sequence,
    build_task_sequence,
    build_transport_with_fill_sequence,
    _renumber_tasks,
)

DATASET_FILE = ROOT / "simple_dataset.json"
DELIVERY_FILE = ROOT / "delivery_for_simulation.json"
DELIVERY_LOCAL = ROOT / "delivery_for_simulation_local.json"
DELIVERY_SCENARIO = ROOT / "delivery_scenario_35.json"


def to_parts(vision_objects):
    parts = []
    for obj in attach_pose_state((vision_objects or {}).get("objects") or []):
        if obj.get("visible_pixels", 0) <= 0 or obj.get("bbox") is None:
            continue
        p = obj["pose_6d"]
        item = {
            "id": obj["id"],
            "class": obj["class"],
            "xyz": [float(p["x"]), float(p["y"]), float(p["z"])],
            "rpy": [float(p["roll"]), float(p["pitch"]), float(p["yaw"])],
            "visible_pixels": obj["visible_pixels"],
            "pose_state": obj.get("pose_state") or infer_pose_state(obj),
        }
        if obj.get("slot_id") is not None:
            item["slot_id"] = obj.get("slot_id")
        if obj.get("dest_coordinate") is not None:
            item["dest_coordinate"] = list(obj.get("dest_coordinate"))
            item["destination"] = obj.get("destination")
        parts.append(item)
    return parts


def make_natural_cmd(sample_id, parts, base_cmd):
    """
    对齐项目书 / 答疑文件的指令风格（XH-202607）：
    - 拿取/抓取 + 放到收纳盒/料箱格子 → perceive→pick→[adjust_pose]→move→place
    - 只拿取（无放置词）→ perceive→pick
    - 「零件最多的区域装箱」/「搬运料箱」/「所有零件装箱」→ 高层任务
    不再使用「抓取指定工业零件」（会永远选最大可见面积零件）。
    """
    if not parts:
        return base_cmd or "抓取指定工业零件"

    ordered = sorted(
        parts, key=lambda x: (-int(x.get("visible_pixels") or 0), x.get("id") or 0)
    )
    n = int(sample_id) if str(sample_id).isdigit() else 1
    target = ordered[(n - 1) % len(ordered)]
    name = target["class"]

    pose_parts = [p for p in parts if p.get("pose_state") in ("upside_down", "tilt")]
    pose_target = pose_parts[(n - 1) % len(pose_parts)] if pose_parts else None
    pose_name = pose_target["class"] if pose_target else name
    slot = (n % 6) + 1

    # 项目书示例优先：抓放/调姿/区域装箱/搬运/多智能体；穿插只抓取
    styles = [
        f"拿取{name}放到收纳盒",  # 标准：拿取+移动+放置
        f"帮我把{name}放到收纳盒",
        f"取出{name}放到料箱的第{slot}个格子中",  # 高阶：指定格子
        f"把{pose_name}放到收纳盒",  # 倒放/倾倒 → 含 adjust_pose
        "帮我把零件最多的区域装箱",  # 答疑 Q8
        "帮我搬运一下料箱盒",  # 答疑 Q8
        "帮我把所有零件装箱",  # 多智能体
        f"给我一个{name}",  # 只拿取（无 move/place）
        f"抓取{name}",
        f"帮我拿一把{name}，并且放到收纳盒",  # 课题解读示例句式
    ]
    return styles[(n - 1) % len(styles)]


def _storage_xyz(vision_objects, dest_xyz):
    tc = (vision_objects or {}).get("target_container") or {}
    if tc.get("xyz"):
        xyz = [float(v) for v in tc["xyz"][:3]]
        while len(xyz) < 3:
            xyz.append(0.0)
        return xyz
    return list(dest_xyz or STORAGE_BOX_XYZ)


def build_tasks(user_cmd, vision_objects, sample_id):
    parts = to_parts(vision_objects)
    if not parts:
        return {
            "sample_id": sample_id,
            "user_cmd": user_cmd,
            "intent": "error",
            "task": {
                "task_id": "error",
                "action": "stop",
                "object": "",
                "coordinate": [0.0, 0.0, 0.0],
                "dest_coordinate": list(STORAGE_BOX_XYZ),
                "destination": STORAGE_BOX_NAME,
                "status": "failed",
                "reason": "无有效零件",
                "tasks": [],
            },
        }

    intent = parse_intent(user_cmd, [p["class"] for p in parts])
    dest_name, dest_xyz, slot_id = resolve_destination_from_cmd(user_cmd)
    batch_intents = (
        INTENT_PACK_REGION,
        INTENT_TRANSPORT,
        INTENT_PICK_PLACE,
        INTENT_PACK_N,
        INTENT_PACK_ALL_BOX,
        INTENT_MULTI_NAMED,
        INTENT_PACK_SLOTS,
    )
    if intent in batch_intents:
        if slot_id is None:
            dest_name = dest_name or STORAGE_BOX_NAME
            dest_xyz = _storage_xyz(vision_objects, dest_xyz or STORAGE_BOX_XYZ)

    if intent == INTENT_PACK_N:
        n = parse_part_count(user_cmd) or 1
        ranked = sorted(parts, key=lambda o: (-int(o.get("visible_pixels") or 0), o.get("id") or 0))
        summary = build_parts_batch_sequence(
            parts=ranked[:n],
            user_cmd=user_cmd,
            dest_xyz=dest_xyz,
            destination=dest_name or STORAGE_BOX_NAME,
            action_name="pack_n_parts",
            reason=f"随便拿{n}个零件放入收纳盒，失败跳过继续",
        )
        return {"sample_id": sample_id, "user_cmd": user_cmd, "intent": intent, "task": summary}

    if intent == INTENT_PACK_ALL_BOX:
        summary = build_parts_batch_sequence(
            parts=parts,
            user_cmd=user_cmd,
            dest_xyz=dest_xyz,
            destination=dest_name or STORAGE_BOX_NAME,
            action_name="pack_all_box",
            reason="所有零件放进收纳盒，失败跳过继续",
        )
        return {"sample_id": sample_id, "user_cmd": user_cmd, "intent": intent, "task": summary}

    if intent == INTENT_MULTI_NAMED:
        names = extract_named_mentions(user_cmd, [p["class"] for p in parts])
        used = set()
        chosen = []
        for name in names:
            hit = None
            for p in parts:
                if p.get("id") in used:
                    continue
                cls = p.get("class") or ""
                if name == cls or name in cls:
                    hit = p
                    break
            if hit:
                used.add(hit.get("id"))
                chosen.append(hit)
        summary = build_parts_batch_sequence(
            parts=chosen,
            user_cmd=user_cmd,
            dest_xyz=dest_xyz,
            destination=dest_name or STORAGE_BOX_NAME,
            action_name="multi_named_pick_place",
            reason="多个具体零件连续 pick-place，失败跳过继续",
        )
        return {"sample_id": sample_id, "user_cmd": user_cmd, "intent": intent, "task": summary}

    if intent == INTENT_PACK_SLOTS:
        n = parse_part_count(user_cmd)
        work = sorted(parts, key=lambda p: int(p.get("id") or 0))
        if n:
            work = work[:n]
        flat = []
        for i, p in enumerate(work):
            sid = p.get("slot_id")
            if sid is None:
                sid = i
            dest = list(p.get("dest_coordinate") or STORAGE_BOX_XYZ)
            seq = build_task_sequence(
                object_name=p["class"],
                pick_xyz=p["xyz"],
                pick_rpy=p.get("rpy") or [0.0, 0.0, 0.0],
                destination=p.get("destination") or f"料盘格子{sid}",
                dest_xyz=dest,
                user_cmd=user_cmd,
                reason=f"按口令生成；入格第{i + 1}件",
                status="pending",
                do_place=True,
                pose_state=p.get("pose_state") or "normal",
                slot_id=sid,
            )
            sub = _renumber_tasks(seq.get("tasks") or [], step_offset=len(flat), id_prefix="t")
            for t in sub:
                t["skip_on_fail"] = True
                t["on_fail"] = {"action": "skip_continue", "reason": "该件规划失败则跳过继续"}
            flat.extend(sub)
        first = work[0] if work else parts[0]
        summary = {
            "task_id": "t_bin_all",
            "action": "pack_bin_slots",
            "object": first["class"],
            "coordinate": list(first["xyz"]),
            "dest_coordinate": list(
                (flat[0].get("dest_coordinate") if flat else None) or STORAGE_BOX_XYZ
            ),
            "destination": "料盘格子",
            "status": "pending",
            "reason": f"按口令生成；共{len(work)}件入格，失败跳过继续",
            "original_cmd": user_cmd,
            "skip_on_fail": True,
            "tasks": flat,
        }
        return {"sample_id": sample_id, "user_cmd": user_cmd, "intent": intent, "task": summary}

    if intent == INTENT_PACK_REGION:
        region, region_key, _ = select_densest_region(parts)
        summary = build_pack_region_sequence(
            region_parts=region,
            user_cmd=user_cmd,
            dest_xyz=dest_xyz,
            destination=dest_name or STORAGE_BOX_NAME,
            region_key=region_key,
        )
        return {
            "sample_id": sample_id,
            "user_cmd": user_cmd,
            "intent": intent,
            "task": summary,
        }

    if intent == INTENT_MULTI_AGENT:
        summary = build_multi_agent_plan(
            parts_3d=parts,
            user_cmd=user_cmd,
            dest_xyz=dest_xyz or STORAGE_BOX_XYZ,
        )
        return {
            "sample_id": sample_id,
            "user_cmd": user_cmd,
            "intent": intent,
            "task": summary,
        }

    if intent == INTENT_TRANSPORT:
        summary = build_transport_with_fill_sequence(
            parts_3d=parts,
            user_cmd=user_cmd,
            filled_count=0,
            dest_xyz=dest_xyz or STORAGE_BOX_XYZ,
        )
        return {
            "sample_id": sample_id,
            "user_cmd": user_cmd,
            "intent": intent,
            "task": summary,
        }

    target, reason, status = choose_target(user_cmd, parts)
    if slot_id is None and dest_name == STORAGE_BOX_NAME:
        dest_xyz = _storage_xyz(vision_objects, dest_xyz)

    summary = build_task_sequence(
        object_name=target["class"],
        pick_xyz=target["xyz"],
        pick_rpy=target["rpy"],
        destination=dest_name,
        dest_xyz=dest_xyz,
        user_cmd=user_cmd,
        reason=reason,
        status="pending" if status in ("valid", "invalid") else status,
        slot_id=slot_id,
        do_place=dest_name is not None,
        pose_state=target.get("pose_state") or "normal",
    )
    return {
        "sample_id": sample_id,
        "user_cmd": user_cmd,
        "intent": intent,
        "task": summary,
    }


def build_pick_task(user_cmd, vision_objects, sample_id):
    return build_tasks(user_cmd, vision_objects, sample_id)


def main():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)

    delivery = []
    scenario = []
    for item in samples:
        sample_id = item.get("sample_id", "0000")
        vision_objects = item.get("vision_objects")
        if vision_objects is None and "objects" in item:
            vision_objects = {
                "objects": item["objects"],
                "target_container": {"xyz": STORAGE_BOX_XYZ},
            }
        parts = to_parts(vision_objects)
        user_cmd = make_natural_cmd(sample_id, parts, item.get("user_cmd", ""))
        entry = build_tasks(user_cmd, vision_objects, sample_id)
        delivery.append(entry)
        if entry.get("intent") in (INTENT_PACK_REGION, INTENT_TRANSPORT) or (
            entry.get("task") or {}
        ).get("pose_state") in ("upside_down", "tilt"):
            scenario.append(entry)

    # 额外保证场景文件里一定有装箱/搬运样例
    if samples:
        vo = samples[0].get("vision_objects") or {
            "objects": samples[0].get("objects") or [],
            "target_container": {"xyz": STORAGE_BOX_XYZ},
        }
        for cmd in ("帮我把零件最多的区域装箱", "帮我搬运一下料箱盒"):
            scenario.insert(0, build_tasks(cmd, vo, samples[0].get("sample_id", "0001")))

    for path in (DELIVERY_FILE, DELIVERY_LOCAL):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(delivery, f, ensure_ascii=False, indent=2)
    with open(DELIVERY_SCENARIO, "w", encoding="utf-8") as f:
        json.dump(scenario[:40], f, ensure_ascii=False, indent=2)

    print(f"[OK] 交付 {len(delivery)} 条 -> {DELIVERY_FILE}")
    print(f"[OK] 场景样例 {min(40, len(scenario))} 条 -> {DELIVERY_SCENARIO}")
    counts = {}
    for e in delivery:
        counts[e.get("intent")] = counts.get(e.get("intent"), 0) + 1
    print("  intent 分布:", counts)
    for e in delivery[:8]:
        t = e["task"]
        print(f"  {e['sample_id']} [{e.get('intent')}] {e['user_cmd']}")
        print(f"    steps({len(t.get('tasks') or [])}): {[x['action'] for x in (t.get('tasks') or [])][:8]}...")


if __name__ == "__main__":
    main()
