"""
多智能体协同：答疑高阶任务
「帮我把所有零件装箱」→ AgentA / AgentB 分区装箱，AgentC 搬运满箱
"""
from __future__ import annotations

from common_config import BOX_CAPACITY, PLACE_AREA_XYZ, STORAGE_BOX_NAME, STORAGE_BOX_XYZ
from nl_parse import cluster_parts_by_grid
from task_sequence import (
    build_pack_region_sequence,
    build_transport_sequence,
    _renumber_tasks,
)


def _top_regions(parts_3d, cell: float = 0.20, top_k: int = 2):
    buckets = cluster_parts_by_grid(parts_3d, cell=cell)
    if not buckets:
        return []
    ranked = sorted(
        buckets.items(),
        key=lambda kv: (len(kv[1]), -abs(kv[0][0]), -abs(kv[0][1])),
        reverse=True,
    )
    return ranked[:top_k]


def build_multi_agent_plan(
    *,
    parts_3d: list,
    user_cmd: str = "帮我把所有零件装箱",
    dest_xyz=None,
    capacity: int = None,
):
    """
    返回多智能体任务包：
    - agents.A / agents.B：分区装箱
    - agents.C：搬运（A 先完成则先搬 A 的箱，B 同理；演示为顺序协议）
    """
    capacity = int(capacity or BOX_CAPACITY)
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    regions = _top_regions(parts_3d, top_k=2)

    # 不足 2 区时：一半零件给 A，一半给 B
    if len(regions) < 2:
        sorted_parts = sorted(
            parts_3d,
            key=lambda o: (o["xyz"][0], o["xyz"][1]),
        )
        mid = max(1, len(sorted_parts) // 2)
        regions = [
            (("split", 0), sorted_parts[:mid]),
            (("split", 1), sorted_parts[mid:] or sorted_parts[:1]),
        ]

    (key_a, parts_a), (key_b, parts_b) = regions[0], regions[1]
    pack_a = build_pack_region_sequence(
        region_parts=parts_a[:capacity],
        user_cmd=user_cmd,
        dest_xyz=dest,
        destination=f"{STORAGE_BOX_NAME}-A",
        capacity=capacity,
        region_key=key_a,
    )
    pack_b = build_pack_region_sequence(
        region_parts=parts_b[:capacity],
        user_cmd=user_cmd,
        dest_xyz=[dest[0] + 0.35, dest[1], dest[2]],
        destination=f"{STORAGE_BOX_NAME}-B",
        capacity=capacity,
        region_key=key_b,
    )
    # C：先搬 A 箱，再搬 B 箱（A 优先完成协议）
    transport_a = build_transport_sequence(user_cmd=user_cmd, box_name="料箱A")
    transport_a["destination"] = "零件放置区"
    transport_a["dest_coordinate"] = list(PLACE_AREA_XYZ)
    transport_a["reason"] = "AgentA 装箱完成后，AgentC 搬运料箱A并叠放"
    transport_a["depends_on"] = "agent_A"
    transport_a["agent_id"] = "agent_C"

    transport_b = build_transport_sequence(user_cmd=user_cmd, box_name="料箱B")
    transport_b["destination"] = "零件放置区"
    transport_b["dest_coordinate"] = [
        PLACE_AREA_XYZ[0],
        PLACE_AREA_XYZ[1] - 0.12,
        PLACE_AREA_XYZ[2],
    ]
    for t in transport_b.get("tasks") or []:
        t["dest_coordinate"] = list(transport_b["dest_coordinate"])
        t["destination"] = "零件放置区"
        if t.get("action") in ("transport", "place", "move"):
            t["coordinate"] = list(transport_b["dest_coordinate"])
    transport_b["reason"] = "AgentB 装箱完成后，AgentC 搬运料箱B并叠放"
    transport_b["depends_on"] = "agent_B"
    transport_b["agent_id"] = "agent_C"

    pack_a["agent_id"] = "agent_A"
    pack_b["agent_id"] = "agent_B"

    # 扁平汇总 tasks（串行演示：A装箱→C搬A→B装箱→C搬B）
    # Dify 深度限制：agents 只放摘要，完整步骤只在顶层 tasks
    flat = []
    for block in (pack_a, transport_a, pack_b, transport_b):
        sub = _renumber_tasks(block.get("tasks") or [], step_offset=len(flat), id_prefix="t")
        for t in sub:
            t["agent_id"] = block.get("agent_id")
            # 压平 on_fail，避免 Dify Depth limit
            if isinstance(t.get("on_fail"), dict):
                t["on_fail_action"] = t["on_fail"].get("action")
                t["on_fail_max_retry"] = t["on_fail"].get("max_retry")
                t.pop("on_fail", None)
        flat.extend(sub)

    def _rk(key):
        if isinstance(key, (list, tuple)):
            return [key[0], key[1]] if len(key) >= 2 else list(key)
        return key

    return {
        "task_id": "t_multi_agent",
        "action": "multi_agent_pack",
        "object": "所有零件",
        "coordinate": list(STORAGE_BOX_XYZ),
        "dest_coordinate": list(PLACE_AREA_XYZ),
        "destination": "零件放置区",
        "status": "pending",
        "reason": "多智能体：A/B分区装箱，C负责满箱搬运叠放",
        "original_cmd": user_cmd,
        "collaboration": "A/B并行装箱；谁先满谁先被C搬运；演示顺序 A→C(A)→B→C(B)",
        "agent_A_role": "packer",
        "agent_A_region": _rk(key_a),
        "agent_A_parts": pack_a.get("region_count", 0),
        "agent_B_role": "packer",
        "agent_B_region": _rk(key_b),
        "agent_B_parts": pack_b.get("region_count", 0),
        "agent_C_role": "transporter",
        "tasks": flat,
        "task_sequence_desc": [
            "1.agent_A 区A装箱",
            "2.agent_C 搬运料箱A叠放",
            "3.agent_B 区B装箱",
            "4.agent_C 搬运料箱B叠放",
        ],
    }
