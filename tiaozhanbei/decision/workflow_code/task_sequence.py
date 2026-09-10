"""
任务序列分解：对齐项目书 + 第③⑤点
- 只拿起来 / 拿起放到盒子
- 倒放/倾倒：先调姿再放置
- 失败闭环：重感知→重抓→调姿→再放
- 最多区域装箱：分区后批量依次装箱
- 箱满：搬运料箱至放置区（不满则先继续装）
"""
from common_config import (
    BOX_CAPACITY,
    PLACE_AREA_XYZ as _PLACE_AREA_XYZ,
    STORAGE_BOX_NAME,
    STORAGE_BOX_XYZ,
    wants_place,
)
from pose_state import POSE_CN, needs_reorient

PLACE_AREA_XYZ = list(_PLACE_AREA_XYZ)


def _step(
    *,
    task_id,
    step,
    action,
    object_name,
    coordinate,
    dest_coordinate,
    destination,
    angle,
    rpy,
    user_cmd,
    reason="",
    status="pending",
    max_retry=3,
    extra=None,
):
    item = {
        "task_id": task_id,
        "step": step,
        "action": action,
        "object": object_name,
        "coordinate": list(coordinate) if coordinate is not None else None,
        "dest_coordinate": list(dest_coordinate) if dest_coordinate is not None else None,
        "destination": destination,
        "angle": angle,
        "rpy": list(rpy),
        "retry": 0,
        "max_retry": max_retry,
        "status": status,
        "reason": reason,
        "original_cmd": user_cmd,
    }
    if extra:
        item.update(extra)
    return item


def build_task_sequence(
    *,
    object_name: str,
    pick_xyz,
    pick_rpy,
    destination,
    dest_xyz,
    user_cmd: str,
    reason: str = "",
    status: str = "pending",
    max_retry: int = 3,
    slot_id=None,
    do_place=None,
    pose_state: str = "normal",
    x_off: float = 0.0,
    yaw_off: float = 0.0,
):
    angle = round(float(pick_rpy[2]) + float(yaw_off), 6)
    rpy = [
        round(float(pick_rpy[0]), 6),
        round(float(pick_rpy[1]), 6),
        angle,
    ]
    pick_xyz = [
        float(pick_xyz[0]) + float(x_off),
        float(pick_xyz[1]),
        float(pick_xyz[2]) if len(pick_xyz) > 2 else 0.0,
    ]

    if do_place is None:
        do_place = wants_place(user_cmd)
        if destination is None or str(destination).strip() in ("", "null", "None", "无"):
            do_place = wants_place(user_cmd)

    if do_place:
        dest_name = destination or STORAGE_BOX_NAME
        if dest_name in ("", "null", "None", "无"):
            dest_name = STORAGE_BOX_NAME
        dest = list(dest_xyz) if dest_xyz is not None else list(STORAGE_BOX_XYZ)
        dest = [dest[0] + float(x_off), dest[1], dest[2] if len(dest) > 2 else 0.0]
    else:
        dest_name = None
        dest = None

    tasks = []
    step = 1
    tasks.append(
        _step(
            task_id="t000",
            step=step,
            action="perceive",
            object_name=object_name,
            coordinate=pick_xyz,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            reason="识别场景目标并定位；判断正常/倒放/倾倒",
            status=status,
            max_retry=max_retry,
            extra={"pose_state": pose_state, "pose_state_cn": POSE_CN.get(pose_state, pose_state)},
        )
    )
    step += 1

    if do_place and needs_reorient(pose_state):
        # 倒放/倾倒：先抓起 → 调姿翻正 → 再移放
        tasks.append(
            _step(
                task_id=f"t{step:03d}",
                step=step,
                action="pick",
                object_name=object_name,
                coordinate=pick_xyz,
                dest_coordinate=dest,
                destination=dest_name,
                angle=angle,
                rpy=rpy,
                user_cmd=user_cmd,
                reason=f"抓取{POSE_CN.get(pose_state, pose_state)}零件",
                status=status,
                max_retry=max_retry,
            )
        )
        step += 1
        flip_rpy = [0.0, 0.0, angle]
        tasks.append(
            _step(
                task_id=f"t{step:03d}",
                step=step,
                action="adjust_pose",
                object_name=object_name,
                coordinate=pick_xyz,
                dest_coordinate=dest,
                destination=dest_name,
                angle=angle,
                rpy=flip_rpy,
                user_cmd=user_cmd,
                reason="将倒放/倾倒零件调整为正常姿态后再放置",
                status="pending",
                max_retry=max_retry,
                extra={"target_pose_state": "normal"},
            )
        )
        step += 1
        tasks.append(
            _step(
                task_id=f"t{step:03d}",
                step=step,
                action="move",
                object_name=object_name,
                coordinate=dest,
                dest_coordinate=dest,
                destination=dest_name,
                angle=angle,
                rpy=flip_rpy,
                user_cmd=user_cmd,
                reason="携带已翻正零件移动至放置点",
                max_retry=max_retry,
            )
        )
        step += 1
        tasks.append(
            _step(
                task_id=f"t{step:03d}",
                step=step,
                action="place",
                object_name=object_name,
                coordinate=dest,
                dest_coordinate=dest,
                destination=dest_name,
                angle=angle,
                rpy=flip_rpy,
                user_cmd=user_cmd,
                reason="以正常姿态放入料箱格子",
                max_retry=max_retry,
                extra={
                    "on_fail": {
                        "action": "retry_place",
                        "max_retry": max_retry,
                        "reason": "若摆放失败则重新感知并重试放置",
                    }
                },
            )
        )
        desc = [
            "1.perceive 识别并判断姿态",
            "2.pick 抓取异常姿态零件",
            "3.adjust_pose 翻正为正常姿态",
            "4.move 移至放置点",
            "5.place 正常姿态放入",
        ]
        return {
            "task_id": "t001",
            "action": "pick",
            "object": object_name,
            "coordinate": list(pick_xyz),
            "dest_coordinate": dest,
            "destination": dest_name,
            "angle": angle,
            "rpy": rpy,
            "retry": 0,
            "max_retry": max_retry,
            "status": status,
            "reason": reason or f"姿态={POSE_CN.get(pose_state, pose_state)}，需调姿后放置",
            "original_cmd": user_cmd,
            "slot_id": slot_id,
            "pose_state": pose_state,
            "tasks": tasks,
            "task_sequence_desc": desc,
        }

    # 仅抓取
    tasks.append(
        _step(
            task_id="t001",
            step=step,
            action="pick",
            object_name=object_name,
            coordinate=pick_xyz,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            reason=reason,
            status=status,
            max_retry=max_retry,
        )
    )
    step += 1

    if not do_place:
        return {
            "task_id": "t001",
            "action": "pick",
            "object": object_name,
            "coordinate": list(pick_xyz),
            "dest_coordinate": None,
            "destination": None,
            "angle": angle,
            "rpy": rpy,
            "retry": 0,
            "max_retry": max_retry,
            "status": status,
            "reason": reason,
            "original_cmd": user_cmd,
            "slot_id": None,
            "pose_state": pose_state,
            "tasks": tasks,
            "task_sequence_desc": [
                "1.perceive 识别定位目标",
                "2.pick 抓取目标零件（仅拿起，不放置）",
            ],
        }

    tasks.append(
        _step(
            task_id="t002",
            step=step,
            action="move",
            object_name=object_name,
            coordinate=dest,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            reason="携带零件移动至放置目标上方",
            max_retry=max_retry,
        )
    )
    step += 1
    tasks.append(
        _step(
            task_id="t003",
            step=step,
            action="place",
            object_name=object_name,
            coordinate=dest,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            max_retry=max_retry,
            extra={
                "on_fail": {
                    "action": "retry_place",
                    "max_retry": max_retry,
                    "reason": "若摆放失败则重新感知并重试放置",
                }
            },
        )
    )

    return {
        "task_id": "t001",
        "action": "pick",
        "object": object_name,
        "coordinate": list(pick_xyz),
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": angle,
        "rpy": rpy,
        "retry": 0,
        "max_retry": max_retry,
        "status": status,
        "reason": reason,
        "original_cmd": user_cmd,
        "slot_id": slot_id,
        "pose_state": pose_state,
        "tasks": tasks,
        "task_sequence_desc": [
            "1.perceive 识别定位目标",
            "2.pick 抓取目标零件",
            "3.move 移动至放置点",
            "4.place 放入收纳盒/料箱格子（失败可 retry_place）",
        ],
    }


def build_recovery_sequence(
    *,
    object_name: str,
    pick_xyz,
    pick_rpy,
    dest_xyz,
    user_cmd: str,
    fault_type: str,
    x_off: float = 0.0,
    yaw_off: float = 0.0,
    max_retry: int = 3,
):
    """第③点失败修正：重新感知 → 重抓 → 调姿 → 再放。"""
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    dest = [dest[0] + x_off, dest[1], dest[2] if len(dest) > 2 else 0.0]
    xyz = [float(pick_xyz[0]) + x_off, float(pick_xyz[1]), float(pick_xyz[2])]
    angle = round(float(pick_rpy[2]) + yaw_off, 6)
    rpy = [0.0, 0.0, angle]
    dest_name = STORAGE_BOX_NAME
    reason_map = {
        "part_fall": "零件掉落，重新抓取",
        "wrong_pose": "姿态异常，调整后重放",
        "pos_offset": "位置偏移，修正坐标后重放",
        "angle_offset": "角度歪斜，调整姿态后重放",
    }
    reason = reason_map.get(fault_type, "执行失败，闭环重规划")

    tasks = [
        _step(
            task_id="t100",
            step=1,
            action="perceive",
            object_name=object_name,
            coordinate=xyz,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            reason="失败后重新感知当前状态",
            max_retry=max_retry,
            extra={"fault_type": fault_type},
        ),
        _step(
            task_id="t101",
            step=2,
            action="pick",
            object_name=object_name,
            coordinate=xyz,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            reason="重新抓取",
            max_retry=max_retry,
        ),
        _step(
            task_id="t102",
            step=3,
            action="adjust_pose",
            object_name=object_name,
            coordinate=xyz,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            reason="调整姿态/补偿偏移",
            max_retry=max_retry,
            extra={"x_off": x_off, "yaw_off": yaw_off},
        ),
        _step(
            task_id="t103",
            step=4,
            action="place",
            object_name=object_name,
            coordinate=dest,
            dest_coordinate=dest,
            destination=dest_name,
            angle=angle,
            rpy=rpy,
            user_cmd=user_cmd,
            reason="再次放置",
            max_retry=max_retry,
            extra={
                "on_fail": {
                    "action": "retry_place",
                    "max_retry": max_retry,
                    "reason": reason,
                }
            },
        ),
    ]
    return {
        "task_id": "t_recovery",
        "action": "retry_place",
        "object": object_name,
        "coordinate": xyz,
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": angle,
        "rpy": rpy,
        "status": "pending",
        "reason": reason,
        "original_cmd": user_cmd,
        "fault_type": fault_type,
        "tasks": tasks,
        "task_sequence_desc": [
            "1.perceive 失败后重感知",
            "2.pick 重新抓取",
            "3.adjust_pose 调整姿态",
            "4.place 再次放置",
        ],
    }


def build_transport_sequence(*, user_cmd: str = "帮我搬运一下料箱盒", box_name: str = "料箱"):
    """第⑤点：箱满后搬运至放置区并叠放。"""
    box_xyz = list(STORAGE_BOX_XYZ)
    place_xyz = list(PLACE_AREA_XYZ)
    tasks = [
        _step(
            task_id="t200",
            step=1,
            action="perceive",
            object_name=box_name,
            coordinate=box_xyz,
            dest_coordinate=place_xyz,
            destination="零件放置区",
            angle=0.0,
            rpy=[0.0, 0.0, 0.0],
            user_cmd=user_cmd,
            reason="确认料箱已装满",
        ),
        _step(
            task_id="t201",
            step=2,
            action="pick",
            object_name=box_name,
            coordinate=box_xyz,
            dest_coordinate=place_xyz,
            destination="零件放置区",
            angle=0.0,
            rpy=[0.0, 0.0, 0.0],
            user_cmd=user_cmd,
            reason="抓取装满的料箱",
        ),
        _step(
            task_id="t202",
            step=3,
            action="transport",
            object_name=box_name,
            coordinate=place_xyz,
            dest_coordinate=place_xyz,
            destination="零件放置区",
            angle=0.0,
            rpy=[0.0, 0.0, 0.0],
            user_cmd=user_cmd,
            reason="搬运料箱至零件放置区",
        ),
        _step(
            task_id="t203",
            step=4,
            action="place",
            object_name=box_name,
            coordinate=place_xyz,
            dest_coordinate=place_xyz,
            destination="零件放置区",
            angle=0.0,
            rpy=[0.0, 0.0, 0.0],
            user_cmd=user_cmd,
            reason="叠放整齐",
        ),
    ]
    return {
        "task_id": "t_transport",
        "action": "transport",
        "object": box_name,
        "coordinate": box_xyz,
        "dest_coordinate": place_xyz,
        "destination": "零件放置区",
        "status": "pending",
        "reason": "料箱已满，执行搬运叠放",
        "original_cmd": user_cmd,
        "tasks": tasks,
        "task_sequence_desc": [
            "1.perceive 确认箱满",
            "2.pick 抓取料箱",
            "3.transport 搬运至放置区",
            "4.place 叠放整齐",
        ],
    }


def _renumber_tasks(tasks, step_offset: int = 0, id_prefix: str = "t"):
    out = []
    for i, t in enumerate(tasks):
        item = dict(t)
        step = step_offset + i + 1
        item["step"] = step
        item["task_id"] = f"{id_prefix}{step:03d}"
        out.append(item)
    return out


def build_pack_region_sequence(
    *,
    region_parts: list,
    user_cmd: str,
    dest_xyz=None,
    destination: str = None,
    capacity: int = None,
    region_key=None,
):
    """
    第⑤点：将零件最多区域依次装箱；倒放/倾倒走调姿。
    输出扁平 tasks，仿真可逐步执行。
    """
    capacity = int(capacity or BOX_CAPACITY)
    dest_name = destination or STORAGE_BOX_NAME
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    parts = list(region_parts or [])[:capacity]
    if not parts:
        return {
            "task_id": "t_pack",
            "action": "pack_region",
            "object": "",
            "coordinate": list(STORAGE_BOX_XYZ),
            "dest_coordinate": dest,
            "destination": dest_name,
            "status": "failed",
            "reason": "最密区域无有效零件",
            "original_cmd": user_cmd,
            "tasks": [],
            "batch": [],
            "task_sequence_desc": [],
        }

    flat = []
    batch = []
    desc = [f"0.perceive 选定零件最多区域{region_key}，共{len(parts)}件（容量{capacity}）"]
    for i, part in enumerate(parts):
        pose_state = part.get("pose_state") or "normal"
        seq = build_task_sequence(
            object_name=part["class"],
            pick_xyz=part["xyz"],
            pick_rpy=part.get("rpy") or [0.0, 0.0, 0.0],
            destination=dest_name,
            dest_xyz=dest,
            user_cmd=user_cmd,
            reason=f"区域装箱第{i + 1}/{len(parts)}件",
            status="pending",
            do_place=True,
            pose_state=pose_state,
        )
        sub = _renumber_tasks(seq["tasks"], step_offset=len(flat), id_prefix="t")
        flat.extend(sub)
        batch.append(
            {
                "index": i + 1,
                "object": part["class"],
                "object_id": part.get("id"),
                "pose_state": pose_state,
                "pose_state_cn": POSE_CN.get(pose_state, pose_state),
                "coordinate": list(part["xyz"]),
                "steps": [t["action"] for t in sub],
            }
        )
        desc.append(
            f"{i + 1}.{part['class']}({POSE_CN.get(pose_state, pose_state)}) → "
            + "→".join(t["action"] for t in sub)
        )

    first = parts[0]
    return {
        "task_id": "t_pack",
        "action": "pack_region",
        "object": first["class"],
        "coordinate": list(first["xyz"]),
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": float((first.get("rpy") or [0, 0, 0])[2]),
        "rpy": list(first.get("rpy") or [0.0, 0.0, 0.0]),
        "retry": 0,
        "max_retry": 3,
        "status": "pending",
        "reason": f"零件最多区域{region_key}共{len(parts)}件，依次装箱（含姿态纠正）",
        "original_cmd": user_cmd,
        "region_key": list(region_key) if region_key is not None else None,
        "region_count": len(parts),
        "box_capacity": capacity,
        "batch": batch,
        "tasks": flat,
        "task_sequence_desc": desc,
        "on_fail": {
            "action": "skip_continue",
            "max_retry": 3,
            "reason": "该件规划/执行失败则跳过，继续下一件",
        },
    }


def build_parts_batch_sequence(
    *,
    parts: list,
    user_cmd: str,
    dest_xyz=None,
    destination: str = None,
    action_name: str = "pack_batch",
    reason: str = "",
    skip_on_fail: bool = True,
):
    """多名/N个/全部零件连续 pick-place；单件规划失败则跳过继续。"""
    dest_name = destination or STORAGE_BOX_NAME
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    selected = []
    skipped = []
    for part in parts or []:
        xyz = part.get("xyz")
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
            skipped.append({"object": part.get("class"), "reason": "无有效坐标，规划失败跳过"})
            continue
        selected.append(part)
    if not selected:
        return {
            "task_id": "t_pack_batch",
            "action": action_name,
            "object": "",
            "coordinate": list(STORAGE_BOX_XYZ),
            "dest_coordinate": dest,
            "destination": dest_name,
            "status": "failed",
            "reason": "没有可规划零件（已全部跳过）",
            "original_cmd": user_cmd,
            "skipped": skipped,
            "skip_on_fail": True,
            "tasks": [],
            "batch": [],
            "task_sequence_desc": [],
        }

    flat = []
    batch = []
    desc = [f"0.perceive 连续抓放{len(selected)}件，失败跳过继续"]
    for i, part in enumerate(selected):
        pose_state = part.get("pose_state") or "normal"
        seq = build_task_sequence(
            object_name=part["class"],
            pick_xyz=part["xyz"],
            pick_rpy=part.get("rpy") or [0.0, 0.0, 0.0],
            destination=dest_name,
            dest_xyz=dest,
            user_cmd=user_cmd,
            reason=f"连续抓放第{i + 1}/{len(selected)}件",
            status="pending",
            do_place=True,
            pose_state=pose_state,
        )
        sub = _renumber_tasks(seq["tasks"], step_offset=len(flat), id_prefix="t")
        if skip_on_fail:
            for t in sub:
                t["skip_on_fail"] = True
                t["on_fail"] = {
                    "action": "skip_continue",
                    "reason": "该件规划/执行失败则跳过，继续下一件",
                }
        flat.extend(sub)
        batch.append(
            {
                "index": i + 1,
                "object": part["class"],
                "object_id": part.get("id"),
                "pose_state": pose_state,
                "pose_state_cn": POSE_CN.get(pose_state, pose_state),
                "coordinate": list(part["xyz"]),
                "steps": [t["action"] for t in sub],
            }
        )
        desc.append(
            f"{i + 1}.{part['class']}({POSE_CN.get(pose_state, pose_state)}) → "
            + "→".join(t["action"] for t in sub)
        )
    first = selected[0]
    return {
        "task_id": "t_pack_batch",
        "action": action_name,
        "object": first["class"],
        "coordinate": list(first["xyz"]),
        "dest_coordinate": dest,
        "destination": dest_name,
        "angle": float((first.get("rpy") or [0, 0, 0])[2]),
        "rpy": list(first.get("rpy") or [0.0, 0.0, 0.0]),
        "retry": 0,
        "max_retry": 3,
        "status": "pending",
        "reason": reason or f"连续抓放{len(selected)}件，失败跳过继续",
        "original_cmd": user_cmd,
        "job_count": len(selected),
        "skipped": skipped,
        "skip_on_fail": True,
        "batch": batch,
        "tasks": flat,
        "task_sequence_desc": desc,
        "on_fail": {
            "action": "skip_continue",
            "reason": "该件规划/执行失败则跳过，继续下一件",
        },
    }


def build_transport_with_fill_sequence(
    *,
    parts_3d: list,
    user_cmd: str = "帮我搬运一下料箱盒",
    filled_count: int = 0,
    capacity: int = None,
    dest_xyz=None,
):
    """
    第⑤点搬运指令：若未满，先从最密区继续装箱至满，再搬运叠放。
    """
    capacity = int(capacity or BOX_CAPACITY)
    dest = list(dest_xyz or STORAGE_BOX_XYZ)
    need = max(0, capacity - int(filled_count or 0))
    flat = []
    batch = []
    desc = []
    pre_pack = None

    if need > 0 and parts_3d:
        from nl_parse import select_densest_region

        region, region_key, _ = select_densest_region(parts_3d)
        pre_pack = build_pack_region_sequence(
            region_parts=region[:need],
            user_cmd=user_cmd,
            dest_xyz=dest,
            capacity=need,
            region_key=region_key,
        )
        flat.extend(pre_pack.get("tasks") or [])
        batch = pre_pack.get("batch") or []
        desc.append(f"0.料箱未满(已装{filled_count}/{capacity})，先补装{len(batch)}件")
        desc.extend(pre_pack.get("task_sequence_desc") or [])

    transport = build_transport_sequence(user_cmd=user_cmd)
    transport_tasks = _renumber_tasks(
        transport["tasks"], step_offset=len(flat), id_prefix="t"
    )
    flat.extend(transport_tasks)
    desc.append(f"{len(batch) + 1}.箱满后搬运叠放 → " + "→".join(t["action"] for t in transport_tasks))

    return {
        "task_id": "t_transport_fill",
        "action": "transport",
        "object": "料箱",
        "coordinate": list(STORAGE_BOX_XYZ),
        "dest_coordinate": list(PLACE_AREA_XYZ),
        "destination": "零件放置区",
        "status": "pending",
        "reason": (
            f"搬运料箱：先补装至满({capacity})再搬运叠放"
            if need > 0
            else "料箱已满，直接搬运叠放"
        ),
        "original_cmd": user_cmd,
        "filled_before": int(filled_count or 0),
        "box_capacity": capacity,
        "box_full": True,
        "pre_pack_count": len(batch),
        "batch": batch,
        "tasks": flat,
        "task_sequence_desc": desc,
    }
