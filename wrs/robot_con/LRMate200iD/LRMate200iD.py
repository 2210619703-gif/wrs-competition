#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/4/8 17:21
# @Author : ZhangXi
"""
FANUC Robot RMI Controller Wrapper (FANUC ER-4iA / LR Mate 200iD)
==================================================================

基于 FANUC Remote Motion Interface (RMI, 手册 B-84184EN/03) 的高层 Python
控制器。通过 TCP/IP 向 R-30iB (Mate) Plus 控制器发送 JSON 指令包来控制机器人，
适用于:
    - ROBOGUIDE 仿真控制器 (Windows 上开发调试, 需加载 R912 RMI 选项)
    - 现场真实 FANUC ER-4iA 控制器 (决赛现场)

对外单位约定:
    - 所有关节角: 弧度 (rad)
    - 所有笛卡尔位置: 米 (m)
    - 姿态: 3x3 旋转矩阵 (与 WRS 一致) 或 FANUC W/P/R (deg)
内部会自动转换成 FANUC 期望的 度(deg) 与 毫米(mm)。

姿态约定说明:
    FANUC 的 W/P/R 是绕固定坐标系 X/Y/Z 的外旋欧拉角 (extrinsic xyz),
    与 WRS `rm.rotmat_*_euler(order='sxyz')` 完全一致, 因此可直接互转。

典型用法::

    from wrs.robot_con.LRMate200iD.LRMate200iD import FanucArmController
    import numpy as np

    with FanucArmController(ip_address="192.168.1.100") as arm:
        arm.set_override(30)                    # 先降速, 现场调试务必低速
        arm.set_uframe_utool(uframe=1, utool=1) # 选择用户坐标系/工具坐标系
        arm.move_j([0, 0, 0, 0, -1.57, 0], speed=20)
        pos = np.array([0.5, 0.0, 0.25])
        rotmat = rm.rotmat_from_euler(np.pi, 0, 0)  # 工具朝下
        arm.move_l(pos, rotmat, speed=100)
        arm.close_gripper()                     # 末端执行器 (DO 口) 抓取
        arm.move_l_rel([0, 0, 0.1])             # 沿 +z 相对抬升 100mm

RMI 关键限制 (来自手册, 现场排错必看):
    - 使用前控制器须: 关闭示教器(TP disabled)、AUTO 模式、无伺服报警、
      当前选中程序不是 RMI_MOVE。
    - 一次最多缓冲 8 条指令; 运动指令的返回包是在该运动执行完成后才返回,
      因此本类默认按 "阻塞" 方式逐条执行 (FINE 终止)。
    - SequenceID 必须从 1 开始且连续递增, 否则进入 HOLD, 需 FRC_Reset。
    - 最后一条运动指令必须是 FINE (或带 NoBlend), 否则 CNT 段不会被执行。
    - 会话结束务必调用 FRC_Abort/FRC_Disconnect, 否则其它 TP 程序无法运行。
"""

from __future__ import annotations

import socket
import json
import time
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import wrs.basis.robot_math as rm
except ImportError:  # 脱离 WRS 环境时的最小回退实现 (与 sxyz 外旋一致)
    from math import atan2, cos, sin, sqrt


    class rm:  # type: ignore
        @staticmethod
        def rotmat_to_euler(rot: np.ndarray, order: str = "sxyz") -> np.ndarray:
            if rot.shape != (3, 3):
                raise ValueError("rot must be a 3x3 matrix")
            sy = -rot[2, 0]
            cy = sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
            pitch = atan2(sy, cy)
            roll = atan2(rot[2, 1], rot[2, 2])
            yaw = atan2(rot[1, 0], rot[0, 0])
            return np.array([roll, pitch, yaw])

        @staticmethod
        def rotmat_from_euler(roll: float, pitch: float, yaw: float,
                              order: str = "sxyz") -> np.ndarray:
            cr, sr = cos(roll), sin(roll)
            cp, sp = cos(pitch), sin(pitch)
            cy, sy = cos(yaw), sin(yaw)
            return np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ])


# RMI 初始握手端口 (固定 16001), 之后使用控制器返回的会话端口
RMI_CONNECT_PORT = 16001

# 良性提示(非故障)错误文本前缀: 控制器在 AUTO 模式下 FRC_Reset/FRC_Initialize
# 可能把这类"状态通知"作为非零 ErrorID 返回, 可安全忽略, 不应中断使能流程。
#   SYST-040  Operation mode AUTO Selected   (仅提示, 正是我们想要的 AUTO 模式)
#   TPIF-157  Menu cannot be displayed        (只是黑白示教器菜单提示, 可忽略)
_BENIGN_ERROR_HINTS = ("SYST-040", "TPIF-157")
# 一个空的 Configuration; Front/Up/Left/Flip/Turn* 让控制器自行选择位形
_DEFAULT_CONFIG = {
    "UToolNumber": 1, "UFrameNumber": 1,
    # ER-4iA 在本任务工作区(容器/金字塔前方, 工具朝下)手动 JOG 时的自然配置。
    # 若某点报 MOTN-018(找不到 IK 解), 通常是该点需要不同的臂配置。
    "Front": 1, "Up": 1, "Left": 0, "Flip": 1,
    "Turn4": 0, "Turn5": 0, "Turn6": 0,
}


class FanucRMIError(RuntimeError):
    """RMI 返回非 0 ErrorID, 或控制器上报 FRC_SystemFault 时抛出。"""

    def __init__(self, message: str, error_id: Optional[int] = None):
        super().__init__(message)
        self.error_id = error_id


class FanucArmController:
    """FANUC 机械臂 RMI 高层控制器。"""

    def __init__(
            self,
            *,
            ip_address: str,
            auto_enable: bool = True,
            gripper_do_port: int = 1,
            uframe: int = 1,
            utool: int = 1,
            recv_timeout: float = 60.0,
            verbose: bool = True,
    ) -> None:
        """
        :param ip_address: 控制器 IP (ROBOGUIDE 本机通常为 127.0.0.1)
        :param auto_enable: 构造时是否自动 reset + FRC_Initialize
        :param gripper_do_port: 控制末端执行器(手爪/吸盘)的数字输出 DO 口号
        :param uframe / utool: 默认用户坐标系 / 工具坐标系编号
        :param recv_timeout: socket 接收超时(秒); 运动指令阻塞期间需足够大
        :param verbose: 是否打印过程信息
        """
        self.ip_address = ip_address
        self.gripper_do_port = gripper_do_port
        self.uframe = uframe
        self.utool = utool
        self.recv_timeout = recv_timeout
        self.verbose = verbose

        self._socket: Optional[socket.socket] = None
        self._buffer = ""           # 跨次接收的粘包缓冲
        self._sequence_id = 1

        self._connect_rmi()
        if auto_enable:
            try:
                self.enable()
            except Exception as e:
                # 使能失败常见于上一次运动故障(MOTN-018)残留, 令 FRC_Initialize
                # 报 INTP-105。先尝试一次软恢复(重连会话), 免去手动冷启动。
                self._log(f"首次使能失败({e}), 尝试软恢复 ...")
                if not self.recover():
                    # 恢复也失败: 干净断开, 否则会在控制器里留下悬挂的 RMI 会话,
                    # 导致下次 FRC_Initialize 报 RMIT-016 (RMI 已被占用/无法启动)。
                    self.close_connection()
                    raise

    # ------------------------------------------------------------------
    # 连接 / 会话管理
    # ------------------------------------------------------------------
    def _log(self, *args) -> None:
        if self.verbose:
            print("[FANUC-RMI]", *args)

    def _connect_rmi(self) -> None:
        """先连 16001 做 FRC_Connect 握手, 再切换到控制器分配的会话端口。"""
        init_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        init_sock.settimeout(5.0)
        try:
            init_sock.connect((self.ip_address, RMI_CONNECT_PORT))
            init_sock.sendall((json.dumps({"Communication": "FRC_Connect"}) + "\r\n").encode("ascii"))
            response = json.loads(init_sock.recv(4096).decode("ascii").strip())
            if response.get("ErrorID", -1) != 0:
                raise FanucRMIError(
                    f"FRC_Connect 被拒绝, ErrorID={response.get('ErrorID')}",
                    response.get("ErrorID"))
            assigned_port = response.get("PortNumber")
            self._log(f"握手成功 v{response.get('MajorVersion')}.{response.get('MinorVersion')}, "
                      f"会话端口={assigned_port}")
        finally:
            init_sock.close()

        time.sleep(0.1)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self.recv_timeout)
        self._socket.connect((self.ip_address, assigned_port))
        self._log(f"已连接到会话端口 {assigned_port}")

    def _send_recv(self, payload: dict) -> dict:
        """发送一个 JSON 包并接收其返回包 (以 \\r\\n 为帧边界, 处理粘包)。"""
        if not self._socket:
            raise FanucRMIError("尚未连接到控制器")
        self._socket.sendall((json.dumps(payload) + "\r\n").encode("ascii"))
        while "\r\n" not in self._buffer:
            chunk = self._socket.recv(4096).decode("ascii")
            if not chunk:
                raise FanucRMIError("连接已断开")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition("\r\n")
        res = json.loads(line.strip())
        # 控制器异步上报的系统故障包
        if res.get("Communication") == "FRC_SystemFault":
            detail = ""
            # 读取控制器真实报警文本 (如 MOTN-049 位置不可达 / MOTN-023 奇异点),
            # 用 _reading_error 防止 read_error 自身再触发 SystemFault 时无限递归。
            if not getattr(self, "_reading_error", False):
                self._reading_error = True
                try:
                    detail = " " + self.read_error(count=3)
                except Exception:
                    pass
                finally:
                    self._reading_error = False
            raise FanucRMIError(
                f"控制器上报 FRC_SystemFault, 出错指令 SequenceID={res.get('SequenceID')}.{detail}")
        return res

    def _check(self, res: dict, what: str) -> dict:
        """校验返回包的 ErrorID; 非 0 抛出 FanucRMIError。"""
        eid = res.get("ErrorID", -1)
        if eid != 0:
            detail = ""
            try:
                detail = " " + self.read_error()
            except Exception:
                pass
            raise FanucRMIError(f"{what} 失败, ErrorID={eid}.{detail}", eid)
        return res

    def _check_tolerant(self, res: dict, what: str) -> dict:
        """同 _check, 但对良性提示(如 SYST-040 AUTO 模式)只告警不抛出。"""
        eid = res.get("ErrorID", -1)
        if eid == 0:
            return res
        detail = ""
        try:
            detail = self.read_error()
        except Exception:
            pass
        if any(hint in detail for hint in _BENIGN_ERROR_HINTS):
            self._log(f"{what} 返回良性提示(忽略): ErrorID={eid} {detail}")
            return res
        raise FanucRMIError(f"{what} 失败, ErrorID={eid}. {detail}", eid)

    def close_connection(self) -> None:
        """安全断开 RMI 会话 (先 Disconnect 再关 socket)。"""
        try:
            if self._socket:
                self._send_recv({"Communication": "FRC_Disconnect"})
        except Exception:
            pass
        finally:
            if self._socket:
                self._socket.close()
                self._socket = None
                self._log("连接已关闭")

    def __enter__(self) -> "FanucArmController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.disable()
        finally:
            self.close_connection()

    # ------------------------------------------------------------------
    # 命令包 (立即生效, 不进入 TP 程序)
    # ------------------------------------------------------------------
    def _soft_reset(self) -> dict:
        """尽力清报警的 FRC_Reset: 即使错误队列残留(良性)告警也只记录不抛出。

        FRC_Reset 只是"尝试清除报警"; 控制器在 AUTO 下常把 SYST-040/TPIF-157
        这类良性通知作为非零 ErrorID 回返, 不应据此中断使能。真正的就绪与否由
        随后的 FRC_Initialize (伺服就绪 / RMI 可启动) 决定。
        """
        res = self._send_recv({"Command": "FRC_Reset"})
        eid = res.get("ErrorID", 0)
        if eid != 0:
            detail = ""
            try:
                detail = self.read_error()
            except Exception:
                pass
            self._log(f"FRC_Reset 残留提示(忽略): ErrorID={eid} {detail}")
        return res

    def enable(self) -> None:
        """清除报警并初始化 RMI_MOVE TP 程序 (使能远程运动)。"""
        self._soft_reset()
        time.sleep(0.3)
        self._check_tolerant(self._send_recv({"Command": "FRC_Initialize", "GroupMask": 1}),
                             "FRC_Initialize")
        self._sequence_id = 1
        time.sleep(0.3)
        # 初始化后设定默认坐标系
        try:
            self.set_uframe_utool(self.uframe, self.utool)
        except FanucRMIError:
            pass
        self._log("已使能, RMI_MOVE 已启动")

    def disable(self) -> None:
        """中止 RMI_MOVE TP 程序 (释放控制权, 便于运行其它 TP 程序)。"""
        try:
            self._send_recv({"Command": "FRC_Abort"})
            self._log("RMI_MOVE 已中止")
        except Exception:
            pass

    def reset(self) -> None:
        """清除控制器报警 (尽力而为, 残留良性告警不抛出)。"""
        self._soft_reset()
        time.sleep(0.3)

    def recover(self, retries: int = 2) -> bool:
        """从运动故障(如 MOTN-018 / INTP-105)软恢复, 无需冷启动控制器。

        原理: 运动故障会让 RMI_MOVE 程序被"中止", 普通 FRC_Reset 清不干净。
        这里做一次"软冷启动"整条会话:
          断开旧会话 -> 重新握手 -> Abort 残留程序 -> 多次 Reset -> 重新 Initialize。
        恢复成功(伺服就绪且 RMI 运行)返回 True; 否则 False(可能仍需手动冷启动)。
        """
        for attempt in range(1, retries + 1):
            self._log(f"软恢复中 (第 {attempt}/{retries} 次) ...")
            try:
                # 1) 彻底断开旧会话: 清掉被中止的 RMI_MOVE 与悬挂 socket
                self.close_connection()
                time.sleep(0.8)
                # 2) 重新握手建立全新会话
                self._connect_rmi()
                time.sleep(0.3)
                # 3) Abort 残留 RMI_MOVE, 多次 Reset 清报警队列
                self.disable()
                time.sleep(0.4)
                for _ in range(2):
                    self._soft_reset()
                    time.sleep(0.4)
                # 4) 重新 Initialize 拉起 RMI_MOVE
                self._check_tolerant(
                    self._send_recv({"Command": "FRC_Initialize", "GroupMask": 1}),
                    "FRC_Initialize")
                self._sequence_id = 1
                time.sleep(0.3)
                try:
                    self.set_uframe_utool(self.uframe, self.utool)
                except FanucRMIError:
                    pass
                if self.is_enabled:
                    self._log(f"软恢复成功 (第 {attempt} 次)")
                    return True
                self._log("软恢复后仍未就绪, 继续重试 ...")
            except Exception as e:
                self._log(f"软恢复第 {attempt} 次失败: {e}")
                time.sleep(1.0)
        self._log("软恢复失败, 可能仍需手动冷启动控制器。")
        return False

    def pause(self) -> None:
        self._check(self._send_recv({"Command": "FRC_Pause"}), "FRC_Pause")

    def resume(self) -> None:
        self._check(self._send_recv({"Command": "FRC_Continue"}), "FRC_Continue")

    def get_status(self) -> dict:
        """返回控制器状态字典 (ServoReady/TPMode/RMIMotionStatus/... )。"""
        return self._check(self._send_recv({"Command": "FRC_GetStatus"}), "FRC_GetStatus")

    @property
    def is_enabled(self) -> bool:
        """RMI 正在运行且伺服就绪时返回 True。"""
        try:
            res = self._send_recv({"Command": "FRC_GetStatus"})
        except Exception:
            return False
        return res.get("ErrorID") == 0 and res.get("ServoReady") == 1 \
            and res.get("RMIMotionStatus") == 1

    def read_error(self, count: int = 1) -> str:
        """读取最近的控制器错误文本 (形如 SRVO-002)。"""
        payload = {"Command": "FRC_ReadError"}
        if count > 1:
            payload["Count"] = int(count)
        res = self._send_recv(payload)
        return str(res.get("ErrorData", ""))

    def set_override(self, percent: int) -> None:
        """设置程序速度倍率 (1~100)。现场调试务必先设小值!"""
        percent = int(np.clip(percent, 1, 100))
        self._check(self._send_recv({"Command": "FRC_SetOverRide", "Value": percent}),
                    "FRC_SetOverRide")
        self._log(f"速度倍率 -> {percent}%")

    def set_uframe_utool(self, uframe: int, utool: int, group: int = 1) -> None:
        """设置当前用户坐标系号与工具坐标系号 (机器人静止时调用)。

        AUTO 模式下控制器常把 SYST-040 等良性通知作为非零 ErrorID 回返, 用
        _check_tolerant 忽略这些无害提示, 避免误判为设置失败而中断流程。
        """
        self._check_tolerant(self._send_recv({
            "Command": "FRC_SetUFrameUTool",
            "UFrameNumber": int(uframe),
            "UToolNumber": int(utool),
            "Group": int(group),
        }), "FRC_SetUFrameUTool")
        self.uframe, self.utool = int(uframe), int(utool)

    # ------------------------------------------------------------------
    # 数字 IO (末端执行器 / 传感器)
    # ------------------------------------------------------------------
    def write_dout(self, port: int, on: bool) -> None:
        """写数字输出口 DO[port] = ON/OFF。"""
        self._check(self._send_recv({
            "Command": "FRC_WriteDOUT",
            "PortNumber": int(port),
            "PortValue": "ON" if on else "OFF",
        }), "FRC_WriteDOUT")

    def read_din(self, port: int) -> int:
        """读数字输入口 DI[port] 的值 (0/1)。"""
        res = self._check(self._send_recv({
            "Command": "FRC_ReadDIN", "PortNumber": int(port),
        }), "FRC_ReadDIN")
        return int(res.get("PortValue", 0))

    def close_gripper(self, wait: float = 0.5) -> None:
        """闭合手爪 / 开启吸盘 (置位 gripper_do_port)。"""
        self.write_dout(self.gripper_do_port, True)
        if wait:
            time.sleep(wait)

    def open_gripper(self, wait: float = 0.5) -> None:
        """张开手爪 / 关闭吸盘 (复位 gripper_do_port)。"""
        self.write_dout(self.gripper_do_port, False)
        if wait:
            time.sleep(wait)

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------
    def get_joint_values(self) -> np.ndarray:
        """读取当前关节角 (弧度, 6 维)。"""
        res = self._check(self._send_recv({"Command": "FRC_ReadJointAngles"}),
                          "FRC_ReadJointAngles")
        j = res.get("JointAngle", {})
        deg = np.array([j.get(f"J{i}", 0.0) for i in range(1, 7)], dtype=float)
        return np.radians(deg)

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """读取当前末端位姿, 返回 (位置[m], 旋转矩阵)。"""
        res = self._check(self._send_recv({"Command": "FRC_ReadCartesianPosition"}),
                          "FRC_ReadCartesianPosition")
        p = res.get("Position", {})
        pos = np.array([p.get("X", 0.0), p.get("Y", 0.0), p.get("Z", 0.0)], dtype=float) / 1000.0
        wpr = np.radians([p.get("W", 0.0), p.get("P", 0.0), p.get("R", 0.0)])
        rot = rm.rotmat_from_euler(wpr[0], wpr[1], wpr[2], order="sxyz")
        return pos, rot

    def get_tcp_speed(self) -> float:
        """读取当前 TCP 速度 (mm/sec)。"""
        res = self._check(self._send_recv({"Command": "FRC_ReadTCPSpeed"}),
                          "FRC_ReadTCPSpeed")
        return float(res.get("Speed", 0.0))

    # ------------------------------------------------------------------
    # 运动指令包 (追加到 RMI_MOVE, 运动完成后才返回 -> 天然阻塞)
    # ------------------------------------------------------------------
    def _next_seq(self) -> int:
        seq = self._sequence_id
        self._sequence_id = 1 if self._sequence_id >= 2147483647 else self._sequence_id + 1
        return seq

    @staticmethod
    def _term(block: bool, term_value: int) -> dict:
        return {"TermType": "FINE" if block else "CNT",
                "TermValue": 0 if block else int(term_value)}

    def _pose_to_position(self, pos_m: np.ndarray, rot: np.ndarray,
                          is_euler: bool) -> dict:
        pos_mm = np.asarray(pos_m, dtype=float) * 1000.0
        if is_euler:  # rot 直接就是 [W, P, R] (弧度)
            wpr_deg = np.degrees(np.asarray(rot, dtype=float))
        else:
            wpr_deg = np.degrees(rm.rotmat_to_euler(np.asarray(rot, dtype=float), order="sxyz"))
        return {
            "X": float(pos_mm[0]), "Y": float(pos_mm[1]), "Z": float(pos_mm[2]),
            "W": float(wpr_deg[0]), "P": float(wpr_deg[1]), "R": float(wpr_deg[2]),
            "Ext1": 0.0, "Ext2": 0.0, "Ext3": 0.0,
        }

    def _config(self, override: Optional[dict] = None) -> dict:
        cfg = dict(_DEFAULT_CONFIG)
        cfg["UFrameNumber"] = self.uframe
        cfg["UToolNumber"] = self.utool
        if override:
            for k in ("Front", "Up", "Left", "Flip", "Turn4", "Turn5", "Turn6"):
                if k in override:
                    cfg[k] = override[k]
        return cfg

    def read_cartesian_raw(self) -> dict:
        """读取当前笛卡尔位姿的完整返回包 (含 Configuration 手臂配置标志)。"""
        return self._check(self._send_recv({"Command": "FRC_ReadCartesianPosition"}),
                           "FRC_ReadCartesianPosition")

    def get_config(self) -> dict:
        """读取当前机器人手臂配置 (Front/Up/Left/Flip/Turn4~6)。

        绝对笛卡尔下发时, 应尽量沿用机器人当前配置, 避免控制器解不出目标配置
        的关节解而抛 MOTN 报警 (进而触发 FRC_SystemFault)。
        """
        return dict(self.read_cartesian_raw().get("Configuration", {}))

    def move_j(
            self,
            joint_angles: Iterable[float],
            *,
            is_radians: bool = True,
            speed: int = 30,
            block: bool = True,
            term_value: int = 100,
    ) -> None:
        """关节运动到给定关节配置 (关节表示 FRC_JointMotionJRep)。speed 为百分比。"""
        angles = np.asarray(list(joint_angles), dtype=float)
        if angles.size < 6:
            raise ValueError("joint_angles 至少需要 6 个值")
        deg = np.degrees(angles) if is_radians else angles
        payload = {
            "Instruction": "FRC_JointMotionJRep",
            "SequenceID": self._next_seq(),
            "JointAngle": {**{f"J{i + 1}": float(deg[i]) for i in range(6)},
                           "J7": 0.0, "J8": 0.0, "J9": 0.0},
            "SpeedType": "Percent",
            "Speed": int(speed),
            **self._term(block, term_value),
        }
        self._check(self._send_recv(payload), "move_j")

    def move_j_cart(
            self,
            pos: Iterable[float],
            rot: np.ndarray,
            *,
            is_euler: bool = False,
            speed: int = 30,
            block: bool = True,
            term_value: int = 100,
            config: Optional[dict] = None,
    ) -> None:
        """关节运动到给定笛卡尔位姿 (FRC_JointMotion, 姿态用位置表示)。speed 为百分比。

        config: 手臂配置覆盖 (Front/Up/Left/...); 建议传入 get_config() 的当前配置。
        """
        payload = {
            "Instruction": "FRC_JointMotion",
            "SequenceID": self._next_seq(),
            "Configuration": self._config(config),
            "Position": self._pose_to_position(np.asarray(list(pos), float), rot, is_euler),
            "SpeedType": "Percent",
            "Speed": int(speed),
            **self._term(block, term_value),
        }
        self._check(self._send_recv(payload), "move_j_cart")

    def move_l(
            self,
            pos: Iterable[float],
            rot: np.ndarray,
            *,
            is_euler: bool = False,
            speed: int = 100,
            block: bool = True,
            term_value: int = 100,
            config: Optional[dict] = None,
    ) -> None:
        """直线运动到给定笛卡尔位姿 (FRC_LinearMotion)。speed 单位 mm/sec。

        config: 手臂配置覆盖 (Front/Up/Left/...); 建议传入 get_config() 的当前配置。
        """
        position = np.asarray(list(pos), dtype=float)
        if position.size != 3:
            raise ValueError("pos 必须是 3 维坐标")
        payload = {
            "Instruction": "FRC_LinearMotion",
            "SequenceID": self._next_seq(),
            "Configuration": self._config(config),
            "Position": self._pose_to_position(position, rot, is_euler),
            "SpeedType": "mmSec",
            "Speed": int(speed),
            **self._term(block, term_value),
        }
        self._check(self._send_recv(payload), "move_l")

    def move_l_rel(
            self,
            d_pos: Iterable[float],
            d_rot_euler=(0.0, 0.0, 0.0),
            *,
            speed: int = 100,
            block: bool = True,
            term_value: int = 100,
    ) -> None:
        """相对(增量)直线运动 (FRC_LinearRelative)。d_pos[m], d_rot_euler[rad]。"""
        payload = {
            "Instruction": "FRC_LinearRelative",
            "SequenceID": self._next_seq(),
            "Configuration": self._config(),
            "Position": self._pose_to_position(np.asarray(list(d_pos), float),
                                               np.asarray(d_rot_euler, float), is_euler=True),
            "SpeedType": "mmSec",
            "Speed": int(speed),
            **self._term(block, term_value),
        }
        self._check(self._send_recv(payload), "move_l_rel")

    def move_j_rel(
            self,
            d_joint_angles: Iterable[float],
            *,
            is_radians: bool = True,
            speed: int = 30,
            block: bool = True,
            term_value: int = 100,
    ) -> None:
        """相对(增量)关节运动 (FRC_JointRelative)。speed 为百分比。"""
        d = np.asarray(list(d_joint_angles), dtype=float)
        if d.size < 6:
            raise ValueError("d_joint_angles 至少需要 6 个值")
        d_deg = np.degrees(d) if is_radians else d
        # FRC_JointRelative 用 Position/Configuration 承载增量; 这里用关节表示更直观:
        payload = {
            "Instruction": "FRC_JointRelative",
            "SequenceID": self._next_seq(),
            "Configuration": self._config(),
            "Position": {
                "X": float(d_deg[0]), "Y": float(d_deg[1]), "Z": float(d_deg[2]),
                "W": float(d_deg[3]), "P": float(d_deg[4]), "R": float(d_deg[5]),
                "Ext1": 0.0, "Ext2": 0.0, "Ext3": 0.0,
            },
            "SpeedType": "Percent",
            "Speed": int(speed),
            **self._term(block, term_value),
        }
        self._check(self._send_recv(payload), "move_j_rel")

    def wait_time(self, seconds: float) -> None:
        """向 TP 程序追加等待指令 WAIT seconds。"""
        self._check(self._send_recv({
            "Instruction": "FRC_WaitTime",
            "SequenceID": self._next_seq(),
            "Time": float(seconds),
        }), "wait_time")

    # ------------------------------------------------------------------
    # 位姿寄存器 (可配合视觉偏移 Offset,PR[] 使用)
    # ------------------------------------------------------------------
    def write_position_register(self, reg: int, pos: Iterable[float], rot: np.ndarray,
                                *, is_euler: bool = False, group: int = 1) -> None:
        self._check(self._send_recv({
            "Command": "FRC_WritePositionRegister",
            "RegisterNumber": int(reg),
            "Configuration": self._config(),
            "Position": self._pose_to_position(np.asarray(list(pos), float), rot, is_euler),
            "Group": int(group),
        }), "FRC_WritePositionRegister")


# 通用别名 (机器人型号无关)
FanucRMIController = FanucArmController


if __name__ == "__main__":
    # ROBOGUIDE 本机仿真默认 IP; 真机改为控制器实际 IP
    ip = "127.0.0.1"
    print("创建 FanucArmController...")
    try:
        with FanucArmController(ip_address=ip, verbose=True) as arm:
            print("is_enabled:", arm.is_enabled)
            print("状态:", arm.get_status())
            print("当前关节角(rad):", arm.get_joint_values())
            pos, rot = arm.get_pose()
            print("当前末端位置(m):", pos)
            # 低速冒烟测试 (确认现场安全后再取消注释)
            # arm.set_override(10)
            # arm.move_l_rel([0, 0, 0.02], speed=50)  # 沿 +z 抬 20mm
        print("会话正常结束。")
    except Exception as e:
        print(f"请确认 FANUC 控制器 IP 可达且已加载 RMI(R912) 选项。错误: {e}")
