# -*- coding: utf-8 -*-
"""fafu 真机的薄封装 + plan 执行器。

对上只暴露执行模块需要的几个动作（回 HOME、走关节路径、开合夹爪、急停），
对下调用 ``FafuRobotController``。所有下发都先过 ``robot.cfg`` 软限位，
并且默认是 **dry-run**：不连硬件，只把要下发的指令打印出来。
真的要动机械臂必须显式 ``dry_run=False``。

SDK 是懒加载的：dry-run 模式下不会 import ``fafu_motor``，
所以在没有硬件、没有编译好 ``.pyd`` 的开发机上也能跑通整条链路。
"""
from __future__ import annotations

import contextlib
import math
import sys
import threading
import time

from real_config import (
    CARRY_LIFT_MIN_RAD,
    DEFAULT_MOVE_SPEED,
    EXIT_HOME_CONF,
    GRASP_EFFORT_RAW,
    GRASP_FORCE_THRESHOLD,
    GRASP_OVERDRIVE_RAD,
    GRIPPER_ACC,
    GRIPPER_VEL,
    HOLD_DRIFT_TOL_RAD,
    HOLD_RETURN_VEL_RAD_S,
    HOLD_TORQUE_RAW,
    JAW_OPEN_ON_ARRIVE,
    MAX_MOVE_SPEED,
    MAX_STEP_RAD,
    MOVE_J_TIMEOUT,
    MOVE_J_TOLERANCE_RAD,
    PARK_CONF,
    PLACE_CONF,
    SERVO_FINISH_MODE,
    SERVO_ACC_RAD_S2,
    SERVO_JERK_RAD_S3,
    SERVO_LOOKAHEAD_S,
    SERVO_MAX_LAG_RAD,
    SERVO_MAX_STEP_RAD,
    SERVO_MAX_VEL_RAD_S,
    SERVO_RATE_HZ,
    SERVO_SETTLE_TIMEOUT,
    SERVO_SETTLE_TOL_RAD,
    SERVO_USE_MIT,
    SERVO_VEL_AT_FULL_SPEED,
    SERVO_IDLE_WATCHDOG_MS,
    SERVO_WATCHDOG_MS,
    SIM_HOME_CONF,
    STREAM_CONTROL_FREQ,
    ZERO_CONF,
    STREAM_SETTLE_TIMEOUT,
    RELEASE_MODE_ON_CLOSE,
    RELEASE_MODES,
    USE_SERVO_J,
    RealRobotConfig,
    ensure_sdk_on_path,
    load_real_config,
)


def _bump_timer_resolution() -> None:
    """把 Windows 定时粒度提到 1ms。

    默认约 15ms，``time.sleep`` 撑不住 100Hz 喂帧：节奏一乱，帧间距忽大忽小，
    动作反而更抖，严重时还会超过看门狗时限被刹车。
    SDK 的 servo_j 示例也是这么做的。
    """
    if sys.platform != "win32":
        return
    try:
        import atexit  # noqa: PLC0415
        import ctypes  # noqa: PLC0415

        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod(1)
        atexit.register(winmm.timeEndPeriod, 1)
    except Exception:
        pass


def _fmt(q) -> str:
    return "[" + ", ".join(f"{math.degrees(v):7.2f}" for v in q) + "]°"


def _patch_stale_fafu_motor(logger=print) -> None:
    """旧版 ``fafu_motor.pyd`` 没有导出 TORQUE_COEFF / FAULT_TABLE。

    Python 控制器会因此直接 ImportError。位置控制（move_j）本来也不读这张表，
    缺的只是 import 时的校验。这里按 C++ ``hightorque_protocol.cpp`` 现表补上，
    让真机能先连上；有编译环境时仍应重编 ``fafu_robot_cpp``。
    """
    import fafu_motor as pm  # noqa: PLC0415

    if getattr(pm, "TORQUE_COEFF", None) and getattr(pm, "FAULT_TABLE", None):
        return

    logger(
        "[real] 当前 fafu_motor.pyd 偏旧（缺 TORQUE_COEFF），"
        "已按协议源码补表以便连接；有空请重编 fafu_robot_cpp"
    )

    if not getattr(pm, "TORQUE_COEFF", None):
        pm.TORQUE_COEFF = {
            "M3508_02": 0.37,
            "M3516_02": 0.37,
            "M3532_02": 0.37,
            "M4530_02": 0.62,
            "M5009_02": 0.71,
            "M5036_02": 0.67,
            "M6036_02": 0.66,
            "M7033_04": 0.84,
            "M7535_02": 0.73,
            "M3532_02_8353": 0.61,
            "M4530_02_8353": 0.64,
            "M5036_02_8353": 0.70,
            "M3536_32": 0.35,
            "M4438_30": 0.64,
            "M4438_32": 0.64,
            "M5043_20": 0.96,
            "M5047_36": 0.64,
            "M6056_36": 0.66,
            "M7256_35": 0.66,
            "M60BM_35": 0.64,
            "MGENERAL": 0.65,
            "MNONE": 1.0,
            "M4538_19": 0.4450,
            "M5046_20": 0.5280,
            "M5047_09": 0.5330,
            "M60SG_35": 0.7942,
        }

    class _FaultInfo:
        def __init__(self, name: str, detail: str = "", hint: str = "") -> None:
            self.name = name
            self.detail = detail
            self.hint = hint

    if not getattr(pm, "FAULT_TABLE", None):
        pm.FAULT_TABLE = {
            0: _FaultInfo("正常", "操作成功，没有错误发生"),
            1: _FaultInfo("DMA 数据流传输错误", "硬件问题…"),
            2: _FaultInfo("DMA 数据流 FIFO 错误", "硬件问题…"),
            3: _FaultInfo("UART 溢出错误", "硬件问题…"),
            4: _FaultInfo("UART 帧错误", "硬件问题…"),
            5: _FaultInfo("UART 噪声错误", "硬件问题…"),
            6: _FaultInfo("UART 缓冲区溢出错误", "硬件问题…"),
            7: _FaultInfo("UART 奇偶校验错误", "硬件问题…"),
            32: _FaultInfo("校准故障", "校准过程中，编码器无法感知到磁铁"),
            33: _FaultInfo("电机驱动故障", "多为欠压，电流不足"),
            34: _FaultInfo("过压", "母线电压过大"),
            35: _FaultInfo("编码器故障", "编码器读数错误"),
            36: _FaultInfo("电机未校准", "电机还未进行校准（电机出厂都会校准一次）"),
            37: _FaultInfo("PWM周期过限", "一般是内部固件错误"),
            38: _FaultInfo("温度过高", "已超过最大配置温度"),
            39: _FaultInfo("起始位置超出限制", "在位置界限之外尝试启动位置控制（出厂默认无位置限制）"),
            40: _FaultInfo("电压过低", "电压太低"),
            41: _FaultInfo("配置已更改", "在操作期间更改了需要停止的配置值",
                           "改完配置要 conf write 再重启电机才生效"),
            42: _FaultInfo("角度无效", "没有可用的有效换相编码器"),
            43: _FaultInfo("位置无效", "没有可用的有效输出编码器"),
            44: _FaultInfo("驱动器使能故障", "驱动芯片异常"),
            45: _FaultInfo("停止位置使用错误",
                           "程序不支持在设置停止位置的同时，还设置加速度或速度限制"),
            46: _FaultInfo("时序错误", "系统检测到操作或事件未在预期的时间窗口内完成"),
            47: _FaultInfo("反电动势前馈错误", "使用反电动势前馈必须使用加速度限制"),
        }

    if not getattr(pm, "describe_fault", None):
        def describe_fault(code: int) -> str:
            info = pm.FAULT_TABLE.get(int(code))
            if info is None:
                return f"未定义故障码 {code} (厂商表3 未收录)"
            text = f"{code} {info.name}"
            if info.detail:
                text += f": {info.detail}"
            if info.hint:
                text += f" [{info.hint}]"
            return text

        pm.describe_fault = describe_fault


class RealArmError(RuntimeError):
    pass


_MODE_POSITION = 0x0A   # 电机位置/速度/力矩模式，回包 mode 字段核对用
_MODE_MIT = 0x0B        # servo 会话残留的 MIT 模式；这个状态下固件会忽略位置帧


class RealArm:
    """真机执行器。用法::

        with RealArm(dry_run=False) as arm:
            arm.go_sim_home()
            arm.follow_path(waypoints)
    """

    def __init__(
        self,
        cfg: RealRobotConfig | None = None,
        *,
        dry_run: bool = True,
        speed: int = DEFAULT_MOVE_SPEED,
        port: str | None = None,
        logger=print,
        stream_path: bool = False,
        control_freq: float = STREAM_CONTROL_FREQ,
        release_mode: str = RELEASE_MODE_ON_CLOSE,
        use_servo: bool = USE_SERVO_J,
    ) -> None:
        self.cfg = cfg or load_real_config()
        self.dry_run = bool(dry_run)
        self.speed = max(1, min(int(speed), MAX_MOVE_SPEED))
        self.port = port
        self.log = logger
        # 整段路径流式下发（连贯）而不是逐点阻塞（一卡一卡）
        self.stream_path = bool(stream_path)
        self.control_freq = max(0.005, float(control_freq))
        if release_mode not in RELEASE_MODES:
            raise ValueError(
                f"release_mode 只能是 {'/'.join(RELEASE_MODES)}，收到 {release_mode!r}"
            )
        self.release_mode = release_mode
        # servo_j 在线流式：自己按固定频率喂中间点，动作最连贯
        self.use_servo = bool(use_servo)
        self._can_servo = False
        self._arm = None
        self._move_style = "acc"
        self._can_stream = False
        self._servo_drive_depth = 0
        self._servo_lock = threading.Lock()
        self._hold_q: list[float] | None = None
        self._hold_send: list[float] | None = None
        self._hold_crawl_noted = False
        self._hold_stop = threading.Event()
        self._hold_paused = False
        self._hold_thread: threading.Thread | None = None
        self._sim_q = list(SIM_HOME_CONF[: self.cfg.num_joints])  # dry-run 下的虚拟位形

    def set_virtual_q(self, q) -> None:
        """dry-run 时把虚拟位形接到指定姿态（比如已经回避让位）。"""
        self._sim_q = list(q)[: self.cfg.num_joints]

    # -- 连接 ---------------------------------------------------------------
    def connect(self) -> "RealArm":
        if self.dry_run:
            self.log(f"[dry-run] 不连接硬件；限位来自 {self.cfg.cfg_path}")
            return self
        if self._arm is not None:
            return self
        ensure_sdk_on_path()
        _patch_stale_fafu_motor(self.log)
        from fafu_robot_controller import FafuRobotController  # noqa: PLC0415

        self.log(f"[real] 连接 fafu 机械臂 port={self.port or self.cfg.port} ...")
        last_err = None
        for attempt in range(1, 4):
            try:
                self._arm = FafuRobotController(
                    cfg_path=self.cfg.cfg_path,
                    port=self.port or None,
                    has_gripper=True,
                    gripper_motor_id=self.cfg.gripper_id,
                    auto_enable=True,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                self._arm = None
                msg = str(e)
                retryable = "did not respond" in msg or "500ms" in msg
                if (not retryable) or attempt >= 3:
                    raise
                self.log(
                    f"[real] 电机还没醒（第 {attempt} 次），0.8s 后再连：{msg}"
                )
                time.sleep(0.8)
        if last_err is not None:
            raise last_err
        driver = getattr(self._arm, "_ht", None)
        if driver is not None and not hasattr(driver, "set_many_pos_vel_acc"):
            self._move_style = "linear"
            self.log(
                "[real] 旧扩展没有 set_many_pos_vel_acc，move_j 改用 style=linear"
            )
        # 流式下发走 0x8090 位置通道（set_many_pos_vel_tqe），和 set_many_pos_vel_acc
        # 无关，所以旧扩展一样能用；真缺了就退回逐点阻塞。
        self._can_stream = hasattr(self._arm, "move_jntspace_path") and (
            driver is None or hasattr(driver, "set_many_pos_vel_tqe")
        )
        if self.stream_path and not self._can_stream:
            self.log("[real] 扩展不支持流式下发，退回逐点阻塞 move_j")
        elif self.stream_path:
            self.log(
                f"[real] 路径连续下发已开启（帧周期 {self.control_freq * 1000:.0f}ms）"
            )
        # servo_j 会话：需要 servo_start / servo_j / servo_end 三件套齐全
        self._can_servo = all(
            hasattr(self._arm, name) for name in ("servo_start", "servo_j", "servo_end")
        )
        if self.use_servo and not self._can_servo:
            self.log("[real] SDK 不支持 servo_j，退回 move_j")
        elif self.use_servo:
            _bump_timer_resolution()
            self.log(
                f"[real] servo_j 在线流式已开启（{SERVO_RATE_HZ:.0f}Hz，"
                f"看门狗 {SERVO_WATCHDOG_MS}ms）"
            )
        self.log(f"[real] 已连接，当前位形 {_fmt(self.joint_values())}")
        return self

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and not self.dry_run

    def _true_hold_and_close(self) -> bool:
        """自己把关节顶住再关串口，绕过 SDK 有 bug 的 hold 路径。

        SDK 的 ``close_connection(joint_release="hold")`` 会调
        ``set_motor_mode(mid, 0x0A)``，而 C++ 里这条路径做了两件要命的事：
        先**无条件** ``stop()`` 切断 PWM（注释自称"5ms 沉降"，但紧跟一个
        超时 200ms 的 ``read_motor_state``），然后补一帧
        ``build_pos_vel_tqe_int16(pos, 0, tqe=0)`` —— 字面量 0 在协议里是
        "最大力矩 = 0"，等于没力气。结果电机停在 0x0A 却顶不住重力，
        比 brake 还差：brake 至少有三相短路阻尼。

        这里改走公共 API ``set_pos_vel_tqe``：它会把 ``tqe_raw=0`` 翻译成
        ``NAN_INT16``（用电机默认最大力矩），位置取电机当前实测位置、速度 0。
        这是关串口前最后一条指令，不会再被模式切换覆盖。

        返回 ``False`` 表示这条路走不通，调用方回退到 SDK 的标准流程。
        """
        drv = getattr(self._arm, "driver", None) or getattr(self._arm, "_ht", None)
        if drv is None or not hasattr(drv, "set_pos_vel_tqe"):
            self.log("[real] 扩展没有 set_pos_vel_tqe，hold 退回 SDK 路径")
            return False

        # servo 会话还开着的话先正常收尾，否则看门狗会在下面切模式时插进来
        if getattr(self._arm, "_servo_active", False):
            try:
                self._arm.servo_end("hold")
            except Exception as e:
                self.log(f"[real] servo_end 失败：{e}")

        # set_pos_vel_tqe 会对目标位置做软限位裁剪。有关节停在限位外时，
        # 裁剪后的目标和当前位置不同，补这一帧会让它猛地弹回限位内 ——
        # 这正是 SDK 那条路径宁可绕过公共 API 的原因。宁可退回 brake。
        margin = math.radians(1.0)
        for i, (v, (lo, hi)) in enumerate(
            zip(self.joint_values(), self.cfg.joint_limits)
        ):
            if v < lo - margin or v > hi + margin:
                self.log(
                    f"[warn] J{i + 1} 在软限位外（{math.degrees(v):.1f}°），"
                    "补帧会被裁剪后猛动，hold 退回 SDK 路径"
                )
                return False

        # 位置用电机侧原始读数（turns），不做单位换算，避免减速比假设出错
        positions: dict[int, float] = {}
        for mid in self.cfg.joint_ids:
            st = None
            if hasattr(drv, "get_cached_state"):
                st = drv.get_cached_state(mid)
            if st is None and hasattr(drv, "read_motor_state"):
                st = drv.read_motor_state(mid, 0.2)
            if st is None:
                self.log(f"[real] 读不到电机 {mid} 状态，hold 退回 SDK 路径")
                return False
            positions[mid] = float(st.position)

        # 后台线程先停掉（和 close_connection 同样的顺序），避免轮询帧
        # 插在我们的保持帧后面
        for check, stop in (("is_polling", "stop_state_polling"),
                            ("is_async_rx", "disable_async_rx")):
            try:
                if getattr(drv, check)():
                    getattr(drv, stop)()
            except Exception:
                pass

        # 夹爪也主动顶住，否则手上夹着的件会在断开后松掉。用抓取那套力矩上限。
        jaw_cap = self.grasp_effort()
        jaw_held = False
        try:
            jst = drv.get_cached_state(self.cfg.gripper_id)
            if jst is None:
                jst = drv.read_motor_state(self.cfg.gripper_id, 0.2)
            if jst is not None:
                drv.set_pos_vel_tqe(
                    self.cfg.gripper_id, float(jst.position), 0.0, jaw_cap
                )
                jaw_held = True
        except Exception as e:
            self.log(f"[real] 夹爪保持失败，退回 brake：{e}")
        if not jaw_held:
            try:
                drv.brake(self.cfg.gripper_id)
            except Exception:
                pass

        # 关节保持帧：必须是关串口前的最后一批指令。
        # 即时回包的 mode 经常还是 0x0B，不能据此去 brake——那会把保持帧撤掉。
        held: list[int] = []
        failed: list[int] = []
        lagged: list[int] = []
        for mid, pos in positions.items():
            try:
                _ok, mode = self._send_hold_frame(drv, mid, pos, HOLD_TORQUE_RAW)
            except Exception as e:
                self.log(f"[real] 电机 {mid} 保持帧失败：{e}")
                failed.append(mid)
                continue
            held.append(mid)
            if mode == _MODE_MIT:
                lagged.append(mid)

        if lagged:
            self.log(
                f"[real] 保持帧已发给 M{lagged}，回包仍是 0x0B（按读数滞后，不撤成 brake）"
            )
        if failed:
            desc = ", ".join(f"M{mid}" for mid in failed)
            self.log(f"[warn] 这些电机保持帧没发出去，改用 brake：{desc}")
            for mid in failed:
                try:
                    drv.brake(mid)
                except Exception:
                    pass
        if not held:
            self.log("[warn] 所有保持帧都没发出去，已全部退到 brake")

        try:
            drv.close()
        except Exception as e:
            self.log(f"[real] 关串口失败：{e}")

        cap_note = "电机默认上限" if HOLD_TORQUE_RAW == 0 else f"上限 {HOLD_TORQUE_RAW}"
        self.log(
            f"[real] 已断开连接（关节主动顶住 {len(held)}/{len(positions)} 个"
            f"），{cap_note}；"
            f"夹爪{'顶住 effort=' + str(jaw_cap) if jaw_held else ' brake'}）"
        )
        if held:
            self.log("[real] 提示：hold 会持续耗电发热，别长时间挂着")
        return True

    def close(self) -> None:
        if self._arm is None:
            return
        try:
            self._stop_joint_hold()
            # hold 档走自己的实现：SDK 的 hold 会先切断 PWM 再补一帧
            # tqe=0（没力气），实测比 brake 还容易下坠。
            if self.release_mode == "hold":
                try:
                    if self._true_hold_and_close():
                        return
                except Exception as e:
                    self.log(f"[real] 自定义 hold 出错，退回 SDK 路径：{e}")
            # stop 会切断 PWM，关节被重力带着往下沉，停在避让位或半空时就是
            # "摔一下"。brake 是短路阻尼：不耗电，但阻止运动，姿态能大致保持住。
            self._arm.close_connection(
                joint_release=self.release_mode, gripper_release="brake"
            )
            self.log(f"[real] 已断开连接（关节 {self.release_mode}，夹爪 brake）")
        finally:
            self._arm = None

    def __enter__(self) -> "RealArm":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self._arm is not None:
            try:
                self._arm.emergency_stop()
                self.log("[real] 执行异常，已急停")
            except Exception as stop_err:  # 急停本身失败不能掩盖原异常
                self.log(f"[real] 急停失败：{stop_err}")
        self.close()

    # -- 状态 ---------------------------------------------------------------
    def joint_values(self) -> list[float]:
        if self._arm is None:
            return list(self._sim_q)
        return [float(v) for v in self._arm.get_joint_values()]

    # -- 运动 ---------------------------------------------------------------
    @property
    def servo_ready(self) -> bool:
        """真机连着、SDK 支持、开关也打开了，才走 servo_j。"""
        return self.use_servo and self._can_servo and self._arm is not None

    def _send_hold_frame(self, drv, mid: int, pos: float, tqe: int) -> tuple[bool, int]:
        """发一帧位置保持，等模式在固件里生效后再读一次。

        ``set_pos_vel_tqe`` 的即时回包经常还是切模式前的旧值（C++ 注释也写了
        immediate ack 不可靠）。拿它当失败去 ``brake``，等于把刚发出去的
        保持帧撤掉——日志里「顶住 0/6、改用 brake」就是这么来的。
        """
        drv.set_pos_vel_tqe(mid, float(pos), 0.0, int(tqe))
        time.sleep(0.015)
        st = None
        if hasattr(drv, "read_motor_state"):
            st = drv.read_motor_state(mid, 0.15)
        if st is None and hasattr(drv, "get_cached_state"):
            st = drv.get_cached_state(mid)
        mode = int(getattr(st, "mode", -1)) if st is not None else -1
        # 发成功就算顶住了。回包仍是 0x0B 只说明读数滞后，不能据此去 brake。
        return True, mode

    def _servo_j(self, q, vel_ff=None) -> bool:
        """线程互斥的 servo_j。``vel_ff`` 为每关节 rad/s，用来带速度前馈。"""
        with self._servo_lock:
            if vel_ff is not None and self._arm is not None:
                dt = 1.0 / SERVO_RATE_HZ
                two_pi = 2.0 * math.pi
                last = [
                    (float(qi) - float(vi) * dt) / two_pi
                    for qi, vi in zip(q, vel_ff)
                ]
                if hasattr(self._arm, "_servo_last_target_turns"):
                    self._arm._servo_last_target_turns = list(last)
                if hasattr(self._arm, "_servo_filtered_target_turns"):
                    self._arm._servo_filtered_target_turns = list(last)
            return bool(self._arm.servo_j(list(q)))

    def _cached_joints(self) -> list[float] | None:
        if self._arm is None:
            return None
        try:
            getter = getattr(self._arm, "get_joint_values", None)
            if getter is None:
                return None
            q = getter(prefer_cache=True)
            return [float(v) for v in q]
        except TypeError:
            try:
                return [float(v) for v in self._arm.get_joint_values()]
            except Exception:
                return None
        except Exception:
            return None

    def _hold_frame_target(self) -> list[float] | None:
        """保持帧只发命令角，绝不跟实测坠落走。

        以前偏离时从实测角往回爬：第一帧相对 last_target 就是猛落，再爬回去
        就是猛抬。现在命令角已经在目标上就继续顶住；只有命令角本身离开目标
        （例如上一段路径被掐断）才按 8°/s 收回。
        """
        goal = self._hold_q
        if goal is None:
            return None
        src = self._hold_send if self._hold_send is not None else list(goal)
        cmd_err = max(abs(a - b) for a, b in zip(src, goal))
        dt = 1.0 / SERVO_RATE_HZ
        step = HOLD_RETURN_VEL_RAD_S * dt
        if cmd_err <= 1e-6:
            self._hold_send = list(goal)
            meas = self._cached_joints()
            if meas is not None:
                err = max(abs(a - b) for a, b in zip(meas, goal))
                if err > HOLD_DRIFT_TOL_RAD and not self._hold_crawl_noted:
                    self._hold_crawl_noted = True
                    self.log(
                        f"[real] 保持偏离 {math.degrees(err):.1f}°，"
                        "继续顶住命令角，不跟坠落"
                    )
                elif err <= HOLD_DRIFT_TOL_RAD:
                    self._hold_crawl_noted = False
            return list(goal)
        if not self._hold_crawl_noted:
            self._hold_crawl_noted = True
            self.log(
                f"[real] 命令角离目标 {math.degrees(cmd_err):.1f}°，"
                f"按 {math.degrees(HOLD_RETURN_VEL_RAD_S):.0f}°/s 收回"
            )
        out = []
        for a, b in zip(src, goal):
            d = b - a
            out.append(b if abs(d) <= step else a + math.copysign(step, d))
        self._hold_send = out
        return out

    def _hold_joints_tick(self) -> None:
        if self._arm is None or not getattr(self._arm, "_servo_active", False):
            return
        q = self._hold_frame_target()
        if q is None:
            return
        self._servo_j(q)

    def _pause_hold_for_gripper(self):
        """夹爪动作不再停续帧。关节和夹爪走不同电机，续帧必须一直发。"""
        self._ensure_servo_started()
        self._set_joint_watchdog(SERVO_IDLE_WATCHDOG_MS)

    def hold_here(self, label: str = "当前位置") -> None:
        """把当前命令角锁成保持目标，之后只续这一帧，不再回抽。"""
        self._ensure_servo_started()
        if self._hold_q is None:
            self._hold_q = list(self.joint_values())
        self._hold_send = list(self._hold_q)
        self._hold_crawl_noted = False
        self._set_servo_caps(HOLD_RETURN_VEL_RAD_S)
        self._set_joint_watchdog(SERVO_IDLE_WATCHDOG_MS)
        self._start_joint_hold()
        self.log(f"[real] 已在{label}保持（续帧顶住，偏离只慢速收回）")

    def _set_joint_watchdog(self, ms: int) -> None:
        if self._arm is None:
            return
        drv = getattr(self._arm, "driver", None) or getattr(self._arm, "_ht", None)
        if drv is None or not hasattr(drv, "set_timeout"):
            return
        timeout = max(0, int(ms))
        for mid in self.cfg.joint_ids:
            try:
                drv.set_timeout(mid, timeout)
            except Exception:
                pass

    def _start_joint_hold(self) -> None:
        """段与段之间、等提示符时继续喂**上次命令角**，不要用实测角。"""
        if self._arm is None:
            return
        if self._hold_thread is not None and self._hold_thread.is_alive():
            return
        self._hold_stop.clear()
        self._hold_paused = False

        def loop() -> None:
            dt = 1.0 / SERVO_RATE_HZ
            t_next = time.perf_counter()
            while not self._hold_stop.is_set():
                now = time.perf_counter()
                if now < t_next:
                    time.sleep(min(dt, t_next - now))
                    continue
                t_next = now + dt
                if self._hold_paused or self._servo_drive_depth > 0:
                    continue
                if self._arm is None or not getattr(self._arm, "_servo_active", False):
                    break
                q = self._hold_frame_target()
                if q is None:
                    continue
                try:
                    self._servo_j(q)
                except Exception:
                    break

        self._hold_thread = threading.Thread(
            target=loop, daemon=True, name="joint-hold"
        )
        self._hold_thread.start()

    def _stop_joint_hold(self) -> None:
        self._hold_stop.set()
        t = self._hold_thread
        if t is not None and t.is_alive():
            t.join(timeout=0.6)
        self._hold_thread = None

    def _ensure_servo_started(self) -> None:
        """整段连接共用一次 servo 会话，避免每段路径 servo_end 后再 start 触发 motor_reset。"""
        if self._arm is None or not self.servo_ready:
            return
        if getattr(self._arm, "_servo_active", False):
            self._start_joint_hold()
            return
        try:
            from fafu_robot_controller import ServoOpts  # noqa: PLC0415
        except Exception:
            from types import SimpleNamespace as ServoOpts  # noqa: N814
        self._arm.servo_start(
            ServoOpts(
                watchdog_ms=SERVO_IDLE_WATCHDOG_MS,
                max_vel=HOLD_RETURN_VEL_RAD_S,
                max_step_rad=min(
                    SERVO_MAX_STEP_RAD,
                    HOLD_RETURN_VEL_RAD_S / SERVO_RATE_HZ * 1.25,
                ),
                max_lag_rad=SERVO_MAX_LAG_RAD,
                rate_hz=SERVO_RATE_HZ,
                lookahead_time=SERVO_LOOKAHEAD_S,
                use_mit=SERVO_USE_MIT,
                is_radians=True,
            )
        )
        if self._hold_q is None:
            self._hold_q = list(self.joint_values())
        self._set_servo_caps(HOLD_RETURN_VEL_RAD_S)
        self._start_joint_hold()
        self.log("[real] servo 会话保持开启（段与段之间续帧顶住，空闲关掉看门狗）")

    def _lock_position_mode(self, *, also_gripper: bool = True) -> int:
        """用单电机位置帧把关节锁进 0x0A，避免下一次 servo_start 去 motor_reset。

        0x8090 会话结束后关节经常残留 MIT ``0x0B``。SDK 的 ``is_enabled``
        只要有一个电机不是 0x0A 就为假，下一趟 ``servo_start`` 会
        ``enable() → motor_reset``——PWM 被切断，臂当场往下沉，夹着的件也松。
        ``set_motor_mode(0x0A)`` 不能用：它先无条件 ``stop()``。
        单电机 ``set_pos_vel_tqe`` 的帧头自带 mode=0x0A，能切过去还带力矩。
        """
        if self._arm is None:
            return 0
        drv = getattr(self._arm, "driver", None) or getattr(self._arm, "_ht", None)
        if drv is None or not hasattr(drv, "set_pos_vel_tqe"):
            return 0
        held = 0
        ids = list(self.cfg.joint_ids)
        if also_gripper:
            ids.append(self.cfg.gripper_id)
        for mid in ids:
            try:
                st = None
                if hasattr(drv, "read_motor_state"):
                    st = drv.read_motor_state(mid, 0.15)
                if st is None and hasattr(drv, "get_cached_state"):
                    st = drv.get_cached_state(mid)
                if st is None:
                    continue
                tqe = (
                    self.grasp_effort()
                    if mid == self.cfg.gripper_id
                    else HOLD_TORQUE_RAW
                )
                ok, mode = self._send_hold_frame(drv, mid, float(st.position), tqe)
                if ok:
                    held += 1
                    if mode == _MODE_MIT:
                        self.log(
                            f"[real] 电机 {mid} 保持帧已发，回包仍是 0x0B（按滞后处理，不撤）"
                        )
            except Exception as e:
                self.log(f"[warn] 电机 {mid} 锁位置模式失败：{e}")
        return held

    @contextlib.contextmanager
    def _servo_session(self):
        """路径喂帧期间占用 servo。会话在第一次运动时打开，断开前才 servo_end。

        以前每段路径都 start/end：结束后电机停在 0x0B，下一段 ``servo_start``
        看见 ``is_enabled=False`` 就 ``motor_reset``，PWM 被切断，臂和夹爪
        都会松一下。现在会话一直开着，空隙由续帧线程顶住。
        """
        self._ensure_servo_started()
        # 走路也不开 150ms 看门狗：算过渡帧 / 读关节时断流就会刹车坠落。
        self._set_joint_watchdog(SERVO_IDLE_WATCHDOG_MS)
        self._servo_drive_depth += 1
        try:
            yield
        finally:
            self._servo_drive_depth -= 1

    def _servo_speed_rad_s(self, speed: int | None) -> float:
        spd = max(1, min(int(speed or self.speed), MAX_MOVE_SPEED))
        return max(1e-3, (spd / 100.0) * SERVO_VEL_AT_FULL_SPEED)

    def commanded_q(self) -> list[float]:
        """最近一次真正发给电机的关节角，不是坠落中的实测角。"""
        if self._hold_send is not None:
            return list(self._hold_send)
        if self._hold_q is not None:
            return list(self._hold_q)
        return list(self.joint_values())

    def _set_servo_caps(self, vel_rad_s: float) -> None:
        """把 servo 的速度/单帧上限压到这次动作的速度，防止断流后按 57°/s 猛追。"""
        if self._arm is None:
            return
        opts = getattr(self._arm, "_servo_opts", None)
        if opts is None:
            return
        vel = min(SERVO_MAX_VEL_RAD_S, max(1e-3, float(vel_rad_s)))
        opts.max_vel = vel
        # 允许 3 帧的节奏误差，避免 93Hz 时把正常步长裁成锯齿；仍远小于猛追。
        opts.max_step_rad = min(
            SERVO_MAX_STEP_RAD,
            max(vel / SERVO_RATE_HZ * 3.0, math.radians(0.35)),
        )

    def _servo_frames(
        self, waypoints, vel_rad_s: float, *, coast: bool = False
    ) -> list[list[float]]:
        """从上次命令角起，按梯形速度曲线插成 100Hz 帧。

        匀速直线启停是方波，机座会抖。这里用 ``SERVO_ACC_RAD_S2`` 加减速。
        ``coast=True`` 时只加速+巡航，减速交给实测到位，避免命令先停臂还在走。
        """
        dt = 1.0 / SERVO_RATE_HZ
        vel = max(1e-4, float(vel_rad_s))
        acc = max(1e-3, float(SERVO_ACC_RAD_S2))
        pts = [self.commanded_q()]
        for wp in waypoints:
            if max(abs(a - b) for a, b in zip(wp, pts[-1])) > 1e-9:
                pts.append(list(wp))
        if len(pts) == 1:
            return [list(pts[0])]

        seglen = [
            max(abs(a - b) for a, b in zip(p0, p1)) for p0, p1 in zip(pts, pts[1:])
        ]
        total = sum(seglen)
        if total < 1e-9:
            return [list(pts[-1])]

        # 半余弦加减速：起停加速度连续，比梯形尖角少晃机座。
        # 峰值加速度是 (π/2)*acc，所以把加速时间拉到 π/2 倍，峰值仍不超过 acc。
        jerk = max(1e-3, float(SERVO_JERK_RAD_S3))
        t_ramp = 0.5 * math.pi * vel / acc
        t_ramp = max(t_ramp, acc / jerk)
        s_ramp = 0.5 * vel * t_ramp
        if 2.0 * s_ramp >= total:
            t_ramp = math.sqrt(0.5 * math.pi * total / acc)
            t_flat = 0.0
            v_peak = total / max(t_ramp, 1e-9)
            s_ramp = 0.5 * total
            coast = False
        else:
            t_flat = (total - 2.0 * s_ramp) / vel
            v_peak = vel
        duration = t_ramp + t_flat + (0.0 if coast else t_ramp)

        def _cos_s(tau: float, t_r: float, v: float) -> float:
            """半余弦加速段路程：v(t)=v/2*(1-cos(πt/tr))。"""
            if t_r < 1e-9:
                return 0.0
            u = min(max(tau / t_r, 0.0), 1.0)
            return 0.5 * v * t_r * (u - math.sin(math.pi * u) / math.pi)

        def s_of_t(t: float) -> float:
            if t <= t_ramp:
                return _cos_s(t, t_ramp, v_peak)
            if coast or t <= t_ramp + t_flat:
                return s_ramp + v_peak * (t - t_ramp)
            tau = min(t - t_ramp - t_flat, t_ramp)
            return s_ramp + v_peak * t_flat + (
                s_ramp - _cos_s(t_ramp - tau, t_ramp, v_peak)
            )

        cum = [0.0]
        for length in seglen:
            cum.append(cum[-1] + length)

        def q_at_s(s: float) -> list[float]:
            s = min(max(s, 0.0), total)
            for i, length in enumerate(seglen):
                if s <= cum[i + 1] + 1e-12:
                    if length < 1e-12:
                        return list(pts[i + 1])
                    u = (s - cum[i]) / length
                    return [a + (b - a) * u for a, b in zip(pts[i], pts[i + 1])]
            return list(pts[-1])

        n = max(1, int(math.ceil(duration / dt)))
        frames = [q_at_s(s_of_t(min(k * dt, duration))) for k in range(1, n + 1)]
        frames[-1] = list(pts[-1])
        return frames

    def _feed_servo_frames(self, frames, dt: float) -> int:
        sent = 0
        t_next = time.perf_counter()
        for frame in frames:
            if self._servo_j(frame):
                sent += 1
                self._hold_send = list(frame)
            t_next += dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()
        return sent

    def _close_with_vel_ff(self, target, cruise: float, dt: float) -> int:
        """命令停在目标，用速度前馈把滞后关节按原速带到位，再按实测减速。"""
        acc = max(1e-3, float(SERVO_ACC_RAD_S2))
        speed = max(1e-4, float(cruise))
        sent = 0
        t_next = time.perf_counter()
        deadline = t_next + 6.0
        while time.perf_counter() < deadline:
            meas = self.joint_values()
            diffs = [t - m for t, m in zip(target, meas)]
            rem = max(abs(d) for d in diffs)
            if rem <= SERVO_SETTLE_TOL_RAD:
                break
            s_stop = 0.5 * speed * speed / acc
            v_des = math.sqrt(max(0.0, 2.0 * acc * rem)) if rem <= s_stop else cruise
            if speed < v_des:
                speed = min(v_des, speed + acc * dt)
            else:
                speed = max(v_des, speed - acc * dt)
            vels = [
                0.0 if abs(d) < 1e-9 else math.copysign(speed, d) for d in diffs
            ]
            if self._servo_j(target, vel_ff=vels):
                sent += 1
                self._hold_send = list(target)
            t_next += dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()
        self._servo_j(target)
        return sent

    def _hold_dwell(self, target, dt: float, seconds: float = 0.15) -> int:
        """原地续同一命令角，不追实测。让余振散掉，避免到位环硬顶机座。"""
        n = max(1, int(round(max(0.0, float(seconds)) * SERVO_RATE_HZ)))
        sent = 0
        for _ in range(n):
            if self._servo_j(target):
                sent += 1
            time.sleep(dt)
        return sent

    def _brief_arrive(self, target, vel: float, dt: float) -> int:
        """抓取点只短顶一下，避免 3 秒到位环把臂刹死再抬。"""
        err = max(abs(a - b) for a, b in zip(target, self.joint_values()))
        if err <= math.radians(2.5):
            return self._hold_dwell(target, dt, 0.12)
        # 最多 0.35s，不够到位也往下走，别在抓取点磨
        speed = max(1e-4, float(vel))
        sent = 0
        t_next = time.perf_counter()
        deadline = t_next + 0.35
        while time.perf_counter() < deadline:
            meas = self.joint_values()
            diffs = [t - m for t, m in zip(target, meas)]
            rem = max(abs(d) for d in diffs)
            if rem <= math.radians(2.0):
                break
            vels = [
                0.0 if abs(d) < 1e-9 else math.copysign(speed, d) for d in diffs
            ]
            if self._servo_j(target, vel_ff=vels):
                sent += 1
            t_next += dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()
        self._servo_j(target)
        return sent + 1

    def _servo_run(
        self,
        frames,
        *,
        phase: str,
        vel: float | None = None,
        must_arrive: bool = False,
        settle: bool = True,
    ) -> list[float]:
        """按固定节奏把帧喂下去，然后确认到位。

        节奏用绝对时刻推进（而不是每轮 ``sleep(dt)``）：sleep 本身有误差，
        累加起来帧间距会越来越飘，动作反而抖。
        """
        dt = 1.0 / SERVO_RATE_HZ
        target = list(frames[-1])
        sent = 0
        cap = float(vel) if vel is not None else self._servo_speed_rad_s(None)
        self._set_servo_caps(cap)
        with self._servo_session():
            sent += self._feed_servo_frames(frames, dt)

            if must_arrive:
                meas = self.joint_values()
                err = max(abs(a - b) for a, b in zip(target, meas))
                if err > SERVO_SETTLE_TOL_RAD:
                    sent += self._close_with_vel_ff(target, cap, dt)
                err = max(abs(a - b) for a, b in zip(target, self.joint_values()))
                if err > SERVO_SETTLE_TOL_RAD:
                    self.log(
                        f"[warn] {phase} 到位残差 {math.degrees(err):.2f}°"
                    )
            elif settle:
                # 帧喂完不等于已到位。残差还在缩小就继续顶。
                # 已经很近（≤2°）且不再缩小，才按重力静差停。
                deadline = time.perf_counter() + SERVO_SETTLE_TIMEOUT
                prev_err = None
                stall = 0
                err = 0.0
                while time.perf_counter() < deadline:
                    err = max(abs(a - b) for a, b in zip(target, self.joint_values()))
                    if err <= SERVO_SETTLE_TOL_RAD:
                        break
                    if prev_err is not None and err > prev_err - math.radians(0.05):
                        stall += 1
                        if stall >= 50 and err <= math.radians(2.0):
                            self.log(
                                f"[warn] {phase} 残差停在 {math.degrees(err):.2f}°，"
                                "按重力静差处理，不再硬顶"
                            )
                            break
                    else:
                        stall = 0
                    prev_err = err
                    self._servo_j(target)
                    time.sleep(dt)
                else:
                    self.log(
                        f"[warn] {phase} 到位残差 {math.degrees(err):.2f}°"
                        f"（超过 {math.degrees(SERVO_SETTLE_TOL_RAD):.1f}° 判定带）"
                    )

            self._hold_q = list(target)
            self._hold_send = list(target)
            self._hold_crawl_noted = False
        self._set_servo_caps(HOLD_RETURN_VEL_RAD_S)
        lag = int(getattr(self._arm, "servo_lag_count", 0) or 0)
        clamp = int(getattr(self._arm, "servo_clamp_count", 0) or 0)
        note = f"，跟随告警 {lag} 次" if lag else ""
        note += f"，单帧裁剪 {clamp} 次" if clamp else ""
        self.log(f"[real] {phase} servo 走完 {sent} 帧{note}")
        return target

    def _servo_goto(
        self,
        target,
        *,
        label: str,
        speed: int | None = None,
        must_arrive: bool = False,
    ):
        """servo_j 版的点到点：一次插到底，不再拆成 25° 一步。"""
        vel = self._servo_speed_rad_s(speed)
        frames = self._servo_frames([target], vel, coast=must_arrive)
        self.log(
            f"[real] 移动到{label}（servo {len(frames)} 帧，"
            f"{math.degrees(vel):.0f}°/s）"
        )
        if self._arm is None:
            arrived = list(frames[-1])
            self._sim_q = arrived
            self._hold_q = arrived
            self._hold_send = arrived
            self.log(f"[real] {label} servo 走完 {len(frames)} 帧（dry-run）")
            return arrived
        return self._servo_run(
            frames, phase=label, vel=vel, must_arrive=must_arrive
        )

    def move_j(self, q, *, speed: int | None = None, note: str = "", allow_large: bool = False):
        """单点关节运动，阻塞。超软限位会被裁剪，跨度过大会拒绝执行。"""
        target, hit = self.cfg.clamp(q)
        if hit:
            self.log(f"[warn] move_j 裁剪了 J{[j + 1 for j in hit]}")
        gap = max(abs(a - b) for a, b in zip(target, self.commanded_q()))
        if gap > MAX_STEP_RAD and not allow_large:
            raise RealArmError(
                f"目标位形距当前 {math.degrees(gap):.1f}°，超过单步上限 "
                f"{math.degrees(MAX_STEP_RAD):.1f}°；确认安全后用 allow_large=True"
            )
        spd = max(1, min(int(speed or self.speed), MAX_MOVE_SPEED))
        self.log(f"  move_j {_fmt(target)} speed={spd} {note}")
        if self._arm is None:
            self._sim_q = target
            return target
        self._arm.move_j(
            target,
            is_radians=True,
            speed=spd,
            block=True,
            tolerance=MOVE_J_TOLERANCE_RAD,
            timeout=MOVE_J_TIMEOUT,
            style=self._move_style,
        )
        return target

    def open_jaw(self, *, note: str = "") -> float:
        """把夹爪张到软限位上限。

        到 HOME / 避让位时用：合着的夹爪在相机里会被 YOLO 认成 pliers，
        而且下一次抓取本来也要先张开。手上夹着东西时调它就会松手。
        """
        return self.gripper_move(self.cfg.gripper_limits[1], note=note or "(张开)")

    def go_conf(
        self,
        target,
        *,
        label: str = "指定位形",
        speed: int | None = None,
        open_jaw: bool = False,
        must_arrive: bool = False,
    ):
        """走到某个固定位形。

        当前位形可能离目标很远（比如 EC 撤离后停在格子上方），所以拆成小步走，
        而不是一次 move_j 甩过去。

        ``open_jaw`` 在**到位之后**张开夹爪。放在到位后是因为路上还夹着东西时
        提前松手，件会掉在半路。
        """
        from traj_adapter import path_between  # noqa: PLC0415  避免模块级循环依赖

        target, hit = self.cfg.clamp(list(target)[: self.cfg.num_joints])
        if hit:
            self.log(f"[warn] {label} 裁剪了 J{[j + 1 for j in hit]}")
        if self.use_servo and (self.servo_ready or self._arm is None):
            # servo 自己就是密集小步，不需要再按 MAX_STEP_RAD 拆成几段 move_j。
            # dry-run（_arm is None）也走这条，日志才和真机一致，不会误打成 move_j。
            self._servo_goto(
                target, label=label, speed=speed, must_arrive=must_arrive
            )
        else:
            wps = path_between(self.commanded_q(), target)
            self.log(f"[real] 移动到{label}（{len(wps)} 步）")
            for i, wp in enumerate(wps, 1):
                self.move_j(
                    wp, speed=speed, note=f"({label} {i}/{len(wps)})", allow_large=True
                )
        if open_jaw and JAW_OPEN_ON_ARRIVE:
            self.open_jaw(note=f"({label} 到位后张开)")
        return target

    def go_sim_home(self, *, speed: int | None = None, open_jaw: bool = True):
        """回到仿真轨迹的起始位形（不是 SDK 的全零 home）。"""
        return self.go_conf(
            SIM_HOME_CONF, label="仿真 HOME", speed=speed, open_jaw=open_jaw
        )

    def go_exit_home(self, *, speed: int | None = None, open_jaw: bool = True):
        """回到退出前的静置位。

        和 ``go_sim_home`` 分开：那个是仿真轨迹的 HOME 锚点；这个只是断电前
        站得住的收臂位形。视觉抓取默认从避让位接上，不再强制回 HOME。
        """
        return self.go_conf(
            EXIT_HOME_CONF, label="静置位", speed=speed, open_jaw=open_jaw
        )

    def go_park(self, *, speed: int | None = None, open_jaw: bool = True):
        """退到相机拍不到的避让位。识别前必须先到这里，否则臂会被当成零件。"""
        arrived = self.go_conf(
            PARK_CONF, label="避让位", speed=speed, open_jaw=open_jaw
        )
        self.hold_here("避让位")
        return arrived

    def go_place(self, *, speed: int | None = None, open_jaw: bool = False):
        """送到现场示教的放置松爪位。默认不合爪、也不提前张爪。"""
        arrived = self.go_conf(
            PLACE_CONF, label="放置松爪位", speed=speed, open_jaw=open_jaw
        )
        self.hold_here("放置松爪位")
        return arrived

    def go_zero(self, *, speed: int | None = None, open_jaw: bool = True):
        """送到电机零位（全 0°）。不是仿真 HOME，也不是退出静置位。"""
        arrived = self.go_conf(
            ZERO_CONF,
            label="零位",
            speed=speed,
            open_jaw=open_jaw,
            must_arrive=True,
        )
        meas = self.joint_values()
        err = max(abs(a - b) for a, b in zip(arrived, meas))
        self.log(f"[real] 零位实测 {_fmt(meas)} 残差 {math.degrees(err):.2f}°")
        return arrived

    def follow_path(self, waypoints, *, speed: int | None = None, phase: str = ""):
        """走完一条关节路径。

        优先 ``servo_j``：自己按固定频率喂密集帧，动作最连贯。
        其次 ``stream_path``：整段交给 SDK 插值下发。
        兜底逐点阻塞 ``move_j``：慢、一卡一卡，但每步都确认到位。
        """
        if not waypoints:
            return
        if self.use_servo and (self.servo_ready or self._arm is None):
            return self._servo_follow(waypoints, speed=speed, phase=phase or "路径")
        if self.stream_path and self._arm is not None and self._can_stream:
            return self._stream_path(waypoints, speed=speed, phase=phase)
        if self.stream_path and self._arm is None:
            # dry-run 也要能看出将来走的是哪种模式
            return self._stream_path_dry(waypoints, speed=speed, phase=phase)
        self.log(f"[real] 走路径 {phase} 共 {len(waypoints)} 点（逐点阻塞）")
        for i, wp in enumerate(waypoints, 1):
            self.move_j(wp, speed=speed, note=f"({phase} {i}/{len(waypoints)})", allow_large=True)

    def _servo_follow(self, waypoints, *, speed: int | None = None, phase: str = ""):
        """servo_j 版的路径跟随：整条路径插成密集帧，一路喂下去不停顿。"""
        path = []
        clamped: set[int] = set()
        for wp in waypoints:
            target, hit = self.cfg.clamp(wp)
            clamped.update(hit)
            path.append(target)
        if clamped:
            self.log(f"[warn] 路径裁剪了 J{[j + 1 for j in sorted(clamped)]}")

        # 从上次命令角接到首点，不是从坠落中的实测角。首点远只是多一段过渡。
        head_gap = max(abs(a - b) for a, b in zip(path[0], self.commanded_q()))
        if head_gap > MAX_STEP_RAD:
            self.log(
                f"[real] {phase} 从命令角接入，距首点 {math.degrees(head_gap):.1f}°，"
                "已插成过渡帧"
            )
        vel = self._servo_speed_rad_s(speed)
        frames = self._servo_frames(path, vel)
        self.log(
            f"[real] 走路径 {phase} 共 {len(path)} 点 → servo {len(frames)} 帧"
            f"（{math.degrees(vel):.0f}°/s）"
        )
        if self._arm is None:
            arrived = list(frames[-1])
            self._sim_q = arrived
            self._hold_q = arrived
            self._hold_send = arrived
            self.log(f"[real] {phase} servo 走完 {len(frames)} 帧（dry-run）")
            return arrived
        return self._servo_run(frames, phase=phase, vel=vel)

    def stream_pick(
        self,
        approach,
        grasp_angle,
        after,
        *,
        speed: int | None = None,
        label: str = "抓取",
    ):
        """一次会话喂完整段：接近 → 合爪 → 抬起/回避让位。

        关节角先插成 100Hz 帧再连续下发。合爪前后不再重新 servo_start，
        也不走 3 秒到位环，避免 pick 时刹停再猛抬。
        """
        from traj_adapter import _max_delta  # noqa: PLC0415

        vel = self._servo_speed_rad_s(speed)
        dt = 1.0 / SERVO_RATE_HZ

        def _clamp_path(wps):
            out = []
            for wp in wps or []:
                q, _ = self.cfg.clamp(wp)
                if not out or _max_delta(out[-1], q) > 1e-9:
                    out.append(list(q))
            return out

        approach = _clamp_path(approach)
        after = _clamp_path(after)
        if after and approach and _max_delta(after[0], approach[-1]) < 1e-6:
            after = after[1:]

        n_app = n_aft = 0
        with self._servo_session():
            if approach:
                frames = self._servo_frames(approach, vel)
                n_app = len(frames)
                self.log(
                    f"[real] {label} 打包接近 {len(approach)} 点 → {n_app} 帧"
                    f"（{math.degrees(vel):.0f}°/s）"
                )
                self._set_servo_caps(vel)
                self._feed_servo_frames(frames, dt)
                self._hold_dwell(frames[-1], dt, 0.15)
                self._hold_q = list(frames[-1])
                self._hold_send = list(frames[-1])
            if grasp_angle is not None:
                self.gripper_grasp(grasp_angle)
                self._hold_dwell(self.commanded_q(), dt, 0.12)
            if after:
                frames = self._servo_frames(after, vel)
                n_aft = len(frames)
                self.log(
                    f"[real] {label} 打包抬起/收尾 {len(after)} 点 → {n_aft} 帧"
                    f"（{math.degrees(vel):.0f}°/s）"
                )
                self._set_servo_caps(vel)
                self._feed_servo_frames(frames, dt)
                self._hold_dwell(frames[-1], dt, 0.15)
                self._hold_q = list(frames[-1])
                self._hold_send = list(frames[-1])
        self._set_servo_caps(HOLD_RETURN_VEL_RAD_S)
        self.log(f"[real] {label} 连续下发完毕（接近 {n_app} + 收尾 {n_aft} 帧）")

    def _stream_path_dry(self, waypoints, *, speed: int | None = None, phase: str = ""):
        spd = max(1, min(int(speed or self.speed), MAX_MOVE_SPEED))
        target, _hit = self.cfg.clamp(waypoints[-1])
        self.log(
            f"[real] 走路径 {phase} 共 {len(waypoints)} 点（连续下发 speed={spd}）"
            f" → 终点 {_fmt(target)}"
        )
        self._sim_q = target
        return target

    def _stream_path(self, waypoints, *, speed: int | None = None, phase: str = ""):
        """整段路径一次性交给 SDK：先插值成密集帧，再逐帧刷位置通道。

        SDK 内部按 ``control_frequency`` 插值，所以这里传的是**原始航点**，
        不需要自己densify；跨度检查仍然做，避免把一个瞬移当成路径发下去。
        """
        path = []
        clamped_joints: set[int] = set()
        for wp in waypoints:
            target, hit = self.cfg.clamp(wp)
            clamped_joints.update(hit)
            path.append(target)
        if clamped_joints:
            self.log(f"[warn] 路径裁剪了 J{[j + 1 for j in sorted(clamped_joints)]}")

        # 从机械臂当前位形接上，否则第一帧就是一个跳变。
        # 首点远时 SDK 自己会按 control_frequency 插中间帧，不是瞬移。
        current = self.commanded_q()
        head_gap = max(abs(a - b) for a, b in zip(path[0], current))
        if head_gap > MAX_STEP_RAD:
            self.log(
                f"[real] {phase} 从当前接入，距首点 {math.degrees(head_gap):.1f}°"
            )
        spd = max(1, min(int(speed or self.speed), MAX_MOVE_SPEED))
        self.log(
            f"[real] 走路径 {phase} 共 {len(path)} 点（连续下发 speed={spd}）"
            f" → 终点 {_fmt(path[-1])}"
        )
        self._arm.move_jntspace_path(
            [current] + path,
            is_radians=True,
            speed=spd,
            control_frequency=self.control_freq,
            start_frame_id=1,  # 跳过等于当前位形的那一帧
        )
        # 流式帧是 block=False 发的，走完不保证已到位。合爪前必须确认到终点，
        # 否则会在半路上夹。
        self._arm.move_j(
            path[-1],
            is_radians=True,
            speed=spd,
            block=True,
            timeout=STREAM_SETTLE_TIMEOUT,
            style=self._move_style,
        )
        return path[-1]

    # -- 夹爪 ---------------------------------------------------------------
    def gripper_move(self, angle_rad: float, *, note: str = ""):
        lo, hi = self.cfg.gripper_limits
        angle = min(max(float(angle_rad), lo), hi)
        self.log(f"  夹爪 → {math.degrees(angle):.1f}° {note}")
        if self._arm is None:
            return angle
        self._pause_hold_for_gripper()
        try:
            self._arm.gripper_control(
                angle,
                is_radians=True,
                vel=GRIPPER_VEL,
                acc=GRIPPER_ACC,
                block=True,
                on_tick=self._hold_joints_tick,
            )
        finally:
            self._hold_paused = False
        return angle

    def grasp_effort(self, effort: int | None = None) -> int:
        """解析固件侧力矩上限：显式入参 > 配置 > robot.cfg。"""
        if effort is not None:
            return int(effort)
        if GRASP_EFFORT_RAW is not None:
            return int(GRASP_EFFORT_RAW)
        return int(self.cfg.gripper_max_torque_raw)

    def gripper_grasp(
        self,
        angle_rad: float,
        *,
        force: int = GRASP_FORCE_THRESHOLD,
        effort: int | None = None,
        overdrive: float = GRASP_OVERDRIVE_RAD,
    ):
        """力控合爪：一直往闭合方向走，碰到零件（力矩/堵转）就停。

        ``angle_rad`` 只是规划给的标称开口，用来打日志对照，**不是停爪目标**。
        以前按标称宽度只再过合几度就停，STL 比实际抓点粗时夹爪还张着一大截
        就已经 ``reached_target``，现场看起来像「没合到底」，手里也是空的。

        现在目标发给夹爪软限位的闭合端；SDK 在 ``|力矩| >= force`` 或合上一段
        后堵转时提前返回。接触后再把目标改到「接触角再往里过合一点」，让位置
        环顶住，而不是继续往限位死合。
        """
        lo, hi = self.cfg.gripper_limits
        nominal = min(max(float(angle_rad), lo), hi)
        cap = self.grasp_effort(effort)
        trip = int(force)
        if trip >= cap:
            # 实测力矩被 cap 限住，够不到 trip ⇒ 力矩判定失效，只能等堵转/超时。
            # 这种退化是静默的（照样报 grasped=True），所以主动压回去并告警。
            trip = max(1, int(cap * 0.75))
            self.log(f"  [warn] 力阈值 {force} ≥ 力矩上限 {cap}，已压到 {trip}")

        self.log(
            f"  夹爪力控合爪：碰到零件就停，合不到就走到限位 "
            f"{math.degrees(lo):.1f}°（标称开口对应 {math.degrees(nominal):.1f}°）"
            f" force={trip} effort={cap}"
        )
        if self._arm is None:
            return None
        self._pause_hold_for_gripper()
        try:
            result = self._arm.grasp(
                target_angle=lo,
                is_radians=True,
                force_threshold=trip,
                effort=cap,
                vel=GRIPPER_VEL,
                timeout=8.0,
                on_tick=self._hold_joints_tick,
            )
            # grasp() 返回后电机还在追原来的限位目标。碰到东西必须立刻改成
            # 接触位，否则会继续往里合到限位。
            if result.grasped and math.isfinite(result.angle_rad):
                hold = min(max(float(result.angle_rad) - abs(overdrive), lo), hi)
                self._arm.gripper_control(
                    hold,
                    effort=cap,
                    is_radians=True,
                    vel=GRIPPER_VEL,
                    acc=GRIPPER_ACC,
                    block=True,
                    timeout=2.0,
                    on_tick=self._hold_joints_tick,
                )
        finally:
            self._hold_paused = False
        self.log(
            f"  抓取结果 grasped={result.grasped} reason={result.reason} "
            f"峰值力矩={result.peak_torque_raw} 实际合了 {result.closed_deg:.1f}°"
            f" 停在 {math.degrees(result.angle_rad):.1f}°"
        )
        if result.reason == "reached_target":
            self.log(
                "  [warn] 已经合到夹爪限位还没顶到东西（力矩没起来），"
                "可能没抓到或力阈值偏高"
            )
        elif not result.grasped:
            self.log(f"  [warn] 未判定为抓取成功（{result.reason}）")
        return result

    def estop(self) -> None:
        self.log("[real] 急停")
        if self._arm is not None:
            self._arm.emergency_stop()


# ---------------------------------------------------------------------------
# plan 执行
# ---------------------------------------------------------------------------
def release_at_place(
    arm: RealArm,
    *,
    delay_s: float = 0.0,
    park_after: bool = False,
) -> None:
    """送到示教放置位（料框）再松爪。

    ``delay_s`` 只给 ``run_vision_real`` 接下一项任务时用：先夹着停一会儿
    再张爪。智能体抓取指令不要走这里。
    """
    arm.go_place(open_jaw=False)
    delay = max(0.0, float(delay_s))
    if delay > 0:
        arm.log(f"[real] 放置位夹持等待 {delay:.1f}s 再松爪")
        time.sleep(delay)
    arm.open_jaw(note="(放置位松爪)")
    if park_after:
        arm.go_park(open_jaw=True)


def execute_plan(
    plan: dict,
    arm: RealArm,
    *,
    go_home_first: bool = True,
    go_home_after: bool = True,
    park_after: bool = False,
    place_after: bool = False,
    release_after: bool = False,
    pause_between: float = 0.5,
    stop_on_fail: bool = False,
) -> dict:
    """执行 ``traj_adapter.adapt`` 产出的 plan。

    单个目标失败不会中断整批：记下来跳到下一个（``stop_on_fail=True`` 可改成
    一失败就停）。返回每个目标的成败汇总。

    ``park_after`` 优先于 ``go_home_after``：视觉链路每轮都要重新识别，停在
    避让位可以直接接下一条指令，回 HOME 反而会挡住相机。

    ``place_after`` 时**每一件**抓完都去示教料框松爪，再直接抓下一件，
    中间不回避让位；全部放完才 ``park_after``。纯抓取只回避让位、夹爪保持。
    ``release_after`` 在放置位张爪；抓取不要开。

    ``go_home_first=False`` 时从**当前**位形接入（避让位已经到了就别再绕回
    HOME）：approach 丢掉 HOME 前缀。纯抓取且 ``park_after`` 时，carry 若以
    HOME 结尾则抬起后停，让最后的回避让位接手。
    """
    from traj_adapter import (  # noqa: PLC0415
        _max_delta,
        cut_after_lift,
        drop_home_prefix,
        path_between,
    )

    if plan.get("errors"):
        raise RealArmError(
            "plan 存在致命问题，拒绝下发：\n  " + "\n  ".join(plan["errors"])
        )
    for w in plan.get("warnings", []):
        arm.log(f"[warn] {w}")

    results = []
    if go_home_first:
        arm.go_sim_home()

    home_q = list(SIM_HOME_CONF[: arm.cfg.num_joints])
    first_path = True
    segs = list(plan["segments"])

    def _is_pre_open(step) -> bool:
        return (
            step.get("kind") == "gripper"
            and step.get("action") == "move"
            and "全开" in str(step.get("note") or "")
        )

    def prepare_wps(step, wps, last_path):
        nonlocal first_path
        phase = step.get("phase", "")
        if (not go_home_first) and first_path and wps:
            wps, skipped = drop_home_prefix(wps, home_q)
            if skipped:
                arm.log(
                    f"[real] 从当前位置接入 {phase}，"
                    f"去掉开头 {skipped} 个 HOME 锚点，保留后续 {len(wps)} 个航点"
                )
            first_path = False
        # 仿真 carry 常回到 HOME。真机抬起后要去料框或避让位，丢掉回 HOME 的尾巴。
        if (
            (park_after or place_after)
            and step is last_path
            and wps
            and _max_delta(wps[-1], home_q) < math.radians(8.0)
        ):
            kept = cut_after_lift(wps, CARRY_LIFT_MIN_RAD)
            if len(kept) < len(wps):
                dest = "示教料框" if place_after else "避让位"
                arm.log(
                    f"[real] {phase} 抬起后去{dest}，"
                    f"丢掉回 HOME 的 {len(wps) - len(kept)} 个航点"
                )
                wps = kept
        return wps

    def extend_unique(dst, src):
        for q in src:
            if not dst or _max_delta(dst[-1], q) > 1e-9:
                dst.append(list(q))

    def go_taught_place(label: str) -> None:
        arm.go_place(open_jaw=False)
        if release_after:
            arm.open_jaw(note=f"({label} 放置位松爪)")

    for si, seg in enumerate(segs):
        label = f"[{seg['index']}] {seg['object']}"
        arm.log(f"[real] === 开始 {label} ===")
        path_steps = [s for s in seg["steps"] if s["kind"] == "path" and s.get("waypoints")]
        last_path = path_steps[-1] if path_steps else None
        gripper_steps = [s for s in seg["steps"] if s["kind"] != "path"]
        grasps = [s for s in gripper_steps if s.get("action") == "grasp"]
        extras = [
            s
            for s in gripper_steps
            if s.get("action") != "grasp" and not _is_pre_open(s)
        ]
        packable = arm.servo_ready and len(grasps) == 1 and not extras
        try:
            if packable:
                approach: list[list[float]] = []
                after: list[list[float]] = []
                seen_grasp = False
                for step in seg["steps"]:
                    if step["kind"] == "path":
                        wps = prepare_wps(
                            step, list(step.get("waypoints") or []), last_path
                        )
                        if not seen_grasp:
                            extend_unique(approach, wps)
                        else:
                            extend_unique(after, wps)
                    elif step.get("action") == "grasp":
                        seen_grasp = True
                if place_after:
                    last = after[-1] if after else (approach[-1] if approach else None)
                    if last is not None and _max_delta(last, PLACE_CONF) > math.radians(1.0):
                        extra = path_between(last, PLACE_CONF)
                        arm.log(
                            f"[real] {label} 抬起后去示教料框，"
                            f"{len(extra)} 个过渡点"
                        )
                        extend_unique(after, extra)
                arm.stream_pick(
                    approach, grasps[0]["angle_rad"], after, label=label
                )
                if place_after:
                    if release_after:
                        arm.open_jaw(note=f"({label} 放置位松爪)")
            else:
                for step in seg["steps"]:
                    if _is_pre_open(step):
                        continue
                    if step["kind"] == "path":
                        wps = prepare_wps(
                            step, list(step.get("waypoints") or []), last_path
                        )
                        arm.follow_path(wps, phase=step.get("phase", ""))
                    elif step["action"] == "grasp":
                        arm.gripper_grasp(step["angle_rad"])
                    else:
                        arm.gripper_move(step["angle_rad"], note=step.get("note", ""))
                if place_after:
                    go_taught_place(label)
            results.append({"index": seg["index"], "object": seg["object"], "ok": True})
            arm.log(f"[real] === 完成 {label} ===")
        except Exception as e:
            results.append(
                {"index": seg["index"], "object": seg["object"], "ok": False, "error": str(e)}
            )
            arm.log(f"[real] {label} 执行失败：{e}")
            if stop_on_fail:
                break
            arm.log("[real] 跳过该目标，继续下一个")
        if pause_between > 0:
            time.sleep(pause_between)

    ok = sum(1 for r in results if r["ok"])
    if park_after:
        try:
            # 放置任务每件已经松过爪；纯抓取夹着最后一件回避让位
            arm.go_park(open_jaw=bool(place_after) or ok == 0)
        except Exception as e:
            arm.log(f"[real] 去避让位失败：{e}")
    elif go_home_after:
        try:
            arm.go_sim_home(open_jaw=bool(place_after) or ok == 0)
        except Exception as e:
            arm.log(f"[real] 回 HOME 失败：{e}")

    arm.log(f"[real] 全部结束：成功 {ok}/{len(results)}")
    return {"results": results, "ok": ok, "total": len(results)}
