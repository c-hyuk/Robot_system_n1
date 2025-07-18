"""
Robot Controller – 중앙 제어 시스템
모든 하위 모듈을 조합해 완전한 로봇 스택을 구동한다.

깨끗하게 정리된 버전: 중복 정의 · 잘못 삽입된 GR00T 인터페이스 코드 · 테스트 블록 중복 등을 제거했다.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from model.inference_engine import RealTimeInferenceEngine, create_inference_engine
from model.action_decoder import ActionDecoderManager, create_action_decoder
from communication.hardware_bridge import PiperHardwareBridge  # 실제 구현체
from control.safety_manager import SafetyManager               # 실제 구현체
# from communication.terminal_interface import TerminalInterface  # TODO

from config.hardware_config import get_hardware_config
from utils.data_types import SystemConfig  # 타입 힌트용 (예: joint 제한 등)

# ────────────────────────────────────────────────────────────────────────────────
# Enum · Dataclass 정의
# ------------------------------------------------------------------------------

class ControllerState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobotControllerConfig:
    # 주기
    control_frequency: float = 10.0  # Hz
    max_loop_time: float = 0.15      # s

    # 모델
    model_path: str = "nvidia/gr00t-1.5b"
    embodiment_name: str = "dual_piper_arm"
    use_mock_data: bool = False

    # 안전
    enable_safety_checks: bool = True
    emergency_stop_enabled: bool = True
    max_consecutive_errors: int = 3

    # 실행 모드
    execution_mode: str = "position"  # position / velocity / trajectory

    # 로깅
    log_frequency: float = 1.0  # Hz

    # CAN 포트
    left_arm_can_port: str = "can0"
    right_arm_can_port: str = "can1"


@dataclass
class SystemState:
    controller_state: str = "idle"
    inference_state: str = "idle"
    hardware_state: str = "disconnected"

    current_frequency: float = 0.0
    avg_loop_time: float = 0.0
    error_count: int = 0

    last_command_time: float = 0.0
    total_commands_sent: int = 0
    uptime_seconds: float = 0.0

    left_arm_positions: Optional[List[float]] = None
    right_arm_positions: Optional[List[float]] = None
    safety_status: bool = True


# ────────────────────────────────────────────────────────────────────────────────
# RobotController 본체
# ------------------------------------------------------------------------------

class RobotController:
    """메인 로봇 컨트롤러."""

    # ‑‑‑ 초기화 ---------------------------------------------------------------
    def __init__(
        self,
        config: Optional[RobotControllerConfig] = None,
        *,
        left_piper=None,
        right_piper=None,
    ) -> None:
        self.cfg = config or RobotControllerConfig()
        self.left_piper = left_piper
        self.right_piper = right_piper

        self.hw_config: SystemConfig = get_hardware_config()

        # 상태
        self.state = ControllerState.IDLE
        self.sys_state = SystemState()

        # 동시성
        self._stop_event = threading.Event()
        self._control_thread: Optional[threading.Thread] = None
        self._command_queue: queue.Queue = queue.Queue(maxsize=10)
        self._action_lock = threading.Lock()
        self._loop_times: List[float] = []

        # 핵심 서브시스템
        self.inference_engine: Optional[RealTimeInferenceEngine] = None
        self.action_decoder: Optional[ActionDecoderManager] = None
        self.hardware_bridge: Optional[PiperHardwareBridge] = None
        self.safety_manager: Optional[SafetyManager] = None
        self.terminal = None  # TODO: TerminalInterface 구현 시 교체

        # 콜백
        self._status_cbs: List[Callable[[SystemState], None]] = []
        self._error_cbs: List[Callable[[Exception], None]] = []

        # 로깅
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("RobotController initialised")

    # ‑‑‑ 콜백 등록 -------------------------------------------------------------
    def add_status_callback(self, fn: Callable[[SystemState], None]) -> None:
        self._status_cbs.append(fn)

    def add_error_callback(self, fn: Callable[[Exception], None]) -> None:
        self._error_cbs.append(fn)

    # ────────────────────────────────────────────────────────────────────────
    # 시스템 Lifecycle
    # ----------------------------------------------------------------------

    def start(self) -> bool:
        if self.state != ControllerState.IDLE:
            self.log.warning("System already started or not idle")
            return False

        self.state = ControllerState.INITIALIZING
        self.log.info("Starting RobotController …")

        try:
            self._init_modules()
            self._run_control_loop()
            return True
        except Exception as exc:  # noqa: BLE001
            self.log.exception("Failed to start RobotController")
            self._notify_error(exc)
            self.state = ControllerState.ERROR
            return False

    def stop(self) -> None:
        self.log.info("Stopping RobotController …")
        self._stop_event.set()

        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=3.0)

        # 하위 모듈 clean‑up
        if self.inference_engine:
            self.inference_engine.stop()
        if self.hardware_bridge:
            self.hardware_bridge.disconnect()
        if self.safety_manager:
            self.safety_manager.stop_monitoring()

        self.state = ControllerState.IDLE
        self.sys_state.controller_state = "idle"
        self.log.info("RobotController stopped")

    # context‑manager sugar
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        self.stop()

    # ────────────────────────────────────────────────────────────────────────
    # 내부 초기화 루틴
    # ----------------------------------------------------------------------

    def _init_modules(self) -> None:
        """모든 서브시스템 초기화."""
        # 1) Inference Engine
        self.inference_engine = create_inference_engine(
            model_path=self.cfg.model_path,
            target_frequency=self.cfg.control_frequency,
            use_mock_data=self.cfg.use_mock_data,
        )
        self.inference_engine.add_action_callback(self._on_action)
        self.inference_engine.add_error_callback(self._notify_error)
        if not self.inference_engine.start():
            raise RuntimeError("InferenceEngine failed to start")

        # 2) Action Decoder
        self.action_decoder = create_action_decoder(
            embodiment_name=self.cfg.embodiment_name,
            execution_mode=self.cfg.execution_mode,
        )

        # 3) Hardware Bridge
        self.hardware_bridge = PiperHardwareBridge(
            left_can_port=self.cfg.left_arm_can_port,
            right_can_port=self.cfg.right_arm_can_port,
            auto_enable=True,
            gripper_enabled=True,
            left_piper=self.left_piper,
            right_piper=self.right_piper,
        )
        if not self.hardware_bridge.connect():
            raise RuntimeError("Hardware bridge connection failed")

        # 4) Safety Manager
        self.safety_manager = SafetyManager(self.hw_config)
        self.safety_manager.start_monitoring()

        self.sys_state.hardware_state = "connected"
        self.sys_state.inference_state = "running"
        self.log.info("All modules initialised successfully")

    # ────────────────────────────────────────────────────────────────────────
    # 제어 루프
    # ----------------------------------------------------------------------

    def _run_control_loop(self) -> None:
        self._stop_event.clear()
        self.state = ControllerState.RUNNING
        self.sys_state.controller_state = "running"
        self._start_ts = time.time()

        self._control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="RobotControlLoop",
        )
        self._control_thread.start()

    def _control_loop(self):
        tgt_dt = 1.0 / self.cfg.control_frequency
        self.log.info("Control loop @ %.2f Hz", self.cfg.control_frequency)

        while not self._stop_event.is_set():
            tic = time.time()

            try:
                self._update_state()
                self._process_terminal()
                self._monitor_perf(tic)
            except Exception as exc:  # noqa: BLE001
                self._notify_error(exc)
                if self.cfg.emergency_stop_enabled:
                    self._trigger_emergency_stop()
                    break

            # ‑‑ 타이밍 보정 ‑‑
            dt = time.time() - tic
            sleep = max(0.0, tgt_dt - dt)
            if sleep:
                time.sleep(sleep)
            if dt > self.cfg.max_loop_time:
                self.log.warning("Control‑loop over‑time (%.1f ms)", dt * 1000)

    # ────────────────────────────────────────────────────────────────────────
    # 액션 핸들러 & 유틸
    # ----------------------------------------------------------------------

    def _on_action(self, token_dict: Dict[str, np.ndarray]) -> None:
        if not self.action_decoder:
            return
        cmds = self.action_decoder.decode_action(token_dict)
        if self.safety_manager and not self.safety_manager.validate_command(cmds):
            self.log.debug("Unsafe command rejected")
            return
        if self.hardware_bridge:
            self.hardware_bridge.send_commands(cmds)
        with self._action_lock:
            self.sys_state.last_command_time = time.time()
            self.sys_state.total_commands_sent += 1

    # ────────────────────────────────────────────────────────────────────────
    # 상태/모니터링
    # ----------------------------------------------------------------------

    def _update_state(self):
        now = time.time()
        self.sys_state.uptime_seconds = now - getattr(self, "_start_ts", now)

        if self.inference_engine:
            eng = self.inference_engine.get_engine_status()
            self.sys_state.current_frequency = eng["actual_frequency"]
            self.sys_state.inference_state = eng["state"]

        if self.hardware_bridge:
            hw = self.hardware_bridge.get_system_status()
            self.sys_state.hardware_state = hw.get("state", "unknown")
            self.sys_state.left_arm_positions = hw.get("left_arm_positions")
            self.sys_state.right_arm_positions = hw.get("right_arm_positions")

    def _monitor_perf(self, tic: float):
        dt = time.time() - tic
        self._loop_times.append(dt)
        if len(self._loop_times) > 50:
            self._loop_times.pop(0)
        self.sys_state.avg_loop_time = sum(self._loop_times) / len(self._loop_times)

        # log at cfg.log_frequency
        if (time.time() - self._start_ts) % (1.0 / self.cfg.log_frequency) < 0.05:
            self._notify_status()

    # ────────────────────────────────────────────────────────────────────────
    # 사용자 인터페이스 (터미널)
    # ----------------------------------------------------------------------

    def _process_terminal(self):
        if not self.terminal:
            return
        cmd = self.terminal.poll()
        if not cmd:
            return
        match cmd.lower().strip():
            case "pause":
                self.state = ControllerState.PAUSED
                self.inference_engine.pause()
            case "resume":
                self.state = ControllerState.RUNNING
                self.inference_engine.resume()
            case "stop":
                self._trigger_emergency_stop()
            case _:
                self.log.info("Unknown terminal cmd: %s", cmd)

    # ────────────────────────────────────────────────────────────────────────
    # 안전 & 에러 핸들링
    # ----------------------------------------------------------------------

    def _trigger_emergency_stop(self):
        self.log.critical("Emergency‑STOP triggered!")
        self.state = ControllerState.EMERGENCY_STOP
        if self.inference_engine:
            self.inference_engine.pause()
        if self.hardware_bridge:
            self.hardware_bridge.emergency_stop()
        if self.safety_manager:
            self.safety_manager.handle_emergency()
        self.sys_state.safety_status = False
        self._notify_status()

    # ────────────────────────────────────────────────────────────────────────
    # 콜백 디스패치
    # ----------------------------------------------------------------------

    def _notify_status(self):
        for cb in self._status_cbs:
            try:
                cb(self.sys_state)
            except Exception:  # noqa: BLE001
                self.log.exception("Status callback failed")

    def _notify_error(self, exc: Exception):
        for cb in self._error_cbs:
            try:
                cb(exc)
            except Exception:  # noqa: BLE001
                self.log.exception("Error callback failed")

# ───────────────────────────────────────────────────────────────────────────────
# 간단한 팩토리 / 테스트
# ------------------------------------------------------------------------------

def create_robot_controller(**kwargs) -> RobotController:
    cfg = RobotControllerConfig(**kwargs)
    return RobotController(cfg)


def _demo():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    controller = create_robot_controller(use_mock_data=True, control_frequency=5.0)

    def printer(state: SystemState):
        print(f"Status: {state.controller_state}  |  Freq: {state.current_frequency:.1f} Hz")

    controller.add_status_callback(printer)

    with controller:
        for _ in range(15):
            time.sleep(1)


if __name__ == "__main__":
    _demo()
