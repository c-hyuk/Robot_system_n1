"""
Robot Controller - 중앙 제어 시스템
모든 모듈을 통합하여 완전한 로봇 시스템 구성
"""

import time
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

# 기존 완성된 모듈들
from model.inference_engine import RealTimeInferenceEngine, create_inference_engine
from model.action_decoder import ActionDecoderManager, create_action_decoder
from config.hardware_config import get_hardware_config
from utils.data_types import SystemConfig

# 새로 구현할 모듈들
from communication.hardware_bridge import PiperHardwareBridge
from control.safety_manager import SafetyManager
# from communication.terminal_interface import TerminalInterface  # TODO: 추후 구현


class ControllerState(Enum):
    """로봇 컨트롤러 상태"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobotControllerConfig:
    """로봇 컨트롤러 설정"""
    # 타이밍 설정
    control_frequency: float = 10.0        # Hz (GR00T 추론 주기와 동일)
    max_loop_time: float = 0.15           # 150ms 제한
    
    # 모델 설정
    model_path: str = "nvidia/gr00t-1.5b"
    embodiment_name: str = "dual_piper_arm"
    use_mock_data: bool = False
    
    # 안전 설정
    enable_safety_checks: bool = True
    emergency_stop_enabled: bool = True
    max_consecutive_errors: int = 3
    
    # 실행 모드
    execution_mode: str = "position"       # position/velocity/trajectory
    auto_recovery: bool = True
    
    # 모니터링
    enable_performance_monitoring: bool = True
    log_frequency: float = 1.0            # 1Hz
    
    # CAN 설정 (Piper arm)
    left_arm_can_port: str = "can0"
    right_arm_can_port: str = "can1"
    can_auto_init: bool = True


@dataclass
class SystemState:
    """시스템 상태 정보"""
    # 전체 시스템 상태
    controller_state: str = "idle"
    inference_state: str = "idle"
    hardware_state: str = "disconnected"
    
    # 성능 메트릭
    current_frequency: float = 0.0
    avg_loop_time: float = 0.0
    error_count: int = 0
    last_command_time: float = 0.0
    
    # 로봇 상태
    left_arm_positions: Optional[List[float]] = None
    right_arm_positions: Optional[List[float]] = None
    safety_status: bool = True
    
    # 통계
    total_commands_sent: int = 0
    total_errors: int = 0
    uptime_seconds: float = 0.0


class RobotController:
    """메인 로봇 컨트롤러"""
    
    def __init__(self, config: Optional[RobotControllerConfig] = None):
        """
        로봇 컨트롤러 초기화
        
        Args:
            config: 컨트롤러 설정
        """
        self.config = config or RobotControllerConfig()
        
        # 하드웨어 설정 로드
        self.hw_config = get_hardware_config()
        
        # 상태 관리
        self.state = ControllerState.IDLE
        self.system_state = SystemState()
        
        # 메인 컨트롤 스레드
        self.control_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 핵심 모듈들
        self.inference_engine: Optional[RealTimeInferenceEngine] = None
        self.action_decoder: Optional[ActionDecoderManager] = None
        self.hardware_bridge: Optional[PiperHardwareBridge] = None
        self.safety_manager: Optional[SafetyManager] = None
        self.terminal_interface = None  # TODO: TerminalInterface 구현 후
        
        # 데이터 큐 및 동기화
        self.command_queue = queue.Queue(maxsize=10)
        self.action_lock = threading.Lock()
        self.last_action: Optional[Dict[str, Any]] = None
        
        # 성능 모니터링
        self.start_time = None
        self.loop_times: List[float] = []
        self.max_loop_time_history = 50
        
        # 콜백 함수들
        self.status_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # 로깅
        self.logger = logging.getLogger("RobotController")
        
        self.logger.info("Robot Controller initialized")
    
    def add_status_callback(self, callback: Callable[[SystemState], None]):
        """상태 업데이트 콜백 추가"""
        self.status_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """에러 콜백 추가"""
        self.error_callbacks.append(callback)
    
    def start_system(self) -> bool:
        """시스템 시작"""
        if self.state != ControllerState.IDLE:
            self.logger.warning(f"Cannot start system from state: {self.state}")
            return False
        
        self.logger.info("Starting robot control system...")
        self.state = ControllerState.INITIALIZING
        
        try:
            # 1. 추론 엔진 초기화
            if not self._initialize_inference_engine():
                raise Exception("Failed to initialize inference engine")
            
            # 2. 액션 디코더 초기화
            if not self._initialize_action_decoder():
                raise Exception("Failed to initialize action decoder")
            
            # 3. 하드웨어 브릿지 초기화 (Mock으로 시작)
            if not self._initialize_hardware_bridge():
                raise Exception("Failed to initialize hardware bridge")
            
            # 4. 안전 관리자 초기화
            if not self._initialize_safety_manager():
                raise Exception("Failed to initialize safety manager")
            
            # 5. 터미널 인터페이스 초기화
            if not self._initialize_terminal_interface():
                raise Exception("Failed to initialize terminal interface")
            
            # 6. 제어 루프 스레드 시작
            self.stop_event.clear()
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            
            # 7. 시스템 상태 업데이트
            self.state = ControllerState.RUNNING
            self.start_time = time.time()
            self.system_state.controller_state = "running"
            
            self.logger.info("✅ Robot control system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start system: {e}")
            self.state = ControllerState.ERROR
            self._notify_error_callbacks(e)
            return False
    
    def stop_system(self):
        """시스템 중지"""
        self.logger.info("Stopping robot control system...")
        
        # 중지 신호 발송
        self.stop_event.set()
        
        # 제어 스레드 종료 대기
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=3.0)
        
        # 각 모듈 정리
        if self.inference_engine:
            self.inference_engine.stop()
        
        if self.hardware_bridge:
            self.hardware_bridge.disconnect()
        
        if self.safety_manager:
            self.safety_manager.stop_monitoring()
        
        self.state = ControllerState.IDLE
        self.system_state.controller_state = "idle"
        
        self.logger.info("Robot control system stopped")
    
    def pause_system(self):
        """시스템 일시정지"""
        if self.state == ControllerState.RUNNING:
            self.state = ControllerState.PAUSED
            self.system_state.controller_state = "paused"
            
            if self.inference_engine:
                self.inference_engine.pause()
            
            self.logger.info("System paused")
    
    def resume_system(self):
        """시스템 재개"""
        if self.state == ControllerState.PAUSED:
            self.state = ControllerState.RUNNING
            self.system_state.controller_state = "running"
            
            if self.inference_engine:
                self.inference_engine.resume()
            
            self.logger.info("System resumed")
    
    def emergency_stop(self):
        """비상 정지"""
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        
        self.state = ControllerState.EMERGENCY_STOP
        self.system_state.controller_state = "emergency_stop"
        self.system_state.safety_status = False
        
        # 추론 엔진 중지
        if self.inference_engine:
            self.inference_engine.pause()
        
        # 하드웨어 비상 정지
        if self.hardware_bridge:
            self.hardware_bridge.emergency_stop()
        
        # 안전 관리자 비상 처리
        if self.safety_manager:
            self.safety_manager.handle_emergency()
        
        # 모든 콜백에 알림
        for callback in self.status_callbacks:
            try:
                callback(self.system_state)
            except Exception as e:
                self.logger.error(f"Status callback error: {e}")
    
    def reset_errors(self):
        """에러 상태 리셋"""
        if self.state in [ControllerState.ERROR, ControllerState.EMERGENCY_STOP]:
            self.state = ControllerState.IDLE
            self.system_state.error_count = 0
            self.system_state.safety_status = True
            self.logger.info("Errors reset, system ready to start")
    
    def _initialize_inference_engine(self) -> bool:
        """추론 엔진 초기화"""
        try:
            self.inference_engine = create_inference_engine(
                model_path=self.config.model_path,
                target_frequency=self.config.control_frequency,
                use_mock_data=self.config.use_mock_data
            )
            
            # 액션 콜백 등록
            self.inference_engine.add_action_callback(self._handle_inference_action)
            self.inference_engine.add_error_callback(self._handle_inference_error)
            
            # 추론 엔진 시작
            if not self.inference_engine.start():
                return False
            
            self.system_state.inference_state = "running"
            self.logger.info("✅ Inference engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Inference engine initialization failed: {e}")
            return False
    
    def _initialize_action_decoder(self) -> bool:
        """액션 디코더 초기화"""
        try:
            self.action_decoder = create_action_decoder(
                embodiment_name=self.config.embodiment_name,
                execution_mode=self.config.execution_mode
            )
            
            self.logger.info("✅ Action decoder initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Action decoder initialization failed: {e}")
            return False
    
    def _initialize_hardware_bridge(self) -> bool:
        """하드웨어 브릿지 초기화"""
        try:
            self.hardware_bridge = PiperHardwareBridge(
                left_can_port=self.config.left_arm_can_port,
                right_can_port=self.config.right_arm_can_port,
                auto_enable=True,
                gripper_enabled=True
            )
            
            if self.hardware_bridge.connect():
                self.system_state.hardware_state = "connected"
                self.logger.info("✅ Piper Hardware bridge initialized")
                return True
            else:
                self.logger.error("❌ Failed to connect to Piper hardware")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Hardware bridge initialization failed: {e}")
            return False
    
    def _initialize_safety_manager(self) -> bool:
        """안전 관리자 초기화"""
        try:
            self.safety_manager = SafetyManager(self.hw_config)
            self.safety_manager.start_monitoring()
            self.logger.info("✅ Safety manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Safety manager initialization failed: {e}")
            return False
    
    def _initialize_terminal_interface(self) -> bool:
        """터미널 인터페이스 초기화 (현재는 Mock)"""
        try:
            # TODO: 실제 TerminalInterface 구현 후 교체
            self.terminal_interface = MockTerminalInterface()
            self.logger.info("✅ Terminal interface initialized (Mock)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Terminal interface initialization failed: {e}")
            return False
    
    def _control_loop(self):
        """메인 제어 루프 (별도 스레드에서 실행)"""
        target_interval = 1.0 / self.config.control_frequency
        
        self.logger.info(f"🔄 Control loop started at {self.config.control_frequency}Hz")
        
        while not self.stop_event.is_set():
            if self.state != ControllerState.RUNNING:
                time.sleep(0.1)
                continue
            
            loop_start = time.time()
            
            try:
                # 1. 사용자 명령 처리
                self._process_user_commands()
                
                # 2. 추론 결과 처리 (콜백으로 이미 처리되지만 큐에서도 확인)
                self._process_action_queue()
                
                # 3. 시스템 상태 업데이트
                self._update_system_state()
                
                # 4. 안전성 검사
                if not self._check_safety():
                    self.logger.warning("Safety check failed")
                    continue
                
                # 5. 성능 모니터링
                self._monitor_performance(loop_start)
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                self.system_state.error_count += 1
                
                if self.system_state.error_count >= self.config.max_consecutive_errors:
                    self.emergency_stop()
                    break
            
            # 타이밍 조절
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_interval - elapsed)
            
            if elapsed > self.config.max_loop_time:
                self.logger.warning(f"Control loop slow: {elapsed*1000:.1f}ms")
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _handle_inference_action(self, action_tokens: Dict[str, np.ndarray]):
        """추론 엔진에서 액션 토큰 수신 시 호출되는 콜백"""
        try:
            # 1. 액션 토큰을 로봇 명령으로 디코딩
            if self.action_decoder is not None:
                robot_commands = self.action_decoder.decode_action(action_tokens)
            else:
                self.logger.error("action_decoder is not initialized")
                return
            # 2. 안전성 검사
            if self.safety_manager is not None:
                if not self.safety_manager.validate_command(robot_commands):
                    self.logger.warning("Unsafe command rejected")
                    return
            else:
                self.logger.error("safety_manager is not initialized")
                return
            # 3. 하드웨어에 명령 전송
            success = self._send_hardware_commands(robot_commands)
            if success:
                # 4. 상태 업데이트
                with self.action_lock:
                    self.last_action = robot_commands
                    self.system_state.last_command_time = time.time()
                    self.system_state.total_commands_sent += 1
            
        except Exception as e:
            self.logger.error(f"Error handling inference action: {e}")
            self.system_state.error_count += 1
    
    def _handle_inference_error(self, error: Exception):
        """추론 엔진 에러 처리"""
        self.logger.error(f"Inference engine error: {error}")
        self.system_state.error_count += 1
        self.system_state.inference_state = "error"
    
    def _send_hardware_commands(self, robot_commands: Dict[str, Any]) -> bool:
        """하드웨어에 명령 전송"""
        try:
            if not self.hardware_bridge:
                return False
            
            success = True
            
            # 각 팔에 명령 전송
            for arm_name, arm_commands in robot_commands.get('arms', {}).items():
                arm_success = self.hardware_bridge.send_arm_command(arm_name, arm_commands)
                success = success and arm_success
            
            return success
            
        except Exception as e:
            self.logger.error(f"Hardware command error: {e}")
            return False
    
    def _process_user_commands(self):
        """사용자 명령 처리"""
        if not self.terminal_interface:
            return
        
        try:
            command = self.terminal_interface.get_user_commands()
            if command:
                self._execute_user_command(command)
        except Exception as e:
            self.logger.error(f"User command processing error: {e}")
    
    def _execute_user_command(self, command: str):
        """사용자 명령 실행"""
        command = command.strip().lower()
        
        if command == "pause":
            self.pause_system()
        elif command == "resume":
            self.resume_system()
        elif command == "stop":
            self.emergency_stop()
        elif command == "reset":
            self.reset_errors()
        elif command == "status":
            self._print_system_status()
        else:
            self.logger.info(f"Unknown command: {command}")
    
    def _process_action_queue(self):
        """액션 큐 처리 (추가적인 액션 처리가 필요한 경우)"""
        try:
            while not self.command_queue.empty():
                action = self.command_queue.get_nowait()
                # 필요시 추가 처리
        except queue.Empty:
            pass
    
    def _update_system_state(self):
        """시스템 상태 업데이트"""
        current_time = time.time()
        
        if self.start_time:
            self.system_state.uptime_seconds = current_time - self.start_time
        
        # 추론 엔진 상태 업데이트
        if self.inference_engine:
            engine_status = self.inference_engine.get_engine_status()
            self.system_state.current_frequency = engine_status.get('actual_frequency', 0.0)
            self.system_state.inference_state = engine_status.get('state', 'unknown')
        
        # 하드웨어 상태 업데이트
        if self.hardware_bridge:
            hw_status = self.hardware_bridge.get_system_status()
            self.system_state.hardware_state = hw_status.get('state', 'unknown')
            
            # 로봇 팔 위치 업데이트
            self.system_state.left_arm_positions = hw_status.get('left_arm_positions', [])
            self.system_state.right_arm_positions = hw_status.get('right_arm_positions', [])
            
            # 안전 관리자에 상태 전달
            if self.safety_manager:
                for arm_name, arm_status in hw_status.get('arms', {}).items():
                    self.safety_manager.update_arm_state(arm_name, arm_status)
    
    def _check_safety(self) -> bool:
        """안전성 검사"""
        if not self.safety_manager:
            return True
        
        try:
            safety_status = self.safety_manager.get_safety_status()
            self.system_state.safety_status = safety_status.get('safe', True)
            return self.system_state.safety_status
        except Exception as e:
            self.logger.error(f"Safety check error: {e}")
            return False
    
    def _monitor_performance(self, loop_start: float):
        """성능 모니터링"""
        loop_time = time.time() - loop_start
        
        # 루프 시간 히스토리 업데이트
        self.loop_times.append(loop_time)
        if len(self.loop_times) > self.max_loop_time_history:
            self.loop_times = self.loop_times[-self.max_loop_time_history:]
        
        # 평균 루프 시간 계산
        self.system_state.avg_loop_time = sum(self.loop_times) / len(self.loop_times)
        
        # 주기적으로 상태 콜백 호출
        if self.start_time is not None:
            if (time.time() - self.start_time) % (1.0 / self.config.log_frequency) < 0.1:
                self._notify_status_callbacks()
    
    def _notify_status_callbacks(self):
        """상태 콜백 호출"""
        for callback in self.status_callbacks:
            try:
                callback(self.system_state)
            except Exception as e:
                self.logger.error(f"Status callback error: {e}")
    
    def _notify_error_callbacks(self, error: Exception):
        """에러 콜백 호출"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error callback error: {e}")
    
    def _print_system_status(self):
        """시스템 상태 출력"""
        print(f"\n🤖 Robot Controller Status:")
        print(f"  State: {self.system_state.controller_state}")
        print(f"  Inference: {self.system_state.inference_state}")
        print(f"  Hardware: {self.system_state.hardware_state}")
        print(f"  Frequency: {self.system_state.current_frequency:.1f}Hz")
        print(f"  Loop Time: {self.system_state.avg_loop_time*1000:.1f}ms")
        print(f"  Commands Sent: {self.system_state.total_commands_sent}")
        print(f"  Errors: {self.system_state.error_count}")
        print(f"  Safety: {'✅' if self.system_state.safety_status else '❌'}")
        print(f"  Uptime: {self.system_state.uptime_seconds:.1f}s")
    
    def get_system_state(self) -> SystemState:
        """시스템 상태 반환"""
        return self.system_state
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 메트릭 반환"""
        metrics = {}
        
        # 추론 엔진 메트릭
        if self.inference_engine:
            metrics.update(self.inference_engine.get_performance_metrics())
        
        # 액션 디코더 메트릭
        if self.action_decoder:
            decoder_stats = self.action_decoder.get_decoder_stats()
            metrics['decoder_avg_time_ms'] = decoder_stats.get('avg_decode_time_ms', 0)
        
        # 컨트롤러 메트릭
        metrics.update({
            'controller_frequency': self.system_state.current_frequency,
            'avg_loop_time_ms': self.system_state.avg_loop_time * 1000,
            'total_commands': self.system_state.total_commands_sent,
            'error_count': self.system_state.error_count,
            'uptime_seconds': self.system_state.uptime_seconds
        })
        
        return metrics
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트 (런타임 중)"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Config updated: {key} = {value}")
        
        # 액션 디코더 모드 변경
        if 'execution_mode' in new_config and self.action_decoder:
            self.action_decoder.set_execution_mode(new_config['execution_mode'])
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_system()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_system()


# Mock 클래스들 (실제 구현 전까지 임시)
class MockPiperHardwareBridge:
    """Mock Piper 하드웨어 브릿지"""
    
    def __init__(self, left_can_port: str, right_can_port: str):
        self.left_can_port = left_can_port
        self.right_can_port = right_can_port
        self.connected = False
        self.logger = logging.getLogger("MockPiperHardwareBridge")
    
    def connect(self) -> bool:
        self.connected = True
        self.logger.info(f"🔗 Mock connected to {self.left_can_port}, {self.right_can_port}")
        return True
    
    def send_arm_command(self, arm_name: str, command: Dict[str, Any]) -> bool:
        self.logger.debug(f"📤 Mock command to {arm_name}: {list(command.keys())}")
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            'state': 'connected' if self.connected else 'disconnected',
            'left_arm_positions': [0.0] * 7,
            'right_arm_positions': [0.0] * 7,
        }


class MockSafetyManager:
    """Mock 안전 관리자"""
    
    def __init__(self, hw_config):
        self.hw_config = hw_config
        self.logger = logging.getLogger("MockSafetyManager")
    
    def validate_command(self, command: Dict[str, Any]) -> bool:
        # 간단한 검증 (실제로는 더 복잡)
        return True
    
    def get_safety_status(self) -> Dict[str, Any]:
        return {'safe': True, 'warnings': []}


class MockTerminalInterface:
    """Mock 터미널 인터페이스"""
    
    def __init__(self):
        self.command_queue = queue.Queue()
        self.logger = logging.getLogger("MockTerminalInterface")
        
        # 시뮬레이션용 명령어들
        self.test_commands = ["status", "pause", "resume"]
        self.command_index = 0
    
    def get_user_commands(self) -> Optional[str]:
        # Mock: 30초마다 테스트 명령어 반환
        if int(time.time()) % 30 == 0:
            cmd = self.test_commands[self.command_index % len(self.test_commands)]
            self.command_index += 1
            return cmd
        return None


# 편의용 함수들
def create_robot_controller(
    model_path: str = "nvidia/gr00t-1.5b",
    control_frequency: float = 10.0,
    execution_mode: str = "position",
    use_mock_data: bool = False
) -> RobotController:
    """로봇 컨트롤러 생성"""
    config = RobotControllerConfig(
        model_path=model_path,
        control_frequency=control_frequency,
        execution_mode=execution_mode,
        use_mock_data=use_mock_data
    )
    return RobotController(config)


def test_robot_controller():
    """로봇 컨트롤러 테스트"""
    print("🤖 Testing Robot Controller...")
    
    # 상태 모니터링 콜백
    def status_callback(state: SystemState):
        print(f"\n📊 Status Update:")
        print(f"  Controller: {state.controller_state}")
        print(f"  Frequency: {state.current_frequency:.1f}Hz")
        print(f"  Commands: {state.total_commands_sent}")
        print(f"  Errors: {state.error_count}")
        print(f"  Uptime: {state.uptime_seconds:.1f}s")
    
    def error_callback(error: Exception):
        print(f"❌ Error occurred: {error}")
    
    try:
        # 로봇 컨트롤러 생성 (Mock 모드)
        controller = create_robot_controller(
            model_path="mock_model_path",
            control_frequency=5.0,  # 5Hz로 테스트
            execution_mode="position",
            use_mock_data=True
        )
        
        # 콜백 등록
        controller.add_status_callback(status_callback)
        controller.add_error_callback(error_callback)
        
        print("✅ Robot Controller created")
        
        # Context manager로 시스템 실행
        with controller:
            print("🚀 Robot Control System started")
            
            # 30초 동안 실행하며 다양한 테스트
            for i in range(30):
                time.sleep(1)
                
                # 5초마다 상태 출력
                if i % 5 == 0:
                    print(f"\n⏰ Second {i+1}:")
                    state = controller.get_system_state()
                    print(f"  State: {state.controller_state}")
                    print(f"  Inference: {state.inference_state}")
                    print(f"  Hardware: {state.hardware_state}")
                    print(f"  Safety: {'✅' if state.safety_status else '❌'}")
                
                # 10초에 일시정지 테스트
                if i == 10:
                    print("\n⏸️ Testing pause...")
                    controller.pause_system()
                
                # 15초에 재개 테스트
                if i == 15:
                    print("\n▶️ Testing resume...")
                    controller.resume_system()
                
                # 25초에 성능 메트릭 출력
                if i == 25:
                    print("\n📈 Performance Metrics:")
                    metrics = controller.get_performance_metrics()
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")
        
        print("✅ Robot Controller test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 실행
    test_robot_controller()