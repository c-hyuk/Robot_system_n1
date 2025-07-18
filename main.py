"""
메인 시스템 실행 파일
전체 로봇 시스템을 초기화하고 실행
"""

import os
import sys
import time
import signal
import argparse
import logging
from typing import Optional
from pathlib import Path

# 시스템 패키지 추가
sys.path.append(str(Path(__file__).parent))

from control.robot_controller import RobotController, RobotControllerConfig, create_robot_controller
from config.hardware_config import initialize_hardware_config, get_hardware_config
from utils.data_types import SystemConfig

# Piper SDK import (DI용)
try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None


class RobotSystem:
    """전체 로봇 시스템 관리자"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        로봇 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.controller: Optional[RobotController] = None
        self.running = False
        
        # 로깅 설정
        self.logger = logging.getLogger("RobotSystem")
        
        # 신호 처리 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🤖 Robot System Manager initialized")
    
    def _signal_handler(self, signum, frame):
        """신호 처리 (Ctrl+C 등)"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def initialize(self, args) -> bool:
        """시스템 초기화"""
        try:
            self.logger.info("🔧 Initializing robot system...")
            
            # 1. 하드웨어 설정 초기화
            hw_config = initialize_hardware_config(self.config_path)
            self.logger.info("✅ Hardware configuration loaded")
            
            # 2. 로봇 컨트롤러 설정
            controller_config = RobotControllerConfig(
                model_path=args.model_path,
                control_frequency=args.frequency,
                execution_mode=args.execution_mode,
                use_mock_data=args.mock_data,
                left_arm_can_port=args.left_can,
                right_arm_can_port=args.right_can,
                enable_safety_checks=not args.disable_safety,
                emergency_stop_enabled=not args.disable_emergency_stop
                # enable_performance_monitoring=args.enable_monitoring  # 제거: 존재하지 않음
            )
            
            # 2.5. PiperInterface 객체 생성 (DI)
            left_piper = None
            right_piper = None
            if C_PiperInterface_V2 is not None:
                left_piper = C_PiperInterface_V2(
                    can_name=args.left_can,
                    judge_flag=False,
                    can_auto_init=True,
                    dh_is_offset=1,
                    start_sdk_joint_limit=True,
                    start_sdk_gripper_limit=True
                )
                right_piper = C_PiperInterface_V2(
                    can_name=args.right_can,
                    judge_flag=False,
                    can_auto_init=True,
                    dh_is_offset=1,
                    start_sdk_joint_limit=True,
                    start_sdk_gripper_limit=True
                )
            
            # 3. 로봇 컨트롤러 생성 (DI 적용)
            self.controller = RobotController(controller_config, left_piper=left_piper, right_piper=right_piper)
            
            # 4. 상태 모니터링 콜백 등록
            self.controller.add_status_callback(self._status_callback)
            self.controller.add_error_callback(self._error_callback)
            
            self.logger.info("✅ Robot system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """시스템 시작"""
        if not self.controller:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.logger.info("🚀 Starting robot system...")
            
            # 하드웨어 연결 상태 확인
            hw_config = get_hardware_config()
            if not hw_config.is_hardware_ready:
                self.logger.warning("⚠️ Hardware not fully ready, continuing anyway...")
            
            # 컨트롤러 시작
            if hasattr(self.controller, 'start') and self.controller.start():
                self.running = True
                self.logger.info("✅ Robot system started successfully")
                
                # 시스템 정보 출력
                self._print_system_info()
                return True
            else:
                self.logger.error("❌ Failed to start robot controller")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ System start failed: {e}")
            return False
    
    def run(self):
        """시스템 메인 루프"""
        if not self.running:
            self.logger.error("System not running")
            return
        
        self.logger.info("🔄 Entering main loop...")
        self.logger.info("Press Ctrl+C to stop the system")
        
        try:
            # 메인 루프
            loop_count = 0
            status_interval = 10  # 10초마다 상태 출력
            
            while self.running:
                time.sleep(1)
                loop_count += 1
                
                # 주기적 상태 확인
                if loop_count % status_interval == 0:
                    self._check_system_health()
                
                # 사용자 입력 처리 (간단한 명령어)
                self._process_simple_commands()
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """시스템 종료"""
        if not self.running:
            return
        
        self.logger.info("🛑 Shutting down robot system...")
        self.running = False
        
        try:
            if self.controller:
                if hasattr(self.controller, 'stop'):
                    self.controller.stop()
            
            self.logger.info("✅ Robot system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _status_callback(self, system_state):
        """시스템 상태 콜백"""
        # 주기적 상태 로깅 (DEBUG 레벨)
        self.logger.debug(f"System status: {system_state.controller_state}, "
                         f"Freq: {system_state.current_frequency:.1f}Hz, "
                         f"Errors: {system_state.error_count}")
    
    def _error_callback(self, error: Exception):
        """에러 콜백"""
        self.logger.error(f"System error: {error}")
    
    def _check_system_health(self):
        """시스템 건강 상태 확인"""
        if not self.controller:
            return
        
        try:
            state = getattr(self.controller, 'get_state', lambda: None)()
            metrics = getattr(self.controller, 'get_metrics', lambda: None)()
            # 성능 경고
            if state:
                if getattr(state, 'current_frequency', 10.0) < 8.0:
                    self.logger.warning(f"Low control frequency: {getattr(state, 'current_frequency', 0):.1f}Hz")
                if getattr(state, 'avg_loop_time', 0.0) > 0.12:
                    self.logger.warning(f"High loop time: {getattr(state, 'avg_loop_time', 0)*1000:.1f}ms")
                if getattr(state, 'error_count', 0) > 10:
                    self.logger.warning(f"High error count: {getattr(state, 'error_count', 0)}")
                if getattr(state, 'error_count', 0) == 0 and getattr(state, 'current_frequency', 0) >= 9.0:
                    self.logger.info(f"✅ System healthy: {getattr(state, 'current_frequency', 0):.1f}Hz, "
                                   f"{getattr(state, 'total_commands_sent', 0)} commands sent")
                
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    def _process_simple_commands(self):
        """간단한 사용자 명령 처리"""
        # TODO: 비동기 입력 처리나 웹 인터페이스 구현
        pass
    
    def _print_system_info(self):
        """시스템 정보 출력"""
        if not self.controller:
            return
        
        print("\n" + "="*60)
        print("🤖 DUAL PIPER ROBOT CONTROL SYSTEM")
        print("="*60)
        
        # 하드웨어 정보
        hw_config = get_hardware_config()
        print(f"📡 Hardware Configuration:")
        print(f"  Control Frequency: {hw_config.system_config.control_frequency}Hz")
        print(f"  Arms: {list(hw_config.system_config.arms.keys())}")
        print(f"  Cameras: {list(hw_config.system_config.cameras.keys())}")
        
        # 컨트롤러 설정
        config = getattr(self.controller, 'get_config', lambda: None)()
        print(f"\n🧠 Controller Configuration:")
        if config:
            print(f"  Model: {getattr(config, 'model_path', '-')}")
            print(f"  Execution Mode: {getattr(config, 'execution_mode', '-')}")
            print(f"  Safety Checks: {'Enabled' if getattr(config, 'enable_safety_checks', False) else 'Disabled'}")
            print(f"  Mock Data: {'Yes' if getattr(config, 'use_mock_data', False) else 'No'}")
            print(f"\n🔌 CAN Configuration:")
            print(f"  Left Arm: {getattr(config, 'left_arm_can_port', '-')}")
            print(f"  Right Arm: {getattr(config, 'right_arm_can_port', '-')}")
        else:
            print("  (No config available)")
        
        print("="*60)
        print("🎮 Available Commands:")
        print("  Ctrl+C: Emergency stop and shutdown")
        print("  (System runs autonomously with GR00T model)")
        print("="*60 + "\n")


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="Dual Piper Robot Control System")
    
    # 모델 설정
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.5-3B",
                        help="GR00T model path or HuggingFace ID")
    parser.add_argument("--frequency", type=float, default=10.0,
                        help="Control frequency in Hz")
    parser.add_argument("--execution-mode", type=str, default="position",
                        choices=["position", "velocity", "trajectory"],
                        help="Action execution mode")
    
    # 하드웨어 설정
    parser.add_argument("--left-can", type=str, default="can0",
                        help="Left arm CAN port")
    parser.add_argument("--right-can", type=str, default="can1", 
                        help="Right arm CAN port")
    
    # 안전 설정
    parser.add_argument("--disable-safety", action="store_true",
                        help="Disable safety checks")
    parser.add_argument("--disable-emergency-stop", action="store_true",
                        help="Disable emergency stop")
    
    # 디버깅/테스트
    parser.add_argument("--mock-data", action="store_true",
                        help="Use mock data instead of real sensors")
    parser.add_argument("--config", type=str,
                        help="Hardware configuration file path")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--enable-monitoring", action="store_true", default=True,
                        help="Enable performance monitoring")
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 (타임스탬프 포함)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"robot_system_{timestamp}.log"
    
    # 로깅 포맷
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 파일 및 콘솔 핸들러 설정
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 일부 라이브러리 로그 레벨 조정
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logger = logging.getLogger("Main")
    logger.info(f"Logging initialized: {log_file}")


def check_dependencies():
    """종속성 확인"""
    logger = logging.getLogger("Dependencies")
    
    try:
        # 필수 패키지 확인
        import torch
        import numpy as np
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ NumPy: {np.__version__}")
        
        # Piper SDK 확인
        if C_PiperInterface_V2 is None:
            logger.warning("⚠️ Piper SDK not available - using mock interface")
        else:
            logger.info("✅ Piper SDK available")
        
        # CUDA 확인
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠️ CUDA not available - using CPU")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False


def check_hardware():
    """하드웨어 확인"""
    logger = logging.getLogger("Hardware")
    
    try:
        # CAN 인터페이스 확인
        import subprocess
        result = subprocess.run(["ifconfig"], capture_output=True, text=True)
        
        can_interfaces = []
        for line in result.stdout.split('\n'):
            if 'can' in line and ':' in line:
                interface = line.split(':')[0].strip()
                can_interfaces.append(interface)
        
        if can_interfaces:
            logger.info(f"✅ CAN interfaces found: {can_interfaces}")
        else:
            logger.warning("⚠️ No CAN interfaces found")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Hardware check failed: {e}")
        return True  # 계속 진행


def main():
    """메인 함수"""
    # 명령행 인수 파싱
    args = parse_arguments()
    
    # 로깅 설정
    setup_logging(args.log_level)
    
    logger = logging.getLogger("Main")
    logger.info("🚀 Starting Dual Piper Robot Control System")
    
    try:
        # 종속성 확인
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            return 1
        
        # 하드웨어 확인
        check_hardware()
        
        # 로봇 시스템 생성
        robot_system = RobotSystem(config_path=args.config)
        
        # 시스템 초기화
        if not robot_system.initialize(args):
            logger.error("❌ System initialization failed")
            return 1
        
        # 시스템 시작
        if not robot_system.start():
            logger.error("❌ System start failed")
            return 1
        
        # 메인 루프 실행
        robot_system.run()
        
        logger.info("✅ System shutdown complete")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)