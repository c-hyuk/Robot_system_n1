"""
통합 데이터 수집 관리자
모든 하드웨어(카메라, 로봇 팔, 텍스트 입력)에서 데이터를 수집하여 통합 관리
"""

import time
import threading
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np

from utils.data_types import RobotData, ModalityData
from data.collectors.vision_collector import VisionCollectorManager
from data.collectors.state_collector import RobotStateCollectorManager  
from data.collectors.text_collector import TextCollectorManager


class MainDataCollector:
    """메인 데이터 수집 관리자"""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.is_running = False
        
        # 개별 수집기들
        self.vision_collector: Optional[VisionCollectorManager] = None
        self.state_collector: Optional[RobotStateCollectorManager] = None
        self.text_collector: Optional[TextCollectorManager] = None
        
        # 데이터 동기화
        self.data_lock = threading.Lock()
        self.latest_robot_data: Optional[RobotData] = None
        self.last_update_time = 0.0
        
        # 통계
        self.total_frames_collected = 0
        self.total_states_collected = 0
        self.total_commands_collected = 0
        self.start_time = None
        
        # 로깅 설정
        self.logger = logging.getLogger("MainDataCollector")
        
        self._initialize_collectors()
    
    def _initialize_collectors(self) -> None:
        """모든 수집기 초기화"""
        try:
            # 비전 수집기 초기화
            self.vision_collector = VisionCollectorManager(use_mock=self.use_mock)
            self.logger.info("Vision collector initialized")
            
            # 상태 수집기 초기화
            self.state_collector = RobotStateCollectorManager(use_mock=self.use_mock)
            self.logger.info("State collector initialized")
            
            # 텍스트 수집기 초기화
            self.text_collector = TextCollectorManager(use_mock=self.use_mock)
            self.logger.info("Text collector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize collectors: {e}")
            raise
    
    def start_collection(self) -> bool:
        """모든 데이터 수집 시작"""
        if self.is_running:
            self.logger.warning("Data collection already running")
            return True
        
        self.logger.info("Starting all data collectors...")
        success_count = 0
        total_collectors = 3
        
        # 비전 수집기 시작
        try:
            if self.vision_collector and self.vision_collector.start_all_cameras():
                success_count += 1
                self.logger.info("✓ Vision collector started")
            else:
                self.logger.error("✗ Failed to start vision collector")
        except Exception as e:
            self.logger.error(f"✗ Vision collector error: {e}")
        
        # 상태 수집기 시작
        try:
            if self.state_collector and self.state_collector.start_all_collectors():
                success_count += 1
                self.logger.info("✓ State collector started")
            else:
                self.logger.error("✗ Failed to start state collector")
        except Exception as e:
            self.logger.error(f"✗ State collector error: {e}")
        
        # 텍스트 수집기 시작
        try:
            if self.text_collector and self.text_collector.start_collection():
                success_count += 1
                self.logger.info("✓ Text collector started")
            else:
                self.logger.error("✗ Failed to start text collector")
        except Exception as e:
            self.logger.error(f"✗ Text collector error: {e}")
        
        # 결과 확인
        if success_count > 0:
            self.is_running = True
            self.start_time = time.time()
            self.logger.info(f"Data collection started ({success_count}/{total_collectors} collectors)")
            return True
        else:
            self.logger.error("Failed to start any collectors")
            return False
    
    def stop_collection(self) -> None:
        """모든 데이터 수집 중지"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping all data collectors...")
        
        # 모든 수집기 중지
        try:
            if self.vision_collector:
                self.vision_collector.stop_all_cameras()
                self.logger.info("✓ Vision collector stopped")
        except Exception as e:
            self.logger.error(f"Error stopping vision collector: {e}")
        
        try:
            if self.state_collector:
                self.state_collector.stop_all_collectors()
                self.logger.info("✓ State collector stopped")
        except Exception as e:
            self.logger.error(f"Error stopping state collector: {e}")
        
        try:
            if self.text_collector:
                self.text_collector.stop_collection()
                self.logger.info("✓ Text collector stopped")
        except Exception as e:
            self.logger.error(f"Error stopping text collector: {e}")
        
        self.is_running = False
        self.logger.info("All data collectors stopped")
    
    def collect_synchronized_data(self) -> Optional[RobotData]:
        """동기화된 로봇 데이터 수집"""
        if not self.is_running:
            return None
        
        with self.data_lock:
            try:
                # 새로운 RobotData 인스턴스 생성
                robot_data = RobotData()
                robot_data.timestamp = time.time()
                
                # 비디오 데이터 수집
                if self.vision_collector:
                    video_data = self.vision_collector.get_all_frames()
                    if video_data:
                        robot_data.video_data = video_data
                        self.total_frames_collected += 1
                
                # 상태 데이터 수집
                if self.state_collector:
                    state_data = self.state_collector.get_all_states()
                    if state_data:
                        robot_data.state_data = state_data
                        self.total_states_collected += 1
                
                # 언어 데이터 수집 (새로운 명령어가 있을 때만)
                if self.text_collector and self.text_collector.has_new_commands():
                    language_data = self.text_collector.get_latest_command()
                    if language_data:
                        robot_data.language_data = language_data
                        self.total_commands_collected += 1
                
                # 데이터 유효성 검증
                if self._validate_robot_data(robot_data):
                    self.latest_robot_data = robot_data
                    self.last_update_time = robot_data.timestamp
                    return robot_data
                else:
                    self.logger.warning("Invalid robot data collected")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Error collecting synchronized data: {e}")
                return None
    
    def _validate_robot_data(self, robot_data: RobotData) -> bool:
        """로봇 데이터 유효성 검증"""
        try:
            # 최소한 비디오나 상태 데이터 중 하나는 있어야 함
            has_video = bool(robot_data.video_data)
            has_state = bool(robot_data.state_data)
            
            if not (has_video or has_state):
                self.logger.warning("No valid video or state data")
                return False
            
            # 타임스탬프 검증
            if robot_data.timestamp <= 0:
                self.logger.warning("Invalid timestamp")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating robot data: {e}")
            return False
    
    def get_latest_data(self) -> Optional[RobotData]:
        """최신 로봇 데이터 반환"""
        return self.latest_robot_data
    
    def get_gr00t_format_data(self) -> Optional[ModalityData]:
        """GR00T 모델용 데이터 형식으로 변환"""
        if not self.latest_robot_data:
            return None
        
        # TODO: GR00T 형식으로 데이터 변환 구현
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        status = {
            'main_collector': {
                'is_running': self.is_running,
                'uptime': time.time() - (self.start_time or time.time()),
                'total_frames': self.total_frames_collected,
                'total_states': self.total_states_collected,
                'total_commands': self.total_commands_collected
            }
        }
        
        # 개별 수집기 상태
        if self.vision_collector:
            status['vision'] = self.vision_collector.get_status()
        
        if self.state_collector:
            status['state'] = self.state_collector.get_status()
        
        if self.text_collector:
            status['text'] = self.text_collector.get_status()
        
        return status
    
    def is_system_ready(self) -> bool:
        """시스템이 준비되었는지 확인"""
        try:
            # 비전 시스템 확인
            vision_ready = (self.vision_collector and 
                          self.vision_collector.is_running and
                          bool(self.vision_collector.get_all_frames()))
            
            # 상태 시스템 확인
            state_ready = (self.state_collector and 
                          self.state_collector.is_running and
                          self.state_collector.is_all_arms_ready())
            
            # 텍스트 시스템 확인 (항상 준비된 것으로 간주)
            text_ready = (self.text_collector and 
                         self.text_collector.is_running)
            
            # 최소 요구사항: 비전 + 상태 시스템이 모두 준비되어야 함
            return vision_ready and state_ready and text_ready
            
        except Exception as e:
            self.logger.error(f"Error checking system readiness: {e}")
            return False
    
    def wait_for_system_ready(self, timeout: float = 30.0) -> bool:
        """시스템이 준비될 때까지 대기"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_system_ready():
                self.logger.info("System is ready")
                return True
            
            time.sleep(0.5)
        
        self.logger.warning(f"System not ready after {timeout} seconds")
        return False
    
    def get_data_rates(self) -> Dict[str, float]:
        """데이터 수집 주파수 반환"""
        if not self.start_time:
            return {}
        
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return {}
        
        return {
            'frames_per_second': self.total_frames_collected / elapsed,
            'states_per_second': self.total_states_collected / elapsed,
            'commands_per_second': self.total_commands_collected / elapsed
        }
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_collection()


# 편의용 함수들
def create_main_collector(use_mock: bool = False) -> MainDataCollector:
    """메인 데이터 수집기 생성"""
    return MainDataCollector(use_mock=use_mock)


def test_integrated_collection(duration: float = 10.0, use_mock: bool = True):
    """통합 데이터 수집 테스트"""
    print(f"Testing integrated data collection for {duration} seconds...")
    print(f"Mode: {'Mock' if use_mock else 'Real Hardware'}")
    
    with create_main_collector(use_mock=use_mock) as collector:
        # 시스템 준비 대기
        print("Waiting for system to be ready...")
        if not collector.wait_for_system_ready(timeout=10.0):
            print("❌ System not ready, continuing anyway...")
        else:
            print("✅ System ready!")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            
            # 데이터 수집
            robot_data = collector.collect_synchronized_data()
            
            if robot_data:
                print(f"\n--- Iteration {iteration} ---")
                print(f"Timestamp: {robot_data.timestamp:.3f}")
                
                # 비디오 데이터 정보
                if robot_data.video_data:
                    print(f"Video streams: {list(robot_data.video_data.keys())}")
                    for key, frame in robot_data.video_data.items():
                        print(f"  {key}: {frame.shape}")
                
                # 상태 데이터 정보
                if robot_data.state_data:
                    print(f"State data: {list(robot_data.state_data.keys())}")
                    for key, state in robot_data.state_data.items():
                        if "joint" in key:
                            joint_str = f"[{', '.join([f'{x:.3f}' for x in state[:3]])}...]"
                            print(f"  {key}: {joint_str}")
                        else:
                            pos_str = f"[{', '.join([f'{x:.3f}' for x in state[:3]])}...]"
                            print(f"  {key}: {pos_str}")
                
                # 언어 데이터 정보
                if robot_data.language_data:
                    instruction = robot_data.language_data.get("annotation.language.instruction", "")
                    print(f"Language: '{instruction}'")
            
            # 상태 정보 출력 (5초마다)
            if iteration % 10 == 0:
                status = collector.get_system_status()
                rates = collector.get_data_rates()
                
                print(f"\n📊 System Status:")
                print(f"  Uptime: {status['main_collector']['uptime']:.1f}s")
                print(f"  System ready: {collector.is_system_ready()}")
                print(f"  Rates: {rates.get('frames_per_second', 0):.1f} fps, "
                      f"{rates.get('states_per_second', 0):.1f} sps, "
                      f"{rates.get('commands_per_second', 0):.3f} cps")
                
                # 개별 수집기 상태
                if 'vision' in status:
                    vision_info = status['vision']
                    if isinstance(vision_info, dict) and 'error' not in vision_info:
                        running_cameras = sum(1 for cam_status in vision_info.values() 
                                            if isinstance(cam_status, dict) and cam_status.get('is_running', False))
                        print(f"  Vision: {running_cameras} cameras running")
                
                if 'state' in status:
                    state_info = status['state']
                    if isinstance(state_info, dict) and 'error' not in state_info:
                        running_arms = sum(1 for arm_status in state_info.values()
                                         if isinstance(arm_status, dict) and arm_status.get('is_running', False))
                        print(f"  State: {running_arms} arms running")
            
            time.sleep(0.5)  # 2Hz로 데이터 수집
        
        # 최종 통계
        final_status = collector.get_system_status()
        final_rates = collector.get_data_rates()
        
        print(f"\n🎯 Final Statistics:")
        print(f"  Total frames: {final_status['main_collector']['total_frames']}")
        print(f"  Total states: {final_status['main_collector']['total_states']}")
        print(f"  Total commands: {final_status['main_collector']['total_commands']}")
        print(f"  Average rates: {final_rates}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 실행
    print("통합 데이터 수집 테스트")
    print("1. Mock 모드 (시뮬레이션)")
    print("2. Real 모드 (실제 하드웨어)")
    
    choice = input("선택하세요 (1/2): ").strip()
    use_mock = choice != "2"
    
    test_integrated_collection(duration=30.0, use_mock=use_mock)