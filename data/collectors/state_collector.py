"""
로봇 상태 데이터 수집기
현재 하드웨어: Dual Piper arm (14 DOF total)
"""

import time
import threading
import queue
from typing import Dict, Optional, List, Any
from abc import ABC, abstractmethod
import numpy as np
import logging

from utils.data_types import ArmConfig, StateData
from config.hardware_config import get_hardware_config


class BaseRobotStateCollector(ABC):
    """로봇 상태 수집기 기본 클래스"""
    
    def __init__(self, arm_config: ArmConfig):
        self.config = arm_config
        self.is_running = False
        self.collection_thread = None
        self.data_queue = queue.Queue(maxsize=50)  # 상태 데이터는 비디오보다 작으므로 더 큰 버퍼
        self.last_state = None
        self.sample_count = 0
        self.start_time = None
        
        # 로깅 설정
        self.logger = logging.getLogger(f"RobotState_{arm_config.name}")
    
    @abstractmethod
    def _initialize_robot(self) -> bool:
        """로봇 연결 초기화 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _read_joint_positions(self) -> Optional[np.ndarray]:
        """관절 위치 읽기 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _read_effector_pose(self) -> Optional[np.ndarray]:
        """엔드이펙터 포즈 읽기 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _cleanup_robot(self) -> None:
        """로봇 연결 정리 (하위 클래스에서 구현)"""
        pass
    
    def start_collection(self) -> bool:
        """상태 수집 시작"""
        if self.is_running:
            self.logger.warning("State collection already running")
            return True
        
        if not self._initialize_robot():
            self.logger.error("Failed to initialize robot connection")
            return False
        
        self.is_running = True
        self.start_time = time.time()
        self.sample_count = 0
        
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info(f"Started state collection: {self.config.name}")
        return True
    
    def stop_collection(self) -> None:
        """상태 수집 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        self._cleanup_robot()
        self.logger.info(f"Stopped state collection: {self.config.name}")
    
    def _collection_loop(self) -> None:
        """상태 수집 루프 (별도 스레드에서 실행)"""
        # 10Hz로 상태 수집 (100ms 간격)
        target_interval = 0.1
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                state_data = self._collect_state()
                if state_data is not None:
                    self._process_and_queue_state(state_data)
                    self.sample_count += 1
                else:
                    self.logger.warning("Failed to collect robot state")
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(0.1)
            
            # 주파수 조절
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _collect_state(self) -> Optional[Dict[str, np.ndarray]]:
        """로봇 상태 수집"""
        joint_positions = self._read_joint_positions()
        effector_pose = self._read_effector_pose()
        
        if joint_positions is None or effector_pose is None:
            return None
        
        return {
            'joint_positions': joint_positions,
            'effector_pose': effector_pose
        }
    
    def _process_and_queue_state(self, state_data: Dict[str, np.ndarray]) -> None:
        """상태 데이터 처리 및 큐에 추가"""
        timestamp = time.time()
        
        # 데이터 검증
        if not self._validate_state_data(state_data):
            self.logger.warning("Invalid state data, skipping")
            return
        
        processed_state = {
            'joint_positions': state_data['joint_positions'],
            'effector_pose': state_data['effector_pose'],
            'timestamp': timestamp,
            'sample_id': self.sample_count
        }
        
        # 큐에 추가
        try:
            self.data_queue.put_nowait(processed_state)
            self.last_state = processed_state
        except queue.Full:
            try:
                self.data_queue.get_nowait()  # 오래된 데이터 제거
                self.data_queue.put_nowait(processed_state)
                self.last_state = processed_state
            except queue.Empty:
                pass
    
    def _validate_state_data(self, state_data: Dict[str, np.ndarray]) -> bool:
        """상태 데이터 검증"""
        try:
            joint_pos = state_data['joint_positions']
            effector_pose = state_data['effector_pose']
            
            # 크기 검증
            if joint_pos.shape[0] != self.config.dof:
                self.logger.error(f"Joint positions size mismatch: expected {self.config.dof}, got {joint_pos.shape[0]}")
                return False
            d
            if effector_pose.shape[0] != self.config.effector_dof:
                self.logger.error(f"Effector pose size mismatch: expected {self.config.effector_dof}, got {effector_pose.shape[0]}")
                return False
            
            # 값 범위 검증 (관절 제한)
            if self.config.joint_limits:
                for i, (joint_name, (min_val, max_val)) in enumerate(self.config.joint_limits.items()):
                    if i < len(joint_pos):
                        if not (min_val <= joint_pos[i] <= max_val):
                            self.logger.warning(f"Joint {joint_name} out of limits: {joint_pos[i]} not in [{min_val}, {max_val}]")
            
            # NaN/Inf 검증
            if np.any(np.isnan(joint_pos)) or np.any(np.isinf(joint_pos)):
                self.logger.error("Invalid joint position values (NaN/Inf)")
                return False
            
            if np.any(np.isnan(effector_pose)) or np.any(np.isinf(effector_pose)):
                self.logger.error("Invalid effector pose values (NaN/Inf)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating state data: {e}")
            return False
    
    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """최신 상태 반환"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return self.last_state
    
    def get_collection_rate(self) -> float:
        """현재 수집 주파수 계산"""
        if self.start_time is None or self.sample_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.sample_count / elapsed if elapsed > 0 else 0.0


class MockRobotStateCollector(BaseRobotStateCollector):
    """Mock 로봇 상태 수집기 (테스트용)"""
    
    def __init__(self, arm_config: ArmConfig):
        super().__init__(arm_config)
        self.time_offset = np.random.random() * 2 * np.pi  # 각 팔마다 다른 움직임
        
    def _initialize_robot(self) -> bool:
        """Mock 로봇 초기화"""
        self.logger.info("Mock robot initialized")
        return True
    
    def _read_joint_positions(self) -> Optional[np.ndarray]:
        """Mock 관절 위치 생성"""
        # 사인파로 부드러운 움직임 시뮬레이션
        t = time.time() + self.time_offset
        positions = np.zeros(self.config.dof)
        
        for i in range(self.config.dof):
            # 각 관절마다 다른 주파수와 진폭
            freq = 0.1 + i * 0.05  # 0.1Hz ~ 0.45Hz
            amplitude = 0.5 + i * 0.1  # 작은 움직임
            positions[i] = amplitude * np.sin(2 * np.pi * freq * t)
        
        return positions.astype(np.float32)
    
    def _read_effector_pose(self) -> Optional[np.ndarray]:
        """Mock 엔드이펙터 포즈 생성"""
        t = time.time() + self.time_offset
        
        # 위치 (x, y, z) + 회전 (roll, pitch, yaw)
        pose = np.zeros(self.config.effector_dof)
        
        # 작은 원형 움직임 시뮬레이션
        radius = 0.1
        freq = 0.2
        pose[0] = radius * np.cos(2 * np.pi * freq * t)  # x
        pose[1] = radius * np.sin(2 * np.pi * freq * t)  # y
        pose[2] = 0.5 + 0.05 * np.sin(2 * np.pi * freq * 2 * t)  # z (위아래 움직임)
        
        # 작은 회전
        pose[3] = 0.1 * np.sin(2 * np.pi * freq * 0.5 * t)  # roll
        pose[4] = 0.1 * np.cos(2 * np.pi * freq * 0.3 * t)  # pitch  
        pose[5] = 0.1 * np.sin(2 * np.pi * freq * 0.7 * t)  # yaw
        
        return pose.astype(np.float32)
    
    def _cleanup_robot(self) -> None:
        """Mock 로봇 정리"""
        self.logger.info("Mock robot cleaned up")


class PiperRobotStateCollector(BaseRobotStateCollector):
    """Piper 로봇 상태 수집기 (실제 하드웨어용)"""
    
    def __init__(self, arm_config: ArmConfig):
        super().__init__(arm_config)
        self.robot_connection = None
        # TODO: Piper 로봇 SDK import 및 초기화
    
    def _initialize_robot(self) -> bool:
        """Piper 로봇 연결 초기화"""
        try:
            # TODO: 실제 Piper 로봇 SDK 사용
            # 예시:
            # import piper_robot_sdk as piper
            # self.robot_connection = piper.connect(self.config.name)
            # return self.robot_connection.is_connected()
            
            self.logger.info("Piper robot connection initialized (TODO: implement actual SDK)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Piper robot: {e}")
            return False
    
    def _read_joint_positions(self) -> Optional[np.ndarray]:
        """Piper 로봇 관절 위치 읽기"""
        try:
            # TODO: 실제 Piper 로봇 SDK 사용
            # 예시:
            # joint_states = self.robot_connection.get_joint_positions()
            # return np.array(joint_states, dtype=np.float32)
            
            # 임시로 Mock 데이터 반환
            return np.zeros(self.config.dof, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to read joint positions: {e}")
            return None
    
    def _read_effector_pose(self) -> Optional[np.ndarray]:
        """Piper 로봇 엔드이펙터 포즈 읽기"""
        try:
            # TODO: 실제 Piper 로봇 SDK 사용
            # 예시:
            # pose = self.robot_connection.get_end_effector_pose()
            # return np.array(pose, dtype=np.float32)
            
            # 임시로 Mock 데이터 반환
            return np.zeros(self.config.effector_dof, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to read effector pose: {e}")
            return None
    
    def _cleanup_robot(self) -> None:
        """Piper 로봇 연결 정리"""
        try:
            # TODO: 실제 Piper 로봇 SDK 사용
            # if self.robot_connection:
            #     self.robot_connection.disconnect()
            
            self.logger.info("Piper robot connection cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up robot connection: {e}")


class RobotStateCollectorManager:
    """로봇 상태 수집 관리자"""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.collectors: Dict[str, BaseRobotStateCollector] = {}
        self.is_running = False
        
        # 하드웨어 설정 로드
        self.hw_config = get_hardware_config()
        
        # 로깅 설정
        self.logger = logging.getLogger("RobotStateCollectorManager")
        
        self._initialize_collectors()
    
    def _initialize_collectors(self) -> None:
        """수집기들 초기화"""
        arm_configs = self.hw_config.system_config.arms
        
        for arm_name, arm_config in arm_configs.items():
            if self.use_mock:
                collector = MockRobotStateCollector(arm_config)
            else:
                # 실제 하드웨어 타입에 따라 적절한 수집기 선택
                if "piper" in arm_name.lower() or "arm" in arm_name.lower():
                    collector = PiperRobotStateCollector(arm_config)
                else:
                    collector = MockRobotStateCollector(arm_config)
            
            self.collectors[arm_name] = collector
            self.logger.info(f"Initialized state collector for {arm_name}")
    
    def start_all_collectors(self) -> bool:
        """모든 상태 수집기 시작"""
        if self.is_running:
            self.logger.warning("State collectors already running")
            return True
        
        success_count = 0
        for name, collector in self.collectors.items():
            if collector.start_collection():
                success_count += 1
                self.logger.info(f"Started state collector: {name}")
            else:
                self.logger.error(f"Failed to start state collector: {name}")
        
        self.is_running = success_count > 0
        self.logger.info(f"Started {success_count}/{len(self.collectors)} state collectors")
        return self.is_running
    
    def stop_all_collectors(self) -> None:
        """모든 상태 수집기 중지"""
        for name, collector in self.collectors.items():
            collector.stop_collection()
            self.logger.info(f"Stopped state collector: {name}")
        
        self.is_running = False
    
    def get_all_states(self) -> StateData:
        """모든 로봇의 최신 상태 수집"""
        states = {}
        
        for arm_name, collector in self.collectors.items():
            state_data = collector.get_latest_state()
            if state_data:
                # GR00T 데이터 키 형식으로 변환
                joint_key = f"state.{arm_name}_joint_position"
                effector_key = f"state.{arm_name}_effector_position"
                
                states[joint_key] = state_data['joint_positions']
                states[effector_key] = state_data['effector_pose']
        
        return states
    
    def get_collector_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 수집기 상태 반환"""
        status = {}
        for name, collector in self.collectors.items():
            status[name] = {
                'is_running': collector.is_running,
                'collection_rate': collector.get_collection_rate(),
                'sample_count': collector.sample_count,
                'queue_size': collector.data_queue.qsize()
            }
        return status
    
    def get_current_joint_positions(self) -> Dict[str, np.ndarray]:
        """현재 모든 관절 위치 반환"""
        positions = {}
        states = self.get_all_states()
        
        for key, value in states.items():
            if "joint_position" in key:
                positions[key] = value
        
        return positions
    
    def get_current_effector_poses(self) -> Dict[str, np.ndarray]:
        """현재 모든 엔드이펙터 포즈 반환"""
        poses = {}
        states = self.get_all_states()
        
        for key, value in states.items():
            if "effector_position" in key:
                poses[key] = value
        
        return poses
    
    def is_all_arms_ready(self) -> bool:
        """모든 로봇 팔이 준비되었는지 확인"""
        if not self.is_running:
            return False
        
        for collector in self.collectors.values():
            if not collector.is_running or collector.last_state is None:
                return False
        
        return True
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_all_collectors()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_all_collectors()


# 편의용 함수들
def create_state_collector(use_mock: bool = False) -> RobotStateCollectorManager:
    """상태 수집기 생성"""
    return RobotStateCollectorManager(use_mock=use_mock)


def test_state_collection(duration: float = 5.0, use_mock: bool = True):
    """상태 수집 테스트"""
    print(f"Testing robot state collection for {duration} seconds...")
    
    with create_state_collector(use_mock=use_mock) as collector:
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            states = collector.get_all_states()
            
            if states:
                sample_count += 1
                print(f"Sample {sample_count}:")
                
                # 관절 위치 출력
                for key, value in states.items():
                    if "joint_position" in key:
                        print(f"  {key}: [{', '.join([f'{x:.3f}' for x in value])}]")
                    elif "effector_position" in key:
                        pos = value[:3]
                        rot = value[3:] if len(value) > 3 else []
                        pos_str = f"pos=[{', '.join([f'{x:.3f}' for x in pos])}]"
                        rot_str = f"rot=[{', '.join([f'{x:.3f}' for x in rot])}]" if len(rot) > 0 else ""
                        print(f"  {key}: {pos_str} {rot_str}")
                
                # 상태 출력 (1초마다)
                if sample_count % 10 == 0:
                    status = collector.get_collector_status()
                    for arm, info in status.items():
                        print(f"  {arm}: {info['collection_rate']:.1f} Hz, queue: {info['queue_size']}")
                    print(f"  All arms ready: {collector.is_all_arms_ready()}")
            
            time.sleep(0.1)
        
        print(f"Test completed. Collected {sample_count} samples.")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_state_collection(duration=10.0, use_mock=True)
