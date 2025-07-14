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
import argparse

from utils.data_types import ArmConfig, StateData
from config.hardware_config import get_hardware_config

# Piper SDK imports (실제 환경에서 사용)
try:
    from piper_sdk.interface.piper_interface_v2 import C_PiperInterface_V2
    from piper_sdk.interface.piper_interface import C_PiperInterface
    PIPER_SDK_AVAILABLE = True
except ImportError:
    PIPER_SDK_AVAILABLE = False
    print("Warning: Piper SDK not available, using mock data")


class BaseRobotStateCollector(ABC):
    """로봇 상태 수집기 기본 클래스"""
    
    def __init__(self, arm_config: ArmConfig, control_frequency: float = 10.0):
        self.config = arm_config
        self.control_frequency = control_frequency
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
        """데이터 수집 루프"""
        self.logger.info(f"State collection loop started for {self.config.name}")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # 관절 위치 읽기
                joint_positions = self._read_joint_positions()
                if joint_positions is None:
                    time.sleep(0.01)  # 실패 시 잠시 대기
                    continue
                
                # 엔드이펙터 포즈 읽기
                effector_pose = self._read_effector_pose()
                if effector_pose is None:
                    time.sleep(0.01)
                    continue
                
                # StateData 객체 생성
                current_time = time.time()
                state_data = {
                    "timestamp": current_time,
                    "joint_positions": joint_positions,
                    "effector_pose": effector_pose,
                    "arm_name": self.config.name
                }
                
                # 큐에 데이터 저장 (큐가 가득 찬 경우 오래된 데이터 제거)
                try:
                    self.data_queue.put_nowait(state_data)
                except queue.Full:
                    try:
                        self.data_queue.get_nowait()  # 오래된 데이터 제거
                        self.data_queue.put_nowait(state_data)
                    except queue.Empty:
                        pass
                
                self.last_state = state_data
                self.sample_count += 1
                
                # 타겟 주파수 유지 (기본 100Hz)
                target_interval = 1.0 / self.control_frequency
                elapsed = time.time() - start_time
                sleep_time = target_interval - elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(0.1)  # 에러 시 잠시 대기
    
    def get_latest_state(self) -> Optional[StateData]:
        """최신 상태 데이터 반환"""
        return self.last_state
    
    def get_all_queued_states(self) -> List[StateData]:
        """큐에 있는 모든 상태 데이터 반환"""
        states = []
        while not self.data_queue.empty():
            try:
                state = self.data_queue.get_nowait()
                states.append(state)
            except queue.Empty:
                break
        return states
    
    def get_status(self) -> Dict[str, Any]:
        """수집기 상태 반환"""
        return {
            'name': self.config.name,
            'is_running': self.is_running,
            'sample_count': self.sample_count,
            'queue_size': self.data_queue.qsize(),
            'last_update': self.last_state["timestamp"] if self.last_state else None,
            'sampling_rate': self.get_sampling_rate()
        }
    
    def get_sampling_rate(self) -> float:
        """현재 샘플링 레이트 반환"""
        if not self.start_time:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.sample_count / elapsed if elapsed > 0 else 0.0


class MockRobotStateCollector(BaseRobotStateCollector):
    """Mock 로봇 상태 수집기 (테스트용)"""
    
    def __init__(self, arm_config: ArmConfig, control_frequency: float = 10.0):
        super().__init__(arm_config, control_frequency)
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
    
    def __init__(self, arm_config: ArmConfig, control_frequency: float = 10.0):
        super().__init__(arm_config, control_frequency)
        self.robot_connection = None
        self.can_port = getattr(arm_config, 'can_port', 'can0')  # CAN 포트 (기본값: can0)
        self.use_v2_interface = getattr(arm_config, 'use_v2_interface', True)  # V2 인터페이스 사용 여부
        self.connection_timeout = 5.0  # 연결 타임아웃
        self.last_joint_data = None
        self.last_pose_data = None
        
    def _initialize_robot(self) -> bool:
        """Piper 로봇 연결 초기화"""
        if not PIPER_SDK_AVAILABLE:
            self.logger.warning("Piper SDK not available, using mock data")
            return self._initialize_mock_robot()
        
        try:
            # Piper SDK 버전에 따라 적절한 인터페이스 선택
            if self.use_v2_interface:
                self.robot_connection = C_PiperInterface_V2(
                    can_name=self.can_port,
                    judge_flag=True,
                    can_auto_init=True,
                    start_sdk_joint_limit=True,
                    start_sdk_gripper_limit=True
                )
            else:
                self.robot_connection = C_PiperInterface(
                    can_name=self.can_port,
                    judge_flag=True,
                    can_auto_init=True,
                    start_sdk_joint_limit=True,
                    start_sdk_gripper_limit=True
                )
            
            # 로봇 연결
            connect_result = self.robot_connection.ConnectPort(
                can_init=True,
                piper_init=True,
                start_thread=True
            )
            
            if not connect_result:
                self.logger.error("Failed to connect to Piper robot")
                return False
            
            # 연결 확인 (타임아웃 적용)
            start_time = time.time()
            while time.time() - start_time < self.connection_timeout:
                if self.robot_connection.get_connect_status():
                    break
                time.sleep(0.1)
            else:
                self.logger.error("Connection timeout")
                return False
            
            # 로봇이 정상적으로 응답하는지 확인
            if not self._wait_for_robot_ready():
                self.logger.error("Robot not ready")
                return False
            
            self.logger.info(f"Piper robot connected successfully on {self.can_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Piper robot: {e}")
            return False
    
    def _initialize_mock_robot(self) -> bool:
        """SDK가 없을 때 Mock 초기화"""
        self.logger.info("Using mock Piper robot data")
        return True
    
    def _wait_for_robot_ready(self, timeout: float = 3.0) -> bool:
        """로봇이 준비될 때까지 대기"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if self.robot_connection is not None and self.robot_connection.isOk():
                    # 첫 번째 관절 데이터를 받을 때까지 대기
                    joint_data = self.robot_connection.GetArmJointFeedBack()
                    if joint_data and joint_data.time_stamp > 0:
                        self.logger.info("Robot is ready and sending data")
                        return True
                time.sleep(0.1)
            except Exception as e:
                self.logger.warning(f"Waiting for robot ready: {e}")
                time.sleep(0.1)
        
        return False
    
    def _read_joint_positions(self) -> Optional[np.ndarray]:
        """Piper 로봇 관절 위치 읽기"""
        if not PIPER_SDK_AVAILABLE or not self.robot_connection:
            return self._read_mock_joint_positions()
        
        try:
            # Piper SDK에서 관절 데이터 읽기
            joint_data = self.robot_connection.GetArmJointFeedBack()
            
            if not joint_data or joint_data.time_stamp <= 0:
                return self.last_joint_data  # 이전 데이터 사용
            
            # 관절 각도 추출 (라디안으로 변환)
            joint_positions = np.array([
                joint_data.joint_state.joint_1 / 1000.0 * np.pi / 180.0,  # 밀리도 -> 라디안
                joint_data.joint_state.joint_2 / 1000.0 * np.pi / 180.0,
                joint_data.joint_state.joint_3 / 1000.0 * np.pi / 180.0,
                joint_data.joint_state.joint_4 / 1000.0 * np.pi / 180.0,
                joint_data.joint_state.joint_5 / 1000.0 * np.pi / 180.0,
                joint_data.joint_state.joint_6 / 1000.0 * np.pi / 180.0
            ], dtype=np.float32)
            
            # 그리퍼 데이터 추가 (있는 경우)
            if hasattr(joint_data.joint_state, 'gripper') and self.config.dof > 6:
                gripper_data = self.robot_connection.GetArmGripperFeedBack()
                if gripper_data and gripper_data.time_stamp > 0:
                    gripper_angle = gripper_data.gripper_state.grippers_angle / 1000000.0  # 마이크로미터 -> 미터
                    joint_positions = np.append(joint_positions, gripper_angle)
            
            # DOF에 맞게 패딩 또는 자르기
            if len(joint_positions) < self.config.dof:
                # 부족한 경우 0으로 패딩
                padded = np.zeros(self.config.dof, dtype=np.float32)
                padded[:len(joint_positions)] = joint_positions
                joint_positions = padded
            elif len(joint_positions) > self.config.dof:
                # 초과하는 경우 자르기
                joint_positions = joint_positions[:self.config.dof]
            
            self.last_joint_data = joint_positions
            return joint_positions
            
        except Exception as e:
            self.logger.error(f"Failed to read joint positions: {e}")
            return self.last_joint_data  # 이전 데이터 반환
    
    def _read_mock_joint_positions(self) -> Optional[np.ndarray]:
        """Mock 관절 위치 생성 (SDK 없을 때)"""
        t = time.time()
        positions = np.zeros(self.config.dof, dtype=np.float32)
        
        for i in range(self.config.dof):
            freq = 0.1 + i * 0.05
            amplitude = 0.3 + i * 0.1
            positions[i] = amplitude * np.sin(2 * np.pi * freq * t)
        
        return positions
    
    def _read_effector_pose(self) -> Optional[np.ndarray]:
        """Piper 로봇 엔드이펙터 포즈 읽기"""
        if not PIPER_SDK_AVAILABLE or not self.robot_connection:
            return self._read_mock_effector_pose()
        
        try:
            # Piper SDK에서 엔드이펙터 포즈 데이터 읽기
            pose_data = self.robot_connection.GetArmEndPoseFeedBack()
            
            if not pose_data or pose_data.time_stamp <= 0:
                return self.last_pose_data  # 이전 데이터 사용
            
            # 포즈 데이터 추출 (위치는 미터, 회전은 라디안)
            pose = np.array([
                pose_data.end_pose.X_axis / 1000.0,  # 밀리미터 -> 미터
                pose_data.end_pose.Y_axis / 1000.0,
                pose_data.end_pose.Z_axis / 1000.0,
                pose_data.end_pose.RX_axis / 1000.0 * np.pi / 180.0,  # 밀리도 -> 라디안
                pose_data.end_pose.RY_axis / 1000.0 * np.pi / 180.0,
                pose_data.end_pose.RZ_axis / 1000.0 * np.pi / 180.0
            ], dtype=np.float32)
            
            # effector_dof에 맞게 조정
            if len(pose) < self.config.effector_dof:
                padded = np.zeros(self.config.effector_dof, dtype=np.float32)
                padded[:len(pose)] = pose
                pose = padded
            elif len(pose) > self.config.effector_dof:
                pose = pose[:self.config.effector_dof]
            
            self.last_pose_data = pose
            return pose
            
        except Exception as e:
            self.logger.error(f"Failed to read effector pose: {e}")
            return self.last_pose_data  # 이전 데이터 반환
    
    def _read_mock_effector_pose(self) -> Optional[np.ndarray]:
        """Mock 엔드이펙터 포즈 생성 (SDK 없을 때)"""
        t = time.time()
        pose = np.zeros(self.config.effector_dof, dtype=np.float32)
        
        # 작은 원형 움직임 시뮬레이션
        radius = 0.05
        freq = 0.1
        pose[0] = 0.3 + radius * np.cos(2 * np.pi * freq * t)  # x
        pose[1] = radius * np.sin(2 * np.pi * freq * t)  # y
        pose[2] = 0.4 + 0.02 * np.sin(2 * np.pi * freq * 2 * t)  # z
        
        if self.config.effector_dof > 3:
            pose[3] = 0.05 * np.sin(2 * np.pi * freq * 0.5 * t)  # roll
            pose[4] = 0.05 * np.cos(2 * np.pi * freq * 0.3 * t)  # pitch
            pose[5] = 0.05 * np.sin(2 * np.pi * freq * 0.7 * t)  # yaw
        
        return pose
    
    def _cleanup_robot(self) -> None:
        """Piper 로봇 연결 정리"""
        try:
            if self.robot_connection and PIPER_SDK_AVAILABLE:
                self.robot_connection.DisconnectPort()
                self.logger.info("Piper robot connection disconnected")
            
            self.robot_connection = None
            
        except Exception as e:
            self.logger.error(f"Error cleaning up robot connection: {e}")
    
    def get_robot_status(self) -> Dict[str, Any]:
        """로봇 특화 상태 정보 반환"""
        status = self.get_status()
        
        if self.robot_connection and PIPER_SDK_AVAILABLE:
            try:
                status.update({
                    'can_port': self.can_port,
                    'connection_ok': self.robot_connection.isOk(),
                    'connection_status': self.robot_connection.get_connect_status(),
                    'interface_version': 'V2' if self.use_v2_interface else 'V1'
                })
                
                # 펌웨어 버전 정보 (가능한 경우)
                try:
                    firmware_version = self.robot_connection.GetPiperFirmwareVersion()
                    if isinstance(firmware_version, str):
                        status['firmware_version'] = firmware_version
                except:
                    pass
                    
            except Exception as e:
                status['robot_error'] = str(e)
        else:
            status.update({
                'can_port': self.can_port,
                'connection_ok': False,
                'mock_mode': True
            })
        
        return status


class RobotStateCollectorManager:
    """로봇 상태 수집 관리자"""
    
    def __init__(self, use_mock: bool = False, control_frequency: float = 10.0):
        self.use_mock = use_mock
        self.collectors: Dict[str, BaseRobotStateCollector] = {}
        self.is_running = False
        
        # 하드웨어 설정 로드
        self.hw_config = get_hardware_config()
        
        # 로깅 설정
        self.logger = logging.getLogger("RobotStateCollectorManager")
        
        self._initialize_collectors(control_frequency)
    
    def _initialize_collectors(self, control_frequency: float) -> None:
        """수집기들 초기화"""
        arm_configs = self.hw_config.system_config.arms
        
        for arm_name, arm_config in arm_configs.items():
            if self.use_mock:
                collector = MockRobotStateCollector(arm_config, control_frequency)
            else:
                # 실제 하드웨어 타입에 따라 적절한 수집기 선택
                if "piper" in arm_name.lower() or "arm" in arm_name.lower():
                    collector = PiperRobotStateCollector(arm_config, control_frequency)
                else:
                    collector = MockRobotStateCollector(arm_config, control_frequency)
            
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
                effector_key = f"state.{arm_name}_effector_pose"
                
                states[joint_key] = state_data["joint_positions"]
                states[effector_key] = state_data["effector_pose"]
        
        return states
    
    def get_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 반환"""
        status = {
            'manager_running': self.is_running,
            'total_collectors': len(self.collectors),
            'collectors': {}
        }
        
        for name, collector in self.collectors.items():
            status['collectors'][name] = collector.get_status()
            
            # Piper 로봇의 경우 추가 상태 정보
            if isinstance(collector, PiperRobotStateCollector):
                status['collectors'][name].update(collector.get_robot_status())
        
        return status
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_all_collectors()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_all_collectors()


def test_piper_state_collector():
    """Piper 상태 수집기 테스트"""
    print("Testing Piper Robot State Collector...")
    
    # Mock 모드로 테스트
    with RobotStateCollectorManager(use_mock=False) as manager:
        print(f"Manager started with {len(manager.collectors)} collectors")
        
        # 상태 정보 출력
        status = manager.get_status()
        print(f"System status: {status}")
        
        # 10초간 데이터 수집 테스트
        for i in range(50):  # 5초간 0.1초 간격
            states = manager.get_all_states()
            if states:
                print(f"Iteration {i+1}: Collected {len(states)} state values")
                for key, value in states.items():
                    if "joint" in key:
                        print(f"  {key}: [{', '.join([f'{x:.3f}' for x in value[:3]])}...]")
                    else:
                        print(f"  {key}: [{', '.join([f'{x:.3f}' for x in value[:3]])}...]")
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="로봇 상태 수집기")
    parser.add_argument(
        '--mock', action='store_true',
        help='Mock 모드로 실행 (실제 하드웨어 대신 시뮬레이션 데이터를 사용)'
    )
    parser.add_argument(
        '--duration', type=float, default=5.0,
        help='데이터 수집 총 시간 (초)'
    )
    parser.add_argument(
        '--interval', type=float, default=0.1,
        help='콘솔에 상태를 출력할 간격 (초)'
    )
    args = parser.parse_args()

    # 로그 포맷 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # 매니저 생성 및 시작
    manager = RobotStateCollectorManager(use_mock=args.mock)
    manager.start_all_collectors()
    start_time = time.time()

    try:
        while time.time() - start_time < args.duration:
            states = manager.get_all_states()
            elapsed = time.time() - start_time
            print(f"[{elapsed:.2f}s] 수집된 상태 항목: {len(states)}")
            for key, val in states.items():
                # 넘파이 배열의 앞 3개 값만 표시
                snippet = ", ".join(f"{x:.3f}" for x in val[:3])
                print(f"  {key}: [{snippet}, ...]")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("사용자에 의해 중단됨")
    finally:
        manager.stop_all_collectors()

if __name__ == "__main__":
    main()