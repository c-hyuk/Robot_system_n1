#!/usr/bin/env python3
# -*-coding:utf8-*-

"""
수정된 PiPER 로봇 상태 데이터 수집기
- Import 경로 수정 (정상 작동하는 코드와 동일)
- 초기화 방식 단순화
- 물리적으로 다른 CAN 포트 사용 (can0, can1)
"""

import time
import threading
import queue
from typing import Dict, Optional, List, Any
import numpy as np
import logging
import argparse
import os
import sys
# from scipy.spatial.transform import Rotation as R

# 정상 작동하는 첫 번째/두 번째 파일과 동일한 import 방식 사용
try:
    from piper_sdk import C_PiperInterface_V2
    PIPER_SDK_AVAILABLE = True
    print("✅ piper_sdk import 성공")
except ImportError as e:
    PIPER_SDK_AVAILABLE = False
    print(f"❌ piper_sdk import 실패: {e}")
    print("해결 방법: cd piper_py/piper_sdk && pip install -e .")


class PiperRobotStateCollector:
    """Piper 로봇 상태 수집기 (단순화된 안정 버전)"""
    
    def __init__(self, can_port: str = "can0", control_frequency: float = 10.0, piper_interface=None):
        self.can_port = can_port
        self.control_frequency = control_frequency
        self.robot_connection = piper_interface
        self.is_running = False
        self.collection_thread = None
        self.data_queue = queue.Queue(maxsize=50)
        self.last_state = None
        self.sample_count = 0
        self.start_time = None
        
        # 로깅 설정
        self.logger = logging.getLogger(f"PiperState_{can_port}")
        
    def emergency_stop_and_restore(self):
        """긴급정지 및 복구 - 첫 번째 파일과 동일한 방식"""
        arm_name = f"로봇 ({self.can_port})"
        self.logger.info(f"=== {arm_name} 긴급정지 및 복구 중... ===")
        if self.robot_connection is None:
            self.logger.error(f"❌ {arm_name} robot_connection is None, cannot perform emergency stop/restore.")
            return False
        try:
            # 긴급정지
            self.robot_connection.MotionCtrl_1(0x01, 0, 0x00)
            self.logger.info(f"✅ {arm_name} 긴급정지 완료")
            time.sleep(0.5)
            # 복구
            self.robot_connection.MotionCtrl_1(0x02, 0, 0x00)
            self.robot_connection.MotionCtrl_1(0x00, 0, 0x00)
            self.robot_connection.MotionCtrl_2(0x01, 0, 0, 0x00)  # StandBy 모드
            self.robot_connection.GripperCtrl(0, 0, 0x02, 0)
            time.sleep(1)
            self.robot_connection.MotionCtrl_2(0x01, 0, 0, 0x00)  # CAN 모드
            self.robot_connection.GripperCtrl(0, 0, 0x03, 0)
            time.sleep(0.05)
            self.robot_connection.EnableArm(7)  # 로봇 팔 활성화
            time.sleep(0.05)
            self.logger.info(f"✅ {arm_name} 정상 리셋 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ {arm_name} 긴급정지/복구 실패: {e}")
            return False
    
    def set_slave_mode(self):
        """Slave 모드 설정 - 첫 번째 파일과 동일한 방식"""
        arm_name = f"로봇 ({self.can_port})"
        self.logger.info(f"=== {arm_name} Slave 모드 설정 중... ===")
        if self.robot_connection is None:
            self.logger.error(f"❌ {arm_name} robot_connection is None, cannot set slave mode.")
            return False
        try:
            self.robot_connection.MasterSlaveConfig(0xFC, 0, 0, 0)  # Slave 모드
            self.logger.info(f"✅ {arm_name} Slave 모드 설정 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ {arm_name} Slave 모드 설정 실패: {e}")
            return False

    def enable_arm(self):
        """로봇 팔 활성화 - 첫 번째 파일과 동일한 방식"""
        arm_name = f"로봇 ({self.can_port})"
        self.logger.info(f"=== {arm_name} 활성화 중... ===")
        if self.robot_connection is None:
            self.logger.error(f"❌ {arm_name} robot_connection is None, cannot enable arm.")
            return False
        try:
            self.robot_connection.EnableArm(7)
            self.robot_connection.GripperCtrl(0, 1000, 0x01, 0)
            # 활성화 상태 확인 (간단한 버전)
            timeout = 5
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # 데이터 수신 확인
                    joint_data = self.robot_connection.GetArmJointMsgs()
                    if joint_data:
                        self.logger.info(f"✅ {arm_name} 활성화 완료")
                        return True
                except Exception:
                    pass
                self.robot_connection.EnableArm(7)
                time.sleep(1)
            self.logger.warning(f"⚠️ {arm_name} 활성화 타임아웃 (하지만 계속 진행)")
            return True  # 타임아웃이어도 진행
        except Exception as e:
            self.logger.error(f"❌ {arm_name} 활성화 실패: {e}")
            return False
    
    def _initialize_robot(self) -> bool:
        """로봇 연결 초기화 - Piper SDK 예제와 동일한 방식으로 초기화"""
        if not PIPER_SDK_AVAILABLE:
            self.logger.warning("Piper SDK not available, using mock data")
            return True
        try:
            # 1. Piper 인터페이스 객체가 없으면 생성
            if self.robot_connection is None:
                self.robot_connection = C_PiperInterface_V2(
                    can_name=self.can_port,
                    judge_flag=True,
                    can_auto_init=True,
                    start_sdk_joint_limit=True,
                    start_sdk_gripper_limit=True
                )
                self.robot_connection.ConnectPort(can_init=True, piper_init=True, start_thread=True)
                time.sleep(1)
            # 2. 긴급정지 및 복구
            self.logger.info(f"[Init] Emergency stop/restore for {self.can_port}")
            if hasattr(self.robot_connection, 'MotionCtrl_1'):
                self.robot_connection.MotionCtrl_1(0x01, 0, 0x00)
                time.sleep(0.5)
                self.robot_connection.MotionCtrl_1(0x02, 0, 0x00)
                self.robot_connection.MotionCtrl_1(0x00, 0, 0x00)
            if hasattr(self.robot_connection, 'MotionCtrl_2') and hasattr(self.robot_connection, 'GripperCtrl'):
                self.robot_connection.MotionCtrl_2(0x01, 0, 0, 0x00)
                self.robot_connection.GripperCtrl(0, 0, 0x02, 0)
                time.sleep(1)
                self.robot_connection.MotionCtrl_2(0x01, 0, 0, 0x00)
                self.robot_connection.GripperCtrl(0, 0, 0x03, 0)
                time.sleep(0.05)
            # 3. 슬레이브 모드
            self.logger.info(f"[Init] Slave mode for {self.can_port}")
            if hasattr(self.robot_connection, 'MasterSlaveConfig'):
                self.robot_connection.MasterSlaveConfig(0xFC, 0, 0, 0)
            # 4. 활성화
            self.logger.info(f"[Init] Enable arm for {self.can_port}")
            if hasattr(self.robot_connection, 'EnableArm') and hasattr(self.robot_connection, 'GetArmLowSpdInfoMsgs'):
                self.robot_connection.EnableArm(7)
                self.robot_connection.GripperCtrl(0, 1000, 0x01, 0)
                for _ in range(5):
                    enable_flag = all([
                        self.robot_connection.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status,
                        self.robot_connection.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status,
                        self.robot_connection.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status,
                        self.robot_connection.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status,
                        self.robot_connection.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status,
                        self.robot_connection.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
                    ])
                    if enable_flag:
                        self.logger.info(f"[Init] Arm enabled for {self.can_port}")
                        break
                    self.robot_connection.EnableArm(7)
                    time.sleep(1)
                else:
                    self.logger.warning(f"[Init] Arm enable timeout for {self.can_port}")
            self.logger.info(f"🎉 {self.can_port} 초기화 완료!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Piper robot: {e}")
            return False
    
    def _read_joint_positions(self) -> Optional[np.ndarray]:
        """관절 위치 읽기"""
        if not PIPER_SDK_AVAILABLE or not self.robot_connection:
            return self._generate_mock_joints()
        
        try:
            # 관절 데이터 읽기
            joint_data = self.robot_connection.GetArmJointMsgs()
            
            if not joint_data:
                return self._generate_mock_joints()
            
            # 관절 각도 추출 (도 -> 라디안 변환)
            factor = np.pi / 180.0 / 1000.0  # 밀리도 -> 라디안
            
            joint_positions = np.array([
                joint_data.joint_state.joint_1 * factor,
                joint_data.joint_state.joint_2 * factor,
                joint_data.joint_state.joint_3 * factor,
                joint_data.joint_state.joint_4 * factor,
                joint_data.joint_state.joint_5 * factor,
                joint_data.joint_state.joint_6 * factor
            ], dtype=np.float32)
            
            return joint_positions
            
        except Exception as e:
            self.logger.error(f"Failed to read joint positions: {e}")
            return self._generate_mock_joints()
    
    def _read_effector_pose(self) -> Optional[np.ndarray]:
        """엔드이펙터 포즈 읽기"""
        if not PIPER_SDK_AVAILABLE or not self.robot_connection:
            return self._generate_mock_pose()
        
        try:
            # 엔드이펙터 포즈 읽기
            pose_data = self.robot_connection.GetArmEndPoseMsgs()
            
            if not pose_data:
                return self._generate_mock_pose()
            
            # 포즈 데이터 추출 (미터 및 라디안으로 변환)
            pose = np.array([
                pose_data.end_pose.X_axis / 1000.0,  # 밀리미터 -> 미터
                pose_data.end_pose.Y_axis / 1000.0,
                pose_data.end_pose.Z_axis / 1000.0,
                pose_data.end_pose.RX_axis * np.pi / 180.0 / 1000.0,  # 밀리도 -> 라디안
                pose_data.end_pose.RY_axis * np.pi / 180.0 / 1000.0,
                pose_data.end_pose.RZ_axis * np.pi / 180.0 / 1000.0
            ], dtype=np.float32)
            
            return pose
            
        except Exception as e:
            self.logger.error(f"Failed to read effector pose: {e}")
            return self._generate_mock_pose()
    
    def _generate_mock_joints(self) -> np.ndarray:
        """Mock 관절 데이터 생성"""
        t = time.time()
        positions = np.zeros(6, dtype=np.float32)
        
        for i in range(6):
            freq = 0.1 + i * 0.05
            amplitude = 0.3
            positions[i] = amplitude * np.sin(2 * np.pi * freq * t)
        
        return positions
    
    def _generate_mock_pose(self) -> np.ndarray:
        """Mock 포즈 데이터 생성"""
        t = time.time()
        
        # 작은 원형 움직임 시뮬레이션
        radius = 0.05
        freq = 0.1
        pose = np.array([
            0.3 + radius * np.cos(2 * np.pi * freq * t),  # x
            radius * np.sin(2 * np.pi * freq * t),        # y
            0.4 + 0.02 * np.sin(2 * np.pi * freq * 2 * t), # z
            0.05 * np.sin(2 * np.pi * freq * 0.5 * t),    # roll
            0.05 * np.cos(2 * np.pi * freq * 0.3 * t),    # pitch
            0.05 * np.sin(2 * np.pi * freq * 0.7 * t)     # yaw
        ], dtype=np.float32)
        
        return pose
    
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
        
        self.logger.info(f"✅ Started state collection: {self.can_port}")
        return True
    
    def stop_collection(self) -> None:
        """상태 수집 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        self._cleanup_robot()
        self.logger.info(f"✅ Stopped state collection: {self.can_port}")
    
    def _collection_loop(self) -> None:
        """데이터 수집 루프"""
        self.logger.info(f"State collection loop started for {self.can_port}")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # 관절 위치 읽기
                joint_positions = self._read_joint_positions()
                if joint_positions is None:
                    time.sleep(0.01)
                    continue
                
                # 엔드이펙터 포즈 읽기
                effector_pose = self._read_effector_pose()
                if effector_pose is None:
                    time.sleep(0.01)
                    continue
                
                # 상태 데이터 생성
                current_time = time.time()
                state_data = {
                    "timestamp": current_time,
                    "joint_positions": joint_positions,
                    "effector_pose": effector_pose,
                    "can_port": self.can_port
                }
                
                # 큐에 데이터 저장
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
                
                # 타겟 주파수 유지
                target_interval = 1.0 / self.control_frequency
                elapsed = time.time() - start_time
                sleep_time = target_interval - elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(0.1)
    
    def _cleanup_robot(self) -> None:
        """로봇 연결 정리"""
        try:
            if self.robot_connection and PIPER_SDK_AVAILABLE:
                # 첫 번째 파일과 유사한 정리 방식
                try:
                    self.robot_connection.DisableArm(7)
                except:
                    pass
                self.logger.info("✅ Robot connection cleaned up")
            
            self.robot_connection = None
            
        except Exception as e:
            self.logger.error(f"Error cleaning up robot connection: {e}")
    
    def get_latest_state(self) -> Optional[dict]:
        """최신 상태 데이터 반환"""
        return self.last_state
    
    def get_all_queued_states(self) -> List[dict]:
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
            'can_port': self.can_port,
            'is_running': self.is_running,
            'sample_count': self.sample_count,
            'queue_size': self.data_queue.qsize(),
            'last_update': self.last_state["timestamp"] if self.last_state else None,
            'sampling_rate': self.get_sampling_rate(),
            'sdk_available': PIPER_SDK_AVAILABLE
        }
    
    def get_sampling_rate(self) -> float:
        """현재 샘플링 레이트 반환"""
        if not self.start_time:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.sample_count / elapsed if elapsed > 0 else 0.0


class RobotStateCollectorManager:
    """로봇 상태 수집 관리자"""
    
    def __init__(self, use_mock: bool = False, left_piper=None, right_piper=None):
        self.use_mock = use_mock
        self.collectors: Dict[str, PiperRobotStateCollector] = {}
        self.is_running = False
        # Dual Arm 설정 (공유 PiperInterface 사용 가능)
        self.left_collector = PiperRobotStateCollector("can0", 10.0, piper_interface=left_piper)
        self.right_collector = PiperRobotStateCollector("can1", 10.0, piper_interface=right_piper)
        self.collectors = {
            "left_arm": self.left_collector,
            "right_arm": self.right_collector
        }
        # 로깅 설정
        self.logger = logging.getLogger("RobotStateCollectorManager")
        
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
    
    def get_all_states(self) -> Dict[str, Any]:
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
    
    def get_status(self) -> Dict[str, Any]:
        """상태 수집기 상태 반환"""
        status = {}
        for name, collector in self.collectors.items():
            status[name] = {
                'is_running': collector.is_running,
                'sample_count': collector.sample_count,
                'queue_size': collector.data_queue.qsize()
            }
        return status


class DualArmStateCollectorManager:
    """양팔 상태 수집 관리자"""
    
    def __init__(self, control_frequency: float = 10.0):
        self.left_collector = PiperRobotStateCollector("can0", control_frequency)
        self.right_collector = PiperRobotStateCollector("can1", control_frequency)
        self.is_running = False
        self.logger = logging.getLogger("DualArmStateManager")
    
    def _split_pose_to_pos_quat(self, pose: np.ndarray) -> tuple:
        # pose: [x, y, z, rx, ry, rz] -> pos(3,), quat(4,)
        pos = pose[:3]
        quat = self._euler_to_quaternion(pose[3], pose[4], pose[5])
        return pos, quat

    def _euler_to_quaternion(self, rx, ry, rz):
        # numpy-only 오일러 → 쿼터니언 (w, x, y, z)
        cy = np.cos(rz * 0.5)
        sy = np.sin(rz * 0.5)
        cp = np.cos(ry * 0.5)
        sp = np.sin(ry * 0.5)
        cr = np.cos(rx * 0.5)
        sr = np.sin(rx * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z], dtype=np.float32)

    def start_all_collectors(self) -> bool:
        """모든 상태 수집기 시작"""
        if self.is_running:
            self.logger.warning("State collectors already running")
            return True
        left_ok = self.left_collector.start_collection()
        right_ok = self.right_collector.start_collection()
        self.is_running = left_ok or right_ok  # 하나라도 성공하면 실행
        if left_ok and right_ok:
            self.logger.info("✅ Both arms started successfully")
        elif left_ok:
            self.logger.warning("⚠️ Only left arm started")
        elif right_ok:
            self.logger.warning("⚠️ Only right arm started")
        else:
            self.logger.error("❌ Failed to start any arm")
        return self.is_running

    def stop_all_collectors(self) -> None:
        """모든 상태 수집기 중지"""
        self.left_collector.stop_collection()
        self.right_collector.stop_collection()
        self.is_running = False
        self.logger.info("✅ All collectors stopped")

    def get_all_states(self) -> Dict[str, Any]:
        """모든 로봇의 최신 상태 수집 (hardware_config.py 표준 포맷)"""
        states = {}
        # 왼팔 상태
        left_state = self.left_collector.get_latest_state()
        if left_state:
            pos, quat = self._split_pose_to_pos_quat(left_state["effector_pose"])
            states["state.left_arm_eef_pos"] = pos
            states["state.left_arm_eef_quat"] = quat
            # gripper 값: 실제 SDK에서 읽을 수 있으면 대체, 없으면 mock
            states["state.left_gripper_qpos"] = np.array([0.0], dtype=np.float32)
        # 오른팔 상태
        right_state = self.right_collector.get_latest_state()
        if right_state:
            pos, quat = self._split_pose_to_pos_quat(right_state["effector_pose"])
            states["state.right_arm_eef_pos"] = pos
            states["state.right_arm_eef_quat"] = quat
            states["state.right_gripper_qpos"] = np.array([0.0], dtype=np.float32)
        return states
    
    def get_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 반환"""
        return {
            'manager_running': self.is_running,
            'left_arm': self.left_collector.get_status(),
            'right_arm': self.right_collector.get_status()
        }


def test_state_collector():
    """상태 수집기 테스트"""
    print("=== Piper Robot State Collector Test ===")
    
    # 로그 포맷 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    manager = DualArmStateCollectorManager(control_frequency=10.0)
    
    try:
        if manager.start_all_collectors():
            print("✅ State collectors started")
            
            # 상태 정보 출력
            status = manager.get_status()
            print(f"System status: {status}")
            
            # 5초간 데이터 수집 테스트
            for i in range(50):  # 5초간 0.1초 간격
                states = manager.get_all_states()
                if states:
                    print(f"Iteration {i+1}: Collected {len(states)} state values")
                    for key, value in states.items():
                        if isinstance(value, np.ndarray):
                            snippet = ", ".join(f"{x:.3f}" for x in value[:3])
                            print(f"  {key}: [{snippet}, ...]")
                time.sleep(0.1)
        else:
            print("❌ Failed to start state collectors")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    finally:
        manager.stop_all_collectors()
        print("✅ Test completed")


def main():
    parser = argparse.ArgumentParser(description="로봇 상태 수집기")
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
    manager = DualArmStateCollectorManager()
    
    if not manager.start_all_collectors():
        print("❌ Failed to start state collectors")
        return 1
    
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
        print("✅ 프로그램 종료")
        return 0


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("❌ 반드시 root 권한으로 실행해야 합니다! (sudo python3 fixed_state_collector.py)")
        exit(1)
    
    exit(main())