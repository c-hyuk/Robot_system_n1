"""
Piper Hardware Bridge
실제 Piper SDK를 사용한 하드웨어 통신 인터페이스
"""

import time
import threading
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

try:
    from piper_sdk import C_PiperInterface_V2
    PIPER_SDK_AVAILABLE = True
except ImportError:
    PIPER_SDK_AVAILABLE = False
    print("⚠️ piper_sdk not installed. Using mock interface.")

if TYPE_CHECKING:
    from piper_sdk import C_PiperInterface_V2

from config.hardware_config import get_hardware_config


class PiperArmState(Enum):
    """Piper 팔 상태"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class PiperArmStatus:
    """Piper 팔 상태 정보"""
    arm_name: str
    state: PiperArmState
    joint_positions: List[float]
    joint_velocities: List[float]
    effector_position: List[float]
    gripper_position: float
    last_update_time: float
    error_count: int
    is_moving: bool
    
    # Piper 특정 상태들
    ctrl_mode: int = 0          # 제어 모드
    arm_status: int = 0         # 팔 상태
    motion_status: int = 0      # 모션 상태
    teach_status: int = 0       # 시교 상태


class PiperHardwareBridge:
    """Piper 로봇 팔 하드웨어 브릿지"""
    
    def __init__(
        self,
        left_can_port: str = "can0",
        right_can_port: str = "can1",
        auto_enable: bool = True,
        gripper_enabled: bool = True,
        left_piper: Optional[C_PiperInterface_V2] = None,
        right_piper: Optional[C_PiperInterface_V2] = None
    ):
        """
        Piper 하드웨어 브릿지 초기화
        
        Args:
            left_can_port: 왼쪽 팔 CAN 포트
            right_can_port: 오른쪽 팔 CAN 포트  
            auto_enable: 자동 enable 여부
            gripper_enabled: 그리퍼 사용 여부
            left_piper: 외부에서 주입된 PiperInterface 객체(왼쪽)
            right_piper: 외부에서 주입된 PiperInterface 객체(오른쪽)
        """
        self.left_can_port = left_can_port
        self.right_can_port = right_can_port
        self.auto_enable = auto_enable
        self.gripper_enabled = gripper_enabled
        
        # 하드웨어 설정
        self.hw_config = get_hardware_config()
        
        # Piper 인터페이스들 (DI 적용)
        self.left_piper: Optional[C_PiperInterface_V2] = left_piper
        self.right_piper: Optional[C_PiperInterface_V2] = right_piper
        
        # 팔 상태 관리
        self.arm_states: Dict[str, PiperArmStatus] = {
            "left_arm": PiperArmStatus(
                arm_name="left_arm",
                state=PiperArmState.DISCONNECTED,
                joint_positions=[0.0] * 7,
                joint_velocities=[0.0] * 7,
                effector_position=[0.0] * 6,
                gripper_position=0.0,
                last_update_time=0.0,
                error_count=0,
                is_moving=False
            ),
            "right_arm": PiperArmStatus(
                arm_name="right_arm", 
                state=PiperArmState.DISCONNECTED,
                joint_positions=[0.0] * 7,
                joint_velocities=[0.0] * 7,
                effector_position=[0.0] * 6,
                gripper_position=0.0,
                last_update_time=0.0,
                error_count=0,
                is_moving=False
            )
        }
        
        # 상태 모니터링 스레드
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.state_update_frequency = 100.0  # Hz (200Hz 읽기를 위한 5ms 간격)
        
        # 명령 실행 제한
        self.max_joint_velocity = 1.0  # rad/s
        self.max_effector_velocity = 0.5  # m/s
        
        # 로깅
        self.logger = logging.getLogger("PiperHardwareBridge")
        
        if not PIPER_SDK_AVAILABLE:
            self.logger.warning("Piper SDK not available, using mock interface")
    
    def connect(self) -> bool:
        """Piper 팔들에 연결"""
        if not PIPER_SDK_AVAILABLE:
            self.logger.warning("Using mock connection")
            return self._mock_connect()
        
        self.logger.info("Connecting to Piper arms...")
        
        try:
            # 왼쪽 팔 연결
            success_left = self._connect_arm("left_arm", self.left_can_port)
            
            # 오른쪽 팔 연결
            success_right = self._connect_arm("right_arm", self.right_can_port)
            
            if success_left and success_right:
                # 상태 모니터링 시작
                self._start_monitoring()
                
                self.logger.info("✅ Successfully connected to both Piper arms")
                return True
            else:
                self.logger.error("❌ Failed to connect to one or more Piper arms")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Connection error: {e}")
            return False
    
    def _connect_arm(self, arm_name: str, can_port: str) -> bool:
        """개별 팔 연결"""
        try:
            # DI: 이미 주입된 PiperInterface가 있으면 재사용
            if arm_name == "left_arm" and self.left_piper is not None:
                piper = self.left_piper
            elif arm_name == "right_arm" and self.right_piper is not None:
                piper = self.right_piper
            else:
                # Piper 인터페이스 생성
                piper = C_PiperInterface_V2(
                    can_name=can_port,
                    judge_flag=False,          # 외부 CAN 장치 사용
                    can_auto_init=True,        # CAN 자동 초기화
                    dh_is_offset=1,           # 최신 펌웨어용 DH 파라미터
                    start_sdk_joint_limit=True,   # SDK 관절 제한 활성화
                    start_sdk_gripper_limit=True  # SDK 그리퍼 제한 활성화
                )
                # 인터페이스 저장
                if arm_name == "left_arm":
                    self.left_piper = piper
                else:
                    self.right_piper = piper

            # 이미 연결된 경우 ConnectPort() 생략
            already_connected = False
            if hasattr(piper, 'get_connect_status'):
                try:
                    already_connected = piper.get_connect_status() is True
                except Exception:
                    already_connected = False

            if not already_connected:
                # CAN 포트 연결
                if not piper.ConnectPort():
                    self.logger.error(f"Failed to connect CAN port: {can_port}")
                    return False
                # 연결 대기
                time.sleep(0.1)
            else:
                self.logger.info(f"{arm_name} PiperInterface already connected, skipping ConnectPort()")
            
            # 팔 활성화 (auto_enable이 True인 경우)
            if self.auto_enable:
                # CAN 명령 제어 모드로 설정
                piper.MotionCtrl_2(0x01, 0x01, 50)  # CAN모드, 관절제어, 속도50
                time.sleep(0.1)
                
                # 팔 enable
                piper.EnableArm(arm_id=1)
                time.sleep(0.1)
                
                self.logger.info(f"✅ {arm_name} enabled on {can_port}")
            
            # 상태 업데이트
            self.arm_states[arm_name].state = PiperArmState.ENABLED if self.auto_enable else PiperArmState.CONNECTED
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect {arm_name}: {e}")
            return False
    
    def disconnect(self):
        """연결 해제"""
        self.logger.info("Disconnecting from Piper arms...")
        
        # 모니터링 중지
        self._stop_monitoring()
        
        # 팔 비활성화
        try:
            if self.left_piper:
                self.left_piper.DisableArm(arm_id=1)
                self.left_piper = None
                
            if self.right_piper:
                self.right_piper.DisableArm(arm_id=1)
                self.right_piper = None
                
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        
        # 상태 리셋
        for arm_state in self.arm_states.values():
            arm_state.state = PiperArmState.DISCONNECTED
        
        self.logger.info("Disconnected from Piper arms")
    
    def send_arm_command(self, arm_name: str, command: Dict[str, Any]) -> bool:
        """팔에 명령 전송"""
        if arm_name not in self.arm_states:
            self.logger.error(f"Unknown arm: {arm_name}")
            return False
        
        arm_state = self.arm_states[arm_name]
        if arm_state.state not in [PiperArmState.CONNECTED, PiperArmState.ENABLED]:
            self.logger.warning(f"Arm {arm_name} not ready for commands")
            return False
        
        try:
            # Piper 인터페이스 선택
            piper = self.left_piper if arm_name == "left_arm" else self.right_piper
            if not piper:
                return False
            
            # 명령 타입별 처리
            if 'joint_positions' in command:
                return self._send_joint_command(piper, command['joint_positions'])
            
            elif 'effector_position' in command and 'effector_rotation' in command:
                return self._send_cartesian_command(
                    piper, 
                    command['effector_position'], 
                    command['effector_rotation']
                )
            
            else:
                self.logger.warning(f"Unknown command format: {list(command.keys())}")
                return False
                
        except Exception as e:
            self.logger.error(f"Command execution error for {arm_name}: {e}")
            arm_state.error_count += 1
            return False
    
    def _send_joint_command(self, piper, joint_positions: List[float]) -> bool:
        """관절 위치 명령 전송"""
        try:
            # 관절 각도 제한 확인
            if len(joint_positions) != 7:
                self.logger.error(f"Expected 7 joint positions, got {len(joint_positions)}")
                return False
            
            # 라디안을 도로 변환 (Piper SDK는 도 단위 사용)
            joint_degrees = [np.degrees(pos) for pos in joint_positions]
            
            # 관절 명령 전송
            success = piper.JointMovJ(
                joint_degrees,          # 관절 각도 (도)
                speed_factor=50,        # 속도 팩터 (1-100)
                roughly_arrive=True     # 대략적 도달 허용
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Joint command error: {e}")
            return False
    
    def _send_cartesian_command(self, piper, position: List[float], rotation: List[float]) -> bool:
        """카르테시안 좌표 명령 전송"""
        try:
            # 위치 + 회전을 하나의 포즈로 결합
            if len(position) != 3 or len(rotation) != 3:
                self.logger.error("Position and rotation must be 3-element lists")
                return False
            
            # 위치는 미터를 밀리미터로 변환
            pose_mm = [
                position[0] * 1000,  # x (mm)
                position[1] * 1000,  # y (mm) 
                position[2] * 1000,  # z (mm)
                np.degrees(rotation[0]),  # roll (도)
                np.degrees(rotation[1]),  # pitch (도)
                np.degrees(rotation[2])   # yaw (도)
            ]
            
            # 카르테시안 명령 전송
            success = piper.PoseMovJ(
                pose_mm,                # 포즈 [x,y,z,rx,ry,rz]
                speed_factor=50,        # 속도 팩터
                roughly_arrive=True     # 대략적 도달 허용
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cartesian command error: {e}")
            return False
    
    def send_gripper_command(self, arm_name: str, gripper_position: float, speed: int = 50) -> bool:
        """그리퍼 명령 전송"""
        if not self.gripper_enabled:
            return True
        
        try:
            piper = self.left_piper if arm_name == "left_arm" else self.right_piper
            if not piper:
                return False
            
            # 그리퍼 위치: 0.0(닫힘) ~ 1.0(열림) → 0~1000 범위로 변환
            gripper_pos_raw = int(gripper_position * 1000)
            gripper_pos_raw = max(0, min(1000, gripper_pos_raw))
            
            success = piper.GripperCtrl(gripper_pos_raw, speed)
            return success
            
        except Exception as e:
            self.logger.error(f"Gripper command error for {arm_name}: {e}")
            return False
    
    def emergency_stop(self):
        """비상 정지"""
        self.logger.critical("🚨 Hardware Emergency Stop")
        
        try:
            # 모든 팔 비상 정지
            for arm_name, piper in [("left_arm", self.left_piper), ("right_arm", self.right_piper)]:
                if piper:
                    # 모션 정지
                    piper.MotionCtrl_2(0x00, 0x00, 0)  # 대기 모드로 설정
                    
                    # 상태 업데이트
                    self.arm_states[arm_name].state = PiperArmState.EMERGENCY_STOP
                    
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
    
    def _start_monitoring(self):
        """상태 모니터링 시작"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("📡 State monitoring started")
    
    def _stop_monitoring(self):
        """상태 모니터링 중지"""
        self.stop_monitoring.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """상태 모니터링 루프"""
        update_interval = 1.0 / self.state_update_frequency
        
        while not self.stop_monitoring.is_set():
            start_time = time.time()
            
            try:
                # 각 팔의 상태 업데이트
                for arm_name, piper in [("left_arm", self.left_piper), ("right_arm", self.right_piper)]:
                    if piper:
                        self._update_arm_state(arm_name, piper)
                        
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
            
            # 주기 조절
            elapsed = time.time() - start_time
            sleep_time = max(0, update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _update_arm_state(self, arm_name: str, piper):
        """개별 팔 상태 업데이트"""
        try:
            arm_state = self.arm_states[arm_name]
            
            # 관절 상태 읽기
            joint_msgs = piper.GetArmJointMsgs()
            if joint_msgs:
                # 관절 위치 (도 → 라디안)
                arm_state.joint_positions = [np.radians(pos) for pos in joint_msgs]
                
            # 팔 상태 읽기
            arm_status = piper.GetArmStatus()
            if arm_status:
                arm_state.ctrl_mode = getattr(arm_status, 'ctrl_mode', 0)
                arm_state.arm_status = getattr(arm_status, 'arm_status', 0)
                arm_state.motion_status = getattr(arm_status, 'motion_status', 0)
                arm_state.teach_status = getattr(arm_status, 'teach_status', 0)
                
                # 움직임 상태 업데이트
                arm_state.is_moving = (arm_state.motion_status == 1)  # 1: 미도달
            
            # 엔드이펙터 포즈 계산 (순운동학 필요 - 현재는 더미)
            # TODO: 실제 순운동학 계산 구현
            arm_state.effector_position = [0.0] * 6
            
            # 그리퍼 상태 (있는 경우)
            if self.gripper_enabled:
                gripper_msgs = piper.GetGripperMsgs()
                if gripper_msgs:
                    # 0~1000 → 0.0~1.0
                    arm_state.gripper_position = gripper_msgs / 1000.0
            
            # 타임스탬프 업데이트
            arm_state.last_update_time = time.time()
            
        except Exception as e:
            self.logger.error(f"State update error for {arm_name}: {e}")
            arm_state.error_count += 1
    
    def get_arm_status(self, arm_name: str) -> Optional[PiperArmStatus]:
        """개별 팔 상태 반환"""
        return self.arm_states.get(arm_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태 반환"""
        current_time = time.time()
        
        status = {
            'bridge_type': 'PiperHardwareBridge',
            'state': 'connected' if all(s.state != PiperArmState.DISCONNECTED for s in self.arm_states.values()) else 'disconnected',
            'timestamp': current_time,
            'arms': {}
        }
        
        # 각 팔 상태
        for arm_name, arm_state in self.arm_states.items():
            status['arms'][arm_name] = {
                'state': arm_state.state.value,
                'joint_positions': arm_state.joint_positions,
                'effector_position': arm_state.effector_position[:3],
                'effector_rotation': arm_state.effector_position[3:6] if len(arm_state.effector_position) >= 6 else [0,0,0],
                'gripper_position': arm_state.gripper_position,
                'is_moving': arm_state.is_moving,
                'error_count': arm_state.error_count,
                'last_update_ago': current_time - arm_state.last_update_time,
                'ctrl_mode': arm_state.ctrl_mode,
                'arm_status': arm_state.arm_status
            }
        
        # 간편한 액세스를 위한 평면화된 데이터
        status['left_arm_positions'] = self.arm_states['left_arm'].joint_positions
        status['right_arm_positions'] = self.arm_states['right_arm'].joint_positions
        
        return status
    
    def is_hardware_connected(self) -> bool:
        """하드웨어 연결 상태 확인"""
        return all(
            state.state in [PiperArmState.CONNECTED, PiperArmState.ENABLED] 
            for state in self.arm_states.values()
        )
    
    def is_system_ready(self) -> bool:
        """시스템 준비 상태 확인"""
        current_time = time.time()
        return (
            self.is_hardware_connected() and
            all(
                current_time - state.last_update_time < 1.0  # 1초 내 업데이트
                for state in self.arm_states.values()
            )
        )
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """하드웨어 정보 반환"""
        return {
            'bridge_type': 'PiperHardwareBridge',
            'sdk_available': PIPER_SDK_AVAILABLE,
            'left_can_port': self.left_can_port,
            'right_can_port': self.right_can_port,
            'auto_enable': self.auto_enable,
            'gripper_enabled': self.gripper_enabled,
            'state_update_frequency': self.state_update_frequency
        }
    
    def _mock_connect(self) -> bool:
        """Mock 연결 (SDK 없을 때)"""
        for arm_state in self.arm_states.values():
            arm_state.state = PiperArmState.ENABLED
            arm_state.last_update_time = time.time()
        
        self.logger.info("🔗 Mock connection established")
        return True
    
    def __enter__(self):
        """Context manager 진입"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.disconnect()


def test_piper_hardware_bridge():
    """Piper 하드웨어 브릿지 테스트"""
    print("🔧 Testing Piper Hardware Bridge...")
    
    try:
        # 하드웨어 브릿지 생성
        bridge = PiperHardwareBridge(
            left_can_port="can0",
            right_can_port="can1", 
            auto_enable=True,
            gripper_enabled=True
        )
        
        print("✅ Piper Hardware Bridge created")
        
        # Context manager로 연결 테스트
        with bridge:
            print("🔗 Connected to Piper arms")
            
            # 하드웨어 정보 출력
            hw_info = bridge.get_hardware_info()
            print(f"\nHardware Info:")
            for key, value in hw_info.items():
                print(f"  {key}: {value}")
            
            # 시스템 상태 확인
            print(f"\nSystem Ready: {bridge.is_system_ready()}")
            print(f"Hardware Connected: {bridge.is_hardware_connected()}")
            
            # 5초 동안 상태 모니터링
            for i in range(5):
                time.sleep(1)
                
                system_status = bridge.get_system_status()
                print(f"\nSecond {i+1}:")
                print(f"  System State: {system_status['state']}")
                
                for arm_name, arm_status in system_status['arms'].items():
                    print(f"  {arm_name}:")
                    print(f"    State: {arm_status['state']}")
                    print(f"    Joints: {[f'{j:.3f}' for j in arm_status['joint_positions'][:3]]}")
                    print(f"    Moving: {arm_status['is_moving']}")
                    print(f"    Errors: {arm_status['error_count']}")
            
            # 명령 전송 테스트
            print(f"\n🎮 Testing commands...")
            
            # 관절 명령 테스트
            test_joint_positions = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            left_cmd = {'joint_positions': test_joint_positions}
            success = bridge.send_arm_command("left_arm", left_cmd)
            print(f"  Left arm joint command: {'✅' if success else '❌'}")
            
            # 카르테시안 명령 테스트
            cartesian_cmd = {
                'effector_position': [0.3, 0.0, 0.4],  # x, y, z (m)
                'effector_rotation': [0.0, 0.0, 0.0]   # roll, pitch, yaw (rad)
            }
            success = bridge.send_arm_command("right_arm", cartesian_cmd)
            print(f"  Right arm cartesian command: {'✅' if success else '❌'}")
            
            # 그리퍼 명령 테스트
            success = bridge.send_gripper_command("left_arm", 0.5, speed=80)
            print(f"  Gripper command: {'✅' if success else '❌'}")
            
            time.sleep(2)
            
            # 최종 상태 확인
            final_status = bridge.get_system_status()
            print(f"\nFinal Status:")
            print(f"  System Ready: {bridge.is_system_ready()}")
            print(f"  Total Errors: {sum(arm['error_count'] for arm in final_status['arms'].values())}")
        
        print("✅ Piper Hardware Bridge test completed")
        
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
    test_piper_hardware_bridge()