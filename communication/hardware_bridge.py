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
    """Dual Piper 로봇과의 하드웨어 인터페이스 (EEF 제어)"""
    
    def __init__(self, 
                 left_can_port: str = "can0",
                 right_can_port: str = "can1", 
                 auto_enable: bool = True,
                 gripper_enabled: bool = True):
        
        self.left_can = left_can_port
        self.right_can = right_can_port
        self.auto_enable = auto_enable
        self.gripper_enabled = gripper_enabled
        self.logger = logging.getLogger("PiperHardwareBridge")
        
        # Piper 인터페이스
        self.arms = {}
        self.connected = False
        
        # 현재 상태
        self.current_state = {
            'left': {'ee_pose': None, 'gripper': 0},
            'right': {'ee_pose': None, 'gripper': 0}
        }
        
    def connect(self):
        """로봇 연결"""
        try:
            # SDK import는 여기서만
            import sys
            sys.path.append("../piper_py/piper_sdk")
            from piper_sdk import C_PiperInterface_V2
            
            # 왼팔 연결
            self.arms['left'] = C_PiperInterface_V2(
                can_name=self.left_can,
                judge_flag=False,
                can_auto_init=True,
                start_sdk_joint_limit=True,
                start_sdk_gripper_limit=True
            )
            self.arms['left'].ConnectPort()
            
            # 오른팔 연결
            self.arms['right'] = C_PiperInterface_V2(
                can_name=self.right_can,
                judge_flag=False,
                can_auto_init=True,
                start_sdk_joint_limit=True,
                start_sdk_gripper_limit=True
            )
            self.arms['right'].ConnectPort()
            
            time.sleep(1.0)
            
            if self.auto_enable:
                self._enable_arms()
                
            self.connected = True
            self.logger.info("Hardware connected successfully")
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.connected = False
    
    def disconnect(self):
        """로봇 연결 해제"""
        if self.connected:
            self._disable_arms()
            for arm in self.arms.values():
                arm.DisconnectPort()
            self.connected = False
    
    def _enable_arms(self):
        """로봇 활성화"""
        for name, arm in self.arms.items():
            # Enable
            arm.EnableArm()
            time.sleep(0.2)
            
            # 슬레이브 모드 설정
            arm.MasterSlaveConfig(0xFC, 0, 0, 0)
            
            # End-Effector 제어 모드 설정 (MOVE P)
            arm.MotionCtrl_2(
                ctrl_mode=0x01,      # CAN 지령 제어
                move_mode=0x00,      # MOVE P (Position/EEF mode)
                move_spd_rate_ctrl=50
            )
            time.sleep(0.1)
            
            self.logger.info(f"{name} arm enabled in EEF mode")
    
    def _disable_arms(self):
        """로봇 비활성화"""
        for name, arm in self.arms.items():
            # 홈 포지션으로 (joint 모드 임시 전환)
            arm.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=50)
            arm.JointCtrl(0, 0, 0, 0, 0, 0)
            time.sleep(0.5)
            arm.DisableArm()
    
    def enable_arm(self, arm_name: str):
        """공통 초기화: Enable, 슬레이브 모드"""
        arm = self.arms[arm_name]
        arm.EnableArm()
        time.sleep(0.2)
        arm.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.1)
        self.logger.info(f"{arm_name} arm enabled and in slave mode")

    def set_eef_control_mode(self, arm_name: str, speed: int = 50):
        """EEF(End-Effector) 제어 모드로 전환"""
        arm = self.arms[arm_name]
        arm.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=speed)
        time.sleep(0.1)
        self.logger.info(f"{arm_name} arm set to EEF control mode")

    def set_joint_control_mode(self, arm_name: str, speed: int = 50):
        """Jointspace(조인트) 제어 모드로 전환"""
        arm = self.arms[arm_name]
        arm.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=speed)
        time.sleep(0.1)
        self.logger.info(f"{arm_name} arm set to Joint control mode")

    def send_joint_command(self, arm_name: str, joint_angles):
        """조인트 각도 명령 전송 (joint_angles: 6개, radian)"""
        import numpy as np
        arm = self.arms[arm_name]
        # radian → 0.001deg 변환
        joint_vals = [int(np.rad2deg(a) * 1000) for a in joint_angles]
        arm.JointCtrl(*joint_vals)
        self.logger.info(f"{arm_name} arm joint command sent: {joint_vals}")
    
    def send_eef_command(self, arm_name: str, eef_cmd):
        """
        End-Effector 명령 전송
        
        Args:
            arm_name: 'left' or 'right'
            eef_cmd: EEFCommand 객체
        """
        if not self.connected or arm_name not in self.arms:
            return
            
        try:
            arm = self.arms[arm_name]
            
            # Position: meters -> 0.001mm
            X = int(eef_cmd.position[0] * 1000000)  # m -> 0.001mm
            Y = int(eef_cmd.position[1] * 1000000)
            Z = int(eef_cmd.position[2] * 1000000)
            
            # Rotation: radians -> 0.001degrees
            RX = int(np.rad2deg(eef_cmd.rotation[0]) * 1000)
            RY = int(np.rad2deg(eef_cmd.rotation[1]) * 1000)
            RZ = int(np.rad2deg(eef_cmd.rotation[2]) * 1000)
            
            # 값 검증 및 로깅
            self.logger.debug(f"EEF command for {arm_name}: pos=({X}, {Y}, {Z}), rot=({RX}, {RY}, {RZ})")
            
            # 안전 제한 검사
            if abs(X) > 1000000 or abs(Y) > 1000000 or abs(Z) > 1000000:  # 1m 제한
                self.logger.warning(f"Position out of safe limits for {arm_name}")
                return
            if abs(RX) > 180000 or abs(RY) > 180000 or abs(RZ) > 180000:  # ±180도 제한
                self.logger.warning(f"Rotation out of safe limits for {arm_name}")
                return
            
            # Send EEF command
            arm.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            
            # Gripper command
            if self.gripper_enabled and hasattr(eef_cmd, 'gripper'):
                gripper_angle = int(eef_cmd.gripper * 70000)  # [0,1] -> [0, 70000] (0.001deg)
                arm.GripperCtrl(
                    gripper_angle=gripper_angle,
                    gripper_effort=1000,  # 1N
                    gripper_code=0x01     # Enable
                )
            
            # Update state
            self.current_state[arm_name]['ee_pose'] = {
                'position': eef_cmd.position,
                'rotation': eef_cmd.rotation
            }
            self.current_state[arm_name]['gripper'] = eef_cmd.gripper
            
        except Exception as e:
            self.logger.error(f"EEF command failed: {e}")
    
    def send_arm_command(self, arm_name: str, cmd):
        """통합 명령 인터페이스 (gr00t_terminal 호환성)"""
        if hasattr(cmd, 'position') and hasattr(cmd, 'rotation'):
            # EEF command
            self.send_eef_command(arm_name, cmd)
        elif isinstance(cmd, dict) and 'ee_pose' in cmd:
            # Dictionary format
            from types import SimpleNamespace
            eef_cmd = SimpleNamespace(
                position=cmd['ee_pose']['position'],
                rotation=cmd['ee_pose']['rotation'],
                gripper=cmd.get('gripper', 0.5)
            )
            self.send_eef_command(arm_name, eef_cmd)
    
    def get_arm_state(self, arm_name: str) -> Optional[Dict]:
        """현재 팔 상태 조회"""
        if not self.connected or arm_name not in self.arms:
            return None
            
        try:
            arm = self.arms[arm_name]
            
            # End-Effector 상태
            ee_msg = arm.GetArmEndPoseMsgs()
            ee_pose = {
                'position': np.array([
                    ee_msg.end_pose.X_axis / 1000000.0,  # 0.001mm -> m
                    ee_msg.end_pose.Y_axis / 1000000.0,
                    ee_msg.end_pose.Z_axis / 1000000.0
                ]),
                'rotation': np.array([
                    np.deg2rad(ee_msg.end_pose.RX_axis / 1000.0),  # 0.001deg -> rad
                    np.deg2rad(ee_msg.end_pose.RY_axis / 1000.0),
                    np.deg2rad(ee_msg.end_pose.RZ_axis / 1000.0)
                ])
            }
            
            # Quaternion 변환 (GR00T state format)
            def euler_to_quaternion(rx, ry, rz):
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
            quat = euler_to_quaternion(*ee_pose['rotation'])  # [w, x, y, z]
            
            # Gripper 상태
            gripper_msg = arm.GetArmGripperMsgs()
            gripper_pos = gripper_msg.gripper_state.grippers_angle / 70000.0  # [0, 1]
            
            return {
                f'{arm_name}_arm_eef_pos': ee_pose['position'],
                f'{arm_name}_arm_eef_quat': quat,
                f'{arm_name}_gripper_qpos': gripper_pos,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"State query failed: {e}")
            return None
    
    def emergency_stop(self):
        """긴급 정지"""
        self.logger.warning("EMERGENCY STOP")
        for arm in self.arms.values():
            try:
                arm.EmergencyStop(0x01)
            except:
                pass