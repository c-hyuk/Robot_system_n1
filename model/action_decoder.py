"""
Action Token 디코더 - End-Effector 기반
GR00T 모델의 EEF 액션 토큰을 실제 로봇 명령으로 변환
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import logging
import time
# from scipy.spatial.transform import Rotation

from utils.data_types import ActionData
from config.hardware_config import get_hardware_config


@dataclass
class EEFCommand:
    """End-Effector 명령"""
    timestamp: float
    position: np.ndarray  # (3,) meters
    rotation: np.ndarray  # (3,) or (6,) based on representation
    gripper: float  # [0, 1]
    

class BaseActionDecoder(ABC):
    """액션 디코더 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def decode_action(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """액션 토큰을 로봇 명령으로 디코딩"""
        pass


class DualPiperActionDecoder(BaseActionDecoder):
    """Dual Piper 로봇용 EEF 액션 디코더"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(config)
        
        # 액션 설정
        self.action_horizon = 16
        self.execution_frequency = 10.0  # Hz
        self.dt = 1.0 / self.execution_frequency
        
        # 작업 공간 제한 (미터 단위)
        self.workspace_limits = {
            'x': (-0.8, 0.8),
            'y': (-0.8, 0.8), 
            'z': (0.0, 1.2)
        }
        
        # 이전 상태 (스무딩용)
        self.previous_commands: Dict[str, Optional[EEFCommand]] = {
            'left': None,
            'right': None
        }
        
        # 이전 trajectory의 마지막 step 저장
        self.previous_trajectory_end: Optional[Dict[str, EEFCommand]] = None
        
        self.logger = logging.getLogger("DualPiperActionDecoder")
    
    def decode_action(self, action_tokens: Dict[str, np.ndarray]) -> List[Dict[str, EEFCommand]]:
        """
        GR00T 액션 토큰을 시간별 EEF 명령으로 변환
        Returns: 시간별 EEF 명령 리스트 (각 step은 dict of arm_name: EEFCommand)
        """
        try:
            # 1. step token이 arm별 key로 들어온 경우 지원
            required_keys = [
                'action.right_arm_eef_pos', 'action.right_arm_eef_rot', 'action.right_gripper_close',
                'action.left_arm_eef_pos', 'action.left_arm_eef_rot', 'action.left_gripper_close'
            ]
            if all(k in action_tokens for k in required_keys):
                left_action = np.concatenate([
                    np.array(action_tokens['action.left_arm_eef_pos']).flatten(),
                    np.array(action_tokens['action.left_arm_eef_rot']).flatten(),
                    np.array(action_tokens['action.left_gripper_close']).flatten()
                ])
                right_action = np.concatenate([
                    np.array(action_tokens['action.right_arm_eef_pos']).flatten(),
                    np.array(action_tokens['action.right_arm_eef_rot']).flatten(),
                    np.array(action_tokens['action.right_gripper_close']).flatten()
                ])
                current_time = time.time()
                cmd_dict = {
                    'timestamp': current_time,
                    'left': self._decode_single_arm_eef(left_action, 'left'),
                    'right': self._decode_single_arm_eef(right_action, 'right')
                }
                return [cmd_dict]
            # 2. 기존 통합 action 처리 (기존 코드 유지)
            if 'action_pred' in action_tokens:
                actions = action_tokens['action_pred']
            elif 'action' in action_tokens:
                actions = action_tokens['action']
            else:
                self.logger.error("No action found in token")
                return []
            
            # Shape 처리: (batch, horizon, dim) -> (horizon, dim)
            if actions.ndim == 3:
                actions = actions[0]  # Remove batch dimension
            elif actions.ndim == 2:
                # (horizon, dim) 형태
                pass
            else:
                self.logger.error(f"Unexpected action shape: {actions.shape}")
                return []
            
            # Action dimension 검증
            expected_dim = 20  # left(10) + right(10)
            if actions.shape[1] != expected_dim:
                self.logger.error(f"Action dimension mismatch: expected {expected_dim}, got {actions.shape[1]}")
                return []
            
            # 시간별 명령 생성
            commands = []
            current_time = time.time()
            
            # 16개 horizon 모두 처리
            horizon_steps = min(self.action_horizon, actions.shape[0])
            self.logger.info(f"Processing {horizon_steps} action steps from horizon {actions.shape[0]}")
            
            for t in range(horizon_steps):
                action_t = actions[t]
                
                # Left arm: first 10 dimensions
                left_action = action_t[:10]
                # Right arm: next 10 dimensions  
                right_action = action_t[10:20]
                
                cmd_dict = {
                    'timestamp': current_time + t * self.dt,
                    'left': self._decode_single_arm_eef(left_action, 'left'),
                    'right': self._decode_single_arm_eef(right_action, 'right')
                }
                commands.append(cmd_dict)
            
            # trajectory smoothing (step 간)
            for i in range(1, len(commands)):
                for arm in ['left', 'right']:
                    prev = commands[i-1][arm]
                    curr = commands[i][arm]
                    if prev is not None and curr is not None:
                        commands[i][arm] = self._blend_eef_command(prev, curr, alpha=0.5)
            
            # trajectory blending (이전 trajectory와 연결) - FIXED: prev_cmd 변수 정의
            if hasattr(self, 'previous_trajectory_end') and self.previous_trajectory_end is not None and len(commands) > 0:
                for arm in ['left', 'right']:
                    prev = self.previous_trajectory_end.get(arm)
                    curr = commands[0][arm]
                    if prev is not None and curr is not None:
                        commands[0][arm] = self._blend_eef_command(prev, curr, alpha=0.5)
            
            # 현재 trajectory의 마지막 step 저장
            if commands:
                self.previous_trajectory_end = commands[-1]
            
            return commands
            
        except Exception as e:
            self.logger.error(f"Action decoding failed: {e}")
            return []
    
    def _decode_single_arm_eef(self, arm_action: np.ndarray, arm_name: str) -> EEFCommand:
        """
        단일 팔의 EEF 액션 디코딩
        Args:
            arm_action: (10,) = position(3) + rotation(6) + gripper(1)
            arm_name: 'left' or 'right'
        """
        # 기본값 초기화
        position = np.zeros(3, dtype=np.float32)
        rotation = np.zeros(3, dtype=np.float32)
        gripper = 0.0
        timestamp = time.time()
        # 1. Position (첫 3개): [-1, 1] -> workspace limits
        if len(arm_action) >= 3:
            norm_pos = arm_action[:3]
            position = self._denormalize_position(norm_pos)
        # 2. Rotation (다음 6개): 6D rotation representation
        if len(arm_action) >= 9:
            rot_6d = arm_action[3:9]
            rotation = self._convert_6d_to_euler(rot_6d)
        # 3. Gripper (마지막 1개): [-1, 1] -> [0, 1]
        if len(arm_action) >= 10:
            gripper = (arm_action[9] + 1) / 2.0
            gripper = np.clip(gripper, 0, 1)
        cmd = EEFCommand(timestamp=timestamp, position=position, rotation=rotation, gripper=gripper)
        # 안전성 검사
        cmd = self._apply_safety_limits(cmd, arm_name)
        return cmd
    
    def _denormalize_position(self, norm_pos: np.ndarray) -> np.ndarray:
        """정규화된 위치를 실제 좌표로 변환"""
        denorm = np.zeros(3)
        
        # [-1, 1] -> workspace limits
        for i, (key, (min_val, max_val)) in enumerate(self.workspace_limits.items()):
            denorm[i] = min_val + (norm_pos[i] + 1) * (max_val - min_val) / 2
            
        return denorm
    
    def _convert_6d_to_euler(self, rot_6d: np.ndarray) -> np.ndarray:
        # 6D representation -> rotation matrix
        x = rot_6d[:3] / (np.linalg.norm(rot_6d[:3]) + 1e-8)
        y = rot_6d[3:6]
        y = y - np.dot(x, y) * x
        y = y / (np.linalg.norm(y) + 1e-8)
        z = np.cross(x, y)
        rot_matrix = np.column_stack([x, y, z])
        # numpy-only: rotation matrix → euler (xyz)
        def matrix_to_euler_xyz(R):
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(R[2,1], R[2,2])
                y = np.arctan2(-R[2,0], sy)
                z = np.arctan2(R[1,0], R[0,0])
            else:
                x = np.arctan2(-R[1,2], R[1,1])
                y = np.arctan2(-R[2,0], sy)
                z = 0
            return np.array([x, y, z], dtype=np.float32)
        euler = matrix_to_euler_xyz(rot_matrix)
        return euler
    
    def _apply_safety_limits(self, cmd: EEFCommand, arm_name: str) -> EEFCommand:
        """안전 제한 적용"""
        # 위치 제한
        for i, (key, (min_val, max_val)) in enumerate(self.workspace_limits.items()):
            cmd.position[i] = np.clip(cmd.position[i], min_val, max_val)
        
        # 회전 제한 (±180도)
        cmd.rotation = np.clip(cmd.rotation, -np.pi, np.pi)
        
        # 스무딩 (이전 명령과의 급격한 변화 방지)
        prev_cmd = self.previous_commands.get(arm_name, None)
        if prev_cmd is not None:
            # 위치 변화 제한 (10cm/step)
            pos_diff = cmd.position - prev_cmd.position
            max_pos_change = 0.1  # meters
            if np.linalg.norm(pos_diff) > max_pos_change:
                cmd.position = prev_cmd.position + pos_diff / np.linalg.norm(pos_diff) * max_pos_change
            
            # 회전 변화 제한 (30deg/step)
            rot_diff = cmd.rotation - prev_cmd.rotation
            max_rot_change = np.deg2rad(30)
            for i in range(3):
                if abs(rot_diff[i]) > max_rot_change:
                    cmd.rotation[i] = prev_cmd.rotation[i] + np.sign(rot_diff[i]) * max_rot_change
        self.previous_commands[arm_name] = cmd
        return cmd

    def _blend_eef_command(self, cmd1: EEFCommand, cmd2: EEFCommand, alpha=0.5) -> EEFCommand:
        import numpy as np
        return EEFCommand(
            timestamp=cmd2.timestamp,
            position=alpha * cmd2.position + (1-alpha) * cmd1.position,
            rotation=alpha * cmd2.rotation + (1-alpha) * cmd1.rotation,
            gripper=alpha * cmd2.gripper + (1-alpha) * cmd1.gripper
        )


# 편의 함수
def create_action_decoder(embodiment_name: str = "dual_piper_arm", 
                         config: Optional[Dict[str, Any]] = None) -> BaseActionDecoder:
    """액션 디코더 생성"""
    if embodiment_name == "dual_piper_arm":
        return DualPiperActionDecoder(config)
    else:
        raise ValueError(f"Unsupported embodiment: {embodiment_name}")
