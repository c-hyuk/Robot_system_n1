"""
Action Token 디코더
GR00T 모델의 액션 토큰을 실제 로봇 명령으로 변환
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import logging

from utils.data_types import ActionData
from config.hardware_config import get_hardware_config


@dataclass
class ActionDecodeConfig:
    """액션 디코딩 설정"""
    # 시간 관련
    action_horizon: int = 16
    execution_frequency: float = 10.0  # Hz
    
    # 스케일링
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    velocity_scale: float = 0.5
    
    # 안전 제한
    max_joint_velocity: float = 1.0  # rad/s
    max_effector_velocity: float = 0.5  # m/s
    max_acceleration: float = 2.0  # rad/s² or m/s²
    
    # 실행 모드
    execution_mode: str = "position"  # "position", "velocity", "trajectory"
    interpolation_method: str = "linear"  # "linear", "cubic", "spline"


class BaseActionDecoder(ABC):
    """액션 디코더 기본 클래스"""
    
    def __init__(self, config: ActionDecodeConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def decode_action(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """액션 토큰을 로봇 명령으로 디코딩"""
        pass


class DualPiperActionDecoder(BaseActionDecoder):
    """Dual Piper 로봇용 액션 디코더"""
    
    def __init__(self, config: Optional[ActionDecodeConfig] = None):
        config = config or ActionDecodeConfig()
        super().__init__(config)
        
        # 하드웨어 설정 로드
        self.hw_config = get_hardware_config()
        self.arm_configs = self.hw_config.system_config.arms
        
        # 액션 키 매핑
        self.action_mapping = self._create_action_mapping()
        
        # 이전 상태 (연속성 확인용)
        self.previous_action: Optional[Dict[str, np.ndarray]] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # 통계
        self.decode_count = 0
        self.total_decode_time = 0.0
    
    def _create_action_mapping(self) -> Dict[str, Dict[str, Any]]:
        """액션 키 매핑 생성"""
        mapping = {}
        
        for arm_name, arm_config in self.arm_configs.items():
            mapping[f"action.{arm_name}_joint_position"] = {
                'type': 'joint_position',
                'arm': arm_name,
                'dof': arm_config.dof,
                'limits': arm_config.joint_limits,
                'max_velocity': arm_config.max_velocity
            }
            
            mapping[f"action.{arm_name}_effector_position"] = {
                'type': 'effector_position',
                'arm': arm_name,
                'dof': arm_config.effector_dof,
                'max_velocity': self.config.max_effector_velocity
            }
        
        return mapping
    
    def decode_action(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """액션 토큰을 로봇 명령으로 디코딩"""
        import time
        start_time = time.time()
        
        try:
            # 1. 액션 토큰 검증
            if not self._validate_action_tokens(action_tokens):
                return self._generate_safe_action()
            
            # 2. 액션 형태에 따른 디코딩
            if self.config.execution_mode == "position":
                decoded = self._decode_position_commands(action_tokens)
            elif self.config.execution_mode == "velocity":
                decoded = self._decode_velocity_commands(action_tokens)
            elif self.config.execution_mode == "trajectory":
                decoded = self._decode_trajectory_commands(action_tokens)
            else:
                self.logger.error(f"Unknown execution mode: {self.config.execution_mode}")
                return self._generate_safe_action()
            
            # 3. 안전성 검사 및 제한 적용
            safe_decoded = self._apply_safety_limits(decoded)
            
            # 4. 연속성 검사
            smooth_decoded = self._ensure_continuity(safe_decoded)
            
            # 5. 최종 변환
            robot_commands = self._convert_to_robot_commands(smooth_decoded)
            
            # 통계 업데이트
            decode_time = time.time() - start_time
            self.total_decode_time += decode_time
            self.decode_count += 1
            
            # 히스토리 업데이트
            self.previous_action = smooth_decoded.copy()
            self._update_execution_history(robot_commands)
            
            return robot_commands
            
        except Exception as e:
            self.logger.error(f"Action decoding failed: {e}")
            return self._generate_safe_action()
    
    def _validate_action_tokens(self, action_tokens: Dict[str, np.ndarray]) -> bool:
        """액션 토큰 유효성 검증"""
        try:
            for key, value in action_tokens.items():
                if key not in self.action_mapping:
                    self.logger.warning(f"Unknown action key: {key}")
                    continue
                
                if not isinstance(value, np.ndarray):
                    self.logger.error(f"Action {key} is not numpy array: {type(value)}")
                    return False
                
                # 크기 검증
                expected_shape = self._get_expected_action_shape(key)
                if value.shape != expected_shape:
                    self.logger.error(f"Action {key} shape mismatch: {value.shape} vs {expected_shape}")
                    return False
                
                # 값 범위 검증
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    self.logger.error(f"Invalid values in action {key}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Action validation error: {e}")
            return False
    
    def _get_expected_action_shape(self, action_key: str) -> Tuple[int, ...]:
        """예상 액션 형태 반환"""
        if action_key not in self.action_mapping:
            return (0,)
        
        mapping = self.action_mapping[action_key]
        
        if self.config.execution_mode == "trajectory":
            # 궤적 모드: (time_steps, dof)
            return (self.config.action_horizon, mapping['dof'])
        else:
            # 위치/속도 모드: (dof,) 또는 (1, dof)
            base_shape = (mapping['dof'],)
            if len(action_key.split('.')) > 1:  # 배치 차원 있는 경우
                return (1, mapping['dof'])
            return base_shape
    
    def _decode_position_commands(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """위치 명령 디코딩"""
        decoded = {}
        
        for key, value in action_tokens.items():
            if key not in self.action_mapping:
                continue
            
            mapping = self.action_mapping[key]
            
            # 정규화된 액션을 실제 범위로 변환
            if mapping['type'] == 'joint_position':
                # 관절 위치: [-1, 1] → [joint_min, joint_max]
                decoded[key] = self._denormalize_joint_positions(value, mapping)
            elif mapping['type'] == 'effector_position':
                # 엔드이펙터 위치: [-1, 1] → 실제 좌표
                decoded[key] = self._denormalize_effector_positions(value, mapping)
        
        return decoded
    
    def _decode_velocity_commands(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """속도 명령 디코딩"""
        decoded = {}
        
        for key, value in action_tokens.items():
            if key not in self.action_mapping:
                continue
            
            mapping = self.action_mapping[key]
            
            # 속도 명령으로 변환
            if mapping['type'] == 'joint_position':
                # 위치 → 속도 변환 (이전 위치와의 차이)
                velocity = self._position_to_velocity(value, key)
                decoded[key.replace('position', 'velocity')] = velocity
            elif mapping['type'] == 'effector_position':
                velocity = self._position_to_velocity(value, key)
                decoded[key.replace('position', 'velocity')] = velocity
        
        return decoded
    
    def _decode_trajectory_commands(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """궤적 명령 디코딩"""
        decoded = {}
        
        for key, value in action_tokens.items():
            if key not in self.action_mapping:
                continue
            
    def _decode_trajectory_commands(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """궤적 명령 디코딩"""
        decoded = {}
        
        for key, value in action_tokens.items():
            if key not in self.action_mapping:
                continue
            
            # 전체 궤적을 보간하여 실행 가능한 명령으로 변환
            if value.ndim == 2:  # (time_steps, dof)
                # 현재 시점의 목표 위치 추출 (첫 번째 타임스텝)
                current_target = value[0]
                decoded[key] = current_target
            else:
                decoded[key] = value
        
        return decoded
    
    def _denormalize_joint_positions(self, normalized_positions: np.ndarray, mapping: Dict[str, Any]) -> np.ndarray:
        """정규화된 관절 위치를 실제 값으로 변환"""
        if mapping['limits'] is None:
            # 기본 제한: [-π, π]
            return normalized_positions * np.pi
        
        # 실제 관절 제한 사용
        denormalized = np.zeros_like(normalized_positions)
        limits = list(mapping['limits'].values())
        
        for i, (min_val, max_val) in enumerate(limits):
            if i < len(normalized_positions):
                # [-1, 1] → [min_val, max_val]
                range_val = max_val - min_val
                denormalized[i] = min_val + (normalized_positions[i] + 1) * range_val / 2
        
        return denormalized
    
    def _denormalize_effector_positions(self, normalized_positions: np.ndarray, mapping: Dict[str, Any]) -> np.ndarray:
        """정규화된 엔드이펙터 위치를 실제 값으로 변환"""
        denormalized = np.zeros_like(normalized_positions)
        
        # 위치 (x, y, z): [-1, 1] → [-2, 2] (미터)
        if len(normalized_positions) >= 3:
            denormalized[:3] = normalized_positions[:3] * 2.0
        
        # 회전 (roll, pitch, yaw): [-1, 1] → [-π, π] (라디안)
        if len(normalized_positions) >= 6:
            denormalized[3:6] = normalized_positions[3:6] * np.pi
        
        return denormalized
    
    def _position_to_velocity(self, current_position: np.ndarray, key: str) -> np.ndarray:
        """위치 명령을 속도 명령으로 변환"""
        if self.previous_action is None or key not in self.previous_action:
            # 이전 위치가 없으면 0 속도
            return np.zeros_like(current_position)
        
        # 속도 = (현재 위치 - 이전 위치) / 시간 간격
        dt = 1.0 / self.config.execution_frequency
        velocity = (current_position - self.previous_action[key]) / dt
        
        # 속도 제한 적용
        mapping = self.action_mapping[key]
        max_vel = mapping.get('max_velocity', self.config.max_joint_velocity)
        velocity = np.clip(velocity, -max_vel, max_vel)
        
        return velocity
    
    def _apply_safety_limits(self, decoded_actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """안전 제한 적용"""
        safe_actions = {}
        
        for key, value in decoded_actions.items():
            if key not in self.action_mapping:
                safe_actions[key] = value
                continue
            
            mapping = self.action_mapping[key]
            safe_value = value.copy()
            
            # 타입별 안전 제한
            if mapping['type'] == 'joint_position':
                # 관절 위치 제한
                if mapping['limits']:
                    limits = list(mapping['limits'].values())
                    for i, (min_val, max_val) in enumerate(limits):
                        if i < len(safe_value):
                            safe_value[i] = np.clip(safe_value[i], min_val, max_val)
                else:
                    safe_value = np.clip(safe_value, -np.pi, np.pi)
                    
            elif mapping['type'] == 'effector_position':
                # 엔드이펙터 위치 제한
                if len(safe_value) >= 3:
                    # 위치 제한: ±2m
                    safe_value[:3] = np.clip(safe_value[:3], -2.0, 2.0)
                if len(safe_value) >= 6:
                    # 회전 제한: ±180도
                    safe_value[3:6] = np.clip(safe_value[3:6], -np.pi, np.pi)
            
            safe_actions[key] = safe_value
        
        return safe_actions
    
    def _ensure_continuity(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """액션 연속성 보장"""
        if self.previous_action is None:
            return actions
        
        smooth_actions = {}
        
        for key, value in actions.items():
            if key not in self.previous_action:
                smooth_actions[key] = value
                continue
            
            prev_value = self.previous_action[key]
            
            # 급격한 변화 감지 및 제한
            diff = value - prev_value
            max_change = self._get_max_change_per_step(key)
            
            # 변화량 제한
            clipped_diff = np.clip(diff, -max_change, max_change)
            smooth_value = prev_value + clipped_diff
            
            smooth_actions[key] = smooth_value
        
        return smooth_actions
    
    def _get_max_change_per_step(self, key: str) -> np.ndarray:
        """스텝당 최대 변화량 계산"""
        if key not in self.action_mapping:
            return np.inf
        
        mapping = self.action_mapping[key]
        dt = 1.0 / self.config.execution_frequency
        
        if mapping['type'] == 'joint_position':
            # 관절 위치: 최대 속도 기반
            max_vel = mapping.get('max_velocity', self.config.max_joint_velocity)
            return np.full(mapping['dof'], max_vel * dt)
            
        elif mapping['type'] == 'effector_position':
            # 엔드이펙터: 위치와 회전 따로 계산
            max_changes = np.zeros(mapping['dof'])
            
            # 위치 (처음 3개)
            if mapping['dof'] >= 3:
                max_changes[:3] = self.config.max_effector_velocity * dt
            
            # 회전 (나머지)
            if mapping['dof'] >= 6:
                max_changes[3:6] = self.config.max_joint_velocity * dt  # 회전 속도
            
            return max_changes
        
        return np.inf
    
    def _convert_to_robot_commands(self, actions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """최종 로봇 명령으로 변환"""
        robot_commands = {
            'timestamp': time.time(),
            'execution_mode': self.config.execution_mode,
            'arms': {}
        }
        
        # 팔별로 그룹화
        for arm_name in self.arm_configs.keys():
            arm_commands = {}
            
            # 관절 위치
            joint_key = f"action.{arm_name}_joint_position"
            if joint_key in actions:
                arm_commands['joint_positions'] = actions[joint_key].tolist()
            
            # 엔드이펙터 위치
            effector_key = f"action.{arm_name}_effector_position"
            if effector_key in actions:
                effector_pos = actions[effector_key]
                arm_commands['effector_position'] = effector_pos[:3].tolist() if len(effector_pos) >= 3 else [0, 0, 0]
                arm_commands['effector_rotation'] = effector_pos[3:6].tolist() if len(effector_pos) >= 6 else [0, 0, 0]
            
            if arm_commands:
                robot_commands['arms'][arm_name] = arm_commands
        
        return robot_commands
    
    def _generate_safe_action(self) -> Dict[str, Any]:
        """안전한 기본 액션 생성"""
        safe_commands = {
            'timestamp': time.time(),
            'execution_mode': 'hold',
            'arms': {}
        }
        
        # 모든 팔을 현재 위치에 유지
        for arm_name in self.arm_configs.keys():
            safe_commands['arms'][arm_name] = {
                'joint_positions': [0.0] * self.arm_configs[arm_name].dof,
                'effector_position': [0.0, 0.0, 0.5],  # 기본 위치
                'effector_rotation': [0.0, 0.0, 0.0]   # 기본 자세
            }
        
        return safe_commands
    
    def _update_execution_history(self, robot_commands: Dict[str, Any]):
        """실행 히스토리 업데이트"""
        self.execution_history.append({
            'timestamp': robot_commands['timestamp'],
            'commands': robot_commands.copy()
        })
        
        # 히스토리 크기 제한
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_decode_stats(self) -> Dict[str, float]:
        """디코딩 통계 반환"""
        avg_decode_time = (self.total_decode_time / self.decode_count 
                          if self.decode_count > 0 else 0)
        
        return {
            'decode_count': self.decode_count,
            'avg_decode_time_ms': avg_decode_time * 1000,
            'total_decode_time': self.total_decode_time,
            'history_length': len(self.execution_history)
        }
    
    def get_last_execution(self) -> Optional[Dict[str, Any]]:
        """마지막 실행 명령 반환"""
        return self.execution_history[-1] if self.execution_history else None
    
    def reset_state(self):
        """디코더 상태 초기화"""
        self.previous_action = None
        self.execution_history.clear()
        self.logger.info("Action decoder state reset")


class ActionDecoderManager:
    """액션 디코더 관리자"""
    
    def __init__(self, embodiment_name: str = "dual_piper_arm", 
                 config: Optional[ActionDecodeConfig] = None):
        self.embodiment_name = embodiment_name
        self.config = config or ActionDecodeConfig()
        
        # Embodiment에 따른 디코더 선택
        if embodiment_name == "dual_piper_arm":
            self.decoder = DualPiperActionDecoder(self.config)
        else:
            raise ValueError(f"Unsupported embodiment: {embodiment_name}")
        
        self.logger = logging.getLogger("ActionDecoderManager")
    
    def decode_action(self, action_tokens: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """액션 토큰 디코딩"""
        return self.decoder.decode_action(action_tokens)
    
    def set_execution_mode(self, mode: str):
        """실행 모드 설정"""
        self.config.execution_mode = mode
        self.logger.info(f"Execution mode set to: {mode}")
    
    def get_decoder_stats(self) -> Dict[str, Any]:
        """디코더 통계 반환"""
        stats = self.decoder.get_decode_stats()
        stats['embodiment_name'] = self.embodiment_name
        stats['execution_mode'] = self.config.execution_mode
        return stats


# 편의용 함수들
def create_action_decoder(embodiment_name: str = "dual_piper_arm", 
                         execution_mode: str = "position") -> ActionDecoderManager:
    """액션 디코더 생성"""
    config = ActionDecodeConfig(execution_mode=execution_mode)
    return ActionDecoderManager(embodiment_name, config)


def test_action_decoder():
    """액션 디코더 테스트"""
    print("Testing action decoder...")
    
    # 디코더 생성
    decoder_manager = create_action_decoder("dual_piper_arm", "position")
    
    # Mock 액션 토큰 생성
    mock_action_tokens = {
        "action.left_arm_joint_position": np.random.uniform(-0.5, 0.5, 7).astype(np.float32),
        "action.right_arm_joint_position": np.random.uniform(-0.5, 0.5, 7).astype(np.float32),
        "action.left_arm_effector_position": np.random.uniform(-0.3, 0.3, 6).astype(np.float32),
        "action.right_arm_effector_position": np.random.uniform(-0.3, 0.3, 6).astype(np.float32),
    }
    
    print("Mock action tokens:")
    for key, value in mock_action_tokens.items():
        print(f"  {key}: {value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
    
    # 여러 번 디코딩 테스트 (연속성 확인)
    print(f"\nDecoding test (5 iterations):")
    
    for i in range(5):
        # 약간의 노이즈 추가로 연속적인 액션 시뮬레이션
        noisy_tokens = {}
        for key, value in mock_action_tokens.items():
            noise = np.random.normal(0, 0.01, value.shape).astype(np.float32)
            noisy_tokens[key] = value + noise
        
        # 디코딩
        robot_commands = decoder_manager.decode_action(noisy_tokens)
        
        print(f"\nIteration {i+1}:")
        print(f"  Timestamp: {robot_commands['timestamp']:.3f}")
        print(f"  Mode: {robot_commands['execution_mode']}")
        print(f"  Arms: {list(robot_commands['arms'].keys())}")
        
        for arm_name, arm_commands in robot_commands['arms'].items():
            if 'joint_positions' in arm_commands:
                joints = arm_commands['joint_positions']
                print(f"    {arm_name} joints: [{joints[0]:.3f}, {joints[1]:.3f}, ..., {joints[-1]:.3f}]")
            
            if 'effector_position' in arm_commands:
                pos = arm_commands['effector_position']
                rot = arm_commands['effector_rotation']
                print(f"    {arm_name} effector: pos={pos}, rot={rot}")
    
    # 통계 출력
    stats = decoder_manager.get_decoder_stats()
    print(f"\nDecoder statistics:")
    print(f"  Embodiment: {stats['embodiment_name']}")
    print(f"  Execution mode: {stats['execution_mode']}")
    print(f"  Decode count: {stats['decode_count']}")
    print(f"  Avg decode time: {stats['avg_decode_time_ms']:.2f}ms")
    print(f"  History length: {stats['history_length']}")
    
    # 다른 실행 모드 테스트
    print(f"\n🔄 Testing velocity mode:")
    decoder_manager.set_execution_mode("velocity")
    
    velocity_commands = decoder_manager.decode_action(mock_action_tokens)
    print(f"  Mode: {velocity_commands['execution_mode']}")
    
    print("✅ Action decoder test completed")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_action_decoder()