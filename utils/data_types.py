"""
기본 데이터 타입 및 구조 정의
현재 하드웨어: Dual Piper arm, 2x D435 intel realsense, 1x Zed21 stereo camera
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
from enum import Enum


class HardwareType(Enum):
    """하드웨어 타입 정의"""
    PIPER_ARM_LEFT = "piper_arm_left"
    PIPER_ARM_RIGHT = "piper_arm_right"
    D435_LEFT = "d435_left"
    D435_RIGHT = "d435_right"
    ZED21_CENTER = "zed21_center"


class DataModalityType(Enum):
    """데이터 모달리티 타입"""
    VIDEO = "video"
    STATE = "state"
    ACTION = "action"
    LANGUAGE = "language"


@dataclass
class CameraConfig:
    """카메라 설정"""
    name: str
    width: int = 640
    height: int = 480
    fps: int = 30
    device_id: Optional[str] = None
    
    # GR00T 처리용 설정
    processed_width: int = 224
    processed_height: int = 224


@dataclass
class ArmConfig:
    """로봇 팔 설정"""
    name: str
    dof: int = 7  # Degrees of Freedom
    joint_limits: Optional[Dict[str, tuple]] = None
    effector_dof: int = 6  # position + orientation
    
    # 제어 관련
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    driver_min_freq: float = 5.0  # 하드웨어가 지원하는 최소 제어 주파수(Hz)


@dataclass
class SystemConfig:
    """전체 시스템 설정"""
    # 하드웨어 설정
    cameras: Dict[str, CameraConfig]
    arms: Dict[str, ArmConfig]
    
    # GR00T 모델 설정
    action_horizon: int = 16
    state_horizon: int = 1
    max_state_dim: int = 64
    max_action_dim: int = 32
    
    # 실행 설정
    control_frequency: float = 10.0  # Hz
    safety_timeout: float = 1.0  # seconds


@dataclass
class RobotControllerConfig:
    """로봇 제어기 설정 (실제 제어 주파수 등)"""
    arms: Dict[str, ArmConfig]
    control_frequency: float = 20.0  # 실제 사용할 제어 주파수(Hz)


class RobotData:
    """로봇 데이터 컨테이너"""
    
    def __init__(self):
        self.video_data: Dict[str, np.ndarray] = {}
        self.state_data: Dict[str, np.ndarray] = {}
        self.action_data: Dict[str, np.ndarray] = {}
        self.language_data: Dict[str, str] = {}
        self.timestamp: Optional[float] = None
    
    @property
    def gr00t_format(self) -> Dict[str, Any]:
        """GR00T 모델에 맞는 형식으로 변환"""
        return {
            "video": self.video_data,
            "state": self.state_data,
            "action": self.action_data,
            "language": self.language_data
        }


# 하드웨어별 데이터 키 정의 (GR00T 컨벤션 따름)
VIDEO_KEYS = [
    "video.left_arm_d435",    # 왼쪽 팔 D435 카메라
    "video.right_arm_d435",   # 오른쪽 팔 D435 카메라  
    "video.center_zed21",     # 중앙 Zed21 스테레오 카메라
]

STATE_KEYS = [
    "state.left_arm_joint_position",     # 왼쪽 팔 관절 위치 (7,)
    "state.right_arm_joint_position",    # 오른쪽 팔 관절 위치 (7,)
    "state.left_effector_position",      # 왼쪽 엔드이펙터 위치 (6,)
    "state.right_effector_position",     # 오른쪽 엔드이펙터 위치 (6,)
]

ACTION_KEYS = [
    "action.left_arm_joint_position",    # 왼쪽 팔 관절 목표 위치 (7,)
    "action.right_arm_joint_position",   # 오른쪽 팔 관절 목표 위치 (7,)
    "action.left_effector_position",     # 왼쪽 엔드이펙터 목표 위치 (6,)
    "action.right_effector_position",    # 오른쪽 엔드이펙터 목표 위치 (6,)
]

LANGUAGE_KEYS = [
    "annotation.language.instruction",    # 터미널 텍스트 명령
]


def create_default_system_config() -> SystemConfig:
    """기본 시스템 설정 생성"""
    
    cameras = {
        "left_arm_d435": CameraConfig(
            name="left_arm_d435",
            width=640, height=480, fps=30
        ),
        "right_arm_d435": CameraConfig(
            name="right_arm_d435", 
            width=640, height=480, fps=30
        ),
        "center_zed21": CameraConfig(
            name="center_zed21",
            width=1280, height=720, fps=30
        )
    }
    
    arms = {
        "left_arm": ArmConfig(
            name="left_arm",
            dof=7, effector_dof=6
        ),
        "right_arm": ArmConfig(
            name="right_arm", 
            dof=7, effector_dof=6
        )
    }
    
    return SystemConfig(
        cameras=cameras,
        arms=arms,
        action_horizon=16,
        state_horizon=1,
        control_frequency=10.0
    )


def validate_data_format(data: Dict[str, Any]) -> bool:
    """데이터 형식 검증"""
    
    # 비디오 데이터 검증
    for video_key in VIDEO_KEYS:
        if video_key in data:
            video_data = data[video_key]
            if not isinstance(video_data, (np.ndarray, torch.Tensor)):
                return False
            # 예상 형태: (T, H, W, C) 또는 (H, W, C)
            if len(video_data.shape) not in [3, 4]:
                return False
    
    # 상태 데이터 검증  
    for state_key in STATE_KEYS:
        if state_key in data:
            state_data = data[state_key]
            if not isinstance(state_data, (np.ndarray, torch.Tensor)):
                return False
            # 관절 위치: (7,), 엔드이펙터: (6,)
            expected_dim = 7 if "joint" in state_key else 6
            if state_data.shape[-1] != expected_dim:
                return False
    
    # 액션 데이터 검증
    for action_key in ACTION_KEYS:
        if action_key in data:
            action_data = data[action_key]
            if not isinstance(action_data, (np.ndarray, torch.Tensor)):
                return False
            # 관절 위치: (7,), 엔드이펙터: (6,)
            expected_dim = 7 if "joint" in action_key else 6
            if action_data.shape[-1] != expected_dim:
                return False
    
    return True


# 타입 힌트용 별칭
VideoData = Dict[str, np.ndarray]
StateData = Dict[str, np.ndarray] 
ActionData = Dict[str, np.ndarray]
LanguageData = Dict[str, str]
ModalityData = Dict[str, Any]
