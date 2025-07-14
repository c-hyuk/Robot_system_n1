"""
하드웨어별 세부 설정 및 초기화
현재 하드웨어: Dual Piper arm, 2x D435 intel realsense, 1x Zed21 stereo camera
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from utils.data_types import SystemConfig, CameraConfig, ArmConfig


class HardwareConfig:
    """하드웨어 설정 관리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.system_config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> SystemConfig:
        """설정 파일 로드 또는 기본 설정 생성"""
        if self.config_path and os.path.exists(self.config_path):
            return self._load_config_from_file()
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> SystemConfig:
        """현재 하드웨어에 맞는 기본 설정 생성"""
        
        # D435i 카메라 설정 (첫 번째 RealSense)
        d435i_left = CameraConfig(
            name="left_arm_d435i",
            width=640, height=480, fps=30,
            device_id="/dev/video2",  # RealSense D435i의 컬러 스트림
            processed_width=224, processed_height=224
        )
        
        # D435 카메라 설정 (두 번째 RealSense)
        d435_right = CameraConfig(
            name="right_arm_d435", 
            width=640, height=480, fps=30,
            device_id="/dev/video10",  # RealSense D435의 컬러 스트림
            processed_width=224, processed_height=224
        )
        
        # ZED 2i 스테레오 카메라 설정
        zed21_center = CameraConfig(
            name="center_zed21",
            width=1344, height=376, fps=15,  # ZED 2i의 실제 지원 해상도
            device_id="/dev/video14",  # ZED 2i의 메인 스트림
            processed_width=224, processed_height=224
        )
        
        # Piper 로봇 팔 설정
        piper_left = ArmConfig(
            name="left_arm",
            dof=7,
            effector_dof=6,
            joint_limits={
                "joint_1": (-3.14, 3.14),
                "joint_2": (-3.14, 3.14), 
                "joint_3": (-3.14, 3.14),
                "joint_4": (-3.14, 3.14),
                "joint_5": (-3.14, 3.14),
                "joint_6": (-3.14, 3.14),
                "joint_7": (-3.14, 3.14),
            },
            max_velocity=1.0,
            max_acceleration=2.0
        )
        
        piper_right = ArmConfig(
            name="right_arm",
            dof=7,
            effector_dof=6,
            joint_limits={
                "joint_1": (-3.14, 3.14),
                "joint_2": (-3.14, 3.14),
                "joint_3": (-3.14, 3.14), 
                "joint_4": (-3.14, 3.14),
                "joint_5": (-3.14, 3.14),
                "joint_6": (-3.14, 3.14),
                "joint_7": (-3.14, 3.14),
            },
            max_velocity=1.0,
            max_acceleration=2.0
        )
        
        return SystemConfig(
            cameras={
                "left_arm_d435i": d435i_left,
                "right_arm_d435": d435_right,
                "center_zed21": zed21_center
            },
            arms={
                "left_arm": piper_left,
                "right_arm": piper_right
            },
            action_horizon=16,      # GR00T 기본값
            state_horizon=1,        # GR00T 기본값
            max_state_dim=64,       # GR00T 기본값
            max_action_dim=32,      # GR00T 기본값
            control_frequency=10.0, # 10Hz 제어
            safety_timeout=1.0      # 1초 안전 타임아웃
        )
    
    def _load_config_from_file(self) -> SystemConfig:
        """파일에서 설정 로드 (향후 구현)"""
        # TODO: JSON/YAML 파일에서 설정 로드
        pass
    
    def get_camera_config(self, camera_name: str) -> CameraConfig:
        """특정 카메라 설정 반환"""
        if camera_name not in self.system_config.cameras:
            raise ValueError(f"Camera {camera_name} not found in configuration")
        return self.system_config.cameras[camera_name]
    
    def get_arm_config(self, arm_name: str) -> ArmConfig:
        """특정 로봇 팔 설정 반환"""
        if arm_name not in self.system_config.arms:
            raise ValueError(f"Arm {arm_name} not found in configuration")
        return self.system_config.arms[arm_name]
    
    def get_gr00t_modality_config(self) -> Dict[str, Any]:
        """GR00T 모델용 modality 설정 반환"""
        from gr00t.data.dataset import ModalityConfig
        
        # 비디오 모달리티
        video_modality = ModalityConfig(
            delta_indices=[0],  # state_horizon=1 이므로 현재 시점만
            modality_keys=[
                "video.left_arm_d435i",
                "video.right_arm_d435", 
                "video.center_zed21"
            ]
        )
        
        # 상태 모달리티
        state_modality = ModalityConfig(
            delta_indices=[0],  # state_horizon=1
            modality_keys=[
                "state.left_arm_joint_position",
                "state.right_arm_joint_position",
                "state.left_effector_position", 
                "state.right_effector_position"
            ]
        )
        
        # 액션 모달리티
        action_modality = ModalityConfig(
            delta_indices=list(range(self.system_config.action_horizon)),  # 16 steps
            modality_keys=[
                "action.left_arm_joint_position",
                "action.right_arm_joint_position",
                "action.left_effector_position",
                "action.right_effector_position"
            ]
        )
        
        # 언어 모달리티
        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.language.instruction"]
        )
        
        return {
            "video": video_modality,
            "state": state_modality, 
            "action": action_modality,
            "language": language_modality
        }
    
    def validate_hardware_connections(self) -> Dict[str, bool]:
        """하드웨어 연결 상태 검증"""
        status = {}
        
        # 카메라 연결 상태 검증
        for camera_name, camera_config in self.system_config.cameras.items():
            try:
                # 실제 카메라 장치 확인 로직 (예시)
                if camera_config.device_id and os.path.exists(camera_config.device_id):
                    status[camera_name] = True
                else:
                    status[camera_name] = False
            except Exception:
                status[camera_name] = False
        
        # 로봇 팔 연결 상태 검증 
        for arm_name in self.system_config.arms.keys():
            try:
                # 실제 로봇 팔 연결 확인 로직 (향후 구현)
                status[arm_name] = True  # 임시로 True
            except Exception:
                status[arm_name] = False
        
        return status
    
    def save_config(self, save_path: Optional[str] = None):
        """현재 설정을 파일로 저장"""
        if save_path is None:
            save_path = self.config_path or "config/system_config.json"
        
        # TODO: JSON으로 설정 저장 구현
        pass
    
    @property
    def is_hardware_ready(self) -> bool:
        """모든 하드웨어가 준비되었는지 확인"""
        status = self.validate_hardware_connections()
        return all(status.values())


# 전역 설정 인스턴스
_hardware_config = None

def get_hardware_config() -> HardwareConfig:
    """전역 하드웨어 설정 인스턴스 반환"""
    global _hardware_config
    if _hardware_config is None:
        _hardware_config = HardwareConfig()
    return _hardware_config


def initialize_hardware_config(config_path: Optional[str] = None) -> HardwareConfig:
    """하드웨어 설정 초기화"""
    global _hardware_config 
    _hardware_config = HardwareConfig(config_path)
    return _hardware_config


# 편의용 함수들
def get_camera_configs() -> Dict[str, CameraConfig]:
    """모든 카메라 설정 반환"""
    return get_hardware_config().system_config.cameras

def get_arm_configs() -> Dict[str, ArmConfig]:
    """모든 로봇 팔 설정 반환"""
    return get_hardware_config().system_config.arms

def get_control_frequency() -> float:
    """제어 주파수 반환"""
    return get_hardware_config().system_config.control_frequency
