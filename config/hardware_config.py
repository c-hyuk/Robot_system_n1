"""
하드웨어별 세부 설정 및 초기화
현재 하드웨어: Dual Piper arm, 2x L515 intel realsense, 1x Zed21 stereo camera
DualPiperDataConfig와 호환되도록 개선됨
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
        
        # DualPiperDataConfig와 일치하는 카메라 명명
        # RealSense 515 카메라 설정 (왼쪽 손목 뷰)
        left_wrist_camera = CameraConfig(
            name="left_wrist_view",  # DualPiperDataConfig와 일치
            width=640, height=480, fps=30,
            device_id="/dev/video2",  # OpenCV Success, 컬러 스트림 확인
            processed_width=224, processed_height=224
        )
        
        # RealSense 515 카메라 설정 (오른쪽 손목 뷰)
        right_wrist_camera = CameraConfig(
            name="right_wrist_view",  # DualPiperDataConfig와 일치
            width=640, height=480, fps=30,
            device_id="/dev/video8",  # OpenCV Success, 컬러 스트림 확인
            processed_width=224, processed_height=224
        )
        
        # ZED 2i 스테레오 카메라 설정 (정면 뷰)
        front_camera = CameraConfig(
            name="front_view",  # DualPiperDataConfig와 일치
            width=1344, height=376, fps=15,  # ZED 2i의 실제 지원 해상도
            device_id="/dev/video12",  # ZED 2i, OpenCV Success, 컬러 스트림 확인
            processed_width=224, processed_height=224
        )
        
        # Piper 로봇 팔 설정 (Piper SDK 공식)
        piper_left = ArmConfig(
            name="left_arm",
            dof=6,
            effector_dof=6,
            joint_limits={
                "joint_1": (-2.6179, 2.6179),
                "joint_2": (0, 3.14),
                "joint_3": (-2.967, 0),
                "joint_4": (-1.745, 1.745),
                "joint_5": (-1.22, 1.22),
                "joint_6": (-2.09439, 2.09439),
            },
            max_velocity=1.0,
            max_acceleration=2.0
        )
        
        piper_right = ArmConfig(
            name="right_arm",
            dof=6,
            effector_dof=6,
            joint_limits={
                "joint_1": (-2.6179, 2.6179),
                "joint_2": (0, 3.14),
                "joint_3": (-2.967, 0),
                "joint_4": (-1.745, 1.745),
                "joint_5": (-1.22, 1.22),
                "joint_6": (-2.09439, 2.09439),
            },
            max_velocity=1.0,
            max_acceleration=2.0
        )
        
        # 실제 action dimension 계산: (3+6+1)*2 = 20
        # max_action_dim=32로 설정하여 향후 확장성 고려
        return SystemConfig(
            cameras={
                "left_wrist_view": left_wrist_camera,    # DualPiperDataConfig 호환
                "right_wrist_view": right_wrist_camera,  # DualPiperDataConfig 호환
                "front_view": front_camera               # DualPiperDataConfig 호환
            },
            arms={
                "left_arm": piper_left,
                "right_arm": piper_right
            },
            action_horizon=16,      # GR00T 기본값
            state_horizon=1,        # GR00T 기본값
            max_state_dim=64,       # GR00T 기본값 (실제: (3+4+1)*2 = 16, 여유공간 포함)
            max_action_dim=32,      # GR00T 기본값 (실제: (3+6+1)*2 = 20, 여유공간 포함)
            control_frequency=10.0, # 10Hz 제어
            safety_timeout=1.0      # 1초 안전 타임아웃
        )
    
    def _load_config_from_file(self) -> SystemConfig:
        """파일에서 설정 로드 (향후 구현)"""
        # TODO: JSON/YAML 파일에서 설정 로드
        # 임시로 기본 config 반환 (linter 오류 방지)
        return self._create_default_config()
    
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
        """DualPiperDataConfig와 호환되는 GR00T 모델용 modality 설정 반환"""
        from gr00t.data.dataset import ModalityConfig
        
        # DualPiperDataConfig와 정확히 일치하는 키 사용
        video_modality = ModalityConfig(
            delta_indices=[0],  # state_horizon=1 이므로 현재 시점만
            modality_keys=[
                "video.right_wrist_view",  # DualPiperDataConfig와 일치
                "video.left_wrist_view",   # DualPiperDataConfig와 일치
                "video.front_view"         # DualPiperDataConfig와 일치
            ]
        )
        
        # DualPiperDataConfig의 state_keys와 일치
        state_modality = ModalityConfig(
            delta_indices=[0],  # state_horizon=1
            modality_keys=[
                "state.right_arm_eef_pos",      # (3,) - x, y, z in meters
                "state.right_arm_eef_quat",     # (4,) - quaternion (w, x, y, z)
                "state.right_gripper_qpos",     # (1,) - gripper position [0, 1]
                "state.left_arm_eef_pos",       # (3,)
                "state.left_arm_eef_quat",      # (4,)
                "state.left_gripper_qpos"       # (1,)
            ]
        )
        
        # DualPiperDataConfig의 action_keys와 일치
        action_modality = ModalityConfig(
            delta_indices=list(range(self.system_config.action_horizon)),  # 16 steps
            modality_keys=[
                "action.right_arm_eef_pos",     # (3,) - target position
                "action.right_arm_eef_rot",     # (6,) - 6D rotation representation
                "action.right_gripper_close",   # (1,) - binary command
                "action.left_arm_eef_pos",      # (3,)
                "action.left_arm_eef_rot",      # (6,)
                "action.left_gripper_close"     # (1,)
            ]
        )
        
        # DualPiperDataConfig의 language_keys와 일치
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
    
    def get_data_dimensions(self) -> Dict[str, int]:
        """실제 데이터 차원 정보 반환"""
        return {
            "state_dim": 16,  # (3+4+1)*2 = right/left (eef_pos + eef_quat + gripper)
            "action_dim": 20, # (3+6+1)*2 = right/left (eef_pos + eef_rot + gripper_close)
            "video_count": 3, # right_wrist, left_wrist, front
            "action_horizon": self.system_config.action_horizon,
            "state_horizon": self.system_config.state_horizon
        }
    
    def get_normalization_config(self) -> Dict[str, Dict[str, str]]:
        """DualPiperDataConfig와 일치하는 정규화 설정 반환"""
        return {
            "state_normalization_modes": {
                "state.right_arm_eef_pos": "min_max",
                "state.right_gripper_qpos": "min_max",
                "state.left_arm_eef_pos": "min_max",
                "state.left_gripper_qpos": "min_max",
            },
            "state_target_rotations": {
                "state.right_arm_eef_quat": "rotation_6d",
                "state.left_arm_eef_quat": "rotation_6d",
            },
            "action_normalization_modes": {
                "action.right_gripper_close": "binary",
                "action.left_gripper_close": "binary",
            }
        }
    
    def validate_hardware_connections(self) -> Dict[str, bool]:
        """하드웨어 연결 상태 검증"""
        status = {}
        
        # 카메라 연결 상태 검증
        for camera_name, camera_config in self.system_config.cameras.items():
            try:
                # 실제 카메라 장치 확인 로직
                if camera_config.device_id and os.path.exists(camera_config.device_id):
                    status[f"camera_{camera_name}"] = True
                else:
                    status[f"camera_{camera_name}"] = False
            except Exception:
                status[f"camera_{camera_name}"] = False
        
        # 로봇 팔 연결 상태 검증 
        for arm_name in self.system_config.arms.keys():
            try:
                # 실제 로봇 팔 연결 확인 로직 (향후 구현)
                # TODO: Piper SDK를 통한 실제 연결 확인
                status[f"arm_{arm_name}"] = True  # 임시로 True
            except Exception:
                status[f"arm_{arm_name}"] = False
        
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
    
    @property
    def data_config_compatible(self) -> bool:
        """DualPiperDataConfig와의 호환성 확인"""
        required_cameras = {"right_wrist_view", "left_wrist_view", "front_view"}
        available_cameras = set(self.system_config.cameras.keys())
        
        required_arms = {"left_arm", "right_arm"}
        available_arms = set(self.system_config.arms.keys())
        
        return (required_cameras.issubset(available_cameras) and 
                required_arms.issubset(available_arms))


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


# 편의용 함수들 (DualPiperDataConfig 호환)
def get_camera_configs() -> Dict[str, CameraConfig]:
    """모든 카메라 설정 반환"""
    return get_hardware_config().system_config.cameras

def get_arm_configs() -> Dict[str, ArmConfig]:
    """모든 로봇 팔 설정 반환"""
    return get_hardware_config().system_config.arms

def get_control_frequency() -> float:
    """제어 주파수 반환"""
    return get_hardware_config().system_config.control_frequency
def get_video_keys() -> list[str]:
    """DualPiperDataConfig 호환 비디오 키 반환"""
    return ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"]

def get_state_keys() -> list[str]:
    """DualPiperDataConfig 호환 상태 키 반환"""
    return [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat", 
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos"
    ]

def get_action_keys() -> list[str]:
    """DualPiperDataConfig 호환 액션 키 반환"""
    return [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos", 
        "action.left_arm_eef_rot",
        "action.left_gripper_close"
    ]
