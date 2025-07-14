"""
GR00T 모델 인터페이스
기존 GR00T Policy를 우리 시스템에 맞게 래핑
"""

import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
import torch
import numpy as np

# GR00T 기본 import
from gr00t.model.policy import Gr00tPolicy, BasePolicy
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# 우리 시스템 import
from utils.data_types import RobotData
from data.integrated_pipeline import IntegratedDataPipeline


class DualPiperGR00TInterface:
    """Dual Piper 로봇용 GR00T 모델 인터페이스"""
    
    def __init__(
        self,
        model_path: str,
        embodiment_name: str = "dual_piper_arm",
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        use_mock_data: bool = False
    ):
        """
        GR00T 모델 인터페이스 초기화
        
        Args:
            model_path: GR00T 모델 체크포인트 경로 또는 HuggingFace Hub ID
            embodiment_name: 로봇 embodiment 이름 (DATA_CONFIG_MAP의 키)
            denoising_steps: Flow matching 디노이징 스텝 수
            device: 실행 장치
            use_mock_data: Mock 데이터 사용 여부
        """
        self.model_path = model_path
        self.embodiment_name = embodiment_name
        self.device = device
        self.use_mock_data = use_mock_data
        
        # 로깅 설정
        self.logger = logging.getLogger("DualPiperGR00TInterface")
        
        # 설정 및 Transform 로드
        self._load_config_and_transforms()
        
        # GR00T Policy 초기화
        self._initialize_gr00t_policy(denoising_steps)
        
        # 데이터 파이프라인 (선택적)
        self.data_pipeline: Optional[IntegratedDataPipeline] = None
        
        # 성능 통계
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        
        self.logger.info(f"GR00T interface initialized for {embodiment_name}")
    
    def _load_config_and_transforms(self):
        """설정 및 Transform 로드"""
        if self.embodiment_name not in DATA_CONFIG_MAP:
            available = list(DATA_CONFIG_MAP.keys())
            raise ValueError(f"Unknown embodiment: {self.embodiment_name}. Available: {available}")
        
        # GR00T 데이터 설정 로드
        self.data_config = DATA_CONFIG_MAP[self.embodiment_name]
        self.modality_config = self.data_config.modality_config()
        self.modality_transform = self.data_config.transform()
        
        self.logger.info(f"Loaded config for embodiment: {self.embodiment_name}")
    
    def _initialize_gr00t_policy(self, denoising_steps: Optional[int]):
        """GR00T Policy 초기화"""
        try:
            # Embodiment tag 설정
            if self.embodiment_name in ["dual_piper_arm"]:
                embodiment_tag = "bimanual_arm"  # GR00T의 기존 태그 중 가장 유사한 것
            else:
                embodiment_tag = self.embodiment_name
            
            # GR00T Policy 생성
            self.policy = Gr00tPolicy(
                model_path=self.model_path,
                embodiment_tag=embodiment_tag,
                modality_config=self.modality_config,
                modality_transform=self.modality_transform,
                denoising_steps=denoising_steps,
                device=self.device
            )
            
            self.logger.info("GR00T policy initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GR00T policy: {e}")
            raise
    
    def start_data_pipeline(self) -> bool:
        """데이터 파이프라인 시작"""
        if self.data_pipeline is not None:
            self.logger.warning("Data pipeline already started")
            return True
        
        try:
            self.data_pipeline = IntegratedDataPipeline(
                embodiment_name=self.embodiment_name,
                use_mock=self.use_mock_data
            )
            
            success = self.data_pipeline.start()
            if success:
                self.logger.info("Data pipeline started")
            else:
                self.logger.error("Failed to start data pipeline")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error starting data pipeline: {e}")
            return False
    
    def stop_data_pipeline(self):
        """데이터 파이프라인 중지"""
        if self.data_pipeline is not None:
            self.data_pipeline.stop()
            self.data_pipeline = None
            self.logger.info("Data pipeline stopped")
    
    def get_action_from_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        관찰 데이터로부터 액션 예측
        
        Args:
            observations: GR00T 형식의 관찰 데이터
            
        Returns:
            예측된 액션 딕셔너리
        """
        start_time = time.time()
        
        try:
            # GR00T Policy로 액션 예측
            action_dict = self.policy.get_action(observations)
            
            # 통계 업데이트
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.total_inferences += 1
            
            # 액션 후처리 (필요시)
            processed_action = self._postprocess_action(action_dict)
            
            return processed_action
            
        except Exception as e:
            self.logger.error(f"Action prediction failed: {e}")
            return self._get_safe_action()
    
    def get_action_from_pipeline(self) -> Optional[Dict[str, Any]]:
        """
        데이터 파이프라인으로부터 액션 예측
        
        Returns:
            예측된 액션 딕셔너리 또는 None
        """
        if self.data_pipeline is None:
            self.logger.error("Data pipeline not started")
            return None
        
        # 파이프라인에서 GR00T 입력 데이터 수집
        gr00t_input = self.data_pipeline.get_gr00t_input()
        if gr00t_input is None:
            self.logger.warning("No data from pipeline")
            return None
        
        # 액션 예측
        return self.get_action_from_observations(gr00t_input)
    
    def _postprocess_action(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """액션 후처리"""
        processed = {}
        
        for key, value in action_dict.items():
            if isinstance(value, torch.Tensor):
                # Tensor를 numpy로 변환
                processed[key] = value.detach().cpu().numpy()
            else:
                processed[key] = value
        
        return processed
    
    def _get_safe_action(self) -> Dict[str, Any]:
        """안전한 기본 액션 (에러 시 사용)"""
        # 모든 관절과 엔드이펙터를 현재 위치로 유지
        safe_action = {}
        
        for action_key in self.data_config.action_keys:
            if "joint_position" in action_key:
                # 7 DOF 관절
                safe_action[action_key] = np.zeros(7, dtype=np.float32)
            elif "effector_position" in action_key:
                # 6 DOF 엔드이펙터 (위치 + 자세)
                safe_action[action_key] = np.zeros(6, dtype=np.float32)
        
        return safe_action
    
    def set_training_mode(self, training: bool = True):
        """훈련/평가 모드 설정"""
        if hasattr(self.policy, 'model'):
            if training:
                self.policy.model.train()
            else:
                self.policy.model.eval()
        
        # Transform도 모드 설정
        if training:
            self.modality_transform.train()
        else:
            self.modality_transform.eval()
        
        self.logger.info(f"Set to {'training' if training else 'evaluation'} mode")
    
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """모달리티 설정 반환"""
        return self.policy.get_modality_config()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        elapsed = time.time() - self.start_time
        avg_inference_time = (self.total_inference_time / self.total_inferences 
                             if self.total_inferences > 0 else 0)
        inferences_per_second = self.total_inferences / elapsed if elapsed > 0 else 0
        
        return {
            'total_inferences': self.total_inferences,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'inferences_per_second': inferences_per_second,
            'uptime_seconds': elapsed
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'model_path': str(self.model_path),
            'embodiment_name': self.embodiment_name,
            'device': str(self.device),
            'denoising_steps': getattr(self.policy, 'denoising_steps', None)
        }
        
        if hasattr(self.policy, 'model'):
            model = self.policy.model
            info.update({
                'model_type': type(model).__name__,
                'action_horizon': getattr(model, 'action_horizon', None),
                'state_horizon': getattr(model, 'state_horizon', None)
            })
        
        return info
    
    def validate_observations(self, observations: Dict[str, Any]) -> bool:
        """관찰 데이터 유효성 검증"""
        try:
            # 필수 키 확인
            expected_keys = set()
            for modality_key in self.modality_config.keys():
                if modality_key in observations:
                    expected_keys.add(modality_key)
            
            if len(expected_keys) == 0:
                self.logger.error("No valid modality keys found in observations")
                return False
            
            # 데이터 타입 및 크기 확인
            for key, value in observations.items():
                if not isinstance(value, (np.ndarray, torch.Tensor)):
                    self.logger.error(f"Invalid data type for {key}: {type(value)}")
                    return False
                
                if hasattr(value, 'shape') and len(value.shape) == 0:
                    self.logger.error(f"Scalar value not allowed for {key}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_data_pipeline()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_data_pipeline()


# 편의용 함수들
def create_dual_piper_interface(
    model_path: str,
    denoising_steps: Optional[int] = None,
    device: str = "cuda",
    use_mock_data: bool = False
) -> DualPiperGR00TInterface:
    """Dual Piper GR00T 인터페이스 생성"""
    return DualPiperGR00TInterface(
        model_path=model_path,
        embodiment_name="dual_piper_arm",
        denoising_steps=denoising_steps,
        device=device,
        use_mock_data=use_mock_data
    )


def test_gr00t_interface():
    """GR00T 인터페이스 테스트"""
    print("Testing GR00T interface...")
    
    # Mock 모델 경로 (실제 테스트 시 실제 경로 사용)
    model_path = "nvidia/gr00t-1.5b"  # 예시 경로
    
    try:
        # 인터페이스 생성 (Mock 모드)
        interface = DualPiperGR00TInterface(
            model_path=model_path,
            embodiment_name="dual_piper_arm",  # 실제로는 존재하지 않을 수 있음
            use_mock_data=True
        )
        
        print("✅ Interface created successfully")
        
        # 모델 정보 출력
        model_info = interface.get_model_info()
        print(f"Model info: {model_info}")
        
        # 모달리티 설정 확인
        modality_config = interface.get_modality_config()
        print(f"Modality config keys: {list(modality_config.keys())}")
        
        # Mock 관찰 데이터 생성
        mock_observations = {
            'video': np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8),
            'state': np.random.uniform(-1, 1, (1, 64), dtype=np.float32),
            'language': "Pick up the red cube"
        }
        
        print("\n🔍 Testing action prediction...")
        
        # 데이터 검증
        is_valid = interface.validate_observations(mock_observations)
        print(f"Observations valid: {is_valid}")
        
        if is_valid:
            # 액션 예측
            action = interface.get_action_from_observations(mock_observations)
            print(f"Predicted action keys: {list(action.keys())}")
            
            for key, value in action.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
        
        # 성능 통계
        stats = interface.get_performance_stats()
        print(f"\nPerformance stats: {stats}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: This is expected if the actual model path doesn't exist")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_gr00t_interface()