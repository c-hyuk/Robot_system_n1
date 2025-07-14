"""
통합 데이터 파이프라인
기존 Collector + GR00T 표준 실험 시스템 조합
"""

import time
from typing import Dict, Any, Optional
import logging

from utils.data_types import RobotData
from data.collectors.main_collector import MainDataCollector

# GR00T 표준 방식으로 설정 import
from gr00t.experiment.data_config import DATA_CONFIG_MAP


class IntegratedDataPipeline:
    """Collector + GR00T 표준 실험 시스템 통합 파이프라인"""
    
    def __init__(self, embodiment_name: str = "dual_piper_arm", use_mock: bool = False):
        self.embodiment_name = embodiment_name
        self.use_mock = use_mock
        
        # 기존 데이터 수집기 (변경 없음)
        self.data_collector = MainDataCollector(use_mock=use_mock)
        
        # GR00T 표준 설정 시스템 사용
        if embodiment_name not in DATA_CONFIG_MAP:
            raise ValueError(f"Unknown embodiment: {embodiment_name}. Available: {list(DATA_CONFIG_MAP.keys())}")
        
        self.config = DATA_CONFIG_MAP[embodiment_name]
        self.modality_config = self.config.modality_config()
        self.transform_pipeline = self.config.transform()
        
        # 상태
        self.is_running = False
        self.logger = logging.getLogger("IntegratedDataPipeline")
    
    def start(self) -> bool:
        """파이프라인 시작"""
        if not self.data_collector.start_collection():
            self.logger.error("Failed to start data collection")
            return False
        
        self.is_running = True
        self.logger.info(f"Integrated pipeline started with embodiment: {self.embodiment_name}")
        return True
    
    def stop(self) -> None:
        """파이프라인 중지"""
        self.data_collector.stop_collection()
        self.is_running = False
        self.logger.info("Integrated pipeline stopped")
    
    def get_gr00t_input(self) -> Optional[Dict[str, Any]]:
        """GR00T 모델 입력 형식으로 데이터 반환"""
        if not self.is_running:
            return None
        
        # 1. 기존 collector에서 데이터 수집
        robot_data = self.data_collector.collect_synchronized_data()
        if robot_data is None:
            return None
        
        # 2. GR00T 형식으로 변환
        gr00t_data = robot_data.gr00t_format
        
        # 3. GR00T Transform 적용 (표준 방식)
        try:
            # 데이터셋 메타데이터 설정 (필요시)
            # self.transform_pipeline.set_metadata(dataset_metadata)
            
            transformed_data = self.transform_pipeline.apply(gr00t_data)
            return transformed_data
        except Exception as e:
            self.logger.error(f"Transform failed: {e}")
            return None
    
    def set_training_mode(self, training: bool = True):
        """훈련/평가 모드 설정"""
        if training:
            self.transform_pipeline.train()
        else:
            self.transform_pipeline.eval()
        
        self.logger.info(f"Set to {'training' if training else 'evaluation'} mode")
    
    def get_modality_config(self) -> Dict[str, Any]:
        """GR00T 모달리티 설정 반환"""
        return self.modality_config
    
    def get_available_embodiments(self) -> list[str]:
        """사용 가능한 embodiment 목록 반환"""
        return list(DATA_CONFIG_MAP.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 반환"""
        collector_status = self.data_collector.get_system_status()
        
        return {
            'pipeline_running': self.is_running,
            'embodiment_name': self.embodiment_name,
            'collector_status': collector_status,
            'transform_mode': 'training' if self.transform_pipeline.training else 'evaluation',
            'available_embodiments': self.get_available_embodiments()
        }
    
    def __enter__(self):
        """Context manager 진입"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop()


def create_pipeline_for_embodiment(embodiment_name: str, use_mock: bool = False) -> IntegratedDataPipeline:
    """특정 embodiment용 파이프라인 생성"""
    return IntegratedDataPipeline(embodiment_name=embodiment_name, use_mock=use_mock)


def test_integrated_pipeline():
    """통합 파이프라인 테스트"""
    print("Testing integrated data pipeline with GR00T standard system...")
    
    # 사용 가능한 embodiment 확인
    pipeline = IntegratedDataPipeline(use_mock=True)
    available_embodiments = pipeline.get_available_embodiments()
    print(f"Available embodiments: {available_embodiments}")
    
    # Dual Piper 테스트
    embodiment_name = "dual_piper_arm"
    if embodiment_name not in available_embodiments:
        print(f"Warning: {embodiment_name} not in DATA_CONFIG_MAP. Using mock test...")
        embodiment_name = "fourier_gr1_arms_only"  # 대체용
    
    with create_pipeline_for_embodiment(embodiment_name, use_mock=True) as pipeline:
        # 시스템 준비 대기
        print("Waiting for system ready...")
        time.sleep(2)
        
        # 모달리티 설정 확인
        modality_config = pipeline.get_modality_config()
        print(f"\nModality configuration:")
        for modality, config in modality_config.items():
            print(f"  {modality}: {len(config.modality_keys)} keys")
            print(f"    Keys: {config.modality_keys}")
            print(f"    Delta indices: {config.delta_indices}")
        
        # 훈련 모드 테스트
        print(f"\n=== Training Mode ({embodiment_name}) ===")
        pipeline.set_training_mode(True)
        
        for i in range(3):
            gr00t_input = pipeline.get_gr00t_input()
            if gr00t_input:
                print(f"Training sample {i+1}:")
                for key, value in gr00t_input.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)} = {value}")
            else:
                print(f"Training sample {i+1}: No data")
            time.sleep(0.5)
        
        # 평가 모드 테스트
        print(f"\n=== Evaluation Mode ({embodiment_name}) ===")
        pipeline.set_training_mode(False)
        
        for i in range(3):
            gr00t_input = pipeline.get_gr00t_input()
            if gr00t_input:
                print(f"Evaluation sample {i+1}:")
                for key, value in gr00t_input.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)} = {value}")
            else:
                print(f"Evaluation sample {i+1}: No data")
            time.sleep(0.5)
        
        # 상태 확인
        status = pipeline.get_status()
        print(f"\nPipeline Status:")
        print(f"  Running: {status['pipeline_running']}")
        print(f"  Embodiment: {status['embodiment_name']}")
        print(f"  Mode: {status['transform_mode']}")
        print(f"  Available embodiments: {len(status['available_embodiments'])}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_integrated_pipeline()

    """Collector + GR00T Transform 통합 파이프라인"""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        
        # 기존 데이터 수집기 (변경 없음)
        self.data_collector = MainDataCollector(use_mock=use_mock)
        
        # GR00T 설정 및 Transform
        self.config = get_dual_piper_config()
        self.transform_pipeline = self.config.transform()
        
        # 상태
        self.is_running = False
        self.logger = logging.getLogger("IntegratedDataPipeline")
    
    def start(self) -> bool:
        """파이프라인 시작"""
        if not self.data_collector.start_collection():
            self.logger.error("Failed to start data collection")
            return False
        
        self.is_running = True
        self.logger.info("Integrated pipeline started")
        return True
    
    def stop(self) -> None:
        """파이프라인 중지"""
        self.data_collector.stop_collection()
        self.is_running = False
        self.logger.info("Integrated pipeline stopped")
    
    def get_gr00t_input(self) -> Optional[Dict[str, Any]]:
        """GR00T 모델 입력 형식으로 데이터 반환"""
        if not self.is_running:
            return None
        
        # 1. 기존 collector에서 데이터 수집
        robot_data = self.data_collector.collect_synchronized_data()
        if robot_data is None:
            return None
        
        # 2. GR00T 형식으로 변환
        gr00t_data = robot_data.gr00t_format
        
        # 3. GR00T Transform 적용
        try:
            transformed_data = self.transform_pipeline.apply(gr00t_data)
            return transformed_data
        except Exception as e:
            self.logger.error(f"Transform failed: {e}")
            return None
    
    def set_training_mode(self, training: bool = True):
        """훈련/평가 모드 설정"""
        if training:
            self.transform_pipeline.train()
        else:
            self.transform_pipeline.eval()
        
        self.logger.info(f"Set to {'training' if training else 'evaluation'} mode")
    
    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 반환"""
        collector_status = self.data_collector.get_system_status()
        
        return {
            'pipeline_running': self.is_running,
            'collector_status': collector_status,
            'transform_mode': 'training' if self.transform_pipeline.training else 'evaluation',
            'config': 'dual_piper_arm'
        }
    
    def __enter__(self):
        """Context manager 진입"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop()


def test_integrated_pipeline():
    """통합 파이프라인 테스트"""
    print("Testing integrated data pipeline...")
    
    with IntegratedDataPipeline(use_mock=True) as pipeline:
        # 시스템 준비 대기
        print("Waiting for system ready...")
        time.sleep(2)
        
        # 훈련 모드 테스트
        print("\n=== Training Mode ===")
        pipeline.set_training_mode(True)
        
        for i in range(3):
            gr00t_input = pipeline.get_gr00t_input()
            if gr00t_input:
                print(f"Training sample {i+1}:")
                for key, value in gr00t_input.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            time.sleep(0.5)
        
        # 평가 모드 테스트
        print("\n=== Evaluation Mode ===")
        pipeline.set_training_mode(False)
        
        for i in range(3):
            gr00t_input = pipeline.get_gr00t_input()
            if gr00t_input:
                print(f"Evaluation sample {i+1}:")
                for key, value in gr00t_input.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            time.sleep(0.5)
        
        # 상태 확인
        status = pipeline.get_status()
        print(f"\nPipeline Status:")
        print(f"  Running: {status['pipeline_running']}")
        print(f"  Mode: {status['transform_mode']}")
        print(f"  Config: {status['config']}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_integrated_pipeline()