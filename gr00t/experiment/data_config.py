# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


###########################################################################################


class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]"""
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
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DualCamDataConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


###########################################################################################


class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


###########################################################################################


class FourierGr1FullUpperBodyDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
        "state.neck",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
        "action.neck",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.front_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_gripper_qpos": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_gripper_qpos": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }
    action_normalization_modes = {
        "action.right_gripper_close": "binary",
        "action.left_gripper_close": "binary",
    }

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaHandDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_hand",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_hand",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_hand": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_hand": "min_max",
    }
    action_normalization_modes = {
        "action.right_hand": "min_max",
        "action.left_hand": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }


###########################################################################################


class SinglePandaGripperDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }


###########################################################################################


class FourierGr1ArmsWaistDataConfig(FourierGr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return super().modality_config()

    def transform(self):
        return super().transform()


###########################################################################################


class OxeDroidDataConfig:
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class AgibotGenie1DataConfig:
    video_keys = [
        "video.top_head",
        "video.hand_left",
        "video.hand_right",
    ]
    state_keys = [
        "state.left_arm_joint_position",
        "state.right_arm_joint_position",
        "state.left_effector_position",
        "state.right_effector_position",
        "state.head_position",
        "state.waist_position",
    ]
    action_keys = [
        "action.left_arm_joint_position",
        "action.right_arm_joint_position",
        "action.left_effector_position",
        "action.right_effector_position",
        "action.head_position",
        "action.waist_position",
        "action.robot_velocity",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_joint_position": "min_max",
                    "state.right_arm_joint_position": "min_max",
                    "state.left_effector_position": "min_max",
                    "state.right_effector_position": "min_max",
                    "state.head_position": "min_max",
                    "state.waist_position": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_arm_joint_position": "min_max",
                    "action.right_arm_joint_position": "min_max",
                    "action.left_effector_position": "min_max",
                    "action.right_effector_position": "min_max",
                    "action.head_position": "min_max",
                    "action.waist_position": "min_max",
                    "action.robot_velocity": "min_max",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################
# NEW: Dual Piper Configuration
###########################################################################################


class DualPiperDataConfig(BaseDataConfig):
    """Dual Piper 로봇용 데이터 설정"""
    
    # 비디오 키 (우리 하드웨어에 맞춰 설정)
    video_keys = [
        "video.left_arm_d435",    # 왼쪽 팔 D435 카메라
        "video.right_arm_d435",   # 오른쪽 팔 D435 카메라  
        "video.center_zed21",     # 중앙 Zed21 스테레오 카메라
    ]
    
    # 상태 키 (Dual Piper arm 구성)
    state_keys = [
        "state.left_arm_joint_position",     # 왼쪽 팔 관절 위치 (7 DOF)
        "state.right_arm_joint_position",    # 오른쪽 팔 관절 위치 (7 DOF)
        "state.left_arm_effector_position",  # 왼쪽 엔드이펙터 위치/자세 (6 DOF)
        "state.right_arm_effector_position", # 오른쪽 엔드이펙터 위치/자세 (6 DOF)
    ]
    
    # 액션 키 (상태와 동일한 구조)
    action_keys = [
        "action.left_arm_joint_position",
        "action.right_arm_joint_position", 
        "action.left_arm_effector_position",
        "action.right_arm_effector_position",
    ]
    
    # 언어 키
    language_keys = ["annotation.language.instruction"]
    
    # GR00T 표준 설정
    observation_indices = [0]        # state_horizon = 1 (현재 상태만)
    action_indices = list(range(16)) # action_horizon = 16
    
    def modality_config(self) -> dict[str, ModalityConfig]:
        """GR00T 모달리티 설정"""
        
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices, 
            modality_keys=self.state_keys,
        )
        
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        
        return modality_configs
    
    def transform(self) -> ComposedModalityTransform:
        """GR00T Transform 파이프라인 생성"""
        
        transforms = [
            # 비디오 변환 (GR00T 표준)
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            
            # 상태 변환
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_joint_position": "min_max",
                    "state.right_arm_joint_position": "min_max", 
                    "state.left_arm_effector_position": "min_max",
                    "state.right_arm_effector_position": "min_max",
                },
            ),
            
            # 액션 변환
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_arm_joint_position": "min_max",
                    "action.right_arm_joint_position": "min_max",
                    "action.left_arm_effector_position": "min_max", 
                    "action.right_arm_effector_position": "min_max",
                },
            ),
            
            # 연결 변환
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            
            # GR00T 특정 변환
            GR00TTransform(
                state_horizon=len(self.observation_indices),   # 1
                action_horizon=len(self.action_indices),       # 16
                max_state_dim=64,                              # GR00T 표준
                max_action_dim=32,                             # GR00T 표준
            ),
        ]
        
        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

DATA_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "fourier_gr1_full_upper_body": FourierGr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "unitree_g1": UnitreeG1DataConfig(),
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "agibot_genie1": AgibotGenie1DataConfig(),


    # 새로 추가: Dual Piper 설정
    "dual_piper_arm": DualPiperDataConfig(),
}

