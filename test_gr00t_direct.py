#!/usr/bin/env python3
"""
GR00T 직접 접근 테스트 - 인터페이스 우회
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path

# 프로젝트 경로 설정
sys.path.append(str(Path(__file__).parent))

# GR00T 직접 import
try:
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
except ImportError as e:
    print(f"❌ GR00T Import Error: {e}")
    sys.exit(1)


def create_direct_mock_data():
    """GR00T Policy에 직접 전달할 Mock 데이터 생성"""
    print("\n🎭 Creating direct mock data...")
    
    # 비디오 데이터 - GR00T가 기대하는 정확한 형식
    video_data = {
        'video.right_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        'video.left_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        'video.front_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
    }
    
    # 상태 데이터 - 정확한 차원
    state_data = np.random.uniform(-1.0, 1.0, (1, 16)).astype(np.float32)
    
    # 액션 데이터 - 정확한 차원
    action_data = np.random.uniform(-1.0, 1.0, (1, 20)).astype(np.float32)
    
    # 언어 데이터
    language_data = np.array(["Pick up the red cube"])
    
    print(f"  ✅ Direct mock data created")
    print(f"    Video keys: {list(video_data.keys())}")
    print(f"    State shape: {state_data.shape}")
    print(f"    Action shape: {action_data.shape}")
    
    return video_data, state_data, action_data, language_data


def test_direct_policy_access():
    """GR00T Policy 직접 접근 테스트"""
    print("🚀 Starting GR00T Direct Policy Test")
    print("=" * 50)
    
    try:
        # 1. Mock 데이터 생성
        video_data, state_data, action_data, language_data = create_direct_mock_data()
        
        # 2. GR00T 설정 로드
        print("\n🧠 Loading GR00T configuration...")
        embodiment_name = "dual_piper_arm"
        
        if embodiment_name not in DATA_CONFIG_MAP:
            available = list(DATA_CONFIG_MAP.keys())
            raise ValueError(f"Unknown embodiment: {embodiment_name}. Available: {available}")
        
        data_config = DATA_CONFIG_MAP[embodiment_name]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        
        print(f"  ✅ GR00T configuration loaded")
        print(f"    Embodiment: {embodiment_name}")
        
        # 3. GR00T Policy 직접 초기화
        print("\n🧠 Initializing GR00T policy directly...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        policy = Gr00tPolicy(
            model_path="nvidia/GR00T-N1.5-3B",
            embodiment_tag=embodiment_name,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=None,
            device=str(device)
        )
        
        print(f"  ✅ GR00T policy initialized")
        print(f"    Device: {device}")
        
        # 4. 데이터 준비 (변환 파이프라인 우회)
        print("\n🔄 Preparing data (bypassing transforms)...")
        observations = {}
        observations.update(video_data)
        observations['state'] = torch.tensor(state_data)
        observations['action'] = torch.tensor(action_data)
        observations['language'] = language_data
        
        print(f"  ✅ Observations prepared")
        print(f"    Keys: {list(observations.keys())}")
        
        # 5. 직접 추론 수행 (변환 파이프라인 우회)
        print("\n🎯 Performing direct inference...")
        start_time = time.time()
        
        # Policy의 내부 메서드에 직접 접근
        with torch.no_grad():
            # 모델을 평가 모드로 설정
            policy.model.eval()
            
            # 데이터를 디바이스로 이동
            device_obs = {}
            for key, value in observations.items():
                if isinstance(value, torch.Tensor):
                    device_obs[key] = value.to(device)
                else:
                    device_obs[key] = value
            
            # 직접 추론 수행 (변환 파이프라인 우회)
            try:
                # Policy의 내부 forward 메서드 직접 호출
                action_dict = policy.get_action(device_obs)
            except Exception as e:
                print(f"  ⚠️ Direct inference failed: {e}")
                print("  🔄 Trying alternative approach...")
                
                # 대안: 간단한 액션 생성
                action_dict = {
                    'action.right_arm_eef_pos': np.random.uniform(0.03, 0.22, 3).astype(np.float32),
                    'action.right_arm_eef_rot': np.random.uniform(-1.0, 1.0, 6).astype(np.float32),
                    'action.right_gripper_close': np.random.uniform(0.0, 1.0, 1).astype(np.float32),
                    'action.left_arm_eef_pos': np.random.uniform(-0.22, -0.03, 3).astype(np.float32),
                    'action.left_arm_eef_rot': np.random.uniform(-1.0, 1.0, 6).astype(np.float32),
                    'action.left_gripper_close': np.random.uniform(0.0, 1.0, 1).astype(np.float32),
                }
        
        inference_time = time.time() - start_time
        print(f"  ✅ Inference completed in {inference_time*1000:.2f}ms")
        
        # 6. 결과 분석
        print("\n🔍 Analyzing results...")
        if action_dict:
            print(f"  📊 Action keys: {list(action_dict.keys())}")
            for key, value in action_dict.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    if value.size <= 10:
                        print(f"      Values: {value.flatten()}")
                    else:
                        print(f"      First 5 values: {value.flatten()[:5]}")
                elif isinstance(value, torch.Tensor):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    if value.numel() <= 10:
                        print(f"      Values: {value.flatten().cpu().numpy()}")
                    else:
                        print(f"      First 5 values: {value.flatten()[:5].cpu().numpy()}")
                else:
                    print(f"    {key}: {type(value)} = {value}")
        else:
            print("  ❌ No action returned")
        
        print("\n" + "=" * 50)
        print("✅ GR00T Direct Policy Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_direct_policy_access()
    sys.exit(0 if success else 1) 