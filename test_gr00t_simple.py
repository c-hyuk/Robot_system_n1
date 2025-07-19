#!/usr/bin/env python3
"""
GR00T 간단한 테스트 - 변환 파이프라인 우회
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

# 필요한 모듈 import
try:
    from model.gr00t_interface import DualPiperGR00TInterface
    from utils.data_types import RobotData
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


def create_simple_mock_data():
    """GR00T 모델이 기대하는 정확한 형식의 Mock 데이터 생성"""
    print("\n🎭 Creating simple mock data...")
    
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
    
    print(f"  ✅ Simple mock data created")
    print(f"    Video keys: {list(video_data.keys())}")
    print(f"    State shape: {state_data.shape}")
    print(f"    Action shape: {action_data.shape}")
    
    return video_data, state_data, action_data, language_data


def test_direct_inference():
    """GR00T 직접 추론 테스트"""
    print("🚀 Starting GR00T Direct Test")
    print("=" * 50)
    
    try:
        # 1. Mock 데이터 생성
        video_data, state_data, action_data, language_data = create_simple_mock_data()
        
        # 2. GR00T 인터페이스 초기화
        print("\n🧠 Initializing GR00T interface...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        gr00t_interface = DualPiperGR00TInterface(
            model_path="nvidia/GR00T-N1.5-3B",
            embodiment_name="dual_piper_arm",
            device=device,
            use_mock_data=True
        )
        
        print(f"  ✅ GR00T interface initialized")
        print(f"    Device: {device}")
        
        # 3. 데이터 준비
        observations = {}
        observations.update(video_data)
        observations['state'] = torch.tensor(state_data)
        observations['action'] = torch.tensor(action_data)
        observations['language'] = language_data
        
        print(f"  ✅ Observations prepared")
        print(f"    Keys: {list(observations.keys())}")
        
        # 4. 추론 수행
        print("\n🎯 Performing inference...")
        start_time = time.time()
        
        action = gr00t_interface.get_action_from_observations(observations)
        
        inference_time = time.time() - start_time
        print(f"  ✅ Inference completed in {inference_time*1000:.2f}ms")
        
        # 5. 결과 분석
        print("\n🔍 Analyzing results...")
        if action:
            print(f"  📊 Action keys: {list(action.keys())}")
            for key, value in action.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    if value.size <= 10:
                        print(f"      Values: {value.flatten()}")
                    else:
                        print(f"      First 5 values: {value.flatten()[:5]}")
                else:
                    print(f"    {key}: {type(value)} = {value}")
        else:
            print("  ❌ No action returned")
        
        print("\n" + "=" * 50)
        print("✅ GR00T Direct Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_direct_inference()
    sys.exit(0 if success else 1) 