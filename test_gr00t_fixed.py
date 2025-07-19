#!/usr/bin/env python3
"""
GR00T 수정된 테스트 - 올바른 데이터 형식 사용
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


def create_mock_data():
    """업데이트된 metadata에 맞는 Mock 데이터 생성"""
    print("\n🎭 Creating mock data for updated metadata...")
    
    # 비디오 데이터 (업데이트된 키)
    video_data = {
        'video.right_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        'video.left_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        'video.front_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
    }
    
    # 상태 데이터 (업데이트된 키)
    state_data = {
        'state.right_arm_eef_pos': np.random.uniform(0.05, 0.2, 3).astype(np.float32),
        'state.right_arm_eef_quat': np.random.uniform(-1.0, 1.0, 4).astype(np.float32),
        'state.right_gripper_qpos': np.random.uniform(0.0, 0.07, 1).astype(np.float32),
        'state.left_arm_eef_pos': np.random.uniform(-0.2, -0.05, 3).astype(np.float32),
        'state.left_arm_eef_quat': np.random.uniform(-1.0, 1.0, 4).astype(np.float32),
        'state.left_gripper_qpos': np.random.uniform(0.0, 0.07, 1).astype(np.float32),
    }
    
    # 액션 데이터 (업데이트된 키)
    action_data = {
        'action.right_arm_eef_pos': np.random.uniform(0.03, 0.22, 3).astype(np.float32),
        'action.right_arm_eef_rot': np.random.uniform(-1.0, 1.0, 6).astype(np.float32),
        'action.right_gripper_close': np.random.uniform(0.0, 1.0, 1).astype(np.float32),
        'action.left_arm_eef_pos': np.random.uniform(-0.22, -0.03, 3).astype(np.float32),
        'action.left_arm_eef_rot': np.random.uniform(-1.0, 1.0, 6).astype(np.float32),
        'action.left_gripper_close': np.random.uniform(0.0, 1.0, 1).astype(np.float32),
    }
    
    # 언어 데이터
    language_data = {
        "annotation.language.instruction": "Pick up the red cube and place it on the table"
    }
    
    print(f"  ✅ Mock data created")
    print(f"    Video keys: {list(video_data.keys())}")
    print(f"    State keys: {list(state_data.keys())}")
    print(f"    Action keys: {list(action_data.keys())}")
    
    return video_data, state_data, action_data, language_data


def create_gr00t_observations(video_data, state_data, action_data, language_data):
    """GR00T 모델이 기대하는 형식으로 데이터 변환"""
    print("\n🔄 Converting to GR00T format...")
    
    try:
        # GR00T가 기대하는 형식으로 데이터 구성
        observations = {}
        
        # 비디오 데이터 - 각 카메라별로 개별 키로 제공
        for camera_name, frame in video_data.items():
            observations[camera_name] = frame  # (1, H, W, C) numpy array
        
        # 상태 데이터 변환
        state_vector = []
        for key in [
            'state.right_arm_eef_pos',
            'state.right_arm_eef_quat', 
            'state.right_gripper_qpos',
            'state.left_arm_eef_pos',
            'state.left_arm_eef_quat',
            'state.left_gripper_qpos',
        ]:
            state_vector.extend(state_data[key])
        
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        observations['state'] = state_tensor
        
        # 액션 데이터 변환
        action_vector = []
        for key in [
            'action.right_arm_eef_pos',
            'action.right_arm_eef_rot',
            'action.right_gripper_close',
            'action.left_arm_eef_pos',
            'action.left_arm_eef_rot',
            'action.left_gripper_close',
        ]:
            action_vector.extend(action_data[key])
        
        action_tensor = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0)
        observations['action'] = action_tensor
        
        # 언어 데이터
        text_data = language_data.get("annotation.language.instruction", "")
        observations['language'] = np.array([text_data])
        
        print(f"  ✅ GR00T observations created")
        print(f"    Video keys: {list(video_data.keys())}")
        print(f"    State shape: {state_tensor.shape}")
        print(f"    Action shape: {action_tensor.shape}")
        
        return observations
        
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        raise


def test_gr00t_inference():
    """GR00T 추론 테스트"""
    print("🚀 Starting GR00T Fixed Test")
    print("=" * 50)
    
    try:
        # 1. Mock 데이터 생성
        video_data, state_data, action_data, language_data = create_mock_data()
        
        # 2. GR00T 형식으로 변환
        observations = create_gr00t_observations(video_data, state_data, action_data, language_data)
        
        # 3. GR00T 인터페이스 초기화
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
        print("✅ GR00T Fixed Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gr00t_inference()
    sys.exit(0 if success else 1) 