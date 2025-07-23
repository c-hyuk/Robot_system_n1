#!/usr/bin/env python3
"""
GR00T Action Horizon 테스트 - N1-2B 모델용
"""

import sys
import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path
import json

# 프로젝트 경로 설정 (상위 경로의 GR00T-N1-2B를 import 경로에 추가)
GR00T_N1_2B_PATH = str(Path(__file__).parent.parent / "GR00T-N1-2B")
sys.path.append(GR00T_N1_2B_PATH)
sys.path.append(str(Path(__file__).parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "gr00t")))

# GR00T 직접 import (로컬 gr00t 모듈 사용)
try:
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
except ImportError as e:
    print(f"❌ GR00T Import Error: {e}")
    sys.exit(1)


def create_horizon_mock_data():
    """Action horizon에 맞는 Mock 데이터 생성 (N1-2B config에 맞춤)"""
    print("\n🎭 Creating horizon mock data...")
    # N1-2B는 action_dim=32, action_horizon=16, state_dim=64 기준
    video_data = {
        'video.right_wrist_view': np.random.randint(0, 255, (1, 1, 224, 224, 3), dtype=np.uint8),
        'video.left_wrist_view': np.random.randint(0, 255, (1, 1, 224, 224, 3), dtype=np.uint8),
        'video.front_view': np.random.randint(0, 255, (1, 1, 224, 224, 3), dtype=np.uint8),
    }
    # state: (batch, horizon, state_dim=64)
    state_data = np.random.uniform(-1.0, 1.0, (1, 1, 64)).astype(np.float32)
    # action: (batch, horizon=16, action_dim=32)
    action_data = np.random.uniform(-1.0, 1.0, (1, 16, 32)).astype(np.float32)
    language_data = np.array(["Pick up the red cube and place it on the table"])
    print(f"  ✅ Horizon mock data created")
    print(f"    Video keys: {list(video_data.keys())}")
    print(f"    State shape: {state_data.shape}")
    print(f"    Action shape: {action_data.shape} (horizon=16, dim=32)")
    return video_data, state_data, action_data, language_data


def test_action_horizon():
    """Action Horizon 테스트 (N1-2B)"""
    print("🚀 Starting GR00T N1-2B Action Horizon Test")
    print("=" * 50)
    try:
        # 1. Mock 데이터 생성
        video_data, state_data, action_data, language_data = create_horizon_mock_data()
        # 2. GR00T 설정 로드
        print("\n🧠 Loading GR00T configuration...")
        embodiment_name = "dual_piper_arm"  # N1-2B도 동일 embodiment 사용
        if embodiment_name not in DATA_CONFIG_MAP:
            available = list(DATA_CONFIG_MAP.keys())
            raise ValueError(f"Unknown embodiment: {embodiment_name}. Available: {available}")
        data_config = DATA_CONFIG_MAP[embodiment_name]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        print("\n[디버그] transform pipeline:")
        for idx, t in enumerate(getattr(modality_transform, 'transforms', [])):
            print(f"  [{idx}] {type(t).__name__}")
        print(f"  ✅ GR00T configuration loaded")
        print(f"    Embodiment: {embodiment_name}")
        print(f"    Action horizon: {len(data_config.action_indices)}")
        print(f"    State horizon: {len(data_config.observation_indices)}")
        # 3. GR00T Policy 직접 초기화
        print("\n🧠 Initializing GR00T N1-2B policy directly...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # N1-2B 모델 경로 지정 (폴더 경로로 수정)
        model_path = GR00T_N1_2B_PATH  # 폴더 경로만 넘김
        config_path = os.path.join(GR00T_N1_2B_PATH, "config.json")
        print("[DEBUG] EmbodimentTag file:", getattr(EmbodimentTag, "__file__", "(no __file__ attribute)"))
        print("[DEBUG] Direct import EmbodimentTag members:", list(EmbodimentTag))
        print("EmbodimentTag module:", EmbodimentTag.__module__)
        with open("/home/rosota/GR00T-N1-2B/experiment_cfg/metadata.json") as f:
            meta = json.load(f)
        print("[DEBUG] metadata.json embodiment_tags keys:", list(meta["embodiment_tags"].keys()))
        # 정책 객체 생성
        policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_name,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=None,
            device=str(device)
        )
        print(f"  ✅ GR00T N1-2B policy initialized")
        print(f"    Device: {device}")
        print(f"    Model action horizon: {policy.model.action_head.config.action_horizon}")
        # 4. 데이터 준비 (action horizon에 맞게)
        print("\n🔄 Preparing data with correct action horizon...")
        observations = {}
        observations.update(video_data)
        observations['state'] = torch.tensor(state_data)
        if observations['state'].ndim == 1:
            observations['state'] = observations['state'].unsqueeze(0)
        observations['action'] = torch.tensor(action_data)
        observations['language'] = language_data
        print(f"  ✅ Observations prepared")
        print(f"    Keys: {list(observations.keys())}")
        print(f"    Action shape: {observations['action'].shape}")
        # 5. 직접 추론 수행
        print("\n🎯 Performing inference with action horizon...")
        start_time = time.time()
        with torch.no_grad():
            policy.model.eval()
            device_obs = {}
            for key, value in observations.items():
                if isinstance(value, torch.Tensor):
                    device_obs[key] = value.cpu()  # 반드시 CPU로 변환
                else:
                    device_obs[key] = value
            try:
                action_dict = policy.get_action(device_obs)
                print(f"  ✅ Direct inference successful")
            except Exception as e:
                print(f"  ⚠️ Direct inference failed: {e}")
                import traceback
                traceback.print_exc()
                print("  ❌ Inference failed. Exiting test.")
                return False
        inference_time = time.time() - start_time
        print(f"  ✅ Inference completed in {inference_time*1000:.2f}ms")
        # 6. 결과 분석
        print("\n🔍 Analyzing results...")
        if action_dict:
            print(f"  📊 Action keys: {list(action_dict.keys())}")
            if isinstance(action_dict, dict):
                if any(key.startswith('action.') for key in action_dict.keys()):
                    print(f"  📈 Single action returned (expected action horizon)")
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
                elif any(key.startswith('action_step_') for key in action_dict.keys()):
                    print(f"  📈 Action horizon returned: {len(action_dict)} steps")
                    for step_key, step_actions in action_dict.items():
                        print(f"    {step_key}:")
                        for action_key, action_value in step_actions.items():
                            if isinstance(action_value, np.ndarray):
                                print(f"      {action_key}: shape={action_value.shape}")
                                if action_value.size <= 5:
                                    print(f"        Values: {action_value.flatten()}")
                                else:
                                    print(f"        First 3 values: {action_value.flatten()[:3]}")
                elif isinstance(next(iter(action_dict.values())), torch.Tensor):
                    action_tensor = next(iter(action_dict.values()))
                    print(f"  📈 Action tensor returned: shape={action_tensor.shape}")
                    if action_tensor.ndim >= 2:
                        print(f"    Action horizon: {action_tensor.shape[0]}")
                        print(f"    Action dimension: {action_tensor.shape[1]}")
                        print(f"    First 3 steps:")
                        for i in range(min(3, action_tensor.shape[0])):
                            print(f"      Step {i}: {action_tensor[i, :5].cpu().numpy()}")
                    else:
                        print(f"    Single action: {action_tensor[:5].cpu().numpy()}")
                else:
                    print(f"  📈 Unknown action format: {type(action_dict)}")
                    for key, value in action_dict.items():
                        print(f"    {key}: {type(value)}")
        else:
            print("  ❌ No action returned")
        print("\n" + "=" * 50)
        print("✅ GR00T N1-2B Action Horizon Test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_action_horizon()
    sys.exit(0 if success else 1) 