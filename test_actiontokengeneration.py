#!/usr/bin/env python3
"""
GR00T 추론 시스템 최소 테스트
단일 추론 수행 및 Action Token 검증
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
    from data.integrated_pipeline import IntegratedDataPipeline
    from utils.data_types import RobotData
    from config.hardware_config import get_hardware_config
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class GR00TMinimalTester:
    """GR00T 최소 테스트 클래스"""
    
    def __init__(self, use_mock_data: bool = True):
        """
        테스터 초기화
        
        Args:
            use_mock_data: Mock 데이터 사용 여부
        """
        self.use_mock_data = use_mock_data
        self.logger = logging.getLogger("GR00TMinimalTester")
        
        # 설정
        self.embodiment_name = "dual_piper_arm"
        # self.embodiment_name = "agibot_genie1"
        self.model_path = "nvidia/GR00T-N1.5-3B"  # 공식 모델 경로로 수정
        
        # 컴포넌트들
        self.gr00t_interface: Optional[DualPiperGR00TInterface] = None
        self.data_pipeline: Optional[IntegratedDataPipeline] = None
        
        print(f"🔧 GR00T Minimal Tester initialized (Mock: {use_mock_data})")
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def check_dependencies(self) -> bool:
        """의존성 확인"""
        print("\n📋 Checking dependencies...")
        
        try:
            # PyTorch 확인
            print(f"  PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"  CUDA: Available (GPU: {torch.cuda.get_device_name()})")
            else:
                print(f"  CUDA: Not available (using CPU)")
            
            # NumPy 확인
            print(f"  NumPy: {np.__version__}")
            
            # GR00T 모듈 확인
            try:
                from gr00t.model.policy import Gr00tPolicy
                print(f"  GR00T Policy: Available")
            except ImportError:
                print(f"  GR00T Policy: Not available (will use mock)")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ❌ Dependency check failed: {e}")
            return False
    
    def create_mock_robot_data(self) -> RobotData:
        """Mock 로봇 데이터 생성"""
        print("\n🎭 Creating mock robot data...")
        
        # 현재 시간
        timestamp = time.time()
        
        # GR00T 포맷에 맞는 Mock 비디오 데이터 (예시: left_arm_d435, right_arm_d435)
        mock_video = {
            'video.left_arm_d435': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'video.right_arm_d435': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        }
        
        # GR00T 포맷에 맞는 Mock 상태 데이터
        mock_state = {
            'state.left_arm_joint_position': np.random.uniform(-1, 1, 7).astype(np.float32),
            'state.right_arm_joint_position': np.random.uniform(-1, 1, 7).astype(np.float32),
            'state.left_effector_position': np.random.uniform(-0.5, 0.5, 6).astype(np.float32),
            'state.right_effector_position': np.random.uniform(-0.5, 0.5, 6).astype(np.float32),
        }
        
        # Mock 액션 데이터 (GR00T 포맷)
        mock_action = {
            'action.left_arm_joint_position': np.random.uniform(-1, 1, 7).astype(np.float32),
            'action.right_arm_joint_position': np.random.uniform(-1, 1, 7).astype(np.float32),
            'action.left_effector_position': np.random.uniform(-0.5, 0.5, 6).astype(np.float32),
            'action.right_effector_position': np.random.uniform(-0.5, 0.5, 6).astype(np.float32),
        }
        
        # Mock 텍스트 명령 (GR00T 포맷)
        mock_text = "Pick up the red cube and place it on the table"
        mock_language = {"annotation.language.instruction": mock_text}
        
        # RobotData 객체 생성 및 속성 할당
        robot_data = RobotData()
        robot_data.timestamp = timestamp
        robot_data.video_data = mock_video
        robot_data.state_data = mock_state
        robot_data.action_data = mock_action
        robot_data.language_data = mock_language
        
        print(f"  ✅ Mock robot data created")
        print(f"    Video frames: {len(mock_video)}")
        print(f"    State keys: {list(mock_state.keys())}")
        print(f"    Text command: {mock_text[:50]}...")
        
        return robot_data
    
    def create_gr00t_observations(self, robot_data: RobotData) -> Dict[str, Any]:
        """로봇 데이터를 GR00T 관찰 형식으로 변환"""
        print("\n🔄 Converting to GR00T observations...")
        
        try:
            # 비디오 데이터 변환 (Batch dimension 추가)
            video_tensor = []
            for camera_name, frame in robot_data.video_data.items():
                # (H, W, C) -> (C, H, W) -> (1, C, H, W)
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                video_tensor.append(frame_tensor)
            video_tensor = torch.cat(video_tensor, dim=0)  # (N_cameras, C, H, W)
            video_tensor = video_tensor.unsqueeze(0)  # (1, N_cameras, C, H, W)
            
            # 상태 데이터 변환 (각 state key별로 이어붙임)
            state_vector = []
            for key in [
                'state.left_arm_joint_position',
                'state.right_arm_joint_position',
                'state.left_effector_position',
                'state.right_effector_position',
            ]:
                state_vector.extend(robot_data.state_data[key])
            state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
            
            # 액션 데이터 변환 (각 action key별로 이어붙임)
            action_vector = []
            for key in [
                'action.left_arm_joint_position',
                'action.right_arm_joint_position',
                'action.left_effector_position',
                'action.right_effector_position',
            ]:
                action_vector.extend(robot_data.action_data[key])
            action_tensor = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0)  # (1, action_dim)

            # 텍스트 데이터 (GR00T 포맷)
            text_data = robot_data.language_data.get("annotation.language.instruction", "")
            language_tensor = np.array([text_data])

            observations = {
                'video': video_tensor,
                'state': state_tensor,
                'action': action_tensor,
                'language': language_tensor
            }
            
            print(f"  ✅ GR00T observations created")
            print(f"    Video shape: {video_tensor.shape}")
            print(f"    State shape: {state_tensor.shape}")
            print(f"    Language: {text_data}")
            
            return observations
            
        except Exception as e:
            print(f"  ❌ Conversion failed: {e}")
            raise
    
    def initialize_gr00t_interface(self) -> bool:
        """GR00T 인터페이스 초기화"""
        print("\n🧠 Initializing GR00T interface...")
        
        try:
            # 디바이스 설정
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # GR00T 인터페이스 생성
            self.gr00t_interface = DualPiperGR00TInterface(
                model_path=self.model_path,
                embodiment_name=self.embodiment_name,
                device=device,
                use_mock_data=self.use_mock_data
            )
            
            # 모델 정보 확인
            model_info = self.gr00t_interface.get_model_info()
            print(f"  ✅ GR00T interface initialized")
            print(f"    Model: {model_info.get('model_path', 'Unknown')}")
            print(f"    Embodiment: {model_info.get('embodiment_name', 'Unknown')}")
            print(f"    Device: {model_info.get('device', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ GR00T interface initialization failed: {e}")
            return False
    
    def test_single_inference(self, observations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """단일 추론 테스트"""
        print("\n🎯 Testing single inference...")
        
        try:
            # 관찰 데이터 검증
            print("  📋 Validating observations...")
            if self.gr00t_interface is None:
                print("  ❌ GR00T interface is None")
                return None
            is_valid = self.gr00t_interface.validate_observations(observations)
            if not is_valid:
                print("  ❌ Observations validation failed")
                return None
            
            print("  ✅ Observations validated")
            
            # 추론 수행
            print("  🚀 Performing inference...")
            start_time = time.time()
            
            action = self.gr00t_interface.get_action_from_observations(observations)
            
            inference_time = time.time() - start_time
            
            print(f"  ✅ Inference completed in {inference_time*1000:.2f}ms")
            
            return action
            
        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
            return None
    
    def analyze_action_token(self, action: Dict[str, Any]):
        """Action Token 분석"""
        print("\n🔍 Analyzing Action Token...")
        
        try:
            print(f"  📊 Action Token Structure:")
            print(f"    Total keys: {len(action)}")
            
            # 각 액션 키 분석
            for key, value in action.items():
                print(f"\n    🔑 {key}:")
                
                if isinstance(value, np.ndarray):
                    print(f"      Type: numpy.ndarray")
                    print(f"      Shape: {value.shape}")
                    print(f"      Dtype: {value.dtype}")
                    print(f"      Range: [{value.min():.4f}, {value.max():.4f}]")
                    
                    # 첫 몇 개 값 출력
                    if value.size <= 20:
                        print(f"      Values: {value.flatten()}")
                    else:
                        print(f"      First 5 values: {value.flatten()[:5]}")
                        
                elif isinstance(value, dict):
                    print(f"      Type: dict")
                    print(f"      Sub-keys: {list(value.keys())}")
                    
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            print(f"        {sub_key}: {sub_value.shape} {sub_value.dtype}")
                        else:
                            print(f"        {sub_key}: {type(sub_value)} = {sub_value}")
                            
                else:
                    print(f"      Type: {type(value)}")
                    print(f"      Value: {value}")
            
            # 액션 해석
            print(f"\n  🎯 Action Interpretation:")
            
            # 일반적인 로봇 액션 패턴 확인
            if 'arm_action' in action:
                arm_action = action['arm_action']
                if isinstance(arm_action, np.ndarray):
                    print(f"    Arm actions detected: {arm_action.shape}")
                    if arm_action.shape[-1] == 14:  # 14 DOF dual arm
                        print(f"    Left arm (7 joints): {arm_action[:7]}")
                        print(f"    Right arm (7 joints): {arm_action[7:]}")
                    
            if 'gripper_action' in action:
                gripper_action = action['gripper_action']
                if isinstance(gripper_action, np.ndarray):
                    print(f"    Gripper actions: {gripper_action}")
                    
            if 'terminate' in action:
                terminate = action['terminate']
                print(f"    Terminate signal: {terminate}")
            
        except Exception as e:
            print(f"  ❌ Action analysis failed: {e}")
    
    def run_test(self) -> bool:
        """전체 테스트 실행"""
        print("🚀 Starting GR00T Minimal Inference Test")
        print("=" * 60)
        
        try:
            # 1. 로깅 설정
            self.setup_logging()
            
            # 2. 의존성 확인
            if not self.check_dependencies():
                print("❌ Dependency check failed")
                return False
            
            # 3. Mock 데이터 생성
            robot_data = self.create_mock_robot_data()
            
            # 4. GR00T 관찰 데이터 변환
            observations = self.create_gr00t_observations(robot_data)
            
            # 5. GR00T 인터페이스 초기화
            if not self.initialize_gr00t_interface():
                print("❌ GR00T interface initialization failed")
                return False
            
            # 6. 단일 추론 테스트
            action = self.test_single_inference(observations)
            
            if action is None:
                print("❌ Inference test failed")
                return False
            
            # 7. Action Token 분석
            self.analyze_action_token(action)
            
            # 8. 성능 통계
            if self.gr00t_interface is not None:
                stats = self.gr00t_interface.get_performance_stats()
                print(f"\n📈 Performance Statistics:")
                print(f"    Total inferences: {stats.get('total_inferences', 0)}")
                print(f"    Average inference time: {stats.get('avg_inference_time_ms', 0):.2f}ms")
                print(f"    Uptime: {stats.get('uptime_seconds', 0):.2f}s")
            else:
                print("[WARN] GR00T interface is None, cannot print performance stats.")
            
            print("\n" + "=" * 60)
            print("✅ GR00T Minimal Inference Test PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # 정리 작업
            if self.gr00t_interface:
                self.gr00t_interface.stop_data_pipeline()


def main():
    """메인 함수"""
    print("GR00T Minimal Inference Test")
    print("This test will perform a single inference to verify the GR00T system")
    
    # 사용자 옵션
    use_mock = True  # 실제 모델 사용시 False로 변경
    
    if use_mock:
        print("\n⚠️  Using MOCK data (no real model required)")
    else:
        print("\n🔥 Using REAL model (requires actual GR00T model)")
    
    # 테스트 실행
    tester = GR00TMinimalTester(use_mock_data=use_mock)
    success = tester.run_test()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("The GR00T inference system is working correctly.")
    else:
        print("\n💥 Test failed!")
        print("Please check the error messages above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
