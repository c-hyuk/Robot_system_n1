#!/usr/bin/env python3
"""
GR00T End-to-End Robot Control - Simple & Direct
기존 모듈들을 활용한 간결한 end-to-end 실행 스크립트
"""

import sys
import os
import time
import logging
import signal
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
import threading

# 경로 설정
sys.path.append(str(Path(__file__).parent))
GR00T_N1_2B_PATH = str(Path(__file__).parent.parent / "GR00T-N1-2B")

# 기존 모듈들 import
from communication.hardware_bridge import PiperHardwareBridge
from model.action_decoder import create_action_decoder
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.embodiment_tags import EmbodimentTag
from data.collectors.vision_collector import VisionCollectorManager
from data.collectors.text_collector import TextCollectorManager
from data.collectors.state_collector import DualArmStateCollectorManager
from control.trajectory_executor import TrajectoryExecutor, ExecutionConfig
from control.safety_manager import SafetyManager
from config.hardware_config import get_hardware_config
import json
from gr00t.data.schema import DatasetMetadata

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GR00T-E2E")


class GR00TRobotSystem:
    """GR00T 로봇 시스템 - 모든 컴포넌트 통합"""
    
    def __init__(self, dry_run: bool = False, mock_vision: bool = False):
        self.dry_run = dry_run
        self.mock_vision = mock_vision
        self.running = False
        
        # 컴포넌트들
        self.hardware_bridge: Optional[PiperHardwareBridge] = None
        self.policy: Optional[Gr00tPolicy] = None
        self.vision_collector: Optional[VisionCollectorManager] = None
        self.text_collector: Optional[TextCollectorManager] = None
        self.state_collector: Optional[DualArmStateCollectorManager] = None
        self.trajectory_executor: Optional[TrajectoryExecutor] = None
        self.safety_manager: Optional[SafetyManager] = None
        self.action_decoder = None
        
    def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("="*60)
            logger.info("🚀 GR00T End-to-End System 초기화")
            logger.info("="*60)
            
            # 1. 하드웨어 브릿지
            if not self.dry_run:
                logger.info("[1/7] 하드웨어 브릿지 초기화...")
                self.hardware_bridge = PiperHardwareBridge(
                    left_can_port="can0",
                    right_can_port="can1",
                    auto_enable=True,
                    gripper_enabled=True
                )
                if not self.hardware_bridge.connect():
                    logger.error("하드웨어 연결 실패")
                    return False
                logger.info("✅ 하드웨어 연결 완료")
            else:
                logger.info("[1/7] Dry-run 모드 - 하드웨어 연결 생략")
            
            # 2. Safety Manager
            logger.info("[2/7] Safety Manager 초기화...")
            hw_config = get_hardware_config()
            self.safety_manager = SafetyManager(hw_config)
            self.safety_manager.start_monitoring()
            logger.info("✅ Safety Manager 시작")
            
            # 3. 데이터 수집기들
            logger.info("[3/7] 비전 수집기 초기화...")
            self.vision_collector = VisionCollectorManager(use_mock=self.mock_vision)
            self.vision_collector.start_all_cameras()
            logger.info(f"✅ 비전 수집기 시작 (mock={self.mock_vision})")
            # 디버깅: vision 프레임 1회 출력
            try:
                vision_frames = self.vision_collector.get_all_frames()
                for k, v in vision_frames.items():
                    logger.info(f"[DEBUG][Vision] {k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
            except Exception as e:
                logger.warning(f"[DEBUG][Vision] 프레임 출력 실패: {e}")
            logger.info("[4/7] 텍스트 수집기 초기화...")
            self.text_collector = TextCollectorManager(use_mock=False)  # 항상 실제 입력
            self.text_collector.start_collection()
            logger.info("✅ 텍스트 수집기 시작")
            
            logger.info("[5/7] 상태 수집기 초기화...")
            self.state_collector = DualArmStateCollectorManager(control_frequency=10.0)
            if self.hardware_bridge:
                # 하드웨어 브릿지의 Piper 인터페이스 공유
                self.state_collector.left_collector.robot_connection = self.hardware_bridge.arms.get('left')
                self.state_collector.right_collector.robot_connection = self.hardware_bridge.arms.get('right')
            self.state_collector.start_all_collectors()
            logger.info("✅ 상태 수집기 시작")
            # 디버깅: 로봇 상태 1회 출력
            try:
                robot_states = self.state_collector.get_all_states()
                for k, v in robot_states.items():
                    logger.info(f"[DEBUG][Robot] {k}: {v}")
            except Exception as e:
                logger.warning(f"[DEBUG][Robot] 상태 출력 실패: {e}")
            
            # 6. Gr00tPolicy 직접 생성
            logger.info("[6/7] GR00T Policy 직접 생성...")
            embodiment_name = "dual_piper_arm"
            data_config = DATA_CONFIG_MAP[embodiment_name]
            modality_config = data_config.modality_config()
            modality_transform = data_config.transform()
            # metadata.json 로드 및 set_metadata 적용
            metadata_path = os.path.join("/home/rosota/GR00T-N1-2B/experiment_cfg", "metadata.json")
            with open(metadata_path, "r") as f:
                metadatas = json.load(f)
            meta_dict = metadatas["embodiment_tags"][embodiment_name]
            metadata = DatasetMetadata.model_validate(meta_dict)
            modality_transform.set_metadata(metadata)
            self.policy = Gr00tPolicy(
                model_path=GR00T_N1_2B_PATH,
                embodiment_tag=embodiment_name,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=None,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("✅ GR00T Policy 생성 완료")
            
            # 7. 액션 디코더 & Trajectory Executor
            logger.info("[7/7] 액션 디코더 및 실행기 초기화...")
            self.action_decoder = create_action_decoder("dual_piper_arm")
            
            if self.hardware_bridge and not self.dry_run:
                self.trajectory_executor = TrajectoryExecutor(
                    hardware_bridge=self.hardware_bridge,
                    safety_manager=self.safety_manager,
                    config=ExecutionConfig(
                        execution_frequency=10.0,
                        blending_alpha=0.5,
                        step_blending_alpha=0.7
                    )
                )
            logger.info("✅ 액션 시스템 준비 완료")
            
            logger.info("\n🎉 시스템 초기화 성공!")
            logger.info("텍스트 명령을 입력하세요. (quit/exit: 종료)")
            return True 
            
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def collect_observations(self) -> dict:
        """모든 센서에서 관찰 데이터 수집"""
        obs = {}
        
        # 1. 비전 데이터
        video_frames = self.vision_collector.get_all_frames()
        # 모든 프레임을 (1, 1, 224, 224, 3)로 맞춤 (해상도 다르면 resize)
        for k, v in video_frames.items():
            # v: (224, 224, 3) 또는 (1, 224, 224, 3) 또는 (H, W, 3) 등
            if v.ndim == 3:
                if v.shape[0] != 224 or v.shape[1] != 224:
                    v = cv2.resize(v, (224, 224))
                v = v[None, None, ...]
            elif v.ndim == 4:
                if v.shape[1] != 224 or v.shape[2] != 224:
                    v[0] = cv2.resize(v[0], (224, 224))
                v = v[None, ...]
            obs[k] = v.astype(np.uint8)
        # 2. 로봇 상태
        robot_states = self.state_collector.get_all_states()
        obs.update(robot_states)
        # 3. 텍스트 명령 (대기)
        text_command = None
        while not text_command and self.running:
            command_data = self.text_collector.get_latest_command()
            if command_data:
                text_command = command_data.get('annotation.language.instruction')
            else:
                time.sleep(0.1)
        if text_command:
            obs['annotation.language.instruction'] = text_command
        # === GR00T 포맷 맞추기 ===
        # 1. video: 이미 위에서 (1,1,224,224,3)으로 맞춤
        # 2. state: metadata 순서대로 concat
        try:
            state_vec = np.concatenate([
                obs['state.right_arm_eef_pos'],
                obs['state.right_arm_eef_quat'],
                obs['state.right_gripper_qpos'],
                obs['state.left_arm_eef_pos'],
                obs['state.left_arm_eef_quat'],
                obs['state.left_gripper_qpos'],
            ], axis=0)
            obs['state'] = state_vec[None, None, :].astype(np.float32)
            # state.* key 삭제
            for k in list(obs.keys()):
                if k.startswith('state.') and k != 'state':
                    del obs[k]
        except Exception as e:
            print(f"[DEBUG] state concat error: {e}")
        # 3. language
        if 'annotation.language.instruction' in obs:
            obs['language'] = np.array([obs['annotation.language.instruction']])
        # 디버깅: obs의 타입/shape 출력
        for k in list(obs.keys()):
            v = obs[k]
            if isinstance(v, (int, float)):
                print(f"[DEBUG] {k}: int/float({v}), numpy array로 변환")
                obs[k] = np.array([v], dtype=np.float32)
            elif isinstance(v, np.ndarray):
                print(f"[DEBUG] {k}: np.ndarray, dtype={v.dtype}, shape={v.shape}")
            elif isinstance(v, str):
                print(f"[DEBUG] {k}: str({v})")
            else:
                print(f"[DEBUG] {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
        return obs, text_command
    
    def run(self):
        """메인 실행 루프"""
        self.running = True
        
        try:
            while self.running:
                # 1. 관찰 데이터 수집
                obs, text_command = self.collect_observations()
                
                if not text_command:
                    continue
                    
                # 종료 명령 체크
                if text_command.lower() in ['quit', 'exit']:
                    logger.info("종료 명령 수신")
                    break
                    
                logger.info(f"명령: '{text_command}'")
                
                # 2. GR00T 추론
                try:
                    # [R2] torch.inference_mode() 사용
                    with torch.inference_mode():
                        self.policy.model.eval()
                        device_obs = {}
                        for k, v in obs.items():
                            # [R0] .cpu() 강제 삭제
                            if isinstance(v, torch.Tensor):
                                device_obs[k] = v
                            else:
                                device_obs[k] = v
                        action_dict = self.policy.get_action(device_obs)
                    
                    if action_dict:
                        # step별로 분리
                        if 'action.right_arm_eef_pos' in action_dict:
                            batch = 0
                            horizon = action_dict['action.right_arm_eef_pos'].shape[1]
                            step_tokens = []
                            for t in range(horizon):
                                step_token = {}
                                for k, v in action_dict.items():
                                    step_token[k] = v[batch, t]
                                step_tokens.append(step_token)
                            trajectory = []
                            for token in step_tokens:
                                step_traj = self.action_decoder.decode_action(token)
                                if step_traj:
                                    trajectory.append(step_traj[0])
                            logger.info(f"Trajectory 생성: {len(trajectory)} steps")
                            # 4. 실행 또는 dry-run
                            if self.dry_run:
                                logger.info("[DRY-RUN] Trajectory:")
                                for i, step in enumerate(trajectory):
                                    logger.info(f"  Step {i}: left={step.get('left')}, right={step.get('right')}")
                            else:
                                if self.hardware_bridge:
                                    for i, step in enumerate(trajectory):
                                        threads = []
                                        for arm_name in ['left', 'right']:
                                            eef_cmd = step.get(arm_name)
                                            if eef_cmd is not None:
                                                t = threading.Thread(target=self.hardware_bridge.send_arm_command, args=(arm_name, eef_cmd))
                                                t.start()
                                                threads.append(t)
                                        for t in threads:
                                            t.join()
                                        logger.info(f"[SEND] Step {i} (both arms sent)")
                                        time.sleep(0.1)
                                else:
                                    logger.warning("Hardware bridge not initialized, cannot send commands.")
                        else:
                            # 기존 방식 (혹시 모를 fallback)
                            trajectory = self.action_decoder.decode_action(action_dict)
                            if trajectory:
                                logger.info(f"Trajectory 생성: {len(trajectory)} steps")
                                if self.dry_run:
                                    logger.info("[DRY-RUN] Trajectory:")
                                    for i, step in enumerate(trajectory[:3]):
                                        logger.info(f"  Step {i}: left={step.get('left')}, right={step.get('right')}")
                                else:
                                    if self.trajectory_executor:
                                        success = self.trajectory_executor.execute_trajectory(trajectory)
                                        if success:
                                            logger.info("✅ Trajectory 실행 완료")
                                        else:
                                            logger.warning("⚠️ Trajectory 실행 중 문제 발생")
                            else:
                                logger.warning("액션 디코딩 실패")
                    else:
                        logger.warning("GR00T 추론 실패")
                        
                except Exception as e:
                    logger.error(f"처리 중 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # 다음 명령 대기
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("\n사용자 중단")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """시스템 종료"""
        logger.info("\n시스템 종료 중...")
        self.running = False
        
        # 역순으로 종료
        if self.trajectory_executor:
            self.trajectory_executor.stop_execution()
            
        if self.state_collector:
            self.state_collector.stop_all_collectors()
            
        if self.text_collector:
            self.text_collector.stop_collection()
            
        if self.vision_collector:
            self.vision_collector.stop_all_cameras()
            
        if self.safety_manager:
            self.safety_manager.stop_monitoring()
            
        if self.hardware_bridge:
            self.hardware_bridge.disconnect()
            
        logger.info("✅ 시스템 종료 완료")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GR00T End-to-End Robot Control")
    parser.add_argument("--dry-run", action="store_true", help="로봇 실제 제어 없이 테스트")
    parser.add_argument("--mock-vision", action="store_true", help="Mock 비전 데이터 사용")
    args = parser.parse_args()
    
    # 시그널 핸들러
    def signal_handler(signum, frame):
        logger.warning("긴급 정지 신호!")
        if system.hardware_bridge:
            system.hardware_bridge.emergency_stop()
        system.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 시스템 생성 및 실행
    system = GR00TRobotSystem(dry_run=args.dry_run, mock_vision=args.mock_vision)
    
    if system.initialize():
        system.run()
    else:
        logger.error("시스템 초기화 실패")
        sys.exit(1)


if __name__ == "__main__":

    main()