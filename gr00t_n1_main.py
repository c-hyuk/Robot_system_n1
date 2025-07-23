#!/usr/bin/env python3
"""
GR00T N1-2B End-to-End Main
- 실시간 Vision/State/Text 수집
- 데이터 transform 및 추론
- 액션 토큰을 로봇에 실시간 전달
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import time
import logging
import numpy as np
import torch
import argparse
import json

GR00T_N1_2B_PATH = str(Path(__file__).parent.parent / "GR00T-N1-2B")
sys.path.append(GR00T_N1_2B_PATH)
sys.path.append(str(Path(__file__).parent))

from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.embodiment_tags import EmbodimentTag
from model.action_decoder import EEFCommand, create_action_decoder
from data.collectors.vision_collector import VisionCollectorManager
from data.collectors.text_collector import TextCollectorManager
from control.robot_controller import RobotController
from control.safety_manager import SafetyManager
from communication.hardware_bridge import PiperHardwareBridge as HardwareBridge
from config.hardware_config import get_hardware_config
from gr00t.data.schema import DatasetMetadata


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    parser = argparse.ArgumentParser(description="GR00T N1-2B End-to-End Main")
    parser.add_argument("--dry-run", action="store_true", help="로봇에 실제 명령을 보내지 않고 액션 토큰만 출력")
    parser.add_argument("--mock-vision", action="store_true", help="Vision 입력을 mock 데이터로 대체")
    return parser.parse_args()


def collect_vision(vision_manager=None, mock_vision=False):
    if mock_vision or vision_manager is None:
        obs = {
            'video.left_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
            'video.right_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
            'video.front_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        }
        return obs
    # VisionCollectorManager에서 프레임 수집
    frames = vision_manager.get_all_frames()
    obs = {}
    for k, v in frames.items():
        # (224,224,3)로 리사이즈 및 RGB 변환 보장
        if v.shape[1] != 224 or v.shape[0] != 224:
            import cv2
            v = cv2.resize(v, (224, 224))
        if v.shape[-1] == 3 and v.dtype != np.uint8:
            v = v.astype(np.uint8)
        obs[k] = v[None, ...]  # (1,224,224,3)
    return obs


def collect_state(hardware_bridge):
    # 실제 하드웨어 브릿지에서 로봇 상태 수집
    # left/right arm state dict 병합
    left_state = hardware_bridge.get_arm_state('left')
    right_state = hardware_bridge.get_arm_state('right')
    obs = {}
    if left_state:
        obs['state.left_arm_eef_pos'] = left_state.get('left_arm_eef_pos', np.zeros(3, dtype=np.float32))
        obs['state.left_arm_eef_quat'] = left_state.get('left_arm_eef_quat', np.zeros(4, dtype=np.float32))
        obs['state.left_gripper_qpos'] = np.array([left_state.get('left_gripper_qpos', 0.0)], dtype=np.float32)
    else:
        obs['state.left_arm_eef_pos'] = np.zeros(3, dtype=np.float32)
        obs['state.left_arm_eef_quat'] = np.zeros(4, dtype=np.float32)
        obs['state.left_gripper_qpos'] = np.zeros(1, dtype=np.float32)
    if right_state:
        obs['state.right_arm_eef_pos'] = right_state.get('right_arm_eef_pos', np.zeros(3, dtype=np.float32))
        obs['state.right_arm_eef_quat'] = right_state.get('right_arm_eef_quat', np.zeros(4, dtype=np.float32))
        obs['state.right_gripper_qpos'] = np.array([right_state.get('right_gripper_qpos', 0.0)], dtype=np.float32)
    else:
        obs['state.right_arm_eef_pos'] = np.zeros(3, dtype=np.float32)
        obs['state.right_arm_eef_quat'] = np.zeros(4, dtype=np.float32)
        obs['state.right_gripper_qpos'] = np.zeros(1, dtype=np.float32)
    return obs


def collect_text(text_collector):
    # 실제 터미널에서 명령 입력
    # return text_collector.get_text()
    return 'test'  # 더미 입력


def transform_data(raw_obs, modality_transform):
    # state vector 합치기 (metadata.json 순서)
    state_vec = np.concatenate([
        raw_obs['state.right_arm_eef_pos'],
        raw_obs['state.right_arm_eef_quat'],
        raw_obs['state.right_gripper_qpos'],
        raw_obs['state.left_arm_eef_pos'],
        raw_obs['state.left_arm_eef_quat'],
        raw_obs['state.left_gripper_qpos'],
    ], axis=0)
    obs = dict(raw_obs)  # shallow copy
    obs['state'] = state_vec[None, :]  # (1, 16)
    # state.* key는 모두 삭제 (state만 남김)
    for k in list(obs.keys()):
        if k.startswith('state.') and k != 'state':
            del obs[k]
    print("[DEBUG] obs keys and shapes before transform:")
    for k, v in obs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', type(v))}")
        else:
            print(f"  {k}: {type(v)}")
    return modality_transform(obs)


def run_inference(policy, transformed_obs):
    with torch.no_grad():
        policy.model.eval()
        device_obs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in transformed_obs.items()}
        return policy.get_action(device_obs)


def execute_action(hardware_bridge, action_token, dry_run=False):
    # action_token: dict, 예시: {'left': {...}, 'right': {...}}
    for arm_name in ['left', 'right']:
        cmd_dict = action_token.get(arm_name)
        if cmd_dict is None:
            continue
        # EEFCommand로 변환 (trajectory_executor.py 참고)
        try:
            eef_cmd = EEFCommand(
                timestamp=cmd_dict.get('timestamp', time.time()),
                position=np.array(cmd_dict['position'], dtype=np.float32),
                rotation=np.array(cmd_dict['rotation'], dtype=np.float32),
                gripper=cmd_dict.get('gripper', 0.5)
            )
        except Exception as e:
            print(f"[ERROR] Invalid action_token for {arm_name}: {e}")
            continue
        if dry_run:
            print(f"[DRY-RUN] {arm_name} EEFCommand: {eef_cmd}")
        else:
            try:
                hardware_bridge.send_arm_command(arm_name, eef_cmd)
            except Exception as e:
                print(f"[ERROR] Failed to send {arm_name} command: {e}")


def main():
    args = parse_arguments()
    setup_logging()
    logger = logging.getLogger("GR00T-N1-2B-Main")
    logger.info(f"🚀 GR00T N1-2B End-to-End Main 시작 (dry-run={args.dry_run}, mock-vision={args.mock_vision})")

    # 1. 하드웨어/로봇/텍스트 collector 초기화
    hardware_bridge = HardwareBridge()
    robot_controller = RobotController(hardware_bridge)
    hw_config = get_hardware_config()
    safety_manager = SafetyManager(hw_config)

    # 텍스트 입력 수집기 초기화
    text_collector = TextCollectorManager()
    text_collector.start_collection()

    # VisionCollectorManager 준비
    vision_manager = None
    if not args.mock_vision:
        vision_manager = VisionCollectorManager(use_mock=False)
        vision_manager.start_all_cameras()

    # 2. 모델/transform 로드
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

    policy = Gr00tPolicy(
        model_path=GR00T_N1_2B_PATH,
        embodiment_tag=embodiment_name,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("모델 및 transform 로드 완료")

    # 3. 액션 디코더 준비
    action_decoder = create_action_decoder(embodiment_name)

    print("\n==== GR00T N1-2B End-to-End Main ====")
    print(f"터미널에서 자연어 명령을 입력하세요. (quit/exit: 종료) [dry-run={args.dry_run}, mock-vision={args.mock_vision}]")

    # 4. 메인 루프
    try:
        while True:
            # 1. 데이터 수집
            vision_obs = collect_vision(vision_manager=vision_manager, mock_vision=args.mock_vision)
            state_obs = collect_state(hardware_bridge)
            # 실제 텍스트 입력 받기 (명령이 들어올 때까지 대기)
            text_input = None
            while not text_input:
                text_data = text_collector.get_latest_command()
                text_input = text_data.get('annotation.language.instruction', None)
                if not text_input:
                    time.sleep(0.1)
            if text_input.lower() in ["quit", "exit"]:
                print("[종료]")
                break

            # 2. 통합 observation 생성
            raw_obs = {**vision_obs, **state_obs, 'annotation.language.instruction': text_input}

            # 3. transform
            try:
                transformed_obs = transform_data(raw_obs, modality_transform)
            except Exception as e:
                logger.error(f"[Transform 오류] {e}")
                import traceback
                traceback.print_exc()
                continue

            # 4. 추론
            try:
                action_token = run_inference(policy, transformed_obs)
            except Exception as e:
                logger.error(f"[추론 오류] {e}")
                import traceback
                traceback.print_exc()
                continue

            # 5. 액션 토큰 → trajectory 변환 (smoothing/blending 포함)
            try:
                trajectory = action_decoder.decode_action(action_token)
                if not trajectory:
                    logger.warning("디코딩된 trajectory가 없습니다.")
                    continue
            except Exception as e:
                logger.error(f"[액션 디코딩 오류] {e}")
                import traceback
                traceback.print_exc()
                continue

            # 6. trajectory 실행 (각 step을 순차적으로 로봇에 전달)
            for step in trajectory:
                for arm_name in ['left', 'right']:
                    eef_cmd = step.get(arm_name)
                    if eef_cmd is None:
                        continue
                    if args.dry_run:
                        print(f"[DRY-RUN] {arm_name} EEFCommand: {eef_cmd}")
                    else:
                        try:
                            hardware_bridge.send_arm_command(arm_name, eef_cmd)
                        except Exception as e:
                            print(f"[ERROR] Failed to send {arm_name} command: {e}")
                time.sleep(0.1)  # trajectory_executor의 dt와 맞추기

            # 추론 사이에 0.5초 대기
            import time
            time.sleep(0.5)

            # 7. 로깅/모니터링
            logger.info(f"Trajectory executed: {len(trajectory)} steps")

    except KeyboardInterrupt:
        logger.info("사용자 종료 요청. 시스템 종료.")
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vision_manager is not None:
            vision_manager.stop_all_cameras()
        if text_collector is not None:
            text_collector.stop_collection()

if __name__ == "__main__":
    main() 