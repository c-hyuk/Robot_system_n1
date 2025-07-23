#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR00T End-to-End 통합 터미널 (실제 시스템 연동)
- 텍스트 명령, 비전(mock/실제), 로봇 상태를 입력받아 GR00T 추론
- Action token 생성 및 실제 로봇 제어 또는 dry-run 출력
- 긴급 정지/Disable 등 안전 기능 내장
"""
import sys
import time
import signal
import argparse
import logging
from pathlib import Path
import numpy as np  # 추가: mock observation 생성용
import torch
import json
from typing import Optional

sys.path.append(str(Path(__file__).parent))

# 실제 시스템 import
from model.gr00t_interface import DualPiperGR00TInterface
from model.action_decoder import create_action_decoder
from data.unified_data_pipeline import UnifiedDataPipeline, CollectionConfig
from communication.hardware_bridge import PiperHardwareBridge
from control.safety_manager import SafetyManager
from control.trajectory_executor import TrajectoryExecutor, ExecutionConfig

# =====================
# Argument/Logging
# =====================
def parse_arguments():
    parser = argparse.ArgumentParser(description="GR00T End-to-End Terminal")
    parser.add_argument("--model-path", type=str, default="/home/rosota/GR00T-N1-2B", help="GR00T N1-2B 모델 폴더 경로")
    parser.add_argument("--mock-vision", action="store_true", help="Vision 입력을 mock 데이터로 대체")
    parser.add_argument("--dry-run", action="store_true", help="로봇을 실제로 움직이지 않고 action token만 출력")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="로깅 레벨")
    parser.add_argument("--left-can", type=str, default="can0", help="Left arm CAN port")
    parser.add_argument("--right-can", type=str, default="can1", help="Right arm CAN port")
    parser.add_argument("--execution-mode", type=str, default="position", choices=["position", "velocity", "trajectory"], help="Action execution mode")
    parser.add_argument("--embodiment", type=str, default="dual_piper_arm", help="로봇 embodiment 이름")
    return parser.parse_args()

def setup_logging(log_level: str):
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

# =====================
# Main Terminal Logic
# =====================
def main():
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger("GR00T-Terminal")
    logger.info("🚀 GR00T End-to-End Terminal 시작")

    # 1. 데이터 파이프라인 (vision/state/text/mock)
    pipeline = UnifiedDataPipeline(
        embodiment_name=args.embodiment,
        config=CollectionConfig(),
        use_mock=args.mock_vision
    )
    pipeline.start()

    # 2. GR00T 인터페이스 (실제 모델)
    gr00t = DualPiperGR00TInterface(
        model_path=args.model_path,
        embodiment_name=args.embodiment,
        use_mock_data=args.mock_vision
    )
    action_decoder = create_action_decoder(
        embodiment_name=args.embodiment
        # execution_mode 인자 제거
    )

    # 4. 로봇 하드웨어 브릿지 (실제 제어, dry-run 시 None)
    hardware_bridge = None
    if not args.dry_run:
        hardware_bridge = PiperHardwareBridge(
            left_can_port=args.left_can,
            right_can_port=args.right_can,
            auto_enable=True,
            gripper_enabled=True
        )
        hardware_bridge.connect()
    
    # 5. Safety Manager (긴급정지 등)
    from config.hardware_config import get_hardware_config
    hw_config = get_hardware_config()
    safety_manager = SafetyManager(hw_config)
    safety_manager.start_monitoring()
    
    # 6. Trajectory Executor (trajectory blending 및 실행)
    trajectory_executor = None
    if not args.dry_run and hardware_bridge:
        trajectory_executor = TrajectoryExecutor(
            hardware_bridge=hardware_bridge,
            safety_manager=safety_manager,
            config=ExecutionConfig(
                execution_frequency=10.0,
                blending_alpha=0.5,
                step_blending_alpha=0.7
            )
        )

    # 긴급 정지 핸들러
    def emergency_handler(signum, frame):
        logger.warning("[EMERGENCY] 시그널 감지! 로봇 즉시 정지/Disable!")
        if hardware_bridge:
            hardware_bridge.emergency_stop()
        safety_manager.handle_emergency()
        sys.exit(1)
    signal.signal(signal.SIGINT, emergency_handler)
    signal.signal(signal.SIGTERM, emergency_handler)

    print("\n==== GR00T End-to-End Terminal ====")
    print("텍스트 명령을 입력하세요. (emergency: 즉시 정지, quit/exit: 종료)")
    print(f"[모드] Vision: {'MOCK' if args.mock_vision else 'REAL'}, Dry-run: {args.dry_run}")

    # 텍스트 수집기 인스턴스 확보
    text_collector = pipeline.collection_layer.text_collector if hasattr(pipeline.collection_layer, 'text_collector') else None
    if text_collector is None:
        from data.collectors.text_collector import TextCollectorManager
        text_collector = TextCollectorManager()
        text_collector.start_collection()

    last_wait_print = 0
    while True:
        try:
            now = time.time()
            # 명령이 없을 때만 1초에 한 번만 출력
            command_data = text_collector.get_latest_command()
            if not command_data:
                if now - last_wait_print > 1.0:
                    print("[DEBUG] Waiting for command...")
                    last_wait_print = now
                time.sleep(0.1)
                continue
            print(f"[DEBUG] Received command_data: {command_data}")
            # 명령어 추출 로직 개선
            if 'processed_command' in command_data:
                user_input = command_data['processed_command']
            elif 'annotation.language.instruction' in command_data:
                user_input = command_data['annotation.language.instruction']
            else:
                print(f"[오류] 명령어 키를 찾을 수 없습니다: {command_data}")
                continue
            print(f"[DEBUG] Processed user_input: {user_input}")
            if user_input.lower() in ["quit", "exit"]:
                print("[종료]")
                break
            if user_input.lower() in ["emergency", "disable"]:
                print("[DEBUG] Emergency/Disable command received. Sending emergency stop.")
                if hardware_bridge:
                    hardware_bridge.emergency_stop()
                safety_manager.handle_emergency()
                continue
            # 1. 올바른 형식의 데이터 수집 (unified_data_pipeline 활용)
            print("[DEBUG] Collecting observations...")
            observations = pipeline.get_groot_input()
            if observations is None:
                print("[경고] 유효한 입력 데이터가 없습니다. (mock observation으로 대체)")
                # metadata.json의 state key 순서에 맞게 mock observation 생성
                observations = create_mock_observations(user_input)
            else:
                # 실제 수집 데이터도 동일하게 state vector로 변환 (metadata.json 순서)
                state_keys = [
                    'state.right_arm_eef_pos', 'state.right_arm_eef_quat', 'state.right_gripper_qpos',
                    'state.left_arm_eef_pos', 'state.left_arm_eef_quat', 'state.left_gripper_qpos'
                ]
                if all(k in observations for k in state_keys):
                    state_vec = np.concatenate([
                        observations['state.right_arm_eef_pos'],
                        observations['state.right_arm_eef_quat'],
                        observations['state.right_gripper_qpos'],
                        observations['state.left_arm_eef_pos'],
                        observations['state.left_arm_eef_quat'],
                        observations['state.left_gripper_qpos'],
                    ], axis=0)
                    observations['state'] = state_vec[None, :]  # (1, 16)
                    # transform 이후 개별 state key 삭제
                    for k in state_keys:
                        if k in observations:
                            del observations[k]
            print(f"[DEBUG] Observations collected: {list(observations.keys())}")
            # 데이터 형식 검증 및 로깅
            print(f"[데이터 검증] Observations keys: {list(observations.keys())}")
            # DualPiperDataConfig 형식 검증
            expected_video_keys = ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"]
            expected_state_keys = [
                "state.right_arm_eef_pos", "state.right_arm_eef_quat", "state.right_gripper_qpos",
                "state.left_arm_eef_pos", "state.left_arm_eef_quat", "state.left_gripper_qpos"
            ]
            expected_language_keys = ["annotation.language.instruction"]
            # 비전 데이터 검증
            video_keys = [k for k in observations.keys() if k.startswith('video.')]
            print(f"  [비전] 발견된 키: {video_keys}")
            for key in expected_video_keys:
                if key in observations:
                    shape = observations[key].shape if hasattr(observations[key], 'shape') else 'N/A'
                    print(f"    ✓ {key}: {shape}")
                else:
                    print(f"    ✗ {key}: 누락")
            # 상태 데이터 검증
            # 개별 state key는 transform 이후 삭제되므로, 검증은 생략하거나 안내 메시지 출력
            print(f"  [상태] (transform 이후 개별 state key는 삭제됨)")
            # 언어 데이터 검증
            language_keys = [k for k in observations.keys() if k.startswith('annotation.')]
            print(f"  [언어] 발견된 키: {language_keys}")
            for key in expected_language_keys:
                if key in observations:
                    value = observations[key][:50] if isinstance(observations[key], str) else str(observations[key])
                    print(f"    ✓ {key}: {value}...")
                else:
                    print(f"    ✗ {key}: 누락")
            # 기타 데이터
            other_keys = [k for k in observations.keys() if not k.startswith(('video.', 'state.', 'annotation.')) and k != 'state']
            if 'state' in observations:
                print(f"  [통합 state 벡터] state: shape={observations['state'].shape}, dtype={getattr(observations['state'], 'dtype', type(observations['state']))}")
            if other_keys:
                print(f"  [기타] {other_keys}")
                for key in other_keys:
                    value = observations[key]
                    if hasattr(value, 'shape'):
                        print(f"    {key}: shape={value.shape}, dtype={getattr(value, 'dtype', type(value))}")
                    elif isinstance(value, str):
                        print(f"    {key}: {value[:50]}...")
                    else:
                        print(f"    {key}: {type(value)}")
            # 2. GR00T 추론 (텍스트+비전+상태)
            print("[DEBUG] Running GR00T inference...")
            if hasattr(gr00t, 'get_action_from_observations'):
                action_token = gr00t.get_action_from_observations(observations)
            else:
                action_token = None
            print(f"[DEBUG] Action token: {action_token}")
            # 3. Action token → 로봇 명령 변환
            print("[DEBUG] Decoding action token...")
            if action_token is not None:
                robot_cmds = action_decoder.decode_action(action_token)
            else:
                robot_cmds = None
            print(f"[DEBUG] Robot commands: {robot_cmds}")
            # 4. 실제 로봇 제어 or dry-run 출력
            print("[DEBUG] Executing robot command or dry-run...")
            if args.dry_run:
                print(f"[DRY-RUN] Action token:")
                if isinstance(action_token, dict):
                    for k, v in action_token.items():
                        if hasattr(v, 'shape'):
                            print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', type(v))}")
                        else:
                            print(f"  {k}: {type(v)}")
                else:
                    print(action_token)
                if robot_cmds:
                    print(f"[DRY-RUN] Trajectory: {len(robot_cmds)} steps")
                    for i, step in enumerate(robot_cmds[:3]):
                        print(f"  Step {i}: {step}")
            else:
                if trajectory_executor and robot_cmds:
                    print(f"[실행] {len(robot_cmds)} steps trajectory 시작")
                    success = trajectory_executor.execute_trajectory(robot_cmds, dry_run=False)
                    print(f"[DEBUG] Trajectory execution result: {success}")
                    if not success:
                        print("[경고] Trajectory 실행 실패")
                    else:
                        print("[완료] Trajectory 실행 성공")
                elif hardware_bridge and robot_cmds:
                    if isinstance(robot_cmds, list) and len(robot_cmds) > 0:
                        first_step = robot_cmds[0]
                        if isinstance(first_step, dict):
                            for arm_name, cmd in first_step.items():
                                if cmd is not None:
                                    print(f"[DEBUG] Sending arm command to {arm_name}: {cmd}")
                                    hardware_bridge.send_arm_command(arm_name, cmd)
                                    print(f"[실행] {arm_name} arm command sent")
                    elif isinstance(robot_cmds, dict):
                        for arm_name, cmd in robot_cmds.items():
                            if cmd is not None:
                                print(f"[DEBUG] Sending arm command to {arm_name}: {cmd}")
                                hardware_bridge.send_arm_command(arm_name, cmd)
                                print(f"[실행] {arm_name} arm command sent")
                    else:
                        print("[경고] 유효한 trajectory 명령이 없습니다.")
                else:
                    print("[경고] 유효한 trajectory 명령이 없습니다.")
        except KeyboardInterrupt:
            print("\n[사용자 강제 종료]")
            if trajectory_executor:
                trajectory_executor.stop_execution()
            if hardware_bridge:
                hardware_bridge.emergency_stop()
            safety_manager.handle_emergency()
            break
        except Exception as e:
            import traceback
            logger.error(f"[오류] {e}")
            print(f"[오류] {e}")
            traceback.print_exc()

    # 종료 처리
    pipeline.stop()
    if trajectory_executor:
        trajectory_executor.stop_execution()
    safety_manager.stop_monitoring()
    if hardware_bridge:
        hardware_bridge.disconnect()
    print("[프로그램 종료 및 리소스 정리 완료]")

def create_mock_observations(user_input: str) -> dict:
    # metadata.json의 state key 순서에 맞게 mock observation 생성 (N1-2B 기준)
    obs = {
        'video.left_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        'video.right_wrist_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        'video.front_view': np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        'state.right_arm_eef_pos': np.random.uniform(-0.2, 0.2, (3,)).astype(np.float32),
        'state.right_arm_eef_quat': np.random.uniform(-1, 1, (4,)).astype(np.float32),
        'state.right_gripper_qpos': np.random.uniform(0, 1, (1,)).astype(np.float32),
        'state.left_arm_eef_pos': np.random.uniform(-0.2, 0.2, (3,)).astype(np.float32),
        'state.left_arm_eef_quat': np.random.uniform(-1, 1, (4,)).astype(np.float32),
        'state.left_gripper_qpos': np.random.uniform(0, 1, (1,)).astype(np.float32),
        'annotation.language.instruction': user_input.strip()
    }
    # (1, 16) state vector로 합치기 (metadata.json 순서)
    state_vec = np.concatenate([
        obs['state.right_arm_eef_pos'],
        obs['state.right_arm_eef_quat'],
        obs['state.right_gripper_qpos'],
        obs['state.left_arm_eef_pos'],
        obs['state.left_arm_eef_quat'],
        obs['state.left_gripper_qpos'],
    ], axis=0)
    obs['state'] = state_vec[None, :]  # (1, 16)
    # shape 검증 출력
    print("[MOCK 검증] mock observation shapes:")
    for k, v in obs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', type(v))}")
        else:
            print(f"  {k}: {type(v)}")
    return obs

if __name__ == "__main__":
    main() 