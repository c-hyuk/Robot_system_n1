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
from typing import Optional

sys.path.append(str(Path(__file__).parent))

# 실제 시스템 import
from model.gr00t_interface import DualPiperGR00TInterface
from model.action_decoder import create_action_decoder
from data.unified_data_pipeline import UnifiedDataPipeline, CollectionConfig
from communication.hardware_bridge import PiperHardwareBridge
from control.safety_manager import SafetyManager

# =====================
# Argument/Logging
# =====================
def parse_arguments():
    parser = argparse.ArgumentParser(description="GR00T End-to-End Terminal")
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.5-3B", help="GR00T 모델 경로")
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
    # action_decoder = create_action_decoder(
    #     embodiment_name=args.embodiment,
    #     execution_mode=args.execution_mode
    # )

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

    while True:
        try:
            user_input = input("\n명령 입력 > ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("[종료]")
                break
            if user_input.lower() in ["emergency", "disable"]:
                if hardware_bridge:
                    hardware_bridge.emergency_stop()
                safety_manager.handle_emergency()
                continue
            # 1. 최신 데이터 취합 (vision/state/text)
            # 최신 state/vision/text 안전 취득
            robot_data = None
            vision_data = None
            state_collector = getattr(getattr(pipeline.collection_layer, 'state_collector', None), 'left_collector', None)
            if state_collector is not None:
                robot_data = getattr(state_collector, 'last_state', None)
            vision_collector = getattr(pipeline.collection_layer, 'vision_collector', None)
            if vision_collector is not None and hasattr(vision_collector, 'get_latest'):
                vision_data = vision_collector.get_latest()
            # 2. GR00T 추론 (텍스트+비전+상태)
            # 관찰 데이터 dict 구성
            observations = {}
            if vision_data is not None:
                if isinstance(vision_data, dict):
                    observations.update(vision_data)
                else:
                    observations['video'] = vision_data
            if robot_data is not None:
                observations['state'] = robot_data
            observations['language'] = user_input
            if hasattr(gr00t, 'get_action_from_observations'):
                action_token = gr00t.get_action_from_observations(observations)
            else:
                action_token = None
            # 3. Action token → 로봇 명령 변환 (주석처리)
            # robot_cmds = action_decoder.decode_action(action_token) if action_token is not None else None
            # 4. 실제 로봇 제어 or dry-run 출력
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
                # print(f"[DRY-RUN] Robot commands: {robot_cmds}")
            # else:
            #     if hardware_bridge and robot_cmds is not None:
            #         # robot_cmds가 dict of arm_name: command 형태일 수 있음
            #         if isinstance(robot_cmds, dict):
            #             for arm_name, cmd in robot_cmds.items():
            #                 if cmd is not None:
            #                     hardware_bridge.send_arm_command(arm_name, cmd)
            #             print(f"[로봇 제어] 명령 전송 완료: {robot_cmds}")
        except KeyboardInterrupt:
            print("\n[사용자 강제 종료]")
            if hardware_bridge:
                hardware_bridge.emergency_stop()
            safety_manager.handle_emergency()
            break
        except Exception as e:
            logger.error(f"[오류] {e}")

    # 종료 처리
    pipeline.stop()
    safety_manager.stop_monitoring()
    if hardware_bridge:
        hardware_bridge.disconnect()
    print("[프로그램 종료 및 리소스 정리 완료]")

if __name__ == "__main__":
    main() 