#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper Single Arm Terminal Controller
- CAN 포트 선택 및 통신 확인
- [i] 로봇 초기화 (Enable, 홈 포지션, 슬레이브 모드)
- [1] 현재 joint 값 출력
- [2] joint 각도 입력 → 이동
- 터미널 메뉴 기반
"""
import sys
import time
import threading
import subprocess

try:
    sys.path.append("../piper_py/piper_sdk")
    from piper_sdk import C_PiperInterface_V2
    PIPER_SDK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Piper SDK import 실패: {e}")
    PIPER_SDK_AVAILABLE = False
    C_PiperInterface_V2 = None

def find_can_ports():
    """사용 가능한 CAN 포트 목록 반환"""
    result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
    ports = []
    for line in result.stdout.split('\n'):
        if 'can' in line and ':' in line:
            port = line.split(':')[1].strip()
            ports.append(port)
    return ports

class PiperSingleArmTerminal:
    def __init__(self, can_port="can0"):
        if not PIPER_SDK_AVAILABLE or C_PiperInterface_V2 is None:
            print("Piper SDK not available. 종료합니다.")
            sys.exit(1)
        self.can_port = can_port
        self.robot = C_PiperInterface_V2(
            can_name=can_port,
            judge_flag=False,
            can_auto_init=True,
            dh_is_offset=1,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True
        )
        self.robot.ConnectPort()
        time.sleep(1)

    def initialize_robot(self):
        print("[로봇 초기화: EnableArm, 홈 포지션, 슬레이브 모드]")
        try:
            if hasattr(self.robot, 'EnableArm'):
                self.robot.EnableArm()
                time.sleep(0.2)
            if hasattr(self.robot, 'MotionCtrl_2'):
                self.robot.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=50)
                time.sleep(0.1)
            if hasattr(self.robot, 'JointCtrl'):
                self.robot.JointCtrl(0, 0, 0, 0, 0, 0)
                time.sleep(0.5)
            if hasattr(self.robot, 'MasterSlaveConfig'):
                self.robot.MasterSlaveConfig(0xFC, 0, 0, 0)
                print("[슬레이브 모드로 전환 완료]")
            print("[초기화 완료]")
        except Exception as e:
            print(f"[초기화 오류] {e}")

    def print_joint_values(self):
        print("[현재 Joint 값 출력]")
        try:
            joint = self.robot.GetArmJointMsgs()
            if joint:
                joint_vals = []
                for i in range(1, 7):
                    val = getattr(joint.joint_state, f'joint_{i}', None)
                    if val is not None:
                        joint_vals.append(val / 1000.0)  # 0.001deg → deg
                print(f"Joint(deg): {[f'{v:.2f}' for v in joint_vals]}")
            else:
                print("[오류] Joint 상태를 읽을 수 없습니다.")
        except Exception as e:
            print(f"[오류] {e}")

    def move_to_joint_position(self):
        print("\n[관절 위치 이동 - 'stop' 입력시 중단]")
        while True:
            try:
                user = input("목표 joint position 입력 (j1 j2 j3 j4 j5 j6, 단위: deg) 또는 'stop': ")
                if user.strip().lower() == 'stop':
                    print("[정지]")
                    break
                vals = user.strip().split()
                if len(vals) != 6:
                    print("6개 값(j1~j6, deg)을 입력하세요.")
                    continue
                joints_deg = [float(v) for v in vals]
                joints = [int(j*1000) for j in joints_deg]  # 0.001deg 단위
                if hasattr(self.robot, 'MotionCtrl_2'):
                    self.robot.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=50)
                    time.sleep(0.1)
                if hasattr(self.robot, 'JointCtrl'):
                    self.robot.JointCtrl(*joints)
                    print(f"[JointCtrl 명령 전송: {joints}]")
                else:
                    print("[오류] JointCtrl 메서드가 없습니다.")
            except KeyboardInterrupt:
                print("[사용자 중단]")
                break
            except Exception as e:
                print(f"[오류] {e}")

    def move_to_ee_position(self):
        print("\n[End-Effector(EE) 위치 이동 - 'stop' 입력시 중단]")
        while True:
            try:
                user = input("목표 EE position 입력 (X Y Z RX RY RZ, 단위: mm deg deg deg deg deg) 또는 'stop': ")
                if user.strip().lower() == 'stop':
                    print("[정지]")
                    break
                vals = user.strip().split()
                if len(vals) != 6:
                    print("6개 값(X Y Z RX RY RZ)을 입력하세요.")
                    continue
                # X, Y, Z: mm → 0.001mm, RX, RY, RZ: deg → 0.001deg
                X = int(float(vals[0]) * 1000)
                Y = int(float(vals[1]) * 1000)
                Z = int(float(vals[2]) * 1000)
                RX = int(float(vals[3]) * 1000)
                RY = int(float(vals[4]) * 1000)
                RZ = int(float(vals[5]) * 1000)
                # 직교좌표 모드로 전환
                if hasattr(self.robot, 'MotionCtrl_2'):
                    self.robot.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=50)
                    time.sleep(0.1)
                if hasattr(self.robot, 'EndPoseCtrl'):
                    self.robot.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                    print(f"[EndPoseCtrl 명령 전송: X={X}, Y={Y}, Z={Z}, RX={RX}, RY={RY}, RZ={RZ}]")
                else:
                    print("[오류] EndPoseCtrl 메서드가 없습니다.")
            except KeyboardInterrupt:
                print("[사용자 중단]")
                break
            except Exception as e:
                print(f"[오류] {e}")

    def cleanup_robot(self):
        print("[로봇 종료 및 정지 상태로 복귀]")
        try:
            # 홈 포지션 이동
            if hasattr(self.robot, 'JointCtrl'):
                self.robot.JointCtrl(0, 0, 0, 0, 0, 0)
                time.sleep(0.5)
            # DisableArm(7) - 모든 축 비활성화
            if hasattr(self.robot, 'DisableArm'):
                try:
                    self.robot.DisableArm(7)
                except Exception as e:
                    print(f"[DisableArm(7) 오류] {e}")
                time.sleep(0.2)
            print("[로봇 정지 및 비활성화 완료]")
        except Exception as e:
            print(f"[정지/종료 오류] {e}")

    def run(self):
        print("\n==== Piper Single Arm Terminal ====")
        print("i: 로봇 초기화 (Enable, 홈 포지션, 슬레이브 모드)")
        print("1: 현재 Joint 값 출력")
        print("2: Joint 각도 직접 이동")
        print("3: EE(End-Effector) 좌표 직접 이동")
        print("q: 종료")
        try:
            while True:
                mode = input("\n명령 선택 (i/1/2/3/q): ").strip().lower()
                if mode == 'i':
                    self.initialize_robot()
                elif mode == '1':
                    self.print_joint_values()
                elif mode == '2':
                    self.move_to_joint_position()
                elif mode == '3':
                    self.move_to_ee_position()
                elif mode == 'q':
                    self.cleanup_robot()
                    print("[종료]")
                    break
                else:
                    print("잘못된 입력입니다. i, 1, 2, 3, q 중 선택하세요.")
        except KeyboardInterrupt:
            print("\n[사용자 강제 종료 감지]")
        finally:
            self.cleanup_robot()
            print("[프로그램 종료 및 로봇 안전 정지]")

if __name__ == "__main__":
    can_ports = find_can_ports()
    print("사용 가능한 CAN 포트:", can_ports)
    can_port = input(f"사용할 CAN 포트 입력 (기본: can0): ").strip() or "can0"
    if can_port not in can_ports:
        print(f"[경고] {can_port} 포트가 시스템에 없습니다. 계속 진행하려면 연결을 확인하세요.")
    controller = PiperSingleArmTerminal(can_port=can_port)
    controller.run() 