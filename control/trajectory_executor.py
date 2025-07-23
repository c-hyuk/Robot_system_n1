"""
Trajectory Executor
GR00T에서 생성된 trajectory를 실제 로봇에 안전하게 실행하는 시스템
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from model.action_decoder import EEFCommand
from communication.hardware_bridge import PiperHardwareBridge
from control.safety_manager import SafetyManager


@dataclass
class TrajectoryStep:
    """Trajectory의 단일 step"""
    timestamp: float
    left_arm: Optional[EEFCommand] = None
    right_arm: Optional[EEFCommand] = None


@dataclass
class ExecutionConfig:
    """실행 설정"""
    execution_frequency: float = 10.0  # Hz
    blending_alpha: float = 0.5  # trajectory 간 blending 계수
    step_blending_alpha: float = 0.7  # step 간 blending 계수
    safety_check_interval: float = 0.1  # 안전 검사 간격
    max_execution_time: float = 30.0  # 최대 실행 시간 (초)


class TrajectoryExecutor:
    """Trajectory 실행 및 blending 관리"""
    
    def __init__(self, 
                 hardware_bridge: PiperHardwareBridge,
                 safety_manager: SafetyManager,
                 config: Optional[ExecutionConfig] = None):
        
        self.hardware_bridge = hardware_bridge
        self.safety_manager = safety_manager
        self.config = config or ExecutionConfig()
        
        self.logger = logging.getLogger("TrajectoryExecutor")
        self.dt = 1.0 / self.config.execution_frequency
        
        # 이전 trajectory의 마지막 step 저장
        self.previous_trajectory_end: Optional[Dict[str, EEFCommand]] = None
        
        # 실행 상태
        self.is_executing = False
        self.current_step = 0
        self.total_steps = 0
    
    def execute_trajectory(self, 
                          trajectory: List[Dict[str, EEFCommand]], 
                          dry_run: bool = False) -> bool:
        """
        Trajectory 실행
        
        Args:
            trajectory: EEFCommand 리스트
            dry_run: 실제 실행하지 않고 시뮬레이션만
            
        Returns:
            성공 여부
        """
        if not trajectory:
            self.logger.warning("Empty trajectory")
            return False
        
        self.is_executing = True
        self.current_step = 0
        self.total_steps = len(trajectory)
        
        try:
            self.logger.info(f"Starting trajectory execution: {len(trajectory)} steps")
            
            # 1. 이전 trajectory와의 blending (첫 번째 step)
            if self.previous_trajectory_end and trajectory:
                blended_first_step = self._blend_trajectory_transition(
                    self.previous_trajectory_end, 
                    trajectory[0]
                )
                if not dry_run:
                    self._execute_single_step(blended_first_step)
                time.sleep(self.dt)
            
            # 2. Trajectory steps 순차 실행
            for i, step in enumerate(trajectory):
                if not self.is_executing:
                    break
                
                # 안전 검사
                if not self._safety_check(step):
                    self.logger.error("Safety check failed")
                    return False
                
                # Step 간 blending (이전 step과)
                if i > 0:
                    blended_step = self._blend_consecutive_steps(
                        trajectory[i-1], step
                    )
                else:
                    blended_step = step
                
                # 실행
                if not dry_run:
                    self._execute_single_step(blended_step)
                
                self.current_step = i + 1
                time.sleep(self.dt)
            
            # 3. 마지막 step 저장 (다음 trajectory blending용)
            if trajectory:
                self.previous_trajectory_end = trajectory[-1]
            
            self.logger.info("Trajectory execution completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Trajectory execution failed: {e}")
            return False
        finally:
            self.is_executing = False
    
    def _blend_trajectory_transition(self, 
                                   prev_end: Dict[str, EEFCommand], 
                                   new_start: Dict[str, EEFCommand]) -> Dict[str, EEFCommand]:
        """Trajectory 간 전환 blending"""
        blended = {}
        
        for arm_name in ['left', 'right']:
            prev_cmd = prev_end.get(arm_name)
            new_cmd = new_start.get(arm_name)
            
            if prev_cmd is not None and new_cmd is not None:
                blended[arm_name] = self._blend_eef_commands(
                    prev_cmd, new_cmd, self.config.blending_alpha
                )
            elif new_cmd is not None:
                blended[arm_name] = new_cmd
            elif prev_cmd is not None:
                blended[arm_name] = prev_cmd
        
        return blended
    
    def _blend_consecutive_steps(self, 
                                prev_step: Dict[str, EEFCommand], 
                                curr_step: Dict[str, EEFCommand]) -> Dict[str, EEFCommand]:
        """연속된 step 간 blending"""
        blended = {}
        
        for arm_name in ['left', 'right']:
            prev_cmd = prev_step.get(arm_name)
            curr_cmd = curr_step.get(arm_name)
            
            if prev_cmd is not None and curr_cmd is not None:
                blended[arm_name] = self._blend_eef_commands(
                    prev_cmd, curr_cmd, self.config.step_blending_alpha
                )
            elif curr_cmd is not None:
                blended[arm_name] = curr_cmd
            elif prev_cmd is not None:
                blended[arm_name] = prev_cmd
        
        return blended
    
    def _blend_eef_commands(self, 
                           cmd1: EEFCommand, 
                           cmd2: EEFCommand, 
                           alpha: float) -> EEFCommand:
        """EEF 명령 blending"""
        return EEFCommand(
            timestamp=cmd2.timestamp,
            position=alpha * cmd2.position + (1-alpha) * cmd1.position,
            rotation=alpha * cmd2.rotation + (1-alpha) * cmd1.rotation,
            gripper=alpha * cmd2.gripper + (1-alpha) * cmd1.gripper
        )
    
    def _execute_single_step(self, step: Dict[str, EEFCommand]) -> None:
        """단일 step 실행"""
        for arm_name, cmd in step.items():
            if cmd is not None:
                self.hardware_bridge.send_arm_command(arm_name, cmd)
    
    def _safety_check(self, step: Dict[str, EEFCommand]) -> bool:
        """안전 검사"""
        try:
            # Safety manager를 통한 검증
            if not self.safety_manager.validate_command(step):
                return False
            
            # 추가적인 검사들...
            for arm_name, cmd in step.items():
                if cmd is not None:
                    # 위치 제한 검사
                    if not self._check_position_limits(cmd.position):
                        self.logger.warning(f"{arm_name} position out of limits")
                        return False
                    
                    # 회전 제한 검사
                    if not self._check_rotation_limits(cmd.rotation):
                        self.logger.warning(f"{arm_name} rotation out of limits")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety check error: {e}")
            return False
    
    def _check_position_limits(self, position: np.ndarray) -> bool:
        """위치 제한 검사"""
        workspace_limits = {
            'x': (-0.8, 0.8),
            'y': (-0.8, 0.8),
            'z': (0.0, 1.2)
        }
        
        for i, (axis, (min_val, max_val)) in enumerate(workspace_limits.items()):
            if not (min_val <= position[i] <= max_val):
                return False
        
        return True
    
    def _check_rotation_limits(self, rotation: np.ndarray) -> bool:
        """회전 제한 검사"""
        # ±180도 제한
        for angle in rotation:
            if not (-np.pi <= angle <= np.pi):
                return False
        
        return True
    
    def stop_execution(self) -> None:
        """실행 중지"""
        self.is_executing = False
        self.logger.info("Trajectory execution stopped")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """실행 상태 반환"""
        return {
            'is_executing': self.is_executing,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress': self.current_step / self.total_steps if self.total_steps > 0 else 0.0
        }
    
    def reset(self) -> None:
        """상태 초기화"""
        self.previous_trajectory_end = None
        self.is_executing = False
        self.current_step = 0
        self.total_steps = 0