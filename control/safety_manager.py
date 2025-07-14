"""
Safety Manager
로봇 안전성 검사 및 위험 상황 관리
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

from config.hardware_config import get_hardware_config, HardwareConfig
from utils.data_types import SystemConfig


class SafetyLevel(Enum):
    """안전 수준"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class SafetyViolationType(Enum):
    """안전 위반 타입"""
    JOINT_LIMIT = "joint_limit"
    VELOCITY_LIMIT = "velocity_limit"
    ACCELERATION_LIMIT = "acceleration_limit"
    WORKSPACE_LIMIT = "workspace_limit"
    COLLISION_RISK = "collision_risk"
    SINGULARITY = "singularity"
    HARDWARE_ERROR = "hardware_error"
    COMMUNICATION_TIMEOUT = "communication_timeout"


@dataclass
class SafetyViolation:
    """안전 위반 정보"""
    violation_type: SafetyViolationType
    severity: SafetyLevel
    arm_name: str
    joint_index: Optional[int] = None
    current_value: float = 0.0
    limit_value: float = 0.0
    message: str = ""
    timestamp: float = 0.0


@dataclass
class SafetyConfig:
    """안전 설정"""
    # 관절 제한
    enable_joint_limits: bool = True
    joint_limit_margin: float = 0.1  # 라디안 (약 5.7도)
    
    # 속도 제한
    enable_velocity_limits: bool = True
    max_joint_velocity: float = 1.5  # rad/s
    max_effector_velocity: float = 0.8  # m/s
    velocity_limit_margin: float = 0.1  # 10% 마진
    
    # 가속도 제한
    enable_acceleration_limits: bool = True
    max_joint_acceleration: float = 3.0  # rad/s²
    max_effector_acceleration: float = 2.0  # m/s²
    
    # 워크스페이스 제한
    enable_workspace_limits: bool = True
    workspace_bounds: Dict[str, Tuple[float, float]] = None  # {axis: (min, max)}
    
    # 충돌 방지
    enable_collision_check: bool = True
    min_distance_between_arms: float = 0.2  # 미터
    collision_check_frequency: float = 50.0  # Hz
    
    # 특이점 회피
    enable_singularity_check: bool = True
    singularity_threshold: float = 0.01  # 조작도 임계값
    
    # 하드웨어 안전
    enable_hardware_monitoring: bool = True
    max_communication_timeout: float = 0.5  # 초
    max_consecutive_errors: int = 5
    
    # 자동 복구
    enable_auto_recovery: bool = True
    recovery_timeout: float = 5.0  # 초


class SafetyManager:
    """로봇 안전 관리자"""
    
    def __init__(self, hw_config: HardwareConfig, config: Optional[SafetyConfig] = None):
        """
        안전 관리자 초기화
        
        Args:
            hw_config: 하드웨어 설정
            config: 안전 설정
        """
        self.hw_config = hw_config
        self.config = config or SafetyConfig()
        
        # 기본 워크스페이스 설정 (설정되지 않은 경우)
        if self.config.workspace_bounds is None:
            self.config.workspace_bounds = {
                'x': (-1.0, 1.0),   # 좌우 1미터
                'y': (-1.0, 1.0),   # 앞뒤 1미터  
                'z': (0.0, 1.5)     # 상하 1.5미터
            }
        
        # 안전 상태 관리
        self.current_safety_level = SafetyLevel.SAFE
        self.active_violations: List[SafetyViolation] = []
        self.violation_history: List[SafetyViolation] = []
        self.max_history_length = 1000
        
        # 팔별 상태 추적
        self.arm_states: Dict[str, Dict[str, Any]] = {}
        self.previous_positions: Dict[str, List[float]] = {}
        self.previous_velocities: Dict[str, List[float]] = {}
        self.position_history: Dict[str, List[List[float]]] = {}
        
        # 타이밍 관리
        self.last_update_time: Dict[str, float] = {}
        self.last_check_time = 0.0
        
        # 안전 모니터링 스레드
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # 긴급 정지 상태
        self.emergency_stop_active = False
        self.emergency_stop_reason = ""
        
        # 통계
        self.total_checks = 0
        self.total_violations = 0
        self.start_time = time.time()
        
        # 로깅
        self.logger = logging.getLogger("SafetyManager")
        
        self.logger.info("Safety Manager initialized")
        
        # 하드웨어별 관절 제한 로드
        self._load_joint_limits()
    
    def _load_joint_limits(self):
        """하드웨어 설정에서 관절 제한 로드"""
        self.joint_limits: Dict[str, List[Tuple[float, float]]] = {}
        
        for arm_name, arm_config in self.hw_config.system_config.arms.items():
            if arm_config.joint_limits:
                limits = []
                for joint_name, (min_val, max_val) in arm_config.joint_limits.items():
                    limits.append((min_val, max_val))
                self.joint_limits[arm_name] = limits
            else:
                # 기본 제한 (±π)
                self.joint_limits[arm_name] = [(-np.pi, np.pi)] * arm_config.dof
        
        self.logger.info(f"Loaded joint limits for {len(self.joint_limits)} arms")
    
    def start_monitoring(self):
        """안전 모니터링 시작"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("🛡️ Safety monitoring started")
    
    def stop_monitoring(self):
        """안전 모니터링 중지"""
        self.stop_monitoring.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        self.logger.info("🛡️ Safety monitoring stopped")
    
    def _monitoring_loop(self):
        """안전 모니터링 루프"""
        check_interval = 1.0 / self.config.collision_check_frequency
        
        while not self.stop_monitoring.is_set():
            start_time = time.time()
            
            try:
                # 정기적인 안전 검사
                self._perform_periodic_safety_checks()
                
            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")
            
            # 주기 조절
            elapsed = time.time() - start_time
            sleep_time = max(0, check_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _perform_periodic_safety_checks(self):
        """주기적 안전 검사"""
        current_time = time.time()
        
        # 통신 타임아웃 검사
        if self.config.enable_hardware_monitoring:
            self._check_communication_timeout(current_time)
        
        # 충돌 위험 검사
        if self.config.enable_collision_check and len(self.arm_states) >= 2:
            self._check_collision_risk()
        
        # 위반 만료 검사
        self._cleanup_expired_violations(current_time)
        
        # 안전 수준 업데이트
        self._update_safety_level()
        
        self.total_checks += 1
        self.last_check_time = current_time
    
    def validate_command(self, command: Dict[str, Any]) -> bool:
        """명령 유효성 검사"""
        if self.emergency_stop_active:
            self.logger.warning("Command rejected: Emergency stop active")
            return False
        
        try:
            violations = []
            
            # 각 팔의 명령 검사
            for arm_name, arm_commands in command.get('arms', {}).items():
                arm_violations = self._validate_arm_command(arm_name, arm_commands)
                violations.extend(arm_violations)
            
            # 위반사항이 있으면 기록하고 거부
            if violations:
                for violation in violations:
                    self._add_violation(violation)
                
                self.logger.warning(f"Command rejected: {len(violations)} safety violations")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Command validation error: {e}")
            return False
    
    def _validate_arm_command(self, arm_name: str, arm_commands: Dict[str, Any]) -> List[SafetyViolation]:
        """개별 팔 명령 검사"""
        violations = []
        
        # 관절 위치 검사
        if 'joint_positions' in arm_commands:
            joint_violations = self._check_joint_limits(arm_name, arm_commands['joint_positions'])
            violations.extend(joint_violations)
        
        # 엔드이펙터 위치 검사
        if 'effector_position' in arm_commands:
            workspace_violations = self._check_workspace_limits(arm_name, arm_commands['effector_position'])
            violations.extend(workspace_violations)
        
        # 속도 검사 (이전 위치가 있는 경우)
        if arm_name in self.previous_positions:
            velocity_violations = self._check_velocity_limits(arm_name, arm_commands)
            violations.extend(velocity_violations)
        
        return violations
    
    def _check_joint_limits(self, arm_name: str, joint_positions: List[float]) -> List[SafetyViolation]:
        """관절 제한 검사"""
        violations = []
        
        if not self.config.enable_joint_limits or arm_name not in self.joint_limits:
            return violations
        
        limits = self.joint_limits[arm_name]
        margin = self.config.joint_limit_margin
        
        for i, (position, (min_limit, max_limit)) in enumerate(zip(joint_positions, limits)):
            # 마진을 포함한 실제 제한
            effective_min = min_limit + margin
            effective_max = max_limit - margin
            
            if position < effective_min:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.JOINT_LIMIT,
                    severity=SafetyLevel.DANGER,
                    arm_name=arm_name,
                    joint_index=i,
                    current_value=position,
                    limit_value=effective_min,
                    message=f"Joint {i} below lower limit: {position:.3f} < {effective_min:.3f}",
                    timestamp=time.time()
                ))
            
            elif position > effective_max:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.JOINT_LIMIT,
                    severity=SafetyLevel.DANGER,
                    arm_name=arm_name,
                    joint_index=i,
                    current_value=position,
                    limit_value=effective_max,
                    message=f"Joint {i} above upper limit: {position:.3f} > {effective_max:.3f}",
                    timestamp=time.time()
                ))
        
        return violations
    
    def _check_workspace_limits(self, arm_name: str, effector_position: List[float]) -> List[SafetyViolation]:
        """워크스페이스 제한 검사"""
        violations = []
        
        if not self.config.enable_workspace_limits or len(effector_position) < 3:
            return violations
        
        position_dict = {'x': effector_position[0], 'y': effector_position[1], 'z': effector_position[2]}
        
        for axis, position in position_dict.items():
            if axis in self.config.workspace_bounds:
                min_bound, max_bound = self.config.workspace_bounds[axis]
                
                if position < min_bound:
                    violations.append(SafetyViolation(
                        violation_type=SafetyViolationType.WORKSPACE_LIMIT,
                        severity=SafetyLevel.WARNING,
                        arm_name=arm_name,
                        current_value=position,
                        limit_value=min_bound,
                        message=f"Effector {axis} below workspace limit: {position:.3f} < {min_bound:.3f}",
                        timestamp=time.time()
                    ))
                
                elif position > max_bound:
                    violations.append(SafetyViolation(
                        violation_type=SafetyViolationType.WORKSPACE_LIMIT,
                        severity=SafetyLevel.WARNING,
                        arm_name=arm_name,
                        current_value=position,
                        limit_value=max_bound,
                        message=f"Effector {axis} above workspace limit: {position:.3f} > {max_bound:.3f}",
                        timestamp=time.time()
                    ))
        
        return violations
    
    def _check_velocity_limits(self, arm_name: str, arm_commands: Dict[str, Any]) -> List[SafetyViolation]:
        """속도 제한 검사"""
        violations = []
        
        if not self.config.enable_velocity_limits or arm_name not in self.previous_positions:
            return violations
        
        current_time = time.time()
        dt = current_time - self.last_update_time.get(arm_name, current_time)
        
        if dt <= 0:
            return violations
        
        # 관절 속도 검사
        if 'joint_positions' in arm_commands:
            current_positions = arm_commands['joint_positions']
            previous_positions = self.previous_positions[arm_name]
            
            if len(current_positions) == len(previous_positions):
                for i, (curr, prev) in enumerate(zip(current_positions, previous_positions)):
                    velocity = abs(curr - prev) / dt
                    max_vel = self.config.max_joint_velocity * (1 - self.config.velocity_limit_margin)
                    
                    if velocity > max_vel:
                        violations.append(SafetyViolation(
                            violation_type=SafetyViolationType.VELOCITY_LIMIT,
                            severity=SafetyLevel.WARNING,
                            arm_name=arm_name,
                            joint_index=i,
                            current_value=velocity,
                            limit_value=max_vel,
                            message=f"Joint {i} velocity too high: {velocity:.3f} > {max_vel:.3f} rad/s",
                            timestamp=current_time
                        ))
        
        return violations
    
    def _check_collision_risk(self):
        """충돌 위험 검사"""
        if len(self.arm_states) < 2:
            return
        
        try:
            # 두 팔의 엔드이펙터 위치 추출
            arm_names = list(self.arm_states.keys())
            if len(arm_names) >= 2:
                arm1_name, arm2_name = arm_names[0], arm_names[1]
                
                arm1_pos = self.arm_states[arm1_name].get('effector_position', [0, 0, 0])
                arm2_pos = self.arm_states[arm2_name].get('effector_position', [0, 0, 0])
                
                if len(arm1_pos) >= 3 and len(arm2_pos) >= 3:
                    # 거리 계산
                    distance = np.linalg.norm(np.array(arm1_pos[:3]) - np.array(arm2_pos[:3]))
                    
                    if distance < self.config.min_distance_between_arms:
                        violation = SafetyViolation(
                            violation_type=SafetyViolationType.COLLISION_RISK,
                            severity=SafetyLevel.CRITICAL,
                            arm_name="both_arms",
                            current_value=distance,
                            limit_value=self.config.min_distance_between_arms,
                            message=f"Arms too close: {distance:.3f}m < {self.config.min_distance_between_arms:.3f}m",
                            timestamp=time.time()
                        )
                        
                        self._add_violation(violation)
        
        except Exception as e:
            self.logger.error(f"Collision check error: {e}")
    
    def _check_communication_timeout(self, current_time: float):
        """통신 타임아웃 검사"""
        for arm_name in self.arm_states:
            last_update = self.last_update_time.get(arm_name, 0)
            timeout = current_time - last_update
            
            if timeout > self.config.max_communication_timeout:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.COMMUNICATION_TIMEOUT,
                    severity=SafetyLevel.CRITICAL,
                    arm_name=arm_name,
                    current_value=timeout,
                    limit_value=self.config.max_communication_timeout,
                    message=f"Communication timeout: {timeout:.3f}s > {self.config.max_communication_timeout:.3f}s",
                    timestamp=current_time
                )
                
                self._add_violation(violation)
    
    def update_arm_state(self, arm_name: str, arm_state: Dict[str, Any]):
        """팔 상태 업데이트"""
        current_time = time.time()
        
        # 이전 위치 저장
        if arm_name in self.arm_states and 'joint_positions' in self.arm_states[arm_name]:
            self.previous_positions[arm_name] = self.arm_states[arm_name]['joint_positions'].copy()
        
        # 현재 상태 저장
        self.arm_states[arm_name] = arm_state.copy()
        self.last_update_time[arm_name] = current_time
        
        # 위치 히스토리 업데이트
        if arm_name not in self.position_history:
            self.position_history[arm_name] = []
        
        if 'joint_positions' in arm_state:
            self.position_history[arm_name].append(arm_state['joint_positions'].copy())
            
            # 히스토리 길이 제한
            if len(self.position_history[arm_name]) > 100:
                self.position_history[arm_name] = self.position_history[arm_name][-100:]
    
    def _add_violation(self, violation: SafetyViolation):
        """안전 위반 추가"""
        self.active_violations.append(violation)
        self.violation_history.append(violation)
        
        # 히스토리 길이 제한
        if len(self.violation_history) > self.max_history_length:
            self.violation_history = self.violation_history[-self.max_history_length:]
        
        self.total_violations += 1
        
        # 심각한 위반의 경우 로그 출력
        if violation.severity in [SafetyLevel.DANGER, SafetyLevel.CRITICAL]:
            self.logger.warning(f"🚨 Safety violation: {violation.message}")
        
        # 긴급 정지 필요성 검사
        if violation.severity == SafetyLevel.CRITICAL:
            self._trigger_emergency_stop(f"Critical safety violation: {violation.message}")
    
    def _cleanup_expired_violations(self, current_time: float):
        """만료된 위반 정리"""
        expiry_time = 5.0  # 5초 후 만료
        
        self.active_violations = [
            v for v in self.active_violations 
            if current_time - v.timestamp < expiry_time
        ]
    
    def _update_safety_level(self):
        """안전 수준 업데이트"""
        if not self.active_violations:
            self.current_safety_level = SafetyLevel.SAFE
            return
        
        # 가장 심각한 위반의 수준으로 설정
        max_severity = max(v.severity for v in self.active_violations)
        self.current_safety_level = max_severity
    
    def _trigger_emergency_stop(self, reason: str):
        """긴급 정지 발동"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.emergency_stop_reason = reason
            self.logger.critical(f"🚨 EMERGENCY STOP: {reason}")
    
    def reset_emergency_stop(self):
        """긴급 정지 해제"""
        self.emergency_stop_active = False
        self.emergency_stop_reason = ""
        self.active_violations.clear()
        self.current_safety_level = SafetyLevel.SAFE
        self.logger.info("Emergency stop reset")
    
    def handle_emergency(self):
        """긴급 상황 처리"""
        self._trigger_emergency_stop("Manual emergency stop")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """안전 상태 반환"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            'safe': self.current_safety_level == SafetyLevel.SAFE and not self.emergency_stop_active,
            'safety_level': self.current_safety_level.value,
            'emergency_stop_active': self.emergency_stop_active,
            'emergency_stop_reason': self.emergency_stop_reason,
            'active_violations': len(self.active_violations),
            'total_violations': self.total_violations,
            'total_checks': self.total_checks,
            'uptime_seconds': uptime,
            'violation_rate': self.total_violations / max(self.total_checks, 1),
            'arms_monitored': list(self.arm_states.keys()),
            'last_check_ago': current_time - self.last_check_time,
            'warnings': [v.message for v in self.active_violations if v.severity == SafetyLevel.WARNING],
            'errors': [v.message for v in self.active_violations if v.severity in [SafetyLevel.DANGER, SafetyLevel.CRITICAL]]
        }
    
    def get_detailed_violations(self) -> List[Dict[str, Any]]:
        """상세 위반 정보 반환"""
        return [
            {
                'type': v.violation_type.value,
                'severity': v.severity.value,
                'arm_name': v.arm_name,
                'joint_index': v.joint_index,
                'current_value': v.current_value,
                'limit_value': v.limit_value,
                'message': v.message,
                'timestamp': v.timestamp,
                'age_seconds': time.time() - v.timestamp
            }
            for v in self.active_violations
        ]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 메트릭 반환"""
        uptime = time.time() - self.start_time
        
        return {
            'checks_per_second': self.total_checks / max(uptime, 1),
            'violation_rate': self.total_violations / max(self.total_checks, 1),
            'avg_violations_per_hour': (self.total_violations / max(uptime, 1)) * 3600,
            'emergency_stops': 1 if self.emergency_stop_active else 0,
            'arms_monitored': len(self.arm_states),
            'uptime_hours': uptime / 3600
        }
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_monitoring()


def test_safety_manager():
    """안전 관리자 테스트"""
    print("🛡️ Testing Safety Manager...")
    
    try:
        # 하드웨어 설정 로드
        hw_config = get_hardware_config()
        
        # 안전 관리자 생성
        safety_manager = SafetyManager(hw_config)
        
        print("✅ Safety Manager created")
        
        # Context manager로 모니터링 시작
        with safety_manager:
            print("🛡️ Safety monitoring started")
            
            # 안전 상태 확인
            status = safety_manager.get_safety_status()
            print(f"\nInitial Safety Status:")
            print(f"  Safe: {status['safe']}")
            print(f"  Level: {status['safety_level']}")
            print(f"  Emergency Stop: {status['emergency_stop_active']}")
            
            # 유효한 명령 테스트
            valid_command = {
                'arms': {
                    'left_arm': {
                        'joint_positions': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        'effector_position': [0.3, 0.0, 0.4]
                    }
                }
            }
            
            valid = safety_manager.validate_command(valid_command)
            print(f"\n✅ Valid command test: {'PASS' if valid else 'FAIL'}")
            
            # 위험한 명령 테스트 (관절 제한 초과)
            dangerous_command = {
                'arms': {
                    'left_arm': {
                        'joint_positions': [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 5 라디안 초과
                    }
                }
            }
            
            dangerous = safety_manager.validate_command(dangerous_command)
            print(f"🚨 Dangerous command test: {'FAIL' if dangerous else 'PASS (correctly rejected)'}")
            
            # 팔 상태 업데이트 테스트
            arm_state = {
                'joint_positions': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'effector_position': [0.3, 0.0, 0.4, 0.0, 0.0, 0.0],
                'is_moving': False,
                'error_count': 0
            }
            
            safety_manager.update_arm_state("left_arm", arm_state)
            safety_manager.update_arm_state("right_arm", arm_state)
            
            print(f"📊 Arm states updated")
            
            # 5초 동안 모니터링 테스트
            for i in range(5):
                time.sleep(1)
                
                status = safety_manager.get_safety_status()
                violations = safety_manager.get_detailed_violations()
                
                print(f"\nSecond {i+1}:")
                print(f"  Safety Level: {status['safety_level']}")
                print(f"  Active Violations: {status['active_violations']}")
                print(f"  Total Checks: {status['total_checks']}")
                print(f"  Arms Monitored: {len(status['arms_monitored'])}")
                
                if violations:
                    print(f"  Violations:")
                    for v in violations[:3]:  # 최대 3개만 표시
                        print(f"    - {v['type']}: {v['message']}")
            
            # 긴급 정지 테스트
            print(f"\n🚨 Testing emergency stop...")
            safety_manager.handle_emergency()
            
            status = safety_manager.get_safety_status()
            print(f"  Emergency Stop Active: {status['emergency_stop_active']}")
            print(f"  Reason: {status['emergency_stop_reason']}")
            
            # 긴급 정지 해제 테스트
            safety_manager.reset_emergency_stop()
            status = safety_manager.get_safety_status()
            print(f"  Emergency Stop Reset: {not status['emergency_stop_active']}")
            
            # 성능 메트릭 출력
            metrics = safety_manager.get_performance_metrics()
            print(f"\n📈 Performance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        print("✅ Safety Manager test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 실행
    test_safety_manager()