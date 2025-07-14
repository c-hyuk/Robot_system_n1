"""
GR00T 추론 엔진
실시간 로봇 제어를 위한 고성능 추론 시스템
"""

import time
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
import torch

from model.gr00t_interface import DualPiperGR00TInterface
from data.integrated_pipeline import IntegratedDataPipeline


class InferenceState(Enum):
    """추론 엔진 상태"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class InferenceConfig:
    """추론 엔진 설정"""
    target_frequency: float = 10.0  # Hz (10Hz = 100ms 간격)
    max_queue_size: int = 5
    timeout_seconds: float = 1.0
    enable_action_smoothing: bool = True
    smoothing_alpha: float = 0.7
    enable_safety_checks: bool = True
    max_consecutive_failures: int = 3


class RealTimeInferenceEngine:
    """실시간 GR00T 추론 엔진"""
    
    def __init__(
        self,
        gr00t_interface: DualPiperGR00TInterface,
        config: Optional[InferenceConfig] = None
    ):
        """
        실시간 추론 엔진 초기화
        
        Args:
            gr00t_interface: GR00T 모델 인터페이스
            config: 추론 엔진 설정
        """
        self.gr00t_interface = gr00t_interface
        self.config = config or InferenceConfig()
        
        # 상태 관리
        self.state = InferenceState.IDLE
        self.inference_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 데이터 큐
        self.action_queue = queue.Queue(maxsize=self.config.max_queue_size)
        
        # 액션 스무딩용 히스토리
        self.action_history: List[Dict[str, np.ndarray]] = []
        self.max_history_length = 5
        
        # 안전성 관리
        self.consecutive_failures = 0
        self.last_safe_action: Optional[Dict[str, np.ndarray]] = None
        
        # 성능 모니터링
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.start_time = None
        self.last_inference_time = None
        
        # 콜백 함수들
        self.action_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # 로깅
        self.logger = logging.getLogger("RealTimeInferenceEngine")
    
    def add_action_callback(self, callback: Callable[[Dict[str, np.ndarray]], None]):
        """액션 출력 콜백 추가"""
        self.action_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """에러 콜백 추가"""
        self.error_callbacks.append(callback)
    
    def start(self) -> bool:
        """추론 엔진 시작"""
        if self.state == InferenceState.RUNNING:
            self.logger.warning("Inference engine already running")
            return True
        
        try:
            # GR00T 인터페이스 데이터 파이프라인 시작
            if not self.gr00t_interface.start_data_pipeline():
                self.logger.error("Failed to start GR00T data pipeline")
                return False
            
            # 평가 모드 설정
            self.gr00t_interface.set_training_mode(False)
            
            # 추론 스레드 시작
            self.stop_event.clear()
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()
            
            self.state = InferenceState.RUNNING
            self.start_time = time.time()
            self.consecutive_failures = 0
            
            self.logger.info(f"Inference engine started at {self.config.target_frequency}Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start inference engine: {e}")
            self.state = InferenceState.ERROR
            self._notify_error_callbacks(e)
            return False
    
    def stop(self):
        """추론 엔진 중지"""
        if self.state == InferenceState.IDLE:
            return
        
        self.logger.info("Stopping inference engine...")
        
        # 중지 신호 발송
        self.stop_event.set()
        
        # 스레드 종료 대기
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        
        # GR00T 인터페이스 정리
        self.gr00t_interface.stop_data_pipeline()
        
        self.state = InferenceState.IDLE
        self.logger.info("Inference engine stopped")
    
    def pause(self):
        """추론 엔진 일시정지"""
        if self.state == InferenceState.RUNNING:
            self.state = InferenceState.PAUSED
            self.logger.info("Inference engine paused")
    
    def resume(self):
        """추론 엔진 재개"""
        if self.state == InferenceState.PAUSED:
            self.state = InferenceState.RUNNING
            self.logger.info("Inference engine resumed")
    
    def _inference_loop(self):
        """추론 루프 (별도 스레드에서 실행)"""
        target_interval = 1.0 / self.config.target_frequency
        
        while not self.stop_event.is_set():
            if self.state != InferenceState.RUNNING:
                time.sleep(0.1)
                continue
            
            loop_start = time.time()
            
            try:
                # 추론 실행
                action = self._perform_inference()
                
                if action is not None:
                    # 액션 후처리
                    processed_action = self._postprocess_action(action)
                    
                    # 안전성 검사
                    if self._validate_action(processed_action):
                        # 액션 큐에 추가
                        self._queue_action(processed_action)
                        
                        # 콜백 호출
                        self._notify_action_callbacks(processed_action)
                        
                        # 성공 카운터 리셋
                        self.consecutive_failures = 0
                        self.last_safe_action = processed_action.copy()
                    else:
                        self._handle_unsafe_action()
                else:
                    self._handle_inference_failure()
                
            except Exception as e:
                self.logger.error(f"Inference loop error: {e}")
                self._handle_inference_failure()
                self._notify_error_callbacks(e)
            
            # 타이밍 조절
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 성능 모니터링 업데이트
            self.last_inference_time = time.time()
    
    def _perform_inference(self) -> Optional[Dict[str, Any]]:
        """추론 수행"""
        inference_start = time.time()
        
        try:
            # GR00T 인터페이스에서 액션 예측
            action = self.gr00t_interface.get_action_from_pipeline()
            
            # 성능 통계 업데이트
            inference_time = time.time() - inference_start
            self.total_inference_time += inference_time
            self.inference_count += 1
            
            return action
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None
    
    def _postprocess_action(self, action: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """액션 후처리"""
        processed = {}
        
        # 타입 변환 및 정리
        for key, value in action.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                processed[key] = value.copy()
            else:
                # 다른 타입은 numpy 배열로 변환 시도
                try:
                    processed[key] = np.array(value, dtype=np.float32)
                except:
                    self.logger.warning(f"Could not convert {key} to numpy array")
                    continue
        
        # 액션 스무딩 적용
        if self.config.enable_action_smoothing:
            processed = self._apply_action_smoothing(processed)
        
        return processed
    
    def _apply_action_smoothing(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """액션 스무딩 적용"""
        if not self.action_history:
            # 첫 번째 액션은 그대로 사용
            self.action_history.append(action.copy())
            return action
        
        smoothed = {}
        prev_action = self.action_history[-1]
        alpha = self.config.smoothing_alpha
        
        for key, value in action.items():
            if key in prev_action:
                # 지수 이동 평균 적용
                smoothed[key] = alpha * prev_action[key] + (1 - alpha) * value
            else:
                smoothed[key] = value
        
        # 히스토리 업데이트
        self.action_history.append(smoothed.copy())
        if len(self.action_history) > self.max_history_length:
            self.action_history = self.action_history[-self.max_history_length:]
        
        return smoothed
    
    def _validate_action(self, action: Dict[str, np.ndarray]) -> bool:
        """액션 유효성 검증"""
        if not self.config.enable_safety_checks:
            return True
        
        try:
            for key, value in action.items():
                # NaN/Inf 검사
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    self.logger.error(f"Invalid values in action {key}: NaN or Inf")
                    return False
                
                # 관절 제한 검사 (간단한 범위 체크)
                if "joint_position" in key:
                    if np.any(np.abs(value) > 3.14):  # ±180도 제한
                        self.logger.warning(f"Joint position out of range in {key}")
                        # 경고만 하고 계속 진행 (클리핑 적용)
                        action[key] = np.clip(value, -3.14, 3.14)
                
                elif "effector_position" in key:
                    # 위치는 ±2m, 회전은 ±180도로 제한
                    if len(value) >= 3:
                        action[key][:3] = np.clip(value[:3], -2.0, 2.0)
                    if len(value) >= 6:
                        action[key][3:6] = np.clip(value[3:6], -3.14, 3.14)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Action validation error: {e}")
            return False
    
    def _handle_unsafe_action(self):
        """안전하지 않은 액션 처리"""
        self.consecutive_failures += 1
        self.logger.warning(f"Unsafe action detected. Failures: {self.consecutive_failures}")
        
        if self.last_safe_action is not None:
            # 마지막 안전한 액션 사용
            self._queue_action(self.last_safe_action)
            self._notify_action_callbacks(self.last_safe_action)
        
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.logger.error("Too many consecutive failures. Entering error state.")
            self.state = InferenceState.ERROR
    
    def _handle_inference_failure(self):
        """추론 실패 처리"""
        self.consecutive_failures += 1
        self.logger.error(f"Inference failure. Count: {self.consecutive_failures}")
        
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.logger.error("Too many consecutive failures. Entering error state.")
            self.state = InferenceState.ERROR
    
    def _queue_action(self, action: Dict[str, np.ndarray]):
        """액션을 큐에 추가"""
        try:
            # 큐가 가득 찬 경우 오래된 액션 제거
            if self.action_queue.full():
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.action_queue.put_nowait(action)
            
        except queue.Full:
            self.logger.warning("Action queue is full")
    
    def _notify_action_callbacks(self, action: Dict[str, np.ndarray]):
        """액션 콜백 호출"""
        for callback in self.action_callbacks:
            try:
                callback(action)
            except Exception as e:
                self.logger.error(f"Action callback error: {e}")
    
    def _notify_error_callbacks(self, error: Exception):
        """에러 콜백 호출"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error callback error: {e}")
    
    def get_latest_action(self, timeout: float = None) -> Optional[Dict[str, np.ndarray]]:
        """최신 액션 반환"""
        timeout = timeout or self.config.timeout_seconds
        
        try:
            return self.action_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_engine_status(self) -> Dict[str, Any]:
        """엔진 상태 반환"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_inference_time = (self.total_inference_time / self.inference_count 
                             if self.inference_count > 0 else 0)
        actual_frequency = self.inference_count / elapsed if elapsed > 0 else 0
        
        return {
            'state': self.state.value,
            'target_frequency': self.config.target_frequency,
            'actual_frequency': actual_frequency,
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'consecutive_failures': self.consecutive_failures,
            'action_queue_size': self.action_queue.qsize(),
            'uptime_seconds': elapsed,
            'last_inference_ago': time.time() - self.last_inference_time if self.last_inference_time else None
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 메트릭 반환"""
        status = self.get_engine_status()
        
        # 실시간 성능 지표
        metrics = {
            'frequency_ratio': status['actual_frequency'] / status['target_frequency'] if status['target_frequency'] > 0 else 0,
            'avg_latency_ms': status['avg_inference_time_ms'],
            'failure_rate': self.consecutive_failures / max(self.inference_count, 1),
            'queue_utilization': status['action_queue_size'] / self.config.max_queue_size
        }
        
        return metrics
    
    def __enter__(self):
        """Context manager 진입"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop()


def create_inference_engine(
    model_path: str,
    target_frequency: float = 10.0,
    use_mock_data: bool = False
) -> RealTimeInferenceEngine:
    """추론 엔진 생성"""
    from model.gr00t_interface import create_dual_piper_interface
    
    # GR00T 인터페이스 생성
    gr00t_interface = create_dual_piper_interface(
        model_path=model_path,
        use_mock_data=use_mock_data
    )
    
    # 추론 엔진 설정
    config = InferenceConfig(target_frequency=target_frequency)
    
    return RealTimeInferenceEngine(gr00t_interface, config)


def test_inference_engine():
    """추론 엔진 테스트"""
    print("Testing real-time inference engine...")
    
    # Mock 액션 콜백
    def action_callback(action):
        print(f"Received action: {list(action.keys())}")
        for key, value in action.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
    
    def error_callback(error):
        print(f"Error occurred: {error}")
    
    try:
        # 추론 엔진 생성 (Mock 모드)
        engine = create_inference_engine(
            model_path="mock_model_path",
            target_frequency=5.0,  # 5Hz로 테스트
            use_mock_data=True
        )
        
        # 콜백 등록
        engine.add_action_callback(action_callback)
        engine.add_error_callback(error_callback)
        
        print("✅ Inference engine created")
        
        # 추론 엔진 시작
        with engine:
            print("🚀 Inference engine started")
            
            # 10초 동안 실행
            for i in range(10):
                time.sleep(1)
                
                # 상태 확인
                status = engine.get_engine_status()
                metrics = engine.get_performance_metrics()
                
                print(f"\nSecond {i+1}:")
                print(f"  State: {status['state']}")
                print(f"  Frequency: {status['actual_frequency']:.1f}Hz (target: {status['target_frequency']}Hz)")
                print(f"  Inference count: {status['inference_count']}")
                print(f"  Avg latency: {status['avg_inference_time_ms']:.1f}ms")
                print(f"  Queue size: {status['action_queue_size']}")
                print(f"  Performance ratio: {metrics['frequency_ratio']:.2f}")
                
                # 액션 수동 수집 테스트
                action = engine.get_latest_action(timeout=0.1)
                if action:
                    print(f"  Manual action fetch: {list(action.keys())}")
        
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: This is expected if the actual model path doesn't exist")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_inference_engine()