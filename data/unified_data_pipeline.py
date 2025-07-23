"""
통합 데이터 파이프라인
GR00T 모델을 위한 실시간 데이터 수집, 처리, 변환 시스템
"""

import time
import threading
import queue
import gc
from typing import Dict, Any, Optional, List, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import logging
import numpy as np
import weakref

# 기존 수집기들 import
from data.collectors.vision_collector import VisionCollectorManager
from data.collectors.state_collector import DualArmStateCollectorManager
from data.collectors.text_collector import TextCollectorManager

# GR00T 표준 시스템
from gr00t.experiment.data_config import DATA_CONFIG_MAP


@dataclass
class CollectionConfig:
    """데이터 수집 설정 파라미터"""
    collection_frequency: float = 10.0  # Hz
    max_retry_attempts: int = 3
    data_buffer_size: int = 20
    sync_tolerance_ms: float = 100.0  # 동기화 허용 오차 (ms)
    memory_cleanup_interval: int = 50  # N번마다 메모리 정리
    max_data_age_seconds: float = 5.0  # 오래된 데이터 제거 기준


@dataclass
class RobotData:
    """로봇 통합 데이터 구조"""
    timestamp: float = field(default_factory=time.time)
    video_data: Optional[Dict[str, np.ndarray]] = None
    state_data: Optional[Dict[str, np.ndarray]] = None
    language_data: Optional[Dict[str, Any]] = None
    
    def has_video_data(self) -> bool:
        return self.video_data is not None and len(self.video_data) > 0
    
    def has_state_data(self) -> bool:
        return self.state_data is not None and len(self.state_data) > 0
    
    def has_language_data(self) -> bool:
        return self.language_data is not None and len(self.language_data) > 0


@dataclass
class ModalityData:
    """GR00T 모델용 멀티모달 데이터 구조"""
    timestamp: float = field(default_factory=time.time)
    modalities: Dict[str, Any] = field(default_factory=dict)
    
    def add_modality(self, key: str, data: Any) -> None:
        if key and data is not None:
            self.modalities[key] = data
    
    def has_data(self) -> bool:
        return len(self.modalities) > 0
    
    def get_modality_keys(self) -> List[str]:
        return list(self.modalities.keys())


@dataclass
class TimestampedData:
    """타임스탬프가 포함된 데이터"""
    timestamp: float
    data: Any
    data_type: str  # 'video', 'state', 'language'
    
    def age(self) -> float:
        return time.time() - self.timestamp


class DataBuffer:
    """스레드 안전한 데이터 버퍼"""
    
    def __init__(self, max_size: int = 20, max_age_seconds: float = 5.0):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.buffer: Deque[TimestampedData] = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, timestamped_data: TimestampedData) -> None:
        with self.lock:
            self.buffer.append(timestamped_data)
            self._cleanup_old_data()
    
    def get_closest_to_timestamp(self, target_timestamp: float, 
                                tolerance_ms: float = 100.0) -> Optional[TimestampedData]:
        with self.lock:
            if not self.buffer:
                return None
            
            tolerance_seconds = tolerance_ms / 1000.0
            closest_data = None
            min_diff = float('inf')
            
            for data in self.buffer:
                diff = abs(data.timestamp - target_timestamp)
                if diff < min_diff and diff <= tolerance_seconds:
                    min_diff = diff
                    closest_data = data
            
            return closest_data
    
    def get_latest(self) -> Optional[TimestampedData]:
        with self.lock:
            return self.buffer[-1] if self.buffer else None
    
    def _cleanup_old_data(self) -> None:
        current_time = time.time()
        while self.buffer and (current_time - self.buffer[0].timestamp) > self.max_age_seconds:
            self.buffer.popleft()
    
    def size(self) -> int:
        with self.lock:
            return len(self.buffer)
    
    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()


class ModalityConverter:
    """GR00T 모델용 데이터 변환기"""
    
    @staticmethod
    def convert_to_groot_format(robot_data: RobotData, 
                               config: Optional[CollectionConfig] = None) -> Optional[ModalityData]:
        try:
            modality_data = ModalityData()
            modality_data.timestamp = robot_data.timestamp
            
            # 1. 상태 데이터 변환 (DualPiperDataConfig 형식에 맞춤)
            # state.xxx 키들을 하나의 벡터로 합쳐 (1, 20) shape의 'state'로 제공
            if robot_data.has_state_data() and robot_data.state_data is not None:
                state_keys = [
                    'state.left_arm_eef_pos',
                    'state.left_arm_eef_quat',
                    'state.left_gripper_qpos',
                    'state.right_arm_eef_pos',
                    'state.right_arm_eef_quat',
                    'state.right_gripper_qpos',
                ]
                state_vecs = []
                for key in state_keys:
                    arr = robot_data.state_data.get(key)
                    if arr is not None:
                        state_vecs.append(arr)
                if state_vecs:
                    state_vector = np.concatenate(state_vecs, axis=0)
                    modality_data.add_modality('state', state_vector[None, :])  # (1, 20)
            
            # 2. 비전 데이터 변환 (DualPiperDataConfig의 video_keys 형식)
            if robot_data.has_video_data() and robot_data.video_data is not None:
                for camera_id, frame in robot_data.video_data.items():
                    if isinstance(frame, np.ndarray) and frame.size > 0:
                        if camera_id in ['right_wrist_view', 'left_wrist_view', 'front_view']:
                            vision_key = f"video.{camera_id}"
                            modality_data.add_modality(vision_key, frame)
                        else:
                            vision_key = f"observation.image.{camera_id}"
                            modality_data.add_modality(vision_key, frame)
            
            # 3. 언어 데이터 변환 (DualPiperDataConfig의 language_keys 형식)
            if robot_data.has_language_data() and robot_data.language_data is not None:
                instruction = robot_data.language_data.get("annotation.language.instruction", "")
                if instruction and isinstance(instruction, str) and instruction.strip():
                    modality_data.add_modality("annotation.language.instruction", instruction.strip())
            
            return modality_data if modality_data.has_data() else None
            
        except Exception as e:
            logging.getLogger("ModalityConverter").error(f"Failed to convert to GR00T format: {e}")
            return None


# ===== 새로운 계층 구조 =====

class DataCollectionLayer:
    """
    하위 레벨 데이터 수집 계층
    - 실제 하드웨어/Mock에서 raw 데이터 수집
    - 버퍼링 및 동기화
    """
    
    def __init__(self, config: Optional[CollectionConfig] = None, use_mock: bool = False):
        self.config = config or CollectionConfig()
        self.use_mock = use_mock
        self.is_running = False
        
        # 수집기들
        self.vision_collector: Optional[VisionCollectorManager] = None
        self.state_collector: Optional[DualArmStateCollectorManager] = None
        self.text_collector: Optional[TextCollectorManager] = None
        
        # 버퍼 시스템
        self.video_buffer = DataBuffer(self.config.data_buffer_size, self.config.max_data_age_seconds)
        self.state_buffer = DataBuffer(self.config.data_buffer_size, self.config.max_data_age_seconds)
        self.language_buffer = DataBuffer(self.config.data_buffer_size, self.config.max_data_age_seconds)
        
        self.background_thread = None
        self.logger = logging.getLogger("DataCollectionLayer")
        
        self._initialize_collectors()
    
    def _initialize_collectors(self) -> None:
        """수집기 초기화"""
        try:
            self.vision_collector = VisionCollectorManager(use_mock=self.use_mock)
            self.logger.info("✓ Vision collector initialized")
            
            self.state_collector = DualArmStateCollectorManager(
                control_frequency=self.config.collection_frequency
            )
            self.logger.info("✓ State collector initialized")
            
            self.text_collector = TextCollectorManager(use_mock=self.use_mock)
            self.logger.info("✓ Text collector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize collectors: {e}")
            raise
    
    def start(self) -> bool:
        """수집 시작"""
        if self.is_running:
            self.logger.warning("Data collection already running")
            return True
        
        success_count = 0
        
        if self.vision_collector and self.vision_collector.start_all_cameras():
            success_count += 1
        
        if self.state_collector and self.state_collector.start_all_collectors():
            success_count += 1
        
        if self.text_collector and self.text_collector.start_collection():
            success_count += 1
        
        if success_count > 0:
            self.is_running = True
            self._start_background_collection()
            self.logger.info(f"✓ Data collection started ({success_count}/3 collectors)")
            return True
        
        return False
    
    def stop(self) -> None:
        """수집 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.background_thread:
            self.background_thread.join(timeout=2.0)
        
        try:
            if self.vision_collector:
                self.vision_collector.stop_all_cameras()
            if self.state_collector:
                self.state_collector.stop_all_collectors()
            if self.text_collector:
                self.text_collector.stop_collection()
        except Exception as e:
            self.logger.error(f"Error stopping collectors: {e}")
        
        self.video_buffer.clear()
        self.state_buffer.clear()
        self.language_buffer.clear()
        
        self.logger.info("✓ Data collection stopped")
    
    def _start_background_collection(self) -> None:
        """백그라운드 수집 스레드 시작"""
        def background_worker():
            while self.is_running:
                try:
                    current_time = time.time()
                    
                    if self.vision_collector:
                        video_data = self.vision_collector.get_all_frames()
                        if video_data:
                            timestamped_video = TimestampedData(current_time, video_data, 'video')
                            self.video_buffer.add(timestamped_video)
                    
                    if self.state_collector:
                        state_data = self.state_collector.get_all_states()
                        if state_data:
                            timestamped_state = TimestampedData(current_time, state_data, 'state')
                            self.state_buffer.add(timestamped_state)
                    
                    if self.text_collector and self.text_collector.has_new_commands():
                        language_data = self.text_collector.get_latest_command()
                        if language_data:
                            timestamped_language = TimestampedData(current_time, language_data, 'language')
                            self.language_buffer.add(timestamped_language)
                    
                    time.sleep(1.0 / self.config.collection_frequency)
                    
                except Exception as e:
                    self.logger.error(f"Error in background collection: {e}")
                    time.sleep(0.1)
        
        self.background_thread = threading.Thread(target=background_worker, daemon=True)
        self.background_thread.start()
    
    def get_synchronized_buffers(self, target_timestamp: float, 
                                tolerance_ms: float) -> Dict[str, Optional[TimestampedData]]:
        """동기화된 버퍼 데이터 반환"""
        return {
            'video': self.video_buffer.get_closest_to_timestamp(target_timestamp, tolerance_ms),
            'state': self.state_buffer.get_closest_to_timestamp(target_timestamp, tolerance_ms),
            'language': self.language_buffer.get_closest_to_timestamp(target_timestamp, tolerance_ms)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """수집 계층 상태"""
        return {
            'is_running': self.is_running,
            'buffers': {
                'video_buffer_size': self.video_buffer.size(),
                'state_buffer_size': self.state_buffer.size(),
                'language_buffer_size': self.language_buffer.size()
            }
        }


class DataProcessingLayer:
    """
    데이터 처리 계층
    - 수집된 데이터를 RobotData로 조합
    - GR00T 형식으로 변환
    """
    
    def __init__(self, config: Optional[CollectionConfig] = None):
        self.config = config or CollectionConfig()
        self.modality_converter = ModalityConverter()
        self.logger = logging.getLogger("DataProcessingLayer")
        
        # 통계
        self.total_frames_collected = 0
        self.total_states_collected = 0
        self.total_commands_collected = 0
    
    def process_synchronized_data(self, buffer_data: Dict[str, Optional[TimestampedData]], 
                                 target_timestamp: float) -> Optional[RobotData]:
        """동기화된 데이터를 RobotData로 변환"""
        try:
            robot_data = RobotData()
            robot_data.timestamp = target_timestamp
            
            if buffer_data['video'] and buffer_data['video'].data:
                robot_data.video_data = buffer_data['video'].data
                self.total_frames_collected += 1
            
            if buffer_data['state'] and buffer_data['state'].data:
                robot_data.state_data = buffer_data['state'].data
                self.total_states_collected += 1
            
            if buffer_data['language'] and buffer_data['language'].data:
                robot_data.language_data = buffer_data['language'].data
                self.total_commands_collected += 1
            
            if self._validate_robot_data(robot_data):
                return robot_data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing synchronized data: {e}")
            return None
    
    def _validate_robot_data(self, robot_data: RobotData) -> bool:
        """데이터 유효성 검증"""
        try:
            has_video = robot_data.has_video_data()
            has_state = robot_data.has_state_data()
            
            if not (has_video or has_state):
                return False
            
            if robot_data.timestamp <= 0:
                return False
            
            age = time.time() - robot_data.timestamp
            if age > self.config.max_data_age_seconds:
                self.logger.warning(f"Data too old: {age:.2f}s")
                return False
            
            if has_state and robot_data.state_data is not None:
                expected_keys = [
                    "state.right_arm_eef_pos", "state.right_arm_eef_quat", 
                    "state.left_arm_eef_pos", "state.left_arm_eef_quat"
                ]
                for key in expected_keys:
                    if key in robot_data.state_data:
                        data = robot_data.state_data[key]
                        if not isinstance(data, np.ndarray) or data.size == 0:
                            self.logger.warning(f"Invalid state data for {key}")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating robot data: {e}")
            return False
    
    def convert_to_groot(self, robot_data: RobotData) -> Optional[ModalityData]:
        """GR00T 형식으로 변환"""
        return self.modality_converter.convert_to_groot_format(robot_data, self.config)
    
    def get_statistics(self) -> Dict[str, int]:
        """처리 통계"""
        return {
            'total_frames': self.total_frames_collected,
            'total_states': self.total_states_collected,
            'total_commands': self.total_commands_collected
        }


class GR00TTransformLayer:
    """
    GR00T 변환 계층
    - GR00T 표준 시스템과의 인터페이스
    """
    
    def __init__(self, embodiment_name: str = "dual_piper_arm"):
        self.embodiment_name = embodiment_name
        
        if embodiment_name not in DATA_CONFIG_MAP:
            raise ValueError(f"Unknown embodiment: {embodiment_name}")
        
        self.config = DATA_CONFIG_MAP[embodiment_name]
        self.modality_config = self.config.modality_config()
        self.transform_pipeline = self.config.transform()
        # metadata.json 로드 및 set_metadata 적용
        import os, json
        from gr00t.data.schema import DatasetMetadata
        exp_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../experiment_cfg")
        metadata_path = os.path.join(exp_cfg_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadatas = json.load(f)
            meta_dict = metadatas.get(embodiment_name)
            if meta_dict is not None:
                metadata = DatasetMetadata.model_validate(meta_dict)
                self.transform_pipeline.set_metadata(metadata)
        self.logger = logging.getLogger("GR00TTransformLayer")
    
    def apply_transform(self, modality_data: ModalityData) -> Optional[Dict[str, Any]]:
        """GR00T transform 적용"""
        try:
            gr00t_data = modality_data.modalities
            transformed_data = self.transform_pipeline.apply(gr00t_data)
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Transform failed: {e}")
            return None
    
    def set_mode(self, training: bool = True):
        """훈련/평가 모드 설정"""
        if training:
            self.transform_pipeline.train()
        else:
            self.transform_pipeline.eval()


class UnifiedDataPipeline:
    """
    통합 데이터 파이프라인
    - 각 계층을 조합하여 완전한 파이프라인 구성
    - 기존 인터페이스 유지
    """
    
    def __init__(self, embodiment_name: str = "dual_piper_arm", 
                 config: Optional[CollectionConfig] = None,
                 use_mock: bool = False):
        
        self.config = config or CollectionConfig()
        self.embodiment_name = embodiment_name
        self.use_mock = use_mock
        
        # 각 계층 초기화
        self.collection_layer = DataCollectionLayer(self.config, use_mock)
        self.processing_layer = DataProcessingLayer(self.config)
        self.transform_layer = GR00TTransformLayer(embodiment_name)
        
        self.is_running = False
        self.latest_robot_data: Optional[RobotData] = None
        self.iteration_count = 0
        self.start_time = None
        
        self.logger = logging.getLogger("UnifiedDataPipeline")
    
    def start(self) -> bool:
        """파이프라인 시작"""
        if not self.collection_layer.start():
            self.logger.error("Failed to start collection layer")
            return False
        
        self.is_running = True
        self.start_time = time.time()
        self.logger.info(f"Pipeline started with embodiment: {self.embodiment_name}")
        return True
    
    def stop(self) -> None:
        """파이프라인 중지"""
        self.collection_layer.stop()
        self.is_running = False
        self.logger.info("Pipeline stopped")
    
    def collect_synchronized_data(self, target_timestamp: Optional[float] = None) -> Optional[RobotData]:
        """동기화된 데이터 수집"""
        if not self.is_running:
            return None
        
        target_timestamp = target_timestamp or time.time()
        
        # 1. 수집 계층에서 버퍼 데이터 가져오기
        buffer_data = self.collection_layer.get_synchronized_buffers(
            target_timestamp, self.config.sync_tolerance_ms
        )
        
        # 2. 처리 계층에서 RobotData로 변환
        robot_data = self.processing_layer.process_synchronized_data(
            buffer_data, target_timestamp
        )
        
        if robot_data:
            self.latest_robot_data = robot_data
            self.iteration_count += 1
            
            # 주기적 메모리 정리
            if self.iteration_count % self.config.memory_cleanup_interval == 0:
                gc.collect()
        
        return robot_data
    
    def get_groot_format_data(self, target_timestamp: Optional[float] = None) -> Optional[ModalityData]:
        """GR00T 형식 데이터 반환"""
        robot_data = self.collect_synchronized_data(target_timestamp)
        if not robot_data:
            return None
        
        return self.processing_layer.convert_to_groot(robot_data)
    
    def get_groot_input(self) -> Optional[Dict[str, Any]]:
        """GR00T 모델 입력 반환"""
        modality_data = self.get_groot_format_data()
        if not modality_data:
            return None
        
        return self.transform_layer.apply_transform(modality_data)
    
    def set_training_mode(self, training: bool = True):
        """훈련/평가 모드 설정"""
        self.transform_layer.set_mode(training)
    
    def get_latest_data(self) -> Optional[RobotData]:
        """최신 로봇 데이터 반환"""
        return self.latest_robot_data
    
    def get_system_status(self) -> Dict[str, Any]:
        """전체 시스템 상태"""
        collection_status = self.collection_layer.get_status()
        processing_stats = self.processing_layer.get_statistics()
        
        return {
            'main_collector': {
                'is_running': self.is_running,
                'uptime': time.time() - (self.start_time or time.time()),
                'total_frames': processing_stats['total_frames'],
                'total_states': processing_stats['total_states'],
                'total_commands': processing_stats['total_commands'],
                'iteration_count': self.iteration_count
            },
            'buffers': collection_status['buffers'],
            'config': {
                'collection_frequency': self.config.collection_frequency,
                'buffer_size': self.config.data_buffer_size,
                'sync_tolerance_ms': self.config.sync_tolerance_ms
            },
            'embodiment_name': self.embodiment_name,
            'transform_mode': 'training' if self.transform_layer.transform_pipeline.training else 'evaluation'
        }
    
    def is_system_ready(self) -> bool:
        """시스템 준비 상태 확인"""
        try:
            if not self.is_running:
                return False
            
            status = self.collection_layer.get_status()
            buffers = status['buffers']
            
            return (buffers['video_buffer_size'] > 0 and 
                   buffers['state_buffer_size'] > 0)
            
        except Exception as e:
            self.logger.error(f"Error checking system readiness: {e}")
            return False
    
    def wait_for_system_ready(self, timeout: float = 30.0) -> bool:
        """시스템 준비 대기"""
        start_time = time.time()
        
        self.logger.info(f"Waiting for system to be ready (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            if self.is_system_ready():
                elapsed = time.time() - start_time
                self.logger.info(f"✓ System is ready (took {elapsed:.2f}s)")
                return True
            
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0:
                status = self.get_system_status()
                buffers = status['buffers']
                self.logger.info(f"Waiting... Video:{buffers['video_buffer_size']}, "
                               f"State:{buffers['state_buffer_size']}, "
                               f"Language:{buffers['language_buffer_size']}")
            
            time.sleep(0.5)
        
        self.logger.warning(f"✗ System not ready after {timeout} seconds")
        return False
    
    def get_data_rates(self) -> Dict[str, float]:
        """데이터 수집 주파수 반환"""
        if not self.start_time:
            return {}
        
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return {}
        
        stats = self.processing_layer.get_statistics()
        
        return {
            'frames_per_second': stats['total_frames'] / elapsed,
            'states_per_second': stats['total_states'] / elapsed,
            'commands_per_second': stats['total_commands'] / elapsed,
            'overall_collection_rate': self.iteration_count / elapsed
        }
    
    def get_modality_config(self) -> Dict[str, Any]:
        """GR00T 모달리티 설정 반환"""
        return self.transform_layer.modality_config
    
    def get_available_embodiments(self) -> List[str]:
        """사용 가능한 embodiment 목록"""
        return list(DATA_CONFIG_MAP.keys())
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
