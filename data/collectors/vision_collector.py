"""
비전 데이터 수집기
현재 하드웨어: 2x D435 Intel RealSense, 1x Zed21 스테레오 카메라
"""

import time
import threading
import queue
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
import cv2
import logging

from utils.data_types import CameraConfig, VideoData
from config.hardware_config import get_hardware_config


class BaseCameraCollector(ABC):
    """카메라 데이터 수집기 기본 클래스"""
    
    def __init__(self, camera_config: CameraConfig):
        self.config = camera_config
        self.is_running = False
        self.capture_thread = None
        self.data_queue = queue.Queue(maxsize=10)  # 최대 10프레임 버퍼
        self.last_frame = None
        self.frame_count = 0
        self.start_time = None
        
        # 로깅 설정
        self.logger = logging.getLogger(f"Camera_{camera_config.name}")
    
    @abstractmethod
    def _initialize_camera(self) -> bool:
        """카메라 초기화 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _capture_frame(self) -> Optional[np.ndarray]:
        """프레임 캡처 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    def _cleanup_camera(self) -> None:
        """카메라 정리 (하위 클래스에서 구현)"""
        pass
    
    def start_capture(self) -> bool:
        """캡처 시작"""
        if self.is_running:
            self.logger.warning("Camera already running")
            return True
        
        if not self._initialize_camera():
            self.logger.error("Failed to initialize camera")
            return False
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info(f"Started camera capture: {self.config.name}")
        return True
    
    def stop_capture(self) -> None:
        """캡처 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        self._cleanup_camera()
        self.logger.info(f"Stopped camera capture: {self.config.name}")
    
    def _capture_loop(self) -> None:
        """캡처 루프 (별도 스레드에서 실행)"""
        target_interval = 1.0 / self.config.fps
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                frame = self._capture_frame()
                if frame is not None:
                    self._process_and_queue_frame(frame)
                    self.frame_count += 1
                else:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.01)  # 짧은 대기 후 재시도
                    
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
            
            # FPS 조절
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_and_queue_frame(self, frame: np.ndarray) -> None:
        """프레임 처리 및 큐에 추가"""
        timestamp = time.time()
        
        # 프레임 전처리 (크기 조정 등)
        processed_frame = self._preprocess_frame(frame)
        
        frame_data = {
            'frame': processed_frame,
            'timestamp': timestamp,
            'frame_id': self.frame_count
        }
        
        # 큐가 가득 찬 경우 오래된 프레임 제거
        try:
            self.data_queue.put_nowait(frame_data)
            self.last_frame = frame_data
        except queue.Full:
            try:
                self.data_queue.get_nowait()  # 오래된 프레임 제거
                self.data_queue.put_nowait(frame_data)
                self.last_frame = frame_data
            except queue.Empty:
                pass
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리"""
        # 원본 크기와 처리용 크기가 다른 경우 리사이즈
        if (frame.shape[1] != self.config.processed_width or 
            frame.shape[0] != self.config.processed_height):
            frame = cv2.resize(
                frame, 
                (self.config.processed_width, self.config.processed_height)
            )
        
        # BGR을 RGB로 변환 (GR00T 모델 입력 형식)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """최신 프레임 반환"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return self.last_frame
    
    def get_fps(self) -> float:
        """현재 FPS 계산"""
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0


class OpenCVCameraCollector(BaseCameraCollector):
    """OpenCV 기반 카메라 수집기 (기본 USB 카메라용)"""
    
    def __init__(self, camera_config: CameraConfig):
        super().__init__(camera_config)
        self.cap = None
    
    def _initialize_camera(self) -> bool:
        """OpenCV VideoCapture 초기화"""
        try:
            # device_id가 문자열인 경우 정수로 변환 시도
            device_id = self.config.device_id
            if isinstance(device_id, str):
                if device_id.startswith('/dev/video'):
                    device_id = int(device_id.split('video')[-1])
                else:
                    device_id = int(device_id)
            
            self.cap = cv2.VideoCapture(device_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Cannot open camera device: {self.config.device_id}")
                return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # 설정 확인
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height}@{actual_fps}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """프레임 캡처"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def _cleanup_camera(self) -> None:
        """카메라 정리"""
        if self.cap:
            self.cap.release()
            self.cap = None


class MockCameraCollector(BaseCameraCollector):
    """Mock 카메라 수집기 (테스트용)"""
    
    def __init__(self, camera_config: CameraConfig):
        super().__init__(camera_config)
        self.frame_counter = 0
    
    def _initialize_camera(self) -> bool:
        """Mock 카메라 초기화"""
        self.logger.info("Mock camera initialized")
        return True
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Mock 프레임 생성"""
        # 컬러풀한 테스트 패턴 생성
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        
        # 그라디언트 패턴
        for i in range(self.config.height):
            for j in range(self.config.width):
                frame[i, j, 0] = (i + self.frame_counter) % 255  # Red
                frame[i, j, 1] = (j + self.frame_counter) % 255  # Green
                frame[i, j, 2] = (i + j + self.frame_counter) % 255  # Blue
        
        # 프레임 번호 텍스트 추가
        cv2.putText(frame, f"Frame: {self.frame_counter}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {self.config.name}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.frame_counter += 1
        return frame
    
    def _cleanup_camera(self) -> None:
        """Mock 카메라 정리"""
        self.logger.info("Mock camera cleaned up")


class VisionCollectorManager:
    """비전 데이터 수집 관리자"""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.collectors: Dict[str, BaseCameraCollector] = {}
        self.is_running = False
        
        # 하드웨어 설정 로드
        self.hw_config = get_hardware_config()
        
        # 로깅 설정
        self.logger = logging.getLogger("VisionCollectorManager")
        
        self._initialize_collectors()
    
    def _initialize_collectors(self) -> None:
        """수집기들 초기화"""
        camera_configs = self.hw_config.system_config.cameras
        
        for camera_name, camera_config in camera_configs.items():
            if self.use_mock:
                collector = MockCameraCollector(camera_config)
            else:
                # 실제 하드웨어 타입에 따라 적절한 수집기 선택
                if "d435" in camera_name.lower():
                    # TODO: RealSenseCollector 구현 후 교체
                    collector = OpenCVCameraCollector(camera_config)
                elif "zed" in camera_name.lower():
                    # TODO: ZedCameraCollector 구현 후 교체
                    collector = OpenCVCameraCollector(camera_config)
                else:
                    collector = OpenCVCameraCollector(camera_config)
            
            self.collectors[camera_name] = collector
            self.logger.info(f"Initialized collector for {camera_name}")
    
    def start_all_cameras(self) -> bool:
        """모든 카메라 시작"""
        if self.is_running:
            self.logger.warning("Cameras already running")
            return True
        
        success_count = 0
        for name, collector in self.collectors.items():
            if collector.start_capture():
                success_count += 1
                self.logger.info(f"Started camera: {name}")
            else:
                self.logger.error(f"Failed to start camera: {name}")
        
        self.is_running = success_count > 0
        self.logger.info(f"Started {success_count}/{len(self.collectors)} cameras")
        return self.is_running
    
    def stop_all_cameras(self) -> None:
        """모든 카메라 중지"""
        for name, collector in self.collectors.items():
            collector.stop_capture()
            self.logger.info(f"Stopped camera: {name}")
        
        self.is_running = False
    
    def get_all_frames(self) -> VideoData:
        """모든 카메라의 최신 프레임 수집"""
        frames = {}
        
        for camera_name, collector in self.collectors.items():
            frame_data = collector.get_latest_frame()
            if frame_data:
                # GR00T 데이터 키 형식으로 변환
                gr00t_key = f"video.{camera_name}"
                frames[gr00t_key] = frame_data['frame']
        
        return frames
    
    def get_camera_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 카메라 상태 반환"""
        status = {}
        for name, collector in self.collectors.items():
            status[name] = {
                'is_running': collector.is_running,
                'fps': collector.get_fps(),
                'frame_count': collector.frame_count,
                'queue_size': collector.data_queue.qsize()
            }
        return status
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_all_cameras()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_all_cameras()


# 편의용 함수들
def create_vision_collector(use_mock: bool = False) -> VisionCollectorManager:
    """비전 수집기 생성"""
    return VisionCollectorManager(use_mock=use_mock)


def test_vision_collection(duration: float = 5.0, use_mock: bool = True):
    """비전 수집 테스트"""
    print(f"Testing vision collection for {duration} seconds...")
    
    with create_vision_collector(use_mock=use_mock) as collector:
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            frames = collector.get_all_frames()
            
            if frames:
                frame_count += 1
                print(f"Frame {frame_count}: {list(frames.keys())}")
                
                # 상태 출력 (1초마다)
                if frame_count % 10 == 0:
                    status = collector.get_camera_status()
                    for camera, info in status.items():
                        print(f"  {camera}: {info['fps']:.1f} fps, queue: {info['queue_size']}")
            
            time.sleep(0.1)
        
        print(f"Test completed. Captured {frame_count} frames.")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_vision_collection(duration=10.0, use_mock=True)