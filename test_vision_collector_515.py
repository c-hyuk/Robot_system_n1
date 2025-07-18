#!/usr/bin/env python3
"""
Vision Collector RealSense 515 테스트
업데이트된 vision_collector.py가 RealSense 515와 제대로 작동하는지 테스트
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# 프로젝트 경로 설정
sys.path.append(str(Path(__file__).parent))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from data.collectors.vision_collector import create_vision_collector, test_vision_collection
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


def test_realsense_515_collector():
    """RealSense 515 수집기 테스트"""
    print("🚀 Testing RealSense 515 Vision Collector")
    print("=" * 50)
    
    try:
        # 1. RealSense 515 수집기 생성 (mock=False로 실제 하드웨어 사용)
        print("\n📷 Creating RealSense 515 collector...")
        collector = create_vision_collector(use_mock=False)
        print("  ✅ Collector created")
        
        # 2. 카메라 상태 확인
        print("\n📊 Checking camera status...")
        status = collector.get_camera_status()
        for camera_name, camera_status in status.items():
            print(f"  {camera_name}:")
            print(f"    Running: {camera_status['is_running']}")
            print(f"    FPS: {camera_status['fps']:.1f}")
            print(f"    Frame Count: {camera_status['frame_count']}")
            print(f"    Queue Size: {camera_status['queue_size']}")
        
        # 3. 실시간 데이터 캡처 테스트
        print("\n🎥 Testing real-time data capture...")
        test_duration = 10  # 10초 테스트
        start_time = time.time()
        frame_count = 0
        
        with collector:  # Context manager 사용
            while time.time() - start_time < test_duration:
                try:
                    # 모든 카메라의 프레임 가져오기
                    frames = collector.get_all_frames()
                    
                    if frames:
                        frame_count += 1
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        
                        # 진행상황 출력 (30프레임마다)
                        if frame_count % 30 == 0:
                            print(f"  📊 Frame {frame_count}: {fps:.1f} FPS, Elapsed: {elapsed:.1f}s")
                            
                            # 프레임 정보 출력
                            for camera_key, frame_data in frames.items():
                                if isinstance(frame_data, np.ndarray):
                                    print(f"    {camera_key}: shape={frame_data.shape}, dtype={frame_data.dtype}")
                                    print(f"      Min: {frame_data.min()}, Max: {frame_data.max()}, Mean: {frame_data.mean():.2f}")
                            
                            # 카메라 상태 업데이트
                            status = collector.get_camera_status()
                            for camera, info in status.items():
                                print(f"    {camera}: {info['fps']:.1f} fps, queue: {info['queue_size']}")
                    
                    # 짧은 대기 (CPU 부하 방지)
                    time.sleep(0.01)
                    
                except KeyboardInterrupt:
                    print("\n  ⏹️ Test interrupted by user")
                    break
                except Exception as e:
                    print(f"  ❌ Error during capture: {e}")
                    break
        
        # 4. 최종 통계
        print("\n📈 Final Statistics:")
        final_status = collector.get_camera_status()
        for camera, info in final_status.items():
            print(f"  {camera}:")
            print(f"    Total Frames: {info['frame_count']}")
            print(f"    Average FPS: {info['fps']:.1f}")
            print(f"    Queue Size: {info['queue_size']}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Total frames captured: {frame_count}")
        print(f"   Test duration: {time.time() - start_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_collector():
    """Mock 수집기 테스트 (하드웨어 없이 테스트)"""
    print("\n🧪 Testing Mock Collector (no hardware required)")
    print("=" * 50)
    
    try:
        # Mock 수집기로 테스트
        test_vision_collection(duration=5.0, use_mock=True)
        print("✅ Mock collector test passed")
        return True
    except Exception as e:
        print(f"❌ Mock collector test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Vision Collector Test Suite")
    print("=" * 50)
    
    # 1. Mock 테스트 (하드웨어 없이)
    mock_success = test_mock_collector()
    
    # 2. RealSense 515 테스트 (하드웨어 필요)
    print("\n" + "=" * 50)
    print("📷 RealSense 515 Hardware Test")
    print("=" * 50)
    
    realsense_success = test_realsense_515_collector()
    
    # 3. 최종 결과
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    print(f"Mock Collector: {'✅ PASSED' if mock_success else '❌ FAILED'}")
    print(f"RealSense 515: {'✅ PASSED' if realsense_success else '❌ FAILED'}")
    
    if mock_success and realsense_success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        sys.exit(1) 