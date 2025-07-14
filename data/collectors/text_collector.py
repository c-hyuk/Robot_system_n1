"""
터미널 텍스트 입력 수집기
사용자의 자연어 명령을 터미널에서 받아 GR00T 모델에 전달
"""

import time
import threading
import queue
import sys
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
import logging

from utils.data_types import LanguageData


class BaseTextCollector(ABC):
    """텍스트 입력 수집기 기본 클래스"""
    
    def __init__(self):
        self.is_running = False
        self.input_thread = None
        self.command_queue = queue.Queue(maxsize=20)
        self.last_command = None
        self.command_count = 0
        self.start_time = None
        
        # 명령어 히스토리
        self.command_history: List[Dict[str, Any]] = []
        self.max_history = 50
        
        # 로깅 설정
        self.logger = logging.getLogger("TextCollector")
    
    @abstractmethod
    def _get_user_input(self) -> Optional[str]:
        """사용자 입력 받기 (하위 클래스에서 구현)"""
        pass
    
    def start_collection(self) -> bool:
        """텍스트 입력 수집 시작"""
        if self.is_running:
            self.logger.warning("Text collection already running")
            return True
        
        self.is_running = True
        self.start_time = time.time()
        self.command_count = 0
        
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        self.logger.info("Started text input collection")
        return True
    
    def stop_collection(self) -> None:
        """텍스트 입력 수집 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
        
        self.logger.info("Stopped text input collection")
    
    def _input_loop(self) -> None:
        """입력 수집 루프 (별도 스레드에서 실행)"""
        self._show_welcome_message()
        
        while self.is_running:
            try:
                user_input = self._get_user_input()
                
                if user_input is not None:
                    self._process_and_queue_command(user_input)
                    
            except KeyboardInterrupt:
                self.logger.info("Text collection interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in input loop: {e}")
                time.sleep(0.1)
    
    def _show_welcome_message(self) -> None:
        """환영 메시지 및 사용법 출력"""
        print("\n" + "="*60)
        print("🤖 GR00T Robot Control System")
        print("="*60)
        print("터미널에서 자연어 명령을 입력하세요.")
        print("예시:")
        print("  - 'Pick up the red cube'")
        print("  - '빨간 컵을 집어줘'") 
        print("  - 'Move to the left'")
        print("  - 'Stop'")
        print("\n명령어:")
        print("  /help    - 도움말 보기")
        print("  /history - 명령 히스토리 보기")
        print("  /clear   - 화면 지우기")
        print("  /quit    - 종료")
        print("="*60)
    
    def _process_and_queue_command(self, user_input: str) -> None:
        """명령어 처리 및 큐에 추가"""
        timestamp = time.time()
        user_input = user_input.strip()
        
        # 빈 입력 무시
        if not user_input:
            return
        
        # 시스템 명령어 처리
        if user_input.startswith('/'):
            self._handle_system_command(user_input)
            return
        
        # 일반 명령어 처리
        command_data = {
            'command': user_input,
            'timestamp': timestamp,
            'command_id': self.command_count
        }
        
        # 명령어 전처리
        processed_command = self._preprocess_command(user_input)
        command_data['processed_command'] = processed_command
        
        # 큐에 추가
        try:
            self.command_queue.put_nowait(command_data)
            self.last_command = command_data
            self.command_count += 1
            
            # 히스토리에 추가
            self._add_to_history(command_data)
            
            # 확인 메시지 출력
            print(f"✓ 명령어 수신: '{user_input}'")
            
        except queue.Full:
            try:
                self.command_queue.get_nowait()  # 오래된 명령어 제거
                self.command_queue.put_nowait(command_data)
                self.last_command = command_data
                self.command_count += 1
                print(f"✓ 명령어 수신 (이전 명령어 덮어씀): '{user_input}'")
            except queue.Empty:
                pass
    
    def _preprocess_command(self, command: str) -> str:
        """명령어 전처리"""
        # 기본 전처리
        processed = command.strip().lower()
        
        # TODO: 추가 전처리 (동의어 처리, 언어 감지 등)
        # - 동의어 통일 ("잡아" → "집어", "움직여" → "이동해")
        # - 언어 감지 및 번역
        # - 줄임말 확장
        # - 특수문자 정리
        
        return processed
    
    def _handle_system_command(self, command: str) -> None:
        """시스템 명령어 처리"""
        command = command.lower()
        
        if command == '/help':
            self._show_help()
        elif command == '/history':
            self._show_history()
        elif command == '/clear':
            self._clear_screen()
        elif command == '/quit' or command == '/exit':
            print("시스템을 종료합니다...")
            self.stop_collection()
        elif command == '/status':
            self._show_status()
        else:
            print(f"알 수 없는 명령어: {command}")
            print("'/help'를 입력하여 사용 가능한 명령어를 확인하세요.")
    
    def _show_help(self) -> None:
        """도움말 출력"""
        print("\n📖 도움말")
        print("-" * 40)
        print("자연어 명령 예시:")
        print("  영어: 'Pick up the red cube'")
        print("       'Move to the left'")
        print("       'Stop current action'")
        print("  한국어: '빨간 상자를 집어줘'")
        print("         '왼쪽으로 이동해'")
        print("         '정지해'")
        print("\n시스템 명령어:")
        print("  /help    - 이 도움말 보기")
        print("  /history - 명령 히스토리 보기")
        print("  /clear   - 화면 지우기")
        print("  /status  - 시스템 상태 보기")
        print("  /quit    - 프로그램 종료")
        print("-" * 40)
    
    def _show_history(self) -> None:
        """명령 히스토리 출력"""
        print(f"\n📝 명령 히스토리 (최근 {min(10, len(self.command_history))}개)")
        print("-" * 50)
        
        if not self.command_history:
            print("히스토리가 없습니다.")
        else:
            recent_history = self.command_history[-10:]  # 최근 10개만
            for i, cmd in enumerate(recent_history, 1):
                timestamp = time.strftime("%H:%M:%S", time.localtime(cmd['timestamp']))
                print(f"{i:2d}. [{timestamp}] {cmd['command']}")
        
        print("-" * 50)
    
    def _clear_screen(self) -> None:
        """화면 지우기"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        self._show_welcome_message()
    
    def _show_status(self) -> None:
        """시스템 상태 출력"""
        print(f"\n📊 시스템 상태")
        print("-" * 30)
        print(f"실행 시간: {time.time() - self.start_time:.1f}초")
        print(f"수신 명령수: {self.command_count}")
        print(f"대기 명령수: {self.command_queue.qsize()}")
        print(f"마지막 명령: {self.last_command['command'] if self.last_command else 'None'}")
        print("-" * 30)
    
    def _add_to_history(self, command_data: Dict[str, Any]) -> None:
        """히스토리에 명령어 추가"""
        self.command_history.append(command_data)
        
        # 히스토리 크기 제한
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
    
    def get_latest_command(self) -> Optional[Dict[str, Any]]:
        """최신 명령어 반환"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def has_pending_commands(self) -> bool:
        """대기 중인 명령어가 있는지 확인"""
        return not self.command_queue.empty()
    
    def clear_pending_commands(self) -> None:
        """대기 중인 모든 명령어 제거"""
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break


class TerminalTextCollector(BaseTextCollector):
    """터미널 기반 텍스트 수집기"""
    
    def _get_user_input(self) -> Optional[str]:
        """터미널에서 사용자 입력 받기"""
        try:
            # 프롬프트 출력
            user_input = input("🤖 명령어 입력> ")
            return user_input
            
        except EOFError:
            # Ctrl+D 등으로 입력 종료
            return None
        except KeyboardInterrupt:
            # Ctrl+C로 인터럽트
            raise


class MockTextCollector(BaseTextCollector):
    """Mock 텍스트 수집기 (테스트용)"""
    
    def __init__(self):
        super().__init__()
        self.mock_commands = [
            "Pick up the red cube",
            "빨간 컵을 집어줘",
            "Move to the left",
            "왼쪽으로 이동해",
            "Stop current action",
            "정지해",
            "Place the object on the table",
            "물건을 테이블에 놓아줘"
        ]
        self.command_index = 0
    
    def _get_user_input(self) -> Optional[str]:
        """Mock 명령어 생성"""
        if not self.is_running:
            return None
        
        # 3초마다 새로운 명령어 생성
        time.sleep(3.0)
        
        if self.command_index >= len(self.mock_commands):
            return None  # 모든 명령어 완료
        
        command = self.mock_commands[self.command_index]
        self.command_index += 1
        
        print(f"\n[Mock Input] {command}")
        return command


class TextCollectorManager:
    """텍스트 수집 관리자"""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.collector: BaseTextCollector = None
        self.is_running = False
        
        # 로깅 설정
        self.logger = logging.getLogger("TextCollectorManager")
        
        self._initialize_collector()
    
    def _initialize_collector(self) -> None:
        """수집기 초기화"""
        if self.use_mock:
            self.collector = MockTextCollector()
        else:
            self.collector = TerminalTextCollector()
        
        self.logger.info(f"Initialized text collector: {'Mock' if self.use_mock else 'Terminal'}")
    
    def start_collection(self) -> bool:
        """텍스트 수집 시작"""
        if self.is_running:
            return True
        
        success = self.collector.start_collection()
        self.is_running = success
        return success
    
    def stop_collection(self) -> None:
        """텍스트 수집 중지"""
        if self.collector:
            self.collector.stop_collection()
        self.is_running = False
    
    def get_latest_command(self) -> LanguageData:
        """최신 명령어를 GR00T 형식으로 반환"""
        command_data = self.collector.get_latest_command()
        
        if command_data:
            # GR00T 데이터 키 형식으로 변환
            return {
                "annotation.language.instruction": command_data['processed_command']
            }
        else:
            return {}
    
    def has_new_commands(self) -> bool:
        """새로운 명령어가 있는지 확인"""
        return self.collector.has_pending_commands()
    
    def get_collector_status(self) -> Dict[str, Any]:
        """수집기 상태 반환"""
        return {
            'is_running': self.is_running,
            'command_count': self.collector.command_count,
            'pending_commands': self.collector.command_queue.qsize(),
            'last_command': self.collector.last_command['command'] if self.collector.last_command else None
        }
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_collection()


# 편의용 함수들
def create_text_collector(use_mock: bool = False) -> TextCollectorManager:
    """텍스트 수집기 생성"""
    return TextCollectorManager(use_mock=use_mock)


def test_text_collection(duration: float = 30.0, use_mock: bool = False):
    """텍스트 수집 테스트"""
    print(f"Testing text collection for {duration} seconds...")
    
    with create_text_collector(use_mock=use_mock) as collector:
        start_time = time.time()
        command_count = 0
        
        while time.time() - start_time < duration:
            if collector.has_new_commands():
                command_data = collector.get_latest_command()
                
                if command_data:
                    command_count += 1
                    instruction = command_data.get("annotation.language.instruction", "")
                    print(f"Command {command_count}: '{instruction}'")
            
            # 상태 출력 (5초마다)
            if int(time.time() - start_time) % 5 == 0:
                status = collector.get_collector_status()
                print(f"Status: {status['command_count']} total, {status['pending_commands']} pending")
            
            time.sleep(0.5)
        
        print(f"Test completed. Processed {command_count} commands.")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    print("텍스트 수집 테스트")
    print("1. Terminal 모드 (실제 입력)")
    print("2. Mock 모드 (자동 명령어)")
    
    choice = input("선택하세요 (1/2): ").strip()
    use_mock = choice == "2"
    
    test_text_collection(duration=60.0, use_mock=use_mock)