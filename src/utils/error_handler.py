# Error handler utility
"""
에러 처리 및 State 덤프

Features:
- State 덤프: 에러 발생 시 전체 State JSON 저장
- 재시도 데코레이터: 자동 재시도 로직
- 에러 컨텍스트: 에러 발생 위치 추적
"""
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps
import time

from src.utils.logger import default_logger

T = TypeVar('T')


def dump_state_on_error(
    state: Dict[str, Any],
    error: Exception,
    stage: str,
    output_dir: str = "data/logs/error_states"
) -> Path:
    """
    에러 발생 시 State를 JSON으로 저장
    
    Args:
        state: LangGraph State
        error: 발생한 예외
        stage: 에러 발생 단계 (예: "planning", "collection")
        output_dir: 저장 디렉토리
    
    Returns:
        저장된 파일 Path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 에러 정보 구조화
    error_dump = {
        "timestamp": timestamp,
        "stage": stage,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        },
        "state": _serialize_state(state)
    }
    
    # 파일 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / f"error_{stage}_{timestamp}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(error_dump, f, indent=2, ensure_ascii=False)
    
    default_logger.error(
        f"State dumped to {file_path}",
        extra={'context': {'stage': stage, 'error_type': type(error).__name__}}
    )
    
    return file_path


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    State를 JSON 직렬화 가능한 형태로 변환
    
    Args:
        state: 원본 State
    
    Returns:
        직렬화 가능한 State
    """
    serialized = {}
    
    for key, value in state.items():
        try:
            # JSON 직렬화 가능한지 테스트
            json.dumps(value)
            serialized[key] = value
        except (TypeError, ValueError):
            # 직렬화 불가능하면 문자열로 변환
            serialized[key] = str(value)
    
    return serialized


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    재시도 데코레이터
    
    Args:
        max_retries: 최대 재시도 횟수
        delay: 초기 대기 시간 (초)
        backoff: 대기 시간 배수 (지수 백오프)
        exceptions: 재시도할 예외 타입들
    
    Example:
        @retry_on_failure(max_retries=3, delay=2.0, backoff=2.0)
        def fetch_data():
            # API 호출 등
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        default_logger.warning(
                            f"Retry {attempt + 1}/{max_retries} after {current_delay}s",
                            extra={
                                'context': {
                                    'function': func.__name__,
                                    'error': str(e)
                                }
                            }
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        default_logger.error(
                            f"All retries failed for {func.__name__}",
                            extra={'context': {'error': str(e)}}
                        )
            
            # 모든 재시도 실패
            raise last_exception
        
        return wrapper
    return decorator


class ErrorContext:
    """
    에러 컨텍스트 관리자 (with문 사용)
    
    Example:
        with ErrorContext(state, stage="planning"):
            # 에러 발생 가능한 코드
            result = planning_agent.execute()
    """
    
    def __init__(
        self,
        state: Dict[str, Any],
        stage: str,
        dump_on_error: bool = True
    ):
        self.state = state
        self.stage = stage
        self.dump_on_error = dump_on_error
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.dump_on_error:
            # 에러 발생 → State 덤프
            dump_state_on_error(self.state, exc_val, self.stage)
        
        # 예외 전파 (False 반환 → 예외 계속 발생)
        return False


