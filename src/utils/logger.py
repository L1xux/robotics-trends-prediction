# Logger utility
"""
Rich 기반 구조화된 로깅 시스템

Features:
- 콘솔 출력: Rich 스타일 (색상, 포맷)
- 파일 저장: 실행별 로그 파일
- 레벨별 필터링: DEBUG, INFO, WARNING, ERROR
- 구조화된 메시지: 컨텍스트 정보 포함
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

# Rich Console (전역)
console = Console()


class StructuredFormatter(logging.Formatter):
    """구조화된 로그 포맷터"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 기본 포맷
        base_format = super().format(record)
        
        # 추가 컨텍스트 (extra 필드)
        if hasattr(record, 'context'):
            context_str = " | ".join(
                f"{k}={v}" for k, v in record.context.items()
            )
            return f"{base_format} | {context_str}"
        
        return base_format


def setup_logger(
    name: str = "report_generator",
    run_id: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: str = "data/logs/pipeline_logs"
) -> logging.Logger:
    """
    로거 설정 및 반환
    
    Args:
        name: 로거 이름
        run_id: 실행 ID (파일명에 사용, None이면 타임스탬프)
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        log_dir: 로그 파일 저장 디렉토리
    
    Returns:
        설정된 Logger 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 1. 콘솔 핸들러 (Rich)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # 2. 파일 핸들러
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / f"run_{run_id}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 파일용 포맷 (상세)
    file_formatter = StructuredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 로거 설정 완료 메시지
    logger.info(f"Logger initialized: run_id={run_id}, log_file={log_file}")
    
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **context
):
    """
    컨텍스트 정보와 함께 로그 출력
    
    Args:
        logger: Logger 인스턴스
        level: 로그 레벨 (info, warning, error, debug)
        message: 로그 메시지
        **context: 추가 컨텍스트 키워드 인자
    
    Example:
        log_with_context(
            logger, 
            'info', 
            'Data collected',
            stage='collection',
            source='arxiv',
            count=120
        )
    """
    log_func = getattr(logger, level.lower())
    log_func(message, extra={'context': context})


# 전역 로거 (기본)
default_logger = setup_logger()


