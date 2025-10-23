# File utilities
"""
파일 입출력 유틸리티

Features:
- JSON 저장/로드 (타입 안전)
- 폴더 생성/관리
- 파일 존재 확인
- 경로 정규화
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    디렉토리 생성 (없으면)
    
    Args:
        path: 디렉토리 경로
    
    Returns:
        Path 객체
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(
    data: Union[Dict, List],
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> Path:
    """
    JSON 파일 저장
    
    Args:
        data: 저장할 데이터 (dict 또는 list)
        file_path: 저장 경로
        indent: 들여쓰기 (가독성)
        ensure_ascii: ASCII 인코딩 여부 (False면 한글 그대로)
    
    Returns:
        저장된 파일 Path
    
    Raises:
        TypeError: 직렬화 불가능한 객체
        IOError: 파일 쓰기 실패
    """
    file_path = Path(file_path)
    
    # 디렉토리 생성
    ensure_dir(file_path.parent)
    
    # JSON 저장
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        return file_path
    except TypeError as e:
        raise TypeError(f"Cannot serialize data to JSON: {e}")
    except IOError as e:
        raise IOError(f"Failed to write file {file_path}: {e}")


def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    """
    JSON 파일 로드
    
    Args:
        file_path: 파일 경로
    
    Returns:
        로드된 데이터 (dict 또는 list)
    
    Raises:
        FileNotFoundError: 파일 없음
        json.JSONDecodeError: JSON 파싱 실패
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {file_path}: {e.msg}",
            e.doc,
            e.pos
        )


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    파일 존재 여부 확인
    
    Args:
        file_path: 파일 경로
    
    Returns:
        존재 여부 (True/False)
    """
    return Path(file_path).exists()


def create_run_folder(
    topic: str,
    base_dir: str = "data/raw"
) -> Path:
    """
    실행별 폴더 생성
    
    Format: {topic}_{YYYYMMDD}_{HHMMSS}
    
    Args:
        topic: 주제 (공백 → 언더스코어)
        base_dir: 기본 디렉토리
    
    Returns:
        생성된 폴더 Path
    
    Example:
        create_run_folder("humanoid robots in manufacturing")
        → data/raw/humanoid_robots_in_manufacturing_20250122_143052/
    """
    # 주제 정규화 (공백 → 언더스코어, 소문자)
    normalized_topic = topic.lower().replace(" ", "_")
    
    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 폴더명
    folder_name = f"{normalized_topic}_{timestamp}"
    
    # 폴더 생성
    folder_path = Path(base_dir) / folder_name
    ensure_dir(folder_path)
    
    # 하위 폴더 생성 (arxiv, trends, news)
    ensure_dir(folder_path / "arxiv")
    ensure_dir(folder_path / "trends")
    ensure_dir(folder_path / "news")
    
    return folder_path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    파일 크기 (bytes)
    
    Args:
        file_path: 파일 경로
    
    Returns:
        파일 크기 (bytes)
    """
    return Path(file_path).stat().st_size


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    디렉토리 내 파일 목록
    
    Args:
        directory: 디렉토리 경로
        pattern: 파일 패턴 (예: "*.json", "*.pdf")
        recursive: 하위 폴더 포함 여부
    
    Returns:
        파일 Path 리스트
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))

