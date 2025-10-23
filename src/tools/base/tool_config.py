"""
Tool 설정 구조
Pydantic BaseModel 기반의 Tool 설정
"""
from src.core.patterns.base_model import ImmutableModel
from typing import Optional, Dict, Any


class ToolConfig(ImmutableModel):
    """
    Immutable Tool Configuration

    Uses ImmutableModel to ensure configuration cannot be changed after creation
    """

    name: str  # Tool 이름 (예: "ArxivTool", "GoogleTrendsTool")
    description: str  # Tool 설명
    timeout: int = 300  # 타임아웃 (초)
    retry_count: int = 3  # 실패 시 재시도 횟수
    cache_enabled: bool = False  # 캐싱 사용 여부
    extra_params: Optional[Dict[str, Any]] = None  # 추가 파라미터

    def __repr__(self) -> str:
        return (
            f"ToolConfig(name={self.name}, "
            f"timeout={self.timeout}s, "
            f"retry={self.retry_count})"
        )