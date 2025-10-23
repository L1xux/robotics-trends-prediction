# ToolConfig (Dataclass)
"""
Tool 설정 구조
각 Tool의 설정을 정의하는 dataclass
"""
from dataclasses import dataclass


@dataclass
class ToolConfig:
    """Tool 기본 설정"""
    
    name: str  # Tool 이름 (예: "ArxivTool", "GoogleTrendsTool")
    description: str  # Tool 설명
    timeout: int = 300  # 타임아웃 (초)
    retry_count: int = 3  # 실패 시 재시도 횟수
    
    def __repr__(self) -> str:
        return (
            f"ToolConfig(name={self.name}, "
            f"timeout={self.timeout}s, "
            f"retry={self.retry_count})"
        )