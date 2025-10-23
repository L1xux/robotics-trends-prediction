"""
Agent 설정 구조
Pydantic BaseModel 기반의 Agent 설정
"""
from src.core.patterns.base_model import BaseModel, ImmutableModel
from typing import Optional


class AgentConfig(ImmutableModel):
    """
    Immutable Agent Configuration

    Uses ImmutableModel to ensure configuration cannot be changed after creation
    """

    name: str  # Agent 이름 (예: "Planning", "DataCollection")
    description: str  # Agent 설명
    model_name: str  # LLM 모델명 (예: "gpt-4o")
    temperature: float = 0.7  # Temperature (0.0 ~ 2.0)
    retry_count: int = 3  # 실패 시 재시도 횟수
    max_iterations: Optional[int] = None  # 최대 반복 횟수
    verbose: bool = False  # 상세 로그 출력

    def __repr__(self) -> str:
        return (
            f"AgentConfig(name={self.name}, "
            f"model={self.model_name}, "
            f"temp={self.temperature})"
        )