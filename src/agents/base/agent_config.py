# AgentConfig (Dataclass)
"""
Agent 설정 구조
각 Agent의 설정을 정의하는 dataclass
"""
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Agent 설정"""
    
    name: str  # Agent 이름 (예: "Planning", "DataCollection")
    description: str  # Agent 설명
    model_name: str  # LLM 모델명 (예: "gpt-4o")
    temperature: float = 0.7  # Temperature (0.0 ~ 2.0)
    retry_count: int = 3  # 실패 시 재시도 횟수
    
    def __repr__(self) -> str:
        return (
            f"AgentConfig(name={self.name}, "
            f"model={self.model_name}, "
            f"temp={self.temperature})"
        )