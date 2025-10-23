# BaseTool (ABC)
"""
모든 Tool의 기본 클래스
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from src.tools.base.tool_config import ToolConfig


class BaseTool(ABC):
    """
    모든 Tool의 기본 클래스
    
    LangChain BaseTool 대신 단순 ABC 사용
    각 Tool은 _run 메서드를 구현해야 함
    """
    
    def __init__(self, config: ToolConfig):
        """
        BaseTool 초기화
        
        Args:
            config: Tool 설정 (ToolConfig)
        """
        self.config = config
        self.name = config.name
        self.description = config.description
    
    @abstractmethod
    def _run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Tool 실행 (동기)
        하위 클래스에서 반드시 구현해야 함
        
        Returns:
            실행 결과 딕셔너리
        """
        pass
    
    async def _arun(self, *args, **kwargs):
        """
        Tool 실행 (비동기)
        기본적으로 지원하지 않음, 필요시 하위 클래스에서 구현
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"