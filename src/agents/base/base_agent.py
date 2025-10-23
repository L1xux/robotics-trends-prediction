# BaseAgent (ABC)
"""
Base Agent 클래스
모든 Agent의 추상 기본 클래스
"""
from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.language_models import BaseChatModel

from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState


class BaseAgent(ABC):
    """
    모든 Agent의 기본 클래스
    하위 Agent는 execute 메서드를 구현해야 함
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any],
        config: AgentConfig
    ):
        """
        Args:
            llm: LangChain BaseChatModel (ChatOpenAI 등)
            tools: Agent가 사용할 Tool 리스트
            config: Agent 설정
        """
        self.llm = llm
        self.tools = tools
        self.config = config
    
    @abstractmethod
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Agent의 메인 실행 로직
        하위 클래스에서 반드시 구현해야 함
        
        Args:
            state: 현재 파이프라인 상태
        
        Returns:
            업데이트된 파이프라인 상태
        """
        pass