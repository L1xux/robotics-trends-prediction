"""
Recollect Tool

Writer Agent가 사용자 피드백을 받아 DataCollectionAgent로 복귀하는 Tool
데이터가 부족하거나 다른 관점의 데이터가 필요할 때 사용 (1회만 가능)
"""

from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class RecollectInput(BaseModel):
    """Recollect Tool 입력 스키마"""
    
    user_feedback: str = Field(
        description="User feedback requesting more data or different perspective"
    )
    additional_keywords: str = Field(
        default="",
        description="Additional keywords to search for (comma-separated)"
    )


class RecollectTool(BaseTool):
    """
    Recollect Tool
    
    사용자 피드백을 받아 DataCollectionAgent를 재실행합니다.
    
    Use when:
    - User wants more data
    - User wants different perspective or missing topics
    - Current data is insufficient for the report
    
    **IMPORTANT: This tool can only be used ONCE per workflow!**
    
    Args:
        user_feedback: User's feedback about what data is needed
        additional_keywords: Additional search keywords
    
    Returns:
        Status message (workflow will restart from data collection)
    """
    
    name: str = "recollect_data"
    description: str = """
    Re-run data collection with user feedback to get more or different data.
    
    Use this tool when user indicates that:
    - More data is needed
    - Different topics or perspectives are missing
    - Current data doesn't cover important aspects
    - Specific companies or technologies are missing
    
    **CRITICAL: This tool can only be used ONCE!** After recollection, only revision is available.
    
    Input:
    - user_feedback: What data is missing or needed
    - additional_keywords: Extra keywords to search (optional)
    
    Output:
    - Workflow will restart from DataCollectionAgent with enhanced keywords
    """
    args_schema: type[BaseModel] = RecollectInput
    
    # Usage tracking
    usage_count: int = Field(default=0, description="Number of times this tool has been used")
    max_usage: int = Field(default=1, description="Maximum number of times this tool can be used")
    
    def __init__(self, **kwargs):
        """Initialize RecollectTool"""
        super().__init__(**kwargs)
    
    def _run(
        self,
        user_feedback: str,
        additional_keywords: str = "",
        run_manager: Optional[Any] = None
    ) -> str:
        """
        데이터 재수집 요청 (동기)
        
        Args:
            user_feedback: 사용자 피드백
            additional_keywords: 추가 키워드
        
        Returns:
            상태 메시지 (실제 재수집은 workflow에서 처리)
        """
        # 사용 횟수 체크
        if self.usage_count >= self.max_usage:
            return (
                f"❌ RecollectTool cannot be used anymore. "
                f"Already used {self.usage_count}/{self.max_usage} times. "
                f"Please use 'revise_report' tool instead for further changes."
            )
        
        print(f"\n🔄 RecollectTool: Requesting data recollection...")
        print(f"   Usage: {self.usage_count + 1}/{self.max_usage}")
        print(f"   Feedback: {user_feedback[:100]}...")
        if additional_keywords:
            print(f"   Additional keywords: {additional_keywords}")
        
        # 사용 횟수 증가
        self.usage_count += 1
        
        # Return simple message - workflow will handle routing
        print(f"✅ RecollectTool: Recollection request registered")
            
        return f"RECOLLECT_REQUESTED: Data recollection needed. Additional keywords: {additional_keywords if additional_keywords else 'None'}. Workflow will restart from DataCollectionAgent."
    
    async def _arun(
        self,
        user_feedback: str,
        additional_keywords: str = "",
        run_manager: Optional[Any] = None
    ) -> str:
        """
        데이터 재수집 요청 (비동기)
        """
        # 동기 버전과 동일
        return self._run(user_feedback, additional_keywords, run_manager)
    
    def reset_usage(self):
        """사용 횟수 리셋 (새 워크플로우 시작 시)"""
        self.usage_count = 0

