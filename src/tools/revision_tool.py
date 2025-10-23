# src/tools/revision_tool.py

"""
Revision Tool (Decision Marker)

Agent가 말투/표현 수정이 필요하다고 판단했을 때 사용하는 간단한 마커
실제 revision 로직은 WriterAgent에서 처리
"""

from typing import Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class RevisionInput(BaseModel):
    """Revision Tool 입력 스키마"""

    reason: str = Field(
        description="Brief reason why revision is needed (from user feedback)"
    )


class RevisionTool(BaseTool):
    """
    Revision Tool (간단한 decision marker)

    Agent가 이 tool을 선택하면 WriterAgent가 revision 로직 실행
    """

    name: str = "revise_report"
    description: str = """
    Mark that the report needs writing improvements (NO data collection).

    Use this tool when user wants:
    - Better phrasing, tone, or style
    - Clearer explanations
    - Better sentence structure
    - Content reorganization

    **This tool does NOT do the actual revision.**
    It signals to WriterAgent that revision is needed.

    Input:
    - reason: Brief summary of what needs to be revised

    Output:
    - Confirmation that revision will be performed
    """
    args_schema: type[BaseModel] = RevisionInput

    def _run(self, reason: str, run_manager: Optional[Any] = None) -> str:
        """간단한 마커 반환"""
        return f"REVISION_NEEDED: {reason}"

    async def _arun(self, reason: str, run_manager: Optional[Any] = None) -> str:
        """간단한 마커 반환 (비동기)"""
        return f"REVISION_NEEDED: {reason}"
