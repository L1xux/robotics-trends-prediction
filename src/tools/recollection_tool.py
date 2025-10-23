# src/tools/recollection_tool.py

"""
Recollection Tool (Decision Marker)

Agent가 데이터 재수집이 필요하다고 판단했을 때 사용하는 간단한 마커
실제 recollection은 graph에서 data_collection_agent로 라우팅
"""

from typing import Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class RecollectionInput(BaseModel):
    """Recollection Tool 입력 스키마"""

    reason: str = Field(
        description="Brief reason why recollection is needed (what data is missing)"
    )


class RecollectionTool(BaseTool):
    """
    Recollection Tool (간단한 decision marker)

    Agent가 이 tool을 선택하면 graph가 data_collection_agent로 라우팅
    """

    name: str = "recollect_data"
    description: str = """
    Mark that additional data collection is needed.

    Use this tool ONLY when user mentions:
    - Missing companies or organizations
    - Missing technologies or topics
    - Insufficient data coverage

    **This tool does NOT collect data.**
    It signals to the graph to route back to data_collection_agent.

    Input:
    - reason: Brief summary of what data is missing

    Output:
    - Confirmation that recollection will be triggered
    """
    args_schema: type[BaseModel] = RecollectionInput

    def _run(self, reason: str, run_manager: Optional[Any] = None) -> str:
        """간단한 마커 반환"""
        return f"RECOLLECTION_NEEDED: {reason}"

    async def _arun(self, reason: str, run_manager: Optional[Any] = None) -> str:
        """간단한 마커 반환 (비동기)"""
        return f"RECOLLECTION_NEEDED: {reason}"
