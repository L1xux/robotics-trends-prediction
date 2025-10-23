"""
Revision Tool

Writer Agent가 사용자 피드백을 받아 RevisionAgent를 호출하는 Tool
간단한 내용 수정에 사용됨
"""

from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class RevisionInput(BaseModel):
    """Revision Tool 입력 스키마"""
    
    current_report: str = Field(
        description="Current full report content (markdown)"
    )
    user_feedback: str = Field(
        description="User feedback for revision"
    )
    feedback_severity: str = Field(
        default="minor",
        description="Feedback severity: 'minor' or 'major'"
    )


class RevisionTool(BaseTool):
    """
    Revision Tool
    
    사용자 피드백을 받아 RevisionAgent를 호출하여 보고서를 수정합니다.
    
    Use when:
    - User wants minor changes to the report
    - User feedback indicates simple revisions (typos, rephrasing, adding details)
    
    Args:
        current_report: Current report markdown content
        user_feedback: User's revision requests
        feedback_severity: "minor" for simple changes
    
    Returns:
        Revised report content
    """
    
    name: str = "revise_report"
    description: str = """
    Revise the current report based on user feedback.
    
    Use this tool when user wants to make changes to the existing report WITHOUT re-collecting data.
    This is for minor to moderate revisions like:
    - Rephrasing sections
    - Adding more details to existing content
    - Fixing errors or typos
    - Reorganizing content
    
    Input:
    - current_report: The full current report (markdown)
    - user_feedback: User's specific feedback and requests
    - feedback_severity: "minor" for simple changes
    
    Output:
    - Revised report (markdown)
    """
    args_schema: type[BaseModel] = RevisionInput
    
    # RevisionAgent will be injected
    revision_agent: Any = Field(default=None, description="RevisionAgent instance")
    
    def __init__(self, revision_agent: Any, **kwargs):
        """
        Initialize RevisionTool
        
        Args:
            revision_agent: RevisionAgent instance to use for revisions
        """
        super().__init__(revision_agent=revision_agent, **kwargs)
    
    def _run(
        self,
        current_report: str,
        user_feedback: str,
        feedback_severity: str = "minor",
        run_manager: Any = None
    ) -> str:
        """
        보고서 수정 실행 - RevisionAgent 호출
        """
        print(f"\n🔧 RevisionTool: Calling RevisionAgent...")
        print(f"   Severity: {feedback_severity}")
        print(f"   Feedback: {user_feedback[:100]}...")
        
        try:
            from src.graph.state import PipelineState
            import asyncio
            import nest_asyncio
            
            # Apply nest_asyncio to allow nested event loops
            nest_asyncio.apply()
            
            # Create state for revision
            revision_state = PipelineState(
                user_input="Revision Request",
                status="needs_revision",
                final_report=current_report,
                review_feedback=user_feedback,
                revision_type=feedback_severity
            )
            
            # Call RevisionAgent
            loop = asyncio.get_event_loop()
            revised_state = loop.run_until_complete(self.revision_agent.execute(revision_state))
            
            revised_report = revised_state.get("final_report", current_report)
            
            print(f"✅ RevisionTool: Revision completed")
            
            # Return the revised report
            return f"REVISION_COMPLETED: Report has been revised based on feedback. The revised report is ready for review."
        
        except Exception as e:
            print(f"❌ RevisionTool error: {e}")
            import traceback
            traceback.print_exc()
            return f"REVISION_ERROR: {str(e)}"
    
    async def _arun(
        self,
        current_report: str,
        user_feedback: str,
        feedback_severity: str = "minor",
        run_manager: Any = None
    ) -> str:
        """
        보고서 수정 실행 (비동기)
        """
        # Just call sync version
        return self._run(current_report, user_feedback, feedback_severity, run_manager)

