"""
Refine Plan Tool - 계획 개선 및 Human Review 처리
"""

import json
from typing import Dict, Any
from langchain.tools import BaseTool as LangChainBaseTool
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.core.models.planning_model import PlanningOutput
from src.cli.human_review import ReviewCLI
from src.utils.planning_util import ResearchPlanningUtil


class ApprovalDecision(BaseModel):
    """Approval decision output schema"""
    is_approved: bool = Field(description="True if user approves the plan, False if user wants changes")


class RefinePlanUtil(LangChainBaseTool):
    """
    계획 검토 및 개선 Tool (Human Review 포함)
    
    이 Tool은:
    1. 초기 계획을 사용자에게 보여주고 피드백 받기
    2. 승인되면 계획 반환
    3. 승인 안되면 LLM으로 계획 개선 후 다시 검토
    4. 최대 10번 반복
    """
    
    name: str = "refine_plan_with_human_review"
    description: str = """Review and refine research plan with human feedback.
    
Input (JSON string):
- initial_plan: Initial research plan dictionary
- max_attempts: Maximum review attempts (default: 10)

This tool will:
1. Show plan to user and collect feedback
2. If approved, return final plan
3. If not approved, refine plan using LLM
4. Repeat until approved or max attempts reached
"""
    
    refinement_tool: ResearchPlanningUtil = Field(description="Refinement tool")
    review_cli: ReviewCLI = Field(default_factory=ReviewCLI, description="Review CLI")
    approval_llm: Any = Field(default=None, exclude=True, description="LLM for approval checking")
    
    def __init__(self, refinement_tool: ResearchPlanningUtil, **kwargs):
        super().__init__(refinement_tool=refinement_tool, **kwargs)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        object.__setattr__(self, 'approval_llm', llm.with_structured_output(ApprovalDecision))
    
    def _run(self, initial_plan: Dict[str, Any], max_attempts: int = 10) -> str:
        """
        Synchronous version (not used, required by BaseTool)
        """
        raise NotImplementedError("Use async version (_arun)")
    
    async def _arun(
        self,
        initial_plan: Dict[str, Any],
        max_attempts: int = 10
    ) -> str:
        """
        계획 검토 및 개선 (Human Review Loop)
        
        Args:
            initial_plan: 초기 계획 (dict)
            max_attempts: 최대 시도 횟수
        
        Returns:
            최종 승인된 계획 (JSON string)
        
        Raises:
            ToolException: 승인 실패 시
        """
        try:
            # Validate initial plan
            planning_output = PlanningOutput(**initial_plan)
            current_plan = planning_output
            
            # Human Review Loop
            for attempt in range(1, max_attempts + 1):
                print(f"\n{'='*60}")
                print(f"📝 Review Attempt {attempt}/{max_attempts}")
                print(f"{'='*60}\n")
                
                # Display plan
                current_plan_dict = current_plan.model_dump() if hasattr(current_plan, 'model_dump') else current_plan
                self.review_cli.display_plan(current_plan_dict)
                
                # Get feedback (직접 input 사용)
                print(f"\n💬 Review the plan (attempt {attempt}/{max_attempts}):")
                print("(Type your feedback or approval keywords like 'ok', 'approve', '좋아요', etc.)")
                feedback = input("\nYour feedback: ").strip()
                
                if not feedback:
                    print("⚠️  Empty feedback. Please provide feedback or approval.")
                    continue
                
                # Check approval
                is_approved = await self._is_approval(feedback)
                print(f"\n🔍 Checking approval: '{feedback}' -> {is_approved}")
                
                if is_approved:
                    print(f"\n✅ Plan approved!")
                    final_plan_dict = current_plan.model_dump()
                    return json.dumps(final_plan_dict, ensure_ascii=False)
                
                # Refine plan
                print(f"\n🔧 Refining plan based on feedback...")
                print(f"📝 Feedback: {feedback}\n")
                
                current_plan_dict = current_plan.model_dump()
                user_topic = current_plan.topic
                
                # Call refinement tool
                result_json = await self.refinement_tool._arun(
                    topic=user_topic,
                    current_plan=current_plan_dict,
                    user_feedback=feedback
                )
                
                # Parse result
                improved_plan_data = json.loads(result_json)
                improved_plan_data.pop('folder_name', None)
                improved_plan_data.pop('reasoning', None)
                
                # Validate
                current_plan = PlanningOutput(**improved_plan_data)
                print(f"✅ Plan refined!\n")
            
            # Max attempts reached
            raise ToolException(
                f"Plan not approved after {max_attempts} attempts. "
                f"Last plan: {current_plan.model_dump_json()}"
            )
        
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error in refine_plan_tool: {str(e)}")
    
    async def _is_approval(self, feedback: str) -> bool:
        """
        하이브리드 승인 판단: 규칙 우선, 애매한 경우 LLM 사용
        """
        feedback_lower = feedback.lower().strip()
        
        clear_approval = [
            "ok", "okay", "yes", "approve", "approved", "accept", "accepted",
            "good", "great", "perfect", "fine", "proceed", "continue",
            "lgtm", "looks good", "sounds good",
            "좋아요", "좋아", "괜찮아요", "괜찮아", "완벽해요", "완벽해",
            "네", "예", "승인", "확인", "진행", "오케이"
        ]
        
        if feedback_lower in clear_approval:
            return True
        
        return await self._is_approval_llm(feedback)
    async def _is_approval_llm(self, feedback: str) -> bool:
        try:
            system_prompt = """You are a feedback classifier. Analyze user feedback and determine if it indicates approval or rejection.

Approval means:
- User is satisfied with the plan
- User wants to proceed without changes
- Examples: "ok", "good", "looks great", "approve", "yes", "fine", "proceed"

Rejection means:
- User wants changes or improvements
- User provides specific feedback or modifications
- Examples: "change X", "add more Y", "reduce Z", "needs improvement"

Set is_approved to true for approval, false for rejection."""

            user_prompt = f'User feedback: "{feedback}"'
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            result: ApprovalDecision = await self.approval_llm.ainvoke(messages)
            
            return result.is_approved
        except Exception as e:
            print(f"⚠️ LLM approval check failed: {e}")
            return False



