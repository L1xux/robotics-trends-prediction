"""
Revision Agent

Human Review 2에서 거부된 보고서를 개선하는 Agent
사용자 피드백을 반영하여 보고서를 수정
"""

from typing import List, Any, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState


class RevisionLLM(BaseAgent):
    """
    Revision Agent
    
    Human Review 2에서 거부된 보고서를 개선
    
    Input:
    - final_report: str (현재 보고서)
    - revision_feedback: str (사용자 피드백)
    - quality_report: Dict (품질 검사 결과)
    
    Output:
    - final_report: str (개선된 보고서)
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any],
        config: AgentConfig
    ):
        super().__init__(llm, tools, config)
        self._setup_chains()
    
    def _setup_chains(self):
        """LCEL Chains 설정"""
        str_parser = StrOutputParser()
        
        # Full Report Revision Chain
        full_revision_prompt = ChatPromptTemplate.from_template(
            """You are an expert report editor specializing in AI and Robotics. Your task is to revise an English technical report based on user feedback.

**Current English Report:**
{current_report}

**User Feedback (may be in Korean, but you must apply it to the English report):**
{user_feedback}

**Revision Instructions:**

1. **Reflect User Feedback**:
   - Accurately reflect all points from the user feedback in the English report.
   - Fully apply the requested changes.

2. **Enhance Quality**:
   - Improve the logical flow and clarity.
   - Add specific examples or data if the feedback implies a lack of detail.

3. **Maintain Style**:
   - Keep a professional and objective tone.
   - Use clear and concise language.
   - Ensure consistent terminology.

**Output**:
Return ONLY the revised, full English report in Markdown format.
Maintain the original structure (SUMMARY, Section 1-6, REFERENCE, APPENDIX)."""
        )
        self.full_revision_chain = full_revision_prompt | self.llm | str_parser
        
        # Section-specific Revision Chain
        section_revision_prompt = ChatPromptTemplate.from_template(
            """당신은 보고서 섹션 개선 전문가입니다.

**섹션명:** {section_name}

**현재 내용:**
{current_content}

**문제점:**
{issues}

**개선 요청:**
{improvement_requests}

**개선 지침:**
- 문제점을 모두 해결
- 개선 요청사항 반영
- 논리적 흐름 유지
- 데이터 기반 강화

개선된 섹션 내용만 반환하세요 (마크다운 형식)."""
        )
        self.section_revision_chain = section_revision_prompt | self.llm | str_parser
    
    async def execute(
        self,
        state: PipelineState,
        revision_feedback: str,
        revision_type: str = "full"
    ) -> PipelineState:
        """
        보고서 개선 실행
        
        Args:
            state: PipelineState
            revision_feedback: 사용자 피드백
            revision_type: "full" 또는 "section"
        
        Returns:
            Updated PipelineState with revised report
        """
        print(f"\n{'='*60}")
        print(f"🔧 Revision Agent")
        print(f"{'='*60}\n")
        
        current_report = state.get("final_report", "")
        quality_report = state.get("quality_report", {})
        
        if not current_report:
            raise ValueError("No final_report found in state for revision")
        
        try:
            if revision_type == "full":
                # Full report revision
                print("📝 Revising full report based on feedback...\n")
                print(f"💬 Feedback: {revision_feedback[:200]}...\n")
                
                revised_report = await self._revise_full_report(current_report, revision_feedback)
                
                # Update state with the revised ENGLISH report
                state["final_report"] = revised_report
                state["revision_count"] = state.get("revision_count", 0) + 1
                
                print(f"✅ Full report revised!")
                print(f"\n📊 Statistics:")
                print(f"   - Original length: {len(current_report)} chars")
                print(f"   - Revised length: {len(revised_report)} chars")
                print(f"   - Revision count: {state['revision_count']}\n")
            
            else:
                # Section-specific revision (future enhancement)
                print("⚠️  Section-specific revision not yet implemented")
                print("   Falling back to full report revision\n")
                
                revised_report = await self._revise_full_report(current_report, revision_feedback)
                state["final_report"] = revised_report
                state["revision_count"] = state.get("revision_count", 0) + 1
            
            return state
        
        except Exception as e:
            print(f"❌ Error in RevisionAgent: {str(e)}")
            raise
    
    async def _revise_full_report(
        self,
        current_report: str,
        user_feedback: str
    ) -> str:
        """
        전체 보고서 개선
        
        Args:
            current_report: 현재 영문 보고서
            user_feedback: 사용자 피드백
        
        Returns:
            개선된 영문 보고서
        """
        # Run revision chain
        revised_report = await self.full_revision_chain.ainvoke({
            "current_report": current_report[:50000],  # Limit to avoid token overflow
            "user_feedback": user_feedback
        })
        
        return revised_report
    
    async def revise_section(
        self,
        section_name: str,
        current_content: str,
        issues: List[str],
        improvement_requests: List[str]
    ) -> str:
        """
        특정 섹션만 개선
        
        Args:
            section_name: 섹션 이름
            current_content: 현재 내용
            issues: 문제점 리스트
            improvement_requests: 개선 요청 리스트
        
        Returns:
            개선된 섹션 내용
        """
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        requests_text = "\n".join(f"- {req}" for req in improvement_requests)
        
        revised_section = await self.section_revision_chain.ainvoke({
            "section_name": section_name,
            "current_content": current_content,
            "issues": issues_text or "No specific issues",
            "improvement_requests": requests_text or "General improvement needed"
        })
        
        return revised_section


