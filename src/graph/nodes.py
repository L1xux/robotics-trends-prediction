# src/graph/nodes.py

"""
LangGraph Nodes (Updated for LCEL Planning Agent)
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime
from functools import partial
import dataclasses
import json

from pydantic import BaseModel

# State
from src.graph.state import PipelineState

# CLI
from src.cli.human_review import ReviewCLI, ProgressDisplay

# Agents (타입 힌트용)
from src.agents.planning_agent import PlanningAgent
from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.content_analysis_agent import ContentAnalysisAgent
from langchain_core.language_models import BaseChatModel


# Progress Display (표시 전용)
progress = ProgressDisplay()


# =========================================================
# Utilities
# =========================================================

def _to_dict_safe(obj: Any) -> Dict[str, Any]:
    """
    임의의 객체(obj)를 dict로 정규화.
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    # pydantic v2
    if isinstance(obj, BaseModel) and hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    # pydantic v1 (혹은 호환)
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass

    # dataclass
    if dataclasses.is_dataclass(obj):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            pass

    # fallback: 자주 쓰는 필드만 수집
    out: Dict[str, Any] = {}
    for key in ("topic", "folder_name", "keywords", "collection_plan"):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


# =========================================================
# Nodes
# =========================================================

async def planning_node(
    state: PipelineState,
    *,
    planning_agent: PlanningAgent,
    refine_plan_tool: Any,
) -> PipelineState:
    """
    Phase 1: Planning (초기 계획 + Human Review + Refinement)
    - 사용자 주제 분석
    - 키워드 확장
    - 폴더명 생성
    - 데이터 수집 계획 수립
    - Human Review 및 개선 (refine_plan_tool)
    """
    progress.show_phase_start(
        "Phase 1: Planning & Review",
        "Analyzing topic, creating plan, and human review",
    )

    try:
        # Step 1: 초기 계획 생성
        progress.show_agent_start("Planning Agent", "Analyzing user topic")
        new_state = await planning_agent.execute(state)
        progress.show_agent_complete(
            "Planning Agent",
            f"Generated {len(new_state.get('keywords', []))} keywords",
        )
        
        # Step 2: Human Review & Refinement (refine_plan_tool)
        progress.show_agent_start("Planning Agent", "Human review & refinement")
        
        planning_output = new_state.get("planning_output")
        if not planning_output:
            raise ValueError("planning_output not found in state")
        
        initial_plan_dict = planning_output.model_dump()
        
        # Call refine_plan_tool (내부에서 ReviewCLI 처리)
        final_plan_json = await refine_plan_tool._arun(
            initial_plan=initial_plan_dict,
            max_attempts=10
        )
        
        # Parse final plan
        final_plan_data = json.loads(final_plan_json)
        from src.core.models.planning_model import PlanningOutput
        final_plan = PlanningOutput(**final_plan_data)
        
        # Update state with final approved plan
        new_state["planning_output"] = final_plan
        new_state["keywords"] = final_plan.keywords
        new_state["human_review_1"] = True
        new_state["status"] = "planning_accepted"
        new_state["updated_at"] = datetime.now().isoformat()
        
        progress.show_agent_complete(
            "Planning Agent",
            "Plan approved - proceeding to data collection"
        )
        
        return new_state

    except Exception as e:
        progress.show_error(f"Planning failed: {str(e)}")
        state["status"] = "planning_failed"
        state["error"] = str(e)
        state["updated_at"] = datetime.now().isoformat()
        raise


async def data_collection_node(
    state: PipelineState,
    *,
    data_collection_agent: DataCollectionAgent,
) -> PipelineState:
    """
    Phase 3: Data Collection
    - ArxivTool / RAGTool / GoogleTrendsTool / NewsCrawlerTool 활용
    - 내부 품질 체크 & 필요시 재시도 (최대 3회, agent 내부 구현)
    """
    progress.show_phase_start(
        "Phase 3: Data Collection",
        "Intelligent trend-based collection: ArXiv → RAG → Trend Analysis → Trends/News",
    )

    try:
        progress.show_agent_start(
            "Data Collection Agent",
            "Starting intelligent trend-based collection: ArXiv (all keywords) → RAG analysis → Trend extraction → Trends/News",
        )

        new_state = await data_collection_agent.execute(state)

        collection_status = new_state.get("collection_status")
        quality_check_result = new_state.get("quality_check_result")
        
        # 결과 로깅
        if collection_status:
            quality_score = getattr(collection_status, 'quality_score', 0.0)
            items = getattr(collection_status, 'items_collected', {})
            
            progress.show_agent_complete(
                "Data Collection Agent",
                f"Quality Score: {quality_score:.2f} | "
                f"Items: arXiv={items.get('arxiv', 0)}, "
                f"Trends={items.get('trends', 0)}, "
                f"News={items.get('news', 0)}"
            )
        else:
            progress.show_agent_complete(
                "Data Collection Agent",
                "Data collection completed"
            )
        
        # Quality check 결과 로깅
        if quality_check_result:
            status = getattr(quality_check_result, 'status', 'unknown')
            if status == "pass":
                progress.show_info("✅ Quality check: PASSED")
            else:
                progress.show_warning("⚠️ Quality check: Proceeded with collected data")

        return new_state

    except Exception as e:
        progress.show_error(f"Data collection failed: {str(e)}")
        state["status"] = "data_collection_failed"
        state["error"] = str(e)
        state["updated_at"] = datetime.now().isoformat()
        raise


async def content_analysis_node(
    state: PipelineState,
    *,
    content_analysis_agent: ContentAnalysisAgent,
) -> PipelineState:
    """
    Phase 4: Content Analysis (LCEL)
    - 트렌드 분류, 섹션 생성, 인용 관리
    - RAG Tool 자동 호출 (agent 내부, 최대 10회)
    """
    progress.show_phase_start(
        "Phase 4: Content Analysis",
        "Analyzing data and generating report sections (LCEL + RAG)",
    )

    try:
        progress.show_agent_start(
            "Content Analysis Agent",
            "Starting LCEL-based analysis workflow (4 stages with internal RAG calls)",
        )

        new_state = await content_analysis_agent.execute(state)

        trends = new_state.get("trends", [])
        sections = new_state.get("section_contents", {})
        citations = new_state.get("citations", [])

        # Trends 티어별 분류
        hot_trends = [t for t in trends if hasattr(t, 'is_hot_trend') and t.is_hot_trend()]
        rising_stars = [t for t in trends if hasattr(t, 'is_rising_star') and t.is_rising_star()]

        progress.show_agent_complete(
            "Content Analysis Agent",
            f"Generated {len(trends)} trends "
            f"(HOT: {len(hot_trends)}, RISING: {len(rising_stars)}), "
            f"{len(sections)} sections, {len(citations)} citations"
        )
        
        progress.show_info(f"📊 Sections: {', '.join(sections.keys())}")

        return new_state

    except Exception as e:
        progress.show_error(f"Content analysis failed: {str(e)}")
        state["status"] = "analysis_failed"
        state["error"] = str(e)
        state["updated_at"] = datetime.now().isoformat()
        raise


async def report_synthesis_node(
    state: PipelineState,
    *,
    report_synthesis_agent: Any,
) -> PipelineState:
    """
    Phase 4: Report Synthesis
    """
    progress.show_phase_start(
        "Phase 4: Report Synthesis",
        "Generating Summary, Introduction, Conclusion, References, Appendix"
    )
    
    try:
        progress.show_agent_start("Report Synthesis Agent", "Synthesizing final sections")
        new_state = await report_synthesis_agent.execute(state)
        progress.show_agent_complete("Report Synthesis Agent", "All synthesis sections generated")
        return new_state
    except Exception as e:
        progress.show_error(f"Report Synthesis failed: {str(e)}")
        state["status"] = "synthesis_failed"
        state["error"] = str(e)
        state["updated_at"] = datetime.now().isoformat()
        raise


async def writer_node(
    state: PipelineState,
    *,
    writer_agent: Any,
) -> PipelineState:
    """
    Phase 5: Writer (with Human-in-the-Loop)
    
    WriterAgent가 다음을 모두 수행:
    1. 보고서 조립
    2. 사용자 피드백 받기 (CLI)
    3. 피드백 분석 (LLM)
    4. 적절한 액션 결정 (revision or recollection)
    """
    progress.show_phase_start("Phase 5: Writer & Review", "Assembling report and getting user feedback")
    
    try:
        progress.show_agent_start("Writer Agent", "Assembling report and managing review")
        new_state = await writer_agent.execute(state)
        
        # Check status
        status = new_state.get("status", "completed")
        
        if status == "completed":
            progress.show_agent_complete("Writer Agent", "Report accepted by user")
        elif status == "needs_revision":
            progress.show_agent_complete("Writer Agent", "Revision requested - routing to RevisionAgent")
        elif status == "needs_recollection":
            progress.show_agent_complete("Writer Agent", "Data recollection requested - routing to DataCollectionAgent")
        
        return new_state
    except Exception as e:
        progress.show_error(f"Writer failed: {str(e)}")
        state["status"] = "writer_failed"
        state["error"] = str(e)
        state["updated_at"] = datetime.now().isoformat()
        raise


async def revision_node(
    state: PipelineState,
    *,
    revision_agent: Any,
) -> PipelineState:
    """
    Phase 6: Revision
    
    사용자 피드백을 기반으로 보고서를 수정하고 다시 WriterAgent로 돌아감
    """
    progress.show_phase_start("Phase 6: Revision", "Revising report based on feedback")
    
    try:
        feedback = state.get("review_feedback", "")
        if not feedback:
            raise ValueError("No review_feedback found in state")
        
        revision_type = state.get("revision_type", "full")
        
        progress.show_agent_start("Revision Agent", f"Processing {revision_type} revision")
        
        # Revise report - pass feedback and revision_type as arguments
        revised_state = await revision_agent.execute(
            state,
            revision_feedback=feedback,
            revision_type=revision_type
        )
        
        # Clear status to re-enter WriterAgent
        revised_state["status"] = "revision_completed"
        
        progress.show_agent_complete("Revision Agent", "Revision completed - routing back to Writer")
        return revised_state
    
    except Exception as e:
        progress.show_error(f"Revision failed: {str(e)}")
        state["status"] = "revision_failed"
        state["error"] = str(e)
        state["updated_at"] = datetime.now().isoformat()
        raise


async def end_node(
    state: PipelineState,
    *,
    writer_agent: Any  # WriterAgent for its translation utility
) -> PipelineState:
    """
    워크플로우 종료 노드
    
    최종 보고서를 DOCX/PDF로 생성
    """
    final_status = state.get("status", "unknown")
    
    if final_status == "planning_rejected":
        progress.show_warning("⚠️ Workflow terminated: Planning rejected by user")
        state["status"] = "workflow_complete"
        state["updated_at"] = datetime.now().isoformat()
        return state
    
    if "failed" in final_status:
        progress.show_error(f"❌ Workflow failed: {final_status}")
        state["status"] = "workflow_complete"
        state["updated_at"] = datetime.now().isoformat()
        return state
    
    # Success - Generate documents
    progress.show_info("✅ Workflow completed successfully")
    
    try:
        # Get final English report
        english_report = state.get("final_report", "")
        
        if not english_report:
            progress.show_warning("⚠️  No report content found, skipping document generation")
            state["status"] = "workflow_complete"
            state["updated_at"] = datetime.now().isoformat()
            return state
        
        # =================================================
        # 최종 번역 수행 (end_node에서 직접)
        # =================================================
        print("\n🌐 Translating final report to Korean for document generation...")
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import asyncio
        
        korean_report = ""
        try:
            # Reuse the translation logic from WriterAgent
            korean_report = await writer_agent._translate_to_korean(english_report)

            print("✅ Final translation complete.")
            state["final_report_korean"] = korean_report

        except Exception as e:
            progress.show_error(f"Final translation failed: {e}")
            progress.show_warning("⚠️  Using English report for document generation due to translation failure.")
            korean_report = english_report  # 전체 번역 실패 시 영문으로 대체
            state["final_report_korean"] = english_report
            state["translation_failed"] = True

        # =================================================
        # 문서 생성
        # =================================================

        # Generate folder name
        folder_name = state.get("folder_name", "report_output")
        output_dir = f"data/reports/{folder_name}"
        
        print(f"\n{'='*60}")
        print(f"📄 Generating Final Documents")
        print(f"{'='*60}\n")
        
        # Step 1: Generate DOCX
        from src.document.docx_generator import generate_docx
        
        docx_path = f"{output_dir}/final_report_korean.docx"
        print(f"📝 Generating DOCX...")
        
        try:
            docx_file = generate_docx(
                markdown_content=korean_report,
                output_path=docx_path,
                title=state.get("user_input", "AI-Robotics Trend Report")
            )
            state["docx_path"] = docx_file
            print(f"✅ DOCX saved: {docx_file}\n")
        except Exception as e:
            progress.show_error(f"DOCX generation failed: {e}")
            docx_file = None
        
        # Step 2: Generate PDF
        if docx_file:
            from src.document.pdf_converter import convert_to_pdf
            
            pdf_path = f"{output_dir}/final_report_korean.pdf"
            print(f"📄 Generating PDF...")
            
            try:
                pdf_file = convert_to_pdf(
                    docx_path=docx_file,
                    pdf_path=pdf_path
                )
                state["pdf_path"] = pdf_file
                print(f"✅ PDF saved: {pdf_file}\n")
            except Exception as e:
                progress.show_error(f"PDF generation failed: {e}")
                state["pdf_path"] = docx_file  # Fallback to DOCX
        
        print(f"{'='*60}")
        print(f"✅ All documents generated successfully!")
        print(f"{'='*60}\n")
    
    except Exception as e:
        progress.show_error(f"Document generation error: {e}")
        import traceback
        traceback.print_exc()
    
    state["status"] = "workflow_complete"
    state["updated_at"] = datetime.now().isoformat()
    return state


# =========================================================
# Binding helper (partial로 의존성 주입)
# =========================================================

def bind_nodes(
    *,
    planning_agent: PlanningAgent,
    data_collection_agent: DataCollectionAgent,
    content_analysis_agent: ContentAnalysisAgent,
    report_synthesis_agent: Any,
    writer_agent: Any,
    revision_agent: Any,
    refine_plan_tool: Any,
    feedback_classifier_tool: Any,
) -> Dict[str, Callable[..., Any]]:
    """
    워크플로우 구성 시 사용할 바운드 노드 함수들을 반환합니다.
    각 노드는 state 하나만 위치 인자로 받고, 나머지 의존성은 partial로 고정합니다.
    """
    return {
        "planning": partial(
            planning_node,
            planning_agent=planning_agent,
            refine_plan_tool=refine_plan_tool
        ),
        "data_collection": partial(
            data_collection_node,
            data_collection_agent=data_collection_agent,
        ),
        "content_analysis": partial(
            content_analysis_node,
            content_analysis_agent=content_analysis_agent,
        ),
        "report_synthesis": partial(
            report_synthesis_node,
            report_synthesis_agent=report_synthesis_agent,
        ),
        "writer": partial(
            writer_node,
            writer_agent=writer_agent,
        ),
        "revision": partial(
            revision_node,
            revision_agent=revision_agent,
        ),
        "end": partial(
            end_node,
            writer_agent=writer_agent  # Pass writer_agent for its translation method
        ),
    }


# =========================================================
# Export
# =========================================================

__all__ = [
    "planning_node",
    "data_collection_node",
    "content_analysis_node",
    "report_synthesis_node",
    "writer_node",
    "revision_node",
    "end_node",
    "bind_nodes",
]