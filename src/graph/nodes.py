"""
LangGraph Nodes (Clean Architecture)

각 노드는 단일 책임 원칙을 따르며, 에러 처리와 로깅을 중앙화
"""

from typing import Dict, Any, Callable
from datetime import datetime
from functools import partial, wraps
import json

from src.graph.state import PipelineState, WorkflowStatus
from src.cli.human_review import ProgressDisplay

# Agents
from src.agents.planning_agent import PlanningAgent
from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.writer_agent import WriterAgent

# LLMs
from src.llms.content_analysis_llm import ContentAnalysisLLM
from src.llms.report_synthesis_llm import ReportSynthesisLLM
from src.llms.revision_llm import RevisionLLM

# Utils
from src.utils.refine_plan_util import RefinePlanUtil

# Progress Display
progress = ProgressDisplay()


# =========================================================
# Decorators
# =========================================================

def handle_node_error(phase_name: str, error_status: str):
    """Decorator for node error handling"""
    def decorator(func):
        @wraps(func)
        async def wrapper(state: PipelineState, **kwargs) -> PipelineState:
            try:
                return await func(state, **kwargs)
            except Exception as e:
                progress.show_error(f"{phase_name} failed: {str(e)}")
                state["status"] = error_status
                state["error"] = str(e)
                state["updated_at"] = datetime.now().isoformat()
                raise
        return wrapper
    return decorator


# =========================================================
# Nodes
# =========================================================

@handle_node_error("Planning", "planning_failed")
async def planning_node(
    state: PipelineState,
    *,
    planning_agent: PlanningAgent,
    refine_plan_tool: RefinePlanUtil,
) -> PipelineState:
    """Phase 1: Planning & Review"""
    progress.show_phase_start("Phase 1: Planning & Review", "Analyzing topic and creating plan")

    # Generate initial plan
    progress.show_agent_start("Planning Agent", "Analyzing topic")
    new_state = await planning_agent.execute(state)
    progress.show_agent_complete("Planning Agent", f"Generated {len(new_state.get('keywords', []))} keywords")

    # Human review & refinement
    progress.show_agent_start("Planning Agent", "Human review")

    planning_output = new_state.get("planning_output")
    if not planning_output:
        raise ValueError("planning_output not found")

    from src.core.models.planning_model import PlanningOutput

    final_plan_json = await refine_plan_tool._arun(
        initial_plan=planning_output.model_dump(),
        max_attempts=10
    )

    final_plan = PlanningOutput(**json.loads(final_plan_json))

    new_state.update({
        "planning_output": final_plan,
        "keywords": final_plan.keywords,
        "human_review_1": True,
        "status": "planning_accepted",
        "updated_at": datetime.now().isoformat()
    })

    progress.show_agent_complete("Planning Agent", "Plan approved")
    return new_state


@handle_node_error("Data Collection", "data_collection_failed")
async def data_collection_node(
    state: PipelineState,
    *,
    data_collection_agent: DataCollectionAgent,
) -> PipelineState:
    """Phase 2: Data Collection"""
    progress.show_phase_start("Phase 2: Data Collection", "ArXiv → RAG → News")

    progress.show_agent_start("Data Collection Agent", "Collecting data")
    new_state = await data_collection_agent.execute(state)

    collection_status = new_state.get("collection_status")
    if collection_status:
        quality_score = getattr(collection_status, 'quality_score', 0.0)
        items = getattr(collection_status, 'items_collected', {})
        progress.show_agent_complete(
            "Data Collection Agent",
            f"Quality: {quality_score:.2f} | arXiv={items.get('arxiv', 0)}, News={items.get('news', 0)}"
        )
    else:
        progress.show_agent_complete("Data Collection Agent", "Completed")

    return new_state


@handle_node_error("Content Analysis", "analysis_failed")
async def content_analysis_node(
    state: PipelineState,
    *,
    content_analysis_agent: ContentAnalysisLLM,
) -> PipelineState:
    """Phase 3: Content Analysis"""
    progress.show_phase_start("Phase 3: Content Analysis", "Analyzing and generating sections")

    progress.show_agent_start("Content Analysis LLM", "Analyzing")
    new_state = await content_analysis_agent.execute(state)

    trends = new_state.get("trends", [])
    sections = new_state.get("section_contents", {})
    citations = new_state.get("citations", [])

    progress.show_agent_complete(
        "Content Analysis LLM",
        f"Trends: {len(trends)}, Sections: {len(sections)}, Citations: {len(citations)}"
    )

    return new_state


@handle_node_error("Report Synthesis", "synthesis_failed")
async def report_synthesis_node(
    state: PipelineState,
    *,
    report_synthesis_agent: ReportSynthesisLLM,
) -> PipelineState:
    """Phase 4: Report Synthesis"""
    progress.show_phase_start("Phase 4: Report Synthesis", "Generating summary and conclusion")

    progress.show_agent_start("Report Synthesis LLM", "Synthesizing")
    new_state = await report_synthesis_agent.execute(state)
    progress.show_agent_complete("Report Synthesis LLM", "Complete")

    return new_state


@handle_node_error("Writer", "writer_failed")
async def writer_node(
    state: PipelineState,
    *,
    writer_agent: WriterAgent,
) -> PipelineState:
    """Phase 5: Writer & Review"""
    progress.show_phase_start("Phase 5: Writer & Review", "Assembling and reviewing")

    progress.show_agent_start("Writer Agent", "Assembling")
    new_state = await writer_agent.execute(state)

    status = new_state.get("status", "completed")
    status_msg = {
        "completed": "Report accepted",
        "needs_revision": "Revision requested",
        "needs_recollection": "Recollection requested"
    }.get(status, status)

    progress.show_agent_complete("Writer Agent", status_msg)
    return new_state


async def end_node(
    state: PipelineState,
    *,
    writer_agent: WriterAgent
) -> PipelineState:
    """Workflow End Node"""
    final_status = state.get("status", "unknown")

    if final_status == "planning_rejected":
        progress.show_warning("⚠️ Workflow terminated: Planning rejected")
        state.update({"status": "workflow_complete", "updated_at": datetime.now().isoformat()})
        return state

    if "failed" in final_status:
        progress.show_error(f"❌ Workflow failed: {final_status}")
        state.update({"status": "workflow_complete", "updated_at": datetime.now().isoformat()})
        return state

    progress.show_info("✅ Workflow completed successfully")

    try:
        english_report = state.get("final_report", "")
        if not english_report:
            progress.show_warning("⚠️ No report content")
            state.update({"status": "workflow_complete", "updated_at": datetime.now().isoformat()})
            return state

        # Translate
        print("\n🌐 Translating to Korean...")
        try:
            korean_report = await writer_agent._translate_to_korean(english_report)
            print("✅ Translation complete")
            state["final_report_korean"] = korean_report
        except Exception as e:
            progress.show_error(f"Translation failed: {e}")
            korean_report = english_report
            state["final_report_korean"] = english_report
            state["translation_failed"] = True

        # Generate documents
        folder_name = state.get("folder_name", "report_output")
        output_dir = f"data/reports/{folder_name}"

        print(f"\n{'='*60}\n📄 Generating Documents\n{'='*60}\n")

        # DOCX
        from src.document.docx_generator import generate_docx

        docx_path = f"{output_dir}/final_report_korean.docx"
        print("📝 Generating DOCX...")

        try:
            docx_file = generate_docx(
                markdown_content=korean_report,
                output_path=docx_path,
                title=state.get("user_input", "AI-Robotics Trend Report")
            )
            state["docx_path"] = docx_file
            print(f"✅ DOCX saved: {docx_file}\n")
        except Exception as e:
            progress.show_error(f"DOCX failed: {e}")
            docx_file = None

        # PDF
        if docx_file:
            from src.document.pdf_converter import convert_to_pdf

            pdf_path = f"{output_dir}/final_report_korean.pdf"
            print("📄 Generating PDF...")

            try:
                pdf_file = convert_to_pdf(docx_path=docx_file, pdf_path=pdf_path)
                state["pdf_path"] = pdf_file
                print(f"✅ PDF saved: {pdf_file}\n")
            except Exception as e:
                progress.show_error(f"PDF failed: {e}")
                state["pdf_path"] = docx_file

        print(f"{'='*60}\n✅ Documents generated!\n{'='*60}\n")

    except Exception as e:
        progress.show_error(f"Document generation error: {e}")
        import traceback
        traceback.print_exc()

    state.update({"status": "workflow_complete", "updated_at": datetime.now().isoformat()})
    return state


# =========================================================
# Binding
# =========================================================

def bind_nodes(
    *,
    planning_agent: PlanningAgent,
    data_collection_agent: DataCollectionAgent,
    content_analysis_agent: ContentAnalysisLLM,
    report_synthesis_agent: ReportSynthesisLLM,
    writer_agent: WriterAgent,
    revision_agent: RevisionLLM,
    refine_plan_tool: RefinePlanUtil,
    feedback_classifier_tool: Any = None,
) -> Dict[str, Callable]:
    """Bind dependencies to nodes"""
    return {
        "planning": partial(planning_node, planning_agent=planning_agent, refine_plan_tool=refine_plan_tool),
        "data_collection": partial(data_collection_node, data_collection_agent=data_collection_agent),
        "content_analysis": partial(content_analysis_node, content_analysis_agent=content_analysis_agent),
        "report_synthesis": partial(report_synthesis_node, report_synthesis_agent=report_synthesis_agent),
        "writer": partial(writer_node, writer_agent=writer_agent),
        "end": partial(end_node, writer_agent=writer_agent),
    }


__all__ = [
    "planning_node",
    "data_collection_node",
    "content_analysis_node",
    "report_synthesis_node",
    "writer_node",
    "end_node",
    "bind_nodes",
]
