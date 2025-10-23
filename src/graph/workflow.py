"""
LangGraph Workflow

전체 AI-Robotics Report Generator 파이프라인 조립

Workflow:
    START
      ↓
    Planning Agent (초기 계획 + Human Review + Refinement)
      ↓
    Data Collection Agent (with internal quality check + retry)
      ↓
    Content Analysis Agent (with LCEL + RAG)
      ↓
    END
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from dotenv import load_dotenv
import asyncio
    
# State
from src.graph.state import PipelineState, create_initial_state

# Nodes - bind_nodes만 import
from src.graph.nodes import bind_nodes

# Agents
from langchain_openai import ChatOpenAI
from src.agents.planning_agent import PlanningAgent
from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.content_analysis_agent import ContentAnalysisAgent
from src.agents.report_synthesis_agent import ReportSynthesisAgent
from src.agents.writer_agent import WriterAgent
from src.agents.revision_agent import RevisionAgent
from src.agents.base.agent_config import AgentConfig

# Tools
from src.tools.planning_tool import PlanningTool, ResearchPlanningTool
from src.tools.refine_plan_tool import RefinePlanTool
from src.tools.feedback_classifier_tool import FeedbackClassifierTool
from src.tools.arxiv_tool import ArxivTool
from src.tools.rag_tool import RAGTool
from src.tools.news_crawler_tool import NewsCrawlerTool
from src.tools.base.tool_config import ToolConfig

# Settings
from src.core.settings import get_settings

load_dotenv()


class WorkflowManager:
    """
    워크플로우 관리 클래스
    
    Agent와 Tool 초기화 및 워크플로우 실행을 담당
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0
    ):
        """
        WorkflowManager 초기화
        
        Args:
            api_key: OpenAI API Key (None이면 환경변수에서 로드)
            model: 사용할 LLM 모델
            temperature: LLM temperature
        """
        self.settings = get_settings()
        self.model = model
        self.temperature = temperature
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
        # Agents (초기화 후 설정됨)
        self.planning_agent = None
        self.data_collection_agent = None
        self.content_analysis_agent = None
        
        # 초기화
        self._initialize_tools_and_agents()
    
    def _initialize_tools_and_agents(self):
        """Tools와 Agents 초기화"""
        
        print("\n" + "="*60)
        print("🔧 Initializing Tools and Agents...")
        print("="*60 + "\n")
        
        # ========================================
        # 1. Tools 초기화
        # ========================================
        
        print("📦 Initializing Tools...")
        
        # PlanningTool
        planning_tool = PlanningTool(llm=self.llm)
        print(f"   ✓ PlanningTool")
        
        # ResearchPlanningTool
        refinement_tool = ResearchPlanningTool(llm=self.llm)
        print(f"   ✓ ResearchPlanningTool")
        
        # RefinePlanTool (Human Review + Refinement)
        self.refine_plan_tool = RefinePlanTool(refinement_tool=refinement_tool)
        print(f"   ✓ RefinePlanTool")
        
        # FeedbackClassifierTool (Human Review 2 - Feedback Classification)
        self.feedback_classifier_tool = FeedbackClassifierTool(llm=self.llm)
        print(f"   ✓ FeedbackClassifierTool")
        
        # ArxivTool
        arxiv_config = ToolConfig(
            name="ArxivTool",
            description="Search and retrieve academic papers from arXiv",
            timeout=300,
            retry_count=3
        )
        arxiv_tool = ArxivTool(config=arxiv_config)
        print(f"   ✓ {arxiv_config.name}")
        
        # RAGTool
        rag_config = ToolConfig(
            name="RAGTool",
            description="Hybrid retrieval from reference documents (FTSG, WEF)",
            timeout=3000,
            retry_count=2
        )
        rag_tool = RAGTool(config=rag_config, settings=self.settings)
        print(f"   ✓ {rag_config.name}")
        
        # NewsCrawlerTool
        news_config = ToolConfig(
            name="NewsCrawlerTool",
            description="Crawl news articles from multiple sources",
            timeout=180,
            retry_count=3
        )
        news_tool = NewsCrawlerTool(config=news_config)
        print(f"   ✓ {news_config.name}\n")
        
        # ========================================
        # 2. Agents 초기화
        # ========================================
        
        print("🤖 Initializing Agents...")
        
        # Planning Agent (with planning tools)
        planning_config = AgentConfig(
            name="PlanningAgent",
            description="Analyzes user input and creates data collection plan",
            model_name=self.model,
            temperature=self.temperature,
            retry_count=3
        )
        self.planning_agent = PlanningAgent(
            llm=self.llm,
            tools=[planning_tool, refinement_tool],  # PlanningTool + ResearchPlanningTool
            config=planning_config
        )
        print(f"   ✓ {planning_config.name}")
        
        # Data Collection Agent (RAG + News tools for ReAct)
        from src.tools.data_collect_tool import (
            RAGToolWrapper,
            NewsCrawlerWrapper
        )
        
        # Wrap RAG and News tools for LangChain compatibility (ArXiv는 직접 실행)
        rag_wrapper = RAGToolWrapper(rag_tool=rag_tool)
        news_wrapper = NewsCrawlerWrapper(news_tool=news_tool)
        
        data_collection_config = AgentConfig(
            name="DataCollectionAgent",
            description="Collects data: ArXiv first, then ReAct with RAG + News",
            model_name=self.model,
            temperature=self.temperature,
            retry_count=3
        )
        self.data_collection_agent = DataCollectionAgent(
            llm=self.llm,
            tools=[rag_wrapper, news_wrapper],  # ReAct Agent가 사용할 tool (RAG + News만)
            raw_tools=[arxiv_tool, rag_tool, news_tool],  # 직접 접근용 (ArXiv, citation)
            config=data_collection_config,
            settings=self.settings
        )
        print(f"   ✓ {data_collection_config.name}")
        
        # Content Analysis Agent (needs RAG tool for citation)
        content_analysis_config = AgentConfig(
            name="ContentAnalysisAgent",
            description="Analyzes collected data and generates report sections",
            model_name=self.model,
            temperature=self.temperature,
            retry_count=2
        )
        self.content_analysis_agent = ContentAnalysisAgent(
            llm=self.llm,
            tools=[rag_tool],  # RAG tool for additional reference
            config=content_analysis_config
        )
        print(f"   ✓ {content_analysis_config.name}")
        
        # Report Synthesis Agent
        report_synthesis_config = AgentConfig(
            name="ReportSynthesisAgent",
            description="Generates Summary, Introduction, Conclusion, References, Appendix",
            model_name=self.model,
            temperature=self.temperature,
            retry_count=2
        )
        self.report_synthesis_agent = ReportSynthesisAgent(
            llm=self.llm,
            tools=[],
            config=report_synthesis_config
        )
        print(f"   ✓ {report_synthesis_config.name}")
        
        # Writer Agent
        writer_config = AgentConfig(
            name="WriterAgent",
            description="Assembles all sections into final markdown report",
            model_name=self.model,
            temperature=self.temperature,
            retry_count=2
        )
        self.writer_agent = WriterAgent(
            llm=self.llm,
            tools=[],
            config=writer_config
        )
        print(f"   ✓ {writer_config.name}")
        
        # Revision Agent
        revision_config = AgentConfig(
            name="RevisionAgent",
            description="Revises report based on user feedback",
            model_name=self.model,
            temperature=self.temperature,
            retry_count=2
        )
        self.revision_agent = RevisionAgent(
            llm=self.llm,
            tools=[],
            config=revision_config
        )
        print(f"   ✓ {revision_config.name}\n")
        
        print("="*60)
        print("✅ All Tools and Agents initialized successfully!")
        print("="*60 + "\n")
    
    def get_agents(self) -> Dict[str, Any]:
        """초기화된 Agents 반환"""
        return {
            'planning_agent': self.planning_agent,
            'data_collection_agent': self.data_collection_agent,
            'content_analysis_agent': self.content_analysis_agent,
            'report_synthesis_agent': self.report_synthesis_agent,
            'writer_agent': self.writer_agent,
            'revision_agent': self.revision_agent
        }
    
    def create_workflow(self) -> StateGraph:
        """
        LangGraph 워크플로우 생성
        
        Returns:
            컴파일된 StateGraph
        """
        # StateGraph 초기화
        workflow = StateGraph(PipelineState)
        
        # ========================================
        # 노드 추가 (bind_nodes 사용)
        # ========================================
        
        # Agents를 nodes에 주입
        agents = self.get_agents()
        
        # bind_nodes()로 모든 노드 바인딩
        bound_nodes = bind_nodes(
            planning_agent=agents["planning_agent"],
            data_collection_agent=agents["data_collection_agent"],
            content_analysis_agent=agents["content_analysis_agent"],
            report_synthesis_agent=agents["report_synthesis_agent"],
            writer_agent=agents["writer_agent"],
            revision_agent=agents["revision_agent"],
            refine_plan_tool=self.refine_plan_tool,
            feedback_classifier_tool=self.feedback_classifier_tool
        )
        
        # 노드 추가
        workflow.add_node("planning", bound_nodes["planning"])
        workflow.add_node("data_collection", bound_nodes["data_collection"])
        workflow.add_node("content_analysis", bound_nodes["content_analysis"])
        workflow.add_node("report_synthesis", bound_nodes["report_synthesis"])
        workflow.add_node("writer", bound_nodes["writer"])
        workflow.add_node("revision", bound_nodes["revision"])
        workflow.add_node("end", bound_nodes["end"])
        
        # ========================================
        # 엣지 추가
        # ========================================
        
        # Import routing functions
        from src.graph.edges import route_after_writer, route_after_revision
        
        # START -> planning (planning 내부에서 human review 처리)
        workflow.add_edge(START, "planning")
        
        # planning -> data_collection (human review 통과 후)
        workflow.add_edge("planning", "data_collection")
        
        # data_collection -> content_analysis
        workflow.add_edge("data_collection", "content_analysis")
        
        # content_analysis -> report_synthesis
        workflow.add_edge("content_analysis", "report_synthesis")
        
        # report_synthesis -> writer
        workflow.add_edge("report_synthesis", "writer")
        
        # writer -> conditional routing (WriterAgent 내부에서 human review 처리)
        workflow.add_conditional_edges(
            "writer",
            route_after_writer,
            {
                "end": "end",  # Accept
                "revision": "revision",  # Minor revision
                "data_collection": "data_collection"  # Data recollection (restart)
            }
        )
        
        # revision -> writer (재조립 및 재검토)
        workflow.add_conditional_edges(
            "revision",
            route_after_revision,
            {
                "writer": "writer"
            }
        )
        
        # end -> END
        workflow.add_edge("end", END)
        
        # ========================================
        # 컴파일
        # ========================================
        
        compiled_workflow = workflow.compile()
        
        return compiled_workflow
    
    async def run_workflow(
        self,
        user_input: str,
        config: Optional[dict] = None
    ) -> PipelineState:
        """
        워크플로우 실행
        
        Args:
            user_input: 사용자 입력 주제
            config: 추가 설정 (optional)
        
        Returns:
            최종 state
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting AI-Robotics Report Generator Workflow")
        print(f"{'='*60}\n")
        print(f"Topic: {user_input}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 초기 state 생성
        initial_state = create_initial_state(user_input)
        
        # 2. 워크플로우 생성
        workflow = self.create_workflow()
        
        # 3. 워크플로우 실행
        try:
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            print(f"\n{'='*60}")
            print(f"✅ Workflow completed successfully!")
            print(f"{'='*60}\n")
            print(f"Final Status: {final_state.get('status')}")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            return final_state
        
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ Workflow failed!")
            print(f"{'='*60}\n")
            print(f"Error: {str(e)}\n")
            raise
    
    def visualize_workflow(self, output_path: str = "workflow_graph.png") -> str:
        """
        워크플로우 그래프 시각화
        
        Args:
            output_path: 저장할 이미지 파일 경로
        
        Returns:
            Mermaid 다이어그램 코드
        """
        try:
            workflow = self.create_workflow()
            
            # Mermaid 다이어그램 생성
            mermaid_code = workflow.get_graph().draw_mermaid()
            
            print(f"\nWorkflow Graph (Mermaid):")
            print("="*60)
            print(mermaid_code)
            print("="*60)
            
            return mermaid_code
        
        except Exception as e:
            print(f"❌ Failed to visualize workflow: {e}")
            return None


# ========================================
# 편의 함수들
# ========================================

def create_workflow_manager(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.0
) -> WorkflowManager:
    """
    WorkflowManager 생성 (편의 함수)
    
    Args:
        api_key: OpenAI API Key
        model: LLM 모델
        temperature: Temperature
    
    Returns:
        초기화된 WorkflowManager
    """
    return WorkflowManager(
        api_key=api_key,
        model=model,
        temperature=temperature
    )


async def run_report_generation(
    user_input: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    config: Optional[dict] = None
) -> PipelineState:
    """
    전체 보고서 생성 파이프라인 실행 (편의 함수)
    
    Args:
        user_input: 사용자 주제
        api_key: OpenAI API Key
        model: LLM 모델
        temperature: Temperature
        config: 추가 설정
    
    Returns:
        최종 state
    """
    manager = create_workflow_manager(
        api_key=api_key,
        model=model,
        temperature=temperature
    )
    
    return await manager.run_workflow(user_input, config=config)


# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    import argparse
    import sys
    import platform
    
    # Windows에서 asyncio 정책 설정
    if platform.system() == 'Windows':
        # ProactorEventLoop 사용 (Windows 기본값, Python 3.8+)
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the workflow now")
    parser.add_argument("--topic", type=str, default="humanoid robots in manufacturing")
    parser.add_argument("--visualize", action="store_true", help="Visualize workflow graph")
    args = parser.parse_args()

    if args.visualize:
        # 워크플로우 시각화
        manager = create_workflow_manager()
        manager.visualize_workflow()
    elif args.run:
        # 워크플로우 실행
        manager = create_workflow_manager()
        
        # asyncio.run() 사용 (Python 3.7+ 권장 방식)
        try:
            final_state = asyncio.run(manager.run_workflow(args.topic))
            
            print(f"\n📊 Results Summary:")
            print(f"   Keywords: {len(final_state.get('keywords', []))} keywords")
            print(f"   Trends: {len(final_state.get('trends', []))} trends")
            print(f"   Sections: {len(final_state.get('section_contents', {}))} sections")
            print(f"   Citations: {len(final_state.get('citations', []))} citations")
            print(f"   Status: {final_state.get('status')}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage:")
        print("  python -m src.graph.workflow --run --topic 'your topic'")
        print("  python -m src.graph.workflow --visualize")