"""
LangGraph Workflow (Clean Architecture with Design Patterns)

전체 AI-Robotics Report Generator 파이프라인
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Core
from src.graph.state import PipelineState, create_initial_state
from src.graph.nodes import bind_nodes
from src.graph.edges import route_after_writer
from src.core.settings import get_settings
from src.agents.base.agent_config import AgentConfig
from src.tools.base.tool_config import ToolConfig

# Agents
from src.agents.planning_agent import PlanningAgent
from src.agents.data_collection_agent import DataCollectionAgent
from src.agents.writer_agent import WriterAgent

# LLMs
from src.llms.content_analysis_llm import ContentAnalysisLLM
from src.llms.report_synthesis_llm import ReportSynthesisLLM
from src.llms.revision_llm import RevisionLLM

# Utils
from src.utils.planning_util import PlanningUtil, ResearchPlanningUtil
from src.utils.refine_plan_util import RefinePlanUtil
from src.utils.feedback_classifier_util import FeedbackClassifierUtil
from src.utils.data_collect_util import RAGUtilWrapper, NewsCrawlerUtilWrapper

# Tools
from src.tools.arxiv_tool import ArxivTool
from src.tools.rag_tool import RAGTool
from src.tools.news_crawler_tool import NewsCrawlerTool
from src.tools.revision_tool import RevisionTool
from src.tools.recollection_tool import RecollectionTool

load_dotenv()


class WorkflowBuilder:
    """
    Workflow Builder (Builder Pattern)
    Agent와 Tool을 IoC Container와 Factory를 통해 생성하고 LangGraph Workflow를 조립
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        self.settings = get_settings()
        self.model = model
        self.temperature = temperature

        # LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key or self.settings.openai_api_key
        )
        
        self.shared_store = {
            "rag": [],
            "news": []
        }

        # Components (lazy initialization)
        self._agents = None
        self._utils = None
        self._tools = None

    def _build_tools(self) -> Dict[str, Any]:
        """Build all tools"""
        if self._tools is not None:
            return self._tools

        self._tools = {
            'arxiv': ArxivTool(
                config=ToolConfig(name="ArxivTool", description="Search arXiv papers", timeout=300, retry_count=3)
            ),
            'rag': RAGTool(
                config=ToolConfig(name="RAGTool", description="Retrieve from reference documents", timeout=3000, retry_count=2),
                settings=self.settings
            ),
            'news': NewsCrawlerTool(
                config=ToolConfig(name="NewsCrawlerTool", description="Crawl news articles", timeout=180, retry_count=3)
            ),
            'revision': RevisionTool(),
            'recollection': RecollectionTool()
        }

        return self._tools

    def _build_utils(self) -> Dict[str, Any]:
        """Build all utilities"""
        if self._utils is not None:
            return self._utils

        # Planning utils
        planning_util = PlanningUtil(llm=self.llm)
        refinement_util = ResearchPlanningUtil(llm=self.llm)
        
        # Tools 가져오기
        tools = self._build_tools()

        self._utils = {
            'planning': planning_util,
            'refinement': refinement_util,
            'refine_plan': RefinePlanUtil(refinement_tool=refinement_util),
            'feedback_classifier': FeedbackClassifierUtil(llm=self.llm),
            
            'rag_wrapper': RAGUtilWrapper(
                rag_tool=tools['rag'], 
                result_store=self.shared_store
            ),
            'news_wrapper': NewsCrawlerUtilWrapper(
                news_tool=tools['news'], 
                result_store=self.shared_store
            )
        }

        return self._utils

    def _build_agents(self) -> Dict[str, Any]:
        """Build all agents and LLMs"""
        if self._agents is not None:
            return self._agents

        tools = self._build_tools()
        utils = self._build_utils()

        # Base config factory
        def create_config(name: str, description: str) -> AgentConfig:
            return AgentConfig(
                name=name, description=description,
                model_name=self.model, temperature=self.temperature, retry_count=3
            )

        self._agents = {
            'planning': PlanningAgent(
                llm=self.llm,
                tools=[utils['planning'], utils['refinement']],
                config=create_config("PlanningAgent", "Planning and data collection strategy")
            ),
            'data_collection': DataCollectionAgent(
                llm=self.llm,
                tools=[utils['rag_wrapper'], utils['news_wrapper']],
                raw_tools=[tools['arxiv'], tools['rag'], tools['news']],
                config=create_config("DataCollectionAgent", "Data collection with quality check"),
                settings=self.settings,
                result_store=self.shared_store
            ),
            'content_analysis': ContentAnalysisLLM(
                llm=self.llm,
                tools=[tools['rag']],
                config=create_config("ContentAnalysisLLM", "Content analysis and section generation")
            ),
            'report_synthesis': ReportSynthesisLLM(
                llm=self.llm,
                tools=[],
                config=create_config("ReportSynthesisLLM", "Summary and conclusion generation")
            ),
            'writer': WriterAgent(
                llm=self.llm,
                tools=[tools['revision'], tools['recollection']],
                config=create_config("WriterAgent", "Report assembly and review")
            ),
        }

        return self._agents

    def build(self) -> StateGraph:
        """Build and compile the workflow"""
        agents = self._build_agents()
        utils = self._build_utils()

        # Create workflow
        workflow = StateGraph(PipelineState)

        # Bind nodes (EvaluationAgent 제거)
        nodes = bind_nodes(
            planning_agent=agents['planning'],
            data_collection_agent=agents['data_collection'],
            content_analysis_agent=agents['content_analysis'],
            report_synthesis_agent=agents['report_synthesis'],
            writer_agent=agents['writer'],
            refine_plan_tool=utils['refine_plan'],
            feedback_classifier_tool=utils['feedback_classifier']
        )

        # Add nodes
        for node_name, node_func in nodes.items():
            workflow.add_node(node_name, node_func)

        # Define Edges
        workflow.add_edge(START, "planning")
        workflow.add_edge("planning", "data_collection")
        workflow.add_edge("data_collection", "content_analysis")
        workflow.add_edge("content_analysis", "report_synthesis")
        workflow.add_edge("report_synthesis", "writer")
        
        workflow.add_conditional_edges(
            "writer",
            route_after_writer,
            {
                "end": END,  # 작성 완료 시 종료
                "writer": "writer",
                "data_collection": "data_collection"
            }
        )
        
        return workflow.compile()


class WorkflowManager:
    """Workflow Manager (Facade Pattern)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        self.builder = WorkflowBuilder(api_key, model, temperature)
        self._workflow = None

    def create_workflow(self) -> StateGraph:
        if self._workflow is None:
            self._workflow = self.builder.build()
        return self._workflow

    async def run_workflow(
        self,
        user_input: str,
        config: Optional[dict] = None
    ) -> PipelineState:
        print(f"\n{'='*60}")
        print(f"🚀 AI-Robotics Report Generator")
        print(f"{'='*60}\n")
        print(f"Topic: {user_input}")
        
        initial_state = create_initial_state(user_input)
        workflow = self.create_workflow()

        try:
            final_state = await workflow.ainvoke(initial_state, config=config)
            print(f"\n✅ Workflow Completed!")
            return final_state
        except Exception as e:
            print(f"\n❌ Workflow Failed: {str(e)}")
            raise

    def visualize_workflow(self) -> str:
        try:
            workflow = self.create_workflow()
            return workflow.get_graph().draw_mermaid()
        except Exception as e:
            return None

def create_workflow_manager(api_key=None, model="gpt-4o-mini", temperature=0.0):
    return WorkflowManager(api_key, model, temperature)

async def run_report_generation(user_input, api_key=None, model="gpt-4o-mini", temperature=0.0, config=None):
    manager = create_workflow_manager(api_key, model, temperature)
    return await manager.run_workflow(user_input, config=config)