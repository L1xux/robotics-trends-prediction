"""
Planning Agent (Tool-Based Architecture with LLM Tool Selection)

사용자 입력 주제를 분석하고 데이터 수집 계획을 수립하는 Agent
- LLM이 필요에 따라 tool을 선택하여 사용
- 초기 계획: PlanningTool 사용
- Refinement: ResearchPlanningUtil 사용
- Human-in-the-loop: 사용자 피드백을 받아 계속 개선
"""

import json
from datetime import datetime
from typing import List, Any, Dict, Optional
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState, WorkflowStatus
from src.core.models.planning_model import PlanningOutput
from src.cli.human_review import ReviewCLI


class ToolType(str, Enum):
    """Tool types for planning"""
    CREATE_PLAN = "create_research_plan"
    REFINE_PLAN = "refine_research_plan"


class PlanningPrompts:
    """Centralized prompts for planning agent"""

    INITIAL_PLAN = """You are a research planning assistant.

User wants to create a trend analysis report on: "{topic}"

Your task:
1. Use the 'create_research_plan' tool to generate an initial research plan
2. The tool will return a JSON with the complete plan

Call the tool now to create the plan."""

    RETRY_TOOL_USE = "Please use the 'create_research_plan' tool to generate the plan."


class PlanningAgent(BaseAgent):
    """
    Planning Agent (Tool-Based Architecture with Autonomous Tool Use)
    
    주제 분석 및 데이터 수집 계획 수립
    
    Responsibilities:
    1. 사용자 주제 분석
    2. 키워드 확장 (30-40개)
    3. 주제 정규화 (normalized_topic)
    4. 데이터 수집 계획 수립 (arXiv, Trends, News)
    5. 사용자 피드백 기반 refinement
    
    Tools:
    - PlanningTool: 초기 계획 생성
    - ResearchPlanningUtil: 피드백 기반 계획 개선
    
    **New: LLM can autonomously choose which tool to use!**
    
    Note:
        - LLM: normalized_topic 생성 + tool selection
        - Python: folder_name 생성 (타임스탬프, 초기 1회만)
    
    Output: PlanningOutput
    """
    
    MAX_ITERATIONS = 100

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any],
        config: AgentConfig
    ):
        super().__init__(llm, tools, config)
        self._planning_tool: Optional[Any] = None
        self._refinement_tool: Optional[Any] = None
        self._llm_with_tools = self.llm.bind_tools(tools)
        self._review_cli = ReviewCLI()
        self._initialize_tools(tools)

    def _initialize_tools(self, tools: List[Any]) -> None:
        """Initialize and categorize tools"""
        tool_map = {
            ToolType.CREATE_PLAN.value: "_planning_tool",
            ToolType.REFINE_PLAN.value: "_refinement_tool"
        }

        for tool in tools:
            if hasattr(tool, 'name') and tool.name in tool_map:
                setattr(self, tool_map[tool.name], tool)
    
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Planning 실행 (Tool-Based with Human-in-the-loop)

        Args:
            state: 현재 파이프라인 상태
                - state["user_input"]: 사용자 입력 주제

        Returns:
            업데이트된 상태
                - state["planning_output"]: PlanningOutput
                - state["folder_name"]: 실행 폴더명 (Python 생성)
                - state["keywords"]: 확장된 키워드 리스트
        """
        self._log_start()

        user_topic = self._validate_user_input(state)
        planning_output = await self._create_initial_plan(user_topic)
        folder_name = self._generate_folder_name(planning_output.normalized_topic)

        state.update({
            "planning_output": planning_output,
            "folder_name": folder_name,
            "keywords": planning_output.keywords,
            "status": WorkflowStatus.PLANNING_COMPLETE.value
        })

        self._log_completion(planning_output, folder_name)
        return state

    def _validate_user_input(self, state: PipelineState) -> str:
        """Validate and extract user input from state"""
        user_topic = state.get("user_input", "")
        if not user_topic:
            raise ValueError("user_input이 State에 없습니다.")
        return user_topic

    def _log_start(self) -> None:
        """Log planning start"""
        print(f"\n{'='*60}")
        print(f"🎯 Planning Agent 실행 중...")
        print(f"{'='*60}\n")

    def _log_completion(self, planning_output: PlanningOutput, folder_name: str) -> None:
        """Log planning completion"""
        print(f"\n✅ Planning Agent 완료!")
        print(f"📁 정규화된 주제: {planning_output.normalized_topic}")
        print(f"📁 폴더명: {folder_name}")
        print(f"🔑 키워드 개수: {len(planning_output.keywords)}")
        print(f"{'='*60}\n")
    
    async def _create_initial_plan(self, user_topic: str) -> PlanningOutput:
        """
        초기 계획 생성 (LLM이 tool을 자율적으로 사용)

        Args:
            user_topic: 사용자 주제

        Returns:
            PlanningOutput
        """
        print(f"📝 주제: {user_topic}")
        print(f"🔄 초기 계획 생성 중 (LLM + Tool)...\n")

        messages = [HumanMessage(content=PlanningPrompts.INITIAL_PLAN.format(topic=user_topic))]

        for iteration in range(self.MAX_ITERATIONS):
            print(f"🔄 Iteration {iteration + 1}/{self.MAX_ITERATIONS}")

            response = await self._llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                self._handle_no_tool_call(response, messages)
                continue

            planning_output = await self._execute_tool_calls(response.tool_calls, messages)
            if planning_output:
                return planning_output

        raise ValueError(f"초기 계획 생성 실패 (최대 반복 {self.MAX_ITERATIONS})")

    def _handle_no_tool_call(self, response, messages: List) -> None:
        """Handle case when LLM doesn't call any tool"""
        print(f"⚠️ LLM didn't call any tool. Response: {response.content[:100]}...")
        print(f"🔄 Prompting LLM to use tool...\n")
        messages.append(HumanMessage(content=PlanningPrompts.RETRY_TOOL_USE))

    async def _execute_tool_calls(self, tool_calls: List, messages: List) -> Optional[PlanningOutput]:
        """Execute tool calls and return PlanningOutput if successful"""
        for tool_call in tool_calls:
            print(f"🛠️ LLM calling tool: {tool_call['name']}")

            tool = self._get_tool(tool_call['name'])
            if not tool:
                print(f"❌ Tool '{tool_call['name']}' not found!")
                continue

            try:
                planning_output = await self._execute_single_tool(tool, tool_call['args'])
                print(f"✅ 초기 계획 생성 성공!\n")
                return planning_output

            except Exception as e:
                self._handle_tool_error(e, tool_call, messages)

        return None

    def _get_tool(self, tool_name: str) -> Optional[Any]:
        """Get tool by name"""
        return (self._planning_tool if tool_name == ToolType.CREATE_PLAN.value
                else self._refinement_tool if tool_name == ToolType.REFINE_PLAN.value
                else None)

    async def _execute_single_tool(self, tool: Any, args: Dict) -> PlanningOutput:
        """Execute single tool and return validated PlanningOutput"""
        result_json = await tool._arun(**args)
        planning_data = json.loads(result_json)

        # Remove unnecessary fields
        planning_data.pop('folder_name', None)
        planning_data.pop('reasoning', None)

        return PlanningOutput(**planning_data)

    def _handle_tool_error(self, error: Exception, tool_call: Dict, messages: List) -> None:
        """Handle tool execution error"""
        error_msg = str(error)
        print(f"❌ Tool execution failed: {error_msg}")
        messages.append(ToolMessage(
            content=f"Error: {error_msg}",
            tool_call_id=tool_call['id']
        ))
    
    @staticmethod
    def _generate_folder_name(normalized_topic: str) -> str:
        """
        폴더명 생성 (Python 코드에서)

        Format: {normalized_topic}_{YYYYMMDD_HHMMSS}

        Args:
            normalized_topic: 정규화된 주제

        Returns:
            폴더명 문자열
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{normalized_topic}_{timestamp}"
