# src/agents/planning_agent.py

"""
Planning Agent (Tool-Based Architecture with LLM Tool Selection)

사용자 입력 주제를 분석하고 데이터 수집 계획을 수립하는 Agent
- LLM이 필요에 따라 tool을 선택하여 사용
- 초기 계획: PlanningTool 사용
- Refinement: ResearchPlanningTool 사용  
- Human-in-the-loop: 사용자 피드백을 받아 계속 개선
"""

import json
from datetime import datetime
from typing import List, Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState
from src.core.models.planning_model import PlanningOutput
from src.cli.human_review import ReviewCLI


class PlanningAgent(BaseAgent):
    """
    Planning Agent (Tool-Based Architecture with Autonomous Tool Use)
    
    주제 분석 및 데이터 수집 계획 수립
    
    Responsibilities:
    1. 사용자 주제 분석
    2. 키워드 확장 (5-15개)
    3. 주제 정규화 (normalized_topic)
    4. 데이터 수집 계획 수립 (arXiv, Trends, News)
    5. 사용자 피드백 기반 refinement
    
    Tools:
    - PlanningTool: 초기 계획 생성
    - ResearchPlanningTool: 피드백 기반 계획 개선
    
    **New: LLM can autonomously choose which tool to use!**
    
    Note:
        - LLM: normalized_topic 생성 + tool selection
        - Python: folder_name 생성 (타임스탬프, 초기 1회만)
    
    Output: PlanningOutput
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any],
        config: AgentConfig
    ):
        super().__init__(llm, tools, config)
        
        # Tools 분리
        self.planning_tool = None  # PlanningTool
        self.refinement_tool = None  # RefinePlanningTool
        
        for tool in tools:
            if hasattr(tool, 'name'):
                if tool.name == "create_research_plan":
                    self.planning_tool = tool
                elif tool.name == "refine_research_plan":
                    self.refinement_tool = tool
        
        # LLM with tool binding
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        # CLI for human review
        self.review_cli = ReviewCLI()
    
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
        print(f"\n{'='*60}")
        print(f"🎯 Planning Agent 실행 중...")
        print(f"{'='*60}\n")
        
        user_topic = state.get("user_input", "")
        
        if not user_topic:
            raise ValueError("user_input이 State에 없습니다.")
        
        # Step 1: 초기 계획 생성 (성공할 때까지 재시도)
        planning_output = await self._create_initial_plan(user_topic)
        
        # Step 2: folder_name 생성 (Python 코드에서, 초기 1회만)
        folder_name = self._generate_folder_name(planning_output.normalized_topic)
        
        # Step 3: State 업데이트
        state["planning_output"] = planning_output
        state["folder_name"] = folder_name
        state["keywords"] = planning_output.keywords
        state["status"] = "planning_complete"
        
        print(f"\n✅ Planning Agent 완료!")
        print(f"📁 정규화된 주제: {planning_output.normalized_topic}")
        print(f"📁 폴더명: {folder_name}")
        print(f"🔑 키워드 개수: {len(planning_output.keywords)}")
        print(f"{'='*60}\n")
        
        return state
    
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
        
        # Conversation history
        messages = [
            HumanMessage(content=f"""You are a research planning assistant.

User wants to create a trend analysis report on: "{user_topic}"

Your task:
1. Use the 'create_research_plan' tool to generate an initial research plan
2. The tool will return a JSON with the complete plan

Call the tool now to create the plan.""")
        ]
        
        max_iterations = 100
        for iteration in range(max_iterations):
            print(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # LLM with tools
            response = await self.llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # Check if tool was called
            if not response.tool_calls:
                print(f"⚠️ LLM didn't call any tool. Response: {response.content[:100]}...")
                print(f"🔄 Prompting LLM to use tool...\n")
                messages.append(HumanMessage(content="Please use the 'create_research_plan' tool to generate the plan."))
                continue
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                print(f"🛠️ LLM calling tool: {tool_call['name']}")
                
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find and execute tool
                tool = self.planning_tool if tool_name == "create_research_plan" else self.refinement_tool
                
                if not tool:
                    print(f"❌ Tool '{tool_name}' not found!")
                    continue
                
                try:
                    # Execute tool
                    result_json = await tool._arun(**tool_args)
                    
                    # Parse result
                    planning_data = json.loads(result_json)
                    planning_data.pop('folder_name', None)
                    planning_data.pop('reasoning', None)
                    
                    # Validate
                    planning_output = PlanningOutput(**planning_data)
                    
                    print(f"✅ 초기 계획 생성 성공!\n")
                    return planning_output
                
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Tool execution failed: {error_msg}")
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call['id']
                    ))
                    
                    # Retry
                    break
        
        raise ValueError(f"초기 계획 생성 실패 (최대 반복 {max_iterations})")
    
    def _generate_folder_name(self, normalized_topic: str) -> str:
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
