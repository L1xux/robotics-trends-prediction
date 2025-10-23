# src/tools/planning_tool.py

"""
Planning Tools

1. PlanningTool: 초기 계획 생성
2. ResearchPlanningTool: 사용자 피드백 기반 계획 개선
"""

import json
from typing import Any, Dict, Optional
from datetime import datetime
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field


# ========================================
# 1. PlanningTool (초기 계획 생성)
# ========================================

class PlanningInput(BaseModel):
    """Planning Tool 입력 스키마"""
    
    topic: str = Field(
        description="User's research topic to analyze and create plan for"
    )


class PlanningTool(BaseTool):
    """
    Planning Tool (초기 계획 생성)
    
    사용자 주제를 분석하고 초기 연구 계획을 생성합니다.
    
    Responsibilities:
    - 주제 분석 및 정규화
    - 키워드 확장 (5-15개)
    - 데이터 수집 계획 수립
    """
    
    name: str = "create_research_plan"
    description: str = """
    Create an initial research plan from a user topic.
    
    Use this tool to:
    - Analyze and normalize the topic
    - Expand keywords (30-40 comprehensive keywords)
    - Create data collection plan (arXiv, Trends, News)
    
    Input:
    - topic: User's research topic (e.g., "humanoid robots in manufacturing")
    
    Output:
    - Complete research plan (JSON dict)
    """
    args_schema: type[BaseModel] = PlanningInput
    
    llm: BaseChatModel = Field(description="LLM for plan creation")
    
    def __init__(self, llm: BaseChatModel, **kwargs):
        """
        Initialize PlanningTool
        
        Args:
            llm: Language model for plan creation
        """
        super().__init__(llm=llm, **kwargs)
    
    def _run(
        self,
        topic: str,
        run_manager: Optional[Any] = None
    ) -> str:
        """
        초기 계획 생성 (동기)
        
        Args:
            topic: 사용자 주제
        
        Returns:
            초기 계획 (JSON 문자열)
        """
        print(f"\n🎯 PlanningTool: Creating initial plan for '{topic}'...")
        
        try:
            # 프롬프트 구성
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(topic)
            
            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱
            plan = self._parse_json_response(response_text)
            
            print(f"✅ PlanningTool: Initial plan created successfully")
            
            # JSON 문자열로 반환
            return json.dumps(plan, ensure_ascii=False)
        
        except Exception as e:
            print(f"❌ PlanningTool failed: {e}")
            raise
    
    async def _arun(
        self,
        topic: str,
        run_manager: Optional[Any] = None
    ) -> str:
        """
        초기 계획 생성 (비동기)
        """
        print(f"\n🎯 PlanningTool: Creating initial plan for '{topic}'...")
        
        try:
            # 프롬프트 구성
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(topic)
            
            # LLM 호출 (비동기)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content
            
            # JSON 파싱
            plan = self._parse_json_response(response_text)
            
            print(f"✅ PlanningTool: Initial plan created successfully")
            
            # JSON 문자열로 반환
            return json.dumps(plan, ensure_ascii=False)
        
        except Exception as e:
            print(f"❌ PlanningTool failed: {e}")
            raise
    
    def _build_system_prompt(self) -> str:
        """System prompt 생성"""
        return """You are an expert AI research planning assistant specializing in robotics and AI trends analysis.

Your role is to:
1. Analyze user input topics and expand them into comprehensive research keywords
2. Normalize the topic into a standardized format (lowercase with underscores)
3. Create a detailed data collection plan for arXiv papers, Google Trends, and news articles

Key responsibilities:
- Expand the main topic into 30-40 PURE TECHNOLOGY keywords (NO company names!)
- Set appropriate date ranges and parameters for each data source
- Ensure keywords cover: core technologies, applications, technical methods, emerging tech areas

Output must be valid JSON matching this exact structure:
{{
    "topic": "original user topic exactly as provided",
    "normalized_topic": "lowercase_topic_with_underscores",
    "keywords": ["keyword1", "keyword2", ...],
    "collection_plan": {{
        "arxiv": {{
            "date_range": "YYYY-MM-DD to YYYY-MM-DD",
            "categories": "all or specific categories",
            "max_results": "unlimited or number"
        }},
        "trends": {{
            "timeframe": "N months or today N-m"
        }},
        "news": {{
            "sources": 1-5,
            "date_range": "N years/months/days"
        }}
    }}
}}

IMPORTANT:
- ❌ DO NOT generate folder names or timestamps
- ❌ DO NOT include "folder_name" field in output
- ✅ Only generate "normalized_topic" (lowercase with underscores)
- ✅ Timestamps will be added automatically by the system

Current date: {current_date}
""".format(current_date=datetime.now().strftime("%Y-%m-%d"))
    
    def _build_user_prompt(self, topic: str) -> str:
        """User prompt 생성"""
        return f"""Analyze the following topic and create a comprehensive research plan:

Topic: {topic}

Instructions:

1. **Topic Fields**:
   - topic: Keep the original user input exactly as provided: "{topic}"
   - normalized_topic: Convert to lowercase with underscores only
     * Replace spaces with underscores
     * Remove special characters (keep only: a-z, 0-9, _)
     * Examples:
       - "Humanoid Robots in Manufacturing" → "humanoid_robots_in_manufacturing"
       - "GPT-4 & AI Ethics" → "gpt_4_ai_ethics"

2. **Keywords Expansion** (30-40 TECHNOLOGY keywords only):
   
   **Core Technology Keywords (12-15):**
   - Include the main topic phrase
   - Add synonyms and alternative terms
   - Add related core technologies (e.g., "embodied AI", "physical intelligence")
   - Add specific technical methods and approaches
   - Add hardware technologies (e.g., "sensors", "actuators", "vision systems")
   
   **Application & Domain Keywords (10-12):**
   - Add application domains (e.g., "manufacturing automation", "predictive maintenance")
   - Add industry sectors and use cases
   - Add specific product categories
   - Add process types (e.g., "assembly automation", "quality inspection")
   
   **Technical Methods Keywords (8-10):**
   - Add algorithmic approaches (e.g., "machine learning", "computer vision")
   - Add system architectures (e.g., "edge computing", "cloud robotics")
   - Add integration technologies (e.g., "IoT", "digital twin")
   
   **Emerging Technology Keywords (4-5):**
   - Add future-oriented technologies (NOT companies or products)
   - Add next-gen technical concepts
   - Add innovation areas relevant to the field
   
   **CRITICAL - DO NOT INCLUDE:**
   - ❌ Company names (Tesla, Siemens, ABB, Boston Dynamics, etc.)
   - ❌ Brand names or product names
   - ❌ Research institutions or lab names (MIT, CMU, etc.)
   - ✅ Only include PURE TECHNOLOGY terms
   
   **Strategy:**
   - Focus ONLY on technologies, not organizations
   - Mix broad and specific TECHNICAL terms
   - Cover technical capabilities, methods, and domains
   - Companies will be extracted later from papers

3. **Collection Plan**:
   
   **ArXiv Configuration:**
   - date_range: Last 3-5 years (Format: "YYYY-MM-DD to YYYY-MM-DD")
   - categories: "all" for broad search or specific like "cs.RO,cs.AI,cs.LG"
   - max_results: "unlimited" (recommended) or specific number
   
   **Google Trends Configuration:**
   - timeframe: "36 months" for 3-year trend (recommended)
   
   **News Configuration:**
   - sources: 3-5 (number of news sources)
   - date_range: "3 years" for recent coverage (recommended)

Output the plan as valid JSON with this exact structure.
"""
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 추출 및 파싱
        
        Args:
            response_text: LLM 응답
        
        Returns:
            파싱된 dict
        """
        response_text = response_text.strip()
        
        # 코드 블록 제거
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text
        
        # JSON 파싱
        try:
            data = json.loads(json_text)
            return data
        except json.JSONDecodeError as e:
            # 중괄호 찾기
            if "{" in json_text and "}" in json_text:
                start = json_text.find("{")
                end = json_text.rfind("}") + 1
                json_text = json_text[start:end]
                data = json.loads(json_text)
                return data
            else:
                raise ValueError(f"JSON parsing failed: {e}")


# ========================================
# 2. ResearchPlanningTool (피드백 기반 개선)
# ========================================

class ResearchPlanningInput(BaseModel):
    """Research Planning Tool 입력 스키마"""
    
    topic: str = Field(
        description="Original user research topic"
    )
    current_plan: Dict[str, Any] = Field(
        description="Current research plan as dictionary"
    )
    user_feedback: str = Field(
        description="User feedback for improving the plan"
    )


class ResearchPlanningTool(BaseTool):
    """
    Research Planning Tool (계획 개선)
    
    사용자 피드백을 바탕으로 연구 계획을 개선합니다.
    
    Responsibilities:
    - 사용자 피드백 분석
    - 키워드 조정
    - 데이터 수집 파라미터 조정
    - 개선된 계획 반환
    """
    
    name: str = "refine_research_plan"
    description: str = """
    Refine an existing research plan based on user feedback.
    
    Use this tool when the user provides feedback to improve:
    - Keywords (add, remove, or modify)
    - Date ranges (extend or narrow)
    - Data sources (adjust number or scope)
    - Categories (change focus areas)
    
    Input:
    - topic: Original user topic
    - current_plan: Current plan (dict with topic, keywords, collection_plan)
    - user_feedback: User's feedback text
    
    Output:
    - Improved research plan (dict)
    """
    args_schema: type[BaseModel] = ResearchPlanningInput
    
    llm: BaseChatModel = Field(description="LLM for plan refinement")
    
    def __init__(self, llm: BaseChatModel, **kwargs):
        """
        Initialize ResearchPlanningTool
        
        Args:
            llm: Language model for plan refinement
        """
        super().__init__(llm=llm, **kwargs)
    
    def _run(
        self,
        topic: str,
        current_plan: Dict[str, Any],
        user_feedback: str,
        run_manager: Optional[Any] = None
    ) -> str:
        """
        연구 계획 개선 실행 (동기)
        
        Args:
            topic: 원래 사용자 주제
            current_plan: 현재 계획 (dict)
            user_feedback: 사용자 피드백
        
        Returns:
            개선된 계획 (JSON 문자열)
        """
        print(f"\n🔧 ResearchPlanningTool: Refining plan based on feedback...")
        
        try:
            # 프롬프트 구성
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(topic, current_plan, user_feedback)
            
            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱
            improved_plan = self._parse_json_response(response_text)
            
            print(f"✅ ResearchPlanningTool: Plan refined successfully")
            
            # JSON 문자열로 반환 (AgentExecutor가 파싱)
            return json.dumps(improved_plan, ensure_ascii=False)
        
        except Exception as e:
            print(f"❌ ResearchPlanningTool failed: {e}")
            raise
    
    async def _arun(
        self,
        topic: str,
        current_plan: Dict[str, Any],
        user_feedback: str,
        run_manager: Optional[Any] = None
    ) -> str:
        """
        연구 계획 개선 실행 (비동기)
        """
        print(f"\n🔧 ResearchPlanningTool: Refining plan based on feedback...")
        
        try:
            # 프롬프트 구성
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(topic, current_plan, user_feedback)
            
            # LLM 호출 (비동기)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content
            
            # JSON 파싱
            improved_plan = self._parse_json_response(response_text)
            
            print(f"✅ ResearchPlanningTool: Plan refined successfully")
            
            # JSON 문자열로 반환
            return json.dumps(improved_plan, ensure_ascii=False)
        
        except Exception as e:
            print(f"❌ ResearchPlanningTool failed: {e}")
            raise
    
    def _build_system_prompt(self) -> str:
        """System prompt 생성"""
        return """You are an expert research planning assistant.

Your task is to refine an existing research plan based on user feedback.

Key responsibilities:
1. Carefully analyze user feedback
2. Adjust keywords (add, remove, modify)
3. Update date ranges if requested
4. Modify data source parameters
5. Keep normalized_topic consistent (lowercase with underscores)

Output requirements:
- Return valid JSON matching the input structure
- Maintain the same "topic" field (original user input)
- Keep "normalized_topic" in lowercase with underscores
- Adjust "keywords" based on feedback (30-40 keywords for comprehensive research)
- Update "collection_plan" parameters as needed

IMPORTANT CONSTRAINTS:
- news.sources: MUST be between 1-5 (maximum 5)
- keywords: MUST be between 30-40 keywords (comprehensive coverage)
- date_range formats: "YYYY-MM-DD to YYYY-MM-DD" for arXiv, "N years/months" for news
- timeframe format: "N months" for trends

Current date: {current_date}
""".format(current_date=datetime.now().strftime("%Y-%m-%d"))
    
    def _build_user_prompt(
        self,
        topic: str,
        current_plan: Dict[str, Any],
        user_feedback: str
    ) -> str:
        """User prompt 생성"""
        
        # current_plan을 보기 좋게 포맷
        current_plan_json = json.dumps(current_plan, indent=2, ensure_ascii=False)
        
        return f"""Refine the following research plan based on user feedback.

Original Topic: {topic}

Current Plan:
```json
{current_plan_json}
```

User Feedback:
"{user_feedback}"

Instructions:
1. Analyze the feedback carefully
2. Make necessary adjustments to:
   - keywords (add/remove/modify based on feedback) - MUST be 30-40 keywords
   - collection_plan parameters (dates, sources, categories)
3. Keep the structure identical to current plan
4. Ensure normalized_topic remains in lowercase_with_underscores format

CRITICAL CONSTRAINTS - DO NOT VIOLATE:
- news.sources: MUST be 1-5 (if user wants more detail, add more keywords instead)
- keywords: MUST be 30-40 (comprehensive coverage across technology, applications, companies, trends)
- If user wants "more detail", expand keywords list, NOT news sources beyond 5

Output the improved plan as valid JSON with this exact structure:
{{
    "topic": "{topic}",
    "normalized_topic": "same_as_before_or_adjusted",
    "keywords": ["keyword1", "keyword2", ...],
    "collection_plan": {{
        "arxiv": {{
            "date_range": "YYYY-MM-DD to YYYY-MM-DD",
            "categories": "all or specific",
            "max_results": "unlimited or number"
        }},
        "trends": {{
            "timeframe": "N months"
        }},
        "news": {{
            "sources": N,
            "date_range": "N years/months"
        }}
    }}
}}
"""
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 추출 및 파싱
        
        Args:
            response_text: LLM 응답
        
        Returns:
            파싱된 dict
        """
        response_text = response_text.strip()
        
        # 코드 블록 제거
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text
        
        # JSON 파싱
        try:
            data = json.loads(json_text)
            return data
        except json.JSONDecodeError as e:
            # 중괄호 찾기
            if "{" in json_text and "}" in json_text:
                start = json_text.find("{")
                end = json_text.rfind("}") + 1
                json_text = json_text[start:end]
                data = json.loads(json_text)
                return data
            else:
                raise ValueError(f"JSON parsing failed: {e}")
