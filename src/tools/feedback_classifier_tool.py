"""
Feedback Classifier Tool

유저 피드백을 분석하여 심각도를 판단하는 Tool
- "진짜 싫음" → Data Collection부터 재시작
- "적당히 싫음" → Revision Agent로 개선
- "좋음" → Accept
"""

import json
from typing import Dict, Any
from langchain.tools import BaseTool as LangChainBaseTool
from langchain_core.tools import ToolException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field


class FeedbackClassifierTool(LangChainBaseTool):
    """
    유저 피드백 심각도 분류 Tool
    
    유저의 보고서 피드백을 분석하여:
    1. "major" - 진짜 싫음 (데이터 수집부터 재시작 필요)
       - 핵심 내용이 완전히 틀림
       - 데이터가 너무 부족함
       - 주제와 완전히 동떨어짐
       - 전체 구조 재편 필요
    
    2. "minor" - 적당히 싫음 (Revision Agent로 개선 가능)
       - 일부 섹션 개선 필요
       - 표현이나 스타일 문제
       - 내용 보완 필요
       - 인용 추가 필요
    
    3. "accept" - 좋음 (승인)
       - 만족스러움
       - 승인 키워드 사용
    """
    
    name: str = "classify_feedback"
    description: str = """Classify user feedback severity to determine next action.
    
Input:
- user_feedback: User's feedback text
- report_content: Current report (for context)

Output: Classification result (major/minor/accept) with reasoning"""
    
    llm: BaseChatModel = Field(description="LLM for feedback classification")
    
    def _run(self, user_feedback: str, report_content: str = "") -> str:
        """
        Synchronous version (not used, required by BaseTool)
        """
        raise NotImplementedError("Use async version (_arun)")
    
    async def _arun(
        self,
        user_feedback: str,
        report_content: str = ""
    ) -> str:
        """
        유저 피드백 분류
        
        Args:
            user_feedback: 유저 피드백 텍스트
            report_content: 현재 보고서 (선택)
        
        Returns:
            분류 결과 JSON string
            {
                "severity": "major" | "minor" | "accept",
                "reasoning": "why this classification",
                "issues": ["issue1", "issue2"],
                "suggested_action": "restart_collection" | "revise_report" | "accept_report"
            }
        """
        try:
            # Setup classification chain
            classification_prompt = ChatPromptTemplate.from_template(
                """당신은 사용자 피드백 분석 전문가입니다.

**사용자 피드백:**
{user_feedback}

**보고서 길이:** {report_length} characters

**분류 기준:**

**1. ACCEPT (승인) - 다음 중 하나라도 해당:**
- 승인 키워드: "ok", "okay", "good", "great", "approve", "accept", "좋아요", "괜찮아요", "승인", "확인"
- 긍정적인 피드백만 있음
- 사소한 개선 요청만 있음

**2. MINOR (작은 개선 필요) - Revision Agent로 해결 가능:**
- 일부 섹션의 표현 개선 필요
- 스타일이나 톤 조정 필요
- 특정 내용 추가/보완 요청
- 인용 추가 필요
- 구조는 OK, 내용 보강 필요
- 예시: "Section 3를 더 자세히", "스타일을 더 전문적으로", "예시를 추가해줘"

**3. MAJOR (큰 문제) - Data Collection부터 재시작 필요:**
- 핵심 데이터가 완전히 부족함 ("데이터가 너무 적다", "자료가 부족하다")
- 주제와 완전히 동떨어진 내용 ("이건 내가 원한 게 아니야", "주제가 틀렸어")
- 전체 구조 재편 필요 ("전체를 다시 작성해줘")
- 핵심 섹션이 완전히 잘못됨 ("기술 분석이 엉망이야", "전망이 완전히 틀렸어")
- 예시: "데이터 수집부터 다시", "완전히 잘못됐어", "이건 쓸모없어"

**판단 지침:**
1. **승인 키워드가 있으면 무조건 ACCEPT**
2. **"데이터 부족", "자료 부족", "처음부터", "완전히 틀렸다" → MAJOR**
3. **"일부 수정", "약간 개선", "스타일 변경", "내용 보완" → MINOR**
4. **애매하면 MINOR** (Revision으로 먼저 시도)

**출력 (JSON만):**
{{
    "severity": "accept" or "minor" or "major",
    "reasoning": "분류 이유 (구체적으로)",
    "issues": [
        "문제점 1",
        "문제점 2",
        ...
    ],
    "suggested_action": "accept_report" or "revise_report" or "restart_collection",
    "confidence": 0.0-1.0 (확신 정도)
}}

JSON만 출력하세요."""
            )
            
            json_parser = JsonOutputParser()
            classification_chain = classification_prompt | self.llm | json_parser
            
            # Run classification
            result = await classification_chain.ainvoke({
                "user_feedback": user_feedback,
                "report_length": len(report_content) if report_content else 0
            })
            
            # Validate result
            self._validate_classification_result(result)
            
            return json.dumps(result, ensure_ascii=False)
        
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error in feedback_classifier_tool: {str(e)}")
    
    def _validate_classification_result(self, result: Dict[str, Any]):
        """
        분류 결과 검증
        
        Args:
            result: Classification result dict
        
        Raises:
            ToolException: If result is invalid
        """
        required_fields = ["severity", "reasoning", "issues", "suggested_action"]
        
        for field in required_fields:
            if field not in result:
                raise ToolException(f"Missing required field in classification result: {field}")
        
        # Validate severity
        if result["severity"] not in ["accept", "minor", "major"]:
            raise ToolException(f"severity must be accept/minor/major, got {result['severity']}")
        
        # Validate suggested_action
        valid_actions = ["accept_report", "revise_report", "restart_collection"]
        if result["suggested_action"] not in valid_actions:
            raise ToolException(f"suggested_action must be one of {valid_actions}, got {result['suggested_action']}")
        
        # Validate issues is a list
        if not isinstance(result["issues"], list):
            raise ToolException("issues must be a list")

