"""
Quality Checker Tool

최종 보고서의 품질을 평가하는 Tool
Human Review 2 전에 사용
"""

import json
from typing import Dict, Any, List
from langchain.tools import BaseTool as LangChainBaseTool
from langchain_core.tools import ToolException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import Field


class QualityCheckerTool(LangChainBaseTool):
    """
    보고서 품질 검사 Tool
    
    다음 항목을 평가:
    1. 완성도 (Completeness)
    2. 정확성 (Accuracy)
    3. 명확성 (Clarity)
    4. 구조 (Structure)
    5. 인용 (Citations)
    
    Output:
    {
        "overall_score": 8.5,  # 0-10
        "section_scores": {
            "summary": 9.0,
            "introduction": 8.5,
            "technical_analysis": 8.0,
            ...
        },
        "strengths": ["..."],
        "improvements": ["..."],
        "recommendation": "accept" or "revise"
    }
    """
    
    name: str = "quality_checker"
    description: str = """Check the quality of a final report.
    
Input:
- report_content: Full report markdown text
- report_structure: Expected sections list

Output: Quality assessment with scores and recommendations"""
    
    llm: BaseChatModel = Field(description="LLM for quality checking")
    
    def _run(self, report_content: str, report_structure: List[str] = None) -> str:
        """
        Synchronous version (not used, required by BaseTool)
        """
        raise NotImplementedError("Use async version (_arun)")
    
    async def _arun(
        self,
        report_content: str,
        report_structure: List[str] = None
    ) -> str:
        """
        보고서 품질 검사
        
        Args:
            report_content: 전체 보고서 마크다운
            report_structure: 예상 섹션 목록
        
        Returns:
            품질 평가 결과 (JSON string)
        """
        if report_structure is None:
            report_structure = [
                "SUMMARY",
                "1. Introduction",
                "2. Technology Trend Analysis",
                "3. Market Trends & Applications",
                "4. 5-Year Forecast",
                "5. Implications for Business",
                "6. Conclusion",
                "REFERENCE",
                "APPENDIX"
            ]
        
        try:
            # Setup quality check chain
            quality_prompt = ChatPromptTemplate.from_template(
                """당신은 전문 보고서 품질 평가 전문가입니다.

**보고서 내용:**
{report_content}

**예상 구조:**
{expected_sections}

**평가 항목:**

1. **완성도 (Completeness)** - 10점 만점
   - 모든 필수 섹션 포함 여부
   - 각 섹션의 충실도
   - 데이터 및 예시의 충분성

2. **정확성 (Accuracy)** - 10점 만점
   - 데이터 기반 분석
   - 논리적 일관성
   - 주장의 타당성

3. **명확성 (Clarity)** - 10점 만점
   - 명확한 표현
   - 구조적 흐름
   - 가독성

4. **전문성 (Professionalism)** - 10점 만점
   - 전문적인 톤
   - 용어의 일관성
   - 형식의 완성도

5. **인용 (Citations)** - 10점 만점
   - 적절한 인용
   - 출처의 신뢰성
   - 인용 형식

**섹션별 평가:**
각 주요 섹션(SUMMARY, Section 1-6)에 대해 0-10점 평가

**평가 출력 (JSON):**
{{
    "overall_score": 평균 점수 (0-10),
    "category_scores": {{
        "completeness": 점수,
        "accuracy": 점수,
        "clarity": 점수,
        "professionalism": 점수,
        "citations": 점수
    }},
    "section_scores": {{
        "summary": 점수,
        "introduction": 점수,
        "technical_analysis": 점수,
        "market_trends": 점수,
        "forecast": 점수,
        "business_implications": 점수,
        "conclusion": 점수
    }},
    "strengths": [
        "강점 1",
        "강점 2",
        "강점 3"
    ],
    "improvements": [
        "개선사항 1",
        "개선사항 2",
        "개선사항 3"
    ],
    "recommendation": "accept" (8.0+) or "revise" (<8.0),
    "detailed_feedback": "상세 피드백 (200-300단어)"
}}

점수 기준:
- 9.0-10.0: Excellent
- 8.0-8.9: Good
- 7.0-7.9: Satisfactory
- 6.0-6.9: Needs Improvement
- <6.0: Poor

JSON만 출력하세요."""
            )
            
            json_parser = JsonOutputParser()
            quality_chain = quality_prompt | self.llm | json_parser
            
            # Run quality check
            result = await quality_chain.ainvoke({
                "report_content": report_content[:30000],  # Limit to avoid token overflow
                "expected_sections": "\n".join(f"- {s}" for s in report_structure)
            })
            
            # Validate result
            self._validate_quality_result(result)
            
            return json.dumps(result, ensure_ascii=False)
        
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error in quality_checker_tool: {str(e)}")
    
    def _validate_quality_result(self, result: Dict[str, Any]):
        """
        품질 검사 결과 검증
        
        Args:
            result: Quality check result dict
        
        Raises:
            ToolException: If result is invalid
        """
        required_fields = [
            "overall_score",
            "category_scores",
            "section_scores",
            "strengths",
            "improvements",
            "recommendation"
        ]
        
        for field in required_fields:
            if field not in result:
                raise ToolException(f"Missing required field in quality result: {field}")
        
        # Validate overall_score range
        if not (0 <= result["overall_score"] <= 10):
            raise ToolException(f"overall_score must be 0-10, got {result['overall_score']}")
        
        # Validate recommendation
        if result["recommendation"] not in ["accept", "revise"]:
            raise ToolException(f"recommendation must be 'accept' or 'revise', got {result['recommendation']}")
        
        # Validate strengths and improvements are lists
        if not isinstance(result["strengths"], list):
            raise ToolException("strengths must be a list")
        
        if not isinstance(result["improvements"], list):
            raise ToolException("improvements must be a list")


class QuickQualityCheckerTool(LangChainBaseTool):
    """
    간단한 품질 검사 Tool (빠른 검증용)
    
    주요 체크리스트만 확인:
    - 필수 섹션 존재 여부
    - 최소 길이 요구사항
    - 인용 존재 여부
    """
    
    name: str = "quick_quality_check"
    description: str = """Quick quality check for report structure and basic requirements."""
    
    def _run(self, report_content: str) -> str:
        """
        Synchronous version
        """
        return self._arun(report_content)
    
    async def _arun(self, report_content: str) -> str:
        """
        간단한 품질 검사
        
        Args:
            report_content: 전체 보고서
        
        Returns:
            Quick check result (JSON string)
        """
        try:
            required_sections = [
                "SUMMARY",
                "## 1. Introduction",
                "## 2.",
                "## 3.",
                "## 4.",
                "## 5.",
                "## 6. Conclusion",
                "## REFERENCE"
            ]
            
            issues = []
            
            # Check required sections
            missing_sections = []
            for section in required_sections:
                if section not in report_content:
                    missing_sections.append(section)
            
            if missing_sections:
                issues.append(f"Missing sections: {', '.join(missing_sections)}")
            
            # Check minimum length
            min_length = 5000  # characters
            if len(report_content) < min_length:
                issues.append(f"Report too short: {len(report_content)} chars (minimum {min_length})")
            
            # Check for citations
            if "[" not in report_content or "]" not in report_content:
                issues.append("No citations found")
            
            # Determine pass/fail
            status = "pass" if len(issues) == 0 else "fail"
            
            result = {
                "status": status,
                "issues": issues,
                "report_length": len(report_content),
                "sections_found": len(required_sections) - len(missing_sections),
                "sections_total": len(required_sections)
            }
            
            return json.dumps(result, ensure_ascii=False)
        
        except Exception as e:
            raise ToolException(f"Error in quick_quality_check: {str(e)}")

