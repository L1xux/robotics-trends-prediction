"""
Content Analysis Agent (LCEL 방식)

수집된 데이터를 분석하여 트렌드 분류, 섹션 생성, 인용 관리를 수행하는 Agent
LCEL을 사용한 4번의 독립적인 LLM 호출
"""
import json
import asyncio
from typing import List, Any, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState, WorkflowStatus
from src.core.models.trend_model import TrendTier
from src.core.models.citation_model import CitationEntry
from config.prompts.analysis_prompts import ANALYSIS_PROMPTS


class ContentAnalysisLLM(BaseAgent):
    """
    Content Analysis Agent (LCEL 방식)
    
    수집된 데이터 분석 및 보고서 내용 생성
    
    Workflow (4번 LLM 호출):
    1. 병렬: Section 2 + Section 3
    2. 순차: Section 4 (Section 2, 3 기반)
    3. 순차: Section 5 (Section 2, 3, 4 기반)
    
    Responsibilities:
    1. 데이터 통합 분석 (arXiv, Trends, News, RAG)
    2. 트렌드 티어 분류 (HOT_TRENDS, RISING_STARS)
    3. 섹션별 내용 생성 (10개 서브섹션)
       - 2.1~2.2: 기술 트렌드 분석
       - 3.1~3.3: 시장 동향 및 산업 적용
       - 4.1~4.2: 5년 기술 전망
       - 5.1~5.3: 기업 시사점
    4. 인용 관리 (CitationEntry)
    
    Output:
    - trends: List[TrendTier]
    - sections: Dict[str, str] (10개 서브섹션)
    - citations: List[CitationEntry]
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
        # Output parser
        json_parser = JsonOutputParser()
        
        # Section 2 Chain
        section_2_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_PROMPTS["system"]),
            ("human", ANALYSIS_PROMPTS["section_2"])
        ])
        self.section_2_chain = section_2_prompt | self.llm | json_parser
        
        # Section 3 Chain
        section_3_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_PROMPTS["system"]),
            ("human", ANALYSIS_PROMPTS["section_3"])
        ])
        self.section_3_chain = section_3_prompt | self.llm | json_parser
        
        # Section 4 Chain
        section_4_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_PROMPTS["system"]),
            ("human", ANALYSIS_PROMPTS["section_4"])
        ])
        self.section_4_chain = section_4_prompt | self.llm | json_parser
        
        # Section 5 Chain
        section_5_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_PROMPTS["system"]),
            ("human", ANALYSIS_PROMPTS["section_5"])
        ])
        self.section_5_chain = section_5_prompt | self.llm | json_parser
    
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Content Analysis 실행 (LCEL 방식)
        
        Args:
            state: 현재 파이프라인 상태
                - state["planning_output"]: PlanningOutput
                - state["arxiv_data"]: arXiv 데이터
                - state["trends_data"]: Trends 데이터
                - state["news_data"]: News 데이터
                - state["rag_results"]: RAG 결과
                - state["expanded_keywords"]: 확장된 키워드
        
        Returns:
            업데이트된 상태
                - state["trends"]: List[TrendTier]
                - state["sections"]: Dict[str, str] (10개 서브섹션)
                - state["citations"]: List[CitationEntry]
        """
        print(f"\n{'='*60}")
        print(f"Content Analysis Agent 실행 중 (LCEL 방식)...")
        print(f"{'='*60}\n")
        
        # State에서 데이터 가져오기
        topic = state.get("planning_output").topic
        keywords = state.get("expanded_keywords", [])
        arxiv_data = state.get("arxiv_data", {})
        trends_data = state.get("trends_data", {})
        news_data = state.get("news_data", {})
        rag_results = state.get("rag_results", {})
        
        try:
            # Step 1: 데이터 요약 생성
            print(f"Step 1: 데이터 요약 생성 중...")
            data_summaries = self._create_data_summaries(
                arxiv_data, trends_data, news_data, rag_results
            )
            print(f"   요약 생성 완료\n")
            
            # Step 2 & 3: Section 2, 3 병렬 실행
            print(f"Step 2 & 3: Section 2, 3 병렬 생성 중...")
            section_2_result, section_3_result = await self._run_parallel_sections(
                topic, keywords, data_summaries
            )
            print(f"   Section 2 완료 (trends: {len(section_2_result.get('trends', []))}개)")
            print(f"   Section 3 완료 (sections: {len(section_2_result.get('sections', {}))}개)\n")
            
            # Step 4: Section 4 순차 실행
            print(f"Step 4: Section 4 생성 중 (Section 2, 3 기반)...")
            section_4_result = await self._run_section_4(
                topic, section_2_result, section_3_result, data_summaries
            )
            print(f"   Section 4 완료\n")
            
            # Step 5: Section 5 순차 실행
            print(f"Step 5: Section 5 생성 중 (Section 2, 3, 4 기반)...")
            section_5_result = await self._run_section_5(
                topic, section_2_result, section_3_result, section_4_result
            )
            print(f"   Section 5 완료\n")
            
            # Step 6: 결과 통합 및 검증
            print(f"Step 6: 결과 통합 및 검증 중...")
            trends, sections, citations = self._integrate_results(
                section_2_result, section_3_result, section_4_result, section_5_result
            )
            
            print(f"   트렌드: {len(trends)}개")
            print(f"   섹션: {len(sections)}개")
            print(f"   인용: {len(citations)}개\n")
            
            # 트렌드 티어별 요약
            hot_trends = [t for t in trends if t.is_hot_trend()]
            rising_stars = [t for t in trends if t.is_rising_star()]
            
            print(f"   HOT_TRENDS (1-2년 상용화): {len(hot_trends)}개")
            for trend in hot_trends[:3]:
                print(f"      - {trend.name} (논문: {trend.paper_count}, 기업 비율: {trend.company_ratio:.2f})")
            
            print(f"\n   RISING_STARS (3-5년 핵심 기술): {len(rising_stars)}개")
            for trend in rising_stars[:3]:
                print(f"      - {trend.name} (논문: {trend.paper_count}, 기업 비율: {trend.company_ratio:.2f})")
            
            # Citation 출력
            print(f"\n   Citations (출처):")
            arxiv_citations = [c for c in citations if c.source_type == "arxiv"]
            news_citations = [c for c in citations if c.source_type == "news"]
            report_citations = [c for c in citations if c.source_type == "report"]
            
            print(f"      - ArXiv 논문: {len(arxiv_citations)}개")
            for c in arxiv_citations[:3]:
                print(f"        [{c.number}] {c.title[:60]}...")
            
            print(f"      - 뉴스 기사: {len(news_citations)}개")
            for c in news_citations[:3]:
                print(f"        [{c.number}] {c.title[:60]}...")
            
            print(f"      - 전문 보고서: {len(report_citations)}개")
            for c in report_citations[:3]:
                print(f"        [{c.number}] {c.title[:60]}...")
            
            print(f"\n{'='*60}")
            print(f"Content Analysis 완료!")
            print(f"{'='*60}\n")
            
            # State 업데이트
            state["trends"] = trends
            state["sections"] = sections  # sections 키로 저장 (표준)
            state["citations"] = citations
            state["status"] = WorkflowStatus.ANALYSIS_COMPLETE.value
            
            return state
        
        except Exception as e:
            print(f"Analysis 실패: {e}")
            print(f"\nContent Analysis Agent 최종 실패\n")
            state["status"] = WorkflowStatus.ANALYSIS_FAILED.value
            state["error"] = str(e)
            raise
    
    async def _run_parallel_sections(
        self,
        topic: str,
        keywords: List[str],
        data_summaries: Dict
    ) -> tuple:
        """Section 2, 3 병렬 실행"""
        # Section 2 입력
        section_2_input = {
            "topic": topic,
            "keywords": ", ".join(keywords),
            "arxiv_count": len(data_summaries.get("arxiv_papers", [])),
            "arxiv_summary": data_summaries["arxiv"],
            "rag_summary": data_summaries["rag"]
        }
        
        # Section 3 입력
        section_3_input = {
            "topic": topic,
            "keywords": ", ".join(keywords),
            "trends_months": data_summaries.get("trends_months", 0),
            "trends_summary": data_summaries["trends"],
            "news_count": len(data_summaries.get("news_articles", [])),
            "news_summary": data_summaries["news"],
            "rag_summary": data_summaries["rag"],  # RAG summary 추가
            "citation_start_number": 100  # Section 2의 인용이 먼저이므로 100부터 시작
        }
        
        # 병렬 실행
        section_2_task = self.section_2_chain.ainvoke(section_2_input)
        section_3_task = self.section_3_chain.ainvoke(section_3_input)
        
        section_2_result, section_3_result = await asyncio.gather(
            section_2_task, section_3_task
        )
        
        return section_2_result, section_3_result
    
    async def _run_section_4(
        self,
        topic: str,
        section_2_result: Dict,
        section_3_result: Dict,
        data_summaries: Dict
    ) -> Dict:
        """Section 4 실행 (Section 2, 3 기반)"""
        # Section 2, 3 내용 추출
        section_2_content = "\n\n".join([
            f"**{key}:**\n{value}"
            for key, value in section_2_result.get("sections", {}).items()
        ])
        
        section_3_content = "\n\n".join([
            f"**{key}:**\n{value}"
            for key, value in section_3_result.get("sections", {}).items()
        ])
        
        # Trends 요약
        trends_summary = "\n".join([
            f"- {trend['name']} ({trend['tier']}): {trend['reasoning'][:100]}..."
            for trend in section_2_result.get("trends", [])
        ])
        
        # 인용 시작 번호 계산
        citation_start = max([
            int(c.get("number", 0)) if isinstance(c.get("number"), (int, str)) else 0
            for c in section_2_result.get("citations", []) + section_3_result.get("citations", [])
        ], default=0) + 1
        
        section_4_input = {
            "topic": topic,
            "section_2": section_2_content,  # 프롬프트 변수명과 일치
            "section_3": section_3_content,  # 프롬프트 변수명과 일치
            "rag_summary": data_summaries["rag"],
            "trends_summary": trends_summary,
            "citation_start_number": citation_start
        }
        
        section_4_result = await self.section_4_chain.ainvoke(section_4_input)
        
        return section_4_result
    
    async def _run_section_5(
        self,
        topic: str,
        section_2_result: Dict,
        section_3_result: Dict,
        section_4_result: Dict
    ) -> Dict:
        """Section 5 실행 (Section 2, 3, 4 기반)"""
        # 내용 추출
        section_2_content = "\n\n".join([
            f"**{key}:**\n{value}"
            for key, value in section_2_result.get("sections", {}).items()
        ])
        
        section_3_content = "\n\n".join([
            f"**{key}:**\n{value}"
            for key, value in section_3_result.get("sections", {}).items()
        ])
        
        section_4_content = "\n\n".join([
            f"**{key}:**\n{value}"
            for key, value in section_4_result.get("sections", {}).items()
        ])
        
        trends_summary = "\n".join([
            f"- {trend['name']} ({trend['tier']}): {trend['reasoning'][:100]}..."
            for trend in section_2_result.get("trends", [])
        ])
        
        # 인용 시작 번호 계산
        citation_start = max([
            int(c.get("number", 0)) if isinstance(c.get("number"), (int, str)) else 0
            for c in (
                section_2_result.get("citations", []) +
                section_3_result.get("citations", []) +
                section_4_result.get("citations", [])
            )
        ], default=0) + 1
        
        section_5_input = {
            "topic": topic,
            "section_2": section_2_content,  # 프롬프트 변수명과 일치
            "section_3": section_3_content,  # 프롬프트 변수명과 일치
            "section_4": section_4_content,  # 프롬프트 변수명과 일치
            "trends_summary": trends_summary,
            "citation_start_number": citation_start
        }
        
        section_5_result = await self.section_5_chain.ainvoke(section_5_input)
        
        return section_5_result
    
    def _integrate_results(
        self,
        section_2_result: Dict,
        section_3_result: Dict,
        section_4_result: Dict,
        section_5_result: Dict
    ) -> tuple:
        """
        모든 섹션 결과 통합
        
        Returns:
            (trends, sections, citations)
        """
        # Trends 추출 (Section 2에서만)
        trends = []
        for trend_data in section_2_result.get("trends", []):
            trend = TrendTier(**trend_data)
            trends.append(trend)
        
        # Sections 통합
        sections = {}
        sections.update(section_2_result.get("sections", {}))
        sections.update(section_3_result.get("sections", {}))
        sections.update(section_4_result.get("sections", {}))
        sections.update(section_5_result.get("sections", {}))
        
        # Remove markdown wrapper from sections
        for key, value in sections.items():
            sections[key] = self._remove_markdown_wrapper(value)
        
        # 필수 섹션 확인
        required_sections = [
            "section_2_1",
            "section_2_2",
            "section_3_1",
            "section_3_2",
            "section_3_3",
            "section_4_1",
            "section_4_2",
            "section_5_1",
            "section_5_2",
            "section_5_3"
        ]
        
        for section in required_sections:
            if section not in sections:
                raise ValueError(f"필수 섹션 누락: {section}")
        
        # Citations 통합 및 정렬
        all_citations = (
            section_2_result.get("citations", []) +
            section_3_result.get("citations", []) +
            section_4_result.get("citations", []) +
            section_5_result.get("citations", [])
        )
        
        # 중복 제거 (number 기준)
        seen_numbers = set()
        unique_citations = []
        for citation_data in all_citations:
            number = citation_data.get("number")
            if number not in seen_numbers:
                seen_numbers.add(number)
                
                # authors가 리스트면 문자열로 변환
                if isinstance(citation_data.get("authors"), list):
                    authors_list = citation_data.get("authors")
                    citation_data["authors"] = ", ".join(authors_list[:3])
                    if len(authors_list) > 3:
                        citation_data["authors"] += " et al."
                
                citation = CitationEntry(**citation_data)
                unique_citations.append(citation)
        
        # 번호순 정렬
        unique_citations.sort(key=lambda c: c.number)
        
        return trends, sections, unique_citations
    
    def _remove_markdown_wrapper(self, text: str) -> str:
        """Remove markdown code block wrapper from LLM response"""
        if not isinstance(text, str):
            return text
        
        text = text.strip()
        
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        
        if text.endswith("```"):
            text = text[:-3].strip()
        
        return text
    
    def _create_data_summaries(
        self,
        arxiv_data: Dict,
        trends_data: Dict,
        news_data: Dict,
        rag_results: Dict
    ) -> Dict[str, Any]:
        """데이터 요약 생성"""
        summaries = {}
        
        # arXiv 요약
        arxiv_papers = arxiv_data.get("papers", [])
        arxiv_summary = f"Total papers: {len(arxiv_papers)}\n\n"
        arxiv_summary += "Sample papers:\n"
        for i, paper in enumerate(arxiv_papers[:5], 1):
            arxiv_summary += f"{i}. {paper.get('title', 'N/A')}\n"
            arxiv_summary += f"   Authors: {', '.join(paper.get('authors', [])[:3])}\n"
            arxiv_summary += f"   Date: {paper.get('published', 'N/A')}\n"
            arxiv_summary += f"   Abstract: {paper.get('abstract', '')[:150]}...\n\n"
        summaries["arxiv"] = arxiv_summary
        summaries["arxiv_papers"] = arxiv_papers
        
        # Trends 요약
        trends_summary = f"Total months: {trends_data.get('total_months', 0)}\n"
        trends_summary += f"Keywords tracked: {', '.join(trends_data.get('keywords', []))}\n\n"
        trends_summary += "Sample data points:\n"
        for i, data_point in enumerate(trends_data.get('data', [])[:3], 1):
            trends_summary += f"{i}. Date: {data_point.get('date', 'N/A')}\n"
            for key, value in data_point.items():
                if key != 'date':
                    trends_summary += f"   {key}: {value}\n"
            trends_summary += "\n"
        summaries["trends"] = trends_summary
        summaries["trends_months"] = trends_data.get('total_months', 0)
        
        # News 요약
        news_articles = news_data.get("articles", [])
        news_summary = f"Total articles: {len(news_articles)}\n"
        news_summary += f"Unique sources: {news_data.get('unique_sources', 0)}\n\n"
        news_summary += "Sample articles:\n"
        for i, article in enumerate(news_articles[:5], 1):
            news_summary += f"{i}. {article.get('title', 'N/A')}\n"
            news_summary += f"   Source: {article.get('source', 'N/A')}\n"
            news_summary += f"   Date: {article.get('published', 'N/A')}\n"
            news_summary += f"   Snippet: {article.get('snippet', '')[:100]}...\n\n"
        summaries["news"] = news_summary
        summaries["news_articles"] = news_articles
        
        # RAG 요약
        rag_summary = f"Total results: {rag_results.get('total_results', 0)}\n\n"
        rag_summary += "Key insights from reference documents:\n"
        for i, result in enumerate(rag_results.get('results', [])[:5], 1):
            rag_summary += f"{i}. Source: {result.get('source', 'N/A')} (Page {result.get('page', 'N/A')})\n"
            rag_summary += f"   Content: {result.get('content', '')[:200]}...\n\n"
        summaries["rag"] = rag_summary
        
        return summaries

