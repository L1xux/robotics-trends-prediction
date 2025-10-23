"""
Report Synthesis Agent

Content Analysis Agent의 출력(섹션 2,3,4,5)을 받아서
Summary, Introduction, Conclusion, References, Appendix를 생성하는 Agent
"""

from typing import List, Any, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState
from src.core.models.citation_model import CitationEntry
from config.prompts.synthesis_prompts import SYNTHESIS_PROMPTS


class ReportSynthesisLLM(BaseAgent):
    """
    Report Synthesis Agent
    
    ContentAnalysisAgent의 출력을 받아서 나머지 섹션들을 생성:
    - SUMMARY (Executive Summary)
    - Section 1: Introduction
    - Section 6: Conclusion
    - REFERENCE (Citations)
    - APPENDIX (추가 자료)
    
    Input (from ContentAnalysisAgent):
    - sections: Dict[str, str] (section_2_1, section_2_2, ..., section_5_3)
    - trends: List[TrendTier]
    - citations: List[CitationEntry]
    
    Output:
    - summary: str
    - section_1: str (Introduction)
    - section_6: str (Conclusion)
    - references: str (formatted citations)
    - appendix: str
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
        str_parser = StrOutputParser()
        
        # Summary Chain
        summary_prompt = ChatPromptTemplate.from_template(
            """당신은 AI/로보틱스 전문 보고서 작가입니다.

다음 분석 결과를 바탕으로 **Executive Summary**를 작성하세요:

**기술 트렌드 분석 (Section 2):**
{section_2}

**시장 동향 및 산업 적용 (Section 3):**
{section_3}

**5년 기술 전망 (Section 4):**
{section_4}

**비즈니스 시사점 (Section 5):**
{section_5}

**핵심 트렌드:**
{key_trends}

**Executive Summary 작성 요구사항:**

1. **길이**: 200-300단어
2. **구조**:
   - **요약문 (Overview)**: 보고서의 핵심 내용을 2-3문장으로 압축
   - **5년 전망 (5-Year Forecast)**: 2025-2030년 주요 기술 발전 전망 (HOT_TRENDS와 RISING_STARS 포함)

3. **톤**: 간결하고 임팩트 있게, 경영진 대상
4. **특징**: 데이터 기반, 구체적 인사이트

**주의**:
- "주요 발견사항", "권장사항" 등은 포함하지 마세요
- 요약문과 5년 전망만 포함하세요
- 각 파트는 명확히 구분되어야 합니다

Executive Summary만 작성하여 반환하세요 (마크다운 형식)."""
        )
        self.summary_chain = summary_prompt | self.llm | str_parser
        
        # Introduction Chain
        intro_prompt = ChatPromptTemplate.from_template(
            """당신은 AI/로보틱스 전문 보고서 작가입니다.

보고서 주제: {topic}

다음 정보를 바탕으로 **1. Introduction**을 작성하세요:

**수집된 데이터:**
- ArXiv 논문: {arxiv_count}편
- 전문 보고서 (RAG): {rag_count}건
- 뉴스 기사: {news_count}건

**보고서 구조:**
- Section 1: Introduction
- Section 2: AI-Robotics Technology Trend Analysis
- Section 3: Market Trends & Applications
- Section 4: 5-Year Forecast (2025-2030)
- Section 5: Implications for Business
- Section 6: Conclusion

**Introduction 작성 요구사항:**

1. **### 배경 (Background)**:
   - AI/로보틱스 산업의 현황과 중요성
   - 본 보고서의 필요성

2. **### 목적 (Purpose)**:
   - 보고서의 목적과 범위
   - 대상 독자

3. **### 방법론 (Methodology)**:
   - 데이터 수집 방법 (ArXiv, 전문 보고서, 뉴스)
   - 분석 접근법 (트렌드 분류, 5년 전망)

4. **### 구조 (Structure)**:
   - 각 섹션의 간략한 설명

**중요**: 각 파트(배경, 목적, 방법론, 구조) 앞에 `###` 마크다운 헤더를 반드시 사용하세요.
길이: 500-700단어
톤: 전문적이고 객관적

Introduction만 작성하여 반환하세요 (마크다운 형식, "## 1. Introduction" 제목 포함)."""
        )
        self.intro_chain = intro_prompt | self.llm | str_parser
        
        # Conclusion Chain
        conclusion_prompt = ChatPromptTemplate.from_template(
            """당신은 AI/로보틱스 전문 보고서 작가입니다.

다음 분석 결과를 바탕으로 **6. Conclusion**을 작성하세요:

**기술 트렌드 분석:**
{section_2}

**시장 동향:**
{section_3}

**5년 기술 전망:**
{section_4}

**비즈니스 시사점:**
{section_5}

**핵심 트렌드:**
{key_trends}

**Conclusion 작성 요구사항:**

1. **핵심 발견사항 (Key Findings)**:
   - 3-5개의 핵심 발견사항
   - 데이터 기반의 구체적인 결론

2. **미래 전망 (Future Outlook)**:
   - 2025-2027: HOT_TRENDS의 주류화
   - 2028-2030: RISING_STARS의 게임체인저
   - 기술 발전 방향

3. **권고사항 (Recommendations)**:
   - 기업/투자자를 위한 권고
   - 연구자/개발자를 위한 권고
   - 정책입안자를 위한 권고

4. **맺음말 (Closing Remarks)**:
   - 보고서의 의의
   - 지속적인 모니터링 필요성

길이: 600-800단어
톤: 통찰력 있고 실행 가능한

Conclusion만 작성하여 반환하세요 (마크다운 형식, "## 6. Conclusion" 제목 포함)."""
        )
        self.conclusion_chain = conclusion_prompt | self.llm | str_parser
    
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Report Synthesis 실행
        
        Args:
            state: PipelineState (ContentAnalysisAgent 출력 포함)
        
        Returns:
            Updated PipelineState with synthesis results
        """
        print(f"\n{'='*60}")
        print(f"🎨 Report Synthesis Agent")
        print(f"{'='*60}\n")
        
        # Get input from ContentAnalysisAgent
        sections = state.get("sections", {})
        trends = state.get("trends", [])
        citations = state.get("citations", [])
        topic = state.get("user_input", "AI-Robotics Trend")
        
        arxiv_data = state.get("arxiv_data", {})
        rag_results = state.get("rag_results", {})
        news_data = state.get("news_data", {})
        
        arxiv_count = arxiv_data.get("total_count", 0) if arxiv_data else 0
        rag_count = rag_results.get("total_results", 0) if rag_results else 0
        news_count = news_data.get("total_articles", 0) if news_data else 0
        
        # Prepare section texts
        section_2 = self._combine_subsections(sections, "section_2")
        section_3 = self._combine_subsections(sections, "section_3")
        section_4 = self._combine_subsections(sections, "section_4")
        section_5 = self._combine_subsections(sections, "section_5")
        
        # Format key trends
        key_trends = self._format_trends(trends)
        
        try:
            # Generate sections concurrently
            print("📝 Generating Summary, Introduction, Conclusion...\n")
            
            # Summary
            print("   🔹 Generating Executive Summary...")
            summary = await self.summary_chain.ainvoke({
                "section_2": section_2,
                "section_3": section_3,
                "section_4": section_4,
                "section_5": section_5,
                "key_trends": key_trends
            })
            print("   ✅ Executive Summary generated\n")
            
            # Introduction
            print("   🔹 Generating Introduction...")
            section_1 = await self.intro_chain.ainvoke({
                "topic": topic,
                "arxiv_count": arxiv_count,
                "rag_count": rag_count,
                "news_count": news_count
            })
            print("   ✅ Introduction generated\n")
            
            # Conclusion
            print("   🔹 Generating Conclusion...")
            section_6 = await self.conclusion_chain.ainvoke({
                "section_2": section_2,
                "section_3": section_3,
                "section_4": section_4,
                "section_5": section_5,
                "key_trends": key_trends
            })
            print("   ✅ Conclusion generated\n")
            
            # Generate References
            print("   🔹 Generating References...")
            references = self._generate_references(citations)
            print(f"   ✅ References generated ({len(citations)} citations)\n")
            
            # Generate Appendix
            print("   🔹 Generating Appendix...")
            appendix = self._generate_appendix(trends, arxiv_count, rag_count, news_count)
            print("   ✅ Appendix generated\n")
            
            # Update state
            state["summary"] = summary
            state["section_1"] = section_1
            state["section_6"] = section_6
            state["references"] = references
            state["appendix"] = appendix
            
            print(f"✅ Report Synthesis Complete!")
            print(f"\n📊 Summary:")
            print(f"   - Summary: {len(summary)} chars")
            print(f"   - Introduction: {len(section_1)} chars")
            print(f"   - Conclusion: {len(section_6)} chars")
            print(f"   - References: {len(citations)} citations")
            print(f"   - Appendix: {len(appendix)} chars\n")
            
            return state
        
        except Exception as e:
            print(f"❌ Error in ReportSynthesisAgent: {str(e)}")
            raise
    
    def _combine_subsections(self, sections: Dict[str, str], section_prefix: str) -> str:
        """
        서브섹션들을 하나로 결합
        
        Args:
            sections: All sections dict
            section_prefix: e.g., "section_2"
        
        Returns:
            Combined section text
        """
        combined = []
        for key, value in sorted(sections.items()):
            if key.startswith(section_prefix):
                combined.append(value)
        
        return "\n\n".join(combined) if combined else "N/A"
    
    def _format_trends(self, trends: List[Any]) -> str:
        """
        트렌드를 포맷팅
        
        Args:
            trends: List of TrendTier objects
        
        Returns:
            Formatted trends text
        """
        if not trends:
            return "No trends classified yet."
        
        hot_trends = [t for t in trends if hasattr(t, 'tier') and t.tier == "HOT_TRENDS"]
        rising_stars = [t for t in trends if hasattr(t, 'tier') and t.tier == "RISING_STARS"]
        
        result = []
        
        if hot_trends:
            result.append("**HOT_TRENDS (1-2년 주류화):**")
            for trend in hot_trends[:5]:
                tech = getattr(trend, 'technology', 'Unknown')
                papers = getattr(trend, 'paper_count', 0)
                result.append(f"- {tech} ({papers} papers)")
        
        if rising_stars:
            result.append("\n**RISING_STARS (3-5년 게임체인저):**")
            for trend in rising_stars[:5]:
                tech = getattr(trend, 'technology', 'Unknown')
                papers = getattr(trend, 'paper_count', 0)
                result.append(f"- {tech} ({papers} papers)")
        
        return "\n".join(result)
    
    def _generate_references(self, citations: List[CitationEntry]) -> str:
        """
        인용 목록을 포맷팅
        
        Args:
            citations: List of CitationEntry objects
        
        Returns:
            Formatted references markdown
        """
        if not citations:
            return "## REFERENCE\n\nNo citations available."
        
        result = ["## REFERENCE\n"]
        
        # Group by source type
        arxiv_cites = [c for c in citations if c.source_type == "arxiv"]
        news_cites = [c for c in citations if c.source_type == "news"]
        report_cites = [c for c in citations if c.source_type == "report"]
        
        if arxiv_cites:
            result.append("### Academic Papers (arXiv)\n")
            for cite in arxiv_cites:
                result.append(f"[{cite.number}] {cite.to_reference_text()}")
        
        if report_cites:
            result.append("\n### Expert Reports\n")
            for cite in report_cites:
                result.append(f"[{cite.number}] {cite.to_reference_text()}")
        
        if news_cites:
            result.append("\n### News Articles\n")
            for cite in news_cites:
                result.append(f"[{cite.number}] {cite.to_reference_text()}")
        
        return "\n".join(result)
    
    def _generate_appendix(
        self,
        trends: List[Any],
        arxiv_count: int,
        rag_count: int,
        news_count: int
    ) -> str:
        """
        부록 생성
        
        Args:
            trends: Trend tiers
            arxiv_count: ArXiv paper count
            rag_count: RAG document count
            news_count: News article count
        
        Returns:
            Formatted appendix markdown
        """
        result = ["## APPENDIX\n"]
        
        # A. Data Collection Summary
        result.append("### A. Data Collection Summary\n")
        result.append(f"- **ArXiv Papers**: {arxiv_count} papers")
        result.append(f"- **Expert Reports**: {rag_count} documents")
        result.append(f"- **News Articles**: {news_count} articles")
        result.append(f"- **Total Data Sources**: {arxiv_count + rag_count + news_count}\n")
        
        # B. Trend Classification Details
        if trends:
            result.append("### B. Trend Classification Details\n")
            
            hot_trends = [t for t in trends if hasattr(t, 'tier') and t.tier == "HOT_TRENDS"]
            rising_stars = [t for t in trends if hasattr(t, 'tier') and t.tier == "RISING_STARS"]
            
            result.append(f"**HOT_TRENDS**: {len(hot_trends)} technologies")
            result.append(f"**RISING_STARS**: {len(rising_stars)} technologies\n")
        
        # C. Methodology Notes
        result.append("### C. Core Methodology\n")
        result.append("This report was generated using a multi-agent system with the following key methodologies:\n")
        
        result.append("1. **Human-in-the-Loop (HITL) for Planning & Writing:**")
        result.append("   - **Planning Phase:** The initial research plan generated by the AI is reviewed and refined by a human expert to ensure alignment and quality.")
        result.append("   - **Writing Phase:** The final draft is reviewed by a human who can request revisions or data recollection, creating an iterative refinement loop.\n")
        
        result.append("2. **ReAct-based Autonomous Data Collection:**")
        result.append("   - An autonomous agent based on the ReAct (Reasoning and Acting) framework dynamically collects data from various sources.")
        result.append("   - **Sources:** arXiv for academic papers, and real-time news crawling for market signals.\n")
        
        result.append("3. **Hybrid Retrieval-Augmented Generation (RAG):**")
        result.append("   - A sophisticated RAG system enhances the agent's knowledge by retrieving information from expert documents.")
        result.append("   - **Method:** It employs a hybrid search approach combining keyword-based search (BM25) and semantic search (Cosine Similarity), with a Maximal Marginal Relevance (MMR) reranker to ensure result diversity and relevance.\n")
        
        result.append("4. **Automated Analysis & Synthesis:**")
        result.append("   - **2-Tier Trend Classification:** Technologies are classified into 'HOT_TRENDS' (1-2 year horizon) and 'RISING_STARS' (3-5 year horizon).")
        result.append("   - **Automated Report Generation:** All sections, including summaries and conclusions, are synthesized by specialized agents, then assembled and translated into the final report.\n")
        
        return "\n".join(result)


