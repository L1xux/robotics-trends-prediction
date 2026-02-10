"""
Report Synthesis Agent (English Only Version)

Receives outputs (Section 2,3,4,5) from Content Analysis Agent
and generates Summary, Introduction, Conclusion, References, and Appendix in ENGLISH.
"""

from typing import List, Any, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState
from src.core.models.citation_model import CitationEntry


class ReportSynthesisLLM(BaseAgent):
    """
    Report Synthesis Agent
    
    Generates the remaining sections based on ContentAnalysisAgent's output:
    - SUMMARY (Executive Summary)
    - Section 1: Introduction
    - Section 6: Conclusion
    - REFERENCE (Citations)
    - APPENDIX
    
    CRITICAL: All outputs are generated strictly in English.
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
        """LCEL Chains Setup with English Prompts"""
        str_parser = StrOutputParser()
        
        # ---------------------------------------------------------------------
        # 1. Summary Chain (English)
        # ---------------------------------------------------------------------
        summary_prompt = ChatPromptTemplate.from_template(
            """You are an expert technical report writer specializing in AI and Robotics.

Your task is to write an **Executive Summary** based on the following analysis:

**Technology Trends (Section 2):**
{section_2}

**Market Trends & Applications (Section 3):**
{section_3}

**5-Year Forecast (Section 4):**
{section_4}

**Business Implications (Section 5):**
{section_5}

**Key Trends:**
{key_trends}

**CRITICAL RULE:**
- **WRITE ONLY IN ENGLISH.**
- Do NOT use any other language.

**Executive Summary Requirements:**

1. **Length**: 200-300 words
2. **Structure**:
   - **Overview**: Compress the core message of the report into 2-3 sentences.
   - **5-Year Forecast**: Summarize major technological shifts for 2025-2030 (mention HOT_TRENDS and RISING_STARS).

3. **Tone**: Concise, impactful, C-level executive targeting.
4. **Style**: Data-driven, specific insights.

**Constraints**:
- Do not include headers like "Key Findings" or "Recommendations" in the summary.
- Distinctly separate 'Overview' and '5-Year Forecast'.

Return ONLY the Executive Summary in Markdown format."""
        )
        self.summary_chain = summary_prompt | self.llm | str_parser
        
        # ---------------------------------------------------------------------
        # 2. Introduction Chain (English)
        # ---------------------------------------------------------------------
        intro_prompt = ChatPromptTemplate.from_template(
            """You are an expert technical report writer specializing in AI and Robotics.

Report Topic: {topic}

Based on the following information, write **1. Introduction**:

**Data Collected:**
- ArXiv Papers: {arxiv_count}
- Expert Reports (RAG): {rag_count}
- News Articles: {news_count}

**Report Structure:**
- Section 1: Introduction
- Section 2: AI-Robotics Technology Trend Analysis
- Section 3: Market Trends & Applications
- Section 4: 5-Year Forecast (2025-2030)
- Section 5: Implications for Business
- Section 6: Conclusion

**CRITICAL RULE:**
- **WRITE ONLY IN ENGLISH.**
- Do NOT use any other language.

**Introduction Requirements:**

1. **### Background**:
   - Current status and importance of the AI/Robotics industry regarding the topic.
   - The necessity of this report.

2. **### Purpose**:
   - Objectives and scope of the report.
   - Target audience (Executives, R&D, Investors).

3. **### Methodology**:
   - Briefly mention data sources (ArXiv, Reports, News).
   - Analysis approach (Trend classification, 5-year forecasting).

4. **### Structure**:
   - Brief overview of what each section covers.

**Format**: Use `###` Markdown headers for each subsection.
**Length**: 400-600 words.
**Tone**: Professional and objective.

Return ONLY the Introduction in Markdown format (start with "## 1. Introduction")."""
        )
        self.intro_chain = intro_prompt | self.llm | str_parser
        
        # ---------------------------------------------------------------------
        # 3. Conclusion Chain (English)
        # ---------------------------------------------------------------------
        conclusion_prompt = ChatPromptTemplate.from_template(
            """You are an expert technical report writer specializing in AI and Robotics.

Based on the following analysis, write **6. Conclusion**:

**Technology Trends:**
{section_2}

**Market Trends:**
{section_3}

**5-Year Forecast:**
{section_4}

**Business Implications:**
{section_5}

**Key Trends:**
{key_trends}

**CRITICAL RULE:**
- **WRITE ONLY IN ENGLISH.**
- Do NOT use any other language.

**Conclusion Requirements:**

1. **Key Findings**:
   - 3-5 major takeaways.
   - Specific, data-backed conclusions.

2. **Future Outlook**:
   - 2025-2027: Mainstreaming of HOT_TRENDS.
   - 2028-2030: Game-changing potential of RISING_STARS.
   - Direction of technological evolution.

3. **Recommendations**:
   - For Companies/Investors.
   - For Researchers/Developers.
   - For Policymakers.

4. **Closing Remarks**:
   - Significance of the report.
   - Need for continuous monitoring.

**Length**: 600-800 words.
**Tone**: Insightful and actionable.

Return ONLY the Conclusion in Markdown format (start with "## 6. Conclusion")."""
        )
        self.conclusion_chain = conclusion_prompt | self.llm | str_parser
    
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute Report Synthesis
        """
        print(f"\n{'='*60}")
        print(f"Report Synthesis Agent (English Only)")
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
            print("Generating Summary, Introduction, Conclusion (in English)...\n")
            
            # Summary
            print("   Generating Executive Summary...")
            summary = await self.summary_chain.ainvoke({
                "section_2": section_2,
                "section_3": section_3,
                "section_4": section_4,
                "section_5": section_5,
                "key_trends": key_trends
            })
            summary = self._remove_markdown_wrapper(summary)
            print("   Executive Summary generated\n")
            
            # Introduction
            print("   Generating Introduction...")
            section_1 = await self.intro_chain.ainvoke({
                "topic": topic,
                "arxiv_count": arxiv_count,
                "rag_count": rag_count,
                "news_count": news_count
            })
            section_1 = self._remove_markdown_wrapper(section_1)
            print("   Introduction generated\n")
            
            # Conclusion
            print("   Generating Conclusion...")
            section_6 = await self.conclusion_chain.ainvoke({
                "section_2": section_2,
                "section_3": section_3,
                "section_4": section_4,
                "section_5": section_5,
                "key_trends": key_trends
            })
            section_6 = self._remove_markdown_wrapper(section_6)
            print("   Conclusion generated\n")
            
            # Generate References
            print("   Generating References...")
            references = self._generate_references(citations)
            print(f"   References generated ({len(citations)} citations)\n")
            
            # Generate Appendix
            print("   Generating Appendix...")
            appendix = self._generate_appendix(trends, arxiv_count, rag_count, news_count)
            print("   Appendix generated\n")
            
            # Update state
            state["summary"] = summary
            state["section_1"] = section_1
            state["section_6"] = section_6
            state["references"] = references
            state["appendix"] = appendix
            
            print(f"Report Synthesis Complete!")
            print(f"\nSummary:")
            print(f"   - Summary: {len(summary)} chars")
            print(f"   - Introduction: {len(section_1)} chars")
            print(f"   - Conclusion: {len(section_6)} chars")
            print(f"   - References: {len(citations)} citations")
            print(f"   - Appendix: {len(appendix)} chars\n")
            
            return state
        
        except Exception as e:
            print(f"Error in ReportSynthesisAgent: {str(e)}")
            raise
    
    def _remove_markdown_wrapper(self, text: str) -> str:
        """Remove markdown code block wrapper from LLM response"""
        text = text.strip()
        
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        
        if text.endswith("```"):
            text = text[:-3].strip()
        
        return text
    
    def _combine_subsections(self, sections: Dict[str, str], section_prefix: str) -> str:
        """Combine subsections"""
        combined = []
        for key, value in sorted(sections.items()):
            if key.startswith(section_prefix):
                combined.append(value)
        
        return "\n\n".join(combined) if combined else "N/A"
    
    def _format_trends(self, trends: List[Any]) -> str:
        """Format trends for the prompt"""
        if not trends:
            return "No trends classified yet."
        
        hot_trends = [t for t in trends if hasattr(t, 'tier') and t.tier == "HOT_TRENDS"]
        rising_stars = [t for t in trends if hasattr(t, 'tier') and t.tier == "RISING_STARS"]
        
        result = []
        
        if hot_trends:
            result.append("**HOT_TRENDS (Mainstream in 1-2 years):**")
            for trend in hot_trends[:5]:
                tech = getattr(trend, 'technology', 'Unknown')
                papers = getattr(trend, 'paper_count', 0)
                result.append(f"- {tech} ({papers} papers)")
        
        if rising_stars:
            result.append("\n**RISING_STARS (Game Changers in 3-5 years):**")
            for trend in rising_stars[:5]:
                tech = getattr(trend, 'technology', 'Unknown')
                papers = getattr(trend, 'paper_count', 0)
                result.append(f"- {tech} ({papers} papers)")
        
        return "\n".join(result)
    
    def _generate_references(self, citations: List[CitationEntry]) -> str:
        """Generate formatted references section"""
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
        """Generate Appendix section"""
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