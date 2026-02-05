"""
Data Collection Agent (ReAct Architecture)
Refactored for modularity, readability, and shared state management.
"""

import json
import time
import traceback
from typing import List, Any, Dict, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState, WorkflowStatus
from src.core.settings import Settings
from src.core.models.citation_model import CitationCollection
from config.prompts.data_collections_prompts import (
    SUFFICIENCY_CHECK_PROMPT,
    SYSTEM_PAPER_KEYWORD_SUMMARY_PROMPT
)

class CollectionConstants:
    """Constants for data collection configuration."""
    MAX_ATTEMPTS = 3
    MAX_AGENT_ITERATIONS = 25
    MAX_EXECUTION_TIME = 1200
    SUFFICIENCY_MODEL = "gpt-4o-mini"
    SUFFICIENCY_TEMPERATURE = 0.3
    ARXIV_MAX_RETRIES = 3
    RETRY_SLEEP_SECONDS = 3
    DEFAULT_ARXIV_CATEGORIES = "cs.RO,cs.AI"
    MAX_RESULTS_PER_KEYWORD = 100
    YEARS_BACK = 3
    MAX_RECENT_PAPERS_ANALYSIS = 20
    MIN_COMPANY_MENTIONS = 2
    MAX_RAW_KEYWORDS_FOR_LLM = 100
    MAX_PAPER_TITLES_FOR_LLM = 15


class DataCollectionAgent(BaseAgent):
    """
    Data Collection Agent (ReAct Architecture).
    Orchestrates ArXiv search and ReAct-based web/doc search using a shared result store.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any],
        config: AgentConfig,
        result_store: Dict,
        raw_tools: Optional[List[Any]] = None,
        settings: Optional[Settings] = None
    ):
        super().__init__(llm, tools, config)
        self.settings = settings or Settings()
        self.result_store = result_store
        self._raw_tools = raw_tools or []
        
        # Identify specific tools
        self._arxiv_tool = self._find_tool_by_name("arxiv")
        
        # Initialize helper LLM for checks
        self._sufficiency_llm = ChatOpenAI(
            model=CollectionConstants.SUFFICIENCY_MODEL,
            temperature=CollectionConstants.SUFFICIENCY_TEMPERATURE
        )

        self._agent_executor = None
        self._setup_react_agent()

    def _find_tool_by_name(self, name_part: str) -> Optional[Any]:
        """Find a tool in raw_tools by partial name match."""
        if self._raw_tools:
            for tool in self._raw_tools:
                tool_name = getattr(tool, "name", "").lower()
                if name_part in tool_name:
                    return tool
        return None
    
    def _setup_react_agent(self) -> None:
        """Setup ReAct Agent with Strict Formatting Rules"""
            
        # [수정 1] 프롬프트에 'Valid Examples' 추가하여 포맷 준수 강제
        react_template = """You are a data collection specialist.
            You must use the provided tools to gather data. DO NOT answer from your own knowledge.

            TOOLS:
            ------
            {tools}

            FORMAT INSTRUCTIONS:
            --------------------
            You MUST use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do next
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action (valid JSON)
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I have collected sufficient data
            Final Answer: the final summary of collected data

            EXAMPLES:
            ---------
            Question: Research trends in humanoid robotics.
            Thought: I need to find forecast reports first.
            Action: search_reference_documents
            Action Input: {{"query": "humanoid robot market forecast"}}
            Observation: Found reports predicting 50% growth...
            Thought: Now I need recent news.
            Action: search_tech_news
            Action Input: {{"keywords": "humanoid robot launch"}}
            Observation: Tesla Optimus update released...
            Thought: I have enough information.
            Final Answer: The humanoid market is growing...

            CURRENT TASK:
            -------------
            Question: {input}
            Thought:{agent_scratchpad}"""

        react_prompt = PromptTemplate(
            template=react_template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )

        react_agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=react_prompt)

        def _handle_parsing_errors(error) -> str:
            """
            LLM이 포맷을 어겼을 때, 재시도를 위한 수정 지침을 반환
            """
            response = str(error).split("`")[-1] # 에러 메시지에서 LLM의 잘못된 출력 추출
            return f"Invalid Format! You missed 'Action:' or 'Final Answer'. \nYour output was: {response}\n\n YOU MUST STRICTLY FOLLOW THIS FORMAT:\nThought: ...\nAction: ...\nAction Input: ...\n\nIf you are done, use:\nThought: I am done.\nFinal Answer: ..."

        self._agent_executor = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,
            max_iterations=CollectionConstants.MAX_AGENT_ITERATIONS,
            max_execution_time=CollectionConstants.MAX_EXECUTION_TIME,
            handle_parsing_errors=_handle_parsing_errors, 
            return_intermediate_steps=True
        )
    
    async def execute(self, state: PipelineState) -> PipelineState:
        """Main execution flow for data collection."""
        print(f"\n{'='*60}\nData Collection Agent Started\n{'='*60}")
        
        planning_output = state.get("planning_output")
        if not planning_output:
            raise ValueError("Missing 'planning_output' in state.")
        
        topic = planning_output.topic
        keywords = state.get("keywords", [])
        
        print(f"Topic: {topic}")
        print(f"Initial Keywords: {', '.join(keywords)}\n")
        
        # State variables
        arxiv_data = None
        expanded_keywords = []
        citations = CitationCollection()
        
        attempt = 0
        while attempt < CollectionConstants.MAX_ATTEMPTS:
            attempt += 1
            print(f"\nCollection Attempt {attempt}/{CollectionConstants.MAX_ATTEMPTS}")
            
            try:
                # --- Phase 1: ArXiv Collection (First attempt only) ---
                if attempt == 1:
                    arxiv_data, expanded_keywords = await self._run_arxiv_phase(keywords, planning_output, citations)
                
                # --- Phase 2: ReAct Agent (RAG + News) ---
                await self._run_react_phase(topic, expanded_keywords, arxiv_data, attempt)
                
                # --- Phase 3: Extract Data & Citations ---
                rag_results, news_data = self._extract_data_from_store(topic, expanded_keywords, citations)
                
                # --- Phase 4: Sufficiency Check ---
                sufficiency = await self._check_sufficiency(
                    topic, keywords, expanded_keywords, arxiv_data, rag_results, news_data
                )
                
                print(f"   Sufficiency Score: {sufficiency.get('overall_score', 0):.2f}")
                
                if sufficiency.get("sufficient", False):
                    print("Data collection sufficient.")
                    break
                
                if attempt < CollectionConstants.MAX_ATTEMPTS:
                    print("Data insufficient. Retrying with expanded scope...")
                    # Logic to further expand keywords could go here
                else:
                    print("Max attempts reached. Proceeding with available data.")
            
            except Exception as e:
                print(f"Error during collection: {e}")
                traceback.print_exc()
                if attempt >= CollectionConstants.MAX_ATTEMPTS:
                    break
                time.sleep(2)

        # Final State Update
        state["arxiv_data"] = arxiv_data or {}
        state["news_data"] = news_data or {}
        state["rag_results"] = rag_results or {}
        state["expanded_keywords"] = expanded_keywords
        state["citations"] = citations
        state["status"] = WorkflowStatus.DATA_COLLECTION_COMPLETE.value
        
        return state

    # --- Helper Methods for Execution Phases ---

    async def _run_arxiv_phase(self, keywords, planning_output, citations) -> Tuple[Dict, List[str]]:
        """Executes ArXiv search and keyword expansion."""
        print(f"Step 1: ArXiv Research...")
        arxiv_data = await self._collect_arxiv(keywords, planning_output)
        
        if arxiv_data and arxiv_data.get("total_count", 0) > 0:
            print(f"   Collected {arxiv_data['total_count']} papers.")
            citations.arxiv_citations.extend(arxiv_data.get("citations", []))
        else:
            print("   No ArXiv papers found.")
        
        print(f"Step 2: Keyword Expansion...")
        expanded_keywords = await self._expand_keywords(arxiv_data, keywords)
        print(f"   Keywords expanded to {len(expanded_keywords)} terms.")
        return arxiv_data, expanded_keywords

    async def _run_react_phase(self, topic, keywords, arxiv_data, attempt):
        """Executes the ReAct agent for RAG and News."""
        print(f"Step 3: ReAct Agent (RAG + News)...")
        question = self._generate_agent_question(topic, keywords, arxiv_data, attempt)
        await self._agent_executor.ainvoke({"input": question})
        print(f"   ReAct Agent finished.")

    def _extract_data_from_store(self, topic, keywords, citations) -> Tuple[Dict, Dict]:
        """Extracts data from the shared result_store and updates citations."""
        print(f"Extracting data from shared store...")
        
        # Extract RAG
        rag_raw = self.result_store.get("rag", [])
        rag_docs, rag_cits = self._process_rag_entries(rag_raw)
        citations.rag_citations.extend(rag_cits)
        
        # Extract News
        news_raw = self.result_store.get("news", [])
        news_arts, news_cits = self._process_news_entries(news_raw)
        citations.news_citations.extend(news_cits)
        
        # Format Results
        rag_results = {
            "query": topic, "search_type": "agent_collected_cached",
            "total_results": len(rag_docs), "documents": rag_docs, "citations": rag_cits
        } if rag_docs else None
        
        news_data = {
            "keywords": keywords, "date_range": "3 years",
            "total_articles": len(news_arts), "articles": news_arts, "citations": news_cits
        } if news_arts else None
        
        if rag_results: print(f"   RAG: {len(rag_docs)} docs")
        if news_data: print(f"   News: {len(news_arts)} articles")
        
        return rag_results, news_data

    # --- Data Processing Helpers ---

    def _process_rag_entries(self, entries: List[Dict]) -> Tuple[List, List]:
        """Deduplicates and processes RAG entries from store."""
        docs, cits = [], []
        seen_content, seen_cits = set(), set()
        
        for entry in entries:
            entry_docs = entry.get("documents", [])
            # Normalize Document objects
            if entry_docs and hasattr(entry_docs[0], 'page_content'):
                 entry_docs = [{"content": d.page_content, "metadata": d.metadata} for d in entry_docs]
            
            for i, doc in enumerate(entry_docs):
                content = doc.get("content", "")
                if content and content not in seen_content:
                    seen_content.add(content)
                    docs.append(doc)
                    
                    # Add citation if available
                    entry_cits = entry.get("citations", [])
                    if i < len(entry_cits):
                        cit = entry_cits[i]
                        if str(cit) not in seen_cits:
                            seen_cits.add(str(cit))
                            cits.append(cit)
        return docs, cits

    def _process_news_entries(self, entries: List[Dict]) -> Tuple[List, List]:
        """Deduplicates and processes News entries from store."""
        arts, cits = [], []
        seen_urls, seen_cits = set(), set()
        
        for entry in entries:
            entry_arts = entry.get("articles", [])
            entry_cits = entry.get("citations", [])
            
            for i, art in enumerate(entry_arts):
                url = art.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    arts.append(art)
                    
                    if i < len(entry_cits):
                        cit = entry_cits[i]
                        if str(cit) not in seen_cits:
                            seen_cits.add(str(cit))
                            cits.append(cit)
        return arts, cits

    # --- Logic Helpers ---

    def _generate_agent_question(self, topic, keywords, arxiv_data, attempt):
        paper_count = arxiv_data.get("total_count", 0) if arxiv_data else 0
        if attempt == 1:
            return f"""Collect comprehensive data for a ROBOTICS/AUTOMATION technology trend report on "{topic}".
I have already collected {paper_count} ArXiv papers and extracted these keywords: {', '.join(keywords[:15])}
Task:
1. Use search_reference_documents to find expert forecasts (5-year predictions) and market analysis.
2. Use search_tech_news to find recent company activities, product launches, and real-world applications.
Collect at least 10 RAG documents and 20 news articles. Provide a final summary."""
        else:
            return f"""Continue collecting more data for "{topic}". Current status: Need more diverse sources.
Use keywords: {', '.join(keywords[:10])}. Focus on recent market activities."""

    async def _collect_arxiv(self, keywords, planning_output):
        for retry in range(CollectionConstants.ARXIV_MAX_RETRIES):
            try:
                if not self._arxiv_tool: return None
                categories = planning_output.collection_plan.arxiv.categories
                if categories.lower() == "all": categories = CollectionConstants.DEFAULT_ARXIV_CATEGORIES
                
                result = self._arxiv_tool.search_by_keywords_parallel(
                    keywords=keywords, categories=categories,
                    max_results_per_keyword=CollectionConstants.MAX_RESULTS_PER_KEYWORD,
                    years_back=CollectionConstants.YEARS_BACK
                )
                if result and result.get("total_count", 0) > 0: return result
                time.sleep(CollectionConstants.RETRY_SLEEP_SECONDS)
            except Exception as e:
                print(f"   ArXiv Error: {e}")
                time.sleep(CollectionConstants.RETRY_SLEEP_SECONDS)
        return None

    async def _expand_keywords(self, arxiv_data, initial_keywords):
        if not arxiv_data or not arxiv_data.get("papers"): return initial_keywords
        
        raw_keywords = set()
        for p in arxiv_data["papers"]:
            for k in p.get("keywords", []): raw_keywords.add(k)
            
        companies = set()
        if "companies_mentioned" in arxiv_data:
            for c, count in arxiv_data["companies_mentioned"].items():
                if count >= CollectionConstants.MIN_COMPANY_MENTIONS: companies.add(c)
                
        recent_papers = []
        for p in arxiv_data["papers"][:CollectionConstants.MAX_RECENT_PAPERS_ANALYSIS]:
            recent_papers.append({"title": p.get("title", ""), "year": p.get("published", "")[:4]})
            
        all_candidates = list(raw_keywords) + list(companies)
        if not all_candidates: return initial_keywords
        
        return await self._filter_emerging_keywords(initial_keywords, all_candidates, list(companies), recent_papers)

    async def _filter_emerging_keywords(self, initial, raw, companies, papers):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        paper_titles = "\n".join([f"- ({p['year']}) {p['title']}" for p in papers[:CollectionConstants.MAX_PAPER_TITLES_FOR_LLM]])
        filter_prompt = ChatPromptTemplate.from_messages(SYSTEM_PAPER_KEYWORD_SUMMARY_PROMPT)
        
        try:
            chain = filter_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({
                "initial_keywords": ", ".join(initial), "paper_titles": paper_titles,
                "raw_keywords": ", ".join(raw[:CollectionConstants.MAX_RAW_KEYWORDS_FOR_LLM]),
                "companies": ", ".join(companies)
            })
            
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                filtered = json.loads(json_match.group(0))
                final = list(set(initial + filtered))
                return sorted(final[:40])
            return initial
        except: return initial

    async def _check_sufficiency(self, topic, initial, expanded, arxiv, rag, news):
        """Checks if enough data has been collected, with auto-pass logic."""
        try:
            arxiv_count = arxiv.get("total_count", 0) if arxiv else 0
            rag_count = rag.get("total_results", 0) if rag else 0
            news_count = news.get("total_articles", 0) if news else 0
            
            print(f"\n   [Sufficiency Check] ArXiv: {arxiv_count}, RAG: {rag_count}, News: {news_count}")

            # Auto-pass criteria
            if arxiv_count >= 10 and (rag_count >= 3 or news_count >= 5):
                print(f"   Auto-Pass: Minimum criteria met.")
                return {"sufficient": True, "overall_score": 0.9}

            prompt = SUFFICIENCY_CHECK_PROMPT.format(
                topic=topic, keywords=", ".join(expanded or initial),
                arxiv_count=arxiv_count, arxiv_date_range="2022-2025",
                arxiv_companies="Various", arxiv_keywords="Analysis",
                rag_count=rag_count, rag_queries="",
                news_count=news_count, news_sources=0, news_date_range="3 years"
            )
            
            response = await self._sufficiency_llm.ainvoke(prompt)
            content = response.content
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception as e:
            print(f"   Sufficiency Check Failed: {e}")
            # Fallback logic
            is_sufficient = (arxiv_count >= 10) and (rag_count + news_count >= 5)
            return {"sufficient": is_sufficient, "overall_score": 0.5}