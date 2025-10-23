"""
Data Collection Agent (ReAct Architecture)

데이터 수집을 담당하는 Agent
- ArXiv 논문 수집 및 키워드 추출 (먼저 실행)
- RAG + News를 ReAct Agent가 자율적으로 사용
- 데이터 충분성 판단 (보고서 작성 가능 여부)
- Citation 정보 수집
"""

import json
import time
from typing import List, Any, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState, WorkflowStatus
from src.core.settings import Settings
from src.core.models.citation_model import (
    CitationCollection
)
from config.prompts.data_collections_prompts import (
    SUFFICIENCY_CHECK_PROMPT,
    SYSTEM_PAPER_KEYWORD_SUMMARY_PROMPT
)


class CollectionConstants:
    """Constants for data collection"""
    MAX_ATTEMPTS = 3
    MAX_AGENT_ITERATIONS = 25
    MAX_EXECUTION_TIME = 1200  # 20 minutes
    SUFFICIENCY_MODEL = "gpt-4o"
    SUFFICIENCY_TEMPERATURE = 0.3
    ARXIV_MAX_RETRIES = 3
    RETRY_SLEEP_SECONDS = 3

    # Arxiv settings
    DEFAULT_ARXIV_CATEGORIES = "cs.RO,cs.AI"
    MAX_RESULTS_PER_KEYWORD = 100
    YEARS_BACK = 3

    # Keyword settings
    MAX_RECENT_PAPERS_ANALYSIS = 20
    MIN_COMPANY_MENTIONS = 2
    MAX_RAW_KEYWORDS_FOR_LLM = 100
    MAX_PAPER_TITLES_FOR_LLM = 15


class DataCollectionAgent(BaseAgent):
    """
    Data Collection Agent (ReAct Architecture)
    
    Workflow:
    1. ArXiv 논문 수집 (먼저 실행, tool 아님)
    2. 논문에서 키워드 추출
    3. ReAct Agent가 RAG + News tool을 자율적으로 사용
       - 확장된 키워드로 검색
       - 충분할 때까지 반복
    4. 데이터 충분성 판단 (보고서 작성 가능 여부)
    5. 부족하면 Agent가 추가 수집
    
    Features:
    - ArXiv: 초기 실행으로 기술 landscape 파악
    - ReAct Agent: RAG + News tool만 사용
    - Citation 자동 생성
    - 충분성 판단: 보고서 목차 기준
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any],
        config: AgentConfig,
        raw_tools: Optional[List[Any]] = None,
        settings: Optional[Settings] = None
    ):
        super().__init__(llm, tools, config)
        self.settings = settings or Settings()
        self._raw_tools = raw_tools or []
        self._arxiv_tool: Optional[Any] = None
        self._rag_tool: Optional[Any] = None
        self._news_tool: Optional[Any] = None
        self._initialize_tools()

        # Sufficiency check LLM
        self._sufficiency_llm = ChatOpenAI(
            model=CollectionConstants.SUFFICIENCY_MODEL,
            temperature=CollectionConstants.SUFFICIENCY_TEMPERATURE
        )

        # ReAct Agent
        self._agent_executor: Optional[AgentExecutor] = None
        self._setup_react_agent()

    def _initialize_tools(self) -> None:
        """Initialize raw tools by index"""
        if len(self._raw_tools) >= 3:
            self._arxiv_tool = self._raw_tools[0]
            self._rag_tool = self._raw_tools[1]
            self._news_tool = self._raw_tools[2]
    
    def _setup_react_agent(self) -> None:
        """Setup ReAct Agent with RAG + News tools only"""
        react_template = """You are a data collection specialist for AI-Robotics trend reports.

You have already collected ArXiv papers and extracted technical keywords.
Now use these tools to gather market trends and expert forecasts:

{tools}

STRICT FORMAT:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (valid JSON)
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have collected sufficient data
Final Answer: summary of collected data

Guidelines:
1. Use search_reference_documents for expert forecasts and technology trends
2. Use search_tech_news for company activities and real-world applications
3. Try different keyword combinations to get diverse sources
4. Collect at least 10 RAG documents and 20 news articles
5. When you have enough data, provide Final Answer

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        react_prompt = PromptTemplate(
            template=react_template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )

        react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )

        self._agent_executor = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,
            max_iterations=CollectionConstants.MAX_AGENT_ITERATIONS,
            max_execution_time=CollectionConstants.MAX_EXECUTION_TIME,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Data Collection 실행
        
        Workflow:
        1. ArXiv 논문 수집 (직접 실행)
        2. 키워드 확장
        3. ReAct Agent로 RAG + News 수집
        4. 충분성 판단
        5. 부족하면 재시도
        """
        print(f"\n{'='*60}")
        print(f"📚 Data Collection Agent 실행 중...")
        print(f"{'='*60}\n")
        
        # State에서 데이터 가져오기
        planning_output = state.get("planning_output")
        if not planning_output:
            raise ValueError("planning_output이 State에 없습니다.")
        
        topic = planning_output.topic
        keywords = state.get("keywords", [])
        
        print(f"📝 주제: {topic}")
        print(f"🔑 초기 키워드: {', '.join(keywords)}\n")
        
        # 최대 재시도
        max_attempts = 3
        attempt = 0
        
        # 수집 결과
        arxiv_data = None
        rag_results = None
        news_data = None
        expanded_keywords = []
        citations = CitationCollection()
        
        while attempt < max_attempts:
            attempt += 1
            print(f"\n{'='*60}")
            print(f"🔄 데이터 수집 시도 {attempt}/{max_attempts}")
            print(f"{'='*60}\n")
            
            try:
                # ========================================
                # Step 1: ArXiv 논문 수집 (먼저 실행)
                # ========================================
                if attempt == 1:  # 첫 시도에만 ArXiv 수집
                    print(f"📄 Step 1: ArXiv 논문 수집 중...")
                    arxiv_data = await self._collect_arxiv(keywords, planning_output)
                    
                    if arxiv_data and arxiv_data.get("total_count", 0) > 0:
                        print(f"   ✅ ArXiv 수집 완료: {arxiv_data['total_count']}편")
                        # Tool에서 생성한 citations 가져오기
                        tool_citations = arxiv_data.get("citations", [])
                        citations.arxiv_citations.extend(tool_citations)
                        print(f"   ✅ ArXiv Citation: {len(tool_citations)}개\n")
                    else:
                        print(f"   ⚠️  ArXiv 데이터 없음\n")
                    
                    # Step 2: 키워드 추출
                    print(f"🔑 Step 2: 논문에서 키워드 추출 중...")
                    expanded_keywords = await self._expand_keywords(arxiv_data, keywords)
                    print(f"   ✅ 추출된 키워드 ({len(expanded_keywords)}개): {', '.join(expanded_keywords[:10])}{'...' if len(expanded_keywords) > 10 else ''}\n")
                
                # ========================================
                # Step 3: ReAct Agent로 RAG + News 수집
                # ========================================
                print(f"🤖 Step 3: ReAct Agent 시작 (RAG + News)...\n")
                
                # Agent에게 줄 질문 생성
                question = self._generate_agent_question(
                    topic, expanded_keywords, arxiv_data, attempt
                )
                
                # ReAct Agent 실행
                result = await self._agent_executor.ainvoke({
                    "input": question
                })
                
                print(f"\n✅ ReAct Agent 완료!\n")
                
                # Wrapper 캐시에서 Agent가 수집한 데이터 추출 (재검색 없음!)
                print(f"📊 Agent 수집 데이터 추출 중 (캐시 사용)...\n")
                
                rag_results, news_data = self._extract_data_from_cache(
                    topic, expanded_keywords
                )
                
                # Citation 추가 (Tool에서 생성한 것 가져오기)
                if rag_results:
                    tool_citations = rag_results.get("citations", [])
                    citations.rag_citations.extend(tool_citations)
                    print(f"   ✅ RAG Citation: {len(tool_citations)}개")
                
                if news_data:
                    tool_citations = news_data.get("citations", [])
                    citations.news_citations.extend(tool_citations)
                    print(f"   ✅ News Citation: {len(tool_citations)}개\n")
                
                # ========================================
                # Step 4: 데이터 충분성 판단
                # ========================================
                print(f"🔍 Step 4: 데이터 충분성 판단 중...")
                sufficiency_result = await self._check_sufficiency(
                    topic, keywords, expanded_keywords,
                    arxiv_data, rag_results, news_data
                )
                
                print(f"   📊 충분성 점수: {sufficiency_result.get('overall_score', 0):.2f}")
                print(f"   판정: {'✅ 충분함' if sufficiency_result.get('sufficient', False) else '⚠️  부족함'}\n")
                
                # 충분하면 종료
                if sufficiency_result.get("sufficient", False):
                    print(f"✅ 데이터 수집 완료! (보고서 작성 가능)\n")
                    break
                
                # 부족하면 재시도
                if attempt < max_attempts:
                    print(f"⚠️  데이터가 부족합니다. 추가 수집을 시도합니다.")
                    print(f"   부족한 영역: {', '.join(sufficiency_result.get('missing_areas', []))}")
                    print(f"   권장사항: {', '.join(sufficiency_result.get('recommendations', []))}\n")
                    
                    # 키워드 추가 확장
                    expanded_keywords = self._further_expand_keywords(expanded_keywords)
                else:
                    print(f"⚠️  최대 시도 횟수 도달. 현재 데이터로 진행합니다.\n")
            
            except Exception as e:
                print(f"❌ 데이터 수집 중 에러: {e}")
                print(f"error")
                
                if attempt >= max_attempts:
                    print(f"\n⚠️  최대 재시도 횟수 도달. 수집된 데이터로 진행합니다.\n")
                    break
                
                print(f"🔄 재시도 중...\n")
                time.sleep(2)
                continue
        
        # 결과 요약
        print(f"\n{'='*60}")
        print(f"📊 데이터 수집 결과 요약")
        print(f"{'='*60}")
        print(f"📄 ArXiv 논문: {arxiv_data.get('total_count', 0) if arxiv_data else 0}편")
        print(f"📖 RAG 결과: {rag_results.get('total_results', 0) if rag_results else 0}개")
        print(f"📰 뉴스 기사: {news_data.get('total_articles', 0) if news_data else 0}개")
        print(f"🔑 추출 키워드: {len(expanded_keywords)}개")
        print(f"📚 인용 정보 (출처):")
        print(f"   - ArXiv 논문: {len(citations.arxiv_citations)}개")
        print(f"   - 뉴스 기사: {len(citations.news_citations)}개")
        print(f"   - RAG 문서: {len(citations.rag_citations)}개")
        print(f"   - 총 {len(citations.get_all_citations())}개 출처")
        print(f"{'='*60}\n")
        
        # State 업데이트
        state["arxiv_data"] = arxiv_data or {}
        state["news_data"] = news_data or {}
        state["rag_results"] = rag_results or {}
        state["expanded_keywords"] = expanded_keywords
        state["citations"] = citations
        state["status"] = WorkflowStatus.DATA_COLLECTION_COMPLETE.value
        
        return state
    
    def _extract_data_from_cache(
        self,
        topic: str,
        expanded_keywords: List[str]
    ) -> tuple:
        """
        Wrapper 캐시에서 Agent가 수집한 데이터 추출 (재검색 불필요!)
        
        Returns:
            (rag_results, news_data)
        """
        # RAG wrapper에서 캐시 가져오기
        rag_wrapper = None
        news_wrapper = None
        
        for tool in self.tools:
            if 'reference' in tool.name.lower() or 'document' in tool.name.lower():
                rag_wrapper = tool
            elif 'news' in tool.name.lower():
                news_wrapper = tool
        
        # RAG 캐시 데이터 통합
        rag_documents = []
        rag_citations = []
        seen_contents = set()
        seen_rag_citations = set()  # Citation 중복 제거용
        
        if rag_wrapper and hasattr(rag_wrapper, 'get_cached_data'):
            cached_rag = rag_wrapper.get_cached_data()
            print(f"   📖 RAG 캐시: {len(cached_rag)}번 검색 수행됨")
            
            for cache_entry in cached_rag:
                entry_docs = cache_entry.get("documents", [])
                entry_citations = cache_entry.get("citations", [])
                
                # Document와 Citation을 함께 처리 (1:1 매칭 가정)
                for i, doc in enumerate(entry_docs):
                    content = doc.get("content", "")
                    if content and content not in seen_contents:
                        seen_contents.add(content)
                        rag_documents.append(doc)
                        
                        # 대응하는 citation 추가 (중복 체크)
                        if i < len(entry_citations):
                            citation = entry_citations[i]
                            citation_key = citation.full_citation if hasattr(citation, 'full_citation') else str(citation)
                            if citation_key not in seen_rag_citations:
                                seen_rag_citations.add(citation_key)
                                rag_citations.append(citation)
        
        # News 캐시 데이터 통합
        news_articles = []
        news_citations = []
        seen_urls = set()
        seen_news_citations = set()  # Citation 중복 제거용
        
        if news_wrapper and hasattr(news_wrapper, 'get_cached_data'):
            cached_news = news_wrapper.get_cached_data()
            print(f"   📰 News 캐시: {len(cached_news)}번 검색 수행됨")
            
            for cache_entry in cached_news:
                entry_articles = cache_entry.get("articles", [])
                entry_citations = cache_entry.get("citations", [])
                
                # Article과 Citation을 함께 처리 (1:1 매칭 가정)
                for i, article in enumerate(entry_articles):
                    url = article.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        news_articles.append(article)
                        
                        # 대응하는 citation 추가 (중복 체크)
                        if i < len(entry_citations):
                            citation = entry_citations[i]
                            citation_key = citation.full_citation if hasattr(citation, 'full_citation') else str(citation)
                            if citation_key not in seen_news_citations:
                                seen_news_citations.add(citation_key)
                                news_citations.append(citation)
        
        # RAG 결과 정리
        rag_results = {
            "query": topic,
            "search_type": "agent_collected_cached",
            "total_results": len(rag_documents),
            "documents": rag_documents,
            "citations": rag_citations  # Citations 추가
        } if rag_documents else None
        
        # News 결과 정리
        news_data = {
            "keywords": expanded_keywords,
            "date_range": "3 years",
            "total_articles": len(news_articles),
            "unique_sources": len(set(a.get("source", "") for a in news_articles)),
            "articles": news_articles,
            "citations": news_citations  # Citations 추가
        } if news_articles else None
        
        # 최종 통계 출력
        if rag_results:
            print(f"   ✅ RAG 데이터: {len(rag_documents)}개 문서 추출됨")
        if news_data:
            print(f"   ✅ News 데이터: {len(news_articles)}개 기사 추출됨\n")
        
        return rag_results, news_data
    
    def _generate_agent_question(
        self,
        topic: str,
        keywords: List[str],
        arxiv_data: Optional[Dict],
        attempt: int
    ) -> str:
        """ReAct Agent에게 줄 질문 생성"""
        paper_count = arxiv_data.get("total_count", 0) if arxiv_data else 0
        
        if attempt == 1:
            # 로봇/기술 관련 키워드만 선별
            question = f"""Collect comprehensive data for a ROBOTICS/AUTOMATION technology trend report on "{topic}".

I have already collected {paper_count} ArXiv papers and extracted these ROBOTICS-SPECIFIC keywords:
{', '.join(keywords[:])}

Your task:
1. Use search_reference_documents to find expert forecasts about ROBOTICS/AI technologies
   - Search for: robotics trends, AI in automation, 5-year predictions for robots
   - Look for: industrial robotics, service robots, autonomous systems
   - Get: future technology forecasts, market adoption predictions
   - Use TECHNOLOGY keywords from the list above
   - Target: At least 10 relevant documents

2. Use search_tech_news to find recent ROBOTICS/AI news
   - Use ONLY technology keywords (e.g., "collaborative robots", "adaptive welding")
   - Focus on: robot deployments, new robot products, automation innovations
   - Search for: specific companies + technology combinations
   - DO NOT use generic terms like "market activities", "announcements"
   - Target: At least 20 diverse articles about robot technologies

**IMPORTANT:** Use SPECIFIC technology keywords, not generic business terms!
When you have enough data, provide a final summary."""
        
        else:
            question = f"""Continue collecting more data for "{topic}".

Current status: Need more diverse sources.

Use the keywords: {', '.join(keywords[:10])}

Focus on:
- Finding different sources and perspectives
- Covering various aspects of the technology
- Getting recent market activities

Continue until you have comprehensive coverage."""
        
        return question
    
    async def _collect_arxiv(
        self,
        keywords: List[str],
        planning_output: Any
    ) -> Optional[Dict[str, Any]]:
        """ArXiv 논문 수집 (키워드별 병렬 검색)"""
        for retry in range(CollectionConstants.ARXIV_MAX_RETRIES):
            try:
                if not self._arxiv_tool:
                    print("   ⚠️  ArXiv Tool을 찾을 수 없습니다.")
                    return None

                # Get categories from collection plan
                categories_str = planning_output.collection_plan.arxiv.categories
                if categories_str.lower() == "all":
                    categories_str = CollectionConstants.DEFAULT_ARXIV_CATEGORIES

                print(f"   📚 카테고리: {categories_str}")
                print(f"   🔑 키워드: {len(keywords)}개 (각 키워드당 병렬 검색)")
                print(f"   🎯 키워드당 최대: {CollectionConstants.MAX_RESULTS_PER_KEYWORD}편\n")

                # Search ArXiv papers in parallel
                result = self._arxiv_tool.search_by_keywords_parallel(
                    keywords=keywords,
                    categories=categories_str,
                    max_results_per_keyword=CollectionConstants.MAX_RESULTS_PER_KEYWORD,
                    years_back=CollectionConstants.YEARS_BACK
                )

                if result and result.get("total_count", 0) > 0:
                    return result
                else:
                    print(f"   ⚠️  논문을 찾지 못했습니다. 재시도 {retry + 1}/{CollectionConstants.ARXIV_MAX_RETRIES}")
                    if retry < CollectionConstants.ARXIV_MAX_RETRIES - 1:
                        time.sleep(CollectionConstants.RETRY_SLEEP_SECONDS)
                        continue
                    return result

            except Exception as e:
                print(f"   ❌ ArXiv 수집 에러 (시도 {retry + 1}/{CollectionConstants.ARXIV_MAX_RETRIES}): {e}")
                if retry < CollectionConstants.ARXIV_MAX_RETRIES - 1:
                    print(f"   🔄 {CollectionConstants.RETRY_SLEEP_SECONDS}초 후 재시도...")
                    time.sleep(CollectionConstants.RETRY_SLEEP_SECONDS)
                    continue
                else:
                    print(f"   error")
                    return None

        return None
    
    
    async def _expand_keywords(
        self,
        arxiv_data: Optional[Dict[str, Any]],
        initial_keywords: List[str]
    ) -> List[str]:
        """
        ArXiv 논문에서 emerging/특이한 기술 키워드 추출
        
        목표: 5년 후 트렌드를 예측할 수 있는 새롭고 구체적인 키워드 발굴
        
        1. 논문 제목/초록에서 키워드 추출
        2. 기업 정보 추출
        3. LLM이 emerging/specific 키워드 선별
           - 일반적 키워드(machine learning) 제외
           - 구체적/새로운 키워드(neuromorphic computing) 우선
        """
        if not arxiv_data or not arxiv_data.get("papers"):
            return initial_keywords  # 논문이 없으면 초기 키워드 사용
        
        # 1. 논문에서 추출된 키워드 수집
        raw_keywords = set()
        for paper in arxiv_data["papers"]:
            paper_keywords = paper.get("keywords", [])
            for kw in paper_keywords:
                raw_keywords.add(kw)
        
        # 2. 논문에서 언급된 기업 추출
        companies_mentioned = set()
        if "companies_mentioned" in arxiv_data:
            for company, count in arxiv_data["companies_mentioned"].items():
                if count >= CollectionConstants.MIN_COMPANY_MENTIONS:
                    companies_mentioned.add(company)
        
        # 3. 논문 제목 분석 (최신 기술 경향 파악)
        recent_papers_info = []
        if arxiv_data.get("papers"):
            # 최신 논문만 분석 (최신 트렌드)
            for paper in arxiv_data["papers"][:CollectionConstants.MAX_RECENT_PAPERS_ANALYSIS]:
                recent_papers_info.append({
                    "title": paper.get("title", ""),
                    "year": paper.get("published", "")[:4]  # YYYY만 추출
                })
        
        # 4. 모든 후보 키워드 결합
        all_candidates = list(raw_keywords) + list(companies_mentioned)
        
        # 논문 키워드가 없으면 초기 키워드 사용
        if not all_candidates:
            return initial_keywords
        
        # 5. LLM으로 emerging/specific 키워드 선별 ⭐
        filtered_keywords = await self._filter_emerging_keywords(
            initial_keywords=initial_keywords,
            raw_keywords=all_candidates,
            companies=list(companies_mentioned),
            recent_papers=recent_papers_info
        )
        
        return filtered_keywords
    
    async def _filter_emerging_keywords(
        self,
        initial_keywords: List[str],
        raw_keywords: List[str],
        companies: Optional[List[str]] = None,
        recent_papers: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        LLM을 사용하여 emerging/specific 키워드 선별
        
        목표: 5년 후 트렌드 예측을 위한 새롭고 구체적인 키워드 발굴
        
        Args:
            initial_keywords: 사용자 질문 기반 초기 키워드
            raw_keywords: arXiv 논문에서 추출된 모든 키워드
            companies: 논문에서 언급된 기업 리스트
            recent_papers: 최신 논문 제목 리스트
        
        Returns:
            emerging/specific 키워드 리스트 (기업 포함)
        """
        companies = companies or []
        recent_papers = recent_papers or []
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # 최신 논문 제목 요약
        paper_titles = "\n".join([f"- ({p['year']}) {p['title']}" for p in recent_papers[:CollectionConstants.MAX_PAPER_TITLES_FOR_LLM]])

        filter_prompt = ChatPromptTemplate.from_messages(SYSTEM_PAPER_KEYWORD_SUMMARY_PROMPT)
        
        try:
            chain = filter_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({
                "initial_keywords": ", ".join(initial_keywords),
                "paper_titles": paper_titles if paper_titles else "No recent papers available",
                "raw_keywords": ", ".join(raw_keywords[:CollectionConstants.MAX_RAW_KEYWORDS_FOR_LLM]),
                "companies": ", ".join(companies) if companies else "None mentioned"
            })
            
            # JSON 파싱
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                filtered = json.loads(json_match.group(0))
                # 초기 키워드는 항상 포함
                final_keywords = list(set(initial_keywords + filtered))
                
                # 통계 출력
                tech_keywords = [kw for kw in filtered if kw not in companies]
                company_keywords = [kw for kw in filtered if kw in companies]
                
                print(f"   🔍 Emerging keyword extraction:")
                print(f"      • Raw candidates: {len(raw_keywords)}")
                print(f"      • Emerging/specific tech: {len(tech_keywords)}")
                print(f"      • Companies identified: {len(company_keywords)}")
                print(f"      • Total (with initial): {len(final_keywords)}")
                print(f"      • Top emerging tech: {', '.join(tech_keywords[:5])}...")
                
                if company_keywords:
                    print(f"      • Companies: {', '.join(company_keywords[:5])}{'...' if len(company_keywords) > 5 else ''}")
                
                return sorted(final_keywords[:40])  # 최대 40개로 제한
            else:
                print(f"   ⚠️  Keyword filtering failed, using initial keywords only")
                return initial_keywords
                
        except Exception as e:
            print(f"   ⚠️  Keyword filtering error: {e}, using initial keywords only")
            return initial_keywords
    
    def _further_expand_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 유지 (재시도 시에도 확장 없음)"""
        # 재시도 시에도 동일한 키워드 사용 (확장 없음)
        return keywords
    
    async def _check_sufficiency(
        self,
        topic: str,
        initial_keywords: List[str],
        expanded_keywords: List[str],
        arxiv_data: Optional[Dict[str, Any]],
        rag_results: Optional[Dict[str, Any]],
        news_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """데이터 충분성 판단 (보고서 작성 가능 여부)"""
        try:
            # 데이터 요약
            arxiv_count = arxiv_data.get("total_count", 0) if arxiv_data else 0
            arxiv_companies = list(arxiv_data.get("company_stats", {}).keys()) if arxiv_data else []
            
            rag_count = rag_results.get("total_results", 0) if rag_results else 0
            rag_queries = rag_results.get("queries", []) if rag_results else []
            
            news_count = news_data.get("total_articles", 0) if news_data else 0
            news_sources_count = news_data.get("unique_sources", 0) if news_data else 0
            
            # Prompt
            prompt = SUFFICIENCY_CHECK_PROMPT.format(
                topic=topic,
                keywords=", ".join(expanded_keywords or initial_keywords),
                arxiv_count=arxiv_count,
                arxiv_date_range="2022-2025",
                arxiv_companies=", ".join(arxiv_companies[:10]),
                arxiv_keywords=", ".join((expanded_keywords or initial_keywords)[:10]),
                rag_count=rag_count,
                rag_queries=", ".join(rag_queries),
                news_count=news_count,
                news_sources=news_sources_count,
                news_date_range="3 years"
            )
            
            # LLM 호출
            response = await self.sufficiency_llm.ainvoke(prompt)
            content = response.content
            
            # JSON 파싱
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            return result
        
        except Exception as e:
            print(f"   ❌ 충분성 판단 에러: {e}")
            
            # 기본 판단
            arxiv_ok = (arxiv_data.get("total_count", 0) if arxiv_data else 0) >= 30
            rag_ok = (rag_results.get("total_results", 0) if rag_results else 0) >= 10
            news_ok = (news_data.get("total_articles", 0) if news_data else 0) >= 20
            
            sufficient = arxiv_ok and rag_ok and news_ok
            
            return {
                "sufficient": sufficient,
                "overall_score": 0.7 if sufficient else 0.5,
                "section_scores": {
                    "section_2": 0.7 if arxiv_ok else 0.3,
                    "section_3": 0.7 if news_ok else 0.3,
                    "section_4": 0.7 if rag_ok else 0.3,
                    "citation": 0.7,
                    "balance": 0.7 if sufficient else 0.5
                },
                "missing_areas": [],
                "recommendations": [],
                "reasoning": "Default judgment"
            }


