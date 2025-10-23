"""
Data Collection Tools for ReAct Agent

RAG 및 News 수집 도구의 LangChain 호환 래퍼
"""

import asyncio
from typing import Optional, Type, List, Dict
from concurrent.futures import ThreadPoolExecutor
from langchain.tools import BaseTool as LangChainBaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


# ============================================================
# RAG Tool Wrapper
# ============================================================

class RAGToolInput(BaseModel):
    """Input schema for RAG Tool"""
    query: str = Field(description="Search query to find relevant information in reference documents (FTSG, WEF reports)")
    top_k: int = Field(default=15, description="Number of documents to retrieve (default: 15)")
    search_type: str = Field(default="hybrid_mmr", description="Search type: 'hybrid_mmr' (recommended), 'dense', 'sparse'")


class RAGUtilWrapper(LangChainBaseTool):
    """
    LangChain wrapper for RAGTool
    
    Searches reference documents (FTSG, WEF reports) for relevant information
    about future technology trends, market forecasts, and industry applications.
    
    Use this tool to:
    - Find future technology predictions (5-year forecasts)
    - Get industry application examples
    - Extract market trends from expert reports
    - Validate ArXiv findings against professional analyses
    """
    
    name: str = "search_reference_documents"
    description: str = """Search reference documents (FTSG, WEF reports) for expert analysis on robotics and AI trends.

Use this when you need:
- Future technology forecasts (5-year predictions)
- Industry application cases
- Market trend analysis
- Expert opinions on emerging technologies

Input: A specific query about technology trends, applications, or forecasts.
Output: Relevant document excerpts from professional reports."""
    
    args_schema: Type[BaseModel] = RAGToolInput
    rag_tool: Optional[any] = Field(default=None, exclude=True)
    data_cache: List[Dict] = Field(default_factory=list, exclude=True)  # 실제 데이터 캐시
    used_queries: set = Field(default_factory=set, exclude=True)  # 이미 사용한 쿼리
    
    def __init__(self, rag_tool, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag_tool', rag_tool)
        object.__setattr__(self, 'data_cache', [])
        object.__setattr__(self, 'used_queries', set())
    
    def _run(
        self,
        query: str,
        top_k: int = 15,
        search_type: str = "hybrid_mmr",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute RAG search"""
        try:
            # 쿼리 정규화 (소문자, 공백 제거)
            query_normalized = query.lower().strip()
            
            # 쿼리 키워드 추출 (유사 쿼리 감지용)
            query_keywords = set(query_normalized.split())
            
            # 중복 쿼리 체크 - 정확히 같은 쿼리
            if query_normalized in self.used_queries:
                return (
                    f"⚠️ Already searched for '{query}'. "
                    f"Please try different keywords or aspects to get diverse information.\n"
                    f"Suggestions: Try more specific terms, different time periods, or related technologies."
                )
            
            # 유사 쿼리 체크 - 70% 이상 키워드가 겹치면 유사로 판단
            for used_query in self.used_queries:
                used_keywords = set(used_query.split())
                if len(query_keywords & used_keywords) / max(len(query_keywords), len(used_keywords)) >= 0.7:
                    return (
                        f"⚠️ Similar query already searched: '{used_query}'. "
                        f"Please try different aspects or more specific terms to avoid duplication.\n"
                        f"Current query: '{query}'"
                    )
            
            # 사용한 쿼리 기록
            self.used_queries.add(query_normalized)
            
            result = self.rag_tool._run(
                query=query,
                top_k=top_k,
                search_type=search_type
            )
            
            # 실제 데이터를 캐시에 저장 (citation용)
            documents = result.get("documents", [])
            citations = result.get("citations", [])
            if documents:
                cache_entry = {
                    "query": query,
                    "documents": documents,
                    "citations": citations,  # Citation 객체 포함
                    "total_results": len(documents)
                }
                self.data_cache.append(cache_entry)
            
            if not documents:
                return "No relevant documents found. Try different keywords."
            
            # Format output for LLM
            formatted_docs = []
            for i, doc in enumerate(documents[:10], 1):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                formatted_docs.append(f"[Document {i}]\n{content[:500]}...\n")
            
            return f"Found {len(documents)} relevant documents:\n\n" + "\n".join(formatted_docs)
        
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def get_cached_data(self) -> List[Dict]:
        """캐시된 데이터 가져오기"""
        return self.data_cache
    
    def clear_cache(self):
        """캐시 초기화"""
        self.data_cache.clear()


# ============================================================
# News Crawler Tool Wrapper
# ============================================================

class NewsCrawlerInput(BaseModel):
    """Input schema for News Crawler Tool"""
    keywords: str = Field(description="Comma-separated keywords to search for news articles (e.g., 'Boston Dynamics, Tesla Bot, warehouse robotics')")
    max_articles: int = Field(default=50, description="Maximum number of articles to collect (default: 50)")


class NewsCrawlerUtilWrapper(LangChainBaseTool):
    """
    LangChain wrapper for NewsCrawlerTool
    
    Collects recent news articles about robotics and AI from multiple sources.
    Searches by keywords and returns articles from the last 3 years.
    
    Use this tool to:
    - Find company announcements and product launches
    - Track industry news and market activities
    - Gather real-world application examples
    - Monitor technology adoption trends
    """
    
    name: str = "search_tech_news"
    description: str = """Search and collect recent tech news articles (last 3 years) about robotics and AI.

Use this when you need:
- Company announcements (e.g., Boston Dynamics, Tesla, Figure AI)
- Product launches and deployments
- Industry adoption news
- Real-world application examples
- Market activity and investments

Input: Keywords (comma-separated) related to robotics/AI companies, products, or applications.
Output: Collection of recent news articles with titles, descriptions, sources, and dates."""
    
    args_schema: Type[BaseModel] = NewsCrawlerInput
    news_tool: Optional[any] = Field(default=None, exclude=True)
    data_cache: List[Dict] = Field(default_factory=list, exclude=True)  # 실제 데이터 캐시
    used_keyword_sets: List[set] = Field(default_factory=list, exclude=True)  # 이미 사용한 키워드 조합
    
    def __init__(self, news_tool, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'news_tool', news_tool)
        object.__setattr__(self, 'data_cache', [])
        object.__setattr__(self, 'used_keyword_sets', [])
    
    def _run(
        self,
        keywords: str,
        max_articles: int = 50,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute news search (synchronous wrapper)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._arun(keywords, max_articles, run_manager))
    
    async def _arun(
        self,
        keywords: str,
        max_articles: int = 50,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute async news search"""
        try:
            # Parse keywords
            keyword_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
            
            if not keyword_list:
                return "Error: No valid keywords provided."
            
            # 키워드 세트로 변환 (순서 무관)
            keyword_set = set(keyword_list)
            
            # 중복 체크 - 이미 사용한 키워드 조합인지 확인
            for used_set in self.used_keyword_sets:
                # 50% 이상 겹치면 중복으로 간주
                overlap = len(keyword_set & used_set) / max(len(keyword_set), len(used_set))
                if overlap >= 0.5:
                    return (
                        f"⚠️ Already searched with similar keywords: {', '.join(used_set)}.\n"
                        f"Please try different keywords to get diverse news coverage.\n"
                        f"Suggestions: Try specific company names, product launches, or different aspects of the technology."
                    )
            
            # 사용한 키워드 세트 기록
            self.used_keyword_sets.append(keyword_set)
            
            # Call news tool (parallel search)
            result = await self.news_tool.search_by_keywords_parallel(
                keywords=keyword_list[:10],  # Max 10 keywords
                date_range="3 years"
            )
            
            articles = result.get("articles", [])
            citations = result.get("citations", [])
            
            # 실제 데이터를 캐시에 저장 (citation용)
            if articles:
                cache_entry = {
                    "keywords": keyword_list,
                    "articles": articles,
                    "citations": citations,  # Citation 객체 포함
                    "total_articles": len(articles),
                    "unique_sources": len(set(a.get("source", "") for a in articles))
                }
                self.data_cache.append(cache_entry)
            
            if not articles:
                return f"No news articles found for keywords: {', '.join(keyword_list)}. Try different or more specific keywords."
            
            # Format output for LLM
            formatted_articles = []
            for i, article in enumerate(articles[:max_articles], 1):
                formatted_articles.append(
                    f"[Article {i}] {article.get('published', 'N/A')}\n"
                    f"Title: {article.get('title', 'N/A')}\n"
                    f"Source: {article.get('source', 'N/A')}\n"
                    f"Description: {article.get('description', 'N/A')[:200]}...\n"
                )
            
            return (
                f"Collected {len(articles)} articles (showing {min(len(articles), max_articles)}):\n\n"
                + "\n".join(formatted_articles[:20])  # Show first 20 for context
                + f"\n\n(Total: {len(articles)} articles available)"
            )
        
        except Exception as e:
            return f"Error collecting news: {str(e)}"
    
    def get_cached_data(self) -> List[Dict]:
        """캐시된 데이터 가져오기"""
        return self.data_cache
    
    def clear_cache(self):
        """캐시 초기화"""
        self.data_cache.clear()

