"""
Data Collection Tools for ReAct Agent

RAG 및 News 수집 도구의 LangChain 호환 래퍼
"""

import asyncio
from typing import Optional, Type, List, Dict, Any, Union
from langchain.tools import BaseTool as LangChainBaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, field_validator

# ============================================================
# RAG Tool Wrapper
# ============================================================

class RAGToolInput(BaseModel):
    """Input schema for RAG Tool"""
    query: str = Field(description="Search query to find relevant information")
    top_k: int = Field(default=15, description="Number of documents to retrieve")
    search_type: str = Field(default="hybrid_mmr", description="Search type")


class RAGUtilWrapper(LangChainBaseTool):
    """LangChain wrapper for RAGTool"""
    
    name: str = "search_reference_documents"
    description: str = """Search reference documents (FTSG, WEF reports) for expert analysis.
Use for: Future forecasts, Industry cases, Market trends."""
    
    args_schema: Type[BaseModel] = RAGToolInput
    rag_tool: Optional[any] = Field(default=None, exclude=True)
    result_store: Dict = Field(default_factory=dict, exclude=True) # [핵심] 공유 딕셔너리
    used_queries: set = Field(default_factory=set, exclude=True)
    
    def __init__(self, rag_tool, result_store: Dict, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag_tool', rag_tool)
        object.__setattr__(self, 'result_store', result_store) # 참조 공유
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
            query_normalized = query.lower().strip()
            
            if query_normalized in self.used_queries:
                return f"⚠️ Already searched for '{query}'. Please try different keywords."
            
            self.used_queries.add(query_normalized)
            
            result = self.rag_tool._run(
                query=query,
                top_k=top_k,
                search_type=search_type
            )
            
            documents = result.get("documents", [])
            citations = result.get("citations", [])
            
            # [핵심] 수집된 데이터를 공유 딕셔너리에 저장
            if documents:
                # Key가 없으면 생성
                if "rag" not in self.result_store:
                    self.result_store["rag"] = []
                
                cache_entry = {
                    "query": query,
                    "documents": documents,
                    "citations": citations,
                    "total_results": len(documents)
                }
                self.result_store["rag"].append(cache_entry)
            
            if not documents:
                return "No relevant documents found. Try different keywords."
            
            # LLM에게는 요약된 텍스트만 반환
            formatted_docs = []
            for i, doc in enumerate(documents[:10], 1):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                formatted_docs.append(f"[Document {i}]\n{content[:500]}...\n")
            
            return f"Found {len(documents)} relevant documents:\n\n" + "\n".join(formatted_docs)
        
        except Exception as e:
            return f"Error searching documents: {str(e)}"



from typing import Union

class NewsCrawlerInput(BaseModel):
    """Input schema for News Crawler Tool"""
    keywords: Union[str, List[str]] = Field(
        description="Keywords to search: either a list ['keyword1', 'keyword2'] or comma-separated string 'keyword1, keyword2'"
    )
    max_articles: int = Field(default=50, description="Maximum number of articles to collect")
    
    @field_validator('keywords', mode='before')
    @classmethod
    def parse_keywords(cls, v):
        """Parse keywords from various formats"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                import json
                parsed = json.loads(v)
                if isinstance(parsed, dict) and "keywords" in parsed:
                    return parsed["keywords"]
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [k.strip() for k in v.split(",") if k.strip()]
        return v


class NewsCrawlerUtilWrapper(LangChainBaseTool):
    """LangChain wrapper for NewsCrawlerTool"""
    
    name: str = "search_tech_news"
    description: str = """Search and collect recent tech news articles (last 3 years).
Use for: Company announcements, Product launches, Market activity."""
    
    args_schema: Type[BaseModel] = NewsCrawlerInput
    news_tool: Optional[any] = Field(default=None, exclude=True)
    result_store: Dict = Field(default_factory=dict, exclude=True) # [핵심] 공유 딕셔너리
    used_keyword_sets: List[set] = Field(default_factory=list, exclude=True)
    
    def __init__(self, news_tool, result_store: Dict, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'news_tool', news_tool)
        object.__setattr__(self, 'result_store', result_store) # 참조 공유
        object.__setattr__(self, 'used_keyword_sets', [])
    
    def _run(
        self,
        keywords: Union[str, List[str]],
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
        keywords: Union[str, List[str]],
        max_articles: int = 50,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute async news search"""
        try:
            if isinstance(keywords, str):
                keyword_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
            else:
                keyword_list = [k.strip().lower() for k in keywords if k.strip()]
            
            if not keyword_list:
                return "Error: No valid keywords provided."
            
            # 중복 체크
            keyword_set = set(keyword_list)
            for used_set in self.used_keyword_sets:
                overlap = len(keyword_set & used_set) / max(len(keyword_set), len(used_set))
                if overlap >= 0.5:
                    return f"⚠️ Already searched with similar keywords: {', '.join(used_set)}."
            
            self.used_keyword_sets.append(keyword_set)
            
            # 뉴스 툴 실행
            result = await self.news_tool.search_by_keywords_parallel(
                keywords=keyword_list[:10],
                date_range="3 years"
            )
            
            articles = result.get("articles", [])
            citations = result.get("citations", [])
            
            # [핵심] 수집된 데이터를 공유 딕셔너리에 저장
            if articles:
                # Key가 없으면 생성
                if "news" not in self.result_store:
                    self.result_store["news"] = []
                    
                cache_entry = {
                    "keywords": keyword_list,
                    "articles": articles,
                    "citations": citations,
                    "total_articles": len(articles),
                    "unique_sources": len(set(a.get("source", "") for a in articles))
                }
                self.result_store["news"].append(cache_entry)
            
            if not articles:
                return f"No news articles found for keywords: {', '.join(keyword_list)}."
            
            # LLM에게는 요약된 텍스트만 반환
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
                + "\n".join(formatted_articles[:20])
                + f"\n\n(Total: {len(articles)} articles available)"
            )
        
        except Exception as e:
            return f"Error collecting news: {str(e)}"