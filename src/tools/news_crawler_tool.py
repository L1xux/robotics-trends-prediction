# NewsCrawlerTool(BaseTool)
"""
NewsCrawlerTool - 뉴스 크롤링 도구

GoogleNews 라이브러리를 사용하여 여러 소스에서 뉴스 기사를 수집합니다.
"""
from gnews import GNews
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.tools.base.base_tool import BaseTool
from src.tools.base.tool_config import ToolConfig
from src.core.models.citation_model import NewsCitation


class NewsCrawlerTool(BaseTool):
    """
    뉴스 크롤링 도구
    
    Features:
    - 키워드 기반 뉴스 검색
    - 날짜 범위 필터링
    - 여러 소스에서 수집
    - 중복 제거
    - Rate limiting 자동 처리
    
    Example:
        tool = NewsCrawlerTool(config)
        result = tool._run(
            keywords=["humanoid robot", "embodied AI"],
            date_range="3 years",
            sources=5
        )
    """
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        # GNews 클라이언트 초기화
        self.gnews = GNews(
            language='en',
            country='US',  # 전세계 검색을 위해 나중에 변경할 수도 있음
            period='7d',  # 기본값 (실제로는 date_range로 덮어씀)
            max_results=100,  # 키워드당 최대 결과
            exclude_websites=[]  # 제외할 사이트 없음
        )
    
    def _run(
        self,
        keywords: List[str],
        date_range: str = "3 years",
        sources: int = 5
    ) -> Dict[str, Any]:
        """
        뉴스 크롤링 실행
        
        Args:
            keywords: 검색 키워드 리스트
            date_range: 검색 날짜 범위
                - "3 years": 최근 3년
                - "36 months": 최근 36개월
                - "90 days": 최근 90일
            sources: 크롤링할 뉴스 소스 개수 (1-5)
        
        Returns:
            {
                "keywords": List[str],
                "date_range": str,
                "total_articles": int,
                "unique_sources": int,
                "articles": [
                    {
                        "title": str,
                        "url": str,
                        "source": str,
                        "published": str,  # YYYY-MM-DD
                        "snippet": str
                    },
                    ...
                ]
            }
        """
        # 재시도 로직
        for attempt in range(self.config.retry_count):
            try:
                # 1. 날짜 범위를 GNews period 형식으로 변환
                period = self._parse_date_range(date_range)
                self.gnews.period = period
                
                # 2. 각 키워드별로 뉴스 수집
                all_articles = []
                for i, keyword in enumerate(keywords, 1):
                    print(f"\n   [{i}/{len(keywords)}] Processing keyword: '{keyword}'")
                    articles = self._fetch_keyword(keyword, sources)
                    all_articles.extend(articles)
                    
                    # Rate limiting 방지 (키워드 간 짧은 대기)
                    if i < len(keywords):
                        time.sleep(0.5)  # 1초 → 0.5초로 단축
                
                # 3. 중복 제거 (URL 기준)
                unique_articles = self._deduplicate(all_articles)
                
                # 4. 날짜 범위 필터링
                filtered_articles = self._filter_by_date(unique_articles, date_range)
                
                # 5. 날짜 순으로 정렬 (최신순)
                filtered_articles.sort(key=lambda x: x['published'], reverse=True)
                
                # 6. 소스 개수 계산
                unique_sources = len(set(article['source'] for article in filtered_articles))
                
                # 결과 요약 출력
                print(f"\n   [News] Collection Summary:")
                print(f"       Total collected: {len(all_articles)} articles")
                print(f"       After deduplication: {len(unique_articles)} articles")
                print(f"       After date filter: {len(filtered_articles)} articles")
                print(f"       Unique sources: {unique_sources}")
                
                # Citations 생성
                citations = []
                for article in filtered_articles:
                    try:
                        citation = NewsCitation(
                            title=article.get("title", "Unknown"),
                            source=article.get("source", "Unknown"),
                            published=article.get("published", "Unknown"),
                            url=article.get("url", "")
                        )
                        citations.append(citation)
                    except Exception as e:
                        print(f"       Citation 생성 실패: {str(e)[:100]}")
                        continue
                
                print(f"       Citations created: {len(citations)}")
                
                return {
                    "keywords": keywords,
                    "date_range": date_range,
                    "total_articles": len(filtered_articles),
                    "unique_sources": unique_sources,
                    "articles": filtered_articles,
                    "citations": citations  # Citation 객체 리스트 추가
                }
            
            except Exception as e:
                if attempt < self.config.retry_count - 1:
                    # 재시도 전 대기
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # 최종 실패
                    return {
                        "keywords": keywords,
                        "date_range": date_range,
                        "total_articles": 0,
                        "unique_sources": 0,
                        "articles": [],
                        "error": f"Failed after {self.config.retry_count} attempts: {str(e)}"
                    }
    
    def _fetch_keyword(self, keyword: str, sources: int) -> List[Dict[str, Any]]:
        """
        단일 키워드로 뉴스 수집 (더 많은 기사 수집)
        
        Args:
            keyword: 검색 키워드
            sources: 목표 소스 개수 (무시하고 최대한 많이 수집)
        
        Returns:
            기사 리스트
        """
        articles = []
        
        try:
            print(f"   [News] Searching keyword: '{keyword}'")
            
            # GNews 검색
            results = self.gnews.get_news(keyword)
            
            if not results:
                print(f"   [News] No results for '{keyword}'")
                return []
            
            print(f"   [News] Found {len(results)} raw results for '{keyword}'")
            
            # 소스별로 그룹화
            source_dict = {}
            for result in results:
                source = result.get('publisher', {}).get('title', 'Unknown')
                if source not in source_dict:
                    source_dict[source] = []
                source_dict[source].append(result)
            
            # 모든 소스에서 기사 수집 (sources 파라미터 무시)
            # 각 소스에서 최대 10개씩 수집하여 다양성 확보
            for source, source_articles in source_dict.items():
                # 해당 소스에서 최대 10개 기사 수집
                for article in source_articles[:10]:
                    try:
                        formatted = self._format_article(article)
                        articles.append(formatted)
                    except Exception as format_error:
                        print(f"   [News] Failed to format article: {format_error}")
                        continue
            
            print(f"   [News] Collected {len(articles)} articles from {len(source_dict)} sources for '{keyword}'")
            
            return articles
        
        except Exception as e:
            print(f"   [News] Failed to fetch news for keyword '{keyword}': {e}")
            import traceback
            print(f"   📜 [News] Traceback: {traceback.format_exc()}")
            return []
    
    def _format_article(self, raw_article: Dict) -> Dict[str, Any]:
        """
        GNews 결과를 표준 형식으로 변환
        
        Args:
            raw_article: GNews 원본 결과
        
        Returns:
            표준화된 기사 딕셔너리
        """
        # 발행일 파싱
        published_date = raw_article.get('published date', '')
        try:
            # "Mon, 15 Jan 2024 12:34:56 GMT" 형식
            dt = datetime.strptime(published_date, "%a, %d %b %Y %H:%M:%S %Z")
            published_str = dt.strftime("%Y-%m-%d")
        except:
            published_str = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "title": raw_article.get('title', ''),
            "url": raw_article.get('url', ''),
            "source": raw_article.get('publisher', {}).get('title', 'Unknown'),
            "published": published_str,
            "snippet": raw_article.get('description', '')
        }
    
    def _deduplicate(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        URL 기준 중복 제거
        
        Args:
            articles: 기사 리스트
        
        Returns:
            중복 제거된 기사 리스트
        """
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article['url']
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles
    
    def _filter_by_date(
        self,
        articles: List[Dict[str, Any]],
        date_range: str
    ) -> List[Dict[str, Any]]:
        """
        날짜 범위로 필터링
        
        Args:
            articles: 기사 리스트
            date_range: "3 years", "36 months", "90 days"
        
        Returns:
            필터링된 기사 리스트
        """
        # 날짜 범위 파싱
        cutoff_date = self._calculate_cutoff_date(date_range)
        
        filtered = []
        for article in articles:
            try:
                article_date = datetime.strptime(article['published'], "%Y-%m-%d")
                if article_date >= cutoff_date:
                    filtered.append(article)
            except:
                # 날짜 파싱 실패 시 포함
                filtered.append(article)
        
        return filtered
    
    def _parse_date_range(self, date_range: str) -> str:
        """
        date_range를 GNews period 형식으로 변환
        
        Args:
            date_range: "3 years", "36 months", "90 days"
        
        Returns:
            GNews period 형식 (예: "3y", "36m", "90d")
        """
        date_range = date_range.lower().strip()
        
        # 숫자 추출
        parts = date_range.split()
        if len(parts) != 2:
            return "7d"  # 기본값
        
        number = parts[0]
        unit = parts[1]
        
        # 단위 변환
        if 'year' in unit:
            return f"{number}y"
        elif 'month' in unit:
            return f"{number}m"
        elif 'day' in unit:
            return f"{number}d"
        else:
            return "7d"  # 기본값
    
    def _calculate_cutoff_date(self, date_range: str) -> datetime:
        """
        date_range로부터 cutoff 날짜 계산
        
        Args:
            date_range: "3 years", "36 months", "90 days"
        
        Returns:
            cutoff datetime
        """
        date_range = date_range.lower().strip()
        parts = date_range.split()
        
        if len(parts) != 2:
            return datetime.now() - timedelta(days=7)  # 기본값
        
        number = int(parts[0])
        unit = parts[1]
        
        now = datetime.now()
        if 'year' in unit:
            return now - timedelta(days=365 * number)
        elif 'month' in unit:
            return now - timedelta(days=30 * number)
        elif 'day' in unit:
            return now - timedelta(days=number)
        else:
            return now - timedelta(days=7)  # 기본값
    
    async def search_by_keywords_parallel(
        self,
        keywords: List[str],
        date_range: str = "3 years"
    ) -> Dict[str, Any]:
        """
        키워드별 병렬 뉴스 수집
        
        Args:
            keywords: 검색 키워드 리스트
            date_range: 날짜 범위
        
        Returns:
            {
                "articles": [...],
                "count": int,
                "keywords_searched": int
            }
        """
        print(f"\n병렬 뉴스 수집 시작: {len(keywords)}개 키워드")
        
        # ThreadPoolExecutor로 병렬 수집
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._search_single_keyword,
                    keyword,
                    date_range
                )
                for keyword in keywords[:10]  # 최대 10개
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 병합
        all_articles = []
        seen_urls = set()
        
        for result in results:
            if isinstance(result, Exception):
                print(f"   ⚠️ 에러 발생: {result}")
                continue
            
            if isinstance(result, list):
                for article in result:
                    url = article.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_articles.append(article)
        
        # 날짜순 정렬
        all_articles.sort(
            key=lambda x: datetime.strptime(x['published'], "%Y-%m-%d") if x.get('published') else datetime.min,
            reverse=True
        )
        
        print(f"   총 {len(all_articles)}개 기사 수집 완료 (중복 제거 후)")
        
        # Citations 생성
        citations = []
        for article in all_articles:
            try:
                citation = NewsCitation(
                    title=article.get("title", "Unknown"),
                    source=article.get("source", "Unknown"),
                    published=article.get("published", "Unknown"),
                    url=article.get("url", "")
                )
                citations.append(citation)
            except Exception as e:
                print(f"   Citation 생성 실패: {str(e)[:100]}")
                continue
        
        print(f"   Citations created: {len(citations)}")
        
        return {
            "articles": all_articles,
            "citations": citations,  # Citation 객체 리스트 추가
            "count": len(all_articles),
            "keywords_searched": len(keywords)
        }
    
    def _search_single_keyword(
        self,
        keyword: str,
        date_range: str
    ) -> List[Dict[str, Any]]:
        """
        단일 키워드로 뉴스 수집 (동기 함수)
        
        Args:
            keyword: 검색 키워드
            date_range: 날짜 범위
        
        Returns:
            기사 리스트
        """
        print(f"   '{keyword}' 수집 중...")
        
        try:
            # GNews period 설정
            period = self._parse_date_range(date_range)
            gnews = GNews(
                language='en',
                country='US',
                period=period,
                max_results=10  # 키워드당 10개
            )
            
            # 검색
            articles = gnews.get_news(keyword)
            
            # 형식 변환
            formatted = []
            for article in articles:
                # 날짜 파싱
                published_date = article.get('published date', '')
                try:
                    # "Mon, 15 Jan 2024 12:34:56 GMT" 형식
                    dt = datetime.strptime(published_date, "%a, %d %b %Y %H:%M:%S %Z")
                    published_str = dt.strftime("%Y-%m-%d")
                except:
                    # 파싱 실패 시 현재 날짜
                    published_str = datetime.now().strftime("%Y-%m-%d")
                
                formatted.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published': published_str,  # 변환된 날짜 사용
                    'source': article.get('publisher', {}).get('title', 'Unknown'),
                    'keyword': keyword
                })
            
            print(f"      ✓ '{keyword}': {len(formatted)}개")
            return formatted
            
        except Exception as e:
            print(f"      ✗ '{keyword}' 실패: {e}")
            return []
    
    async def _arun(self, *args, **kwargs):
        """비동기 실행 (LangChain 호환)"""
        raise NotImplementedError("NewsCrawlerTool does not support async execution")

