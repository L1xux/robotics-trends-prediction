"""
ArxivTool - arXiv 논문 검색 도구 (개선 버전)

개선사항:
1. 페이징 에러 시 수집한 데이터 보존
2. 500개 제한 (충분한 데이터 수집)
3. 부분 성공 반환 기능
4. 더 나은 에러 핸들링

변경 이력:
- v1.0: 초기 버전
- v1.1: 페이징 에러 대응, 부분 결과 반환 추가
- v1.2: 제한을 100 → 500으로 증가
"""
import arxiv
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.tools.base.base_tool import BaseTool
from src.tools.base.tool_config import ToolConfig
from src.core.models.citation_model import ArXivCitation


class ArxivTool(BaseTool):
    """
    arXiv 논문 검색 도구
    
    Features:
    - 키워드 기반 검색
    - 날짜 범위 필터링
    - 카테고리 필터링
    - 최대 결과 개수 제한 (500개까지)
    - 페이징 에러 시 부분 결과 반환
    - 기업명 추출 (논문에서 언급된 기업)
    - 키워드 추출 (제목/초록 기반)
    
    Example:
        tool = ArxivTool(config)
        result = tool._run(
            keywords=["humanoid robot", "manufacturing"],
            date_range="2022-01-01 to 2025-10-22",
            categories="cs.RO,cs.AI",
            max_results="500"
        )
    """
    
    # 주요 AI/로봇 기업 리스트 (대소문자 무시)
    COMPANIES = {
        # 빅테크
        "Google", "Meta", "Microsoft", "Apple", "Amazon", "Tesla", "NVIDIA",
        "DeepMind", "OpenAI", "Anthropic",
        
        # 로봇 전문
        "Boston Dynamics", "Figure AI", "Agility Robotics", "ANYbotics",
        "Universal Robots", "ABB Robotics", "KUKA", "FANUC", "Yaskawa",
        
        # 자동차
        "Toyota", "BMW", "Mercedes", "Waymo", "Cruise", "Ford", "GM",
        
        # 중국 빅테크
        "Baidu", "Alibaba", "Tencent", "ByteDance", "Huawei",
        
        # 기타
        "SpaceX", "Neuralink", "iRobot", "Fetch Robotics", "Rethink Robotics"
    }
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        # arxiv Client 설정 (보수적)
        self.client = arxiv.Client(
            page_size=50,       # 페이지당 50개
            delay_seconds=4,    # 요청 간 4초 대기 (rate limit 회피)
            num_retries=2       # 재시도 2회
        )
        
        # 기업명 매칭용 정규식 패턴 생성
        self.company_patterns = [
            re.compile(r'\b' + re.escape(company) + r'\b', re.IGNORECASE)
            for company in self.COMPANIES
        ]
    
    def _run(
        self,
        keywords: List[str],
        date_range: str,
        categories: str = "all",
        max_results: str = "unlimited"
    ) -> Dict[str, Any]:
        """
        arXiv 논문 검색 실행
        
        Args:
            keywords: 검색 키워드 리스트
            date_range: 날짜 범위 (예: "2022-01-01 to 2025-10-22")
            categories: arXiv 카테고리 (all 또는 "cs.RO,cs.AI")
            max_results: 최대 결과 개수 (unlimited 또는 숫자)
                - unlimited: 100개로 제한 (arXiv API 한계)
                - 숫자: 해당 개수만큼 (최대 100)
        
        Returns:
            {
                "total_count": int,
                "papers": [
                    {
                        "title": str,
                        "authors": List[str],
                        "abstract": str,
                        "url": str,
                        "published": str,
                        "categories": List[str]
                    },
                    ...
                ],
                "warning": str (optional, 부분 결과인 경우)
            }
        """
        # 재시도 로직 밖에서 best_papers 관리
        best_papers = []
        
        # 재시도 로직
        for attempt in range(self.config.retry_count):
            try:
                # 1. 쿼리 생성
                query = self._build_query(keywords, categories)
                
                # 2. 날짜 범위 파싱
                start_date, end_date = self._parse_date_range(date_range)
                
                # 3. max_results 파싱 (500개 제한)
                if max_results == "unlimited":
                    max_results_int = 500  # 충분한 데이터 수집
                else:
                    max_results_int = min(int(max_results), 500)
                
                print(f"Searching arXiv: max_results={max_results_int}")
                if attempt > 0:
                    print(f"   (Retry {attempt}/{self.config.retry_count})")
                
                # 4. arXiv 검색
                search = arxiv.Search(
                    query=query,
                    max_results=max_results_int,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                # 5. 결과 수집 (예외 처리 추가)
                papers = []
                count = 0
                filtered_out = 0
                start_time = time.time()
                
                print(f"   Starting paper collection...")
                print(f"   Date filter: {start_date} to {end_date}")
                
                try:
                    for result in self.client.results(search):
                        count += 1
                        
                        # 안전 장치: max_results_int 개수만큼 수집하면 중단
                        if count >= max_results_int:
                            print(f"   Reached {max_results_int} papers, stopping...")
                            break
                        
                        # 진행 상황 표시 (50개마다)
                        if count % 50 == 0:
                            elapsed = time.time() - start_time
                            print(f"   Processed {count} papers, collected {len(papers)} (filtered out: {filtered_out}) - {elapsed:.1f}s")
                        
                        # 날짜 필터링
                        try:
                            published_date = result.published.date()
                            if start_date <= published_date <= end_date:
                                # 기업명 추출 (제목 + 초록)
                                text_to_analyze = f"{result.title} {result.summary}"
                                companies = self._extract_companies(text_to_analyze)
                                
                                # 주요 키워드 추출 (제목 기반)
                                extracted_keywords = self._extract_keywords(result.title, result.summary)
                                
                                paper = {
                                    "title": result.title,
                                    "authors": [author.name for author in result.authors],
                                    "abstract": result.summary,
                                    "url": result.entry_id,
                                    "published": published_date.strftime("%Y-%m-%d"),
                                    "categories": result.categories,
                                    "companies": companies,  # 언급된 기업
                                    "keywords": extracted_keywords  # 추출된 키워드
                                }
                                papers.append(paper)
                            else:
                                filtered_out += 1
                                if filtered_out == 1 or filtered_out % 20 == 0:
                                    print(f"   Filtered out {filtered_out} papers (outside date range)")
                        except Exception as paper_error:
                            print(f"   Skipping paper: {paper_error}")
                            continue
                
                except Exception as fetch_error:
                    # 페칭 중 에러 발생 - 지금까지 수집한 papers는 유지
                    print(f"   Fetch interrupted at {count} papers: {fetch_error}")
                    print(f"   Saving {len(papers)} papers collected so far...")
                    print(f"   Stats: processed={count}, collected={len(papers)}, filtered_out={filtered_out}")
                
                # 수집한 papers 업데이트
                if len(papers) > len(best_papers):
                    best_papers = papers
                
                elapsed_total = time.time() - start_time
                print(f"   Collected {len(papers)} papers (from {count} processed, filtered out: {filtered_out}, {elapsed_total:.1f}s)")
                
                # papers가 있으면 성공으로 간주
                if len(papers) > 0:
                    # 기업 통계 생성
                    company_stats = self._generate_company_stats(papers)
                    
                    result = {
                        "total_count": len(papers),
                        "papers": papers,
                        "company_stats": company_stats  # 기업별 논문 수
                    }
                    
                    # 부분 결과인 경우 경고 추가
                    if count < max_results_int:
                        result["warning"] = f"Partial results: collected {count}/{max_results_int} papers"
                    
                    # 기업 통계 출력
                    if company_stats:
                        top_companies = sorted(company_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                        print(f"   Top companies: {', '.join([f'{c}({n})' for c, n in top_companies])}")
                    
                    print(f"ArXiv search successful: {len(papers)} papers")
                    return result
                
                # papers가 0개면 재시도
                raise ValueError(f"No papers collected (processed {count})")
            
            except Exception as e:
                print(f"ArXiv search failed (attempt {attempt + 1}/{self.config.retry_count}): {e}")
                
                if attempt < self.config.retry_count - 1:
                    wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s...
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 최종 실패 - best_papers라도 반환
                    if len(best_papers) > 0:
                        print(f"Returning {len(best_papers)} papers from previous attempts")
                        return {
                            "total_count": len(best_papers),
                            "papers": best_papers,
                            "warning": "Partial results from retry attempts"
                        }
                    else:
                        print(f"No papers collected after {self.config.retry_count} attempts")
                        return {
                            "total_count": 0,
                            "papers": [],
                            "error": f"Failed after {self.config.retry_count} attempts: {str(e)}"
                        }
    
    def _build_query(self, keywords: List[str], categories: str) -> str:
        """
        arXiv 쿼리 문자열 생성
        
        Args:
            keywords: 검색 키워드 리스트 (최대 10개 권장)
            categories: 카테고리 (all 또는 "cs.RO,cs.AI")
        
        Returns:
            arXiv API 쿼리 문자열
        
        Note:
            키워드가 너무 많으면 쿼리가 복잡해져 API 에러 발생 가능
            권장: 10개 이하 (caller에서 제한해야 함)
        """
        # 키워드가 너무 많으면 경고
        if len(keywords) > 10:
            print(f"   Warning: {len(keywords)} keywords provided (recommended: ≤10)")
            print(f"   Complex queries may cause ArXiv API errors")
       
        cleaned_keywords = [kw.replace("_", " ") for kw in keywords]
        
        # 키워드를 OR로 연결
        keyword_query = " OR ".join([f'all:"{kw}"' for kw in cleaned_keywords])
        
        # 카테고리 필터
        if categories != "all":
            cat_list = [cat.strip() for cat in categories.split(",")]
            cat_query = " OR ".join([f'cat:{cat}' for cat in cat_list])
            return f"({keyword_query}) AND ({cat_query})"
        
        return keyword_query
    
    def _parse_date_range(self, date_range: str) -> tuple:
        """
        날짜 범위 파싱
        
        Args:
            date_range: "2022-01-01 to 2025-10-22" 형식
        
        Returns:
            (start_date, end_date) tuple
        
        Raises:
            ValueError: 날짜 형식이 잘못된 경우
        """
        try:
            parts = date_range.lower().split(" to ")
            if len(parts) != 2:
                raise ValueError(f"Invalid date_range format: {date_range}")
            
            start_date = datetime.strptime(parts[0].strip(), "%Y-%m-%d").date()
            end_date = datetime.strptime(parts[1].strip(), "%Y-%m-%d").date()
            
            return start_date, end_date
        
        except ValueError as e:
            raise ValueError(
                f"date_range must be in format 'YYYY-MM-DD to YYYY-MM-DD': {e}"
            )
    
    def search_by_keywords_parallel(
        self,
        keywords: List[str],
        categories: str = "all",
        max_results_per_keyword: int = 100,
        years_back: int = 3
    ) -> Dict[str, Any]:
        """
        각 키워드별로 개별 검색을 병렬로 수행 (최신 논문 위주)
        
        Args:
            keywords: 검색 키워드 리스트
            categories: arXiv 카테고리
            max_results_per_keyword: 키워드당 최대 결과 수
            years_back: 몇 년 전부터 검색할지 (기본: 3년)
        
        Returns:
            중복 제거된 논문 결과
        """
        print(f"\nStarting parallel search for {len(keywords)} keywords")
        print(f"   Date range: Last {years_back} years")
        print(f"   Max per keyword: {max_results_per_keyword}")
        
        # 날짜 범위 계산 (최근 N년)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        # ThreadPoolExecutor를 사용하여 병렬 검색
        all_papers = []
        seen_ids = set()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 각 키워드별로 검색 작업 제출
            futures = []
            for i, keyword in enumerate(keywords):
                print(f"   Submitting search {i+1}/{len(keywords)}: '{keyword}'")
                future = executor.submit(
                    self._search_single_keyword,
                    keyword=keyword,
                    date_range=date_range,
                    categories=categories,
                    max_results=max_results_per_keyword,
                    keyword_index=i+1,
                    total_keywords=len(keywords)
                )
                futures.append(future)
            
            # 결과 수집 및 중복 제거
            for future in futures:
                try:
                    result = future.result()
                    if result and result.get("papers"):
                        for paper in result["papers"]:
                            paper_id = paper["url"].split("/")[-1]  # arXiv ID 추출
                            if paper_id not in seen_ids:
                                seen_ids.add(paper_id)
                                all_papers.append(paper)
                except Exception as e:
                    print(f"   Keyword search failed: {e}")
        
        # 발행일 기준으로 정렬 (최신순)
        all_papers.sort(key=lambda p: p["published"], reverse=True)
        
        # 기업 통계 생성
        company_stats = self._generate_company_stats(all_papers)
        
        print(f"\nParallel search complete!")
        print(f"   Total unique papers: {len(all_papers)}")
        print(f"   Duplicates removed: {sum(len(f.result().get('papers', [])) for f in futures if f.done()) - len(all_papers)}")
        
        # 기업 통계 출력
        if company_stats:
            top_companies = sorted(company_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top companies: {', '.join([f'{c}({n})' for c, n in top_companies])}")
        
        # Citations 생성
        citations = []
        for paper in all_papers:
            try:
                url = paper.get("url", "")
                arxiv_id = url.split("/")[-1] if url else "unknown"
                
                citation = ArXivCitation(
                    authors=paper.get("authors", []),
                    title=paper.get("title", "Unknown"),
                    arxiv_id=arxiv_id,
                    published=paper.get("published", "Unknown"),
                    url=url
                )
                citations.append(citation)
            except Exception as e:
                print(f"   Citation 생성 실패: {str(e)[:100]}")
                continue
        
        print(f"   Citations created: {len(citations)}")
        
        return {
            "total_count": len(all_papers),
            "papers": all_papers,
            "citations": citations,  # Citation 객체 리스트 추가
            "company_stats": company_stats,  # 기업별 논문 수
            "search_method": "parallel_by_keyword",
            "keywords_searched": len(keywords),
            "date_range": date_range
        }
    
    def _search_single_keyword(
        self,
        keyword: str,
        date_range: str,
        categories: str,
        max_results: int,
        keyword_index: int,
        total_keywords: int
    ) -> Dict[str, Any]:
        """
        단일 키워드로 검색 (내부 헬퍼 메서드)
        
        Args:
            keyword: 검색 키워드
            date_range: 날짜 범위
            categories: 카테고리
            max_results: 최대 결과 수
            keyword_index: 현재 키워드 인덱스
            total_keywords: 전체 키워드 수
        
        Returns:
            검색 결과
        """
        print(f"\n   [{keyword_index}/{total_keywords}] Searching: '{keyword}'")
        
        try:
            # 기존 _run 메서드 재사용 (단일 키워드)
            result = self._run(
                keywords=[keyword],  # 단일 키워드
                date_range=date_range,
                categories=categories,
                max_results=str(max_results)
            )
            
            papers_count = len(result.get("papers", []))
            print(f"   [{keyword_index}/{total_keywords}] Found {papers_count} papers for '{keyword}'")
            
            return result
        
        except Exception as e:
            print(f"   [{keyword_index}/{total_keywords}] Failed for '{keyword}': {e}")
            return {"papers": [], "total_count": 0}
    
    def _extract_companies(self, text: str) -> List[str]:
        """
        논문 텍스트에서 기업명 추출
        
        Args:
            text: 분석할 텍스트 (제목 + 초록)
        
        Returns:
            발견된 기업명 리스트
        """
        found_companies = []
        text_lower = text.lower()
        
        for company in self.COMPANIES:
            # 대소문자 무시하고 매칭
            if company.lower() in text_lower:
                # 중복 제거를 위해 정규화된 이름 사용
                if company not in found_companies:
                    found_companies.append(company)
        
        return found_companies
    
    def _extract_keywords(self, title: str, abstract: str) -> List[str]:
        """
        논문 제목과 초록에서 주요 키워드 추출
        
        간단한 규칙 기반 추출:
        - 대문자로 시작하는 연속된 단어 (기술 용어)
        - 하이픈으로 연결된 단어
        - 특정 패턴의 기술 용어
        
        Args:
            title: 논문 제목
            abstract: 논문 초록
        
        Returns:
            추출된 키워드 리스트 (최대 10개)
        """
        keywords = []
        
        # 주요 로봇/AI 기술 키워드 패턴
        tech_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',  # "Reinforcement Learning", "Deep Neural Network"
            r'\b\w+(?:-\w+)+\b',  # "Multi-Agent", "End-to-End"
            r'\b(?:AI|ML|RL|DRL|CNN|RNN|LSTM|GAN|VAE|NLP|CV)\b',  # 약어
        ]
        
        # 제목과 초록의 일부만 분석 (성능 고려)
        text_to_analyze = f"{title} {abstract[:500]}"
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_to_analyze)
            keywords.extend(matches)
        
        # 중복 제거 및 정제
        keywords = list(set(keywords))
        
        # 불용어 제거 (너무 일반적인 단어)
        stopwords = {"The", "This", "These", "Our", "We", "In", "On", "For", "To", "From"}
        keywords = [kw for kw in keywords if kw not in stopwords]
        
        # 최대 10개만 반환 (빈도순으로 정렬하면 더 좋지만, 간단하게)
        return keywords[:10]
    
    def _generate_company_stats(self, papers: List[Dict]) -> Dict[str, int]:
        """
        논문 리스트에서 기업별 통계 생성
        
        Args:
            papers: 논문 리스트 (각 논문에 "companies" 필드 포함)
        
        Returns:
            {기업명: 논문 수} 딕셔너리
        """
        company_counts = {}
        
        for paper in papers:
            companies = paper.get("companies", [])
            for company in companies:
                company_counts[company] = company_counts.get(company, 0) + 1
        
        return company_counts
    
    async def _arun(self, *args, **kwargs):
        """비동기 실행 (LangChain 호환)"""
        raise NotImplementedError("ArxivTool does not support async execution")




