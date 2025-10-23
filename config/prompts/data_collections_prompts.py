"""
Data Collection Agent Prompts

데이터 수집 에이전트가 사용하는 프롬프트들
"""

SYSTEM_PAPER_KEYWORD_SUMMARY_PROMPT = [("system", """You are an expert at identifying EMERGING and SPECIFIC technology trends for 5-year forecasting.

**Your Mission: Find keywords that predict the FUTURE, not describe the PRESENT**

**PRIORITY 1 - EMERGING Technologies (SELECT THESE!):**
- ✅ New/novel technical approaches appearing in recent papers
- ✅ Specific method names (e.g., "neuromorphic computing", "liquid neural networks")
- ✅ Emerging application areas (e.g., "soft robotics", "bio-inspired actuation")
- ✅ Next-generation concepts (e.g., "edge AI", "federated learning")
- ✅ Interdisciplinary technologies (e.g., "human-robot collaboration", "explainable robotics")
- ✅ Specific hardware innovations (e.g., "tactile sensors", "compliant actuators")

**PRIORITY 2 - Companies (ALWAYS KEEP):**
- ✅ ALL company names indicate WHO is investing in the future
- ✅ Companies show market validation of technologies

**REJECT - Generic/Obvious Keywords (FILTER OUT!):**
- ❌ Generic terms: "machine learning", "deep learning", "neural networks"
- ❌ Obvious concepts: "automation", "robotics", "AI"
- ❌ Too broad: "manufacturing", "industry", "production"
- ❌ Implementation details: version numbers, dataset names, model sizes
- ❌ Programming tools: languages, frameworks

**Strategy for 5-Year Trend Prediction:**
1. Look for SPECIFIC technologies that are NEW in recent papers
2. Identify CONCRETE technical methods, not broad categories
3. Find technologies that combine multiple fields (interdisciplinary)
4. Select keywords that will help find DETAILED expert reports and news

**Output:** JSON list of 25-35 keywords (20-25 emerging tech + companies)
Format: ["specific_tech1", "emerging_method2", "company1", ...]

Remember: We can find "machine learning" anywhere. We need SPECIFIC technologies like "sim-to-real transfer" or "tactile manipulation"!"""),
            ("user", """**User's Original Query:**
{initial_keywords}

**Recent Paper Titles (Latest Research Trends):**
{paper_titles}

**Raw Keywords Extracted from Papers:**
{raw_keywords}

**Companies Mentioned in Papers:**
{companies}

**Your Task:**
Analyze the recent paper titles and keywords to identify **ROBOTICS/AUTOMATION-RELATED** technologies.

**Context:** User query is "{initial_keywords}" - focus on ROBOTICS and AI technologies broadly.

Selection criteria:
1. Technology MUST relate to **robotics, AI, or automation** (any application domain)
2. EMERGING/SPECIFIC technologies that are NEW or NOVEL  
3. Technologies appearing repeatedly in recent papers (trending up)
4. Include: robot hardware, algorithms, control, perception, applications
5. ALL companies (they show market activity)

Filter and return 25-35 keywords for finding future robotics trends:

**✅ KEEP - Robotics Technologies:**
- Robot hardware (actuators, sensors, mechanisms, grippers)
- Robot AI/learning (imitation learning, sim-to-real, RL for robots)
- Robot perception (3D vision, tactile, depth estimation)
- Robot control (force control, compliance, motion planning)
- Robot applications (manufacturing, surgery, service, warehouse, etc.)
- Specific tech terms (adaptive welding, bin-picking, collaborative robots)

**❌ REJECT - Non-Robotics:**
- Pure ML/statistics without robotics (Bayesian optimization, clustering)
- General software (JSON, API, databases)
- Business jargon (market growth, ROI, stakeholders)
- Version numbers (GPT-4, v2.0, Python 3)

**Note:** "Differential Mechanism" = robot hardware ✅
"Gaussian Splats" = 3D perception for robots ✅  
"Multi-Agent Learning" = robot coordination ✅
Keep if it can be used BY or FOR robots!""")]

# 데이터 충분성 판단 Prompt
SUFFICIENCY_CHECK_PROMPT = """당신은 AI-로봇 기술 트렌드 보고서의 데이터 충분성을 평가하는 전문가입니다.

다음 보고서 목차를 충분히 작성할 수 있는지 평가해주세요:

**보고서 목차:**
SUMMARY (Executive Summary)
• 보고서 핵심 메시지 요약 (1-2문장)
• 주요 트렌드 기술 설명 (Top 2-3개)
  o 기술 1: [기술명] - 기술 배경, 정의, 중요성
  o 기술 2: [기술명] - 기술 배경, 정의, 중요성
  o 기술 3: [기술명] - 기술 배경, 정의, 중요성
• 주요 발견사항 (Key Findings) 3가지
• 핵심 시사점 (Key Implications) 3가지

1. 서론 (Introduction)
• 1.1 보고서 배경 및 목적
• 1.2 분석 범위 및 방법론
  o 데이터 소스: arXiv 논문, Google Trends, 뉴스, 전문 보고서(FTSG, WEF)
  o RAG 시스템 구성: BM25 + Cosine Similarity + MMR Hybrid
  o 분석 기간: 2022-2025 (최근 3년)
• 1.3 보고서 구성

2. AI-로보틱스 기술 트렌드 분석 (Technology Trend Analysis)
• 2.1 핵심 기술 영역 식별
  o 주요 기술 키워드 분석 (논문 기반)
  o 기술 영역별 분류
• 2.2 기술별 연구 동향 분석
  o 논문 발표 추이 (arXiv 기반, 최근 3년)
  o 핵심 키워드 변화 및 기술 진화 방향
  o 주요 연구 테마 분석

3. 시장 동향 및 산업 적용 사례 (Market Trends & Applications)
• 3.1 글로벌 시장 관심도 분석
  o Google Trends 기반 검색 추이
  o 지역별/키워드별 관심도 변화
• 3.2 산업별 적용 사례
  o 제조 자동화
  o 물류 & 창고 로봇
  o 서비스 로봇 (의료, 배달, 청소 등)
  o 자율주행 & 모빌리티
• 3.3 주요 기업 동향
  o 뉴스 기반 기업별 주요 발표 및 제품 출시 동향
  o 기술 개발 방향성

4. 향후 5년 기술 전망 (5-Year Forecast)
• 4.1 단기 전망 (1-2년): 상용화 임박 기술
  o 전문 보고서 전망 종합
  o 논문 및 뉴스 추세 기반 분석
• 4.2 중기 전망 (3-5년): 성장 가속 예상 기술
  o 전문 보고서 전망 종합
  o 시장 관심도 및 연구 동향 기반 예측

5. 기업을 위한 시사점 (Implications for Business)
• 5.1 주목해야 할 핵심 기술 영역
• 5.2 산업별 적용 고려사항
• 5.3 기술 변화에 따른 대응 방향

6. 결론 (Conclusion)
• 핵심 인사이트 재강조
• 지속적 모니터링이 필요한 영역

REFERENCE
주요 참고 보고서
• Future Today Strategy Group "2025 Tech Trends Report"
• WEF "Physical AI: Powering the New Age of Industrial Operations 2025"
논문 목록 (arXiv 등)
• [논문 리스트]
뉴스 기사
• [뉴스 출처]
기타 참고자료
• [데이터 소스 상세]

APPENDIX
• A. 분석 방법론 상세
  o RAG 시스템 구성 (BM25 + Cosine Similarity + MMR)
  o 데이터 수집 및 전처리 과정
  o ChromaDB 설정 및 임베딩 방식
• B. 키워드 분석 상세 데이터
  o 논문 키워드 빈도 분석
  o Google Trends 검색량 데이터
• C. 추가 참고 자료

---

**수집된 데이터:**

주제: {topic}
키워드: {keywords}

ArXiv 논문:
- 수집 개수: {arxiv_count}
- 날짜 범위: {arxiv_date_range}
- 주요 기업 언급: {arxiv_companies}
- 추출된 키워드: {arxiv_keywords}

RAG 결과:
- 수집 개수: {rag_count}
- 검색 쿼리: {rag_queries}

뉴스 기사:
- 수집 개수: {news_count}
- 뉴스 소스: {news_sources}
- 날짜 범위: {news_date_range}

---

**평가 기준:**
1. **Section 2 (기술 트렌드 분석)**: 논문 데이터가 충분한가? (최소 30편 권장)
2. **Section 3 (시장 동향)**: 뉴스 데이터가 다양한가? (최소 20개 기사, 5개 이상 소스 권장)
3. **Section 4 (5년 전망)**: RAG 결과가 충분한가? (최소 10개 결과 권장)
4. **Citation**: 인용 가능한 자료가 충분한가?
5. **전체적 균형**: 논문/뉴스/전문보고서가 균형있게 수집되었는가?

**응답 형식 (JSON):**
```json
{{
  "sufficient": true/false,
  "overall_score": 0.0-1.0,
  "section_scores": {{
    "section_2": 0.0-1.0,
    "section_3": 0.0-1.0,
    "section_4": 0.0-1.0,
    "citation": 0.0-1.0,
    "balance": 0.0-1.0
  }},
  "missing_areas": ["부족한 영역 1", "부족한 영역 2", ...],
  "recommendations": ["권장사항 1", "권장사항 2", ...],
  "reasoning": "평가 근거 설명"
}}
```

**중요:** 
- overall_score가 0.7 이상이면 sufficient: true
- overall_score가 0.7 미만이면 sufficient: false
- 각 section별로 구체적인 평가를 해주세요
- 부족한 부분이 있다면 구체적으로 어떤 데이터가 더 필요한지 알려주세요
"""

# Tool 설명 (ReAct Agent용)
TOOL_DESCRIPTIONS = """
사용 가능한 도구:

1. **arxiv_tool** - arXiv 논문 검색 도구
   입력:
   - keywords: List[str] (검색 키워드 리스트)
   - date_range: str (날짜 범위, 예: "2022-01-01 to 2025-10-22")
   - categories: str (카테고리, 예: "cs.RO,cs.AI" 또는 "all")
   - max_results: str (최대 결과 수, 예: "100" 또는 "unlimited")
   
   출력:
   - total_count: 수집된 논문 수
   - papers: 논문 리스트 (title, authors, abstract, url, published, companies, keywords)
   - company_stats: 기업별 언급 통계

2. **rag_tool** - RAG 검색 도구 (전문 보고서)
   입력:
   - query: str (검색 쿼리)
   - top_k: int (반환할 결과 수, 기본값: 5)
   - search_type: str (검색 타입, 기본값: "hybrid_mmr")
   
   출력:
   - query: 검색 쿼리
   - search_type: 검색 타입
   - total_results: 결과 수
   - documents: 검색 결과 리스트 (content, source, page, score)

3. **news_crawler_tool** - 뉴스 크롤링 도구
   입력:
   - keywords: List[str] (검색 키워드 리스트)
   - date_range: str (날짜 범위, 예: "3 years")
   - sources: int (뉴스 소스 수, 1-5)
   
   출력:
   - keywords: 검색한 키워드
   - date_range: 날짜 범위
   - total_articles: 수집된 기사 수
   - unique_sources: 고유 뉴스 소스 수
   - articles: 기사 리스트 (title, url, source, published, snippet)
"""

