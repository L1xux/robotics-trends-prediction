"""
Content Analysis Agent 프롬프트

Content Analysis Agent가 수집된 데이터를 분석하고 보고서 섹션을 생성하는 프롬프트
"""

# System Prompt (공통)
ANALYSIS_SYSTEM_PROMPT = """You are an expert AI/Robotics research analyst specializing in **5-year trend forecasting** and technical report writing.

**Mission: Predict 5-Year Future Trends (2025-2030)**

You must analyze current research (arXiv papers), expert reports (RAG), and market signals (news) to forecast technologies that will be mainstream in 5 years.

**Trend Classification (2-Tier System):**
1. **HOT_TRENDS (1-2년 내 상용화)**
   - Paper count: 100+ papers
   - Company participation: 40%+ of major companies
   - Market readiness: High investment, active deployment
   - Timeline: Commercialization by 2027

2. **RISING_STARS (3-5년 핵심 기술)**
   - Paper count: 30-100 papers
   - Company participation: 20-40% of companies
   - Market readiness: Active R&D, pilot projects
   - Timeline: Breakthrough by 2030

**CRITICAL OUTPUT REQUIREMENT:**
- Output ONLY raw JSON
- NO markdown code blocks (```json```)
- Start directly with {{ and end with }}
- Ensure valid JSON format
"""

# Section 2: Technology Trend Analysis
SECTION_2_PROMPT = """**Topic:** {topic}
**Keywords:** {keywords}

**Data Sources:**

**ArXiv Papers:**
{arxiv_summary}

**Expert Reports (RAG):**
{rag_summary}

**Your Task:**
Generate Section 2: AI-Robotics Technology Trend Analysis

**Section 2.1: 최신 기술 트렌드 분석**
- Analyze latest research trends from arXiv papers
- Identify emerging keywords that signal future breakthroughs
- Discuss which research themes have **5-year commercialization potential**
- Connect current research trends with expert forecasts
- Reference specific papers with citations

**Section 2.2: 2-Tier 기술 분류**
- Classify 3-5 key technologies into HOT_TRENDS or RISING_STARS
- For each technology:
  * Count relevant papers
  * Estimate company participation ratio (based on papers and news)
  * Provide clear reasoning for classification
- Explain why each technology will be important in 5 years

**Output Format:**
{{
    "trends": [
        {{
            "name": "Technology Name",
            "tier": "HOT_TRENDS" or "RISING_STARS",
            "paper_count": int,
            "company_ratio": float (0.0 to 1.0, or 0 to 100 will be auto-converted),
            "reasoning": "Why this technology will be mainstream in 1-2 or 3-5 years..."
        }},
        ...
    ],
    "sections": {{
        "section_2_1": "Detailed analysis with citations [1], [2]...",
        "section_2_2": "Technology classification and forecast with citations [3], [4]..."
    }},
    "citations": [
        {{
            "number": 1,
            "source_type": "arxiv" or "report",
            "title": "Paper or report title",
            "authors": ["Author 1", "Author 2"],
            "url": "https://...",
            "date": "YYYY-MM-DD"
        }},
        ...
    ]
}}
"""

# Section 3: Market Trends & Applications
SECTION_3_PROMPT = """**Topic:** {topic}
**Keywords:** {keywords}

**News Data:**
{news_summary}

**Expert Reports (RAG):**
{rag_summary}

**Your Task:**
Generate Section 3: Market Trends & Applications

**Section 3.1: 시장 동향 분석**
- Analyze market trends from news articles
- Identify growing market segments
- Discuss market size and growth predictions

**Section 3.2: 산업별 적용 사례**
- Highlight successful implementations and use cases
- Discuss specific applications by industry

**Section 3.3: 주요 기업 동향**
- Identify key companies and their activities (from news)
- Discuss major announcements, product launches, partnerships
- Analyze technology development directions

**Output Format:**
{{
    "sections": {{
        "section_3_1": "Market trend analysis with citations [X], [Y]...",
        "section_3_2": "Industry applications with citations [Z]...",
        "section_3_3": "Company activities with citations [W]..."
    }},
    "citations": [
        {{
            "number": {citation_start_number},
            "source_type": "news",
            "title": "News article title",
            "url": "https://...",
            "date": "YYYY-MM-DD",
            "publisher": "Publisher name"
        }},
        ...
    ]
}}
"""

# Section 4: 5-Year Forecast
SECTION_4_PROMPT = """**Topic:** {topic}

**Section 2 (Technology Trends):**
{section_2}

**Section 3 (Market Trends):**
{section_3}

**Key Trends (2-Tier Classification):**
{trends_summary}

**Expert Forecasts (RAG):**
{rag_summary}

**Your Task:**
Generate Section 4: 5-Year Forecast (2025-2030)

**This is the CORE forecasting section!** Be specific and bold in predictions.

**Section 4.1: 단기 전망 (2025-2027): 상용화 임박 기술**
- Focus on HOT_TRENDS technologies
- Predict specific commercialization timelines
- Discuss market readiness and deployment scenarios
- Cite expert forecasts from RAG

**Section 4.2: 중장기 전망 (2028-2030): 혁신 기술 전망**
- Focus on RISING_STARS technologies
- Predict breakthrough moments and game-changers
- Synthesize all data sources:
  * Research momentum (arXiv): Which technologies have exponential growth?
  * Expert forecasts (RAG): What do FTSG/WEF predict for 2030?
  * Market signals (News): Where is investment flowing?
- Identify which RISING_STARS will become HOT_TRENDS by 2030
- Predict specific applications and market size by 2030
- Cite expert reports heavily [X], [Y]...

**Output Format:**
{{
    "sections": {{
        "section_4_1": "Short-term forecast with citations...",
        "section_4_2": "Long-term forecast with citations..."
    }},
    "citations": [
        {{
            "number": {citation_start_number},
            "source_type": "report",
            "title": "Expert report title",
            "author": "FTSG or WEF",
            "year": 2023
        }},
        ...
    ]
}}
"""

# Section 5: Implications for Business
SECTION_5_PROMPT = """**Topic:** {topic}

**Section 2 (Technology Trends):**
{section_2}

**Section 3 (Market Trends):**
{section_3}

**Section 4 (5-Year Forecast):**
{section_4}

**Key Trends:**
{trends_summary}

**Your Task:**
Generate Section 5: Implications for Business

**Section 5.1: 기술 변화가 산업에 미치는 영향**
- Analyze how forecasted technologies will transform industries
- Discuss disruption and opportunities

**Section 5.2: 기업의 대응 전략**
- Provide industry-specific guidance based on Section 3
- Discuss implementation challenges and solutions

**Section 5.3: 기술 변화에 따른 대응 방향**
- Suggest response strategies based on Section 4 forecasts
- Provide investment and development recommendations

**Output Format:**
{{
    "sections": {{
        "section_5_1": "Industry impact analysis with citations...",
        "section_5_2": "Corporate strategy recommendations with citations...",
        "section_5_3": "Future response directions with citations..."
    }},
    "citations": [
        {{
            "number": {citation_start_number},
            "source_type": "report",
            "title": "...",
            "author": "...",
            "year": int
        }},
        ...
    ]
}}
"""

# 프롬프트 딕셔너리
ANALYSIS_PROMPTS = {
    "system": ANALYSIS_SYSTEM_PROMPT,
    "section_2": SECTION_2_PROMPT,
    "section_3": SECTION_3_PROMPT,
    "section_4": SECTION_4_PROMPT,
    "section_5": SECTION_5_PROMPT
}

