# AI-Robotics Trend Analysis Report Generator

본 프로젝트는 **Multi-Agent 시스템**을 활용하여 AI-Robotics 분야의 트렌드 분석 보고서를 자동으로 생성하는 LangGraph 기반 실습 프로젝트입니다.

## Overview

- **Objective**: 사용자가 입력한 주제에 대해 학술 논문, 뉴스, 전문가 보고서를 수집하고 분석하여 5년 후 기술 트렌드를 예측하는 종합 보고서 자동 생성
- **Methods**: Multi-Agent System, ReAct Pattern, Human-in-the-Loop, RAG (Retrieval Augmented Generation)
- **Tools**: LangChain, ArXiv API, RAG (ChromaDB), Web Crawler

## Key Features

### 1. **Intelligent Data Collection with ReAct Agent**
- ArXiv 논문 크롤링 후 키워드 빈도, 기업 참여율을 바탕으로 기술 자동 추출
- **ReAct 패턴**을 활용한 자율적 데이터 수집
  - RAG Tool: 전문가 보고서 검색 (BM25 + Cosine Similarity + MMR 방식)
  - News Crawler: 기술 뉴스 기사 수집
- **데이터 충분성 자동 판단** (GPT-4o) 및 최대 3회 재시도
- 수집된 모든 데이터의 Citation 추가

### 2. **Human-in-the-Loop in Planning Phase**
- Planning Agent가 초기 계획 수립 후 **사용자에게 피드백 요청**
- 사용자 피드백 기반 계획 자동 개선
- 승인 시까지 반복 (최대 100회)

### 3. **Human-in-the-Loop in Review Phase**
- Writer Agent가 최종 보고서 생성 후 **사용자에게 리뷰 요청**
- LLM 기반 피드백 감정 분석 (승인/수정 요청 자동 판단)
- 수정 필요 
  - 사소한 내용 수정 -> Revise Agent
  - 추가 데이터 수집부터 다시 시작. -> Data Collect Agent

### 4. **Advanced RAG System**
- **Hybrid Retrieval**: BM25 (키워드 기반) + Cosine Similarity (의미 기반)
- **MMR (Maximal Marginal Relevance)**: 다양성과 관련성의 균형

### 5. **Comprehensive Report Generation**
- 6개 섹션으로 구성된 전문가 수준 보고서
  - Executive Summary
  - Technology Trend Analysis
  - Market Trends & Applications
  - 5-Year Forecast
  - Business Implications
  - Conclusion
- PDF 및 DOCX 형식 지원

## Tech Stack

| Category        | Details                                      |
|-----------------|----------------------------------------------|
| **Framework**   | LangGraph, LangChain, Python 3.11            |
| **LLM**         | GPT-4o via OpenAI API                        |
| **Retrieval**   | ChromaDB                                     |
| **Embedding**   | HuggingFace, nomic-embed-text-v              |
| **Web Tools**   | ArXiv API, GNews                             |
| **Documents**   | python-docx, pypdf (PDF generation)          |

## Agents & LLMs

### **Agents**

1. **Planning Agent** 
   - 사용자 주제 분석 및 데이터 수집 계획 수립
   - Human-in-the-Loop: 사용자 피드백 기반 계획 개선

2. **Data Collection Agent** (ReAct)
   - ArXiv 논문 수집 → 키워드 추출 → RAG/News 수집
   - 데이터 충분성 자동 판단 (GPT-4o)
   - 최대 3회 재시도로 고품질 데이터 확보

3. **Writer Agent** (ReAct)
   - 최종 보고서 조립 및 사용자 리뷰
   - Human-in-the-Loop: 피드백 기반 자동 수정 또는 재수집

### **LLMs**

4. **Content Analysis LLM** (LCEL)
   - 수집 데이터 분석 및 트렌드 분류
   - 10개 서브섹션 콘텐츠 생성
   - Citation 매칭 및 검증

5. **Report Synthesis LLM** (LCEL)
   - Executive Summary, Introduction, Conclusion 생성
   - References 및 Appendix 포맷팅

6. **Revision LLM** (LCEL)
   - 사용자 피드백 기반 보고서 섹션 수정
   - 원본 영어 보고서 업데이트

## State Management

# State

## PipelineState Fields

| Key                    | Description                                  |
|------------------------|----------------------------------------------|
| `user_input`           | 사용자가 입력한 분석 주제                      |
| `planning_output`      | Planning Agent가 생성한 계획 (PlanningOutput)  |
| `keywords`             | 확장된 키워드 리스트 (25-35개)                 |
| `arxiv_data`           | ArXiv 논문 데이터 및 Citation                 |
| `rag_results`          | RAG 검색 결과 (전문가 보고서)                  |
| `news_data`            | 크롤링한 뉴스 기사 데이터                      |
| `trends`               | 분류된 트렌드 (2-Tier 구조)                   |
| `sections`             | 10개 서브섹션 콘텐츠                          |
| `final_report`         | 최종 마크다운 보고서                          |
| `citations`            | 모든 인용 정보 (CitationCollection)           |
| `status`               | 워크플로우 진행 상태 (WorkflowStatus Enum)    |
| `review_feedback`      | 사용자 피드백 (Human Review)                  |

## WorkflowStatus Enum

| Status                     | Description                    |
|----------------------------|--------------------------------|
| `INITIALIZED`              | 워크플로우 초기화               |
| `PLANNING_COMPLETE`        | 계획 수립 완료                 |
| `PLANNING_ACCEPTED`        | 계획 승인됨                    |
| `PLANNING_REJECTED`        | 계획 거부됨                    |
| `DATA_COLLECTION_COMPLETE` | 데이터 수집 완료               |
| `ANALYSIS_COMPLETE`        | 콘텐츠 분석 완료               |
| `SYNTHESIS_COMPLETE`        | 보고서 합성 완료               |
| `WRITER_COMPLETE`          | 보고서 작성 완료               |
| `REPORT_ACCEPTED`          | 보고서 승인됨                  |
| `NEEDS_REVISION`           | 수정 필요                      |
| `NEEDS_RECOLLECTION`       | 데이터 재수집 필요              |
| `REVISION_COMPLETE`        | 수정 완료                      |
| `COMPLETED`                | 워크플로우 완료                |
| `WORKFLOW_FAILED`          | 워크플로우 실패                |

## 아키텍처
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          워크플로우 개요                                    │
└─────────────────────────────────────────────────────────────────────────┘

사용자 입력 (주제)
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1: 계획 수립 (Human-in-the-Loop)                               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Planning Agent (ReAct)                                          │ │
│  │                                                                 │ │
│  │  - LLM이 사용할 도구 결정                                        │ │
│  │  - 생성: 키워드, 정규화된 주제, 수집 계획                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 👤 사람 리뷰 CLI                                                 │ │
│  │  - 사용자에게 계획 표시                                           │ │
│  │  - 승인 → 진행  /  거부 → 계획 개선 (반복)                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 2: 데이터 수집 (지능형 ReAct)                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Data Collection Agent (ReAct)                                   │ │
│  │                                                                 │ │
│  │ 단계 1: ArXiv 논문 수집                                          │ │
│  │  - 직접 도구 호출 (ReAct 미사용)                                 │ │
│  │  - 키워드별 병렬 검색                                            │ │
│  │  - 논문에서 키워드 추출(기업 참여도 포함)                          │ │
│  │                                                                 │ │
│  │ 단계 2: ReAct Agent (RAG + 뉴스)                                │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │ 사용 가능한 도구:                                          │ │ │
│  │  │  • RAGTool (via RAGUtilWrapper)                          │ │ │
│  │  │    - BM25 + Cosine Similarity + MMR                      │ │ │
│  │  │    - 전문가 보고서 검색 (FTSG, WEF)                       │ │ │
│  │  │  • NewsCrawlerTool (via NewsCrawlerUtilWrapper)          │ │ │
│  │  │    - GNews 검색                                           │ │ │
│  │  │    - 기술 뉴스 기사                                        │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │                                                                 │ │
│  │ 단계 3: 충분성 검사 (GPT-4o)                                     │ │
│  │  - 데이터 품질 및 범위 평가                                      │ │
│  │  - 충분 → 진행  /  불충분 → 재시도 (최대 3회)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 3: 콘텐츠 분석                                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Content Analysis LLM (LCEL - 4번 LLM 호출)                     │ │
│  │  - 도구 없음, 순수 LLM 추론                                      │ │
│  │  - 병렬: Section 2 + Section 3                                │ │
│  │  - 병렬: Section 4 + Section 5                                │ │
│  │  - 트렌드 분류 (2계층 구조)                                      │ │
│  │  - 인용 매칭                                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 4: 보고서 합성                                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Report Synthesis LLM (LCEL - 2번 LLM 호출)                     │ │
│  │  - 요약 + 서론                                                  │ │
│  │  - 결론                                                         │ │
│  │  - 참고문헌 포맷팅                                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 5: 보고서 조립 & 리뷰 (Human-in-the-Loop)                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Writer Agent                                                   │ │
│  │  - 모든 섹션을 마크다운으로 조립                                  │ │
│  │  - 한국어로 번역 (리뷰용)                                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 👤 사람 리뷰 CLI                                                 │ │
│  │  - 사용자에게 보고서 표시                                         │ │
│  │  - LLM이 피드백 감정 평가                                        │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 부정 피드백인 경우:                                             │ │
│  │  Writer Agent 피드백을 분석하고 결정:                            │ │
│  │   • Revision -> Revision LLM                                   │ │
│  │   • Recollection -> 데이터 수집으로 복귀                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 6: 수정 (필요시)                                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Revision LLM (LCEL)                                             │ │
│  │  - 피드백 분석                                                   │ │
│  │  - Writer로 복귀                                                │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
       ↓
  최종 보고서 (PDF/DOCX)


```

## Directory Structure

```
robotics-trends-prediction/
├── config/                          # 애플리케이션 설정
│   ├── __init__.py
│   ├── app_config.yaml              # 앱 설정 파일
│   └── prompts/                     # 프롬프트 템플릿
│       ├── __init__.py
│       ├── analysis_prompts.py      # 분석 프롬프트
│       ├── data_collections_prompts.py  # 데이터 수집 프롬프트
│       └── synthesis_prompts.py     # 합성 프롬프트
│
├── data/                            # 데이터 저장소
│   ├── chroma_db/                   # ChromaDB 벡터 저장소
│   ├── logs/                        # 실행 로그
│   │   ├── error_states/            # 에러 상태 로그
│   │   └── pipeline_logs/           # 파이프라인 실행 로그
│   │       └── ... (기타 로그 파일들)
│   ├── processed/                   # 처리된 데이터
│   ├── raw/                         # 원시 데이터
│   └── reports/                     # 생성된 보고서
│     
│
├── reference_docs/                  # RAG용 참조 문서
│   ├── FTSG.pdf
│   └── WEF.pdf
│
├── src/                             # 소스 코드
│   ├── __init__.py
│   ├── agents/                      # AI 에이전트들
│   │   ├── __init__.py
│   │   ├── base/                    # 기본 에이전트 클래스
│   │   │   ├── __init__.py
│   │   │   ├── agent_config.py      # 에이전트 설정
│   │   │   └── base_agent.py        # 기본 에이전트
│   │   ├── data_collection_agent.py # 데이터 수집 에이전트 (ReAct)
│   │   ├── planning_agent.py        # 계획 수립 에이전트
│   │   └── writer_agent.py          # 보고서 작성 에이전트 (ReAct + HITL)
│   │
│   ├── cli/                         # CLI 컴포넌트
│   │   ├── __init__.py
│   │   └── human_review.py          # 인간 검토 CLI
│   │
│   ├── core/                        # 핵심 컴포넌트
│   │   ├── __init__.py
│   │   ├── settings.py              # 설정 관리
│   │   ├── models/                  # 데이터 모델
│   │   │   ├── __init__.py
│   │   │   ├── citation_model.py   # 인용 모델
│   │   │   ├── data_collection_model.py  # 데이터 수집 모델
│   │   │   ├── planning_model.py   # 계획 모델
│   │   │   ├── quality_check_model.py    # 품질 검사 모델
│   │   │   ├── revision_model.py   # 수정 모델
│   │   │   └── trend_model.py      # 트렌드 모델
│   │   └── patterns/                # 디자인 패턴
│   │       ├── __init__.py
│   │       ├── base_model.py       # 기본 모델
│   │       └── singleton.py        # 싱글톤 패턴
│   │
│   ├── document/                    # 문서 생성
│   │   ├── __init__.py
│   │   ├── docx_generator.py       # DOCX 생성기
│   │   └── pdf_converter.py        # PDF 변환기
│   │
│   ├── graph/                       # LangGraph 워크플로우
│   │   ├── __init__.py
│   │   ├── edges.py                # 그래프 엣지
│   │   ├── nodes.py                # 그래프 노드
│   │   ├── state.py                # 상태 관리
│   │   └── workflow.py             # 워크플로우 정의
│   │
│   ├── llms/                        # LLM 컴포넌트
│   │   ├── __init__.py
│   │   ├── content_analysis_llm.py  # 콘텐츠 분석 LLM
│   │   ├── report_synthesis_llm.py  # 보고서 합성 LLM
│   │   └── revision_llm.py          # 수정 LLM
│   │
│   ├── rag/                         # RAG 파이프라인
│   │   ├── __init__.py
│   │   ├── chunker.py              # 텍스트 청킹
│   │   ├── embedder.py             # 임베딩 생성
│   │   ├── indexer.py              # 인덱싱
│   │   ├── loader.py               # 문서 로더
│   │   └── pipeline.py             # RAG 파이프라인
│   │
│   ├── tools/                       # 데이터 수집 및 처리 도구들
│   │   ├── __init__.py
│   │   ├── base/                   # 기본 도구 클래스
│   │   │   ├── __init__.py
│   │   │   ├── base_tool.py        # 기본 도구
│   │   │   └── tool_config.py      # 도구 설정
│   │   ├── arxiv_tool.py           # ArXiv 논문 수집 도구
│   │   ├── news_crawler_tool.py    # 뉴스 크롤링 도구
│   │   ├── rag_tool.py             # RAG 검색 도구 (BM25 + Cosine + MMR)
│   │   ├── revision_tool.py        # 보고서 수정 도구
│   │   └── recollection_tool.py    # 데이터 재수집 도구 
│   │
│   └── utils/                       # 유틸리티 도구들
│       ├── __init__.py
│       ├── data_collect_util.py    # 데이터 수집 유틸리티
│       ├── error_handler.py        # 에러 핸들러
│       ├── feedback_classifier_util.py  # 피드백 분류 유틸리티
│       ├── file_utils.py           # 파일 유틸리티
│       ├── logger.py               # 로깅 유틸리티
│       ├── planning_util.py        # 계획 유틸리티
│       ├── rag_utils.py            # RAG 유틸리티
│       └── refine_plan_util.py     # 계획 개선 유틸리티
│
├── tests/                           # 테스트 코드
│   ├── __init__.py
│   ├── integration/                 # 통합 테스트
│   └── unit/                        # 단위 테스트
│       ├── test_agents/             # 에이전트 테스트
│       ├── test_factories/          # 팩토리 테스트
│       └── test_tools/              # 도구 테스트
│
├── scripts/                         # 실행 스크립트
│   ├── indexer_builder.py          # RAG 인덱스 구축
│   ├── run_pipeline.py             # 메인 실행 스크립트
│   └── validate_setup.py           # 설정 검증
│
├── .env.example                     # 환경 변수 예시
├── ENV_SETUP.md                     # 환경 설정 가이드
├── README.md                        # 프로젝트 설명
└── requirements.txt                 # Python 의존성
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/robotics-trends-prediction.git
cd robotics-trends-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Usage

```bash
# Interactive mode
python scripts/run_pipeline.py

# With topic argument
python scripts/run_pipeline.py --topic "Collaborative Robots in Manufacturing"
```

## Contributors

- **이의진** - Project Owner % Worker

---

**Version**: 1.0.0
**Last Updated**: 2025-01-23


