# Robotics Trends Prediction Pipeline

로보틱스 트렌드 분석 보고서를 자동 생성하는 AI 프로그램입니다.

## 기술

### 핵심 프레임워크
- **Python 3.11+**
- **LangChain / LangGraph**
- **OpenAI GPT-4o / GPT-4o-mini**

### 데이터 & 임베딩
- **ArXiv API** - 학술 논문 검색
- **GNews API** - 뉴스 기사 크롤링
- **ChromaDB** - 벡터 데이터베이스 (RAG)
- **HuggingFace Embeddings** - 문서 임베딩

### 검증 & 출력
- **Ragas** - LLM 품질 평가 (Faithfulness, Relevancy)
- **Pydantic** - 스키마 검증 및 타입 안정성
- **python-docx / docx2pdf** - 문서 생성

## 아키텍처

### 워크플로우

```
사용자 입력
    ↓
┌───────────────┐
│ Planning      │  연구 계획 생성 및 사용자 승인
│ Agent         │
└───────────────┘
    ↓
┌───────────────┐
│ Data          │←─┐  학술/참조/뉴스 데이터 수집 (ReAct)
│ Collection    │  │
└───────────────┘  │
    ↓              │
Content Analysis   │  트렌드 분석 및 섹션 생성
    ↓              │
Report Synthesis   │  요약 및 결론 작성
    ↓              │
┌───────────────┐  │
│ Writer        │ ─┘  보고서 조립 및 사용자 리뷰 (ReAct)
│ Agent         │     - Revision: 보고서 수정
└───────────────┘     - Recollection: 데이터 재수집
    ↓
End Node             한국어 번역 + DOCX/PDF 생성
    ↓
Evaluation           Ragas 품질 평가
```

**피드백 루프**
- **Writer Agent (Revision)**: 보고서 수정 → Writer 재실행
- **Writer Agent (Recollection)**: 데이터 부족 → Data Collection 재실행

## 에이전트 및 LLM

### 1. Planning Agent
- 사용자 주제를 분석하고 연구 계획 생성
- 30-40개의 관련 키워드 생성
- 계획 개선을 위한 사용자 피드백 수집
- Human In the Loop 방식으로 유저로부터 승인 절차

### 2. Data Collection Agent
- ArXiv에서 학술 논문 검색
- 참조 문서(FTSG, WEF 보고서)를 위한 RAG 시스템 쿼리
- 기술 뉴스 기사 크롤링
- 재시도 로직을 통한 품질 검사 수행
- 자율적 도구 선택을 위한 ReAct 아키텍처

### 3. Content Analysis LLM
- 수집된 데이터에서 새로운 트렌드 분석
- 10개의 구조화된 보고서 섹션 생성
- 인용 추출 및 포맷팅

### 4. Report Synthesis LLM
- 경영진 요약 생성
- 서론, 본론, 결론 등 섹션마다 내용물 작성
- 참고문헌 및 부록 컴파일

### 5. Writer Agent
- 모든 섹션을 최종 보고서로 조립
- 유저로부터 품질 검사 수행
- **Feedback Classifier Tool**: 사용자 피드백을 "revision", "recollection", "approved"로 분류
- **Revision Tool**: 보고서 내용 수정 (Writer 내에서 처리)
- **Recollection Tool**: 데이터 재수집 필요 시 Data Collection Agent로 라우팅
- 반복적 개선 사이클 관리 (최대 5회 재시도)

### 6. Evaluation LLM
- Ragas 메트릭을 사용한 보고서 품질 평가
- 소스 데이터에 대한 충실성 측정
- 사용자 쿼리에 대한 답변 관련성 평가
- 품질 보증으로 워크플로우 이후 실행

## 프로젝트 구조

```
robotics-trends-prediction/
├── src/
│   ├── agents/                              # 에이전트 구현
│   │   ├── base/
│   │   │   ├── base_agent.py                # 에이전트 베이스 클래스
│   │   │   └── agent_config.py              # 에이전트 설정
│   │   ├── planning_agent.py                # 연구 계획 생성 (HITL)
│   │   ├── data_collection_agent.py         # 데이터 수집 (ReAct)
│   │   ├── writer_agent.py                  # 보고서 작성 및 리뷰 (ReAct)
│   │   └── evaluation_agent.py              # Ragas 품질 평가
│   ├── llms/                                # LLM 기반 모듈 (도구 없음)
│   │   ├── content_analysis_llm.py          # 트렌드 분석 및 섹션 생성
│   │   ├── report_synthesis_llm.py          # 요약 및 결론 작성
│   │   └── revision_llm.py                  # (사용 안 함)
│   ├── tools/                               # LangChain 도구
│   │   ├── base/
│   │   │   ├── base_tool.py                 # 도구 베이스 클래스
│   │   │   └── tool_config.py               # 도구 설정
│   │   ├── arxiv_tool.py                    # ArXiv 논문 검색
│   │   ├── rag_tool.py                      # RAG 참조 문서 검색
│   │   ├── news_crawler_tool.py             # GNews 뉴스 수집
│   │   ├── revision_tool.py                 # 보고서 수정
│   │   └── recollection_tool.py             # 데이터 재수집 트리거
│   ├── utils/                               # 유틸리티 및 래퍼
│   │   ├── planning_util.py                 # 연구 계획 생성 유틸
│   │   ├── refine_plan_util.py              # 계획 수정 및 승인 감지
│   │   ├── data_collect_util.py             # RAG/News 도구 래퍼
│   │   ├── feedback_classifier_util.py      # 피드백 분류 (Revision/Recollection)
│   │   ├── rag_utils.py                     # RAG 유틸리티
│   │   ├── file_utils.py                    # 파일 I/O
│   │   ├── logger.py                        # 로깅
│   │   └── error_handler.py                 # 에러 처리
│   ├── graph/                               # LangGraph 워크플로우
│   │   ├── workflow.py                      # 워크플로우 빌더 및 매니저
│   │   ├── nodes.py                         # 노드 함수 구현
│   │   ├── edges.py                         # 조건부 라우팅 로직
│   │   └── state.py                         # 공유 상태 정의 (PipelineState)
│   ├── core/                                # 핵심 설정 및 모델
│   │   ├── models/                          # Pydantic 모델
│   │   │   ├── planning_model.py            # 계획 스키마
│   │   │   ├── data_collection_model.py     # 데이터 수집 스키마
│   │   │   ├── citation_model.py            # 인용 스키마
│   │   │   ├── quality_check_model.py       # 품질 검사 스키마
│   │   │   ├── revision_model.py            # 수정 결정 스키마
│   │   │   └── trend_model.py               # 트렌드 분류 스키마
│   │   ├── patterns/                        # 디자인 패턴
│   │   │   ├── base_model.py                # 베이스 모델
│   │   │   └── singleton.py                 # 싱글톤 패턴
│   │   └── settings.py                      # 전역 설정 (Singleton)
│   ├── rag/                                 # RAG 파이프라인
│   │   ├── pipeline.py                      # RAG 전체 파이프라인
│   │   ├── loader.py                        # 문서 로더 (PDF)
│   │   ├── chunker.py                       # 문서 청킹
│   │   ├── embedder.py                      # 임베딩 생성
│   │   └── indexer.py                       # ChromaDB 인덱싱
│   ├── document/                            # 문서 생성
│   │   ├── docx_generator.py                # Markdown → DOCX
│   │   └── pdf_converter.py                 # DOCX → PDF
│   └── cli/                                 # CLI 인터페이스
│       └── human_review.py                  # 사용자 피드백 입력
├── scripts/                                 # 실행 스크립트
│   ├── run_pipeline.py                      # 메인 파이프라인 실행
│   └── indexer_builder.py                   # RAG 인덱스 빌드
├── config/                                  # 설정 파일
│   ├── app_config.yaml                      # 앱 설정
│   └── prompts/                             # 프롬프트 템플릿
│       ├── analysis_prompts.py              # 분석 프롬프트
│       ├── synthesis_prompts.py             # 합성 프롬프트
│       └── data_collections_prompts.py      # 데이터 수집 프롬프트
├── data/                                    # 데이터 디렉토리
│   ├── raw/                                 # 원본 수집 데이터
│   ├── processed/                           # 처리된 데이터
│   ├── reports/                             # 생성된 보고서 (MD/DOCX/PDF)
│   ├── chroma_db/                           # ChromaDB 벡터 스토어
│   └── logs/                                # 파이프라인 로그
├── reference_docs/                          # RAG용 참조 문서
│   ├── FTSG.pdf                             # 참조 문서 1
│   └── WEF.pdf                              # 참조 문서 2
├── tests/                                   # 테스트
│   ├── unit/                                # 단위 테스트
│   └── integration/                         # 통합 테스트
├── requirements.txt                         # Python 의존성
└── README.md                                # 프로젝트 문서
```

## 사용법

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd robotics-trends-prediction

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 설정

1. 템플릿에서 `.env` 파일 생성:
```bash
cp env.example .env
```

2. OpenAI API 키 추가:
```
OPENAI_API_KEY=your_api_key_here
```

3. (선택사항) 참조 문서 사용 시 RAG 인덱스 구축:
```bash
python scripts/indexer_builder.py
```

### 파이프라인 실행

#### 대화형 모드
```bash
python scripts/run_pipeline.py
```

#### 커맨드라인 모드
```bash
python scripts/run_pipeline.py --topic "제조업의 휴머노이드 로봇"
```

### 파이프라인 흐름

1. **주제 입력**: 연구 주제 제공
2. **계획**: 생성된 계획 검토 및 승인/수정
   - 수정 요청 시: Refine Plan Tool이 피드백을 반영하여 계획 재생성
3. **데이터 수집**: 여러 소스에서 자동 수집
4. **분석**: AI가 데이터를 분석하고 구조화된 콘텐츠 생성
5. **합성**: 요약 및 결론 생성
6. **리뷰**: 최종 보고서 검토 및 필요시 수정 요청
   - **승인**: End Node로 진행
   - **수정 요청**: Revision Tool이 보고서 수정 후 Writer Agent로 재시도
   - **데이터 재수집 요청**: Data Collection Agent로 돌아가서 추가 데이터 수집
7. **출력**: 다중 포맷(MD, DOCX, PDF)으로 보고서 수신
8. **평가**: Ragas 메트릭으로 품질 검사

### 출력 파일

보고서는 `data/reports/{topic}_{timestamp}/`에 저장됩니다:
- `final_report.md` - 영문 마크다운 보고서
- `final_report_korean.docx` - 한글 DOCX 보고서
- `final_report_korean.pdf` - 한글 PDF 보고서

평가 메트릭은 콘솔에 표시되고 파이프라인 상태에 저장됩니다.

### 사용자 리뷰 포인트

1. **계획 수립 후**: 키워드, 날짜 범위, 데이터 소스 검토
   - 승인: "ok", "approve", "좋아요"
   - 수정: "키워드 수를 줄여줘", "날짜 범위를 2020-2025로 변경"

2. **작성 후**: 최종 보고서 검토 및 개선을 위한 피드백 제공
   - 승인: "ok", "approve", "좋아요"
   - 수정: "결론 부분을 더 구체적으로 작성해줘"
   - 데이터 재수집: "자율주행 로봇에 대한 데이터를 더 수집해줘"







