# 환경변수 설정 가이드

## 필수 환경변수

프로젝트를 실행하기 위해 다음 환경변수를 설정해야 합니다:

### 🔑 필수 설정 (Required)

```bash
# OpenAI API 키 (반드시 설정 필요)
OPENAI_API_KEY=your_openai_api_key_here
```

### ⚙️ 선택적 설정 (Optional - 기본값 있음)

```bash
# OpenAI 설정
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7

# Embedding 모델 설정
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# ChromaDB 설정
CHROMADB_PATH=data/chromadb
CHROMADB_COLLECTION=reference_reports

# 데이터 수집 설정
ARXIV_START_DATE=2022-01-01
TRENDS_TIMEFRAME=today 36-m
NEWS_SOURCES_COUNT=5

# 품질 검사 설정
MAX_RETRY_COUNT=3
MIN_ARXIV_PAPERS=30
MIN_COMPANY_RATIO=0.20

# RAG 설정
RAG_MAX_CALLS=10
RAG_TOP_K=5
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200

# 병렬 처리 설정
MAX_WORKERS=3

# 로깅 설정
LOG_LEVEL=INFO
```

## 설정 방법

### 1. .env 파일 생성

프로젝트 루트에 `.env` 파일을 생성하고 위의 환경변수들을 복사하여 붙여넣으세요.

### 2. 필수 값 설정

`OPENAI_API_KEY`에 실제 OpenAI API 키를 입력하세요:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. 선택적 값 조정

필요에 따라 다른 환경변수들의 값을 조정할 수 있습니다. 기본값으로도 충분히 작동합니다.

## 환경변수 설명

- **OPENAI_API_KEY**: OpenAI API 키 (필수)
- **OPENAI_MODEL**: 사용할 OpenAI 모델 (기본값: gpt-4o-mini)
- **OPENAI_TEMPERATURE**: 모델의 창의성 수준 (0.0-1.0, 기본값: 0.7)
- **EMBEDDING_MODEL**: 임베딩 모델 (기본값: nomic-ai/nomic-embed-text-v1.5)
- **CHROMADB_PATH**: ChromaDB 저장 경로
- **CHROMADB_COLLECTION**: ChromaDB 컬렉션명
- **ARXIV_START_DATE**: arXiv 논문 검색 시작 날짜
- **TRENDS_TIMEFRAME**: Google Trends 검색 기간
- **NEWS_SOURCES_COUNT**: 크롤링할 뉴스 소스 개수
- **MAX_RETRY_COUNT**: 최대 재시도 횟수
- **MIN_ARXIV_PAPERS**: 최소 arXiv 논문 수 (품질 검사 기준)
- **MIN_COMPANY_RATIO**: 최소 기업 참여율 (품질 검사 기준)
- **RAG_***: RAG 시스템 관련 설정들
- **MAX_WORKERS**: 병렬 처리 워커 수
- **LOG_LEVEL**: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
