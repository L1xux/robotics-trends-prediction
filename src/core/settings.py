"""
전역 설정 관리 (Pydantic BaseSettings with Singleton)
환경변수 기반 설정
"""
from pydantic_settings import BaseSettings, SettingsConfigDict  
from pydantic import Field
from pathlib import Path
from typing import Optional
from src.core.patterns.singleton import Singleton


class SettingsMeta(Singleton, type(BaseSettings)):
    """
    Metaclass combining Singleton and BaseSettings
    Ensures Settings is a singleton
    """
    pass


class Settings(BaseSettings, metaclass=SettingsMeta):
    """애플리케이션 전역 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore", 
        case_sensitive=False
    )
    
    # ===== OpenAI =====
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.2, env="OPENAI_TEMPERATURE")
    
    # ===== Embedding Model =====
    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1",  
        env="EMBEDDING_MODEL"
    )
    
    # ===== ChromaDB =====
    chromadb_path: Path = Field(
        default=Path("data/chroma_db"),  # ← 실제 ChromaDB 위치
        env="CHROMADB_PATH"
    )
    chromadb_collection: str = Field(
        default="documents",  # ← collection 이름
        env="CHROMADB_COLLECTION"
    )
    
    # ===== Data Paths =====
    data_raw_path: Path = Field(default=Path("data/raw"))
    data_processed_path: Path = Field(default=Path("data/processed"))
    data_reports_path: Path = Field(default=Path("data/reports"))
    reference_docs_path: Path = Field(default=Path("reference_docs"))
    
    # ===== Logs =====
    logs_path: Path = Field(default=Path("data/logs"))
    pipeline_logs_path: Path = Field(default=Path("data/logs/pipeline_logs"))
    error_states_path: Path = Field(default=Path("data/logs/error_states"))
    
    # ===== Data Collection =====
    arxiv_start_date: str = Field(default="2022-01-01", env="ARXIV_START_DATE")
    google_trends_timeframe: str = Field(default="today 36-m", env="TRENDS_TIMEFRAME")
    news_sources_count: int = Field(default=5, env="NEWS_SOURCES_COUNT")
    
    # ===== Quality Check =====
    max_retry_count: int = Field(default=3, env="MAX_RETRY_COUNT")
    min_arxiv_papers: int = Field(default=30, env="MIN_ARXIV_PAPERS")
    min_company_ratio: float = Field(default=0.20, env="MIN_COMPANY_RATIO")
    
    # ===== RAG =====
    rag_max_calls: int = Field(default=10, env="RAG_MAX_CALLS")
    rag_top_k: int = Field(default=5, env="RAG_TOP_K")
    rag_chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")  # token 기준
    rag_chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    
    # ===== Parallel Processing =====
    max_workers: int = Field(default=3, env="MAX_WORKERS")
    
    # ===== Logging =====
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # ===== 대문자 Alias (하위 호환성) =====
    @property
    def OPENAI_API_KEY(self) -> str:
        """대문자 alias (deprecated, use openai_api_key)"""
        return self.openai_api_key
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        """대문자 alias (deprecated, use embedding_model)"""
        return self.embedding_model
    
    @property
    def CHROMADB_PATH(self) -> Path:
        """대문자 alias (deprecated, use chromadb_path)"""
        return self.chromadb_path
    
    @property
    def CHROMADB_COLLECTION(self) -> str:
        """대문자 alias (deprecated, use chromadb_collection)"""
        return self.chromadb_collection
    
    @property
    def RAW_DATA_DIR(self) -> Path:
        """대문자 alias (deprecated, use data_raw_path)"""
        return self.data_raw_path
    
    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        """대문자 alias (deprecated, use data_processed_path)"""
        return self.data_processed_path
    
    @property
    def REPORTS_DIR(self) -> Path:
        """대문자 alias (deprecated, use data_reports_path)"""
        return self.data_reports_path


def get_settings() -> Settings:
    """
    설정 싱글톤 반환

    Settings는 이제 Singleton metaclass를 사용하므로
    직접 인스턴스화해도 항상 같은 객체 반환
    """
    return Settings()