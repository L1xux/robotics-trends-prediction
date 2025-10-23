"""RAG 파이프라인 통합"""
from typing import List, Optional
from pathlib import Path
from .loader import PDFLoader
from .chunker import SemanticChunker
from .embedder import Embedder
from .indexer import ChromaDBIndexer
from src.utils.logger import default_logger as logger
from src.utils.rag_utils import get_pdf_files, validate_chunks


class RAGPipeline:
    """전체 RAG 파이프라인을 관리하는 클래스"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1",
        persist_directory: str = "../../data/chroma_db",
        collection_name: str = "documents",
        trust_remote_code: bool = False,  # ✅ 추가: HF remote code 신뢰 여부
    ):
        """
        Args:
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 겹침 크기
            embedding_model: 임베딩 모델명
            persist_directory: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
            trust_remote_code: HF 모델 로드시 커스텀 코드 신뢰 여부 (예: nomic-ai/*)
        """
        logger.info("RAG 파이프라인 초기화 중...")

        self.loader = PDFLoader()
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = Embedder(
            model_name=embedding_model,
            trust_remote_code=trust_remote_code,  # ✅ 전달
        )
        self.indexer = ChromaDBIndexer(
            persist_directory=persist_directory,
            collection_name=collection_name
        )

        logger.info("RAG 파이프라인 초기화 완료")

    def process_directory(
        self,
        directory: str,
        reset_collection: bool = False
    ) -> None:
        """
        디렉토리의 모든 PDF 파일 처리

        Args:
            directory: PDF 파일이 있는 디렉토리
            reset_collection: 기존 컬렉션 초기화 여부
        """
        logger.info(f"디렉토리 처리 시작: {directory}")

        # 기존 컬렉션 초기화 (옵션)
        if reset_collection:
            self.indexer.reset()

        # PDF 파일 목록 가져오기
        pdf_files = get_pdf_files(directory)
        if not pdf_files:
            logger.warning(f"PDF 파일을 찾을 수 없습니다: {directory}")
            return

        logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")

        # 1단계: 로드
        documents = self.loader.load_multiple([str(f) for f in pdf_files])

        # 2단계: 청킹
        chunks = self.chunker.chunk_multiple(documents)

        # 청크 유효성 검증
        if not validate_chunks(chunks):
            raise ValueError("유효하지 않은 청크가 생성되었습니다")

        # 3단계: 임베딩
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        # 4단계: 인덱싱
        self.indexer.index(chunks_with_embeddings)

        # 통계 출력
        stats = self.indexer.get_stats()
        logger.info(f"파이프라인 완료: {stats}")

    def process_files(
        self,
        file_paths: List[str],
        reset_collection: bool = False
    ) -> None:
        """
        특정 PDF 파일들 처리

        Args:
            file_paths: PDF 파일 경로 리스트
            reset_collection: 기존 컬렉션 초기화 여부
        """
        logger.info(f"파일 처리 시작: {len(file_paths)}개")

        if reset_collection:
            self.indexer.reset()

        # 1단계: 로드
        documents = self.loader.load_multiple(file_paths)

        # 2단계: 청킹
        chunks = self.chunker.chunk_multiple(documents)

        # 3단계: 임베딩
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        # 4단계: 인덱싱
        self.indexer.index(chunks_with_embeddings)

        # 통계 출력
        stats = self.indexer.get_stats()
        logger.info(f"파이프라인 완료: {stats}")

    def search(self, query: str, n_results: int = 5) -> List[dict]:
        """
        쿼리 검색

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.embedder.embed(query)

        # 검색
        results = self.indexer.search(
            query_embedding=query_embedding.tolist(),
            n_results=n_results
        )

        return results
