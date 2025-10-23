"""ChromaDB 인덱스 빌더 스크립트"""
import sys
from pathlib import Path
import argparse
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="PDF 문서를 ChromaDB에 인덱싱합니다."
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="reference_docs",
        help="PDF 파일이 있는 디렉토리 (기본값: reference_docs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="청크 최대 크기 (기본값: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="청크 겹침 크기 (기본값: 200)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="임베딩 모델 (기본값: nomic-ai/nomic-embed-text-v1)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./data/chroma_db",
        help="ChromaDB 저장 경로 (기본값: ./data/chroma_db)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="documents",
        help="컬렉션 이름 (기본값: documents)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 컬렉션을 초기화하고 새로 시작",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="HF 모델의 커스텀 코드를 신뢰하고 로드 (예: nomic-ai/* 계열)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RAG 인덱스 빌더 시작")
    logger.info("=" * 60)
    logger.info(f"문서 디렉토리: {args.docs_dir}")
    logger.info(f"청크 크기: {args.chunk_size}")
    logger.info(f"청크 겹침: {args.chunk_overlap}")
    logger.info(f"임베딩 모델: {args.embedding_model}")
    logger.info(f"DB 경로: {args.db_path}")
    logger.info(f"컬렉션: {args.collection_name}")
    logger.info(f"초기화 모드: {args.reset}")
    logger.info(f"trust_remote_code: {args.trust_remote_code}")
    logger.info("=" * 60)

    try:
        pipeline = RAGPipeline(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            persist_directory=args.db_path,
            collection_name=args.collection_name,
            trust_remote_code=args.trust_remote_code,  # 파이프라인/임베더로 전달
        )

        pipeline.process_directory(
            directory=args.docs_dir,
            reset_collection=args.reset,
        )

        logger.info("=" * 60)
        logger.info("인덱싱 완료!")
        logger.info("=" * 60)

        stats = pipeline.indexer.get_stats()
        logger.info(f"컬렉션: {stats.get('collection_name')}")
        logger.info(f"총 문서 수: {stats.get('total_documents')}")
        logger.info(f"저장 위치: {stats.get('persist_directory')}")

    except Exception as e:
        logger.error(f"인덱싱 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
