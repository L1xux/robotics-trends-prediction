"""임베딩 생성"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.logger import default_logger as logger


class Embedder:
    """텍스트를 벡터 임베딩으로 변환하는 클래스"""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        trust_remote_code: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: 사용할 임베딩 모델명
            trust_remote_code: HF 모델 로드시 커스텀 코드 신뢰 여부 (예: nomic-ai/*)
            device: 'cpu' 또는 'cuda' 등 디바이스 지정 (기본: 자동)
        """
        logger.info(f"임베딩 모델 로드 중: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=trust_remote_code,
            device=device,
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"임베딩 차원: {self.dimension}")

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        단일 텍스트 임베딩

        Args:
            text: 입력 텍스트
            normalize: 임베딩 L2 정규화 여부

        Returns:
            임베딩 벡터 (np.ndarray, shape: [dim])
        """
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        배치 임베딩

        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            normalize: 임베딩 L2 정규화 여부

        Returns:
            임베딩 벡터 배열 (np.ndarray, shape: [N, dim])
        """
        logger.info(f"{len(texts)}개 텍스트 임베딩 시작")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) >= batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        logger.info("임베딩 완료")
        return embeddings

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        청크에 임베딩 추가

        Args:
            chunks: 청크 리스트 (각 항목에 'content' 키가 있어야 함)

        Returns:
            임베딩이 추가된 청크 리스트 ('embedding' 키 추가)
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks
