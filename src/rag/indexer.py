# RAG Indexer
"""ChromaDB 인덱싱"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from pathlib import Path
from src.utils.logger import default_logger as logger

class ChromaDBIndexer:
    """ChromaDB에 문서를 인덱싱하는 클래스"""
    
    def __init__(
        self,
        persist_directory: str = "../../data/chroma_db",
        collection_name: str = "documents"
    ):
        """
        Args:
            persist_directory: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChromaDB 초기화: {persist_directory}")
        
        # ChromaDB 클라이언트 생성
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings collection"}
        )
        
        logger.info(f"컬렉션 '{collection_name}' 준비 완료")
    
    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        청크를 ChromaDB에 인덱싱
        
        Args:
            chunks: 임베딩이 포함된 청크 리스트
        """
        if not chunks:
            logger.warning("인덱싱할 청크가 없습니다")
            return
        
        logger.info(f"{len(chunks)}개 청크 인덱싱 시작")
        
        # 데이터 준비
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            # 고유 ID 생성
            chunk_id = f"{chunk['metadata'].get('filename', 'unknown')}_{idx}"
            ids.append(chunk_id)
            
            # 문서 내용
            documents.append(chunk['content'])
            
            # 임베딩
            embeddings.append(chunk['embedding'].tolist())
            
            # 메타데이터 (임베딩 제외)
            metadata = {k: str(v) for k, v in chunk['metadata'].items() 
                       if k != 'embedding'}
            metadatas.append(metadata)
        
        # ChromaDB에 추가
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"인덱싱 완료: 총 {len(ids)}개 문서")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        유사도 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            n_results: 반환할 결과 수
            where: 필터 조건
            
        Returns:
            검색 결과
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 반환"""
        count = self.collection.count()
        return {
            "collection_name": self.collection.name,
            "total_documents": count,
            "persist_directory": str(self.persist_directory)
        }
    
    def reset(self) -> None:
        """컬렉션 초기화"""
        logger.warning("컬렉션 초기화 중...")
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Document embeddings collection"}
        )
        logger.info("컬렉션 초기화 완료")