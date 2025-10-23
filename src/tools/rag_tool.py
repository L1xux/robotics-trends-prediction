"""
RAGTool - Hybrid Retrieval 도구

ChromaDB(dense) + BM25(sparse) + MMR(diversity)를 결합한 검색으로
FTSG, WEF 참고 문서에서 관련 정보를 검색합니다.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import chromadb
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.schema import Document

from src.tools.base.base_tool import BaseTool
from src.tools.base.tool_config import ToolConfig
from src.core.settings import Settings
from src.core.models.citation_model import RAGCitation


class RAGTool(BaseTool):
    """
    Advanced RAG 검색 도구 (BM25 + Cosine + MMR 지원)

    Features:
    - Similarity search: 의미적 유사도만 (dense only)
    - MMR search: 의미적 유사도 + 다양성 (dense + diversity)
    - Hybrid search: 의미적 + 키워드 매칭 (dense + sparse)
    - Hybrid MMR search: BM25 + Cosine + MMR (ALL, 최강 조합! 권장)

    Search Types:
    - "similarity": 벡터 유사도만 (빠름, 관련성 우선)
    - "mmr": 벡터 유사도 + MMR 다양성 (중복 최소화)
    - "hybrid": 벡터 + BM25 키워드 (균형잡힌 검색)
    - "hybrid_mmr": 벡터 + BM25 + MMR (관련성 + 키워드 + 다양성, 권장!)
    """

    def __init__(self, config: ToolConfig, settings: Optional[Settings] = None):
        super().__init__(config)
        self.settings = settings or Settings()

        # ===== 공통 컬렉션명 통일 =====
        self.collection_name = getattr(self.settings, "CHROMADB_COLLECTION", None) or "documents"

        # ChromaDB 경로 확인 및 디버깅
        chromadb_path = self.settings.CHROMADB_PATH
        print(f"[RAGTool] ChromaDB Path: {chromadb_path}")
        print(f"[RAGTool] ChromaDB Exists: {chromadb_path.exists()}")
        print(f"[RAGTool] Collection Name: {self.collection_name}")
        print(f"[RAGTool] Embedding Model: {self.settings.EMBEDDING_MODEL}")

        # ChromaDB 초기화
        self.chroma_client = chromadb.PersistentClient(
            path=str(chromadb_path)
        )

        # 사용 가능한 컬렉션 확인
        try:
            collections = self.chroma_client.list_collections()
            print(f"[RAGTool] Available collections: {[c.name for c in collections]}")
            
            # 컬렉션 존재 확인
            collection = self.chroma_client.get_collection(self.collection_name)
            doc_count = collection.count()
            print(f"[RAGTool] Collection '{self.collection_name}' has {doc_count} documents")
        except Exception as e:
            print(f"[RAGTool] Warning: Could not access collection '{self.collection_name}': {e}")

        # SentenceTransformer Embeddings (nomic-ai)
        print(f"[RAGTool] Loading embedding model: {self.settings.EMBEDDING_MODEL}")
        
        # HuggingFaceEmbeddings 사용 (Langchain 호환)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True, 'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # SentenceTransformer 모델 (MMR용)
        self.sentence_transformer = SentenceTransformer(
            self.settings.EMBEDDING_MODEL,
            trust_remote_code=True,
            device='cpu'
        )
        
        print(f"[RAGTool] Embedding model loaded successfully")

        # Vector store (dense retriever)
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,   # ✅ 통일
            embedding_function=self.embeddings
        )

        # Documents 로드 (BM25용)
        self.documents = self._load_all_documents()

        # BM25 retriever (sparse retriever) 초기화 (비어있으면 None)
        if not self.documents:
            print(f"[RAGTool] No documents for BM25 (collection='{self.collection_name}'). BM25 disabled.")
            self.bm25_retriever = None
        else:
            print(f"[RAGTool] BM25 retriever initialized with {len(self.documents)} documents")
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)

        # Hybrid retriever (ensemble) - 동적 생성
        self.hybrid_retriever = None

    def _load_all_documents(self) -> List[Document]:
        """
        ChromaDB에서 모든 문서 로드 (BM25용)

        Returns:
            Document 리스트
        """
        try:
            col = self.chroma_client.get_collection(self.collection_name)
            total = col.count()
            if total == 0:
                return []
            
            data = col.get(limit=total, include=["documents", "metadatas"])
            docs = []
            docs_raw = data.get("documents", []) or []
            metas_raw = data.get("metadatas", []) or []
            
            for doc, meta in zip(docs_raw, metas_raw):
                if doc and str(doc).strip():
                    docs.append(Document(page_content=doc, metadata=meta or {}))
            print(f"[RAGTool] Loaded {len(docs)} documents for BM25")
            return docs
        except Exception as e:
            print(f"[RAGTool] Warning: Could not load documents for BM25 (collection='{self.collection_name}'): {e}")
            return []

    def _run(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid_mmr",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> Dict[str, Any]:
        """
        RAG 검색 실행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            search_type: 검색 타입
                - "similarity": 유사도 검색 (dense only)
                - "mmr": MMR 검색 (dense + diversity)
                - "hybrid": Hybrid 검색 (dense + sparse)
                - "hybrid_mmr": BM25 + Cosine + MMR (ALL, 권장!)
            dense_weight: Dense retrieval 가중치 (hybrid용, 0.0 ~ 1.0)
            sparse_weight: Sparse retrieval 가중치 (hybrid용, 0.0 ~ 1.0)
            fetch_k: Stage 1에서 가져올 후보 개수 (hybrid_mmr용)
            lambda_mult: MMR 다양성 파라미터 (mmr용, 0.0=다양성↑, 1.0=관련성↑)
        """
        try:
            print(f"[RAGTool] Searching with type='{search_type}', top_k={top_k}")
            print(f"[RAGTool] Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            docs: List[Document] = []

            if search_type == "similarity":
                # 1. Similarity search (dense only)
                docs = self.vector_store.similarity_search(query, k=top_k)

            elif search_type == "mmr":
                # 2. MMR search (dense + diversity)
                docs = self.vector_store.max_marginal_relevance_search(
                    query, k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )

            elif search_type == "hybrid":
                # 3. Hybrid search (dense + sparse)
                docs = self._hybrid_search(query, top_k, dense_weight, sparse_weight)

            elif search_type == "hybrid_mmr":
                # 4. Hybrid + MMR (BM25 + Cosine + MMR)
                docs = self._hybrid_mmr_search(
                    query, top_k, dense_weight, sparse_weight, fetch_k, lambda_mult
                )

            else:
                raise ValueError(
                    f"Invalid search_type: {search_type}. "
                    f"Must be one of: similarity, mmr, hybrid, hybrid_mmr"
                )

            print(f"[RAGTool] Found {len(docs)} documents")

            # 결과 포맷팅
            results = []
            for doc in docs[:top_k]:
                result = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "score": doc.metadata.get("score", 0.0),
                }
                results.append(result)

            print(f"[RAGTool] Returning {len(results)} results")

            # Citations 생성
            citations = []
            for doc in docs[:top_k]:
                try:
                    citation = RAGCitation(
                        source=doc.metadata.get("source", "Reference"),
                        page=str(doc.metadata.get("page", "")) if doc.metadata.get("page") else None,
                        section=doc.metadata.get("section", None)
                    )
                    citations.append(citation)
                except Exception as e:
                    print(f"[RAGTool] Citation 생성 실패: {str(e)[:100]}")
                    continue
            
            print(f"[RAGTool] Citations created: {len(citations)}")

            return {
                "query": query,
                "search_type": search_type,
                "total_results": len(results),
                "documents": results,  # ✅ 'documents' 키로 통일
                "results": results,  # 하위 호환성 유지
                "citations": citations  # Citation 객체 리스트 추가
            }

        except Exception as e:
            print(f"[RAGTool] Error: {e}")
            return {
                "query": query,
                "search_type": search_type,
                "total_results": 0,
                "documents": [],
                "results": [],
                "error": f"RAG search failed: {str(e)}"
            }

    def search_with_filter(
        self,
        query: str,
        source_filter: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> Dict[str, Any]:
        """
        필터링된 검색 (특정 문서만)
        """
        try:
            filter_dict = {"source": source_filter} if source_filter else None

            if search_type == "mmr":
                docs = self.vector_store.max_marginal_relevance_search(
                    query, k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter_dict
                )
            else:
                docs = self.vector_store.similarity_search(
                    query, k=top_k, filter=filter_dict
                )

            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                })

            return {
                "query": query,
                "source_filter": source_filter,
                "search_type": search_type,
                "total_results": len(results),
                "results": results
            }

        except Exception as e:
            return {
                "query": query,
                "total_results": 0,
                "results": [],
                "error": f"Filtered search failed: {str(e)}"
            }

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        sparse_weight: float
    ) -> List[Document]:
        """
        Hybrid search (BM25 + Vector)
        """
        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

        retrievers = [dense_retriever]
        weights = [dense_weight]

        # BM25가 있을 때만 포함
        if self.bm25_retriever is not None:
            self.bm25_retriever.k = top_k
            retrievers.append(self.bm25_retriever)
            weights.append(sparse_weight)

        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )

        # LangChain API 변경 대응: get_relevant_documents 또는 invoke
        return ensemble_retriever.invoke(query)

    def _hybrid_mmr_search(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        fetch_k: int,
        lambda_mult: float
    ) -> List[Document]:
        """
        Hybrid + MMR search (BM25 + Cosine Similarity + MMR)

        Two-stage approach:
        1. Stage 1 (Hybrid): BM25 + Vector로 fetch_k개 후보 검색
        2. Stage 2 (MMR): 후보들에 MMR 적용하여 top_k개 선택
        """
        candidate_docs = self._hybrid_search(query, fetch_k, dense_weight, sparse_weight)
        if not candidate_docs:
            return []

        # Query embedding (SentenceTransformer 사용)
        query_embedding = self.sentence_transformer.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # MMR 적용
        selected_docs = self._apply_mmr(
            query_embedding, candidate_docs, top_k, lambda_mult
        )
        return selected_docs

    def _apply_mmr(
        self,
        query_embedding: List[float],
        candidate_docs: List[Document],
        top_k: int,
        lambda_mult: float
    ) -> List[Document]:
        r"""
        MMR = arg max [λ * sim(q, d) - (1-λ) * max sim(d, s)]
                       d∈R∖S              s∈S
        """
        # 문서 임베딩 생성 (SentenceTransformer 사용)
        doc_texts = [doc.page_content for doc in candidate_docs]
        doc_embeddings = self.sentence_transformer.encode(
            doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Query embedding을 numpy array로 변환
        query_embedding = np.array(query_embedding)

        # Cosine similarity
        def cosine_sim(a, b):
            return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))

        # Query와 각 문서 간 similarity
        query_doc_sims = [cosine_sim(query_embedding, de) for de in doc_embeddings]

        # MMR 선택
        selected_indices: List[int] = []
        remaining_indices = list(range(len(candidate_docs)))

        for _ in range(min(top_k, len(candidate_docs))):
            best_score = -float("inf")
            best_idx = None

            for idx in remaining_indices:
                relevance = query_doc_sims[idx]
                if selected_indices:
                    max_sim_to_selected = max(
                        cosine_sim(doc_embeddings[idx], doc_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim_to_selected = 0.0

                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return [candidate_docs[i] for i in selected_indices]

    async def _arun(self, *args, **kwargs):
        """비동기 실행 (LangChain 호환)"""
        raise NotImplementedError("RAGTool does not support async execution")


