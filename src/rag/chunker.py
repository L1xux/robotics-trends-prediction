"""Semantic 청킹 처리"""
from typing import List, Dict, Any
from src.utils.logger import default_logger as logger

class SemanticChunker:
    """의미 기반 청킹을 수행하는 클래스"""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 300,
        separators: List[str] = None
    ):
        """
        Args:
            chunk_size: 청크 최대 크기 (문자 수)
            chunk_overlap: 청크 간 겹치는 부분 크기
            separators: 청크 분리 기준 (우선순위 순)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 빈 문자열("") 제거, 큰 단위 -> 작은 단위 우선순위
        self.separators = separators or [
            "\n\n\n",  # 큰 섹션 구분
            "\n\n",    # 단락 구분
            "\n",      # 줄바꿈
            ". ",      # 문장 끝
            "! ",
            "? ",
            " ",       # 단어 구분
        ]
    
    def chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = document['content']
        metadata = document['metadata']
        
        logger.info(f"청킹 시작: {metadata.get('filename', 'unknown')}")
        
        # 텍스트 청킹
        chunks = self._split_text(content, start_sep_idx=0)
        
        # 메타데이터와 함께 청크 생성
        result_chunks = []
        for idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': idx,
                'chunk_size': len(chunk_text)
            })
            result_chunks.append({
                'content': chunk_text,
                'metadata': chunk_metadata
            })
        
        logger.info(f"총 {len(result_chunks)}개 청크 생성")
        return result_chunks

    def _split_text(self, text: str, start_sep_idx: int = 0) -> List[str]:
        """분리자 우선순위를 내려가며 재귀적으로 텍스트를 분할"""
        t = text.strip()
        if not t:
            return []
        if len(t) <= self.chunk_size:
            return [t]

        # 현재 우선순위부터 순차 시도
        for i in range(start_sep_idx, len(self.separators)):
            sep = self.separators[i]
            if sep and sep in t:
                out: List[str] = []
                splits = t.split(sep)
                current = ""

                for idx, piece in enumerate(splits):
                    # 이번 루프에서 piece 뒤에 sep을 붙일지 결정 (마지막 제외)
                    add_sep = (idx < len(splits) - 1)
                    piece_plus = (piece + sep) if add_sep else piece

                    # 현재 청크에 붙여도 되는지
                    if len(current) + len(piece_plus) <= self.chunk_size:
                        current += piece_plus
                    else:
                        # 현재 청크를 배출
                        if current.strip():
                            out.append(current.strip())
                        # 새로 시작해야 하는 piece가 너무 크면,
                        # 더 미세한 분리자(i+1)로 재귀 분할
                        if len(piece_plus) > self.chunk_size:
                            out.extend(self._split_text(piece_plus, start_sep_idx=i+1))
                            current = ""
                        else:
                            current = piece_plus

                # 마지막 청크 추가
                if current.strip():
                    out.append(current.strip())

                # 만들어진 것이 있고 모두 규격 이하면 반환
                if out:
                    return out

        # 모든 분리자에서 규격화 실패 → 강제 슬라이싱 (overlap 적용)
        return self._force_slice(t)

    def _force_slice(self, text: str) -> List[str]:
        """모든 분리자 실패 시 강제 분할"""
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def chunk_multiple(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        logger.info(f"전체 {len(all_chunks)}개 청크 생성 완료")
        return all_chunks
