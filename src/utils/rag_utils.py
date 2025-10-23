"""공통 유틸리티 함수"""
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pdf_files(directory: str) -> List[Path]:
    """디렉토리에서 PDF 파일 목록 가져오기"""
    path = Path(directory)
    return list(path.glob("*.pdf"))

def validate_chunks(chunks: List[Dict[str, Any]]) -> bool:
    """청크 유효성 검증"""
    if not chunks:
        return False
    
    required_keys = ['content', 'metadata']
    for chunk in chunks:
        if not all(key in chunk for key in required_keys):
            return False
    return True