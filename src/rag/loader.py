"""PDF 문서 로더"""
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from src.utils.logger import default_logger as logger

class PDFLoader:
    """PDF 파일을 로드하는 클래스"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        단일 PDF 파일 로드
        
        Args:
            file_path: PDF 파일 경로
            
        Returns:
            문서 정보 딕셔너리 (content, metadata)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        if path.suffix not in self.supported_formats:
            raise ValueError(f"지원하지 않는 파일 형식: {path.suffix}")
        
        logger.info(f"PDF 로드 중: {file_path}")
        
        # PDF 읽기
        with open(path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # 전체 텍스트 추출
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            # 메타데이터
            metadata = {
                'source': str(path),
                'filename': path.name,
                'total_pages': len(pdf_reader.pages),
                'file_type': 'pdf'
            }
            
            # PDF 메타데이터 추가
            if pdf_reader.metadata:
                metadata.update({
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', '')
                })
        
        return {
            'content': full_text,
            'metadata': metadata
        }
    
    def load_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        여러 PDF 파일 로드
        
        Args:
            file_paths: PDF 파일 경로 리스트
            
        Returns:
            문서 정보 리스트
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"파일 로드 실패 ({file_path}): {e}")
                continue
        
        logger.info(f"총 {len(documents)}개 문서 로드 완료")
        return documents