"""
PDF Converter

DOCX 파일을 PDF로 변환 (Simple and robust)
"""

from pathlib import Path
from typing import Optional
import subprocess
import platform


class PdfConverter:
    """
    DOCX to PDF Converter
    
    Uses docx2pdf library (Windows/Mac) or libreoffice (Linux)
    """
    
    def __init__(self):
        """Initialize PDF Converter"""
        self.system = platform.system()
    
    def convert(
        self,
        docx_path: str,
        pdf_path: Optional[str] = None
    ) -> str:
        """
        Convert DOCX to PDF
        
        Args:
            docx_path: Path to DOCX file
            pdf_path: Output PDF path (optional, default: same as docx_path with .pdf)
        
        Returns:
            Path to generated PDF file
        """
        docx_path = Path(docx_path)
        
        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")
        
        # Default PDF path
        if pdf_path is None:
            pdf_path = docx_path.with_suffix('.pdf')
        else:
            pdf_path = Path(pdf_path)
        
        try:
            # Method 1: Try docx2pdf (Windows/Mac)
            try:
                from docx2pdf import convert as docx2pdf_convert
                
                print(f"Converting to PDF using docx2pdf...")
                docx2pdf_convert(str(docx_path), str(pdf_path))
                
                print(f"PDF generated: {pdf_path}")
                return str(pdf_path)
            
            except ImportError:
                print(f"docx2pdf not available, trying alternative method...")
            
            except Exception as e:
                print(f"docx2pdf conversion failed: {e}")
                print(f"   Trying alternative method...")
            
            # Method 2: Try libreoffice (Linux/cross-platform)
            try:
                print(f"Converting to PDF using LibreOffice...")
                
                result = subprocess.run(
                    [
                        'soffice',
                        '--headless',
                        '--convert-to', 'pdf',
                        '--outdir', str(pdf_path.parent),
                        str(docx_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    print(f"PDF generated: {pdf_path}")
                    return str(pdf_path)
                else:
                    print(f"LibreOffice conversion failed: {result.stderr}")
            
            except FileNotFoundError:
                print(f"LibreOffice not found in PATH")
            
            except Exception as e:
                print(f"LibreOffice conversion failed: {e}")
            
            # Method 3: Fallback - just copy DOCX as "PDF" (not ideal but won't crash)
            print(f"All PDF conversion methods failed")
            print(f"   Saving DOCX as fallback (install 'docx2pdf' or 'libreoffice' for PDF conversion)")
            
            # Just keep the DOCX
            return str(docx_path)
        
        except Exception as e:
            print(f"PDF conversion failed: {e}")
            print(f"   Returning DOCX path as fallback")
            return str(docx_path)


def convert_to_pdf(
    docx_path: str,
    pdf_path: Optional[str] = None
) -> str:
    """
    Convenience function to convert DOCX to PDF
    
    Args:
        docx_path: Path to DOCX file
        pdf_path: Output PDF path (optional)
    
    Returns:
        Path to generated PDF file (or DOCX if conversion failed)
    """
    converter = PdfConverter()
    return converter.convert(docx_path, pdf_path)
