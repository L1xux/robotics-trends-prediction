"""
Citation Models for Data Collection

Citation formats for different data sources:
- ArXiv papers: APA style academic citations
- News articles: News-style citations with source and date
- RAG documents: Reference document citations with page/section
"""

from typing import List, Dict, Any, Optional
from src.core.patterns.base_model import BaseModel, Field


class CitationEntry(BaseModel):
    """
    Citation Entry for report content
    
    Used in analysis phase to reference sources within report sections.
    Simpler than full Citation objects, designed for inline references.
    """
    number: int = Field(description="Citation number (e.g., 1, 2, 3...)")
    source_type: str = Field(description="Type of source: 'arxiv', 'news', 'report'")
    title: str = Field(description="Title of the source")
    url: Optional[str] = Field(default=None, description="URL if available")
    authors: Optional[str] = Field(default=None, description="Authors (for papers)")
    source: Optional[str] = Field(default=None, description="Source name (for news/reports)")
    date: Optional[str] = Field(default=None, description="Publication date")
    
    def to_inline_ref(self) -> str:
        """Generate inline reference string [number]"""
        return f"[{self.number}]"
    
    def to_reference_text(self) -> str:
        """Generate reference list text"""
        if self.source_type == "arxiv":
            return f"{self.authors or 'Unknown'}. {self.title}. {self.url or ''}"
        elif self.source_type == "news":
            return f"{self.title}. {self.source or 'Unknown'}. {self.date or ''}"
        elif self.source_type == "report":
            return f"{self.title}. {self.source or 'Reference Document'}"
        else:
            return f"{self.title}"


class Citation(BaseModel):
    """
    Base citation model
    """
    full_citation: str = Field(description="Complete citation in standard format")
    short_citation: str = Field(description="Short citation for inline use")
    source_type: str = Field(description="Type of source: 'arxiv', 'news', 'rag'")
    url: Optional[str] = Field(default=None, description="Source URL if available")
    date: Optional[str] = Field(default=None, description="Publication date")


class ArXivCitation(Citation):
    """
    ArXiv paper citation (APA style)
    """
    authors: List[str] = Field(description="List of authors")
    title: str = Field(description="Paper title")
    arxiv_id: str = Field(description="arXiv identifier")
    published: str = Field(description="Publication date")
    
    def __init__(self, **data):
        # Format full citation
        authors_str = ', '.join(data.get('authors', [])[:3])
        if len(data.get('authors', [])) > 3:
            authors_str += ' et al.'
        
        full_citation = f"{authors_str} ({data.get('published', 'Unknown')}). {data.get('title', 'Unknown')}. arXiv preprint {data.get('arxiv_id', 'unknown')}"
        short_citation = f"({authors_str}, {data.get('published', 'Unknown')})"
        
        # 자식 클래스 필드만 추출
        child_fields = {
            'authors': data.get('authors'),
            'title': data.get('title'),
            'arxiv_id': data.get('arxiv_id'),
            'published': data.get('published')
        }
        
        super().__init__(
            full_citation=full_citation,
            short_citation=short_citation,
            source_type="arxiv",
            url=data.get('url'),
            date=data.get('published'),
            **child_fields
        )


class NewsCitation(Citation):
    """
    News article citation
    """
    title: str = Field(description="Article title")
    source: str = Field(description="News source")
    published: str = Field(description="Publication date")
    
    def __init__(self, **data):
        # Format full citation
        full_citation = f"{data.get('title', 'Unknown')}. {data.get('source', 'Unknown')}. {data.get('published', 'Unknown')}. Retrieved from {data.get('url', 'Unknown URL')}"
        short_citation = f"({data.get('source', 'Unknown')}, {data.get('published', 'Unknown')})"
        
        # 자식 클래스 필드만 추출
        child_fields = {
            'title': data.get('title'),
            'source': data.get('source'),
            'published': data.get('published')
        }
        
        super().__init__(
            full_citation=full_citation,
            short_citation=short_citation,
            source_type="news",
            url=data.get('url'),
            date=data.get('published'),
            **child_fields
        )


class RAGCitation(Citation):
    """
    RAG document citation
    """
    source: str = Field(description="Document source (e.g., FTSG, WEF)")
    page: Optional[str] = Field(default=None, description="Page number if available")
    section: Optional[str] = Field(default=None, description="Section if available")
    
    def __init__(self, **data):
        # Format full citation
        source = data.get('source', 'Reference Document')
        page = data.get('page', '')
        section = data.get('section', '')
        
        if page:
            full_citation = f"{source} (Page {page})"
        elif section:
            full_citation = f"{source} ({section})"
        else:
            full_citation = f"{source} (Reference Document)"
        
        short_citation = f"({source})"
        
        # 자식 클래스 필드만 추출
        child_fields = {
            'source': source,
            'page': page,
            'section': section
        }
        
        super().__init__(
            full_citation=full_citation,
            short_citation=short_citation,
            source_type="rag",
            url=data.get('url'),
            date=data.get('date'),
            **child_fields
        )


class CitationCollection(BaseModel):
    """
    Collection of all citations for a report
    """
    arxiv_citations: List[ArXivCitation] = Field(default_factory=list)
    news_citations: List[NewsCitation] = Field(default_factory=list)
    rag_citations: List[RAGCitation] = Field(default_factory=list)
    
    def get_all_citations(self) -> List[Citation]:
        """Get all citations as a single list"""
        return self.arxiv_citations + self.news_citations + self.rag_citations
    
    def get_citations_by_type(self, source_type: str) -> List[Citation]:
        """Get citations by source type"""
        if source_type == "arxiv":
            return self.arxiv_citations
        elif source_type == "news":
            return self.news_citations
        elif source_type == "rag":
            return self.rag_citations
        else:
            return []
    
    def format_reference_list(self) -> str:
        """Format all citations as a reference list"""
        references = []
        
        # ArXiv papers
        for i, citation in enumerate(self.arxiv_citations, 1):
            references.append(f"{i}. {citation.full_citation}")
        
        # News articles
        start_num = len(self.arxiv_citations) + 1
        for i, citation in enumerate(self.news_citations, start_num):
            references.append(f"{i}. {citation.full_citation}")
        
        # RAG documents
        start_num = len(self.arxiv_citations) + len(self.news_citations) + 1
        for i, citation in enumerate(self.rag_citations, start_num):
            references.append(f"{i}. {citation.full_citation}")
        
        return "\n".join(references)