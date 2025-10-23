"""
Pydantic Models for AI-Robotics Report Generator

All models are exported for easy import:
    from core.models import QualityCheckResult, TrendTier, ...
"""

from .planning_model import (
    PlanningOutput,
    CollectionPlan,
    ArxivConfig,
    TrendsConfig,
    NewsConfig
)

from .quality_check_model import (
    QualityCheckResult,
    RetryAction
)

from .data_collection_model import (
    DataCollectionStatus
)

from .revision_model import (
    RevisionDecision
)

from .trend_model import (
    TrendTier
)

from .citation_model import (
    CitationEntry,
    Citation,
    ArXivCitation,
    NewsCitation,
    RAGCitation,
    CitationCollection
)

__all__ = [
    # Planning
    "PlanningOutput",
    "CollectionPlan",
    "ArxivConfig",
    "TrendsConfig",
    "NewsConfig",
    
    # Quality Check
    "QualityCheckResult",
    "RetryAction",
    
    # Data Collection
    "DataCollectionStatus",
    
    # Revision
    "RevisionDecision",
    
    # Trend Analysis
    "TrendTier",
    
    # Citation
    "CitationEntry",
    "Citation",
    "ArXivCitation",
    "NewsCitation",
    "RAGCitation",
    "CitationCollection",
]