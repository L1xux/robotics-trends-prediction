"""
Data Collection 관련 모델

DataCollectionStatus: 데이터 수집 상태 및 품질
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Literal, Optional


class DataCollectionStatus(BaseModel):
    """
    데이터 수집 상태
    
    source별 수집 결과, 품질 점수, 수집 개수를 추적
    
    Example:
        {
            "status": "success",
            "quality_score": 0.85,
            "items_collected": {
                "arxiv": 120,
                "trends": 36,
                "news": 53
            },
            "last_error": null
        }
    """
    
    status: Literal["success", "failed", "retry_needed"] = Field(
        description="수집 상태"
    )
    
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="데이터 품질 점수 (0.0 ~ 1.0)"
    )
    
    items_collected: Dict[str, int] = Field(
        description="Source별 수집된 항목 개수 (arxiv, trends, news)",
        min_length=3,
        max_length=3
    )
    
    last_error: Optional[str] = Field(
        default=None,
        description="마지막 에러 메시지 (있을 경우)"
    )
    
    @field_validator('items_collected')
    @classmethod
    def validate_items_collected(cls, v: Dict[str, int]) -> Dict[str, int]:
        required_sources = {"arxiv", "trends", "news"}
        
        # 필수 소스 확인
        if set(v.keys()) != required_sources:
            raise ValueError(
                f"items_collected must contain exactly: {required_sources}, "
                f"got: {set(v.keys())}"
            )
        
        # 모든 값이 0 이상
        for source, count in v.items():
            if count < 0:
                raise ValueError(
                    f"items_collected[{source}] must be >= 0, got: {count}"
                )
        
        return v
    
    @field_validator('last_error')
    @classmethod
    def validate_last_error(cls, v: Optional[str], info) -> Optional[str]:
        status = info.data.get('status')
        
        # failed 상태면 last_error 필수
        if status == 'failed' and not v:
            raise ValueError("last_error must be provided when status='failed'")
        
        return v
    
    def get_total_items(self) -> int:
        """전체 수집 개수 합계"""
        return sum(self.items_collected.values())
    
    def get_source_success_rate(self) -> Dict[str, float]:
        """
        Source별 성공률 (수집 개수 기반)
        
        Returns:
            {"arxiv": 1.0, "trends": 0.8, "news": 0.6}
        """
        total = self.get_total_items()
        if total == 0:
            return {source: 0.0 for source in self.items_collected}
        
        return {
            source: count / total
            for source, count in self.items_collected.items()
        }