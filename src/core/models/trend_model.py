"""
Trend 분석 관련 모델

TrendTier: 기술 트렌드 2-Tier 분류
"""
from pydantic import BaseModel, Field, field_validator


class TrendTier(BaseModel):
    """
    기술 트렌드 2-Tier 분류
    
    Tier 기준:
    - HOT_TRENDS: 논문 100편+, 기업 참여율 40%+ → 1-2년 내 상용화 예상
    - RISING_STARS: 논문 30-100편, 기업 참여율 20-40% → 3-5년 핵심 기술
    
    Example:
        {
            "name": "Humanoid Robots",
            "tier": "HOT_TRENDS",
            "paper_count": 152,
            "company_ratio": 45,  # 45% 또는 0.45 모두 허용
            "reasoning": "높은 논문 수와 기업 참여율로 상용화 임박"
        }
    """
    
    name: str = Field(
        min_length=2,
        description="기술명"
    )
    
    tier: str = Field(
        pattern=r'^(HOT_TRENDS|RISING_STARS)$',
        description="분류 티어 (HOT_TRENDS: 1-2년 상용화, RISING_STARS: 3-5년 핵심)"
    )
    
    paper_count: int = Field(
        ge=0,
        description="관련 논문 개수"
    )
    
    company_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="기업 참여율 (0.0 ~ 1.0 비율 또는 0 ~ 100% 자동 변환)"
    )
    
    reasoning: str = Field(
        min_length=10,
        description="분류 이유 (최소 10자 이상)"
    )
    
    @field_validator('company_ratio', mode='before')
    @classmethod
    def convert_percentage_to_ratio(cls, v):
        """
        퍼센트 값을 비율로 자동 변환
        
        - 0~1 사이 값: 그대로 유지 (이미 비율)
        - 1~100 사이 값: 100으로 나누어 비율로 변환
        - 100 초과 값: 에러 발생
        """
        if isinstance(v, (int, float)):
            if 0 <= v <= 1:
                return float(v)  # 이미 비율 형태
            elif 1 < v <= 100:
                return float(v / 100)  # 퍼센트를 비율로 변환
            else:
                raise ValueError(f"company_ratio는 0-100% 범위여야 합니다. 입력값: {v}")
        return v
    
    def is_hot_trend(self) -> bool:
        """HOT_TRENDS 여부 확인"""
        return self.tier == "HOT_TRENDS"
    
    def is_rising_star(self) -> bool:
        """RISING_STARS 여부 확인"""
        return self.tier == "RISING_STARS"
    
    def get_company_percentage(self) -> float:
        """기업 참여율을 퍼센트로 반환"""
        return self.company_ratio * 100
    
    def __str__(self) -> str:
        """사용자 친화적인 문자열 표현"""
        return (f"{self.name} ({self.tier}): "
                f"논문 {self.paper_count}편, "
                f"기업 참여율 {self.get_company_percentage():.1f}%")