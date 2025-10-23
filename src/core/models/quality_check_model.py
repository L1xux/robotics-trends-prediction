"""
Quality Check 관련 모델

QualityCheckResult: 데이터 품질 검증 결과
RetryAction: 재시도 액션 상세 정보
"""
from src.core.patterns.base_model import BaseModel, Field, field_validator
from typing import Dict, List, Literal, Optional, Any





class RetryAction(BaseModel):
    """
    재수집 액션 상세
    
    action:
        - "none": 재수집 불필요
        - "expand_keywords": 키워드 확장 후 재수집
        - "adjust_params": 파라미터 조정 후 재수집
    """
    
    action: Literal["none", "expand_keywords", "adjust_params"] = Field(
        description="재수집 액션 타입"
    )
    
    keywords: Optional[List[str]] = Field(
        default=None,
        description="확장할 키워드 리스트 (action='expand_keywords' 시)"
    )
    
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="조정할 파라미터 (action='adjust_params' 시)"
    )
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        action = info.data.get('action')
        
        # expand_keywords 시 keywords 필수
        if action == 'expand_keywords':
            if not v or len(v) == 0:
                raise ValueError("keywords must be provided when action='expand_keywords'")
            
            # 빈 문자열 제거
            filtered = [k.strip() for k in v if k.strip()]
            if not filtered:
                raise ValueError("keywords must contain at least one non-empty string")
            
            return filtered
        
        return v
    
    @field_validator('params')
    @classmethod
    def validate_params(cls, v: Optional[Dict[str, Any]], info) -> Optional[Dict[str, Any]]:
        action = info.data.get('action')
        
        # adjust_params 시 params 필수
        if action == 'adjust_params':
            if not v or len(v) == 0:
                raise ValueError("params must be provided when action='adjust_params'")
        
        return v


class QualityCheckResult(BaseModel):
    """
    데이터 품질 검증 결과
    
    LLM이 수집된 데이터를 평가하여 Pass/Retry 판단
    
    Example:
        {
            "status": "retry",
            "issues": {
                "arxiv": null,
                "news": "too_few_sources"
            },
            "retry_plan": {
                "arxiv": {"action": "none"},
                "news": {
                    "action": "adjust_params",
                    "params": {"sources": 5}
                }
            },
            "reasoning": "News 데이터가 부족하여 소스를 확장합니다..."
        }
    """
    
    status: Literal["pass", "retry"] = Field(
        description="검증 결과 (pass: 진행, retry: 재수집)"
    )
    
    issues: Dict[str, Optional[str]] = Field(
        description="Source별 이슈 (arxiv, news)",
        min_length=2,  # 2개 소스 필수
        max_length=2
    )
    
    retry_plan: Dict[str, RetryAction] = Field(
        description="Source별 재수집 계획",
        min_length=2,
        max_length=2
    )
    
    reasoning: str = Field(
        description="판단 근거 상세 설명",
        min_length=20  # 최소 20자 이상
    )
    
    @field_validator('issues')
    @classmethod
    def validate_issues(cls, v: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        required_sources = {"arxiv", "news"}
        
        if set(v.keys()) != required_sources:
            raise ValueError(
                f"issues must contain exactly: {required_sources}, "
                f"got: {set(v.keys())}"
            )
        
        return v
    
    @field_validator('retry_plan')
    @classmethod
    def validate_retry_plan(cls, v: Dict[str, RetryAction], info) -> Dict[str, RetryAction]:
        required_sources = {"arxiv", "news"}
        
        if set(v.keys()) != required_sources:
            raise ValueError(
                f"retry_plan must contain exactly: {required_sources}, "
                f"got: {set(v.keys())}"
            )
        
        # status가 'pass'면 모든 action이 'none'이어야 함
        status = info.data.get('status')
        if status == 'pass':
            non_none_actions = [
                source for source, action in v.items() 
                if action.action != 'none'
            ]
            if non_none_actions:
                raise ValueError(
                    f"When status='pass', all retry actions must be 'none'. "
                    f"Found non-none actions for: {non_none_actions}"
                )
        
        return v