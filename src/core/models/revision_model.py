"""
Revision 관련 모델

RevisionDecision: 수정 유형 결정 (SMALL/LARGE)
"""
from src.core.patterns.base_model import BaseModel, Field


class RevisionDecision(BaseModel):
    """
    수정 유형 결정
    
    LLM이 Human Feedback을 분석하여 수정 유형 판단
    
    decision:
        - SMALL: 오타, 표현 수정, 포맷 변경 등 (Revision Agent 처리)
        - LARGE: 내용 추가/삭제, 데이터 재분석, 구조 변경 등 (Content Analysis 재실행)
    
    Example:
        {
            "decision": "LARGE",
            "reason": "Section 3에 추가 사례가 필요하므로 데이터 재분석이 필요합니다."
        }
    """
    
    decision: str = Field(
        pattern=r'^(SMALL|LARGE)$',
        description="수정 유형 (SMALL: Revision Agent, LARGE: Content Analysis)"
    )
    
    reason: str = Field(
        min_length=10,
        description="판단 이유 (최소 10자 이상)"
    )

