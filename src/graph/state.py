"""
LangGraph State Definition

전체 파이프라인에서 사용하는 공유 상태 정의
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal
from typing_extensions import NotRequired


class PipelineState(TypedDict):
    """
    AI-Robotics Report Generator Pipeline State
    
    전체 워크플로우에서 공유되는 상태
    각 노드는 이 state를 입력받고, 수정된 state를 반환
    
    Workflow Phases:
    1. Planning → Human Review 1 → Data Collection → Content Analysis
    2. (Future) Report Synthesis → Writer → Human Review 2 → Revision → PDF Generation
    """
    
    # ===== Phase 0: User Input =====
    user_input: str  # 사용자가 입력한 주제
    
    # ===== Phase 1: Planning =====
    planning_output: NotRequired[Any]  # PlanningOutput 객체 (from src.core.models.planning_model)
    folder_name: NotRequired[str]  # 실행 폴더명: {topic}_{YYYYMMDD}_{HHMMSS}
    keywords: NotRequired[List[str]]  # Planning Agent가 생성한 키워드 리스트
    
    # ===== Phase 2: Human Review 1 =====
    human_review_1: NotRequired[bool]  # True: Accept, False: Reject
    human_review_1_feedback: NotRequired[str]  # Reject 시 피드백 (optional)
    
    # ===== Phase 3: Data Collection =====
    arxiv_data: NotRequired[Dict[str, Any]]  # arXiv 논문 데이터
    trends_data: NotRequired[Dict[str, Any]]  # Google Trends 데이터
    news_data: NotRequired[Dict[str, Any]]  # Tech News 기사 데이터
    expanded_keywords: NotRequired[List[str]]  # 확장된 키워드 (arXiv + RAG 기반)
    
    # ===== Phase 4: Quality Check (Internal in Data Collection Agent) =====
    collection_status: NotRequired[Any]  # DataCollectionStatus 객체 (품질 점수, 수집 개수)
    quality_check_result: NotRequired[Any]  # QualityCheckResult 객체 (LLM 품질 평가)
    retry_count: NotRequired[int]  # 현재 재시도 횟수 (default: 0)
    max_retries: NotRequired[int]  # 최대 재시도 횟수 (default: 3)
    
    # ===== Phase 5: Content Analysis =====
    rag_results: NotRequired[Dict[str, Any]]  # RAG Tool 검색 결과 (FTSG, WEF 보고서)
    trends: NotRequired[List[Any]]  # List[TrendTier] - 2-Tier 분류 결과
    sections: NotRequired[Dict[str, str]]  # 10개 서브섹션 내용 (section_2_1, section_2_2, ...)
    section_contents: NotRequired[Dict[str, str]]  # Alias for sections (legacy)
    citations: NotRequired[List[Any]]  # List[CitationEntry] - 본문에서 사용된 인용
    
    # ===== Phase 6: Report Synthesis =====
    summary: NotRequired[str]  # Executive Summary
    section_1: NotRequired[str]  # Section 1: Introduction
    introduction: NotRequired[str]  # Alias for section_1 (legacy)
    section_6: NotRequired[str]  # Section 6: Conclusion
    conclusion: NotRequired[str]  # Alias for section_6 (legacy)
    references: NotRequired[str]  # References (formatted citations)
    appendix: NotRequired[str]  # Appendix
    
    # ===== Phase 7: Writer =====
    final_report: NotRequired[str]  # 최종 보고서 전문 (마크다운)
    report_content: NotRequired[str]  # Alias for final_report (legacy)
    report_generated_at: NotRequired[str]  # 보고서 생성 시간
    quality_report: NotRequired[Dict[str, Any]]  # Quality check result
    
    # ===== Phase 8: Human Review 2 =====
    human_review_2: NotRequired[bool]  # True if review completed
    review_feedback: NotRequired[str]  # User feedback
    feedback_classification: NotRequired[Dict[str, Any]]  # Feedback severity analysis
    revision_type: NotRequired[Literal["minor", "major"]]  # Revision type needed
    
    # ===== Phase 9: Revision =====
    revision_count: NotRequired[int]  # Number of revisions performed
    revision_decision: NotRequired[Literal["SMALL", "LARGE"]]  # Legacy field
    
    # ===== Phase 10: PDF Generation (Future) =====
    pdf_path: NotRequired[str]  # 생성된 PDF 파일 경로
    docx_path: NotRequired[str]  # 중간 DOCX 파일 경로
    translation_failed: NotRequired[bool] # 한국어 번역 실패 여부
    
    # ===== Workflow Control =====
    status: str  # 현재 워크플로우 상태
        # Phase 1-3:
        # - "initialized"
        # - "planning_complete", "planning_accepted", "planning_rejected", "planning_refined"
        # - "data_collection_complete", "data_collection_failed"
        # - "analysis_complete", "analysis_failed"
        # Phase 4-7:
        # - "synthesis_complete", "synthesis_failed"
        # - "writer_complete", "writer_failed"
        # - "report_accepted", "needs_minor_revision", "needs_major_revision"
        # - "revision_complete", "revision_failed"
        # - "review_failed"
        # Final:
        # - "workflow_complete", "workflow_failed"
    
    error: NotRequired[str]  # 에러 발생 시 에러 메시지
    
    # ===== Metadata =====
    created_at: NotRequired[str]  # 워크플로우 시작 시간 (ISO format)
    updated_at: NotRequired[str]  # 마지막 업데이트 시간 (ISO format)


# Helper function to initialize state
def create_initial_state(user_input: str) -> PipelineState:
    """
    초기 State 생성
    
    Args:
        user_input: 사용자 입력 주제
    
    Returns:
        초기화된 PipelineState
    """
    from datetime import datetime
    
    return PipelineState(
        user_input=user_input,
        status="initialized",
        retry_count=0,
        max_retries=3,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )


# Type hints for common state operations
StateUpdate = Dict[str, Any]  # Partial state update