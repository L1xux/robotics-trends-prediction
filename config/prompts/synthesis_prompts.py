"""
Report Synthesis Agent 프롬프트 모음
"""



SYNTHESIS_PROMPTS = {
    "report_synthesis": """
당신은 AI 로보틱스 트렌드 보고서를 작성하는 전문 작가입니다.

분석 결과:
- 기술 분석: {technical_analysis}
- 시장 분석: {market_analysis}
- 연구 동향: {research_trends}
- 종합 분석: {comprehensive_analysis}

보고서 계획:
{report_plan}

다음 구조로 전문적인 보고서를 작성하세요:

# {report_title}

## Executive Summary
- 핵심 발견사항 요약
- 주요 트렌드 하이라이트
- 미래 전망 개요

## 1. 기술 트렌드 분석
- 핵심 기술 동향
- 기술 발전사항
- 새로운 기술 등장

## 2. 시장 동향 분석
- 시장 규모 및 성장률
- 주요 플레이어 동향
- 시장 트렌드

## 3. 연구 동향 분석
- 활발한 연구 분야
- 연구 방법론 변화
- 학술적 기여도

## 4. 종합 분석 및 미래 전망
- 핵심 발견사항
- 상호 연관성 분석
- 미래 전망 (단기/중기/장기)
- 리스크 및 기회

## 결론 및 권고사항
- 주요 결론
- 산업계 권고사항
- 연구자 권고사항

보고서는 다음 특징을 가져야 합니다:
- 전문적이고 객관적인 톤
- 데이터 기반의 분석
- 명확한 구조와 흐름
- 실행 가능한 인사이트
""",

    "section_refinement": """
특정 섹션을 개선하세요:

섹션명: {section_name}
현재 내용: {current_content}
개선 요청: {improvement_request}

개선된 섹션 내용을 반환하세요.
""",

    "executive_summary": """
Executive Summary를 작성하세요:

전체 분석 결과: {full_analysis}
핵심 트렌드: {key_trends}
미래 전망: {future_outlook}

다음 요소를 포함한 Executive Summary를 작성하세요:
- 핵심 발견사항 3-5개
- 가장 중요한 트렌드 하이라이트
- 미래 전망 요약
- 주요 권고사항

길이: 300-500단어
톤: 간결하고 임팩트 있게
"""
}
