"""
Writer Agent (ReAct-based)

모든 섹션들을 하나의 완성된 마크다운 보고서로 조립하고,
사용자 피드백을 받아 Agent가 자율적으로 Tool을 선택하여 처리
"""

from typing import List, Any, Dict, Optional
from datetime import datetime
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState, WorkflowStatus


class WriterConstants:
    """Constants for Writer Agent"""
    MAX_AGENT_ITERATIONS = 10
    MAX_EXECUTION_TIME = 600  # 10 minutes
    SENTIMENT_TEMPERATURE = 0.3

    # Section prefixes
    SECTION_2_PREFIX = "section_2_"
    SECTION_3_PREFIX = "section_3_"
    SECTION_4_PREFIX = "section_4_"
    SECTION_5_PREFIX = "section_5_"

    # Total sections
    TOTAL_SECTIONS = 6


class FeedbackSentiment(str, Enum):
    """Feedback sentiment types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"


class WriterAgent(BaseAgent):
    """Writer Agent (ReAct-based with Tools)"""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Any],
        config: AgentConfig
    ):
        """
        Initialize WriterAgent

        Args:
            llm: Language model
            tools: Agent tools (RevisionTool, RecollectionTool)
            config: Agent configuration
        """
        super().__init__(llm, tools, config)
        self._react_agent: Any = None
        self._agent_executor: Any = None

        # Setup ReAct Agent
        if tools:
            self._setup_react_agent()

    def _setup_react_agent(self) -> None:
        """Setup ReAct Agent with tools"""
        self._react_agent = self._create_react_agent()
        self._agent_executor = AgentExecutor(
            agent=self._react_agent,
            tools=self.tools,
            verbose=True,
            max_iterations=WriterConstants.MAX_AGENT_ITERATIONS,
            max_execution_time=WriterConstants.MAX_EXECUTION_TIME,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def _create_react_agent(self):
        """
        ReAct Agent 생성

        Agent가 자율적으로 tool을 선택하여 작업 수행
        """
        # langchain-hub 의존성을 제거하고, 표준 ReAct 프롬프트를 직접 정의합니다.
        # 이 프롬프트는 'hwchase17/react-chat'에서 사용하는 것과 동일한 구조입니다.
        template = """Assistant is a large language model trained by Google.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text in response to a wide range of prompts and questions, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand vast amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

TOOLS:
------
Assistant has access to the following tools:
{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
```

When you have a response to say to the human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {input}
{agent_scratchpad}"""

        react_prompt = PromptTemplate.from_template(template)

        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        최종 보고서 작성 및 사용자 피드백 처리

        Agent가 자율적으로 Tool을 선택하여 처리
        """
        print(f"\n{'='*60}")
        print(f"✍️  Writer Agent (ReAct)")
        print(f"{'='*60}\n")

        try:
            # Step 1: 영문 보고서 조립 (또는 기존 보고서 사용)
            if state.get("revision_count", 0) > 0 and state.get("final_report"):
                final_report = state["final_report"]
                print("\n📝 Using previously revised English report for re-review.")
            else:
                final_report = self._assemble_report(state)
                state["final_report"] = final_report
                state["report_generated_at"] = datetime.now().isoformat()

            # Step 2: 한국어로 번역 (CLI 리뷰용)
            print("\n🌐 Translating report to Korean for user review...")
            korean_report_for_review = await self._translate_to_korean(final_report)
            print("✅ Translation for review complete.\n")

            # Step 3: 사용자에게 한국어 보고서 표시
            print("\n" + "="*60)
            print("📄 Final Report Draft (Korean)")
            print("="*60 + "\n")

            print(korean_report_for_review)

            print("\n" + "="*60)
            print("👤 Please review the full report above")
            print("="*60 + "\n")

            # Step 4: 사용자 피드백 받기
            print("💬 Your feedback:")
            print("   - Type 'ok', 'accept', 'good', 'approve' to accept the report")
            print("   - Or provide specific feedback for improvements")
            print()

            feedback = input("Your feedback: ").strip()

            # 빈 피드백 - 승인으로 간주
            if not feedback:
                print("\n⚠️  No feedback provided. Treating as acceptance.")
                state["status"] = WorkflowStatus.COMPLETED.value
                state["review_feedback"] = None
                return state

            # LLM을 사용해 피드백 감정 평가
            print(f"\n🤖 Evaluating feedback sentiment...\n")

            sentiment = await self._evaluate_feedback_sentiment(feedback)

            if sentiment == FeedbackSentiment.POSITIVE.value:
                # 사용자 만족 - 보고서 승인
                print("\n✅ Feedback indicates satisfaction - Report accepted!")
                state["status"] = WorkflowStatus.COMPLETED.value
                state["review_feedback"] = None
                return state

            # 부정적 피드백 - ReAct Agent에게 처리 위임
            print(f"\n📝 Received feedback: {feedback[:150]}...")
            print(f"🤖 ReAct Agent will analyze and choose tool...\n")

            # Agent Executor 실행 - Agent가 도구 선택
            if self._agent_executor:
                agent_input = f"""Analyze the following user feedback and decide which tool to use.

User Feedback: "{feedback}"

Choose the appropriate tool:
- If feedback is about WRITING style/tone/clarity → use revise_report
- If feedback mentions MISSING data/companies/topics → use recollect_data

Provide a brief reason for your choice."""

                # Agent 실행 - Tool 자동 호출
                result = await self._agent_executor.ainvoke({"input": agent_input})

                # Agent가 내린 최종 결론(output)과 중간 단계(intermediate_steps)를 가져옴
                output = result.get("output", "No output from agent.")
                intermediate_steps = result.get("intermediate_steps", [])

                print(f"\n🤖 Agent Decision: {output}\n")

                # 중간 단계에서 어떤 Tool이 사용되었는지 직접 확인하여 분기 (가장 안정적인 방법)
                action_taken = None
                if intermediate_steps:
                    # 마지막으로 실행된 AgentAction을 가져옵니다.
                    # intermediate_steps는 [(AgentAction, tool_output), ...] 형태의 리스트입니다.
                    last_step = intermediate_steps[-1]
                    action_taken = last_step[0]

                if action_taken and action_taken.tool == "revise_report":
                    # RevisionTool 선택됨 - 실제 revision 수행
                    print(f"✅ Agent chose RevisionTool - Performing revision...")
                    
                    revised_report = await self._perform_revision(final_report, feedback)

                    state["final_report"] = revised_report
                    state["status"] = WorkflowStatus.REVISION_COMPLETE.value
                    state["review_feedback"] = None
                    state["revision_count"] = state.get("revision_count", 0) + 1

                    print(f"✅ Revision complete (revision #{state['revision_count']})")

                elif action_taken and action_taken.tool == "recollect_data":
                    # RecollectionTool 선택됨 - graph가 data_collection으로 라우팅
                    print(f"✅ Agent chose RecollectionTool - Routing to data_collection_agent")

                    state["status"] = WorkflowStatus.NEEDS_RECOLLECTION.value
                    state["review_feedback"] = feedback

                    print(f"🔄 Workflow will route back to data collection")

                else:
                    # 불명확한 출력, 기본적으로 revision
                    print(f"⚠️  Unclear agent output, defaulting to revision")

                    revised_report = await self._perform_revision(final_report, feedback)

                    state["final_report"] = revised_report
                    state["status"] = WorkflowStatus.REVISION_COMPLETE.value
                    state["review_feedback"] = None
                    state["revision_count"] = state.get("revision_count", 0) + 1
            else:
                # No agent executor, default to revision
                print(f"⚠️  No agent executor, performing revision")

                revised_report = await self._perform_revision(final_report, feedback)

                state["final_report"] = revised_report
                state["status"] = WorkflowStatus.REVISION_COMPLETE.value
                state["review_feedback"] = None
                state["revision_count"] = state.get("revision_count", 0) + 1

            return state

        except Exception as e:
            print(f"❌ Error in WriterAgent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    async def _perform_revision(self, current_report: str, user_feedback: str) -> str:
        """
        실제 revision 수행 (LLM 사용)

        Args:
            current_report: 현재 영문 보고서
            user_feedback: 사용자 피드백

        Returns:
            수정된 영문 보고서
        """
        print(f"\n✏️  Revising report based on feedback...")

        try:
            system_prompt = """You are an expert technical writer and editor specializing in AI-Robotics industry reports.

Your task is to revise the WRITING STYLE, TONE, and EXPRESSIONS of an English report based on user feedback.

**CRITICAL CONSTRAINT**:
- DO NOT add new data, companies, or technologies
- DO NOT collect or insert new information
- ONLY improve the writing style, tone, clarity, and structure using EXISTING content

Key responsibilities:
1. Carefully analyze the user's feedback
2. Improve writing style and expressions
3. Enhance clarity and readability
4. Reorganize content if needed
5. Maintain all factual information and citations

What you CAN change:
- Writing style and tone
- Sentence structure and phrasing
- Paragraph organization
- Clarity of explanations
- Logical flow

What you CANNOT change:
- Add new data or facts
- Remove or change citations
- Add companies/technologies not in the original
- Change technical accuracy

Output requirements:
- Return the COMPLETE revised report in markdown format
- Keep all sections that don't need changes as-is
- Maintain all markdown formatting
- Keep all citation numbers [1], [2], etc. intact
- Use ONLY the existing data and information"""

            user_prompt = f"""Revise the writing style and expressions of the following English report based on user feedback.

**IMPORTANT**: DO NOT add new data. Only improve the writing using existing content.

Current Report:
```markdown
{current_report[:30000]}
```

User Feedback:
"{user_feedback}"

Instructions:
1. Read the user feedback carefully
2. Identify which aspects of WRITING need improvement
3. Revise the writing style, tone, and expressions
4. Return the COMPLETE revised report in markdown format

CRITICAL RULES:
- Use ONLY existing data and information from the current report
- DO NOT add new companies, technologies, or facts
- DO NOT collect or insert new data
- ONLY improve phrasing, clarity, structure, and tone
- Keep all citation numbers [1], [2], etc. exactly as they are

Output the complete revised report below:"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            revised_report = response.content

            print(f"✅ Revision complete")
            print(f"   Original length: {len(current_report)} chars")
            print(f"   Revised length: {len(revised_report)} chars\n")

            return revised_report

        except Exception as e:
            print(f"❌ Revision failed: {e}")
            # Return original on error
            return current_report

    def _assemble_report(self, state: PipelineState) -> str:
        """
        최종 보고서 조립 (영문)
        """
        print("📝 Assembling final report...\n")

        # Get all components
        topic = state.get("user_input", "AI-Robotics Trend Analysis")
        summary = state.get("summary", "")
        section_1 = state.get("section_1", "")
        sections = state.get("sections", {})
        section_6 = state.get("section_6", "")
        references = state.get("references", "")
        appendix = state.get("appendix", "")

        report_parts = []

        # Title and metadata
        report_parts.append(self._generate_title(topic))
        report_parts.append("")

        # Executive Summary
        if summary:
            report_parts.append("## SUMMARY")
            report_parts.append("")
            report_parts.append(summary)
            report_parts.append("")

        # Section 1: Introduction
        if section_1:
            report_parts.append(section_1)
            report_parts.append("")

        # Section 2: Technology Trend Analysis
        section_2 = self._assemble_section(sections, "section_2", "2. AI-Robotics Technology Trend Analysis")
        if section_2:
            report_parts.append(section_2)
            report_parts.append("")

        # Section 3: Market Trends & Applications
        section_3 = self._assemble_section(sections, "section_3", "3. Market Trends & Applications")
        if section_3:
            report_parts.append(section_3)
            report_parts.append("")

        # Section 4: 5-Year Forecast
        section_4 = self._assemble_section(sections, "section_4", "4. 5-Year Forecast (2025-2030)")
        if section_4:
            report_parts.append(section_4)
            report_parts.append("")

        # Section 5: Implications for Business
        section_5 = self._assemble_section(sections, "section_5", "5. Implications for Business")
        if section_5:
            report_parts.append(section_5)
            report_parts.append("")

        # Section 6: Conclusion
        if section_6:
            report_parts.append(section_6)
            report_parts.append("")

        # References
        if references:
            report_parts.append(references)
            report_parts.append("")

        # Appendix
        if appendix:
            report_parts.append(appendix)
            report_parts.append("")

        # Combine all parts
        final_report = "\n".join(report_parts)

        # Statistics
        word_count = len(final_report.split())
        char_count = len(final_report)
        section_count = len([s for s in sections.keys()]) + (WriterConstants.TOTAL_SECTIONS - 4)

        print(f"✅ Final Report Assembled!")
        print(f"\n📊 Report Statistics:")
        print(f"   - Total Words: {word_count:,}")
        print(f"   - Total Characters: {char_count:,}")
        print(f"   - Sections: {section_count}")
        print(f"   - Citations: {len(state.get('citations', []))}")
        print(f"\n📄 Report Structure:")
        print(f"   ✓ SUMMARY")
        print(f"   ✓ 1. Introduction")
        print(f"   ✓ 2. Technology Trend Analysis ({self._count_subsections(sections, 'section_2')} subsections)")
        print(f"   ✓ 3. Market Trends & Applications ({self._count_subsections(sections, 'section_3')} subsections)")
        print(f"   ✓ 4. 5-Year Forecast ({self._count_subsections(sections, 'section_4')} subsections)")
        print(f"   ✓ 5. Implications for Business ({self._count_subsections(sections, 'section_5')} subsections)")
        print(f"   ✓ 6. Conclusion")
        print(f"   ✓ REFERENCE")
        print(f"   ✓ APPENDIX\n")

        return final_report

    async def _evaluate_feedback_sentiment(self, feedback: str) -> str:
        """
        LLM을 사용해 사용자 피드백의 감정 평가

        Args:
            feedback: User feedback text

        Returns:
            FeedbackSentiment value ("positive" or "negative")
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a feedback sentiment analyzer.

Analyze the user's feedback and determine if it indicates:
1. **{FeedbackSentiment.POSITIVE.value}** - User is satisfied and accepts the report
   - Examples: "good", "ok", "looks great", "approve", "nice work", "좋아", "승인", "완료"

2. **{FeedbackSentiment.NEGATIVE.value}** - User wants changes or is not satisfied
   - Examples: "change this", "add more", "fix", "revise", "missing", "needs improvement"

Respond with ONLY ONE WORD: "{FeedbackSentiment.POSITIVE.value}" or "{FeedbackSentiment.NEGATIVE.value}"
"""),
                ("user", f"""User feedback: "{{feedback}}"

Is this {FeedbackSentiment.POSITIVE.value} (accept) or {FeedbackSentiment.NEGATIVE.value} (needs changes)?
Respond with only: {FeedbackSentiment.POSITIVE.value} or {FeedbackSentiment.NEGATIVE.value}""")
            ])

            chain = prompt | self.llm
            response = await chain.ainvoke({"feedback": feedback})

            if hasattr(response, 'content'):
                result = response.content.strip().lower()
            else:
                result = str(response).strip().lower()

            # Extract positive/negative
            if FeedbackSentiment.POSITIVE.value in result:
                return FeedbackSentiment.POSITIVE.value
            elif FeedbackSentiment.NEGATIVE.value in result:
                return FeedbackSentiment.NEGATIVE.value
            else:
                # Default to negative (safer - allows user to provide more feedback)
                return FeedbackSentiment.NEGATIVE.value

        except Exception as e:
            print(f"⚠️  Sentiment evaluation error: {e}")
            # Default to negative (safer)
            return FeedbackSentiment.NEGATIVE.value

    def _generate_title(self, topic: str) -> str:
        """보고서 제목 생성"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return "\n".join([
            f"# {topic.title()}",
            "",
            f"**AI-Robotics Industry Trend Analysis Report**",
            "",
            f"*Generated: {current_date}*",
            "",
            "---",
            ""
        ])

    def _assemble_section(
        self,
        sections: Dict[str, str],
        section_prefix: str,
        section_title: str
    ) -> str:
        """서브섹션 조립"""
        subsections = {k: v for k, v in sections.items() if k.startswith(section_prefix)}

        if not subsections:
            return ""

        sorted_subsections = sorted(subsections.items(), key=lambda x: x[0])
        result = [f"## {section_title}", ""]

        for key, content in sorted_subsections:
            result.append(content)
            result.append("")

        return "\n".join(result)

    def _count_subsections(self, sections: Dict[str, str], section_prefix: str) -> int:
        """서브섹션 개수"""
        return len([k for k in sections.keys() if k.startswith(section_prefix)])

    async def _translate_to_korean(self, english_report: str) -> str:
        """
        (CLI 리뷰용) 영문 보고서를 한국어로 번역. 에러 발생 시 재시도.
        """
        sections = english_report.split("\n## ")
        translated_sections = []

        max_section_retries = 3
        for i, section in enumerate(sections):
            if i == 0:
                chunk = section
            else:
                chunk = "## " + section

            if len(chunk.strip()) < 10:
                translated_sections.append(chunk)
                continue

            for attempt in range(max_section_retries):
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a professional Korean translator specializing in technical and business documents.

Translate the following English markdown report to Korean while:
1. Maintaining all markdown formatting (headers, lists, bold, italic, etc.)
2. Keeping citation numbers [1], [2], etc. as-is
3. Preserving technical terms when appropriate (e.g., AI, IoT, robotics)
4. Using natural, professional Korean business language
5. Keeping the document structure exactly the same

Output ONLY the translated Korean markdown, nothing else."""),
                        ("user", "{text}")
                    ])

                    chain = prompt | self.llm
                    response = await chain.ainvoke({"text": chunk})

                    translated = response.content if hasattr(response, 'content') else str(response)
                    translated_sections.append(translated.strip())
                    break

                except Exception as e:
                    print(f"  ❌ Translation error for section {i+1} (Attempt {attempt + 1}/{max_section_retries}): {e}")
                    if attempt < max_section_retries - 1:
                        await asyncio.sleep(2)
                    else:
                        print(f"  ❌ All retries failed for section {i+1}. Using original English text for this section.")
                        translated_sections.append(chunk)  # Fallback to English

        return "\n\n".join(translated_sections)
