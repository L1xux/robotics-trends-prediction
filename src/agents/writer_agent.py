"""
Writer Agent (ReAct-based) - English Only Version

Assembles all sections into a complete Markdown report and handles user feedback 
using an Agent that autonomously selects tools (Revision/Recollection).
"""

from typing import List, Any, Dict, Optional
from datetime import datetime
from enum import Enum
import textwrap 

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
        Create ReAct Agent with EXPLICIT NEWLINES to prevent formatting issues.
        """
        # [Fix] Explicitly define format instructions to ensure correct line breaks
        format_instructions = (
            "To use a tool, please use the following format:\n\n"
            "```\n"
            "Thought: Do I need to use a tool? Yes\n"
            "Action: the tool to use, should be one of [{tool_names}]\n"
            "Action Input: the input to the tool\n"
            "Observation: the result of the tool\n"
            "```\n\n"
            "When you have a response to say to the human, or if you do not need to use a tool, you MUST use the format:\n\n"
            "```\n"
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: [your response here]\n"
            "```"
        )

        template = f"""Assistant is a large language model trained by Google.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text in response to a wide range of prompts and questions, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand vast amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

TOOLS:
------
Assistant has access to the following tools:
{{tools}}

{format_instructions}

Begin!

New input: {{input}}
{{agent_scratchpad}}"""

        react_prompt = PromptTemplate.from_template(template)

        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Assemble final report and handle user feedback (English Only)
        """
        print(f"\n{'='*60}")
        print(f"Writer Agent (ReAct - English Only)")
        print(f"{'='*60}\n")

        try:
            # Step 1: Assemble English Report (or use existing)
            if state.get("revision_count", 0) > 0 and state.get("final_report"):
                final_report = state["final_report"]
                print("\nUsing previously revised report for re-review.")
            else:
                final_report = self._assemble_report(state)
                state["final_report"] = final_report
                state["report_generated_at"] = datetime.now().isoformat()

            # Step 2: Show English Report to User
            print("\n" + "="*60)
            print("Final Report Draft")
            print("="*60 + "\n")

            print(final_report)

            print("\n" + "="*60)
            print("Please review the full report above")
            print("="*60 + "\n")

            # Step 3: Get User Feedback
            print("Your feedback:")
            print("   - Type 'ok', 'accept', 'good', 'approve' to accept the report")
            print("   - Or provide specific feedback for improvements")
            print()

            feedback = input("Your feedback: ").strip()

            # Empty feedback -> Treat as acceptance
            if not feedback:
                print("\nNo feedback provided. Treating as acceptance.")
                state["status"] = WorkflowStatus.COMPLETED.value
                state["review_feedback"] = None
                return state

            # Evaluate Sentiment
            print(f"\nEvaluating feedback sentiment...\n")

            sentiment = await self._evaluate_feedback_sentiment(feedback)

            if sentiment == FeedbackSentiment.POSITIVE.value:
                # User satisfied -> Accept Report
                print("\nFeedback indicates satisfaction - Report accepted!")
                state["status"] = WorkflowStatus.COMPLETED.value
                state["review_feedback"] = None
                return state

            # Negative Feedback -> Delegate to ReAct Agent
            print(f"\nReceived feedback: {feedback[:150]}...")
            print(f"ReAct Agent will analyze and choose tool...\n")

            if self._agent_executor:
                agent_input = f"""Analyze the following user feedback and decide which tool to use.

User Feedback: "{feedback}"

Choose the appropriate tool:
- If feedback is about WRITING style/tone/clarity → use revise_report
- If feedback mentions MISSING data/companies/topics → use recollect_data

Provide a brief reason for your choice."""

                # Run Agent -> Select Tool
                result = await self._agent_executor.ainvoke({"input": agent_input})

                output = result.get("output", "No output from agent.")
                intermediate_steps = result.get("intermediate_steps", [])

                print(f"\nAgent Decision: {output}\n")

                # Check which tool was actually used
                action_taken = None
                if intermediate_steps:
                    last_step = intermediate_steps[-1]
                    action_taken = last_step[0]

                if action_taken and action_taken.tool == "revise_report":
                    # RevisionTool Selected
                    print(f"Agent chose RevisionTool - Performing revision...")
                    
                    revised_report = await self._perform_revision(final_report, feedback)

                    state["final_report"] = revised_report
                    state["status"] = WorkflowStatus.REVISION_COMPLETE.value
                    state["review_feedback"] = None
                    state["revision_count"] = state.get("revision_count", 0) + 1

                    print(f"Revision complete (revision #{state['revision_count']})")

                elif action_taken and action_taken.tool == "recollect_data":
                    # RecollectionTool Selected
                    print(f"Agent chose RecollectionTool - Routing to data_collection_agent")

                    state["status"] = WorkflowStatus.NEEDS_RECOLLECTION.value
                    state["review_feedback"] = feedback

                    print(f"Workflow will route back to data collection")

                else:
                    # Unclear or Default -> Revision
                    print(f"Unclear agent output, defaulting to revision")

                    revised_report = await self._perform_revision(final_report, feedback)

                    state["final_report"] = revised_report
                    state["status"] = WorkflowStatus.REVISION_COMPLETE.value
                    state["review_feedback"] = None
                    state["revision_count"] = state.get("revision_count", 0) + 1
            else:
                # No Executor -> Default Revision
                print(f"No agent executor, performing revision")

                revised_report = await self._perform_revision(final_report, feedback)

                state["final_report"] = revised_report
                state["status"] = WorkflowStatus.REVISION_COMPLETE.value
                state["review_feedback"] = None
                state["revision_count"] = state.get("revision_count", 0) + 1

            return state

        except Exception as e:
            print(f"Error in WriterAgent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    async def _perform_revision(self, current_report: str, user_feedback: str) -> str:
        """
        Perform actual revision using LLM
        """
        print(f"\nRevising report based on feedback...")

        try:
            # [Fix] Indentation cleanup using textwrap to prevent whitespace issues in LLM prompt
            system_prompt = textwrap.dedent("""\
                You are an expert technical writer and editor specializing in AI-Robotics industry reports.

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

                Output requirements:
                - Return the COMPLETE revised report in markdown format
                - Keep all sections that don't need changes as-is
                - Maintain all markdown formatting
                - Keep all citation numbers [1], [2], etc. intact
                - Use ONLY the existing data and information""")

            user_prompt = f"""Revise the writing style and expressions of the following English report based on user feedback.

            **IMPORTANT**: DO NOT add new data. Only improve the writing using existing content.

            Current Report:
            ```markdown
            {current_report[:30000]}
            User Feedback: "{user_feedback}"

            Instructions:

            Read the user feedback carefully

            Identify which aspects of WRITING need improvement

            Revise the writing style, tone, and expressions

            Return the COMPLETE revised report in markdown format

            Output the complete revised report below:"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            revised_report = response.content

            print(f"Revision complete")
            print(f"   Original length: {len(current_report)} chars")
            print(f"   Revised length: {len(revised_report)} chars\n")

            return revised_report

        except Exception as e:
            print(f"Revision failed: {e}")
            # Return original on error
            return current_report

    def _assemble_report(self, state: PipelineState) -> str:
        """
        Assemble final report (English)
        """
        print("Assembling final report...\n")

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

        print(f"Final Report Assembled!")
        print(f"\nReport Statistics:")
        print(f"   - Total Words: {word_count:,}")
        print(f"   - Total Characters: {char_count:,}")
        print(f"   - Sections: {section_count}")
        print(f"   - Citations: {len(state.get('citations', []))}")
        print(f"\nReport Structure:")
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
        Evaluate feedback sentiment using LLM
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a feedback sentiment analyzer.
    Analyze the user's feedback and determine if it indicates:

    {FeedbackSentiment.POSITIVE.value} - User is satisfied and accepts the report

    Examples: "good", "ok", "looks great", "approve", "nice work", "yes"

    {FeedbackSentiment.NEGATIVE.value} - User wants changes or is not satisfied

    Examples: "change this", "add more", "fix", "revise", "missing", "needs improvement"

    Respond with ONLY ONE WORD: "{FeedbackSentiment.POSITIVE.value}" or "{FeedbackSentiment.NEGATIVE.value}" """), ("user", f"""User feedback: "{{feedback}}"

    Is this {FeedbackSentiment.POSITIVE.value} (accept) or {FeedbackSentiment.NEGATIVE.value} (needs changes)? Respond with only: {FeedbackSentiment.POSITIVE.value} or {FeedbackSentiment.NEGATIVE.value}""") ])

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
                return FeedbackSentiment.NEGATIVE.value

        except Exception as e:
            print(f"Sentiment evaluation error: {e}")
            return FeedbackSentiment.NEGATIVE.value

    def _generate_title(self, topic: str) -> str:
        """Generate Report Title"""
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
        """Assemble subsections"""
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
        """Count subsections"""
        return len([k for k in sections.keys() if k.startswith(section_prefix)])