"""
Writer Agent (ReAct-based)

모든 섹션들을 하나의 완성된 마크다운 보고서로 조립하고,
사용자 피드백을 받아 Agent가 자율적으로 Tool을 선택하여 처리
"""

from typing import List, Any, Dict
from datetime import datetime
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor
import asyncio

from src.agents.base.base_agent import BaseAgent
from src.agents.base.agent_config import AgentConfig
from src.graph.state import PipelineState

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
            tools: [TranslateTool, RevisionTool, RecollectTool]
            config: Agent configuration
        """
        super().__init__(llm, tools, config)
        
        # ReAct Agent 생성
        if tools:
            self.react_agent = self._create_react_agent()
            self.agent_executor = AgentExecutor(
                agent=self.react_agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                max_execution_time=600,  # 10분
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
        else:
            self.react_agent = None
            self.agent_executor = None
    
    def _create_react_agent(self):
        """
        ReAct Agent 생성
        
        Agent가 자율적으로 tool을 선택하여 작업 수행
        """
        react_prompt = PromptTemplate.from_template("""You are a Writer Agent for a research report generation system.

Your task is to:
1. Assemble the report from all sections
2. Translate to Korean using TranslateTool
3. Show the report to user and get feedback
4. Based on feedback, decide which tool to use:
   - If user accepts → Done
   - If minor changes needed → Use RevisionTool
   - If data is missing → Use RecollectTool

Available Tools:
{tools}

Tool Names: {tool_names}

Use the following format:

Question: the task you must complete
Thought: think about what to do next
Action: the tool to use (must be one of [{tool_names}])
Action Input: the input to the tool
Observation: the tool's output
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the result

Begin!

Question: {input}
{agent_scratchpad}""")
        
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
            # Determine the English report to work with
            # If a revision has occurred, use the revised final_report directly.
            # Otherwise, assemble it from individual sections.
            if state.get("revision_count", 0) > 0 and state.get("final_report"):
                final_report = state["final_report"]
                print("\n📝 Using previously revised English report for re-review.")
            else:
                final_report = self._assemble_report(state)
                state["final_report"] = final_report
                state["report_generated_at"] = datetime.now().isoformat()
            
            # Step 2: Translate to Korean for CLI review
            print("\n🌐 Translating report to Korean for user review...")
            korean_report_for_review = await self._translate_to_korean(final_report)
            print("✅ Translation for review complete.\n")

            # Step 3: Show FULL Korean report to user for review
            print("\n" + "="*60)
            print("📄 Final Report Draft (Korean)")
            print("="*60 + "\n")
            
            # 전체 한국어 보고서 출력
            print(korean_report_for_review)
            
            print("\n" + "="*60)
            print("👤 Please review the full report above")
            print("="*60 + "\n")
            
            # Step 4: Get user feedback (on the Korean version)
            print("💬 Your feedback:")
            print("   - Type 'ok', 'accept', 'good', 'approve' to accept the report")
            print("   - Or provide specific feedback for improvements")
            print()
            
            feedback = input("Your feedback: ").strip()
            
            # Empty feedback - treat as accept
            if not feedback:
                print("\n⚠️  No feedback provided. Treating as acceptance.")
                state["status"] = "completed"
                state["review_feedback"] = None
                return state
            
            # Use LLM to evaluate if feedback is positive (accept) or negative (needs changes)
            print(f"\n🤖 Evaluating feedback sentiment...\n")
            
            sentiment = await self._evaluate_feedback_sentiment(feedback)
            
            if sentiment == "positive":
                # User is satisfied - accept report
                print("\n✅ Feedback indicates satisfaction - Report accepted!")
                state["status"] = "completed"
                state["review_feedback"] = None
                return state
            
            # Negative sentiment - User wants changes
            # Let ReAct Agent handle it with TOOLS
            print(f"\n📝 Received feedback: {feedback[:150]}...")
            print(f"🤖 ReAct Agent will decide and use appropriate tool...\n")
            
            # Use Agent Executor - Agent will decide which tool to use
            if self.agent_executor:
                agent_input = f"""The user provided feedback on the final report.

User Feedback: "{feedback}"

**IMPORTANT**: You must use the original English report for revisions, even though the user is reviewing the Korean version.

Analyze the feedback and decide what to do:
1. If user wants content changes (rephrasing, adding details, fixing errors) → Use the `revise_report` tool. You **MUST** pass the full original English report to the `current_report` argument.
2. If user mentions missing data/topics/companies → Use the `recollect_data` tool.

Use the appropriate tool with the user feedback and the **full original English report** which is provided below.

Original English Report:
{final_report}"""

                # Run agent - Agent will call tools automatically
                result = await self.agent_executor.ainvoke({"input": agent_input})
                
                # Parse tool output
                output = result.get("output", "")
                
                print(f"\n🤖 Agent Output: {output[:200]}...")
                
                # Check what tool was used
                if "REVISION_COMPLETED" in output or "REVISION_ERROR" in output:
                    # RevisionTool was called and completed
                    print(f"\n✅ RevisionTool executed - Report revised. Looping back to Writer for re-review.")
                    state["status"] = "revision_completed"  # Will loop back to writer for re-review
                    state["review_feedback"] = None
                    
                elif "RECOLLECT_REQUESTED" in output:
                    # RecollectTool was called
                    print(f"\n✅ RecollectTool executed - Data recollection requested")
                    state["status"] = "needs_recollection"
                    state["review_feedback"] = feedback
                    
                    # Try to extract keywords from output
                    import re
                    keywords_match = re.search(r'Additional keywords: (.*?)\.', output)
                    if keywords_match:
                        keywords_str = keywords_match.group(1).strip()
                        if keywords_str and keywords_str != 'None':
                            state["additional_keywords"] = [k.strip() for k in keywords_str.split(',')]
                        else:
                            state["additional_keywords"] = []
                    else:
                        state["additional_keywords"] = []
                
                else:
                    # No clear tool output, default to revision
                    print(f"\n⚠️  Unclear agent output, defaulting to revision")
                    state["status"] = "needs_revision"
                    state["review_feedback"] = feedback
                    state["revision_type"] = "minor"
            else:
                # No agent executor, default to revision
                state["status"] = "needs_revision"
                state["review_feedback"] = feedback
                state["revision_type"] = "minor"
            
            return state
        
        except Exception as e:
            print(f"❌ Error in WriterAgent: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
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
        section_count = len([s for s in sections.keys()]) + 2
        
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
            "positive" (만족/승인) or "negative" (불만족/수정 요청)
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a feedback sentiment analyzer.

Analyze the user's feedback and determine if it indicates:
1. **positive** - User is satisfied and accepts the report
   - Examples: "good", "ok", "looks great", "approve", "nice work", "좋아", "승인", "완료"
   
2. **negative** - User wants changes or is not satisfied
   - Examples: "change this", "add more", "fix", "revise", "missing", "needs improvement"

Respond with ONLY ONE WORD: "positive" or "negative"
"""),
                ("user", """User feedback: "{feedback}"

Is this positive (accept) or negative (needs changes)?
Respond with only: positive or negative""")
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({"feedback": feedback})
            
            if hasattr(response, 'content'):
                result = response.content.strip().lower()
            else:
                result = str(response).strip().lower()
            
            # Extract positive/negative
            if "positive" in result:
                return "positive"
            elif "negative" in result:
                return "negative"
            else:
                # Default to negative (safer - allows user to provide more feedback)
                return "negative"
        
        except Exception as e:
            print(f"⚠️  Sentiment evaluation error: {e}")
            # Default to negative (safer)
            return "negative"
    
    async def _classify_feedback(
        self,
        feedback: str,
        report_content: str,
        state: PipelineState
    ) -> Dict[str, Any]:
        """
        LLM을 사용해 사용자 피드백 분석
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a feedback analyzer for a research report generation system.

Analyze the user's feedback and determine the appropriate action:

1. **revision**: Simple content improvements, rephrasing, adding details, fixing errors
   - Use when feedback asks for better explanation, more detail, reorganization

2. **recollect**: Need to collect different or additional data
   - Use ONLY when feedback explicitly mentions missing companies, technologies, or data sources
   - Extract additional keywords if applicable

Respond in JSON format:
{{
    "recommended_action": "revision" or "recollect",
    "severity": "minor" or "moderate" or "major",
    "reasoning": "Brief explanation",
    "additional_keywords": ["keyword1", "keyword2"]
}}"""),
                ("user", """Topic: {topic}

User Feedback: {feedback}

Analyze and respond:""")
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "topic": state.get("user_input", ""),
                "feedback": feedback
            })
            
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            # Parse JSON
            import json
            import re
            
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
            if json_match:
                result = json_match.group(1)
            
            classification = json.loads(result)
            return classification
        
        except Exception as e:
            print(f"⚠️  Feedback classification error: {e}")
            return {
                "recommended_action": "revision",
                "severity": "minor",
                "reasoning": "Error in classification, defaulting to revision",
                "additional_keywords": []
            }
    
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
                        translated_sections.append(chunk) # Fallback to English for this section
        
        return "\n\n".join(translated_sections)
