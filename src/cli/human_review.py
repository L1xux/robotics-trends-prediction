# src/cli/human_review.py

"""
Human Review CLI

Human-in-the-loop 리뷰를 위한 CLI 인터페이스
"""

from typing import Dict, Any, Optional, Literal
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich import box
from rich.prompt import Prompt
import json

console = Console()


class ReviewCLI:
    """Human Review CLI Interface"""
    
    def __init__(self):
        self.console = console
    
    def display_plan(self, planning_output: Dict[str, Any]) -> None:
        """
        계획을 표시만 함 (피드백은 별도로 받음)
        
        Args:
            planning_output: Planning output dictionary
        """
        self.console.clear()
        
        # 제목
        self.console.print(
            Panel.fit(
                "[bold cyan]📋 Research Planning Output[/bold cyan]",
                border_style="cyan"
            )
        )
        self.console.print()
        
        # Topic & Folder
        self.console.print(
            Panel(
                f"[bold]Topic:[/bold] {planning_output.get('topic', 'N/A')}\n"
                f"[bold]Folder:[/bold] {planning_output.get('folder_name', 'N/A')}",
                title="[bold]Basic Info[/bold]",
                border_style="blue",
                padding=(1, 2)
            )
        )
        self.console.print()
        
        # Keywords
        keywords = planning_output.get('keywords', [])
        keyword_table = Table(
            title=f"🔑 Keywords ({len(keywords)} total)",
            box=box.ROUNDED,
            border_style="green"
        )
        keyword_table.add_column("Index", justify="right", style="dim")
        keyword_table.add_column("Keyword", style="bold green")
        
        for idx, keyword in enumerate(keywords, 1):
            keyword_table.add_row(str(idx), keyword)
        
        self.console.print(keyword_table)
        self.console.print()
        
        # Collection Plan
        collection_plan = planning_output.get('collection_plan', {})
        
        plan_table = Table(
            title="📊 Collection Plan",
            box=box.ROUNDED,
            border_style="yellow"
        )
        plan_table.add_column("Source", style="bold")
        plan_table.add_column("Configuration", style="cyan")
        
        # arXiv
        arxiv = collection_plan.get('arxiv', {})
        arxiv_config = (
            f"Date Range: {arxiv.get('date_range', 'N/A')}\n"
            f"Categories: {arxiv.get('categories', 'N/A')}\n"
            f"Max Results: {arxiv.get('max_results', 'N/A')}"
        )
        plan_table.add_row("arXiv", arxiv_config)
        
        # Google Trends
        trends = collection_plan.get('trends', {})
        trends_config = f"Timeframe: {trends.get('timeframe', 'N/A')}"
        plan_table.add_row("Google Trends", trends_config)
        
        # News
        news = collection_plan.get('news', {})
        news_config = (
            f"Sources: {news.get('sources', 'N/A')}\n"
            f"Date Range: {news.get('date_range', 'N/A')}"
        )
        plan_table.add_row("Tech News", news_config)
        
        self.console.print(plan_table)
        self.console.print()
    
    def display_planning_review(self, planning_output: Dict[str, Any]) -> bool:
        """
        레거시 메서드 (하위 호환성)
        
        Args:
            planning_output: Planning output dictionary
        
        Returns:
            True if accepted, False if rejected
        """
        self.display_plan(planning_output)
        
        # Decision
        self.console.print("[bold yellow]Do you approve this plan?[/bold yellow]")
        
        decision = Prompt.ask(
            "[bold]Your decision[/bold]",
            choices=["yes", "no"],
            default="yes"
        )
        
        return decision.lower() == "yes"
    
    def display_final_review(
        self,
        report_content: str,
        quality_report: Dict[str, Any]
    ) -> tuple[Literal["accept", "revise"], Optional[str]]:
        """
        최종 리포트 리뷰 표시
        
        Args:
            report_content: 전체 리포트 마크다운 내용
            quality_report: QualityReport 딕셔너리
                - overall_score
                - section_scores
                - strengths
                - improvements
        
        Returns:
            tuple: (decision, feedback)
                - decision: "accept" or "revise"
                - feedback: 수정 요청 사항 (revise 시에만)
        """
        self.console.clear()
        
        # 제목
        self.console.print(
            Panel.fit(
                "[bold magenta]📄 Final Report Review[/bold magenta]",
                border_style="magenta"
            )
        )
        self.console.print()
        
        # Quality Score
        score = quality_report['overall_score']
        score_color = "green" if score >= 8.0 else "yellow" if score >= 6.0 else "red"
        
        self.console.print(
            Panel(
                f"[bold {score_color}]{score:.1f} / 10.0[/bold {score_color}]",
                title="Overall Quality Score",
                border_style=score_color
            )
        )
        self.console.print()
        
        # Section Scores
        section_table = Table(
            title="📊 Section Scores",
            box=box.ROUNDED,
            border_style="blue"
        )
        section_table.add_column("Section", style="bold")
        section_table.add_column("Score", justify="center")
        
        for section, sec_score in quality_report['section_scores'].items():
            sec_color = "green" if sec_score >= 8.0 else "yellow" if sec_score >= 6.0 else "red"
            section_table.add_row(
                section,
                f"[{sec_color}]{sec_score:.1f}[/{sec_color}]"
            )
        
        self.console.print(section_table)
        self.console.print()
        
        # Strengths & Improvements
        layout = Layout()
        layout.split_row(
            Layout(name="strengths"),
            Layout(name="improvements")
        )
        
        # Strengths
        strengths_text = "\n".join([f"• {s}" for s in quality_report['strengths']])
        layout["strengths"].update(
            Panel(
                strengths_text,
                title="[bold green]✓ Strengths[/bold green]",
                border_style="green",
                padding=(1, 2)
            )
        )
        
        # Improvements
        improvements_text = "\n".join([f"• {i}" for i in quality_report['improvements']])
        layout["improvements"].update(
            Panel(
                improvements_text,
                title="[bold yellow]⚠ Suggested Improvements[/bold yellow]",
                border_style="yellow",
                padding=(1, 2)
            )
        )
        
        self.console.print(layout)
        self.console.print()
        
        # Report Preview (first 1000 chars)
        preview_length = min(1000, len(report_content))
        preview = report_content[:preview_length]
        if len(report_content) > preview_length:
            preview += "\n\n... (truncated)"
        
        self.console.print(
            Panel(
                preview,
                title="[bold]Report Preview[/bold]",
                border_style="dim",
                padding=(1, 2)
            )
        )
        self.console.print()
        
        # User Decision
        self.console.print("[bold yellow]Review the final report above.[/bold yellow]")
        
        decision = Prompt.ask(
            "[bold]What would you like to do?[/bold]",
            choices=["accept", "revise"],
            default="accept"
        )
        
        feedback = None
        if decision == "revise":
            self.console.print()
            self.console.print("[bold cyan]Please describe what needs to be revised:[/bold cyan]")
            self.console.print("[dim](Press Ctrl+D or Ctrl+Z when done)[/dim]")
            
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            
            feedback = "\n".join(lines).strip()
            
            if not feedback:
                self.console.print("[bold red]Error:[/bold red] Revision feedback cannot be empty.")
                self.console.print("[yellow]Treating as 'accept' instead.[/yellow]")
                decision = "accept"
                feedback = None
            else:
                self.console.print()
                self.console.print("[bold green]✓[/bold green] Revision request received.")
        else:
            self.console.print("[bold green]✓[/bold green] Report accepted! Generating PDF...")
        
        self.console.print()
        return decision, feedback


class ProgressDisplay:
    """Real-time Progress Display"""
    
    def __init__(self):
        self.console = console
        self.current_phase = None
        self.current_agent = None
    
    def show_phase_start(self, phase: str, description: str):
        """새 Phase 시작 표시"""
        self.current_phase = phase
        
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[bold cyan]{phase}[/bold cyan]\n[dim]{description}[/dim]",
                border_style="cyan",
                padding=(1, 2)
            )
        )
    
    def show_agent_start(self, agent_name: str, task: str):
        """Agent 작업 시작"""
        self.current_agent = agent_name
        
        self.console.print(
            f"[bold blue]►[/bold blue] [bold]{agent_name}[/bold]: {task}"
        )
    
    def show_agent_complete(self, agent_name: str, result: str):
        """Agent 작업 완료"""
        self.console.print(
            f"[bold green]✓[/bold green] [bold]{agent_name}[/bold] completed: {result}"
        )
    
    def show_tool_call(self, tool_name: str, parameters: Dict[str, Any]):
        """Tool 호출 표시"""
        self.console.print(
            f"  [dim]→[/dim] Calling [cyan]{tool_name}[/cyan]..."
        )
    
    def show_tool_result(self, tool_name: str, status: str):
        """Tool 결과 표시"""
        status_icon = "✓" if status == "success" else "✗"
        status_color = "green" if status == "success" else "red"
        
        self.console.print(
            f"  [dim]←[/dim] [{status_color}]{status_icon}[/{status_color}] [cyan]{tool_name}[/cyan] {status}"
        )
    
    def show_quality_check(self, attempt: int, max_attempts: int, result: str):
        """Quality Check 결과"""
        if result == "pass":
            self.console.print(
                f"[bold green]✓[/bold green] Quality Check passed (attempt {attempt}/{max_attempts})"
            )
        else:
            self.console.print(
                f"[bold yellow]⟳[/bold yellow] Quality Check retry needed (attempt {attempt}/{max_attempts})"
            )
    
    def show_rag_retrieval(self, query: str, num_results: int):
        """RAG 검색 표시"""
        self.console.print(
            f"  [dim]→[/dim] RAG retrieval: [cyan]{query}[/cyan] ({num_results} docs)"
        )
    
    def show_error(self, error_message: str):
        """에러 표시"""
        self.console.print(
            Panel(
                f"[bold red]Error:[/bold red] {error_message}",
                border_style="red",
                padding=(1, 2)
            )
        )
    
    def show_warning(self, warning_message: str):
        """경고 표시"""
        self.console.print(
            f"[bold yellow]⚠[/bold yellow] {warning_message}"
        )
    
    def show_info(self, info_message: str):
        """정보 표시"""
        self.console.print(
            f"[bold blue]ℹ[/bold blue] {info_message}"
        )


# Convenience Functions
def planning_review(planning_output: Dict[str, Any]) -> bool:
    """Planning Review 단축 함수"""
    cli = ReviewCLI()
    return cli.display_planning_review(planning_output)


def final_review(
    report_content: str,
    quality_report: Dict[str, Any]
) -> tuple[Literal["accept", "revise"], Optional[str]]:
    """Final Review 단축 함수"""
    cli = ReviewCLI()
    return cli.display_final_review(report_content, quality_report)


if __name__ == "__main__":
    # Test Planning Review
    test_planning = {
        "topic": "humanoid robots in manufacturing",
        "folder_name": "humanoid_robots_manufacturing_20250122_143052",
        "keywords": [
            "humanoid robot",
            "manufacturing automation",
            "industrial robot",
            "physical AI",
            "embodied intelligence"
        ],
        "collection_plan": {
            "arxiv": {
                "date_range": "2022-01-01 to 2025-10-22",
                "categories": "all",
                "max_results": "unlimited"
            },
            "trends": {
                "timeframe": "36 months"
            },
            "news": {
                "sources": 5,
                "date_range": "3 years"
            }
        }
    }
    
    console.print("[bold]Testing Planning Display...[/bold]")
    cli = ReviewCLI()
    cli.display_plan(test_planning)
    
    console.print("\n[bold]Testing Progress Display...[/bold]")
    progress = ProgressDisplay()
    
    progress.show_phase_start("Phase 3: Data Collection", "Collecting data from arXiv, Trends, and News")
    progress.show_agent_start("Data Collection Agent", "Starting parallel collection")
    progress.show_tool_call("arxiv_tool", {"keywords": ["humanoid robot"], "max_results": 100})
    progress.show_tool_result("arxiv_tool", "success")
    progress.show_agent_complete("Data Collection Agent", "Collected 342 papers, 36 months trends, 127 articles")