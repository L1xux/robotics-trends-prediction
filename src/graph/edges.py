"""
LangGraph Conditional Edges

Workflow routing 로직
"""

from typing import Literal
from src.graph.state import PipelineState


def route_after_writer(
    state: PipelineState
) -> Literal["end", "revision", "data_collection"]:
    """
    Writer Agent 이후 routing (Human feedback 포함)
    
    - completed → end (PDF generation)
    - needs_revision → revision
    - needs_recollection → data_collection (재시작)
    """
    status = state.get("status", "")
    
    if status == "completed":
        return "end"
    elif status == "needs_revision":
        return "revision"
    elif status == "needs_recollection":
        return "data_collection"
    elif status == "revision_completed":
        # Revision 후 다시 Writer로 (재조립 및 재검토)
        return "writer"
    else:
        # Default: end
        return "end"


def route_after_revision(
    state: PipelineState
) -> Literal["writer"]:
    """
    Revision 이후 routing
    
    항상 writer로 돌아가서 재조립 및 재검토
    """
    return "writer"
