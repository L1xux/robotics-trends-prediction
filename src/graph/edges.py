"""
LangGraph Conditional Edges (Clean Architecture)

Workflow routing logic with clear status management
"""

from typing import Literal

from src.graph.state import PipelineState, WorkflowStatus


# =========================================================
# Routing Functions
# =========================================================

def route_after_writer(
    state: PipelineState
) -> Literal["end", "writer", "data_collection"]:
    """
    Route after Writer Agent

    Status mapping:
    - completed → end (Generate documents)
    - revision_complete → writer (Re-review revised report)
    - needs_recollection → data_collection (Restart data collection)
    - default → end (Safety fallback)

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    status = state.get("status", "")

    # Status to route mapping
    route_map = {
        WorkflowStatus.COMPLETED: "end",
        WorkflowStatus.REVISION_COMPLETE: "writer",  # Loop back to writer for re-review
        WorkflowStatus.NEEDS_RECOLLECTION: "data_collection",
    }

    route = route_map.get(status, "end")

    print(f"[Router] Writer → {route} (status: {status})")

    return route


def route_after_revision(
    state: PipelineState
) -> Literal["writer"]:
    """
    Route after Revision Agent

    Always returns to Writer for re-assembly and re-review

    Args:
        state: Current pipeline state

    Returns:
        Next node name (always "writer")
    """
    # Log routing decision (optional)
    # print(f"[Router] Revision → writer")

    return "writer"


# =========================================================
# Exports
# =========================================================

__all__ = [
    "route_after_writer",
    "route_after_revision",
]
