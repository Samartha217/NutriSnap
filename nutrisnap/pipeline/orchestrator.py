"""
LangGraph StateGraph orchestrator.

Pattern: Sequential Multi-Agent Pipeline with Conditional Early Exit.
Any agent that sets pipeline_failed=True short-circuits to END.

Compiled once at module import — PIPELINE is the singleton to use.
"""
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.agent1_extraction import run_agent1
from pipeline.agent2_grounding import run_agent2
from pipeline.agent3_scoring import run_agent3
from observability.logger import get_logger

logger = get_logger(__name__)


def _should_continue(state: PipelineState) -> str:
    """Conditional edge — stop the entire pipeline if any agent failed."""
    if state.get("pipeline_failed"):
        return "end"
    return "continue"


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("agent1", run_agent1)
    graph.add_node("agent2", run_agent2)
    graph.add_node("agent3", run_agent3)

    graph.set_entry_point("agent1")

    graph.add_conditional_edges(
        "agent1",
        _should_continue,
        {"continue": "agent2", "end": END},
    )
    graph.add_conditional_edges(
        "agent2",
        _should_continue,
        {"continue": "agent3", "end": END},
    )
    graph.add_edge("agent3", END)

    return graph.compile()


# Compile once at startup — reused for every request
PIPELINE = build_pipeline()
logger.info("pipeline_compiled", extra={"nodes": ["agent1", "agent2", "agent3"]})
