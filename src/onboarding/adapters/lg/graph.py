"""The LangGraph onboarding graph.

Two conditional branch points, matching the MAF workflow decision for decision:

1. after ``risk_gate`` — a record that cannot be onboarded escalates; everything
   else drafts
2. after ``reflect``   — repair loop, confidence fallback, or finish
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from onboarding.adapters.lg import nodes
from onboarding.adapters.lg.state import GraphState
from onboarding.core.concepts import Concept, concept


@concept(Concept.WORKFLOW_DECOMPOSITION, Concept.CONDITIONAL_BRANCHING, Concept.AGENTIC_FIRST)
def build_graph() -> StateGraph:
    """Assemble the graph. Compiling it is the adapter's job."""
    g = StateGraph(GraphState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("plan", nodes.plan)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("risk_gate", nodes.risk_gate)
    g.add_node("draft_email", nodes.draft_email)
    g.add_node("build_tasks", nodes.build_tasks)
    g.add_node("reflect", nodes.reflect)
    g.add_node("repair", nodes.repair)
    g.add_node("deliver", nodes.deliver)
    g.add_node("escalate", nodes.escalate)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "plan")
    g.add_edge("plan", "rewrite_query")
    g.add_edge("rewrite_query", "risk_gate")

    # 1. blocking validation errors or a prompt-injection attempt stop the run;
    #    everything else drafts. Planning always runs first, so even a record
    #    that never reaches a model still gets its remediation task list — and
    #    the MAF switch-case group makes the identical two-way choice.
    g.add_conditional_edges(
        "risk_gate",
        nodes.route_after_risk,
        {"escalate": "escalate", "draft": "draft_email"},
    )

    g.add_edge("draft_email", "build_tasks")
    g.add_edge("build_tasks", "reflect")

    # 2. reflection: bounded repair loop, then confidence fallback or finish
    g.add_conditional_edges(
        "reflect",
        nodes.route_after_reflect,
        {"repair": "repair", "escalate": "escalate", "finalize": "deliver"},
    )
    g.add_edge("repair", "reflect")
    # Registration and mail happen only on the path that passed reflection.
    g.add_edge("deliver", "finalize")

    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    return g


CONDITIONAL_BRANCH_POINTS = ("risk_gate", "reflect")
