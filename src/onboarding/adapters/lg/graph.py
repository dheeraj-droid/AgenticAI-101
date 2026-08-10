"""The LangGraph onboarding graph.

Three conditional branch points, matching the MAF workflow decision for
decision:

1. after ``risk_gate``       — escalate / human approval / draft (one of three)
2. after ``human_approval``  — approve continues, reject escalates
3. after ``reflect``         — repair loop, confidence fallback, or finish
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from onboarding.adapters.lg import nodes
from onboarding.adapters.lg.state import GraphState
from onboarding.core.concepts import Concept, concept


@concept(Concept.WORKFLOW_DECOMPOSITION, Concept.CONDITIONAL_BRANCHING, Concept.AGENTIC_FIRST)
def build_graph() -> StateGraph:
    """Assemble the graph. Compiling with a checkpointer is the adapter's job."""
    g = StateGraph(GraphState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("plan", nodes.plan)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("risk_gate", nodes.risk_gate)
    g.add_node("human_approval", nodes.human_approval)
    g.add_node("draft_email", nodes.draft_email)
    g.add_node("build_tasks", nodes.build_tasks)
    g.add_node("reflect", nodes.reflect)
    g.add_node("repair", nodes.repair)
    g.add_node("escalate", nodes.escalate)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "plan")
    g.add_edge("plan", "rewrite_query")
    g.add_edge("rewrite_query", "risk_gate")

    # 1. one-of-three: blocking errors escalate, high risk needs a human,
    #    everything else drafts. Planning always runs first, so even a record
    #    that never reaches a model still gets its remediation task list —
    #    and the MAF switch-case group makes the identical three-way choice.
    g.add_conditional_edges(
        "risk_gate",
        nodes.route_after_risk,
        {"escalate": "escalate", "approve_needed": "human_approval", "draft": "draft_email"},
    )

    # 2. what the human decided
    g.add_conditional_edges(
        "human_approval",
        nodes.route_after_decision,
        {"approve": "draft_email", "reject": "escalate"},
    )

    g.add_edge("draft_email", "build_tasks")
    g.add_edge("build_tasks", "reflect")

    # 3. reflection: bounded repair loop, then confidence fallback or finish
    g.add_conditional_edges(
        "reflect",
        nodes.route_after_reflect,
        {"repair": "repair", "escalate": "escalate", "finalize": "finalize"},
    )
    g.add_edge("repair", "reflect")

    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    return g


CONDITIONAL_BRANCH_POINTS = ("risk_gate", "human_approval", "reflect")
