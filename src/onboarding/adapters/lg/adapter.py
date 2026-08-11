"""LangGraph adapter: an explicit graph with conditional branches."""

from __future__ import annotations

from typing import ClassVar

from onboarding.adapters.callers import LangChainLlmCaller
from onboarding.adapters.lg import nodes
from onboarding.adapters.lg.graph import build_graph
from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.llm import llm_spec
from onboarding.core.rules import Capabilities
from onboarding.core.schemas import (
    CustomerRecord,
    Framework,
    OnboardingResult,
    OnboardingState,
)


class LangGraphAdapter:
    """Multi-step graph: the control flow is data, and every branch is inspectable."""

    name: ClassVar[Framework] = "langgraph"
    capabilities: ClassVar[Capabilities] = Capabilities(
        multi_step=True,
        conditional_branching=True,
        tools=False,
        agent_count="multi",
        notes=(
            "Explicit graph: the control flow is data, so every branch can be inspected "
            "before the run rather than discovered during it. The repair loop is a real "
            "cycle in the graph, not a retry in a wrapper."
        ),
        extras={"conditional_branch_points": "2", "nodes": "11"},
    )

    def __init__(self, *, allow_send: bool = False) -> None:
        self.allow_send = allow_send

    def _caller(self):
        if not llm_spec().configured:
            return None
        return LangChainLlmCaller()

    @concept(Concept.AGENTIC_FIRST, Concept.CONDITIONAL_BRANCHING)
    async def run(
        self, record: CustomerRecord, *, run_id: str, record_path: str | None = None
    ) -> OnboardingResult:
        paths().ensure_runs()
        nodes.set_context(llm=self._caller(), allow_send=self.allow_send)
        state = steps.new_state(record, run_id, self.name)
        sink = default_sink(run_id, record.record_id, self.name)
        sink.emit("run_started", framework=self.name, company=record.company_name)

        graph = build_graph().compile()
        final = await graph.ainvoke({"state": state.model_dump(mode="json")})
        return steps.finalize(OnboardingState.model_validate(final["state"]), sink)
