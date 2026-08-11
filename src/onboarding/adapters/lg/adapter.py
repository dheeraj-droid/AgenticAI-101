"""LangGraph adapter: stateful, durable, resumable across processes."""

from __future__ import annotations

from typing import Any, ClassVar

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from onboarding.adapters.callers import LangChainLlmCaller
from onboarding.adapters.lg import nodes
from onboarding.adapters.lg.graph import build_graph
from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.hitl import ResumeIndex
from onboarding.core.llm import llm_spec
from onboarding.core.rules import Capabilities
from onboarding.core.schemas import (
    ApprovalDecision,
    CustomerRecord,
    Framework,
    OnboardingResult,
    OnboardingState,
)


class LangGraphAdapter:
    """Multi-step graph with conditional branches and a SQLite checkpointer."""

    name: ClassVar[Framework] = "langgraph"
    capabilities: ClassVar[Capabilities] = Capabilities(
        multi_step=True,
        conditional_branching=True,
        hitl_pause=True,
        durable_resume=True,
        tools=False,
        agent_count="multi",
        statefulness="stateful",
        checkpoint_backend="SqliteSaver (.runs/langgraph.sqlite)",
        notes=(
            "Explicit graph: the control flow is data, and every branch is inspectable "
            "before the run. interrupt() suspends mid-graph and the checkpoint survives "
            "process exit."
        ),
        extras={"conditional_branch_points": "3", "nodes": "11"},
    )

    def __init__(self, *, allow_send: bool = False) -> None:
        self.allow_send = allow_send

    def _caller(self):
        if not llm_spec().configured:
            return None
        return LangChainLlmCaller()

    @concept(Concept.AGENTIC_FIRST, Concept.STATELESS_VS_STATEFUL, Concept.DURABLE_STATE)
    async def run(
        self, record: CustomerRecord, *, run_id: str, record_path: str | None = None
    ) -> OnboardingResult:
        paths().ensure_runs()
        nodes.set_context(llm=self._caller(), record_path=record_path, allow_send=self.allow_send)
        state = steps.new_state(record, run_id, self.name)
        sink = default_sink(run_id, record.record_id, self.name)
        sink.emit("run_started", framework=self.name, company=record.company_name)

        config = {"configurable": {"thread_id": run_id}}
        # The saver context stays open for the whole run: closing it mid-graph
        # would drop the connection the checkpointer writes through.
        async with AsyncSqliteSaver.from_conn_string(str(paths().langgraph_db)) as saver:
            graph = build_graph().compile(checkpointer=saver)
            final = await graph.ainvoke({"state": state.model_dump(mode="json")}, config=config)
            return await self._result(final, graph, config, sink, run_id)

    @concept(Concept.HUMAN_IN_THE_LOOP, Concept.DURABLE_STATE)
    async def resume(self, run_id: str, decision: ApprovalDecision) -> OnboardingResult:
        """Resume a paused run from the on-disk checkpoint, in a fresh process."""
        entry = ResumeIndex().get(run_id)
        nodes.set_context(llm=self._caller(), record_path=entry.record_path, allow_send=self.allow_send)
        sink = default_sink(run_id, entry.record_id, self.name)

        config = {"configurable": {"thread_id": entry.thread_id or run_id}}
        async with AsyncSqliteSaver.from_conn_string(str(paths().langgraph_db)) as saver:
            graph = build_graph().compile(checkpointer=saver)
            final = await graph.ainvoke(
                Command(resume=decision.model_dump(mode="json")), config=config
            )
            result = await self._result(final, graph, config, sink, run_id)
        ResumeIndex().mark_resolved(run_id, result.status)
        return result

    async def _result(self, final: dict[str, Any], graph, config, sink, run_id: str) -> OnboardingResult:
        """Turn the graph's final channel values into the shared output schema."""
        state = OnboardingState.model_validate(final["state"])
        snapshot = await graph.aget_state(config)
        if snapshot.next:  # the graph is parked at an interrupt
            state.status = "blocked_awaiting_approval"
        return steps.finalize(state, sink, resume_token=run_id)
