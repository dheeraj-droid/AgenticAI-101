"""CrewAI adapter: a role-based crew instead of one agent or one graph.

Where this sits in the comparison:

* **Multi-agent.** Two agents with distinct roles hand work to each other. The
  reviewer never sees the writer's reasoning, only its output — a separate
  context window is the actual mechanism behind "a second pair of eyes".
* **Sequential process, fixed at build time.** ``Process.sequential`` gives an
  ordering the other single-agent adapter has no equivalent of, but there is no
  branching: unlike MAF and LangGraph, a crew cannot route a record down a
  different path. Every conditional in this adapter is plain Python around the
  crew, which is honest about what the framework actually provides.
* **Stateless.** ``memory=False``, no checkpointer. Like LangChain, it can pause
  for approval but cannot be resumed.

Everything outside the drafting step is the same ``core.steps`` pipeline the
other three call, so any difference in the deterministic output is a real
divergence rather than a difference in business logic.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from onboarding.adapters.callers import CrewLlmCaller
from onboarding.adapters.crew import tools
from onboarding.adapters.crew.crew import EmailDraft, build_crew
from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.errors import ResumeNotSupportedError
from onboarding.core.hitl import ResumeIndex, record_approval_required
from onboarding.core.rules import Capabilities
from onboarding.core.schemas import (
    ApprovalDecision,
    CustomerRecord,
    Framework,
    OnboardingResult,
    OnboardingState,
    WelcomeEmail,
)


class CrewAdapter:
    """Two agents, one sequential process, no durable state."""

    name: ClassVar[Framework] = "crew"
    capabilities: ClassVar[Capabilities] = Capabilities(
        multi_step=True,
        conditional_branching=False,
        hitl_pause=True,
        durable_resume=False,
        tools=True,
        agent_count="multi",
        statefulness="stateless",
        checkpoint_backend="none (memory=False)",
        notes=(
            "The only multi-agent adapter: a copywriter drafts and a separate compliance "
            "reviewer checks, each with its own context window. The order is fixed at "
            "build time and the crew cannot branch, so conditional routing lives in the "
            "adapter rather than in the framework."
        ),
        extras={"agents": "2", "crew_tasks": "2", "graph_nodes": "n/a"},
    )

    def __init__(self, *, allow_send: bool = False) -> None:
        self.allow_send = allow_send

    @concept(Concept.SINGLE_VS_MULTI_AGENT, Concept.STATELESS_VS_STATEFUL, Concept.AGENT_VS_LLM_APP)
    async def run(
        self, record: CustomerRecord, *, run_id: str, record_path: str | None = None
    ) -> OnboardingResult:
        paths().ensure_runs()
        state = steps.new_state(record, run_id, self.name)
        sink = default_sink(run_id, record.record_id, self.name)
        sink.emit("run_started", framework=self.name, company=record.company_name)

        state = steps.plan(steps.perceive(state, sink), sink)

        if state.has_blocking_errors():
            state = steps.escalate(state, sink)
            state = steps.notify_already_registered(state, sink, allow_send=self.allow_send)
            return steps.finalize(state, sink)

        if state.requires_human_approval():
            record_approval_required(state, sink, record_path=record_path, index=ResumeIndex())
            return steps.finalize(state, sink)

        state = await self._draft_with_crew(state, sink)
        state = await steps.act_build_tasks(state, sink, llm=CrewLlmCaller())
        state = steps.reflect(state, sink)

        # The crew already reviewed itself. We validate again anyway, and repair
        # through the same path as every other adapter — self-review is evidence,
        # not authority.
        if steps.needs_repair(state):
            state = await steps.repair_email(state, CrewLlmCaller(), sink)
            state = steps.reflect(state, sink)
        if steps.should_escalate(state):
            state = steps.escalate(state, sink)
        else:
            state = steps.register_customer(state, sink)
            state = steps.send_notifications(state, sink, allow_send=self.allow_send)

        return steps.finalize(state, sink)

    @concept(Concept.ACTION, Concept.SINGLE_VS_MULTI_AGENT)
    async def _draft_with_crew(self, state: OnboardingState, sink) -> OnboardingState:
        """Run the crew, then record what came back.

        ``kickoff`` is synchronous and CPU-bound between tool calls, so it runs
        on a worker thread rather than stalling the caller's event loop.
        """
        crew, prompt_refs = build_crew()
        tools.set_current_state(state)
        try:
            output = await asyncio.to_thread(
                crew.kickoff,
                {
                    "company_name": state.record.company_name,
                    "rewritten_query": state.plan.rewritten_query if state.plan else "",
                },
            )
        finally:
            tools.set_current_state(None)

        draft = _extract_draft(output)
        # Two agents, so at least two completions; tool loops add more.
        state.llm_calls += len(getattr(crew, "tasks", []) or [1])
        for ref in prompt_refs:
            if ref not in state.prompt_refs:
                state.prompt_refs.append(ref)
        instructions_ref = next((r for r in prompt_refs if r.id == "agent_instructions"), None)
        state.email = WelcomeEmail(
            subject=draft.subject, body=draft.body, prompt_ref=instructions_ref
        )
        sink.emit(
            "email_drafted",
            subject=state.email.subject,
            word_count=state.email.word_count,
            crew_agents=2,
        )
        return state

    @concept(Concept.STATELESS_VS_STATEFUL)
    async def resume(self, run_id: str, decision: ApprovalDecision) -> OnboardingResult:
        raise ResumeNotSupportedError(
            f"the CrewAI adapter runs with memory=False, so run {run_id!r} cannot be resumed: "
            "a crew kickoff is a single call with no checkpoint to return to. Re-run the "
            "record from the start, or use --framework maf|langgraph, which persist state."
        )


def _extract_draft(output: Any) -> EmailDraft:
    """Prefer the crew's structured output; fall back to parsing its text."""
    for attribute in ("pydantic", "json_dict"):
        candidate = getattr(output, attribute, None)
        if isinstance(candidate, EmailDraft):
            return candidate
        if isinstance(candidate, dict):
            try:
                return EmailDraft.model_validate(candidate)
            except Exception:
                pass

    from onboarding.core.llm import extract_json_object

    text = str(getattr(output, "raw", "") or output)
    return EmailDraft.model_validate(extract_json_object(text))
