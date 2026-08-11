"""LangChain adapter: a single, stateless, tool-using agent.

The contrast this adapter exists to draw:

* **Single agent, not a graph.** One ``create_agent`` loop decides its own tool
  order. There is no topology to inspect before the run.
* **Stateless.** No graph, no thread id — the whole run is one call, and the
  order in which tools fire is only knowable afterwards, from the transcript.
* **Not trusted to police itself.** The agent has a ``check_business_rules``
  tool, but the adapter re-runs the same validators server-side afterwards.
  An agent that skips its own checks does not get a free pass.
"""

from __future__ import annotations

from typing import Any, ClassVar

from onboarding.adapters.callers import LangChainLlmCaller
from onboarding.adapters.lc import tools
from onboarding.adapters.lc.agent import EmailDraft, build_agent
from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.rules import Capabilities
from onboarding.core.schemas import (
    CustomerRecord,
    Framework,
    OnboardingResult,
    OnboardingState,
    WelcomeEmail,
)


class LangChainAdapter:
    """One tool-using agent, no graph."""

    name: ClassVar[Framework] = "langchain"
    capabilities: ClassVar[Capabilities] = Capabilities(
        multi_step=False,
        conditional_branching=False,
        tools=True,
        agent_count="single",
        notes=(
            "The agent picks its own tool order, so the control flow only exists at "
            "runtime — there is no topology to inspect before the run, only a transcript "
            "to read after it."
        ),
        extras={"agent_tools": "6", "graph_nodes": "n/a"},
    )

    def __init__(self, *, allow_send: bool = False) -> None:
        self.allow_send = allow_send

    @concept(Concept.SINGLE_VS_MULTI_AGENT, Concept.AGENT_VS_LLM_APP)
    async def run(
        self, record: CustomerRecord, *, run_id: str, record_path: str | None = None
    ) -> OnboardingResult:
        paths().ensure_runs()
        state = steps.new_state(record, run_id, self.name)
        sink = default_sink(run_id, record.record_id, self.name)
        sink.emit("run_started", framework=self.name, company=record.company_name)

        # Perception and planning are policy, not agent discretion: they run the
        # same way here as in the two graphs, which is what keeps the
        # deterministic half of the output identical across frameworks.
        steps.trace(state, "perceive", "step", "policy, not agent discretion")
        state = steps.plan(steps.perceive(state, sink), sink)
        steps.trace(state, "plan", "step")

        if state.must_escalate():
            steps.trace(state, "escalate", "step")
            state = steps.escalate(state, sink)
            state = steps.notify_already_registered(state, sink, allow_send=self.allow_send)
            return steps.finalize(state, sink)

        state = await self._draft_with_agent(state, sink)
        state = await steps.act_build_tasks(state, sink, llm=LangChainLlmCaller())
        steps.trace(state, "reflect", "step")
        state = steps.reflect(state, sink)

        if steps.needs_repair(state):
            steps.trace(state, "repair", "step")
            state = await steps.repair_email(state, LangChainLlmCaller(), sink)
            state = steps.reflect(state, sink)
        if steps.should_escalate(state):
            state = steps.escalate(state, sink)
        else:
            steps.trace(state, "deliver", "step")
            state = steps.register_customer(state, sink)
            state = steps.send_notifications(state, sink, allow_send=self.allow_send)

        return steps.finalize(state, sink)

    @concept(Concept.ACTION, Concept.SINGLE_VS_MULTI_AGENT)
    async def _draft_with_agent(self, state: OnboardingState, sink) -> OnboardingState:
        """Hand the drafting to the agent, then record what it produced."""
        agent, prompt_refs = build_agent()
        tools.set_current_state(state)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": _kickoff(state)}]}
            )
        finally:
            tools.set_current_state(None)

        steps.trace(state, "create_agent", "agent", "one loop, tool order chosen at runtime")
        for name in _tool_call_names(result):
            steps.trace(state, name, "tool", "the agent decided to call this")

        draft = _extract_draft(result)
        state.llm_calls += 1
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
            tool_calls=len(_tool_call_names(result)),
        )
        return state


def _kickoff(state: OnboardingState) -> str:
    return (
        f"Draft the welcome email for {state.record.company_name}. "
        "Start by calling get_customer_context."
    )


def _extract_draft(result: dict[str, Any]) -> EmailDraft:
    """Prefer the structured response; fall back to parsing the last message."""
    structured = result.get("structured_response")
    if isinstance(structured, EmailDraft):
        return structured
    if isinstance(structured, dict):
        return EmailDraft.model_validate(structured)

    from onboarding.core.llm import extract_json_object

    messages = result.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if not content:
            continue
        try:
            return EmailDraft.model_validate(extract_json_object(str(content)))
        except Exception:
            continue
    raise ValueError("the LangChain agent returned no usable email draft")


def _tool_call_names(result: dict[str, Any]) -> list[str]:
    """The tools the agent actually called, in order.

    This is the whole evidence that LangChain ran: nobody wrote this sequence
    down, the agent picked it, and it can differ between two runs of the same
    record. A graph cannot produce a list like this.
    """
    names: list[str] = []
    for message in result.get("messages") or []:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(str(name))
    return names
