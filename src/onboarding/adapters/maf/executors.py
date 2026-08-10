"""Microsoft Agent Framework executors.

One executor per pipeline phase. Each ``@handler`` unpacks the state, calls one
``core.steps`` function and forwards the result — no business logic lives here.

State travels as a serialised dict rather than the pydantic model itself so the
``FileCheckpointStorage`` type allow-list stays small and the checkpoint file is
plain JSON.
"""

# NOTE: deliberately no `from __future__ import annotations` here.
# @response_handler validates the ctx parameter using the raw
# inspect.signature annotation rather than get_type_hints, so postponed
# (string) annotations make it reject an otherwise valid WorkflowContext.
from typing import Any, Never

from agent_framework import Executor, WorkflowContext, handler, response_handler

from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.hitl import ResumeIndex, apply_decision, record_approval_required
from onboarding.core.llm import LlmCaller
from onboarding.core.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    OnboardingResult,
    OnboardingState,
)

StatePayload = dict[str, Any]

# Set by the adapter before each run.
_CONTEXT: dict[str, Any] = {"llm": None, "record_path": None}


def set_context(*, llm: LlmCaller | None, record_path: str | None = None) -> None:
    _CONTEXT["llm"] = llm
    _CONTEXT["record_path"] = record_path


def _load(payload: StatePayload) -> OnboardingState:
    return OnboardingState.model_validate(payload)


def _dump(state: OnboardingState) -> StatePayload:
    return state.model_dump(mode="json")


def _sink(state: OnboardingState):
    return default_sink(state.run_id, state.record.record_id, state.framework)


def _pending_key(run_id: str) -> str:
    """Workflow-state key holding the state of a run parked at the approval gate."""
    return f"pending_approval:{run_id}"


def _llm() -> LlmCaller:
    llm = _CONTEXT["llm"]
    if llm is None:
        from onboarding.core.errors import LlmNotConfiguredError

        raise LlmNotConfiguredError("(the MAF workflow reached an executor that needs a model)")
    return llm


class IngestExecutor(Executor):
    """Perception: validate, mask PII, scan for injection, chunk."""

    def __init__(self, id: str = "ingest") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.PERCEPTION)
    async def ingest(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        state = _load(payload)
        await ctx.send_message(_dump(steps.perceive(state, _sink(state))))


class PlanExecutor(Executor):
    """Planning: risk assessment plus least-to-most decomposition."""

    def __init__(self, id: str = "plan") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.PLANNING, Concept.LEAST_TO_MOST, Concept.QUERY_REWRITING)
    async def plan(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        state = _load(payload)
        await ctx.send_message(_dump(steps.plan(state, _sink(state))))


class RiskGateExecutor(Executor):
    """Source of the first switch-case group. Routing happens on the edges."""

    def __init__(self, id: str = "risk_gate") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.CONDITIONAL_BRANCHING)
    async def gate(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        await ctx.send_message(payload)


class ApprovalExecutor(Executor):
    """Human-in-the-loop checkpoint.

    ``ctx.request_info`` suspends the workflow and the checkpoint is written to
    disk, so the decision can arrive from a different process entirely.
    """

    def __init__(self, id: str = "approval") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.HUMAN_IN_THE_LOOP, Concept.DURABLE_STATE)
    async def request_approval(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        state = _load(payload)
        request = record_approval_required(
            state,
            _sink(state),
            checkpoint_id=state.run_id,
            record_path=_CONTEXT["record_path"],
            index=ResumeIndex(),
        )
        # The response handler only receives the original request, so park the
        # full state in workflow state — it rides along in the checkpoint and is
        # still there when a different process resumes the run.
        ctx.set_state(_pending_key(state.run_id), _dump(state))
        await ctx.request_info(request, ApprovalDecision, request_id=state.run_id)

    @response_handler
    @concept(Concept.HUMAN_IN_THE_LOOP)
    async def on_decision(
        self,
        request: ApprovalRequest,
        response: ApprovalDecision,
        ctx: WorkflowContext[dict[str, Any]],
    ) -> None:
        payload = ctx.get_state(_pending_key(request.run_id))
        state = _load(payload)
        await ctx.send_message(_dump(apply_decision(state, response, _sink(state))))


class DraftEmailExecutor(Executor):
    """Action: the welcome email. The only executor that must reach a model."""

    def __init__(self, id: str = "draft_email") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.ACTION, Concept.CHAIN_OF_THOUGHT)
    async def draft(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        state = _load(payload)
        await ctx.send_message(_dump(await steps.act_draft_email(state, _llm(), _sink(state))))


class TaskListExecutor(Executor):
    """Action: the internal task list."""

    def __init__(self, id: str = "task_list") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.ACTION, Concept.WORKFLOW_DECOMPOSITION)
    async def build(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        state = _load(payload)
        updated = await steps.act_build_tasks(state, _sink(state), llm=_CONTEXT["llm"])
        await ctx.send_message(_dump(updated))


class ReflectExecutor(Executor):
    """Reflection: run the output validators and score confidence."""

    def __init__(self, id: str = "reflect") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.REFLECTION)
    async def review(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        state = _load(payload)
        await ctx.send_message(_dump(steps.reflect(state, _sink(state))))


class RepairExecutor(Executor):
    """Reflection: one critique-and-retry pass, then hard redaction."""

    def __init__(self, id: str = "repair") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.REFLECTION, Concept.NO_FABRICATED_CLAIMS)
    async def repair(self, payload: StatePayload, ctx: WorkflowContext[dict[str, Any]]) -> None:
        state = _load(payload)
        await ctx.send_message(_dump(await steps.repair_email(state, _llm(), _sink(state))))


class EscalateExecutor(Executor):
    """Terminal: route to the human review queue."""

    def __init__(self, id: str = "escalate") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.CONFIDENCE_FALLBACK)
    async def escalate(self, payload: StatePayload, ctx: WorkflowContext[Never, OnboardingResult]) -> None:
        state = _load(payload)
        sink = _sink(state)
        await ctx.yield_output(steps.finalize(steps.escalate(state, sink), sink, resume_token=state.run_id))


class FinalizeExecutor(Executor):
    """Terminal: assemble the shared output schema."""

    def __init__(self, id: str = "finalize") -> None:
        super().__init__(id=id)

    @handler
    @concept(Concept.AUDIT_LOGGING)
    async def finalize(self, payload: StatePayload, ctx: WorkflowContext[Never, OnboardingResult]) -> None:
        state = _load(payload)
        sink = _sink(state)
        await ctx.yield_output(steps.finalize(state, sink, resume_token=state.run_id))
