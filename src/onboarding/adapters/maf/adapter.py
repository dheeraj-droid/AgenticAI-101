"""Microsoft Agent Framework adapter: graph workflow, tools, durable checkpoints."""

from __future__ import annotations

from typing import Any, ClassVar

from agent_framework import FileCheckpointStorage, WorkflowRunState

from onboarding.adapters.callers import MafLlmCaller
from onboarding.adapters.maf import executors
from onboarding.adapters.maf.workflow import (
    ALLOWED_CHECKPOINT_TYPES,
    WORKFLOW_NAME,
    build_workflow,
)
from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.errors import OnboardingError
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


class MafAdapter:
    """Executor graph with switch-case routing, request_info HITL and file checkpoints."""

    name: ClassVar[Framework] = "maf"
    capabilities: ClassVar[Capabilities] = Capabilities(
        multi_step=True,
        conditional_branching=True,
        hitl_pause=True,
        durable_resume=True,
        tools=True,
        agent_count="multi",
        statefulness="stateful",
        checkpoint_backend="FileCheckpointStorage (.runs/maf_checkpoints)",
        notes=(
            "Typed executors and edges. request_info() suspends the workflow and the run "
            "resumes via run(responses={request_id: decision}). Switch-case edge groups make "
            "one-of-N routing explicit rather than a chain of ifs."
        ),
        extras={"switch_case_groups": "2", "executors": "10", "agent_tools": "5"},
    )

    def __init__(self, *, allow_send: bool = False) -> None:
        self.allow_send = allow_send

    def _storage(self) -> FileCheckpointStorage:
        paths().ensure_runs()
        return FileCheckpointStorage(
            paths().maf_checkpoints,
            allowed_checkpoint_types=ALLOWED_CHECKPOINT_TYPES,
        )

    def _caller(self):
        if not llm_spec().configured:
            return None
        return MafLlmCaller()

    @concept(Concept.AGENTIC_FIRST, Concept.SINGLE_VS_MULTI_AGENT, Concept.DURABLE_STATE)
    async def run(
        self, record: CustomerRecord, *, run_id: str, record_path: str | None = None
    ) -> OnboardingResult:
        executors.set_context(llm=self._caller(), record_path=record_path, allow_send=self.allow_send)
        state = steps.new_state(record, run_id, self.name)
        sink = default_sink(run_id, record.record_id, self.name)
        sink.emit("run_started", framework=self.name, company=record.company_name)

        storage = self._storage()
        workflow = build_workflow(storage)
        result = await workflow.run(state.model_dump(mode="json"))
        return self._result(result, state, sink, run_id)

    @concept(Concept.HUMAN_IN_THE_LOOP, Concept.DURABLE_STATE)
    async def resume(self, run_id: str, decision: ApprovalDecision) -> OnboardingResult:
        """Resume a paused workflow from its on-disk checkpoint, in a fresh process."""
        entry = ResumeIndex().get(run_id)
        executors.set_context(llm=self._caller(), record_path=entry.record_path, allow_send=self.allow_send)
        sink = default_sink(run_id, entry.record_id, self.name)

        storage = self._storage()
        workflow = build_workflow(storage)
        checkpoints = await storage.list_checkpoints(workflow_name=WORKFLOW_NAME)
        checkpoint_id = _latest_checkpoint_id(checkpoints)
        if checkpoint_id is None:
            raise OnboardingError(
                f"no MAF checkpoint found for run {run_id!r}; the workflow cannot be resumed"
            )

        outcome = await workflow.run(
            responses={entry.request_id or run_id: decision},
            checkpoint_id=checkpoint_id,
            checkpoint_storage=storage,
        )
        result = self._result(outcome, None, sink, run_id)
        ResumeIndex().mark_resolved(run_id, result.status)
        return result

    def _result(
        self, outcome: Any, state: OnboardingState | None, sink, run_id: str
    ) -> OnboardingResult:
        """Prefer the workflow's own yielded output; fall back to the paused state."""
        outputs = [o for o in outcome.get_outputs() if isinstance(o, OnboardingResult)]
        if outputs:
            return outputs[-1]

        # No terminal executor ran: the workflow is parked on a pending request.
        # The executors work on serialised copies, so the caller's `state` object
        # is still pristine — take the snapshot written at the pause instead, so a
        # blocked run reports its findings, risk and task list like any other.
        pending = outcome.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
        try:
            parked = ResumeIndex().get(run_id).state()
        except OnboardingError:
            parked = None
        state = parked or state
        if state is None:
            raise OnboardingError(
                f"MAF run {run_id!r} produced no output and no state to fall back on "
                f"(final state: {outcome.get_final_state()})"
            )
        if pending:
            state.status = "blocked_awaiting_approval"
        return steps.finalize(state, sink, resume_token=run_id)


def _latest_checkpoint_id(checkpoints: list[Any]) -> str | None:
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda c: getattr(c, "timestamp", 0) or 0)
    return getattr(latest, "checkpoint_id", None)
