"""Human-in-the-loop checkpoint.

Behaviour (as specified): when a record is high-risk the run **pauses**, writes an
``approval_required`` audit entry with the draft attached, marks the record
BLOCKED and stops. It does not auto-continue.

The paused state is persisted to disk, so a *later, separate process* can resume
it:

    onboarding run --framework lg --record fixtures/customers/enterprise_high_value.json
    # -> blocked_awaiting_approval, prints a run_id
    onboarding resume --run-id <id> --decision approve

``ResumeIndex`` is the small directory that maps a run_id to whichever
framework's checkpoint holds it. The framework-native checkpointers (LangGraph
``SqliteSaver``, MAF ``FileCheckpointStorage``) hold the actual state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from onboarding.core.audit import JsonlAuditSink
from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.errors import RunNotFoundError
from onboarding.core.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    OnboardingState,
)


class ResumeEntry(BaseModel):
    """Everything needed to pick a paused run back up in a fresh process."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    record_id: str
    framework: Literal["maf", "langchain", "langgraph"]
    status: str = "blocked_awaiting_approval"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Framework-native handles.
    thread_id: str | None = None  # LangGraph
    checkpoint_id: str | None = None  # MAF
    request_id: str | None = None  # MAF request_info correlation id
    record_path: str | None = None
    approval_request: ApprovalRequest | None = None
    state_snapshot: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def state(self) -> OnboardingState | None:
        if self.state_snapshot is None:
            return None
        return OnboardingState.model_validate(self.state_snapshot)


class ResumeIndex:
    """A tiny JSON directory of paused runs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths().resume_index
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def put(self, entry: ResumeEntry) -> None:
        data = self._load()
        data[entry.run_id] = json.loads(entry.model_dump_json())
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def get(self, run_id: str) -> ResumeEntry:
        data = self._load()
        if run_id not in data:
            known = ", ".join(sorted(data)) or "(none)"
            raise RunNotFoundError(f"no paused run {run_id!r}. Known runs: {known}")
        return ResumeEntry.model_validate(data[run_id])

    def list(self) -> list[ResumeEntry]:
        return [ResumeEntry.model_validate(v) for v in self._load().values()]

    def mark_resolved(self, run_id: str, status: str) -> None:
        data = self._load()
        if run_id in data:
            data[run_id]["status"] = status
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@concept(Concept.HUMAN_IN_THE_LOOP, Concept.AUTONOMOUS_VS_ASSISTIVE)
def build_approval_request(state: OnboardingState) -> ApprovalRequest:
    """Describe, for a human, what needs signing off and why."""
    risk = state.risk
    return ApprovalRequest(
        run_id=state.run_id,
        record_id=state.record.record_id,
        company_name=state.record.company_name,
        reasons=list(risk.reasons) if risk else [],
        risk_band=risk.band if risk else "high",
        annual_contract_value_usd=state.record.commercial_terms.annual_contract_value_usd,
        draft_subject=state.email.subject if state.email else None,
        draft_body=state.email.body if state.email else None,
    )


@concept(Concept.HUMAN_IN_THE_LOOP, Concept.DURABLE_STATE, Concept.AUDIT_LOGGING)
def record_approval_required(
    state: OnboardingState,
    sink: JsonlAuditSink,
    *,
    thread_id: str | None = None,
    checkpoint_id: str | None = None,
    request_id: str | None = None,
    record_path: str | None = None,
    index: ResumeIndex | None = None,
) -> ApprovalRequest:
    """Log the pause, persist the resume handle, and mark the record blocked.

    This is the single implementation of "pause, log, stop" — MAF and LangGraph
    both call it, so their HITL behaviour is identical by construction.
    """
    request = build_approval_request(state)
    event_id = sink.emit(
        "approval_required",
        run_id=state.run_id,
        reasons=request.reasons,
        risk_band=request.risk_band,
        annual_contract_value_usd=str(request.annual_contract_value_usd),
        draft_subject=request.draft_subject,
        draft_body=request.draft_body,
        resume_hint=f"onboarding resume --run-id {state.run_id} --decision approve|reject",
    )
    state.audit_event_ids.append(event_id)
    state.status = "blocked_awaiting_approval"

    (index or ResumeIndex()).put(
        ResumeEntry(
            run_id=state.run_id,
            record_id=state.record.record_id,
            framework=state.framework,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            request_id=request_id,
            record_path=record_path,
            approval_request=request,
            # A snapshot of everything decided before the pause. The framework
            # checkpoint is the resume mechanism; this is what lets us *report*
            # a blocked run (its findings, risk and task list) without having to
            # crack open a framework-specific checkpoint format.
            state_snapshot=state.model_dump(mode="json"),
        )
    )
    return request


@concept(Concept.HUMAN_IN_THE_LOOP)
def apply_decision(state: OnboardingState, decision: ApprovalDecision, sink: JsonlAuditSink) -> OnboardingState:
    """Record a human decision on a paused run."""
    state.approval_decision = decision.decision
    event_id = sink.emit(
        "approval_decided",
        decision=decision.decision,
        decided_by=decision.decided_by,
        note=decision.note,
    )
    state.audit_event_ids.append(event_id)
    if decision.decision == "reject":
        state.status = "rejected"
        state.escalation_queue.append(
            f"rejected by {decision.decided_by}"
            + (f": {decision.note}" if decision.note else "")
        )
    return state
