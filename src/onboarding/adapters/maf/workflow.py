"""The Microsoft Agent Framework onboarding workflow.

Two switch-case edge groups plus two conditional edges give the same four
decision points as the LangGraph implementation:

1. ``risk_gate``  — switch-case: needs approval / blocking errors / default
2. ``approval``   — conditional edges on the human's decision
3. ``reflect``    — switch-case: repair / escalate / finalize
"""

from __future__ import annotations

from typing import Any

from agent_framework import Case, CheckpointStorage, Default, Workflow, WorkflowBuilder

from onboarding.adapters.maf.executors import (
    ApprovalExecutor,
    DraftEmailExecutor,
    EscalateExecutor,
    FinalizeExecutor,
    IngestExecutor,
    PlanExecutor,
    ReflectExecutor,
    RepairExecutor,
    RiskGateExecutor,
    TaskListExecutor,
)
from onboarding.core import steps
from onboarding.core.concepts import Concept, concept
from onboarding.core.schemas import OnboardingState

# FileCheckpointStorage will only deserialise types on this allow-list.
ALLOWED_CHECKPOINT_TYPES: list[str] = [
    "onboarding.core.schemas:OnboardingState",
    "onboarding.core.schemas:ApprovalRequest",
    "onboarding.core.schemas:ApprovalDecision",
    "onboarding.core.schemas:OnboardingResult",
]

WORKFLOW_NAME = "customer_onboarding"

EXECUTOR_IDS = (
    "ingest",
    "plan",
    "risk_gate",
    "approval",
    "draft_email",
    "task_list",
    "reflect",
    "repair",
    "escalate",
    "finalize",
)


def _state(payload: Any) -> OnboardingState:
    """Rehydrate the routing view of the message travelling along an edge."""
    return OnboardingState.model_validate(payload)


# --- routing predicates: the same core functions LangGraph routes on --------


@concept(Concept.CONDITIONAL_BRANCHING, Concept.HUMAN_IN_THE_LOOP)
def needs_approval(payload: Any) -> bool:
    return _state(payload).requires_human_approval()


@concept(Concept.CONDITIONAL_BRANCHING)
def has_blocking_errors(payload: Any) -> bool:
    return _state(payload).has_blocking_errors()


@concept(Concept.CONDITIONAL_BRANCHING)
def approved(payload: Any) -> bool:
    return _state(payload).approval_decision == "approve"


@concept(Concept.CONDITIONAL_BRANCHING)
def rejected(payload: Any) -> bool:
    return _state(payload).approval_decision != "approve"


@concept(Concept.CONDITIONAL_BRANCHING, Concept.NO_FABRICATED_CLAIMS)
def needs_repair(payload: Any) -> bool:
    return steps.needs_repair(_state(payload))


@concept(Concept.CONDITIONAL_BRANCHING, Concept.CONFIDENCE_FALLBACK)
def should_escalate(payload: Any) -> bool:
    return steps.should_escalate(_state(payload))


@concept(Concept.WORKFLOW_DECOMPOSITION, Concept.CONDITIONAL_BRANCHING, Concept.AGENTIC_FIRST)
def build_workflow(checkpoint_storage: CheckpointStorage | None = None) -> Workflow:
    """Assemble the workflow graph.

    Fresh executor instances per call, so two concurrent runs never share state.
    """
    ingest = IngestExecutor()
    plan = PlanExecutor()
    risk_gate = RiskGateExecutor()
    approval = ApprovalExecutor()
    draft_email = DraftEmailExecutor()
    task_list = TaskListExecutor()
    reflect = ReflectExecutor()
    repair = RepairExecutor()
    escalate = EscalateExecutor()
    finalize = FinalizeExecutor()

    builder = WorkflowBuilder(
        name=WORKFLOW_NAME,
        description="Validate a customer record, draft a welcome email, generate tasks, log results",
        start_executor=ingest,
        checkpoint_storage=checkpoint_storage,
        output_from=[finalize, escalate],
    )

    builder.add_edge(ingest, plan)
    builder.add_edge(plan, risk_gate)

    # 1. blocking errors never reach a model; high risk goes to a human first
    builder.add_switch_case_edge_group(
        risk_gate,
        [
            Case(condition=has_blocking_errors, target=escalate),
            Case(condition=needs_approval, target=approval),
            Default(target=draft_email),
        ],
    )

    # 2. what the human decided
    builder.add_edge(approval, draft_email, condition=approved)
    builder.add_edge(approval, escalate, condition=rejected)

    builder.add_edge(draft_email, task_list)
    builder.add_edge(task_list, reflect)

    # 3. bounded repair loop, then confidence fallback or finish
    builder.add_switch_case_edge_group(
        reflect,
        [
            Case(condition=needs_repair, target=repair),
            Case(condition=should_escalate, target=escalate),
            Default(target=finalize),
        ],
    )
    builder.add_edge(repair, reflect)

    return builder.build()
