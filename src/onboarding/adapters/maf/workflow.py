"""The Microsoft Agent Framework onboarding workflow.

Two switch-case edge groups give the same two decision points as the LangGraph
implementation:

1. ``risk_gate`` — switch-case: a record that cannot be onboarded / default
2. ``reflect``   — switch-case: repair / escalate / finalize
"""

from __future__ import annotations

from typing import Any

from agent_framework import Case, Default, Workflow, WorkflowBuilder

from onboarding.adapters.maf.executors import (
    DeliverExecutor,
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

WORKFLOW_NAME = "customer_onboarding"

EXECUTOR_IDS = (
    "ingest",
    "plan",
    "risk_gate",
    "draft_email",
    "task_list",
    "reflect",
    "repair",
    "deliver",
    "escalate",
    "finalize",
)


def _state(payload: Any) -> OnboardingState:
    """Rehydrate the routing view of the message travelling along an edge."""
    return OnboardingState.model_validate(payload)


# --- routing predicates: the same core functions LangGraph routes on --------


@concept(Concept.CONDITIONAL_BRANCHING)
def must_escalate(payload: Any) -> bool:
    return _state(payload).must_escalate()


@concept(Concept.CONDITIONAL_BRANCHING, Concept.NO_FABRICATED_CLAIMS)
def needs_repair(payload: Any) -> bool:
    return steps.needs_repair(_state(payload))


@concept(Concept.CONDITIONAL_BRANCHING, Concept.CONFIDENCE_FALLBACK)
def should_escalate(payload: Any) -> bool:
    return steps.should_escalate(_state(payload))


@concept(Concept.WORKFLOW_DECOMPOSITION, Concept.CONDITIONAL_BRANCHING, Concept.AGENTIC_FIRST)
def build_workflow() -> Workflow:
    """Assemble the workflow graph.

    Fresh executor instances per call, so two concurrent runs never share state.
    """
    ingest = IngestExecutor()
    plan = PlanExecutor()
    risk_gate = RiskGateExecutor()
    draft_email = DraftEmailExecutor()
    task_list = TaskListExecutor()
    reflect = ReflectExecutor()
    repair = RepairExecutor()
    deliver = DeliverExecutor()
    escalate = EscalateExecutor()
    finalize = FinalizeExecutor()

    builder = WorkflowBuilder(
        name=WORKFLOW_NAME,
        description="Validate a customer record, draft a welcome email, generate tasks, log results",
        start_executor=ingest,
        output_from=[finalize, escalate],
    )

    builder.add_edge(ingest, plan)
    builder.add_edge(plan, risk_gate)

    # 1. a record with blocking errors or an injection attempt never reaches a model
    builder.add_switch_case_edge_group(
        risk_gate,
        [
            Case(condition=must_escalate, target=escalate),
            Default(target=draft_email),
        ],
    )

    builder.add_edge(draft_email, task_list)
    builder.add_edge(task_list, reflect)

    # 2. bounded repair loop, then confidence fallback or finish
    builder.add_switch_case_edge_group(
        reflect,
        [
            Case(condition=needs_repair, target=repair),
            Case(condition=should_escalate, target=escalate),
            Default(target=deliver),
        ],
    )
    builder.add_edge(repair, reflect)
    # Registration and mail happen only on the path that passed reflection.
    builder.add_edge(deliver, finalize)

    return builder.build()
