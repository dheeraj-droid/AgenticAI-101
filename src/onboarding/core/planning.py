"""Planning: least-to-most decomposition, query rewriting, deterministic task list.

The task list is a *pure function of the record*. That is deliberate: it makes
the internal task list byte-identical across all four frameworks and safe to
assert on exactly. The model may only ever add clearly-marked ``origin="llm"``
tasks on top, and those are excluded from the equality check.
"""

from __future__ import annotations

from onboarding.core.concepts import Concept, concept
from onboarding.core.rules import (
    BASE_TASKS,
    PRODUCT_TASKS,
    REGION_TASKS,
    REMEDIATION_TASK,
    TIER_TASKS,
    TaskTemplate,
)
from onboarding.core.schemas import (
    CustomerRecord,
    Finding,
    OnboardingTask,
    Plan,
    PlanStep,
    RiskAssessment,
)


def _to_task(t: TaskTemplate) -> OnboardingTask:
    return OnboardingTask(
        task_id=t.task_id,
        title=t.title,
        owner_role=t.owner_role,
        due_offset_days=t.due_offset_days,
        priority=t.priority,  # type: ignore[arg-type]
        origin="rule",
    )


@concept(Concept.PLANNING, Concept.LEAST_TO_MOST, Concept.WORKFLOW_DECOMPOSITION)
def decompose(record: CustomerRecord, risk: RiskAssessment, findings: list[Finding]) -> Plan:
    """Least-to-most: order the work so each step depends only on earlier ones.

    The plan is data, not prose — every framework executes the same ordered
    goals, which is what makes their traces comparable.
    """
    blocking = any(f.severity == "error" for f in findings)
    if blocking:
        strategy = "remediation"
    elif record.tier == "enterprise" or risk.band == "high":
        strategy = "enterprise"
    else:
        strategy = "standard"

    steps: list[PlanStep] = [
        PlanStep(order=1, goal="Validate the customer record against onboarding policy"),
        PlanStep(order=2, goal="Mask personally identifiable information", depends_on=[1]),
        PlanStep(order=3, goal="Assess onboarding risk", depends_on=[1, 2]),
    ]
    if strategy == "remediation":
        steps.append(
            PlanStep(order=4, goal="Route the record for data correction instead of drafting", depends_on=[3])
        )
        return Plan(steps=steps, rewritten_query=rewrite_query(record, strategy), strategy=strategy)

    if strategy == "enterprise":
        # High-value work gets an extra grounding pass before anything is written.
        steps.append(
            PlanStep(order=4, goal="Re-check the approved commercial terms", depends_on=[3])
        )
        next_order = 5
    else:
        next_order = 4

    steps.extend(
        [
            PlanStep(order=next_order, goal="Draft the welcome email from masked context", depends_on=[2, 3]),
            PlanStep(order=next_order + 1, goal="Generate the internal onboarding task list", depends_on=[1]),
            PlanStep(
                order=next_order + 2,
                goal="Review the draft against tone, PII and discount policy",
                depends_on=[next_order],
            ),
            PlanStep(order=next_order + 3, goal="Log the result to the audit trail", depends_on=[next_order + 2]),
        ]
    )
    return Plan(steps=steps, rewritten_query=rewrite_query(record, strategy), strategy=strategy)


@concept(Concept.QUERY_REWRITING, Concept.CONTEXT_AWARE_PROMPT)
def rewrite_query(record: CustomerRecord, strategy: str) -> str:
    """Expand the bare task into an explicit, context-enriched instruction.

    Query rewriting done deterministically: no PII, no model call, and the same
    expansion for every framework.
    """
    products = ", ".join(record.products) if record.products else "no products yet"
    bits = [
        f"Draft a {strategy}-track welcome email for a {record.tier}-tier customer",
        f"in the {record.region.upper()} region",
        f"onboarding onto {products}",
    ]
    if record.requested_go_live:
        bits.append(f"targeting go-live on {record.requested_go_live.isoformat()}")
    if strategy == "enterprise":
        bits.append("with a named CSM and solutions architect being assigned")
    return "; ".join(bits) + "."


@concept(Concept.ACTION, Concept.WORKFLOW_DECOMPOSITION)
def derive_tasks(record: CustomerRecord, findings: list[Finding], risk: RiskAssessment) -> list[OnboardingTask]:
    """The internal task list. Pure function — identical across frameworks."""
    templates: list[TaskTemplate] = []
    blocking = any(f.severity == "error" for f in findings)

    if blocking:
        templates.append(REMEDIATION_TASK)
    else:
        templates.extend(BASE_TASKS)
        templates.extend(TIER_TASKS.get(record.tier, ()))
        templates.extend(REGION_TASKS.get(record.region, ()))
        for product in record.products:
            template = PRODUCT_TASKS.get(product.lower())
            if template:
                templates.append(template)

    # De-duplicate by task_id, keeping the most urgent occurrence.
    priority_rank = {"p0": 0, "p1": 1, "p2": 2}
    best: dict[str, TaskTemplate] = {}
    for t in templates:
        current = best.get(t.task_id)
        if current is None or priority_rank[t.priority] < priority_rank[current.priority]:
            best[t.task_id] = t

    tasks = [_to_task(t) for t in best.values()]
    return sorted(tasks, key=lambda t: (priority_rank[t.priority], t.due_offset_days, t.task_id))
