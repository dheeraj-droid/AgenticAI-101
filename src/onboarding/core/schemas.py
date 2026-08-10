"""The shared input schema, intermediate types, and the identical output schema.

Every framework consumes ``CustomerRecord`` and produces ``OnboardingResult``.
``OnboardingResult.deterministic()`` projects out everything an LLM is allowed to
vary, leaving the part that must be byte-identical across all three — that
projection is what the comparison runner and the golden tests assert on.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

Tier = Literal["starter", "growth", "enterprise"]
Region = Literal["us", "eu", "apac", "other"]
Severity = Literal["error", "warning", "info"]
RunStatus = Literal[
    "completed",
    "blocked_awaiting_approval",
    "rejected",
    "escalated",
    "failed",
]
Framework = Literal["maf", "langchain", "langgraph"]

# Value objects are frozen; only OnboardingState is mutable.
_VALUE = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class Contact(BaseModel):
    model_config = _VALUE

    full_name: str
    email: EmailStr
    phone: str | None = None
    title: str | None = None


class DeclaredDiscount(BaseModel):
    """The ONLY legal source of a discount claim.

    If a commercial concession is not in this list, no generated email may
    mention it. ``core.discounts`` enforces that mechanically.
    """

    model_config = _VALUE

    label: str
    kind: Literal["percent", "fixed_usd", "free_months", "waived_fee"]
    value: Decimal
    approved_by: str


class CommercialTerms(BaseModel):
    model_config = _VALUE

    annual_contract_value_usd: Decimal = Field(ge=0)
    currency: Literal["USD"] = "USD"
    declared_discounts: list[DeclaredDiscount] = Field(default_factory=list)
    contract_start: date
    term_months: int = Field(ge=1, le=120)


class CustomerRecord(BaseModel):
    """The shared input schema — identical for all three frameworks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    company_name: str
    tier: Tier
    region: Region
    primary_contact: Contact
    additional_contacts: list[Contact] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    commercial_terms: CommercialTerms
    signup_notes: str = ""
    requested_go_live: date | None = None
    source_system: str = "manual"

    @property
    def all_contacts(self) -> list[Contact]:
        return [self.primary_contact, *self.additional_contacts]


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    model_config = _VALUE

    code: str
    severity: Severity
    field_path: str
    message: str


class PiiEntity(BaseModel):
    model_config = _VALUE

    entity_type: str
    start: int
    end: int
    score: float
    source_field: str
    engine: Literal["presidio", "regex"]


class RedactionReport(BaseModel):
    model_config = _VALUE

    entities: list[PiiEntity] = Field(default_factory=list)
    masked_text_by_field: dict[str, str] = Field(default_factory=dict)
    # Maps placeholder -> a short hash of the original. Never the raw value:
    # this report is written to the audit log.
    placeholder_digests: dict[str, str] = Field(default_factory=dict)
    engine: Literal["presidio", "regex"] = "presidio"

    @property
    def entity_types(self) -> list[str]:
        return sorted({e.entity_type for e in self.entities})


class InjectionSignal(BaseModel):
    model_config = _VALUE

    pattern_id: str
    matched_span: str  # already PII-masked before it lands here
    field_path: str
    severity: Literal["block", "flag"]


class NotesChunk(BaseModel):
    model_config = _VALUE

    index: int
    text: str
    token_estimate: int


class Perception(BaseModel):
    """Output of the Perception phase. ``safe_context`` is the ONLY thing the LLM sees."""

    model_config = _VALUE

    findings: list[Finding] = Field(default_factory=list)
    redaction: RedactionReport = Field(default_factory=RedactionReport)
    injection_signals: list[InjectionSignal] = Field(default_factory=list)
    safe_context: dict[str, str] = Field(default_factory=dict)
    chunks: list[NotesChunk] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    model_config = _VALUE

    order: int
    goal: str
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = _VALUE

    steps: list[PlanStep] = Field(default_factory=list)
    rewritten_query: str = ""
    strategy: Literal["standard", "enterprise", "remediation"] = "standard"


class RiskAssessment(BaseModel):
    model_config = _VALUE

    tier_risk: bool = False
    value_risk: bool = False
    injection_risk: bool = False
    validation_risk: bool = False
    score: float = 0.0
    band: Literal["low", "medium", "high"] = "low"
    reasons: list[str] = Field(default_factory=list)
    requires_human_approval: bool = False


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class PromptRef(BaseModel):
    """Identifies exactly which prompt version produced a piece of output."""

    model_config = _VALUE

    id: str
    version: int
    checksum: str


class WelcomeEmail(BaseModel):
    model_config = _VALUE

    subject: str
    body: str
    prompt_ref: PromptRef | None = None

    @property
    def word_count(self) -> int:
        return len(self.body.split())


class OnboardingTask(BaseModel):
    model_config = _VALUE

    task_id: str
    title: str
    owner_role: str
    due_offset_days: int
    priority: Literal["p0", "p1", "p2"]
    origin: Literal["rule", "llm"] = "rule"


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


class RuleViolation(BaseModel):
    model_config = _VALUE

    rule_id: str
    detail: str
    span: str | None = None
    remediation: Literal["repaired", "redacted", "escalated"] | None = None


class Reflection(BaseModel):
    model_config = _VALUE

    violations: list[RuleViolation] = Field(default_factory=list)
    repair_attempts: int = 0
    confidence: float = 1.0
    passed: bool = True


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    model_config = _VALUE

    run_id: str
    record_id: str
    company_name: str
    reasons: list[str] = Field(default_factory=list)
    risk_band: Literal["low", "medium", "high"] = "high"
    annual_contract_value_usd: Decimal = Decimal(0)
    draft_subject: str | None = None
    draft_body: str | None = None


class ApprovalDecision(BaseModel):
    model_config = _VALUE

    decision: Literal["approve", "reject"]
    decided_by: str = "cli"
    note: str = ""
    decided_at: datetime | None = None


# ---------------------------------------------------------------------------
# Output — identical across all three frameworks
# ---------------------------------------------------------------------------


class DeterministicOutcome(BaseModel):
    """The part of a run that must NOT vary between frameworks or between models.

    Everything an LLM authors is excluded by construction; what remains is
    decided by ``core`` alone, so an inequality here is a real divergence.
    """

    model_config = _VALUE

    record_id: str
    status: RunStatus
    finding_codes: list[str]
    pii_entity_types: list[str]
    injection_pattern_ids: list[str]
    risk_band: Literal["low", "medium", "high"]
    requires_human_approval: bool
    plan_goals: list[str]
    rule_task_ids: list[str]
    violation_rule_ids: list[str]
    # Deliberately NOT included: prompt_refs. Each framework legitimately uses a
    # different prompt set (the graphs render welcome_email; the LangChain agent
    # renders agent_instructions and drives tools), so requiring them to match
    # would assert something untrue. They are reported side by side instead.


class OnboardingResult(BaseModel):
    """The shared output schema. All three adapters return exactly this."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    record_id: str
    framework: Framework
    status: RunStatus

    findings: list[Finding] = Field(default_factory=list)
    pii_entity_types: list[str] = Field(default_factory=list)
    pii_engine: Literal["presidio", "regex"] = "presidio"
    injection_signals: list[InjectionSignal] = Field(default_factory=list)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    plan: Plan = Field(default_factory=Plan)

    welcome_email: WelcomeEmail | None = None
    tasks: list[OnboardingTask] = Field(default_factory=list)
    reflection: Reflection = Field(default_factory=Reflection)
    confidence: float = 0.0
    escalation_queue: list[str] = Field(default_factory=list)

    prompt_refs: list[PromptRef] = Field(default_factory=list)
    audit_log_path: str = ""
    audit_event_ids: list[str] = Field(default_factory=list)

    resume_supported: bool = False
    resume_token: str | None = None

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int = 0
    llm_calls: int = 0
    error: str | None = None

    def deterministic(self) -> DeterministicOutcome:
        return DeterministicOutcome(
            record_id=self.record_id,
            status=self.status,
            finding_codes=sorted(f.code for f in self.findings),
            pii_entity_types=sorted(set(self.pii_entity_types)),
            injection_pattern_ids=sorted({s.pattern_id for s in self.injection_signals}),
            risk_band=self.risk.band,
            requires_human_approval=self.risk.requires_human_approval,
            plan_goals=[s.goal for s in self.plan.steps],
            rule_task_ids=sorted(t.task_id for t in self.tasks if t.origin == "rule"),
            violation_rule_ids=sorted(v.rule_id for v in self.reflection.violations),
        )


# ---------------------------------------------------------------------------
# Mutable pipeline state — the carrier passed between steps and persisted
# ---------------------------------------------------------------------------


class OnboardingState(BaseModel):
    """Mutable state threaded through the pipeline by every framework.

    This is what gets checkpointed (MAF ``FileCheckpointStorage``, LangGraph
    ``SqliteSaver``), so it must stay JSON-serialisable.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    framework: Framework
    record: CustomerRecord

    perception: Perception | None = None
    plan: Plan | None = None
    risk: RiskAssessment | None = None
    email: WelcomeEmail | None = None
    tasks: list[OnboardingTask] = Field(default_factory=list)
    reflection: Reflection | None = None

    approval_decision: Literal["approve", "reject"] | None = None
    status: RunStatus = "completed"
    escalation_queue: list[str] = Field(default_factory=list)
    prompt_refs: list[PromptRef] = Field(default_factory=list)
    audit_event_ids: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    repair_attempts: int = 0
    started_at: datetime | None = None
    error: str | None = None

    # -- routing predicates, shared verbatim by MAF and LangGraph -----------

    def has_blocking_errors(self) -> bool:
        p = self.perception
        return bool(p and any(f.severity == "error" for f in p.findings))

    def requires_human_approval(self) -> bool:
        return bool(self.risk and self.risk.requires_human_approval)
