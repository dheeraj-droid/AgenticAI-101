"""Business rules — the single source of truth.

Every threshold, policy and template used by any framework lives here. The
adapters import this module; none of them define a rule of their own. A test
(``test_adapter_thinness``) enforces that mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from onboarding.core.concepts import Concept, concept


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Numeric policy. Changing a value here changes all four frameworks at once."""

    # A record at or above this ACV is treated as high risk.
    HIGH_VALUE_ACV_USD: Decimal = Decimal("100000")
    # Tiers that are high risk regardless of value.
    ENTERPRISE_TIERS: tuple[str, ...] = ("enterprise",)
    # A validation warning counts toward the risk score (conservative default).
    WARNING_IS_RISK: bool = True
    # Below this, we do not ship the draft — it goes to the escalation queue.
    MIN_CONFIDENCE: float = 0.62
    # How many times reflection may ask for a repair before giving up.
    MAX_REPAIR_ATTEMPTS: int = 1
    # Acceptable length band for a welcome email body.
    MIN_EMAIL_WORDS: int = 60
    MAX_EMAIL_WORDS: int = 250
    # Notes are chunked before analysis at roughly this size.
    CHUNK_TOKEN_TARGET: int = 120
    CHUNK_TOKEN_OVERLAP: int = 20


RULES = Thresholds()


# --- tone policy -----------------------------------------------------------

BANNED_PHRASES: tuple[str, ...] = (
    "dear sir or madam",
    "to whom it may concern",
    "as an ai",
    "as a language model",
    "i cannot",
    "guaranteed",
    "risk-free",
    "act now",
    "limited time",
    "no obligation",
    "best in the world",
    "unbeatable",
)

WARMTH_LEXICON: tuple[str, ...] = (
    "welcome",
    "delighted",
    "glad",
    "pleased",
    "excited",
    "looking forward",
    "thank you",
    "thanks",
    "happy",
)

CTA_LEXICON: tuple[str, ...] = (
    "next step",
    "next steps",
    "kick-off",
    "kickoff",
    "onboarding call",
    "get started",
    "reach out",
    "let us know",
    "schedule",
    "book",
    "reply",
)

MAX_EXCLAMATIONS = 2


# --- internal task templates ----------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    task_id: str
    title: str
    owner_role: str
    due_offset_days: int
    priority: str


BASE_TASKS: tuple[TaskTemplate, ...] = (
    TaskTemplate("create-crm-account", "Create CRM account and link contract", "ops", 1, "p0"),
    TaskTemplate("provision-workspace", "Provision customer workspace", "ops", 2, "p0"),
    TaskTemplate("send-welcome-email", "Send the approved welcome email", "csm", 1, "p0"),
    TaskTemplate("schedule-kickoff", "Schedule the onboarding kick-off call", "csm", 3, "p1"),
)

TIER_TASKS: dict[str, tuple[TaskTemplate, ...]] = {
    "starter": (
        TaskTemplate("share-self-serve-guide", "Share the self-serve setup guide", "csm", 2, "p2"),
    ),
    "growth": (
        TaskTemplate("assign-csm", "Assign a named customer success manager", "csm", 2, "p1"),
        TaskTemplate("plan-training", "Plan the product training session", "csm", 7, "p2"),
    ),
    "enterprise": (
        TaskTemplate("assign-csm", "Assign a named customer success manager", "csm", 1, "p0"),
        TaskTemplate("assign-solutions-architect", "Assign a solutions architect", "se", 2, "p0"),
        TaskTemplate("security-review", "Run the security and compliance review", "security", 5, "p0"),
        TaskTemplate("draft-success-plan", "Draft the mutual success plan", "csm", 10, "p1"),
    ),
}

REGION_TASKS: dict[str, tuple[TaskTemplate, ...]] = {
    "eu": (
        TaskTemplate("gdpr-dpa", "Confirm the GDPR data processing agreement", "legal", 5, "p0"),
        TaskTemplate("set-eu-residency", "Set EU data residency on the tenant", "ops", 3, "p0"),
    ),
    "apac": (
        TaskTemplate("confirm-apac-support", "Confirm APAC support-hours coverage", "support", 5, "p1"),
    ),
    "us": (),
    "other": (
        TaskTemplate("confirm-data-residency", "Confirm data residency requirements", "legal", 5, "p1"),
    ),
}

PRODUCT_TASKS: dict[str, TaskTemplate] = {
    "analytics": TaskTemplate("enable-analytics", "Enable the analytics module", "ops", 3, "p1"),
    "api": TaskTemplate("issue-api-keys", "Issue production API keys", "ops", 2, "p1"),
    "sso": TaskTemplate("configure-sso", "Configure SSO with the customer IdP", "ops", 4, "p0"),
    "support-premium": TaskTemplate(
        "activate-premium-support", "Activate premium support routing", "support", 2, "p1"
    ),
}

# Tasks added because the record could not be processed cleanly.
REMEDIATION_TASK = TaskTemplate(
    "fix-record-data", "Correct the customer record and re-run onboarding", "ops", 1, "p0"
)

@dataclass(frozen=True, slots=True)
class RiskTrigger:
    """Why a record scored as it did. Kept as data so the reasons are comparable."""

    code: str
    reason: str


@concept(Concept.POLICY_CONSTRAINED, Concept.AUTONOMOUS_VS_ASSISTIVE)
def risk_triggers(
    *,
    tier: str,
    annual_contract_value_usd: Decimal,
    has_injection_block: bool,
    has_validation_warning: bool,
    has_validation_error: bool,
) -> list[RiskTrigger]:
    """The one place that decides how risky a record is.

    The score this feeds drives the planning strategy and the confidence
    threshold. Whether a record is *stopped* is a separate, narrower question —
    see ``OnboardingState.must_escalate``.
    """
    triggers: list[RiskTrigger] = []
    if tier in RULES.ENTERPRISE_TIERS:
        triggers.append(RiskTrigger("TIER", f"tier '{tier}' is handled as high risk"))
    if annual_contract_value_usd >= RULES.HIGH_VALUE_ACV_USD:
        triggers.append(
            RiskTrigger(
                "VALUE",
                f"contract value {annual_contract_value_usd} >= threshold "
                f"{RULES.HIGH_VALUE_ACV_USD}",
            )
        )
    if has_injection_block:
        triggers.append(
            RiskTrigger("INJECTION", "a prompt-injection attempt was detected in free text")
        )
    if has_validation_error:
        triggers.append(RiskTrigger("VALIDATION_ERROR", "the record has blocking errors"))
    elif has_validation_warning and RULES.WARNING_IS_RISK:
        triggers.append(RiskTrigger("VALIDATION_WARNING", "the record has validation warnings"))
    return triggers


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a given framework adapter can actually do — shown in the report."""

    multi_step: bool
    conditional_branching: bool
    tools: bool
    agent_count: str  # "single" | "multi"
    notes: str = ""
    extras: dict[str, str] = field(default_factory=dict)
