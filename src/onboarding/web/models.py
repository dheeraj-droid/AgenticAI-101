"""Request and response shapes for the demo page.

Kept separate from the app so the mapping from *what a web form can send* to
``CustomerRecord`` is one readable function. The form is untrusted input: it
arrives from a browser, and everything it carries is validated by pydantic here
before any of it reaches the pipeline.
"""

from __future__ import annotations

import re
import uuid
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from onboarding.core.schemas import (
    CommercialTerms,
    Contact,
    CustomerRecord,
    OnboardingResult,
    PlanName,
    Region,
)

# The form asks for a plan, because that is what a business user knows. Tier is
# what drives onboarding policy, so it is derived rather than asked for twice.
PLAN_TO_TIER: dict[PlanName, str] = {"free": "starter", "pro": "growth", "pro+": "enterprise"}


class OnboardRequest(BaseModel):
    """One submission of the form."""

    model_config = ConfigDict(extra="forbid")

    framework: str = "langgraph"
    company_name: str = Field(min_length=1, max_length=120)
    full_name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    phone: str = Field(default="", max_length=40)
    plan: PlanName = "pro"
    region: Region = "us"
    products: str = ""
    annual_contract_value_usd: Decimal = Field(default=Decimal("12000"), ge=0)
    term_months: int = Field(default=12, ge=1, le=120)
    signup_notes: str = Field(default="", max_length=4000)
    record_id: str = ""

    def to_record(self) -> CustomerRecord:
        """Build the shared input schema every framework consumes."""
        return CustomerRecord(
            record_id=self.record_id.strip() or _record_id(self.company_name),
            company_name=self.company_name.strip(),
            tier=PLAN_TO_TIER[self.plan],  # type: ignore[arg-type]
            region=self.region,
            plan=self.plan,
            primary_contact=Contact(
                full_name=self.full_name.strip(),
                email=self.email,
                phone=self.phone.strip() or None,
            ),
            products=[p.strip() for p in self.products.split(",") if p.strip()],
            commercial_terms=CommercialTerms(
                annual_contract_value_usd=self.annual_contract_value_usd,
                contract_start=date.today(),
                term_months=self.term_months,
            ),
            signup_notes=self.signup_notes.strip(),
            source_system="web",
        )


def _record_id(company_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", company_name).strip("-").upper()[:20] or "CUST"
    return f"{slug}-{uuid.uuid4().hex[:6].upper()}"


class MailView(BaseModel):
    """One outbound message, as the page shows it."""

    model_config = ConfigDict(extra="forbid")

    audience: Literal["team", "customer"]
    to: str
    subject: str
    body: str
    transmitted: bool
    reason: str


class FindingView(BaseModel):
    """One validation finding, structured so the page can style the severity."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: str
    message: str


class TaskView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    owner_role: str
    priority: str
    due_offset_days: int
    status: str = "pending"


class OnboardResponse(BaseModel):
    """Everything the page needs to render one run."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    framework: str
    run_id: str
    record_id: str
    company_name: str
    contact_name: str
    plan: str
    status: str
    outcome: Literal["registered", "duplicate", "blocked"]
    headline: str
    duplicate: bool
    registered: bool
    findings: list[FindingView] = Field(default_factory=list)
    pii_entity_types: list[str] = Field(default_factory=list)
    injection_signals: list[str] = Field(default_factory=list)
    risk_band: str = "low"
    confidence: float = 0.0
    escalation_queue: list[str] = Field(default_factory=list)
    tasks: list[TaskView] = Field(default_factory=list)
    mail: list[MailView] = Field(default_factory=list)
    llm_calls: int = 0
    duration_ms: int = 0


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    tool_calls: list[str] = Field(default_factory=list)
    framework: str


def describe_outcome(result: OnboardingResult) -> tuple[str, str]:
    """The page's one-line verdict: (outcome, headline).

    Derived from the result rather than from which endpoint was called, so the
    page cannot claim something the pipeline did not do.
    """
    duplicate = any(f.code == "ALREADY_REGISTERED" for f in result.findings)
    if duplicate:
        return "duplicate", "Already signed up — we sent them a note saying so."
    if result.registered:
        return "registered", "Registered. Welcome email sent, task list sent to support."
    return "blocked", "Not registered — the run stopped before anything was drafted."
