"""Read-only question answering over the customer registry.

Two rules govern this module, and both are enforced by tests rather than
convention:

**The model can only read.** Every function here is a pure query. The registry's
write path (``registry.append_customer``) is deliberately not imported, so no
tool built on this module can insert, edit or delete a customer. Changing the
table is something deterministic pipeline code does; the agent only ever
describes what is already there.

**Contact details stay masked, names do not.** The agreed policy is that a
customer's *name* and *plan* are ordinary business facts an employee may see,
while email and phone are masked before anything reaches the model. That keeps
"how many people are on the pro plan?" answerable while a leaked phone number
stays impossible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from onboarding.core.concepts import Concept, concept
from onboarding.core.registry import (
    PlanBreakdown,
    RegistryRow,
    customers_on_plan,
    find_customer,
    plan_breakdown,
    read_all,
)

KNOWN_PLANS: tuple[str, ...] = ("free", "pro", "pro+")


@dataclass(frozen=True, slots=True)
class MaskedCustomer:
    """A registry row as the model is allowed to see it."""

    record_id: str
    customer_name: str
    company_name: str
    plan: str
    email: str  # masked
    phone: str  # masked
    status: str
    registered_at: str

    def as_line(self) -> str:
        return (
            f"{self.customer_name} ({self.company_name}) — plan: {self.plan}, "
            f"email: {self.email}, phone: {self.phone}, status: {self.status}"
        )


def _mask_email(email: str) -> str:
    """Keep the shape, lose the identifier: ``d***@b***.com``."""
    email = email.strip()
    if "@" not in email:
        return "<EMAIL_REDACTED>" if email else ""
    local, _, domain = email.partition("@")
    parts = domain.split(".")
    host = parts[0] if parts else ""
    tld = "." + ".".join(parts[1:]) if len(parts) > 1 else ""
    return f"{local[:1]}***@{host[:1]}***{tld}"


def _mask_phone(phone: str) -> str:
    """Keep the country prefix and last two digits; hide the rest."""
    phone = phone.strip()
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 4:
        return "<PHONE_REDACTED>"
    prefix = "+" if phone.startswith("+") else ""
    return f"{prefix}{digits[:2]}***{digits[-2:]}"


@concept(Concept.PII_DETECTION)
def mask_row(row: RegistryRow) -> MaskedCustomer:
    """Apply the Q&A privacy policy to one row."""
    return MaskedCustomer(
        record_id=row.record_id,
        customer_name=row.customer_name,
        company_name=row.company_name,
        plan=row.plan,
        email=_mask_email(row.email),
        phone=_mask_phone(row.phone),
        status=row.status,
        registered_at=row.registered_at,
    )


# ---------------------------------------------------------------------------
# The read-only query surface. Everything a model can call lives here.
# ---------------------------------------------------------------------------


@concept(Concept.ACTION)
def lookup(query: str) -> list[MaskedCustomer]:
    """Find customers by name, email, company or record id."""
    return [mask_row(row) for row in find_customer(query)]


@concept(Concept.ACTION)
def on_plan(plan: str) -> list[MaskedCustomer]:
    """Every customer on a given plan."""
    return [mask_row(row) for row in customers_on_plan(plan)]


@concept(Concept.ACTION)
def breakdown() -> PlanBreakdown:
    """How many customers are on each plan."""
    return plan_breakdown()


@concept(Concept.ACTION)
def all_customers() -> list[MaskedCustomer]:
    return [mask_row(row) for row in read_all()]


def describe_breakdown() -> str:
    """A sentence the model can quote directly, with the counts already correct.

    Arithmetic is done here rather than left to the model — counting rows is
    exactly the kind of thing a 3B model gets subtly wrong, and the answer is
    cheap to compute exactly.
    """
    counts = breakdown()
    if counts.total == 0:
        return "The registry is empty — no customers have been onboarded yet."
    parts = [f"{counts.of(plan)} on {plan}" for plan in KNOWN_PLANS]
    other = {
        plan: n for plan, n in counts.counts.items() if plan not in KNOWN_PLANS
    }
    for plan, n in sorted(other.items()):
        parts.append(f"{n} on {plan}")
    return f"{counts.total} customer(s) registered: " + ", ".join(parts) + "."


def registry_context(limit: int = 50) -> str:
    """The masked table, formatted for a prompt."""
    rows = all_customers()
    if not rows:
        return "The customer registry is empty."
    lines = [describe_breakdown(), "", "Customers (contact details are masked):"]
    lines.extend(f"  - {row.as_line()}" for row in rows[:limit])
    if len(rows) > limit:
        lines.append(f"  ... and {len(rows) - limit} more")
    return "\n".join(lines)
