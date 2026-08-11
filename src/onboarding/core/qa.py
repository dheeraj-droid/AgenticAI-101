"""Read-only question answering over the customer registry.

Two rules govern this module, and both are enforced by tests rather than
convention:

**The model can only read.** Every function here is a pure query. The registry's
write path (``registry.append_customer``) is deliberately not imported, so no
tool built on this module can insert, edit or delete a customer. Changing the
table is something deterministic pipeline code does; the agent only ever
describes what is already there.

**Phone numbers never reach the model at all.** Not masked, not partially — the
field is simply absent from everything this module returns, so there is nothing
for a model to leak or reconstruct. Names, emails, companies and plans are
ordinary business facts an employee may see, and are passed through in full.
"""

from __future__ import annotations

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
from onboarding.core.tasks import summarise

KNOWN_PLANS: tuple[str, ...] = ("free", "pro", "pro+")


@dataclass(frozen=True, slots=True)
class VisibleCustomer:
    """A registry row as the model is allowed to see it.

    Note the absence of a phone field. It is not masked here — it is never
    carried, so no code path downstream can accidentally surface it.
    """

    record_id: str
    customer_name: str
    company_name: str
    plan: str
    email: str
    status: str
    registered_at: str
    tasks_total: int = 0
    tasks_completed: int = 0

    def as_line(self) -> str:
        tasks = (
            f", tasks: {self.tasks_completed}/{self.tasks_total} done"
            if self.tasks_total
            else ""
        )
        return (
            f"{self.customer_name} ({self.company_name}) — plan: {self.plan}, "
            f"email: {self.email}, status: {self.status}{tasks}"
        )


@concept(Concept.PII_DETECTION)
def visible(row: RegistryRow) -> VisibleCustomer:
    """Project a registry row down to what the model may see.

    The phone number is dropped here and nowhere reintroduced.
    """
    summary = summarise(row.record_id)
    return VisibleCustomer(
        record_id=row.record_id,
        customer_name=row.customer_name,
        company_name=row.company_name,
        plan=row.plan,
        email=row.email,
        status=row.status,
        registered_at=row.registered_at,
        tasks_total=summary.total,
        tasks_completed=summary.completed,
    )


# ---------------------------------------------------------------------------
# The read-only query surface. Everything a model can call lives here.
# ---------------------------------------------------------------------------


@concept(Concept.ACTION)
def lookup(query: str) -> list[VisibleCustomer]:
    """Find customers by name, email, company or record id."""
    return [visible(row) for row in find_customer(query)]


@concept(Concept.ACTION)
def on_plan(plan: str) -> list[VisibleCustomer]:
    """Every customer on a given plan."""
    return [visible(row) for row in customers_on_plan(plan)]


@concept(Concept.ACTION)
def breakdown() -> PlanBreakdown:
    """How many customers are on each plan."""
    return plan_breakdown()


@concept(Concept.ACTION)
def all_customers() -> list[VisibleCustomer]:
    return [visible(row) for row in read_all()]


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
    lines = [describe_breakdown(), "", "Customers (phone numbers are not available):"]
    lines.extend(f"  - {row.as_line()}" for row in rows[:limit])
    if len(rows) > limit:
        lines.append(f"  ... and {len(rows) - limit} more")
    return "\n".join(lines)
