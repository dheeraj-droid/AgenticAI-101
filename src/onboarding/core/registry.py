"""The customer registry — a CSV table of who has been onboarded.

Columns are deliberately few: name, email, phone, plan, plus the bookkeeping
needed to trace a row back to its run.

**Read/write split.** Everything a model can reach lives in the "queries"
section below and is pure — it opens the file read-only and returns data. The
single write path, ``append_customer``, is called from deterministic pipeline
code and is never exposed as a tool. A test asserts no model-facing tool can
reach it, so the agent can answer questions about the registry but can never
change it.

CSV rather than a database because the point is that you can open it. That
choice costs concurrent-write safety, so the writer takes an exclusive lock on a
sidecar ``.csv.lock`` file and re-reads before appending.

The lock comes from ``filelock`` rather than ``fcntl``: fcntl is Unix-only, and
importing it broke every adapter on Windows. Cross-platform locking is fiddly
enough to be worth a dependency rather than a hand-rolled branch.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from filelock import FileLock

from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.schemas import CustomerRecord

FIELDNAMES = (
    "record_id",
    "customer_name",
    "email",
    "phone",
    "plan",
    "company_name",
    "status",
    "registered_at",
    "run_id",
)

DuplicateKind = Literal["record_id", "email", "company", "none"]

# Long enough to outlast a slow write, short enough to surface a stuck process.
LOCK_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class RegistryRow:
    """One customer in the table."""

    record_id: str
    customer_name: str
    email: str
    phone: str
    plan: str
    company_name: str = ""
    status: str = "registered"
    registered_at: str = ""
    run_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in FIELDNAMES}


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    """Why an incoming record looks like one we already have."""

    kind: DuplicateKind
    existing: RegistryRow | None = None
    detail: str = ""

    @property
    def is_duplicate(self) -> bool:
        return self.kind != "none"


@dataclass(slots=True)
class PlanBreakdown:
    """Aggregate view used to answer "how many people are on the pro plan"."""

    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def of(self, plan: str) -> int:
        return self.counts.get(_normalise_plan(plan), 0)


def registry_path() -> Path:
    override = os.environ.get("ONBOARDING_REGISTRY")
    return Path(override) if override else paths().runs / "registry.csv"


def lock_path(target: Path | None = None) -> Path:
    """Sidecar lock file guarding writes to the registry."""
    return (target or registry_path()).with_suffix(".csv.lock")


def _normalise_plan(plan: str) -> str:
    cleaned = plan.strip().lower().replace(" ", "")
    aliases = {"proplus": "pro+", "pro-plus": "pro+", "pro_plus": "pro+", "freemium": "free"}
    return aliases.get(cleaned, cleaned)


def _normalise_company(name: str) -> str:
    """Strip the noise that makes the same company look like two."""
    cleaned = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    stop = {"inc", "ltd", "llc", "limited", "gmbh", "corp", "corporation", "co", "plc", "group", "the"}
    return " ".join(word for word in cleaned.split() if word not in stop)


def _email_domain(email: str) -> str:
    return email.split("@")[-1].strip().lower() if "@" in email else ""


# ---------------------------------------------------------------------------
# Queries — pure, read-only. These are what the model is allowed to reach.
# ---------------------------------------------------------------------------


def read_all(path: Path | None = None) -> list[RegistryRow]:
    """Every row in the table. Returns [] when the registry does not exist yet."""
    target = path or registry_path()
    if not target.exists():
        return []
    with open(target, newline="", encoding="utf-8") as fh:
        return [
            RegistryRow(**{name: (row.get(name) or "") for name in FIELDNAMES})
            for row in csv.DictReader(fh)
        ]


@concept(Concept.PERCEPTION)
def find_duplicate(record: CustomerRecord, path: Path | None = None) -> DuplicateMatch:
    """Decide whether we have already onboarded this customer.

    Three tests, most to least certain. A company-name or shared-domain hit is a
    *suspicion*, not a verdict — the caller routes it to a human rather than
    silently refusing, because a genuine second team at the same company is a
    normal thing to onboard.
    """
    rows = read_all(path)
    if not rows:
        return DuplicateMatch(kind="none")

    email = str(record.primary_contact.email).strip().lower()
    for row in rows:
        if row.record_id == record.record_id:
            return DuplicateMatch("record_id", row, f"record_id {record.record_id!r} is already registered")

    for row in rows:
        if row.email.strip().lower() == email:
            return DuplicateMatch("email", row, f"{row.customer_name} is already registered with this email")

    company = _normalise_company(record.company_name)
    domain = _email_domain(email)
    for row in rows:
        if company and _normalise_company(row.company_name) == company:
            return DuplicateMatch(
                "company",
                row,
                f"{record.company_name!r} looks like the already-registered {row.company_name!r}",
            )
        if domain and _email_domain(row.email) == domain and domain not in _GENERIC_DOMAINS:
            return DuplicateMatch(
                "company",
                row,
                f"shares the email domain {domain!r} with the registered {row.customer_name}",
            )
    return DuplicateMatch(kind="none")


_GENERIC_DOMAINS = frozenset({"gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "example.com"})


def plan_breakdown(path: Path | None = None) -> PlanBreakdown:
    """How many customers are on each plan."""
    rows = read_all(path)
    counts: dict[str, int] = {}
    for row in rows:
        key = _normalise_plan(row.plan)
        counts[key] = counts.get(key, 0) + 1
    return PlanBreakdown(counts=counts, total=len(rows))


def customers_on_plan(plan: str, path: Path | None = None) -> list[RegistryRow]:
    wanted = _normalise_plan(plan)
    return [row for row in read_all(path) if _normalise_plan(row.plan) == wanted]


def find_customer(query: str, path: Path | None = None) -> list[RegistryRow]:
    """Look a customer up by name, email, company or record id."""
    needle = query.strip().lower()
    if not needle:
        return []
    return [
        row
        for row in read_all(path)
        if needle in row.customer_name.lower()
        or needle in row.email.lower()
        or needle in row.company_name.lower()
        or needle == row.record_id.lower()
    ]


# ---------------------------------------------------------------------------
# The single write path — deterministic code only, never a model tool.
# ---------------------------------------------------------------------------


@concept(Concept.ACTION, Concept.AUDIT_LOGGING)
def append_customer(
    record: CustomerRecord,
    *,
    run_id: str = "",
    status: str = "registered",
    path: Path | None = None,
) -> RegistryRow:
    """Add one customer to the table.

    Takes an exclusive cross-platform lock and re-checks for a duplicate
    *inside* the lock, so two runs racing on the same record cannot both insert.
    """
    target = path or registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    row = RegistryRow(
        record_id=record.record_id,
        customer_name=record.primary_contact.full_name,
        email=str(record.primary_contact.email),
        phone=record.primary_contact.phone or "",
        plan=record.effective_plan,
        company_name=record.company_name,
        status=status,
        registered_at=datetime.now(UTC).isoformat(timespec="seconds"),
        run_id=run_id,
    )

    with FileLock(str(lock_path(target)), timeout=LOCK_TIMEOUT_SECONDS):
        # Re-read inside the lock: another run may have inserted this record
        # between our duplicate check and getting here.
        is_new_file = not target.exists() or target.stat().st_size == 0
        with open(target, "a+", newline="", encoding="utf-8") as fh:
            if is_new_file:
                csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()
            else:
                fh.seek(0)
                existing = {r.get("record_id") for r in csv.DictReader(fh)}
                if record.record_id in existing:
                    raise AlreadyRegisteredError(
                        f"{record.record_id} was registered by another run while this one was working"
                    )
                fh.seek(0, os.SEEK_END)
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row.as_dict())
            fh.flush()
            os.fsync(fh.fileno())
    return row


class AlreadyRegisteredError(Exception):
    """Raised when a record is inserted twice concurrently."""


def export_csv(destination: Path, path: Path | None = None) -> Path:
    """Copy the registry somewhere you want to keep it."""
    rows = read_all(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return destination
