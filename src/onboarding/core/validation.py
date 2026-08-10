"""Deterministic validation of a customer record.

Pure function of the record — no model, no I/O. All three frameworks call this,
so the ``findings`` list is identical across them and is safe to assert on
exactly in the golden tests.
"""

from __future__ import annotations

import re
from datetime import date

from onboarding.core.concepts import Concept, concept
from onboarding.core.rules import RULES
from onboarding.core.schemas import CustomerRecord, Finding

_PHONE_OK = re.compile(r"^\+?[\d\s().-]{7,20}$")
KNOWN_PRODUCTS = {"analytics", "api", "sso", "support-premium", "core", "reporting"}


@concept(Concept.PERCEPTION, Concept.POLICY_CONSTRAINED)
def validate_record(record: CustomerRecord, *, today: date | None = None) -> list[Finding]:
    """Return every validation finding, sorted by code for stable comparison."""
    today = today or date.today()
    findings: list[Finding] = []

    def add(code: str, severity: str, field_path: str, message: str) -> None:
        findings.append(
            Finding(code=code, severity=severity, field_path=field_path, message=message)  # type: ignore[arg-type]
        )

    if not record.company_name.strip():
        add("MISSING_COMPANY_NAME", "error", "company_name", "company name is empty")
    elif len(record.company_name.strip()) < 2:
        add("SUSPICIOUS_COMPANY_NAME", "warning", "company_name", "company name is implausibly short")

    contact = record.primary_contact
    if not contact.full_name.strip():
        add("MISSING_CONTACT_NAME", "error", "primary_contact.full_name", "primary contact has no name")
    if contact.phone and not _PHONE_OK.match(contact.phone):
        add(
            "MALFORMED_CONTACT_PHONE",
            "warning",
            "primary_contact.phone",
            "primary contact phone is not a recognisable phone number",
        )
    if not contact.phone:
        add(
            "MISSING_CONTACT_PHONE",
            "info",
            "primary_contact.phone",
            "no phone number for the primary contact",
        )

    if not record.products:
        add("NO_PRODUCTS", "error", "products", "no products are attached to this record")
    else:
        unknown = sorted({p for p in record.products if p.lower() not in KNOWN_PRODUCTS})
        if unknown:
            add(
                "UNKNOWN_PRODUCT",
                "warning",
                "products",
                f"unrecognised product code(s): {', '.join(unknown)}",
            )

    terms = record.commercial_terms
    if terms.annual_contract_value_usd <= 0:
        add(
            "ZERO_CONTRACT_VALUE",
            "error",
            "commercial_terms.annual_contract_value_usd",
            "annual contract value must be greater than zero",
        )
    if record.tier == "enterprise" and terms.annual_contract_value_usd < RULES.HIGH_VALUE_ACV_USD:
        add(
            "TIER_VALUE_MISMATCH",
            "warning",
            "commercial_terms.annual_contract_value_usd",
            "enterprise tier with a contract value below the enterprise threshold",
        )
    if terms.contract_start > today:
        add(
            "CONTRACT_START_IN_FUTURE",
            "info",
            "commercial_terms.contract_start",
            "the contract start date is in the future",
        )

    for d in terms.declared_discounts:
        if not d.approved_by.strip():
            add(
                "UNAPPROVED_DISCOUNT",
                "error",
                "commercial_terms.declared_discounts",
                f"declared discount '{d.label}' has no approver",
            )
        if d.kind == "percent" and not (0 < d.value <= 100):
            add(
                "IMPLAUSIBLE_DISCOUNT",
                "error",
                "commercial_terms.declared_discounts",
                f"declared discount '{d.label}' has an implausible percentage",
            )

    if record.requested_go_live:
        if record.requested_go_live < today:
            add(
                "GO_LIVE_IN_PAST",
                "warning",
                "requested_go_live",
                "the requested go-live date is in the past",
            )
        elif (record.requested_go_live - today).days < 7:
            add(
                "GO_LIVE_TOO_SOON",
                "warning",
                "requested_go_live",
                "the requested go-live date leaves less than a week to onboard",
            )

    return sorted(findings, key=lambda f: (f.code, f.field_path))


def has_errors(findings: list[Finding]) -> bool:
    return any(f.severity == "error" for f in findings)


def has_warnings(findings: list[Finding]) -> bool:
    return any(f.severity == "warning" for f in findings)
