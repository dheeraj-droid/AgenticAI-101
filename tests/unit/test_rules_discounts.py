"""The "no fabricated discounts" guarantee.

This is the load-bearing test of the whole project: it proves the rule is
enforced by a validator, not by asking a model politely.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from onboarding.core.discounts import RULE_ID, VALIDATOR, input_number_set, render_allowlist
from onboarding.core.schemas import CommercialTerms, DeclaredDiscount

NO_DISCOUNTS = CommercialTerms(
    annual_contract_value_usd=Decimal("150000"),
    contract_start=date(2026, 9, 1),
    term_months=12,
)

WITH_PERCENT = CommercialTerms(
    annual_contract_value_usd=Decimal("150000"),
    contract_start=date(2026, 9, 1),
    term_months=12,
    declared_discounts=[
        DeclaredDiscount(label="multi-year", kind="percent", value=Decimal("12"), approved_by="rvp")
    ],
)

WITH_WAIVER = CommercialTerms(
    annual_contract_value_usd=Decimal("80000"),
    contract_start=date(2026, 9, 1),
    term_months=12,
    declared_discounts=[
        DeclaredDiscount(label="setup", kind="waived_fee", value=Decimal("0"), approved_by="cfo")
    ],
)

# Phrasings that must be caught when nothing is approved.
FABRICATED = [
    "We're pleased to offer 15% off your first year.",
    "You'll get ten percent off as a welcome gift.",
    "Enjoy half off for the first quarter.",
    "We've applied a $5,000 credit to your account.",
    "That's 2500 USD off the standard rate.",
    "Your first three months are free.",
    "The next 2 months are on us.",
    "We're including a free trial of the analytics module.",
    "We'll waive the setup fee for you.",
    "There is no onboarding fee for your team.",
    "The implementation fee will be waived.",
    "As a special rate just for you, pricing is reduced.",
    "We've added a loyalty bonus to your account.",
    "A rebate will be issued after your first quarter.",
    "You qualify for our promotional pricing.",
]

# Phrasings that must NOT fire — no concession is being claimed.
CLEAN = [
    "Welcome aboard. We're glad to have you.",
    "Your 12-month term begins on 1 September 2026.",
    "Your team will be onboarded across 40 stores.",
    "We'll schedule your kick-off call this week.",
    "Your analytics dashboards will be ready before the autumn push.",
    "The security review must complete before production data is loaded.",
    "Feel free to reply with any questions.",
    "Your named CSM will reach out within two business days.",
]


@pytest.mark.parametrize("text", FABRICATED)
def test_fabricated_claims_are_caught(text: str) -> None:
    violations = VALIDATOR.validate(text, NO_DISCOUNTS)
    assert violations, f"undetected fabricated discount: {text!r}"
    assert all(v.rule_id == RULE_ID for v in violations)


@pytest.mark.parametrize("text", CLEAN)
def test_clean_prose_is_not_flagged(text: str) -> None:
    assert VALIDATOR.validate(text, NO_DISCOUNTS) == [], f"false positive on: {text!r}"


def test_declared_discount_may_be_mentioned() -> None:
    assert VALIDATOR.validate("You're receiving 12% off under your multi-year agreement.", WITH_PERCENT) == []


def test_wrong_amount_is_still_a_violation() -> None:
    """Approving 12% does not license the model to claim 30%."""
    violations = VALIDATOR.validate("You're receiving 30% off.", WITH_PERCENT)
    assert violations and violations[0].rule_id == RULE_ID


def test_declared_waiver_may_be_mentioned() -> None:
    assert VALIDATOR.validate("We've waived the setup fee as agreed.", WITH_WAIVER) == []


def test_contract_value_quoted_from_the_record_is_not_a_discount() -> None:
    from onboarding.core.schemas import Contact, CustomerRecord

    record = CustomerRecord(
        record_id="T-1",
        company_name="Test Co",
        tier="growth",
        region="us",
        primary_contact=Contact(full_name="A B", email="a@b.com"),
        products=["core"],
        commercial_terms=NO_DISCOUNTS,
    )
    text = "Your annual contract value is $150000 and your term is 12 months."
    assert VALIDATOR.validate(text, NO_DISCOUNTS, allowed_numbers=input_number_set(record)) == []


def test_redaction_removes_the_offending_sentence() -> None:
    text = "Welcome to the platform. We're pleased to offer 15% off your first year. Let's book a call."
    violations = VALIDATOR.validate(text, NO_DISCOUNTS)
    redacted = VALIDATOR.redact(text, violations)
    assert "15%" not in redacted
    assert "Welcome to the platform." in redacted
    assert "Let's book a call." in redacted


@pytest.mark.parametrize("text", FABRICATED)
def test_redaction_is_a_hard_guarantee(text: str) -> None:
    """After redaction there is never a surviving violation. This is the invariant
    that makes the promise structural rather than best-effort."""
    body = f"Welcome aboard. {text} We look forward to working with you."
    violations = VALIDATOR.validate(body, NO_DISCOUNTS)
    redacted = VALIDATOR.redact(body, violations)
    assert VALIDATOR.validate(redacted, NO_DISCOUNTS) == []


def test_redaction_is_idempotent() -> None:
    text = "Welcome. Enjoy 20% off. Speak soon."
    once = VALIDATOR.redact(text, VALIDATOR.validate(text, NO_DISCOUNTS))
    twice = VALIDATOR.redact(once, VALIDATOR.validate(once, NO_DISCOUNTS))
    assert once == twice


def test_allowlist_says_none_when_nothing_is_approved() -> None:
    block = render_allowlist(NO_DISCOUNTS)
    assert "NONE" in block
    assert "must not mention" in block


def test_allowlist_itemises_approved_concessions() -> None:
    block = render_allowlist(WITH_PERCENT)
    assert "12% off" in block
    assert "multi-year" in block
    assert "rvp" in block
