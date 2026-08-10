"""Validation, risk thresholds, confidence fallback, planning, chunking, tone."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from onboarding.core.chunking import chunk_notes, estimate_tokens
from onboarding.core.confidence import below_threshold, escalation_reasons, score_confidence
from onboarding.core.planning import decompose, derive_tasks, rewrite_query
from onboarding.core.risk import assess_risk
from onboarding.core.rules import RULES
from onboarding.core.schemas import (
    CommercialTerms,
    Contact,
    CustomerRecord,
    Finding,
    InjectionSignal,
    RuleViolation,
)
from onboarding.core.tone import extract_slots, jaccard, validate_tone
from onboarding.core.validation import has_errors, has_warnings, validate_record


def make_record(**overrides) -> CustomerRecord:
    base = {
        "record_id": "T-1",
        "company_name": "Test Co",
        "tier": "growth",
        "region": "us",
        "primary_contact": Contact(full_name="Ada Lovelace", email="ada@test.co", phone="+1 415 555 0100"),
        "products": ["core"],
        "commercial_terms": CommercialTerms(
            annual_contract_value_usd=Decimal("50000"),
            contract_start=date(2026, 1, 1),
            term_months=12,
        ),
    }
    base.update(overrides)
    return CustomerRecord(**base)


# --- validation ------------------------------------------------------------


def test_valid_record_has_no_errors(valid_record) -> None:
    assert not has_errors(validate_record(valid_record))


def test_invalid_record_reports_every_problem(invalid_record) -> None:
    codes = {f.code for f in validate_record(invalid_record)}
    assert {"MISSING_CONTACT_NAME", "NO_PRODUCTS", "ZERO_CONTRACT_VALUE", "UNAPPROVED_DISCOUNT"} <= codes
    assert has_errors(validate_record(invalid_record))


def test_findings_are_sorted_for_stable_comparison(invalid_record) -> None:
    findings = validate_record(invalid_record)
    assert findings == sorted(findings, key=lambda f: (f.code, f.field_path))


def test_go_live_in_the_past_is_a_warning() -> None:
    record = make_record(requested_go_live=date(2020, 1, 1))
    codes = {f.code for f in validate_record(record, today=date(2026, 1, 1))}
    assert "GO_LIVE_IN_PAST" in codes
    assert has_warnings(validate_record(record, today=date(2026, 1, 1)))


def test_go_live_too_soon_is_a_warning() -> None:
    today = date(2026, 1, 1)
    record = make_record(requested_go_live=today + timedelta(days=3))
    assert "GO_LIVE_TOO_SOON" in {f.code for f in validate_record(record, today=today)}


# --- risk ------------------------------------------------------------------


def test_enterprise_tier_always_needs_approval() -> None:
    record = make_record(tier="enterprise")
    risk = assess_risk(record, [], [])
    assert risk.requires_human_approval and risk.tier_risk


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (RULES.HIGH_VALUE_ACV_USD - 1, False),
        (RULES.HIGH_VALUE_ACV_USD, True),  # the boundary is inclusive
        (RULES.HIGH_VALUE_ACV_USD + 1, True),
    ],
)
def test_high_value_threshold_boundary(value: Decimal, expected: bool) -> None:
    record = make_record(
        commercial_terms=CommercialTerms(
            annual_contract_value_usd=Decimal(value),
            contract_start=date(2026, 1, 1),
            term_months=12,
        )
    )
    assert assess_risk(record, [], []).value_risk is expected


def test_injection_forces_approval() -> None:
    signal = InjectionSignal(
        pattern_id="IGNORE_PREVIOUS", matched_span="ignore previous", field_path="signup_notes", severity="block"
    )
    risk = assess_risk(make_record(), [], [signal])
    assert risk.requires_human_approval and risk.injection_risk


def test_low_risk_record_runs_autonomously(valid_record) -> None:
    risk = assess_risk(valid_record, validate_record(valid_record), [])
    assert not risk.requires_human_approval
    assert risk.band == "low"


# --- confidence ------------------------------------------------------------


def test_clean_run_scores_full_confidence() -> None:
    assert score_confidence(findings=[], injection_signals=[], violations=[]) == 1.0


def test_violations_reduce_confidence() -> None:
    violations = [RuleViolation(rule_id="TONE", detail="cold")]
    assert score_confidence(findings=[], injection_signals=[], violations=violations) < 1.0


def test_confidence_never_goes_negative() -> None:
    findings = [Finding(code="X", severity="error", field_path="a", message="m") for _ in range(20)]
    assert score_confidence(findings=findings, injection_signals=[], violations=[]) == 0.0


def test_threshold_routes_to_escalation() -> None:
    assert below_threshold(RULES.MIN_CONFIDENCE - 0.01)
    assert not below_threshold(RULES.MIN_CONFIDENCE)
    assert escalation_reasons(RULES.MIN_CONFIDENCE - 0.01, [])


# --- planning --------------------------------------------------------------


def test_plan_is_least_to_most_ordered(valid_record) -> None:
    risk = assess_risk(valid_record, [], [])
    plan = decompose(valid_record, risk, [])
    orders = [s.order for s in plan.steps]
    assert orders == sorted(orders)
    for step in plan.steps:
        assert all(dep < step.order for dep in step.depends_on), "a step depends on a later step"


def test_blocking_errors_switch_to_the_remediation_track(invalid_record) -> None:
    findings = validate_record(invalid_record)
    plan = decompose(invalid_record, assess_risk(invalid_record, findings, []), findings)
    assert plan.strategy == "remediation"
    assert not any("welcome email" in s.goal.lower() for s in plan.steps)


def test_enterprise_plan_inserts_the_approval_step(enterprise_record) -> None:
    risk = assess_risk(enterprise_record, [], [])
    plan = decompose(enterprise_record, risk, [])
    assert plan.strategy == "enterprise"
    assert any("human approval" in s.goal.lower() for s in plan.steps)


def test_query_rewrite_expands_context(valid_record) -> None:
    rewritten = rewrite_query(valid_record, "standard")
    assert valid_record.tier in rewritten
    assert "US" in rewritten
    for product in valid_record.products:
        assert product in rewritten


def test_task_list_is_deterministic(valid_record) -> None:
    findings = validate_record(valid_record)
    risk = assess_risk(valid_record, findings, [])
    first = derive_tasks(valid_record, findings, risk)
    second = derive_tasks(valid_record, findings, risk)
    assert [t.task_id for t in first] == [t.task_id for t in second]
    assert all(t.origin == "rule" for t in first)


def test_task_ids_are_unique(enterprise_record) -> None:
    findings = validate_record(enterprise_record)
    tasks = derive_tasks(enterprise_record, findings, assess_risk(enterprise_record, findings, []))
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_eu_region_adds_gdpr_tasks(enterprise_record) -> None:
    findings = validate_record(enterprise_record)
    ids = {t.task_id for t in derive_tasks(enterprise_record, findings, assess_risk(enterprise_record, findings, []))}
    assert {"gdpr-dpa", "set-eu-residency"} <= ids


def test_approval_task_added_when_human_needed(enterprise_record) -> None:
    findings = validate_record(enterprise_record)
    risk = assess_risk(enterprise_record, findings, [])
    assert "obtain-human-approval" in {t.task_id for t in derive_tasks(enterprise_record, findings, risk)}


# --- chunking --------------------------------------------------------------


def test_short_notes_are_a_single_chunk() -> None:
    chunks = chunk_notes("One short sentence about onboarding.")
    assert len(chunks) == 1


def test_long_notes_are_split_with_overlap() -> None:
    text = " ".join(f"This is sentence number {i} about the onboarding process." for i in range(60))
    chunks = chunk_notes(text, target_tokens=60, overlap_tokens=15)
    assert len(chunks) > 1
    assert all(c.token_estimate > 0 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_notes_produce_no_chunks() -> None:
    assert chunk_notes("") == []
    assert chunk_notes("   ") == []


def test_token_estimate_is_positive_for_text() -> None:
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0


# --- tone ------------------------------------------------------------------


GOOD_BODY = (
    "Hi there, welcome aboard. We are delighted to have you on the platform and are looking "
    "forward to getting your team up and running quickly. Your workspace is being prepared now, "
    "and your named contact will be in touch shortly to walk you through the setup in detail. "
    "As a next step, please reply with a couple of times that suit you for the onboarding call "
    "and we will get it scheduled. Thanks again for choosing us. Regards, The Onboarding Team"
)


def test_good_draft_passes_tone() -> None:
    assert validate_tone("Welcome to the platform", GOOD_BODY) == []


def test_banned_phrase_is_caught() -> None:
    body = GOOD_BODY.replace("Hi there", "Dear Sir or Madam")
    assert any("banned phrase" in v.detail for v in validate_tone("Welcome", body))


def test_cold_draft_is_caught() -> None:
    body = "Your account exists. Please proceed to the next step when convenient. " * 8
    assert any("cold" in v.detail for v in validate_tone("Account", body))


def test_too_short_is_caught() -> None:
    assert any("too short" in v.detail for v in validate_tone("Welcome", "Welcome! Thanks. Next step: reply."))


def test_shouty_draft_is_caught() -> None:
    body = GOOD_BODY + " ACTION REQUIRED IMMEDIATE URGENT RESPONSE NEEDED"
    assert any("shouty" in v.detail for v in validate_tone("Welcome", body))


def test_slot_extraction_is_structural() -> None:
    slots = extract_slots("Welcome to Test Co", GOOD_BODY, "Test Co")
    assert slots["has_greeting"] and slots["has_next_steps"] and slots["has_warmth"]


def test_jaccard_bounds() -> None:
    assert jaccard(GOOD_BODY, GOOD_BODY) == 1.0
    assert jaccard("", GOOD_BODY) == 0.0
    assert 0.0 <= jaccard("welcome aboard the platform", GOOD_BODY) <= 1.0
