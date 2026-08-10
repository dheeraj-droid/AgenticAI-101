"""Live tests. Run these with an endpoint configured:

    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b-instruct \
    LLM_API_KEY=ollama uv run pytest -m llm

They skip (never silently pass) when no endpoint is set — there is no stub model
in this project, so without one they are simply not runnable.

Prose is never compared verbatim. Tier 1 asserts the deterministic outcome is
identical; Tier 2 asserts invariants that must hold for any model; Tier 3
measures cross-framework agreement loosely enough to survive a 3B local model.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from onboarding.adapters.base import FRAMEWORKS, get_adapter
from onboarding.core.audit import new_run_id
from onboarding.core.discounts import VALIDATOR, input_number_set
from onboarding.core.pii import contains_raw_pii
from onboarding.core.rules import RULES
from onboarding.core.tone import extract_slots, jaccard, validate_tone

pytestmark = pytest.mark.llm

# Fixtures that actually reach the drafting step.
DRAFTING_FIXTURES = ["valid_smb", "pii_heavy"]

# Deliberately low: a 3B local model and a frontier model must both clear it.
# The measured values are printed in the comparison report so it can be tuned
# from evidence rather than guesswork.
MIN_PAIRWISE_JACCARD = 0.20


async def _run_all(record, path):
    results = {}
    for framework in FRAMEWORKS:
        adapter = get_adapter(framework)
        results[framework] = await adapter.run(
            record, run_id=new_run_id(record.record_id, framework), record_path=str(path)
        )
    return results


# --- Tier 1: exact equality on everything policy decides -------------------


@pytest.mark.parametrize("fixture", DRAFTING_FIXTURES)
async def test_deterministic_outcomes_identical_live(fixture, record, record_path, llm_configured) -> None:
    results = await _run_all(record(fixture), record_path(fixture))
    outcomes = [r.deterministic() for r in results.values()]
    assert all(o == outcomes[0] for o in outcomes), (
        f"frameworks diverged on {fixture}: "
        f"{ {fw: r.deterministic().model_dump() for fw, r in results.items()} }"
    )


@pytest.mark.parametrize("fixture", DRAFTING_FIXTURES)
async def test_every_framework_produces_an_email(fixture, record, record_path, llm_configured) -> None:
    results = await _run_all(record(fixture), record_path(fixture))
    for framework, result in results.items():
        assert result.welcome_email is not None, f"{framework} produced no draft"
        assert result.welcome_email.subject.strip(), f"{framework} produced an empty subject"
        assert result.llm_calls >= 1


# --- Tier 2: invariants any model must satisfy -----------------------------


@pytest.mark.parametrize("fixture", DRAFTING_FIXTURES)
async def test_no_fabricated_discounts_live(fixture, record, record_path, llm_configured) -> None:
    """The business rule that matters most, checked on real model output."""
    customer = record(fixture)
    results = await _run_all(customer, record_path(fixture))
    for framework, result in results.items():
        email = result.welcome_email
        assert email is not None
        text = f"{email.subject}\n{email.body}"
        violations = VALIDATOR.validate(
            text, customer.commercial_terms, allowed_numbers=input_number_set(customer)
        )
        assert violations == [], f"{framework} shipped an ungrounded commercial claim: {violations}"


@pytest.mark.parametrize("fixture", DRAFTING_FIXTURES)
async def test_no_pii_leaks_into_prose(fixture, record, record_path, llm_configured) -> None:
    customer = record(fixture)
    results = await _run_all(customer, record_path(fixture))
    for framework, result in results.items():
        text = f"{result.welcome_email.subject}\n{result.welcome_email.body}"
        assert contains_raw_pii(text, customer) == [], f"{framework} leaked PII into the draft"


@pytest.mark.parametrize("fixture", DRAFTING_FIXTURES)
async def test_tone_and_length_hold(fixture, record, record_path, llm_configured) -> None:
    results = await _run_all(record(fixture), record_path(fixture))
    for framework, result in results.items():
        email = result.welcome_email
        assert RULES.MIN_EMAIL_WORDS <= email.word_count <= RULES.MAX_EMAIL_WORDS, (
            f"{framework} produced {email.word_count} words"
        )
        assert validate_tone(email.subject, email.body) == [], f"{framework} breached tone policy"


async def test_company_name_is_grounded(record, record_path, llm_configured) -> None:
    customer = record("valid_smb")
    results = await _run_all(customer, record_path("valid_smb"))
    for framework, result in results.items():
        text = f"{result.welcome_email.subject}\n{result.welcome_email.body}".lower()
        assert customer.company_name.lower() in text, f"{framework} did not mention the customer"


# --- Tier 3: loose cross-framework prose agreement -------------------------


async def test_frameworks_agree_on_structure(record, record_path, llm_configured) -> None:
    """Not text equality — the same structural slots must be filled."""
    customer = record("valid_smb")
    results = await _run_all(customer, record_path("valid_smb"))
    slots = {
        fw: extract_slots(r.welcome_email.subject, r.welcome_email.body, customer.company_name)
        for fw, r in results.items()
    }
    for framework, values in slots.items():
        assert values["has_next_steps"], f"{framework} gave the customer no next step"
        assert values["has_warmth"], f"{framework} produced a cold email"
        assert values["mentions_company"], f"{framework} never named the company"


async def test_prose_overlap_is_plausible(record, record_path, llm_configured) -> None:
    """A floor, not a target: catches a framework silently prompting with
    different context, without demanding identical wording."""
    results = await _run_all(record("valid_smb"), record_path("valid_smb"))
    bodies = {fw: r.welcome_email.body for fw, r in results.items()}
    for a, b in combinations(sorted(bodies), 2):
        score = jaccard(bodies[a], bodies[b])
        assert score >= MIN_PAIRWISE_JACCARD, f"{a} vs {b} overlap only {score:.2f}"


# --- injection defense against a real model --------------------------------


async def test_injected_instructions_are_not_obeyed(record, record_path, llm_configured) -> None:
    """The poisoned record blocks before drafting, so the payload never reaches
    a model at all — the strongest possible outcome."""
    customer = record("injection_attempt")
    results = await _run_all(customer, record_path("injection_attempt"))
    for framework, result in results.items():
        assert result.status == "blocked_awaiting_approval", framework
        assert result.welcome_email is None, f"{framework} drafted from a poisoned record"


# --- resume, end to end ----------------------------------------------------


@pytest.mark.parametrize("framework", ["maf", "langgraph"])
async def test_approve_resume_produces_an_email(
    framework, enterprise_record, record_path, llm_configured
) -> None:
    """The full HITL round trip: block, approve later, then draft."""
    from onboarding.core.schemas import ApprovalDecision

    adapter = get_adapter(framework)
    blocked = await adapter.run(
        enterprise_record,
        run_id=new_run_id(enterprise_record.record_id, framework),
        record_path=str(record_path("enterprise_high_value")),
    )
    assert blocked.status == "blocked_awaiting_approval"

    resumed = await adapter.resume(
        blocked.run_id, ApprovalDecision(decision="approve", decided_by="tester")
    )
    assert resumed.status in ("completed", "escalated")
    assert resumed.welcome_email is not None
    text = f"{resumed.welcome_email.subject}\n{resumed.welcome_email.body}"
    assert VALIDATOR.validate(
        text,
        enterprise_record.commercial_terms,
        allowed_numbers=input_number_set(enterprise_record),
    ) == []
