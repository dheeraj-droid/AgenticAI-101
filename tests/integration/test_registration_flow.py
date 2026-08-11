"""The register-and-notify stage, and duplicate handling in the pipeline.

Model-free: the steps are called directly on a hand-built state, so the whole
registration and mail path is verifiable without an endpoint.
"""

from __future__ import annotations

import pytest

from onboarding.adapters.base import FRAMEWORKS, get_adapter
from onboarding.core import steps
from onboarding.core.audit import JsonlAuditSink, new_run_id
from onboarding.core.registry import append_customer, read_all
from onboarding.core.schemas import Reflection, WelcomeEmail


def _sink(state) -> JsonlAuditSink:
    return JsonlAuditSink(run_id=state.run_id, record_id=state.record.record_id, framework="test")


def _ready_state(record, **overrides):
    """A state that has legitimately reached the delivery stage."""
    state = steps.new_state(record, "run-1", "langgraph")
    sink = _sink(state)
    state = steps.plan(steps.perceive(state, sink), sink)
    state.email = WelcomeEmail(subject="Welcome", body="Hello and welcome aboard.")
    state.reflection = Reflection(violations=[], confidence=1.0, passed=True)
    state.status = "completed"
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


# --- the happy path --------------------------------------------------------


def test_a_clean_run_registers_and_mails(valid_record) -> None:
    state = _ready_state(valid_record)
    sink = _sink(state)
    state = steps.register_customer(state, sink)
    assert state.registered is True

    rows = read_all()
    assert len(rows) == 1
    assert rows[0].customer_name == valid_record.primary_contact.full_name
    assert rows[0].plan == "pro"

    state = steps.send_notifications(state, sink)
    assert len(state.mail_outbox) == 2  # team and customer
    assert any("team" in path for path in state.mail_outbox)
    assert any("customer" in path for path in state.mail_outbox)


def test_registration_is_idempotent(valid_record) -> None:
    state = _ready_state(valid_record)
    sink = _sink(state)
    state = steps.register_customer(state, sink)
    state = steps.register_customer(state, sink)
    assert len(read_all()) == 1


# --- the paths that must not register --------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("status", "blocked_awaiting_approval", "waiting on a human"),
        ("status", "rejected", "a human said no"),
        ("status", "escalated", "escalated"),
    ],
)
def test_unfinished_runs_never_register(valid_record, field, value, why) -> None:
    state = _ready_state(valid_record, **{field: value})
    steps.register_customer(state, _sink(state))
    assert read_all() == [], f"a run that was {why} reached the registry"


def test_a_failed_reflection_never_registers(valid_record) -> None:
    from onboarding.core.schemas import RuleViolation

    state = _ready_state(valid_record)
    state.reflection = Reflection(
        violations=[RuleViolation(rule_id="NO_FABRICATED_DISCOUNTS", detail="invented 20% off")],
        confidence=0.4,
        passed=False,
    )
    steps.register_customer(state, _sink(state))
    assert read_all() == []


def test_an_unapproved_high_risk_run_never_registers(enterprise_record) -> None:
    state = _ready_state(enterprise_record)
    assert state.requires_human_approval()
    steps.register_customer(state, _sink(state))
    assert read_all() == []


def test_an_approved_high_risk_run_does_register(enterprise_record) -> None:
    state = _ready_state(enterprise_record, approval_decision="approve")
    steps.register_customer(state, _sink(state))
    assert len(read_all()) == 1


def test_no_mail_without_registration(valid_record) -> None:
    state = _ready_state(valid_record, status="escalated")
    state = steps.send_notifications(state, _sink(state))
    assert state.mail_outbox == []


# --- duplicates seen by the pipeline ---------------------------------------


async def test_a_second_run_of_the_same_record_is_blocked(valid_record, record_path) -> None:
    """An exact duplicate is a blocking error, so it escalates and never re-registers."""
    append_customer(valid_record, run_id="earlier-run")

    adapter = get_adapter("langgraph")
    result = await adapter.run(
        valid_record,
        run_id=new_run_id(valid_record.record_id, "langgraph"),
        record_path=str(record_path("valid_smb")),
    )
    assert result.status == "escalated"
    assert "ALREADY_REGISTERED" in {f.code for f in result.findings}
    assert result.registered is False
    assert result.llm_calls == 0
    assert len(read_all()) == 1, "the duplicate was written to the registry"


async def test_a_same_company_signup_goes_to_a_human(valid_record, record_path) -> None:
    """A suspected duplicate is a judgement call, so it routes to approval."""
    other = valid_record.model_copy(
        update={
            "record_id": "OTHER-1",
            "company_name": "Brightleaf Coffee Roasters Ltd",
            "primary_contact": valid_record.primary_contact.model_copy(
                update={"email": "someone.else@brightleafcoffee.com"}
            ),
        }
    )
    append_customer(other, run_id="earlier")
    adapter = get_adapter("langgraph")
    result = await adapter.run(
        valid_record,
        run_id=new_run_id(valid_record.record_id, "langgraph"),
        record_path=str(record_path("valid_smb")),
    )
    assert "POSSIBLE_DUPLICATE" in {f.code for f in result.findings}
    assert result.status == "blocked_awaiting_approval"
    assert result.registered is False


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_all_frameworks_agree_on_duplicates(framework, valid_record, record_path) -> None:
    append_customer(valid_record, run_id="earlier")
    adapter = get_adapter(framework)
    result = await adapter.run(
        valid_record,
        run_id=new_run_id(valid_record.record_id, framework),
        record_path=str(record_path("valid_smb")),
    )
    assert result.status == "escalated"
    assert result.registered is False
    assert "ALREADY_REGISTERED" in {f.code for f in result.findings}
