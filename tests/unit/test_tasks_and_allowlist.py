"""Per-customer task checklists, and the guards on real mail delivery.

Two things are being pinned down here:

* a customer's checklist lives in its own CSV, is created pending, and the
  completion counts the agent quotes come from that file rather than the model;
* nothing can be transmitted to a real inbox unless the address was explicitly
  approved, because a demo form accepts whatever address is typed into it.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from email import message_from_string, policy
from pathlib import Path

import pytest

from onboarding.core import tasks
from onboarding.core.audit import JsonlAuditSink
from onboarding.core.mailer import (
    allowlist,
    build_already_registered_mail,
    build_team_mail,
    deliver,
    is_allowed,
    support_address,
)
from onboarding.core.schemas import (
    CommercialTerms,
    Contact,
    CustomerRecord,
    OnboardingTask,
)


def make_record(record_id="T-1", email="ada@testco.com") -> CustomerRecord:
    return CustomerRecord(
        record_id=record_id,
        company_name="Test Co",
        tier="growth",
        region="us",
        primary_contact=Contact(full_name="Ada Lovelace", email=email, phone="+1 415 555 0100"),
        products=["core"],
        commercial_terms=CommercialTerms(
            annual_contract_value_usd=Decimal("50000"),
            contract_start=date(2026, 1, 1),
            term_months=12,
        ),
    )


def make_tasks(n=3) -> list[OnboardingTask]:
    return [
        OnboardingTask(
            task_id=f"task-{i}",
            title=f"Task {i}",
            owner_role="ops",
            due_offset_days=i,
            priority="p1",
        )
        for i in range(n)
    ]


# --- the checklist file ----------------------------------------------------


def test_each_customer_gets_their_own_file() -> None:
    tasks.write_tasks("A-1", make_tasks(2))
    tasks.write_tasks("A-2", make_tasks(5))

    assert tasks.tasks_path("A-1") != tasks.tasks_path("A-2")
    assert tasks.summarise("A-1").total == 2
    assert tasks.summarise("A-2").total == 5


def test_tasks_start_pending() -> None:
    tasks.write_tasks("A-1", make_tasks(3))
    summary = tasks.summarise("A-1")
    assert (summary.total, summary.completed, summary.pending) == (3, 0, 3)
    assert len(tasks.pending_tasks("A-1")) == 3
    assert tasks.completed_tasks("A-1") == []


def test_marking_a_task_moves_it_to_completed() -> None:
    tasks.write_tasks("A-1", make_tasks(3))
    assert tasks.mark("A-1", "task-1") is True

    summary = tasks.summarise("A-1")
    assert (summary.completed, summary.pending) == (1, 2)
    assert [r.task_id for r in tasks.completed_tasks("A-1")] == ["task-1"]
    # The rest of the row survives the rewrite untouched.
    done = tasks.completed_tasks("A-1")[0]
    assert (done.title, done.owner_role, done.priority) == ("Task 1", "ops", "p1")
    assert done.updated_at


def test_marking_an_unknown_task_reports_failure_and_changes_nothing() -> None:
    tasks.write_tasks("A-1", make_tasks(2))
    before = tasks.tasks_path("A-1").read_bytes()
    assert tasks.mark("A-1", "no-such-task") is False
    assert tasks.tasks_path("A-1").read_bytes() == before


def test_a_customer_with_no_checklist_is_not_an_error() -> None:
    """The agent asks about people who may never have been onboarded."""
    assert tasks.read_tasks("nobody") == []
    assert tasks.summarise("nobody").total == 0
    assert "no tasks recorded" in tasks.summarise("nobody").describe("Nobody")


def test_the_completion_sentence_is_computed_not_generated() -> None:
    tasks.write_tasks("A-1", make_tasks(4))
    tasks.mark("A-1", "task-0")
    tasks.mark("A-1", "task-2")
    assert tasks.summarise("A-1").describe("Ada") == (
        "Ada has 2 of 4 tasks completed (2 still pending)."
    )


def test_a_record_id_cannot_escape_the_tasks_directory() -> None:
    """Record ids come off a web form, so they are not trusted as path parts."""
    path = tasks.tasks_path("../../etc/passwd")
    assert path.parent == tasks.tasks_dir()


def test_all_summaries_covers_every_checklist_on_disk() -> None:
    tasks.write_tasks("A-1", make_tasks(1))
    tasks.write_tasks("A-2", make_tasks(2))
    assert {s.record_id: s.total for s in tasks.all_summaries()} == {"A-1": 1, "A-2": 2}


# --- the recipient allowlist ------------------------------------------------


def test_no_allowlist_means_nobody_is_approved(monkeypatch) -> None:
    monkeypatch.delenv("ONBOARDING_ALLOWED_RECIPIENTS", raising=False)
    assert allowlist() == set()
    assert is_allowed("ada@testco.com") is False


def test_the_allowlist_is_matched_case_and_space_insensitively(monkeypatch) -> None:
    monkeypatch.setenv("ONBOARDING_ALLOWED_RECIPIENTS", " Ada@Testco.com , bob@testco.com ")
    assert is_allowed("ada@testco.com") is True
    assert is_allowed("  ADA@TESTCO.COM ") is True
    assert is_allowed("mallory@elsewhere.com") is False


def test_a_star_opens_the_gate_completely(monkeypatch) -> None:
    monkeypatch.setenv("ONBOARDING_ALLOWED_RECIPIENTS", "*")
    assert is_allowed("anyone@anywhere.com") is True


def test_customer_mail_to_an_unapproved_address_is_never_transmitted(monkeypatch) -> None:
    """The important one: a typo in the form must not reach a stranger."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.invalid")
    monkeypatch.setenv("ONBOARDING_ALLOWED_RECIPIENTS", "ada@testco.com")
    sink = JsonlAuditSink(run_id="r", record_id="T-1", framework="maf")

    result = deliver(
        build_already_registered_mail(make_record(email="typo@stranger.com"), "r"),
        sink,
        allow_send=True,
    )

    assert result.sent is False
    assert "ONBOARDING_ALLOWED_RECIPIENTS" in result.reason
    # It still lands in the outbox, so the run is inspectable either way.
    assert result.path is not None and result.path.exists()


def test_team_mail_goes_to_the_configured_support_address(monkeypatch) -> None:
    monkeypatch.setenv("ONBOARDING_SUPPORT_EMAIL", "support@testco.com")
    mail = build_team_mail(make_record(), make_tasks(2), "r", registered=True)
    assert mail.to == "support@testco.com" == support_address()


# --- the duplicate notice ---------------------------------------------------


def test_the_duplicate_notice_needs_no_model() -> None:
    mail = build_already_registered_mail(make_record(), "r")
    assert mail.audience == "customer"
    assert mail.to == "ada@testco.com"
    assert "already" in mail.subject.lower()
    assert "Ada Lovelace" in mail.body
    # Nothing was drafted, so there is no prompt to leak or discount to invent.
    assert "%" not in mail.body


@pytest.mark.parametrize("framework", ["maf", "langchain", "langgraph"])
def test_a_returning_customer_is_told_they_already_signed_up(framework, monkeypatch) -> None:
    """The whole point of the duplicate branch, checked end to end.

    A duplicate is a blocking finding, so the run escalates and never reaches
    the delivery node — the notice has to come off the escalate path instead.
    """
    import asyncio

    from onboarding.adapters.base import get_adapter
    from onboarding.core.registry import append_customer

    record = make_record()
    append_customer(record, run_id="seed")

    result = asyncio.run(get_adapter(framework).run(record, run_id=f"dup-{framework}"))

    assert result.status == "escalated"
    assert result.registered is False
    assert result.llm_calls == 0, "a duplicate must not cost a model call"
    assert len(result.mail_outbox) == 1
    written = message_from_string(
        Path(result.mail_outbox[0]).read_text(encoding="utf-8"), policy=policy.default
    )
    assert written["X-Onboarding-Audience"] == "customer"
    assert "already have an account" in written.get_content()
