"""The customer registry, duplicate detection, and mail delivery."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from onboarding.core.audit import JsonlAuditSink
from onboarding.core.mailer import (
    build_customer_mail,
    build_team_mail,
    deliver,
    list_outbox,
    personalise,
)
from onboarding.core.registry import (
    AlreadyRegisteredError,
    append_customer,
    export_csv,
    find_customer,
    find_duplicate,
    plan_breakdown,
    read_all,
    registry_path,
)
from onboarding.core.schemas import (
    CommercialTerms,
    Contact,
    CustomerRecord,
    OnboardingTask,
    WelcomeEmail,
)


def make_record(
    record_id="T-1", company="Test Co", name="Ada Lovelace", email="ada@testco.com", tier="growth", plan=None
) -> CustomerRecord:
    return CustomerRecord(
        record_id=record_id,
        company_name=company,
        tier=tier,
        region="us",
        primary_contact=Contact(full_name=name, email=email, phone="+1 415 555 0100"),
        products=["core"],
        commercial_terms=CommercialTerms(
            annual_contract_value_usd=Decimal("50000"),
            contract_start=date(2026, 1, 1),
            term_months=12,
        ),
        plan=plan,
    )


# --- plan derivation -------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "expected"), [("starter", "free"), ("growth", "pro"), ("enterprise", "pro+")]
)
def test_plan_defaults_from_tier(tier, expected) -> None:
    assert make_record(tier=tier).effective_plan == expected


def test_explicit_plan_wins() -> None:
    assert make_record(tier="starter", plan="pro+").effective_plan == "pro+"


def test_existing_fixtures_still_parse(valid_record, enterprise_record) -> None:
    """The plan field is optional, so no fixture needed editing."""
    assert valid_record.effective_plan == "pro"
    assert enterprise_record.effective_plan == "pro+"


# --- registry --------------------------------------------------------------


def test_registry_starts_empty() -> None:
    assert read_all() == []
    assert plan_breakdown().total == 0


def test_append_and_read_back() -> None:
    row = append_customer(make_record(), run_id="run-1")
    assert row.plan == "pro"
    rows = read_all()
    assert len(rows) == 1
    assert rows[0].customer_name == "Ada Lovelace"
    assert rows[0].email == "ada@testco.com"
    assert registry_path().exists()


def test_header_written_once() -> None:
    append_customer(make_record("T-1", email="a@one.com"), run_id="r1")
    append_customer(make_record("T-2", company="Other Co", email="b@two.com"), run_id="r2")
    text = registry_path().read_text()
    assert text.count("record_id,customer_name") == 1
    assert len(read_all()) == 2


def test_plan_breakdown_counts() -> None:
    append_customer(make_record("T-1", company="A", email="a@a.com", tier="starter"))
    append_customer(make_record("T-2", company="B", email="b@b.com", tier="growth"))
    append_customer(make_record("T-3", company="C", email="c@c.com", tier="enterprise"))
    append_customer(make_record("T-4", company="D", email="d@d.com", tier="growth"))

    counts = plan_breakdown()
    assert counts.total == 4
    assert counts.of("free") == 1
    assert counts.of("pro") == 2
    assert counts.of("pro+") == 1
    assert counts.of("pro plus") == 1  # alias normalisation


def test_find_customer_by_several_keys() -> None:
    append_customer(make_record("T-9", company="Brightleaf", name="Dana Wu", email="dana@brightleaf.com"))
    assert find_customer("dana")
    assert find_customer("brightleaf")
    assert find_customer("T-9")
    assert find_customer("nobody") == []


def test_export_writes_a_readable_csv(tmp_path) -> None:
    append_customer(make_record())
    destination = export_csv(tmp_path / "out.csv")
    text = destination.read_text()
    assert "Ada Lovelace" in text
    assert text.splitlines()[0].startswith("record_id,")


# --- duplicate detection ---------------------------------------------------


def test_no_duplicate_in_an_empty_registry() -> None:
    assert find_duplicate(make_record()).is_duplicate is False


def test_same_record_id_is_an_exact_duplicate() -> None:
    append_customer(make_record("T-1"))
    match = find_duplicate(make_record("T-1", company="Renamed", email="new@elsewhere.com"))
    assert match.kind == "record_id"


def test_same_email_is_an_exact_duplicate() -> None:
    append_customer(make_record("T-1", email="ada@testco.com"))
    match = find_duplicate(make_record("T-2", company="Different", email="ADA@testco.com"))
    assert match.kind == "email"


def test_same_company_is_a_suspicion_not_a_verdict() -> None:
    """A second team at the same company is legitimate — a human decides."""
    append_customer(make_record("T-1", company="Meridian Freight Group"))
    match = find_duplicate(make_record("T-2", company="Meridian Freight Ltd", email="other@meridian.eu"))
    assert match.kind == "company"


def test_shared_domain_flags_a_suspicion() -> None:
    append_customer(make_record("T-1", company="Alpha", email="ada@acme.io"))
    match = find_duplicate(make_record("T-2", company="Beta", email="bob@acme.io"))
    assert match.kind == "company"


def test_generic_domains_do_not_trigger_false_duplicates() -> None:
    append_customer(make_record("T-1", company="Alpha", email="ada@gmail.com"))
    match = find_duplicate(make_record("T-2", company="Beta", email="bob@gmail.com"))
    assert match.is_duplicate is False


def test_concurrent_insert_of_the_same_id_is_refused() -> None:
    append_customer(make_record("T-1"))
    with pytest.raises(AlreadyRegisteredError):
        append_customer(make_record("T-1"))


# --- mail ------------------------------------------------------------------


def _sink() -> JsonlAuditSink:
    return JsonlAuditSink(run_id="run-1", record_id="T-1", framework="test")


def test_team_mail_lists_the_checklist() -> None:
    tasks = [
        OnboardingTask(
            task_id="provision-workspace",
            title="Provision customer workspace",
            owner_role="ops",
            due_offset_days=2,
            priority="p0",
        )
    ]
    mail = build_team_mail(make_record(), tasks, "run-1", registered=True)
    assert mail.audience == "team"
    assert "Provision customer workspace" in mail.body
    assert "pro" in mail.body


def test_customer_mail_resolves_masked_placeholders() -> None:
    record = make_record(name="Ada Lovelace", email="ada@testco.com")
    email = WelcomeEmail(subject="Welcome, <PERSON_1>", body="Hi <PERSON_1>, we reach you at <EMAIL_ADDRESS_1>.")
    mail = build_customer_mail(record, email, "run-1")
    assert "<PERSON_1>" not in mail.body
    assert "Ada Lovelace" in mail.body
    assert "ada@testco.com" in mail.body
    assert mail.to == "ada@testco.com"


def test_unknown_placeholders_are_left_alone() -> None:
    assert "<PERSON_9>" in personalise("Hi <PERSON_9>", make_record())


def test_delivery_writes_to_the_outbox_and_does_not_send() -> None:
    mail = build_team_mail(make_record(), [], "run-1", registered=True)
    result = deliver(mail, _sink())
    assert result.sent is False
    assert result.path is not None and result.path.exists()
    assert "outbox" in result.reason
    assert list_outbox()


def test_send_without_smtp_host_still_does_not_transmit(monkeypatch) -> None:
    monkeypatch.delenv("SMTP_HOST", raising=False)
    result = deliver(build_team_mail(make_record(), [], "r", registered=True), _sink(), allow_send=True)
    assert result.sent is False
    assert "SMTP_HOST" in result.reason


def test_customer_mail_is_never_transmitted_without_approval(monkeypatch) -> None:
    """The guard that stops a demo run emailing a real person."""
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid")
    email = WelcomeEmail(subject="Welcome", body="Hello there")
    mail = build_customer_mail(make_record(), email, "run-1")
    result = deliver(mail, _sink(), allow_send=True, approved=False)
    assert result.sent is False
    assert "approved" in result.reason


def test_delivery_is_audited() -> None:
    sink = _sink()
    deliver(build_team_mail(make_record(), [], "run-1", registered=True), sink)
    events = [e for e in sink.read_all() if e.event_type == "mail_sent"]
    assert events and events[-1].payload["transmitted"] is False
