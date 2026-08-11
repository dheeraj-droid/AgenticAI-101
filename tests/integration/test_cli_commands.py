"""The CLI commands that need no model.

These exist because a command can rot silently: nothing imports it, so a rename
in ``core`` breaks it and the suite stays green. Every command here is invoked
for real through Typer's runner.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from onboarding.cli.main import app
from onboarding.core.registry import append_customer
from onboarding.core.schemas import CommercialTerms, Contact, CustomerRecord

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich elides table cells to fit the terminal; give it room to print them."""
    monkeypatch.setenv("COLUMNS", "220")


def make_record(record_id="T-1", tier="growth") -> CustomerRecord:
    return CustomerRecord(
        record_id=record_id,
        company_name="Test Co",
        tier=tier,
        region="us",
        primary_contact=Contact(
            full_name="Ada Lovelace", email="ada@testco.com", phone="+44 20 7946 0142"
        ),
        products=["core"],
        commercial_terms=CommercialTerms(
            annual_contract_value_usd=Decimal("1000"),
            contract_start=date(2026, 1, 1),
            term_months=12,
        ),
    )


def invoke(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


# --- commands that must keep working ----------------------------------------


def test_doctor_reports_every_adapter() -> None:
    output = invoke("doctor")
    for framework in ("maf", "langchain", "langgraph", "crew"):
        assert framework in output


def test_registry_show_on_an_empty_registry() -> None:
    assert "empty" in invoke("registry", "show").lower()


def test_registry_show_masks_the_phone_number_by_default() -> None:
    append_customer(make_record())
    output = invoke("registry", "show")

    assert "7946 0142" not in output
    assert "***0142" in output
    assert "Ada" in output and "ada@testco.com" in output


def test_registry_show_reveals_on_request() -> None:
    """A human operator reading their own CSV is not the threat model here."""
    append_customer(make_record())
    assert "0142" in invoke("registry", "show", "--reveal")


def test_registry_show_filters_by_plan() -> None:
    append_customer(make_record("T-1", tier="growth"))
    append_customer(make_record("T-2", tier="starter"))
    assert "T-1" in invoke("registry", "show", "--plan", "pro")
    assert "T-2" not in invoke("registry", "show", "--plan", "pro")


def test_registry_export_writes_a_csv(tmp_path) -> None:
    append_customer(make_record())
    out = tmp_path / "customers.csv"
    invoke("registry", "export", "--out", str(out))
    assert "ada@testco.com" in out.read_text(encoding="utf-8")


def test_prompts_verify_passes() -> None:
    invoke("prompts", "verify")


def test_concepts_lists_the_bindings() -> None:
    assert "PERCEPTION" in invoke("concepts").upper()


def test_outbox_on_an_empty_outbox() -> None:
    invoke("outbox")


@pytest.mark.parametrize("framework", ["maf", "langchain", "langgraph", "crew"])
def test_run_reports_a_blocking_finding_without_a_model(framework) -> None:
    """A record with a blocking error never reaches drafting, in any framework."""
    from pathlib import Path

    from onboarding.core.config import paths

    record = make_record().model_copy(update={"products": []})
    path = Path(paths().root) / "blocked.json"
    path.write_text(record.model_dump_json(), encoding="utf-8")

    output = invoke("run", "-f", framework, "-r", str(path))
    assert "NO_PRODUCTS" in output or "escalated" in output
