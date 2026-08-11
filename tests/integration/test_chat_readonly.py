"""The chat agent can read the registry and nothing else.

Two guarantees, both structural rather than prompt-based:

1. **No write capability.** No tool handed to a model can insert, edit or delete
   a customer, send mail, or approve a run. Checked by walking the call graph of
   every exposed tool, so adding a write-capable tool fails the build.
2. **Contact details stay masked.** Names, companies and plans reach the model;
   email addresses and phone numbers never do.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from onboarding.chat import tools as chat_tools
from onboarding.core import qa
from onboarding.core.registry import append_customer
from onboarding.core.schemas import CommercialTerms, Contact, CustomerRecord

SRC = Path(__file__).resolve().parents[2] / "src" / "onboarding"

# Anything that changes state. None of these may be reachable from a chat tool.
FORBIDDEN_CALLS = {
    "append_customer",
    "export_csv",
    "deliver",
    "_transmit",
    "record_approval_required",
    "apply_decision",
    "register_customer",
    "send_notifications",
    "emit",
    "put",
    "write_text",
    "open",
}


def make_record(record_id="C-1", company="Acme", name="Ada Lovelace", email="ada@acme.com", tier="growth"):
    return CustomerRecord(
        record_id=record_id,
        company_name=company,
        tier=tier,
        region="us",
        primary_contact=Contact(full_name=name, email=email, phone="+44 20 7946 0142"),
        products=["core"],
        commercial_terms=CommercialTerms(
            annual_contract_value_usd=Decimal("1000"),
            contract_start=date(2026, 1, 1),
            term_months=12,
        ),
    )


# --- 1. read-only -----------------------------------------------------------


@pytest.mark.parametrize("fn", chat_tools.READ_ONLY_TOOLS, ids=lambda f: f.__name__)
def test_no_chat_tool_can_write(fn) -> None:
    """Walk each tool's body for a call that mutates anything."""
    tree = ast.parse(inspect.getsource(fn))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        assert name not in FORBIDDEN_CALLS, f"{fn.__name__} calls {name!r}, which mutates state"


def test_qa_module_never_imports_the_write_path() -> None:
    """core.qa is the only thing the chat tools reach; it must stay read-only."""
    tree = ast.parse((SRC / "core" / "qa.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert "append_customer" not in imported
    assert "export_csv" not in imported
    for banned in ("mailer", "hitl"):
        assert banned not in ast.dump(tree), f"core.qa reaches {banned}"


def test_chat_tools_never_import_the_write_path() -> None:
    tree = ast.parse((SRC / "chat" / "tools.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "mailer" not in node.module, "a chat tool can reach the mailer"
            for alias in node.names:
                assert alias.name != "append_customer", "a chat tool can write to the registry"


def test_the_registry_is_unchanged_after_answering_questions() -> None:
    """Belt and braces: exercise every tool, then check the file is untouched."""
    from onboarding.core.registry import registry_path

    append_customer(make_record("C-1"))
    rows_before = qa.all_customers()
    bytes_before = registry_path().read_bytes()

    chat_tools.count_customers_by_plan()
    chat_tools.list_customers_on_plan("pro")
    chat_tools.find_customer_details("Ada")
    chat_tools.list_all_customers()
    chat_tools.describe_registry()

    assert qa.all_customers() == rows_before
    assert registry_path().read_bytes() == bytes_before, "a chat tool modified the registry file"


# --- 2. masking -------------------------------------------------------------


def test_email_and_phone_are_masked_for_the_model() -> None:
    append_customer(make_record(email="ada@acme.com"))
    row = qa.all_customers()[0]
    assert "ada@acme.com" not in row.email
    assert row.email == "a***@a***.com"
    assert "7946" not in row.phone


def test_name_company_and_plan_stay_visible() -> None:
    """The agreed policy: these are business facts, not personal identifiers."""
    append_customer(make_record(name="Ada Lovelace", company="Acme", tier="growth"))
    row = qa.all_customers()[0]
    assert row.customer_name == "Ada Lovelace"
    assert row.company_name == "Acme"
    assert row.plan == "pro"


def test_no_raw_contact_details_in_any_tool_output() -> None:
    append_customer(make_record(email="ada@acme.com"))
    blob = " ".join(
        [
            chat_tools.count_customers_by_plan(),
            chat_tools.list_customers_on_plan("pro"),
            chat_tools.find_customer_details("Ada"),
            chat_tools.list_all_customers(),
            chat_tools.describe_registry(),
        ]
    )
    assert "ada@acme.com" not in blob
    assert "7946 0142" not in blob
    assert "+44 20 7946 0142" not in blob


def test_registry_context_is_masked() -> None:
    append_customer(make_record(email="ada@acme.com"))
    assert "ada@acme.com" not in qa.registry_context()


# --- counting is done in code, not by the model -----------------------------


def test_plan_counts_are_exact() -> None:
    for i, tier in enumerate(["starter", "growth", "growth", "enterprise"]):
        append_customer(make_record(f"C-{i}", company=f"Co{i}", email=f"u{i}@co{i}.com", tier=tier))
    described = chat_tools.count_customers_by_plan()
    assert "4 customer(s)" in described
    assert "1 on free" in described
    assert "2 on pro" in described
    assert "1 on pro+" in described


def test_empty_registry_says_so() -> None:
    assert "empty" in chat_tools.count_customers_by_plan().lower()
    assert "empty" in chat_tools.list_all_customers().lower()


def test_unknown_customer_is_reported_not_invented() -> None:
    append_customer(make_record())
    assert "No customer matches" in chat_tools.find_customer_details("Nobody Here")


# --- the tool surface itself ------------------------------------------------


def test_tool_surface_is_stable_and_documented() -> None:
    assert set(chat_tools.TOOL_NAMES) == {
        "count_customers_by_plan",
        "list_customers_on_plan",
        "find_customer_details",
        "list_all_customers",
        "describe_registry",
    }
    for fn in chat_tools.READ_ONLY_TOOLS:
        assert fn.__doc__, f"{fn.__name__} has no docstring for the model to read"
