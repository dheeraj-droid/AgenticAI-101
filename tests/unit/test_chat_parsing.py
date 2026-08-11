"""Parsing of framework-specific response shapes.

These test *our* extraction helpers against the shapes the SDKs actually
produce — they do not stand in for a model. The real conversations are covered
by ``tests/llm/test_chat_live.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from onboarding.chat.session import _flatten, _last_text, _maf_tool_names, _tool_names

# --- MAF: contents are discriminated by a `type` field ---------------------


@dataclass
class FakeContent:
    type: str
    name: str | None = None
    text: str | None = None


@dataclass
class FakeMessage:
    role: str = "assistant"
    contents: list[FakeContent] = field(default_factory=list)


@dataclass
class FakeMafResponse:
    messages: list[FakeMessage] = field(default_factory=list)


def test_maf_tool_names_match_on_the_type_discriminator() -> None:
    """MAF identifies a tool call by content.type, not by the Python class."""
    response = FakeMafResponse(
        messages=[
            FakeMessage(contents=[FakeContent(type="function_call", name="count_customers_by_plan")]),
            FakeMessage(role="tool", contents=[FakeContent(type="function_result", name="count_customers_by_plan")]),
            FakeMessage(contents=[FakeContent(type="text", text="There are 3.")]),
        ]
    )
    assert _maf_tool_names(response) == ["count_customers_by_plan"]


def test_maf_ignores_results_and_text() -> None:
    response = FakeMafResponse(
        messages=[FakeMessage(contents=[FakeContent(type="text", text="hi"), FakeContent(type="usage")])]
    )
    assert _maf_tool_names(response) == []


def test_maf_collects_several_calls_in_order() -> None:
    response = FakeMafResponse(
        messages=[
            FakeMessage(contents=[FakeContent(type="function_call", name="count_customers_by_plan")]),
            FakeMessage(contents=[FakeContent(type="function_call", name="list_customers_on_plan")]),
        ]
    )
    assert _maf_tool_names(response) == ["count_customers_by_plan", "list_customers_on_plan"]


def test_maf_handles_an_empty_response() -> None:
    assert _maf_tool_names(FakeMafResponse()) == []
    assert _maf_tool_names(object()) == []


# --- LangChain / LangGraph: tool_calls on the message ----------------------


@dataclass
class FakeLcMessage:
    content: Any = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    type: str = "ai"


def test_langchain_tool_names_are_collected() -> None:
    messages = [
        FakeLcMessage(tool_calls=[{"name": "count_customers_by_plan", "args": {}}]),
        FakeLcMessage(content="There are 3."),
    ]
    assert _tool_names(messages) == ["count_customers_by_plan"]


def test_last_text_skips_tool_messages() -> None:
    """A tool's return value is not the agent's answer."""
    messages = [
        FakeLcMessage(content="thinking"),
        FakeLcMessage(content="5 customer(s) registered", type="tool"),
        FakeLcMessage(content="There are 3 customers on pro."),
    ]
    assert _last_text(messages) == "There are 3 customers on pro."


def test_last_text_falls_back_past_empty_messages() -> None:
    messages = [FakeLcMessage(content="The answer."), FakeLcMessage(content="")]
    assert _last_text(messages) == "The answer."


def test_last_text_of_nothing_is_empty() -> None:
    assert _last_text([]) == ""


def test_flatten_handles_content_blocks() -> None:
    """Some providers return a list of blocks rather than a string."""
    assert _flatten("plain") == "plain"
    assert _flatten([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _flatten(["a", "b"]) == "ab"
    assert _flatten(None) == ""
