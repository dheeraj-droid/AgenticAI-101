"""The interactive console.

The console can write — it runs the pipeline, approves records, resets state.
That makes one property load-bearing: the *model* must still not be able to
trigger any of it. Commands are dispatched from typed input only, and the chat
path is reached exactly when the input is not a command.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from onboarding.cli import shell
from onboarding.core.errors import OnboardingError
from onboarding.core.hitl import ResumeIndex

SRC = Path(__file__).resolve().parents[2] / "src" / "onboarding"


# --- dispatch ---------------------------------------------------------------


def test_every_documented_command_exists() -> None:
    """/help must not advertise a command that isn't wired up."""
    for name, _ in shell.HELP:
        command = name.split()[0].lstrip("/")
        if command in ("quit",):  # handled in the input loop, not the table
            continue
        assert command in shell._COMMANDS, f"/help lists /{command} but it is not implemented"


def test_every_command_is_documented() -> None:
    documented = {n.split()[0].lstrip("/") for n, _ in shell.HELP}
    aliases = {"h", "?", "fw"}
    for command in shell._COMMANDS:
        if command in aliases:
            continue
        assert command in documented, f"/{command} works but /help does not mention it"


def test_unknown_command_is_reported_not_sent_to_the_model(monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(shell, "_ask", lambda q, s: asked.append(q))
    shell._dispatch("/nonsense", shell.ShellState())
    assert asked == [], "an unknown /command was forwarded to the agent"


def test_plain_text_goes_to_the_agent(monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(shell, "_ask", lambda q, s: asked.append(q))
    shell._dispatch("how many customers are on pro?", shell.ShellState())
    assert asked == ["how many customers are on pro?"]


def test_commands_never_reach_the_agent(monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(shell, "_ask", lambda q, s: asked.append(q))
    monkeypatch.setattr(shell, "_cmd_fixtures", lambda rest, state: None)
    shell._dispatch("/fixtures", shell.ShellState())
    assert asked == []


# --- the write boundary -----------------------------------------------------


def test_the_shell_does_not_hand_write_tools_to_the_model() -> None:
    """The chat session built here is the same read-only one used elsewhere."""
    source = inspect.getsource(shell._ask)
    assert "build_session" in source
    assert "READ_ONLY_TOOLS" not in source or "tools=" not in source


def test_chat_session_tools_stay_read_only() -> None:
    """Belt and braces: the console must not extend the agent's tool surface."""
    from onboarding.chat.tools import TOOL_NAMES

    tree = ast.parse((SRC / "cli" / "shell.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "tools":
            pytest.fail("the shell passes a `tools=` argument, which could widen the agent")
    assert "count_customers_by_plan" in TOOL_NAMES


# --- record resolution ------------------------------------------------------


def test_fixture_resolves_by_bare_name() -> None:
    assert shell._resolve_record("valid_smb").stem == "valid_smb"
    assert shell._resolve_record("valid_smb.json").stem == "valid_smb"


def test_unknown_fixture_lists_the_options() -> None:
    with pytest.raises(OnboardingError, match="Available:"):
        shell._resolve_record("no_such_record")


# --- approval resolution ----------------------------------------------------


async def _block_one(record, record_path) -> str:
    from onboarding.adapters.base import get_adapter
    from onboarding.core.audit import new_run_id

    adapter = get_adapter("langgraph")
    result = await adapter.run(
        record,
        run_id=new_run_id(record.record_id, "langgraph"),
        record_path=str(record_path("enterprise_high_value")),
    )
    assert result.status == "blocked_awaiting_approval"
    return result.run_id


async def test_a_note_is_not_mistaken_for_a_run_id(enterprise_record, record_path) -> None:
    """`/reject needs VP sign-off` must read as a note, not a run called "needs"."""
    run_id = await _block_one(enterprise_record, record_path)
    resolved, note = shell._pick_run("needs VP sign-off")
    assert resolved == run_id
    assert note == "needs VP sign-off"


async def test_an_explicit_run_id_wins(enterprise_record, record_path) -> None:
    run_id = await _block_one(enterprise_record, record_path)
    resolved, note = shell._pick_run(f"{run_id} looks fine")
    assert resolved == run_id
    assert note == "looks fine"


async def test_bare_approve_picks_the_only_pending_run(enterprise_record, record_path) -> None:
    run_id = await _block_one(enterprise_record, record_path)
    resolved, note = shell._pick_run("")
    assert resolved == run_id
    assert note == ""


def test_approving_with_nothing_pending_says_so() -> None:
    with pytest.raises(OnboardingError, match="no runs are waiting"):
        shell._pick_run("")


async def test_several_pending_runs_require_a_choice(
    enterprise_record, injection_record, record_path
) -> None:
    from onboarding.adapters.base import get_adapter
    from onboarding.core.audit import new_run_id

    await _block_one(enterprise_record, record_path)
    adapter = get_adapter("langgraph")
    await adapter.run(
        injection_record,
        run_id=new_run_id(injection_record.record_id, "langgraph"),
        record_path=str(record_path("injection_attempt")),
    )
    assert len(ResumeIndex().list()) == 2
    with pytest.raises(OnboardingError, match="name the one you mean"):
        shell._pick_run("")


# --- state ------------------------------------------------------------------


def test_switching_framework_drops_the_chat_session() -> None:
    """The session is bound to a framework, so it must be rebuilt on a switch."""
    state = shell.ShellState(session=object())
    shell._cmd_framework("maf", state)
    assert state.framework == "maf"
    assert state.session is None


def test_switching_to_an_unknown_framework_is_refused() -> None:
    state = shell.ShellState()
    shell._cmd_framework("nonsense", state)
    assert state.framework == "langgraph", "the framework changed despite an invalid name"


def test_framework_accepts_short_aliases() -> None:
    state = shell.ShellState()
    shell._cmd_framework("lc", state)
    assert state.framework == "langchain"
