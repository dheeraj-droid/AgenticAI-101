"""One interactive console for the whole assistant.

Launch it once and everything happens from inside:

    onboarding shell

Two kinds of input, and the split matters:

``/command``
    A human-typed instruction. These do the real work — running the pipeline,
    approving a blocked record, resetting state. They are ordinary Python calls
    into the same code the CLI uses; no model is involved in deciding to run
    them.

anything else
    A question for the read-only chat agent.

That division is the whole point. The agent gains no new powers by living
inside a console that *can* write: it still only sees the read-only query tools,
and every mutation is something you typed yourself.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from onboarding.adapters.base import FRAMEWORKS, get_adapter, load_record
from onboarding.core.audit import JsonlAuditSink, new_run_id
from onboarding.core.config import llm_spec, paths
from onboarding.core.errors import OnboardingError
from onboarding.core.hitl import ResumeIndex
from onboarding.core.schemas import ApprovalDecision

console = Console()


@dataclass
class ShellState:
    """What the console remembers between commands."""

    framework: str = "langgraph"
    send: bool = False
    session: Any = None  # the chat session, built lazily on first question
    last_run_id: str | None = None


HELP = [
    ("/run <fixture>", "Run the onboarding pipeline over a record"),
    ("/fixtures", "List the available customer records"),
    ("/framework <name>", "Switch framework: maf | langchain | langgraph"),
    ("/pending", "Show runs waiting on a human decision"),
    ("/approve [run_id]", "Approve a blocked run (defaults to the only one pending)"),
    ("/reject [run_id] [note]", "Reject a blocked run"),
    ("/registry [reveal]", "Show the customer table"),
    ("/outbox [name]", "List the mail produced, or print one message"),
    ("/audit [n]", "Show the last n audit events"),
    ("/compare", "Run every framework over every fixture"),
    ("/concepts [layer]", "Where each agentic-AI principle lives in the code"),
    ("/doctor", "Check the environment"),
    ("/reset", "Delete .runs — registry, audit log, checkpoints, outbox"),
    ("/help", "This list"),
    ("/quit", "Leave"),
]


def start(framework: str = "langgraph", send: bool = False) -> None:
    """Run the console until the user leaves."""
    state = ShellState(framework=framework, send=send)
    _banner(state)
    while True:
        try:
            line = console.input(f"[bold cyan]{state.framework} >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "/q", "quit", "exit"):
            return
        try:
            _dispatch(line, state)
        except OnboardingError as exc:
            console.print(f"[red]{type(exc).__name__}[/]: {exc}")
        except Exception as exc:  # a bad command must never kill the session
            console.print(f"[red]{type(exc).__name__}[/]: {exc}")


def _banner(state: ShellState) -> None:
    spec = llm_spec()
    console.print("\n[bold]Customer Onboarding Assistant[/]")
    console.print(
        f"  framework [bold]{state.framework}[/] · model "
        f"[bold]{spec.model or '(none configured)'}[/] · profile {spec.profile}"
    )
    console.print(
        "  Type [bold]/help[/] for commands, or just ask a question about "
        "onboarded customers.\n"
    )


def _dispatch(line: str, state: ShellState) -> None:
    if not line.startswith("/"):
        _ask(line, state)
        return

    command, _, rest = line[1:].partition(" ")
    rest = rest.strip()
    handler = _COMMANDS.get(command.lower())
    if handler is None:
        console.print(f"[yellow]Unknown command[/] /{command}. Try [bold]/help[/].")
        return
    handler(rest, state)


# ---------------------------------------------------------------------------
# Commands. Each delegates to the same code the CLI uses.
# ---------------------------------------------------------------------------


def _cmd_help(rest: str, state: ShellState) -> None:
    table = Table(title="Commands", show_header=False)
    table.add_column("", style="bold")
    table.add_column("")
    for name, description in HELP:
        table.add_row(name, description)
    console.print(table)
    console.print("Anything not starting with / is a question for the read-only agent.\n")


def _fixture_paths() -> list[Path]:
    return sorted(paths().fixtures.glob("*.json"))


def _cmd_fixtures(rest: str, state: ShellState) -> None:
    table = Table(title="Customer records")
    table.add_column("name")
    table.add_column("company")
    table.add_column("tier")
    table.add_column("plan")
    for path in _fixture_paths():
        record = load_record(path)
        table.add_row(path.stem, record.company_name, record.tier, record.effective_plan)
    console.print(table)


def _resolve_record(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    for path in _fixture_paths():
        if path.stem == name or path.name == name:
            return path
    known = ", ".join(p.stem for p in _fixture_paths())
    raise OnboardingError(f"no record named {name!r}. Available: {known}")


def _cmd_run(rest: str, state: ShellState) -> None:
    if not rest:
        console.print("Usage: [bold]/run <fixture>[/]. See [bold]/fixtures[/].")
        return
    from onboarding.cli.main import _print_result

    path = _resolve_record(rest)
    record = load_record(path)
    adapter = get_adapter(state.framework, allow_send=state.send)
    run_id = new_run_id(record.record_id, adapter.name)
    result = asyncio.run(adapter.run(record, run_id=run_id, record_path=str(path)))
    state.last_run_id = result.run_id
    _print_result(result, in_shell=True)
    # A newly registered customer changes what the agent can talk about.
    state.session = None


def _cmd_framework(rest: str, state: ShellState) -> None:
    if not rest:
        console.print(f"Current framework: [bold]{state.framework}[/]. Options: {', '.join(FRAMEWORKS)}")
        return
    try:
        adapter = get_adapter(rest)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return
    state.framework = adapter.name
    state.session = None  # rebuild the chat on the new framework
    console.print(f"Switched to [bold]{adapter.name}[/] — {adapter.capabilities.notes}")


def _cmd_pending(rest: str, state: ShellState) -> None:
    from onboarding.cli.main import pending

    pending()


def _blocked_runs() -> list:
    return [e for e in ResumeIndex().list() if e.status == "blocked_awaiting_approval"]


def _pick_run(rest: str) -> tuple[str, str]:
    """Resolve a run id and an optional note.

    The first word is only taken as a run id if it actually is one, so
    ``/reject needs VP sign-off`` reads the whole thing as a note rather than
    hunting for a run called "needs".
    """
    known = {e.run_id for e in ResumeIndex().list()}
    first, _, remainder = rest.partition(" ")
    if first and first in known:
        return first, remainder.strip()

    blocked = _blocked_runs()
    if not blocked:
        raise OnboardingError("no runs are waiting for approval")
    if len(blocked) > 1:
        ids = "\n  ".join(e.run_id for e in blocked)
        raise OnboardingError(
            f"several runs are pending — name the one you mean:\n  {ids}"
        )
    return blocked[0].run_id, rest.strip()


def _decide(rest: str, state: ShellState, decision: str) -> None:
    from onboarding.cli.main import _print_result

    run_id, note = _pick_run(rest)
    entry = ResumeIndex().get(run_id)
    adapter = get_adapter(entry.framework, allow_send=state.send)
    result = asyncio.run(
        adapter.resume(
            run_id,
            ApprovalDecision(decision=decision, decided_by="shell", note=note),  # type: ignore[arg-type]
        )
    )
    _print_result(result, in_shell=True)
    state.session = None


def _cmd_approve(rest: str, state: ShellState) -> None:
    _decide(rest, state, "approve")


def _cmd_reject(rest: str, state: ShellState) -> None:
    _decide(rest, state, "reject")


def _cmd_registry(rest: str, state: ShellState) -> None:
    from onboarding.cli.main import registry_show

    registry_show(plan=None, reveal=rest.strip().lower() in ("reveal", "--reveal", "raw"))


def _cmd_outbox(rest: str, state: ShellState) -> None:
    from onboarding.cli.main import outbox

    outbox(show=rest or None)


def _cmd_audit(rest: str, state: ShellState) -> None:
    from onboarding.cli.main import audit

    limit = int(rest) if rest.isdigit() else 20
    audit(run_id=None, limit=limit)


def _cmd_compare(rest: str, state: ShellState) -> None:
    from onboarding.cli.main import compare

    # compare exits non-zero on divergence; the table has already printed.
    with contextlib.suppress(SystemExit):
        compare(fixtures=None, out=None, json_out=None, only=None)
    state.session = None


def _cmd_concepts(rest: str, state: ShellState) -> None:
    from onboarding.cli.main import concepts

    concepts(framework=rest or None, fmt="table")


def _cmd_doctor(rest: str, state: ShellState) -> None:
    from onboarding.cli.main import doctor

    doctor()


def _cmd_reset(rest: str, state: ShellState) -> None:
    runs = paths().runs
    if not runs.exists():
        console.print("Nothing to reset.")
        return
    answer = console.input(f"Delete [bold]{runs}[/] and everything in it? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        console.print("Left alone.")
        return
    shutil.rmtree(runs)
    paths().ensure_runs()
    state.session = None
    state.last_run_id = None
    console.print("Reset. The registry, audit log, checkpoints and outbox are empty.")


_COMMANDS = {
    "help": _cmd_help,
    "h": _cmd_help,
    "?": _cmd_help,
    "run": _cmd_run,
    "fixtures": _cmd_fixtures,
    "framework": _cmd_framework,
    "fw": _cmd_framework,
    "pending": _cmd_pending,
    "approve": _cmd_approve,
    "reject": _cmd_reject,
    "registry": _cmd_registry,
    "outbox": _cmd_outbox,
    "audit": _cmd_audit,
    "compare": _cmd_compare,
    "concepts": _cmd_concepts,
    "doctor": _cmd_doctor,
    "reset": _cmd_reset,
}


# ---------------------------------------------------------------------------
# Chat — everything that is not a /command
# ---------------------------------------------------------------------------


def _ask(question: str, state: ShellState) -> None:
    """Put a question to the read-only agent, building the session on demand."""
    from onboarding.chat.session import build_session

    if state.session is None:
        state.session = build_session(state.framework)
    try:
        turn = asyncio.run(state.session.ask(question))
    except OnboardingError:
        raise
    except Exception as exc:
        spec = llm_spec()
        console.print(
            f"[red]Could not reach the model at {spec.base_url}[/] "
            f"({type(exc).__name__}: {str(exc)[:140]})"
        )
        return
    if turn.tool_calls:
        console.print(f"[dim]  tools: {', '.join(turn.tool_calls)}[/]")
    console.print(f"[bold green]agent >[/] {turn.answer}\n")


def audit_sink_path() -> str:
    return str(JsonlAuditSink().path)
