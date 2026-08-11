"""The demo: run one agent, then talk to it.

    onboarding demo --framework maf

Onboards a customer end to end — validates, masks, drafts the welcome email,
builds the task list, registers them, writes the mail — then drops straight into
a conversation about the customer it just processed.

Run it three times, once per framework, and you have watched the same workflow
in Microsoft Agent Framework, LangChain and LangGraph.
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel

from onboarding.adapters.base import get_adapter, load_record
from onboarding.core.audit import new_run_id
from onboarding.core.config import paths
from onboarding.core.errors import OnboardingError
from onboarding.core.schemas import CustomerRecord, OnboardingResult

console = Console()


def run_demo(framework: str = "langgraph", record_name: str = "valid_smb") -> None:
    """Onboard one customer, show the result, then chat about them."""
    path = _find_record(record_name)
    record = load_record(path)
    adapter = get_adapter(framework)

    console.print(
        f"\n[bold]{adapter.name}[/] — onboarding [bold]{record.company_name}[/] "
        f"({record.effective_plan} plan)\n"
    )

    result = asyncio.run(
        adapter.run(record, run_id=new_run_id(record.record_id, adapter.name), record_path=str(path))
    )
    _show(result)

    if result.status == "blocked_awaiting_approval":
        console.print(
            "[yellow]This record needs human approval before anything is sent, so no email "
            "was drafted.[/] Try a record that clears the gate, e.g. "
            "[bold]--record valid_smb[/].\n"
        )

    _chat(framework, record)


def _find_record(name: str):
    candidate = paths().fixtures / f"{name}.json"
    if candidate.exists():
        return candidate
    from pathlib import Path

    if Path(name).exists():
        return Path(name)
    known = ", ".join(p.stem for p in sorted(paths().fixtures.glob("*.json")))
    raise OnboardingError(f"no record named {name!r}. Available: {known}")


def _show(result: OnboardingResult) -> None:
    """Print what the agent decided and what it wrote."""
    console.print(f"  status      [bold]{result.status}[/]")
    console.print(f"  risk        {result.risk.band}")
    console.print(f"  PII masked  {', '.join(result.pii_entity_types) or 'none found'}")
    if result.injection_signals:
        console.print(
            f"  [red]injection[/]   {', '.join(s.pattern_id for s in result.injection_signals)}"
        )
    console.print(f"  tasks       {len(result.tasks)}")
    console.print(f"  registered  {'yes' if result.registered else 'no'}")
    if result.mail_outbox:
        console.print(f"  mail        {len(result.mail_outbox)} message(s) in .runs/outbox/")
    console.print()

    if result.welcome_email:
        console.print(
            Panel(
                result.welcome_email.body,
                title=f"[bold]{result.welcome_email.subject}[/]",
                border_style="green",
            )
        )
        console.print()


def _chat(framework: str, record: CustomerRecord) -> None:
    """Talk to the agent about the customer it just onboarded."""
    from onboarding.chat.session import build_session

    console.print(
        "[bold]Ask me about this customer[/] — or about anyone onboarded so far.\n"
        "[dim]e.g. \"what plan are they on?\", \"how many customers are on pro?\"  "
        "Type exit to finish.[/]\n"
    )
    session = build_session(framework, record)
    while True:
        try:
            question = console.input("[bold cyan]you >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            return
        try:
            turn = asyncio.run(session.ask(question))
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}[/]: {str(exc)[:160]}\n")
            continue
        if turn.tool_calls:
            console.print(f"[dim]  tools: {', '.join(turn.tool_calls)}[/]")
        console.print(f"[bold green]agent >[/] {turn.answer}\n")
