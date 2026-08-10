"""Command line interface.

    onboarding doctor
    onboarding run --framework lg --record fixtures/customers/valid_smb.json
    onboarding resume --run-id <id> --decision approve
    onboarding pending
    onboarding compare
    onboarding concepts --framework maf
    onboarding prompts verify
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from onboarding.adapters.base import FRAMEWORKS, get_adapter, load_record
from onboarding.core.audit import JsonlAuditSink, new_run_id
from onboarding.core.config import llm_spec, paths
from onboarding.core.errors import OnboardingError
from onboarding.core.hitl import ResumeIndex
from onboarding.core.schemas import ApprovalDecision, OnboardingResult

app = typer.Typer(
    help="Customer Onboarding Assistant — one core, three frameworks.",
    no_args_is_help=True,
    add_completion=False,
)
prompts_app = typer.Typer(help="Inspect and verify the versioned prompt library.")
app.add_typer(prompts_app, name="prompts")

console = Console()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Check the environment without calling a model."""
    from onboarding.core.pii import get_pii_engine
    from onboarding.core.prompts import PromptLibrary

    table = Table(title="onboarding doctor", show_lines=False)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    spec = llm_spec()
    table.add_row(
        "LLM endpoint",
        "[green]ok[/]" if spec.configured else "[yellow]not configured[/]",
        f"{spec.profile}: {spec.base_url or '(unset)'} / {spec.model or '(unset)'}",
    )

    engine = get_pii_engine()
    table.add_row(
        "PII engine",
        "[green]presidio[/]" if engine.name == "presidio" else "[yellow]regex fallback[/]",
        "spaCy en_core_web_sm present"
        if engine.name == "presidio"
        else "install the `nlp` extra for NER-based detection",
    )

    try:
        lib = PromptLibrary.load()
        table.add_row("Prompt library", "[green]ok[/]", f"{len(lib.all_specs())} prompts, checksums verified")
    except Exception as exc:
        table.add_row("Prompt library", "[red]failed[/]", str(exc)[:90])

    for framework in FRAMEWORKS:
        try:
            adapter = get_adapter(framework)
            table.add_row(f"Adapter: {framework}", "[green]ok[/]", adapter.capabilities.checkpoint_backend)
        except Exception as exc:
            table.add_row(f"Adapter: {framework}", "[red]failed[/]", str(exc)[:90])

    runs = paths().ensure_runs()
    table.add_row("Run directory", "[green]writable[/]" if runs.exists() else "[red]missing[/]", str(runs))
    table.add_row("Fixtures", "[green]ok[/]", f"{len(list(paths().fixtures.glob('*.json')))} records")

    console.print(table)
    if not spec.configured:
        console.print(
            "\n[yellow]No LLM configured.[/] Everything policy-driven still runs; drafting will "
            "fail loudly. Copy [bold].env.example[/] to [bold].env[/] and start Ollama."
        )


# ---------------------------------------------------------------------------
# run / resume
# ---------------------------------------------------------------------------


@app.command()
def run(
    record: Annotated[Path, typer.Option("--record", "-r", help="Path to a customer record JSON")],
    framework: Annotated[str, typer.Option("--framework", "-f", help="maf | langchain | langgraph")] = "langgraph",
    json_out: Annotated[bool, typer.Option("--json", help="Print the full result as JSON")] = False,
) -> None:
    """Run the assistant over one customer record."""
    customer = load_record(record)
    adapter = get_adapter(framework)
    run_id = new_run_id(customer.record_id, adapter.name)
    try:
        result = asyncio.run(adapter.run(customer, run_id=run_id, record_path=str(record)))
    except OnboardingError as exc:
        console.print(f"[red]{type(exc).__name__}[/]: {exc}")
        raise typer.Exit(code=1) from exc

    if json_out:
        console.print_json(result.model_dump_json())
    else:
        _print_result(result)


@app.command()
def resume(
    run_id: Annotated[str, typer.Option("--run-id", help="The run_id printed when the run blocked")],
    decision: Annotated[str, typer.Option("--decision", help="approve | reject")] = "approve",
    by: Annotated[str, typer.Option("--by", help="Who is making the decision")] = "cli",
    note: Annotated[str, typer.Option("--note", help="Optional note recorded in the audit log")] = "",
) -> None:
    """Resume a run that paused at the human-approval checkpoint."""
    if decision not in ("approve", "reject"):
        console.print("[red]--decision must be 'approve' or 'reject'[/]")
        raise typer.Exit(code=2)

    entry = ResumeIndex().get(run_id)
    adapter = get_adapter(entry.framework)
    try:
        result = asyncio.run(
            adapter.resume(
                run_id,
                ApprovalDecision(decision=decision, decided_by=by, note=note),  # type: ignore[arg-type]
            )
        )
    except OnboardingError as exc:
        console.print(f"[red]{type(exc).__name__}[/]: {exc}")
        raise typer.Exit(code=1) from exc
    _print_result(result)


@app.command()
def pending() -> None:
    """List runs waiting on a human decision."""
    entries = [e for e in ResumeIndex().list() if e.status == "blocked_awaiting_approval"]
    if not entries:
        console.print("No runs are waiting for approval.")
        return
    table = Table(title="Awaiting human approval")
    table.add_column("run_id")
    table.add_column("framework")
    table.add_column("company")
    table.add_column("reasons")
    table.add_column("resumable")
    for entry in entries:
        request = entry.approval_request
        resumable = "[green]yes[/]" if entry.framework != "langchain" else "[red]no (stateless)[/]"
        table.add_row(
            entry.run_id,
            entry.framework,
            request.company_name if request else entry.record_id,
            "; ".join(request.reasons) if request else "—",
            resumable,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    fixtures: Annotated[Path | None, typer.Option("--fixtures", help="Directory of record JSON files")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Markdown report path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="JSON report path")] = None,
    only: Annotated[list[str] | None, typer.Option("--only", help="Limit to these fixture stems")] = None,
) -> None:
    """Run every framework over every fixture and write the side-by-side report."""
    from onboarding.cli.compare import run_comparison
    from onboarding.cli.report import render_json, render_markdown

    report = asyncio.run(run_comparison(fixture_dir=fixtures, only=only))
    markdown = render_markdown(report)

    target = out or (paths().docs / "comparison.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown)

    json_target = json_out or (paths().runs / "comparison.json")
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(render_json(report))

    table = Table(title="Comparison summary")
    table.add_column("fixture")
    for framework in report.frameworks:
        table.add_column(framework)
    table.add_column("identical")
    for fixture in report.fixtures:
        cells = {o.framework: o for o in report.for_fixture(fixture)}
        row = [fixture]
        for framework in report.frameworks:
            outcome = cells.get(framework)
            row.append(outcome.result.status if outcome and outcome.result else "[red]error[/]")
        if not report.comparable(fixture):
            row.append("[yellow]n/a[/]")
        else:
            row.append("[green]yes[/]" if not report.divergences(fixture) else "[red]no[/]")
        table.add_row(*row)
    console.print(table)
    console.print(f"\nMarkdown: [bold]{target}[/]\nJSON:     [bold]{json_target}[/]")
    if report.skipped:
        console.print(
            f"[yellow]Not compared[/] (fewer than two frameworks produced a result): "
            f"{', '.join(report.skipped)}"
        )
    if not report.identical:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# concepts
# ---------------------------------------------------------------------------


@app.command()
def concepts(
    framework: Annotated[str | None, typer.Option("--framework", help="Filter to one layer")] = None,
    fmt: Annotated[str, typer.Option("--format", help="table | md")] = "table",
) -> None:
    """Show where each agentic-AI concept is implemented."""
    from onboarding.core.concepts import Concept, load_all_layers, registry

    load_all_layers()
    bindings = registry()
    if framework:
        bindings = [b for b in bindings if b.layer == framework]

    if fmt == "md":
        # Plain print: rich would wrap the table rows and corrupt the Markdown.
        typer.echo(_concepts_markdown(bindings))
        return

    table = Table(title="Concept → implementation")
    table.add_column("Concept")
    table.add_column("Layer")
    table.add_column("Location")
    for concept_value in Concept:
        matches = [b for b in bindings if b.concept is concept_value]
        if not matches:
            table.add_row(str(concept_value), "[yellow]—[/]", "[yellow]not bound[/]")
            continue
        for i, binding in enumerate(sorted(matches, key=lambda b: (b.layer, b.location))):
            table.add_row(str(concept_value) if i == 0 else "", binding.layer, binding.location)
    console.print(table)


def _concepts_markdown(bindings) -> str:
    from onboarding.core.concepts import Concept

    lines = ["| Concept | Layer | Implementation |", "| --- | --- | --- |"]
    for concept_value in Concept:
        matches = sorted(
            (b for b in bindings if b.concept is concept_value), key=lambda b: (b.layer, b.location)
        )
        if not matches:
            lines.append(f"| {concept_value} | — | _not bound_ |")
        for binding in matches:
            lines.append(f"| {concept_value} | `{binding.layer}` | `{binding.location}` |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


@prompts_app.command("verify")
def prompts_verify() -> None:
    """Verify every prompt's checksum and the pinned index."""
    from onboarding.core.prompts import PromptLibrary

    try:
        lib = PromptLibrary.load()
    except Exception as exc:
        console.print(f"[red]prompt library invalid:[/] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Prompt library")
    table.add_column("id")
    table.add_column("version")
    table.add_column("pinned")
    table.add_column("status")
    table.add_column("checksum")
    for spec in lib.all_specs():
        pinned = lib.pinned.get(spec.id) == spec.version
        table.add_row(
            spec.id,
            str(spec.version),
            "[green]active[/]" if pinned else "—",
            spec.status,
            spec.compute_checksum()[:22] + "…",
        )
    console.print(table)
    console.print("[green]All checksums match.[/]")


@prompts_app.command("rechecksum")
def prompts_rechecksum() -> None:
    """Recompute and write every prompt checksum after an intentional edit."""
    from onboarding.core.prompts import PromptSpec

    changed = 0
    for file in sorted(paths().prompts.glob("*.v*.json")):
        data = json.loads(file.read_text())
        previous = data.get("checksum", "")
        data["checksum"] = ""
        actual = PromptSpec.model_validate(data).compute_checksum()
        if actual != previous:
            data["checksum"] = actual
            file.write_text(json.dumps(data, indent=2) + "\n")
            console.print(f"updated [bold]{file.name}[/] -> {actual[:22]}…")
            changed += 1
    console.print(f"{changed} prompt(s) updated." if changed else "All checksums already current.")


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@app.command()
def audit(
    run_id: Annotated[str | None, typer.Option("--run-id", help="Filter to one run")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Show at most this many events")] = 40,
) -> None:
    """Show the audit trail."""
    sink = JsonlAuditSink()
    events = sink.events_for_run(run_id) if run_id else sink.read_all()
    if not events:
        console.print("No audit events recorded yet.")
        return
    table = Table(title=f"Audit trail{f' — {run_id}' if run_id else ''}")
    table.add_column("time")
    table.add_column("framework")
    table.add_column("event")
    table.add_column("payload")
    for event in events[-limit:]:
        table.add_row(
            event.ts.strftime("%H:%M:%S"),
            event.framework,
            event.event_type,
            json.dumps(event.payload)[:90],
        )
    console.print(table)


def _print_result(result: OnboardingResult) -> None:
    colour = {
        "completed": "green",
        "blocked_awaiting_approval": "yellow",
        "rejected": "red",
        "escalated": "yellow",
        "failed": "red",
    }.get(result.status, "white")

    table = Table(title=f"{result.framework} — {result.record_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("status", f"[{colour}]{result.status}[/]")
    table.add_row("run_id", result.run_id)
    table.add_row("risk", f"{result.risk.band} ({result.risk.score})")
    if result.risk.reasons:
        table.add_row("approval reasons", "\n".join(result.risk.reasons))
    table.add_row("findings", ", ".join(f.code for f in result.findings) or "—")
    table.add_row("PII masked", ", ".join(result.pii_entity_types) or "—")
    if result.injection_signals:
        table.add_row(
            "[red]injection[/]", ", ".join(s.pattern_id for s in result.injection_signals)
        )
    table.add_row("tasks", f"{len(result.tasks)} ({sum(t.origin == 'rule' for t in result.tasks)} from policy)")
    table.add_row("confidence", str(result.confidence))
    if result.reflection.violations:
        table.add_row(
            "[yellow]violations[/]",
            "\n".join(f"{v.rule_id}: {v.detail}" for v in result.reflection.violations),
        )
    if result.escalation_queue:
        table.add_row("escalation", "\n".join(result.escalation_queue))
    table.add_row("resume", result.resume_token or "[red]not resumable[/]")
    console.print(table)

    if result.welcome_email:
        console.print(f"\n[bold]Subject:[/] {result.welcome_email.subject}\n")
        console.print(result.welcome_email.body)

    if result.status == "blocked_awaiting_approval":
        if result.resume_supported:
            console.print(
                f"\n[yellow]Waiting for a human.[/] Resume with:\n"
                f"  onboarding resume --run-id {result.run_id} --decision approve"
            )
        else:
            console.print(
                "\n[yellow]Waiting for a human.[/] This adapter is stateless and cannot be "
                "resumed — re-run the record after approval."
            )


if __name__ == "__main__":
    app()
