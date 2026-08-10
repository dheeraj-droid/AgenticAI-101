"""Render the comparison as Markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import combinations

from onboarding.adapters.base import get_adapter
from onboarding.cli.compare import ComparisonReport, RunOutcome
from onboarding.core.config import llm_spec
from onboarding.core.tone import extract_slots, jaccard


def _fmt(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "—"
    if value is None or value == "":
        return "—"
    return str(value)


def _cell(text: object) -> str:
    return _fmt(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: ComparisonReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Framework comparison: Customer Onboarding Assistant")
    add("")
    add(
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} — "
        f"{len(report.fixtures)} fixtures × {len(report.frameworks)} frameworks._"
    )
    add("")

    # --- environment -------------------------------------------------------
    spec = llm_spec()
    add("## Environment")
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    for key, value in spec.redacted().items():
        add(f"| {key} | `{_cell(value)}` |")
    add(f"| LLM configured | {'yes' if report.llm_configured else '**no** — drafting steps skipped'} |")
    add(f"| PII engine | `{report.pii_engine}` |")
    add("")
    if not report.llm_configured:
        add(
            "> No LLM endpoint is configured, so no welcome email was drafted. Everything "
            "decided by policy — validation, masking, injection defense, risk, planning, the "
            "task list and the approval gate — still ran, and is compared below."
        )
        add("")

    # --- capability matrix -------------------------------------------------
    add("## Capability matrix")
    add("")
    caps = {fw: get_adapter(fw).capabilities for fw in report.frameworks}
    rows = [
        ("Multi-step", lambda c: "yes" if c.multi_step else "no"),
        ("Conditional branching", lambda c: "yes" if c.conditional_branching else "no"),
        ("HITL pause", lambda c: "yes" if c.hitl_pause else "no"),
        ("Durable resume", lambda c: "yes" if c.durable_resume else "**no**"),
        ("Tools", lambda c: "yes" if c.tools else "no"),
        ("Agents", lambda c: c.agent_count),
        ("Statefulness", lambda c: c.statefulness),
        ("Checkpoint backend", lambda c: c.checkpoint_backend),
    ]
    add("| Capability | " + " | ".join(report.frameworks) + " |")
    add("| --- |" + " --- |" * len(report.frameworks))
    for label, getter in rows:
        add(f"| {label} | " + " | ".join(_cell(getter(caps[fw])) for fw in report.frameworks) + " |")
    add("")
    for fw in report.frameworks:
        add(f"- **{fw}** — {caps[fw].notes}")
    add("")

    # --- per fixture -------------------------------------------------------
    add("## Results by fixture")
    add("")
    for fixture in report.fixtures:
        cells = report.for_fixture(fixture)
        add(f"### `{fixture}`")
        add("")
        add(_result_table(cells, report.frameworks))
        add("")

        divergences = report.divergences(fixture)
        if not report.comparable(fixture):
            add(
                "**Not compared** — fewer than two frameworks produced a result "
                "(see the errors above)."
            )
            add("")
        elif divergences:
            add("**Divergence:**")
            add("")
            add("| Field | " + " | ".join(report.frameworks) + " |")
            add("| --- |" + " --- |" * len(report.frameworks))
            for field, values in divergences.items():
                add(
                    f"| `{field}` | "
                    + " | ".join(_cell(values.get(fw, "—")) for fw in report.frameworks)
                    + " |"
                )
            add("")
        else:
            add("**IDENTICAL** — all frameworks agree on every deterministic field.")
            add("")

        prose = _prose_section(cells)
        if prose:
            add(prose)
            add("")

    # --- headline ----------------------------------------------------------
    add("## Verdict")
    add("")
    compared = [f for f in report.fixtures if report.comparable(f)]
    if report.identical:
        add(
            f"All frameworks produced identical deterministic outcomes on every compared "
            f"fixture ({len(compared)} of {len(report.fixtures)}). Any remaining difference "
            "is in LLM-authored prose, which is compared structurally rather than verbatim."
        )
    else:
        diverging = [f for f in compared if report.divergences(f)]
        add(f"Deterministic outcomes diverged on: {', '.join(f'`{f}`' for f in diverging) or 'none'}.")
    if report.skipped:
        add("")
        add(
            f"Not compared (fewer than two frameworks produced a result): "
            f"{', '.join(f'`{f}`' for f in report.skipped)}."
        )
    add("")
    return "\n".join(lines)


def _result_table(cells: list[RunOutcome], frameworks: tuple[str, ...]) -> str:
    by_fw = {c.framework: c for c in cells}
    rows: list[tuple[str, callable]] = [
        ("status", lambda r: r.status),
        ("risk band", lambda r: r.risk.band),
        ("needs approval", lambda r: "yes" if r.risk.requires_human_approval else "no"),
        ("findings", lambda r: sorted(f.code for f in r.findings)),
        ("PII entities", lambda r: r.pii_entity_types),
        ("injection", lambda r: sorted({s.pattern_id for s in r.injection_signals})),
        ("plan strategy", lambda r: r.plan.strategy),
        ("rule tasks", lambda r: len([t for t in r.tasks if t.origin == "rule"])),
        ("llm tasks", lambda r: len([t for t in r.tasks if t.origin == "llm"])),
        ("violations", lambda r: sorted({v.rule_id for v in r.reflection.violations})),
        ("confidence", lambda r: r.confidence),
        ("email words", lambda r: r.welcome_email.word_count if r.welcome_email else "—"),
        ("prompt versions", lambda r: sorted(f"{p.id}@v{p.version}" for p in r.prompt_refs)),
        ("llm calls", lambda r: r.llm_calls),
        ("resume token", lambda r: r.resume_token or "**none**"),
        ("duration ms", lambda r: r.duration_ms),
    ]
    lines = ["| Field | " + " | ".join(frameworks) + " |", "| --- |" + " --- |" * len(frameworks)]
    for label, getter in rows:
        values = []
        for fw in frameworks:
            cell = by_fw.get(fw)
            if cell is None or cell.result is None:
                values.append("_error_" if cell and cell.error else "—")
            else:
                values.append(_cell(getter(cell.result)))
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    errors = [(fw, by_fw[fw].error) for fw in frameworks if fw in by_fw and by_fw[fw].error]
    if errors:
        lines.append("")
        for fw, err in errors:
            lines.append(f"- **{fw} failed:** `{_cell(err)}`")
    return "\n".join(lines)


def _prose_section(cells: list[RunOutcome]) -> str:
    """Compare the drafted prose structurally, never verbatim."""
    drafts = {
        c.framework: c.result.welcome_email
        for c in cells
        if c.result and c.result.welcome_email
    }
    if len(drafts) < 1:
        return ""
    company = next((c.result.record_id for c in cells if c.result), "")
    lines = ["**Prose comparison** (structural — wording is expected to differ):", ""]
    lines.append("| Framework | Subject | Slots |")
    lines.append("| --- | --- | --- |")
    for fw, email in drafts.items():
        slots = extract_slots(email.subject, email.body, company)
        satisfied = ", ".join(k for k, v in slots.items() if v) or "none"
        lines.append(f"| {fw} | {_cell(email.subject)} | {_cell(satisfied)} |")
    if len(drafts) > 1:
        lines.append("")
        lines.append("| Pair | Content-word Jaccard |")
        lines.append("| --- | --- |")
        for a, b in combinations(sorted(drafts), 2):
            score = jaccard(drafts[a].body, drafts[b].body)
            lines.append(f"| {a} ↔ {b} | {score:.2f} |")
    return "\n".join(lines)


def render_json(report: ComparisonReport) -> str:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "llm_configured": report.llm_configured,
        "pii_engine": report.pii_engine,
        "frameworks": list(report.frameworks),
        "fixtures": report.fixtures,
        "identical": report.identical,
        "runs": [
            {
                "fixture": o.fixture,
                "framework": o.framework,
                "error": o.error,
                "duration_ms": o.duration_ms,
                "result": json.loads(o.result.model_dump_json()) if o.result else None,
                "deterministic": json.loads(o.deterministic().model_dump_json())
                if o.deterministic()
                else None,
            }
            for o in report.outcomes
        ],
        "divergences": {f: {k: _stringify(v) for k, v in report.divergences(f).items()} for f in report.fixtures},
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _stringify(values: dict) -> dict:
    return {k: _fmt(v) for k, v in values.items()}
