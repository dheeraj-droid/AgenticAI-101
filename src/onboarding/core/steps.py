"""The pipeline. Every framework calls these functions and adds nothing of its own.

The agent loop maps onto the module directly:

* **Perception** — ``perceive``: validate, mask PII, scan for injection, chunk.
* **Planning**   — ``plan``: risk assessment, least-to-most decomposition, query rewrite.
* **Action**     — ``act_draft_email``, ``act_build_tasks``.
* **Reflection** — ``reflect``, ``repair_email``: output validators, retry, escalate.

Only ``act_draft_email``, ``act_build_tasks`` and ``repair_email`` touch a model,
and they do it through the injected ``LlmCaller`` protocol — so this module never
imports LangChain, LangGraph or agent_framework, and those frameworks never
re-implement any of the logic here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from onboarding.core.audit import JsonlAuditSink
from onboarding.core.chunking import chunk_notes
from onboarding.core.concepts import Concept, concept
from onboarding.core.confidence import below_threshold, escalation_reasons, score_confidence
from onboarding.core.discounts import VALIDATOR, input_number_set, render_allowlist
from onboarding.core.injection import InjectionScanner, has_blocking_signal, neutralise
from onboarding.core.llm import LlmCaller
from onboarding.core.pii import contains_raw_pii, get_pii_engine, redact_record
from onboarding.core.planning import decompose, derive_tasks
from onboarding.core.prompts import PromptLibrary, library
from onboarding.core.risk import assess_risk
from onboarding.core.rules import RULES
from onboarding.core.schemas import (
    CustomerRecord,
    OnboardingResult,
    OnboardingState,
    OnboardingTask,
    Perception,
    Plan,
    Reflection,
    RiskAssessment,
    RuleViolation,
    WelcomeEmail,
)
from onboarding.core.tone import validate_tone
from onboarding.core.validation import has_errors, validate_record

PII_LEAK_RULE = "PII_LEAK"

_SCANNER = InjectionScanner()


def new_state(record: CustomerRecord, run_id: str, framework: str) -> OnboardingState:
    return OnboardingState(
        run_id=run_id,
        framework=framework,  # type: ignore[arg-type]
        record=record,
        started_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


@concept(Concept.PERCEPTION, Concept.PII_DETECTION, Concept.PROMPT_INJECTION_DEFENSE, Concept.CHUNKING)
def perceive(state: OnboardingState, sink: JsonlAuditSink) -> OnboardingState:
    """Ingest the record: validate, mask, scan for injection, chunk the notes.

    Masking happens *before* anything else touches the text, so no downstream
    step — and certainly no model — ever sees a raw value.
    """
    record = state.record

    findings = validate_record(record)
    sink.emit(
        "record_validated",
        finding_codes=[f.code for f in findings],
        errors=sum(f.severity == "error" for f in findings),
        warnings=sum(f.severity == "warning" for f in findings),
    )

    engine = get_pii_engine(record)
    redaction = redact_record(record, engine)
    sink.emit(
        "pii_masked",
        engine=redaction.engine,
        entity_types=redaction.entity_types,
        entity_count=len(redaction.entities),
    )

    masked_notes = redaction.masked_text_by_field.get("signup_notes", record.signup_notes)
    chunks = chunk_notes(masked_notes)

    signals: list = []
    for chunk in chunks or []:
        signals.extend(_SCANNER.scan(chunk.text, "signup_notes"))
    if not chunks:
        signals.extend(_SCANNER.scan(masked_notes, "signup_notes"))
    # De-duplicate across chunks; the same attempt can straddle an overlap.
    seen: set[str] = set()
    deduped = []
    for s in sorted(signals, key=lambda s: s.pattern_id):
        if s.pattern_id not in seen:
            seen.add(s.pattern_id)
            deduped.append(s)
    sink.emit(
        "injection_scanned",
        pattern_ids=[s.pattern_id for s in deduped],
        blocking=has_blocking_signal(deduped),
        chunks=len(chunks),
    )

    perception = Perception(
        findings=findings,
        redaction=redaction,
        injection_signals=deduped,
        safe_context=build_safe_context(record, redaction.masked_text_by_field),
        chunks=chunks,
    )
    state.perception = perception
    return state


@concept(Concept.CONTEXT_AWARE_PROMPT, Concept.PERCEPTION)
def build_safe_context(record: CustomerRecord, masked: dict[str, str]) -> dict[str, str]:
    """The ONLY dictionary that is ever formatted into a prompt.

    Note what is absent: no raw email, no raw phone, no raw contact name. The
    company name and commercial terms are business facts, not personal data, and
    the model needs them to write a grounded email.
    """
    terms = record.commercial_terms
    return {
        "company_name": record.company_name,
        "tier": record.tier,
        "region": record.region,
        "product_list": ", ".join(record.products) or "none recorded",
        "contract_start": terms.contract_start.isoformat(),
        "term_months": str(terms.term_months),
        "go_live": record.requested_go_live.isoformat() if record.requested_go_live else "not set",
        "masked_contact": masked.get("primary_contact.full_name", "").strip() or "<PERSON_1>",
        "masked_notes": masked.get("signup_notes", ""),
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@concept(Concept.PLANNING, Concept.LEAST_TO_MOST, Concept.QUERY_REWRITING)
def plan(state: OnboardingState, sink: JsonlAuditSink) -> OnboardingState:
    """Assess risk, then decompose the work least-to-most."""
    perception = state.perception
    assert perception is not None, "perceive() must run before plan()"

    risk = assess_risk(state.record, perception.findings, perception.injection_signals)
    state.risk = risk
    sink.emit(
        "risk_assessed",
        band=risk.band,
        score=risk.score,
        requires_human_approval=risk.requires_human_approval,
        reasons=risk.reasons,
    )

    p = decompose(state.record, risk, perception.findings)
    state.plan = p
    sink.emit("plan_created", strategy=p.strategy, goals=[s.goal for s in p.steps])

    # The rule-derived task list is a pure function of the record, so it is
    # produced here rather than after drafting. That means a blocked, rejected or
    # escalated run still hands ops its checklist — and the list is identical
    # across frameworks on every path, not just the happy one.
    state.tasks = derive_tasks(state.record, perception.findings, risk)
    sink.emit("tasks_generated", rule_tasks=[t.task_id for t in state.tasks], llm_tasks=[])
    return state


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


def _email_prompt_vars(state: OnboardingState, lib: PromptLibrary) -> dict[str, Any]:
    perception = state.perception
    assert perception is not None
    ctx = dict(perception.safe_context)
    notes = ctx.pop("masked_notes", "")
    return {
        "rewritten_query": state.plan.rewritten_query if state.plan else "",
        "declared_discounts_block": render_allowlist(state.record.commercial_terms),
        "untrusted_notes": neutralise(notes) if notes.strip() else "(none provided)",
        "min_words": RULES.MIN_EMAIL_WORDS,
        "max_words": RULES.MAX_EMAIL_WORDS,
        **ctx,
    }


def render_system_prompt(lib: PromptLibrary | None = None) -> tuple[str, Any]:
    lib = lib or library()
    return lib.render(
        "system_policy",
        discount_policy=(
            "Mention a discount, credit, free period or waived fee ONLY if it appears in the "
            "approved-concessions block of the task. If that block says NONE, say nothing about "
            "money at all."
        ),
        tone_policy=(
            "Warm, professional, concise. Plain English. No hype, no superlatives, no urgency "
            "tactics, at most two exclamation marks. Always give exactly one clear next step."
        ),
    )


@concept(Concept.ACTION, Concept.CHAIN_OF_THOUGHT, Concept.POLICY_CONSTRAINED)
async def act_draft_email(
    state: OnboardingState,
    llm: LlmCaller,
    sink: JsonlAuditSink,
    lib: PromptLibrary | None = None,
) -> OnboardingState:
    """Draft the welcome email. The only unavoidable LLM call in the pipeline."""
    lib = lib or library()
    system, system_ref = render_system_prompt(lib)
    user, email_ref = lib.render("welcome_email", **_email_prompt_vars(state, lib))

    for ref in (system_ref, email_ref):
        if ref not in state.prompt_refs:
            state.prompt_refs.append(ref)
        sink.emit("prompt_rendered", prompt_id=ref.id, version=ref.version, checksum=ref.checksum)

    reply = await llm.complete_json(system=system, user=user)
    state.llm_calls += 1
    sink.emit("llm_called", purpose="welcome_email", prompt_id=email_ref.id, version=email_ref.version)

    email = WelcomeEmail(
        subject=str(reply.get("subject", "")).strip(),
        body=str(reply.get("body", "")).strip(),
        prompt_ref=email_ref,
    )
    state.email = email
    sink.emit("email_drafted", subject=email.subject, word_count=email.word_count)
    return state


@concept(Concept.ACTION, Concept.WORKFLOW_DECOMPOSITION)
async def act_build_tasks(
    state: OnboardingState,
    sink: JsonlAuditSink,
    llm: LlmCaller | None = None,
    lib: PromptLibrary | None = None,
    *,
    max_llm_tasks: int = 3,
) -> OnboardingState:
    """Build the internal task list.

    The rule-derived tasks are authoritative and identical across frameworks.
    A model may only append clearly-marked ``origin="llm"`` extras, which are
    excluded from the cross-framework equality assertion.
    """
    perception = state.perception
    assert perception is not None and state.risk is not None
    # plan() already derived the authoritative rule tasks; re-derive so this step
    # is still correct if called standalone, then layer optional LLM extras on top.
    tasks = derive_tasks(state.record, perception.findings, state.risk)

    if llm is not None and max_llm_tasks > 0:
        lib = lib or library()
        system, _ = render_system_prompt(lib)
        ctx = perception.safe_context
        user, ref = lib.render(
            "task_enrichment",
            existing_tasks="\n".join(f"- {t.task_id}: {t.title}" for t in tasks),
            company_name=ctx["company_name"],
            tier=ctx["tier"],
            region=ctx["region"],
            product_list=ctx["product_list"],
            untrusted_notes=neutralise(ctx.get("masked_notes", "")) or "(none provided)",
            max_tasks=max_llm_tasks,
        )
        if ref not in state.prompt_refs:
            state.prompt_refs.append(ref)
        try:
            reply = await llm.complete_json(system=system, user=user)
            state.llm_calls += 1
            tasks.extend(_parse_llm_tasks(reply, existing={t.task_id for t in tasks}, limit=max_llm_tasks))
        except Exception as exc:  # enrichment is optional — never fail the run for it
            sink.emit("llm_called", purpose="task_enrichment", ok=False, error=str(exc)[:200])

    state.tasks = tasks
    sink.emit(
        "tasks_generated",
        rule_tasks=[t.task_id for t in tasks if t.origin == "rule"],
        llm_tasks=[t.task_id for t in tasks if t.origin == "llm"],
    )
    return state


def _slugify(title: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-")
    # A title of pure punctuation slugifies to nothing; don't emit a bare "llm-".
    return f"llm-{slug}" if slug else "llm-task"


def _parse_llm_tasks(reply: dict[str, Any], existing: set[str], limit: int) -> list[OnboardingTask]:
    raw = reply.get("tasks") or []
    if not isinstance(raw, list):
        return []
    out: list[OnboardingTask] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        task_id = _slugify(title)
        if task_id in existing:
            continue
        existing.add(task_id)
        priority = str(item.get("priority", "p2")).lower()
        try:
            out.append(
                OnboardingTask(
                    task_id=task_id,
                    title=title[:120],
                    owner_role=str(item.get("owner_role", "ops"))[:20],
                    due_offset_days=max(0, min(90, int(item.get("due_offset_days", 7)))),
                    priority=priority if priority in ("p0", "p1", "p2") else "p2",  # type: ignore[arg-type]
                    origin="llm",
                )
            )
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


@concept(Concept.REFLECTION, Concept.NO_FABRICATED_CLAIMS, Concept.TONE_POLICY, Concept.CONFIDENCE_FALLBACK)
def review_output(state: OnboardingState) -> list[RuleViolation]:
    """Run every output validator against the current draft.

    Identical for all three frameworks — this is where "no fabricated discounts",
    tone and PII-leak enforcement actually happens, regardless of what any agent
    claims about its own output.
    """
    email = state.email
    if email is None:
        return []
    text = f"{email.subject}\n{email.body}"
    violations: list[RuleViolation] = []
    violations.extend(
        VALIDATOR.validate(
            text,
            state.record.commercial_terms,
            allowed_numbers=input_number_set(state.record),
        )
    )
    violations.extend(validate_tone(email.subject, email.body))
    for leaked in contains_raw_pii(text, state.record):
        violations.append(
            RuleViolation(
                rule_id=PII_LEAK_RULE,
                detail="the draft contains an unmasked personal identifier",
                span=leaked[:8] + "...",
            )
        )
    return violations


@concept(Concept.REFLECTION, Concept.CONFIDENCE_FALLBACK)
def reflect(state: OnboardingState, sink: JsonlAuditSink) -> OnboardingState:
    """Validate the draft, score confidence, and decide whether it can ship."""
    perception = state.perception
    assert perception is not None
    violations = review_output(state)

    for v in violations:
        sink.emit("rule_violation", rule_id=v.rule_id, detail=v.detail, span=v.span)

    confidence = score_confidence(
        findings=perception.findings,
        injection_signals=perception.injection_signals,
        violations=violations,
        repair_attempts=state.repair_attempts,
    )
    state.reflection = Reflection(
        violations=violations,
        repair_attempts=state.repair_attempts,
        confidence=confidence,
        passed=not violations and not below_threshold(confidence),
    )
    sink.emit(
        "reflection_completed",
        violations=[v.rule_id for v in violations],
        confidence=confidence,
        passed=state.reflection.passed,
        repair_attempts=state.repair_attempts,
    )
    return state


def needs_repair(state: OnboardingState) -> bool:
    """Routing predicate: is another repair pass both needed and permitted?"""
    r = state.reflection
    return bool(r and r.violations and state.repair_attempts < RULES.MAX_REPAIR_ATTEMPTS)


def should_escalate(state: OnboardingState) -> bool:
    """Routing predicate: has this run run out of road?"""
    r = state.reflection
    if r is None:
        return False
    return bool(r.violations) or below_threshold(r.confidence)


@concept(Concept.REFLECTION, Concept.NO_FABRICATED_CLAIMS)
async def repair_email(
    state: OnboardingState,
    llm: LlmCaller,
    sink: JsonlAuditSink,
    lib: PromptLibrary | None = None,
) -> OnboardingState:
    """One critique-and-retry pass, then hard redaction if it still fails.

    The redaction is what makes the guarantee structural: if the model will not
    remove an ungrounded claim, we delete the sentence ourselves.
    """
    email = state.email
    reflection = state.reflection
    if email is None or reflection is None or not reflection.violations:
        return state

    lib = lib or library()
    state.repair_attempts += 1
    system, _ = render_system_prompt(lib)
    user, ref = lib.render(
        "repair_discount",
        previous_subject=email.subject,
        previous_body=email.body,
        violation_list="\n".join(f"- {v.rule_id}: {v.detail}" for v in reflection.violations),
        declared_discounts_block=render_allowlist(state.record.commercial_terms),
        min_words=RULES.MIN_EMAIL_WORDS,
        max_words=RULES.MAX_EMAIL_WORDS,
    )
    if ref not in state.prompt_refs:
        state.prompt_refs.append(ref)

    try:
        reply = await llm.complete_json(system=system, user=user)
        state.llm_calls += 1
        state.email = WelcomeEmail(
            subject=str(reply.get("subject", email.subject)).strip(),
            body=str(reply.get("body", email.body)).strip(),
            prompt_ref=email.prompt_ref,
        )
        sink.emit("llm_called", purpose="repair", attempt=state.repair_attempts)
    except Exception as exc:
        sink.emit("llm_called", purpose="repair", ok=False, error=str(exc)[:200])

    # Whatever came back, re-validate and redact anything still ungrounded.
    remaining = review_output(state)
    discount_violations = [v for v in remaining if v.rule_id != PII_LEAK_RULE and v.span]
    if discount_violations and state.email is not None:
        redacted_body = VALIDATOR.redact(state.email.body, discount_violations)
        state.email = WelcomeEmail(
            subject=state.email.subject,
            body=redacted_body,
            prompt_ref=state.email.prompt_ref,
        )
        sink.emit("rule_violation", rule_id="REDACTED", detail="offending sentences removed from the draft")
    return state


@concept(Concept.CONFIDENCE_FALLBACK, Concept.HUMAN_IN_THE_LOOP)
def escalate(state: OnboardingState, sink: JsonlAuditSink, extra_reason: str = "") -> OnboardingState:
    """Route to the human review queue instead of shipping."""
    reflection = state.reflection
    # Only judge confidence if a draft was actually produced. A run rejected at
    # the approval gate never reached reflection, and reporting "confidence 0.0"
    # there would be noise, not a reason.
    reasons = (
        escalation_reasons(reflection.confidence, reflection.violations) if reflection else []
    )
    perception = state.perception
    if perception and has_errors(perception.findings):
        reasons.append("the record has blocking validation errors and cannot be onboarded as-is")
    if extra_reason:
        reasons.append(extra_reason)
    if not reasons:
        reasons.append("escalated by workflow policy")

    state.escalation_queue.extend(reasons)
    if state.status not in ("blocked_awaiting_approval", "rejected"):
        state.status = "escalated"
    sink.emit("escalated", reasons=reasons, status=state.status)
    return state


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------


@concept(Concept.AUDIT_LOGGING)
def finalize(state: OnboardingState, sink: JsonlAuditSink, *, resume_token: str | None = None) -> OnboardingResult:
    """Assemble the shared output schema. All three adapters end here."""
    perception = state.perception
    reflection = state.reflection
    started = state.started_at or datetime.now(UTC)
    finished = datetime.now(UTC)

    if state.status == "completed" and reflection and not reflection.passed:
        state.status = "escalated"

    result = OnboardingResult(
        run_id=state.run_id,
        record_id=state.record.record_id,
        framework=state.framework,
        status=state.status,
        findings=list(perception.findings) if perception else [],
        pii_entity_types=perception.redaction.entity_types if perception else [],
        pii_engine=perception.redaction.engine if perception else "regex",
        injection_signals=list(perception.injection_signals) if perception else [],
        risk=state.risk or RiskAssessment(),
        plan=state.plan or Plan(),
        welcome_email=state.email,
        tasks=list(state.tasks),
        reflection=reflection or Reflection(),
        confidence=reflection.confidence if reflection else 0.0,
        escalation_queue=list(state.escalation_queue),
        prompt_refs=list(state.prompt_refs),
        audit_log_path=str(sink.path),
        audit_event_ids=list(state.audit_event_ids) + list(sink.event_ids),
        resume_supported=resume_token is not None,
        resume_token=resume_token,
        started_at=started,
        finished_at=finished,
        duration_ms=int((finished - started).total_seconds() * 1000),
        llm_calls=state.llm_calls,
        error=state.error,
    )
    sink.emit(
        "run_finished",
        status=result.status,
        confidence=result.confidence,
        llm_calls=result.llm_calls,
        duration_ms=result.duration_ms,
    )
    return result
