"""What the web page actually does, with no HTTP in sight.

Two jobs: run one onboarding and describe the result, and hold a chat session
per browser tab. Keeping this out of ``app.py`` means the flow can be tested
without a server.

The same read/write split as everywhere else applies: the chat session gets the
read-only tool surface, and nothing reachable from the page's chat endpoint can
change a customer, a task or the outbox.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email import message_from_string, policy
from pathlib import Path
from typing import Any

from onboarding.adapters.base import get_adapter
from onboarding.core.audit import default_sink, new_run_id
from onboarding.core.config import paths
from onboarding.core.schemas import CustomerRecord, OnboardingResult
from onboarding.core.tasks import all_summaries, read_tasks
from onboarding.web.models import (
    ChatResponse,
    FindingView,
    MailView,
    OnboardRequest,
    OnboardResponse,
    PlanCount,
    Stats,
    TaskView,
    TraceView,
    describe_outcome,
)

# How many chat sessions to keep. A browser tab that is closed and reopened
# starts a new one; this is a demo, not a session store.
MAX_SESSIONS = 32


@dataclass
class Session:
    """One browser tab: the customer it onboarded, and the chat about them."""

    session_id: str
    framework: str
    record: CustomerRecord
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    chat: Any = None


class Demo:
    """The demo's whole server-side state: a bounded bag of chat sessions."""

    def __init__(self, *, allow_send: bool = False) -> None:
        self.allow_send = allow_send
        self._sessions: dict[str, Session] = {}

    async def onboard(self, request: OnboardRequest) -> OnboardResponse:
        """Run one record through one framework and describe what happened."""
        record = request.to_record()
        adapter = get_adapter(request.framework, allow_send=self.allow_send)
        run_id = new_run_id(record.record_id, adapter.name)
        result = await adapter.run(record, run_id=run_id)

        session = Session(
            session_id=uuid.uuid4().hex, framework=request.framework, record=record
        )
        self._remember(session)
        return _describe(result, record, session.session_id, request.framework)

    async def ask(self, session_id: str, question: str) -> ChatResponse:
        """Answer a question about the registry, in this tab's framework."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        if session.chat is None:
            from onboarding.chat.session import build_session

            session.chat = build_session(session.framework, session.record)
        turn = await session.chat.ask(question)
        return ChatResponse(
            answer=turn.answer, tool_calls=turn.tool_calls, framework=session.framework
        )

    def _remember(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        while len(self._sessions) > MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.created_at)
            del self._sessions[oldest.session_id]


def _describe(
    result: OnboardingResult, record: CustomerRecord, session_id: str, framework: str
) -> OnboardResponse:
    outcome, headline = describe_outcome(result)
    statuses = {row.task_id: row.status for row in read_tasks(record.record_id)}
    return OnboardResponse(
        session_id=session_id,
        framework=framework,
        run_id=result.run_id,
        record_id=result.record_id,
        company_name=record.company_name,
        contact_name=record.primary_contact.full_name,
        plan=record.effective_plan,
        status=result.status,
        outcome=outcome,  # type: ignore[arg-type]
        headline=headline,
        duplicate=outcome == "duplicate",
        registered=result.registered,
        findings=[
            FindingView(code=f.code, severity=f.severity, message=f.message)
            for f in result.findings
        ],
        pii_entity_types=list(result.pii_entity_types),
        injection_signals=[s.pattern_id for s in result.injection_signals],
        risk_band=result.risk.band,
        confidence=result.confidence,
        escalation_queue=list(result.escalation_queue),
        tasks=[
            TaskView(
                task_id=t.task_id,
                title=t.title,
                owner_role=t.owner_role,
                priority=t.priority,
                due_offset_days=t.due_offset_days,
                status=statuses.get(t.task_id, "pending"),
            )
            for t in result.tasks
        ],
        mail=_mail_views(result),
        trace=[
            TraceView(seq=t.seq, name=t.name, kind=t.kind, detail=t.detail) for t in result.trace
        ],
        llm_calls=result.llm_calls,
        duration_ms=result.duration_ms,
    )


def _mail_views(result: OnboardingResult) -> list[MailView]:
    """Read the messages back off disk, with their delivery verdict.

    The ``.eml`` file is the message; whether it was actually transmitted lives
    in the audit log, because that is the record that survives the process. The
    two are matched on the outbox path.
    """
    verdicts = _delivery_verdicts(result.run_id)
    views: list[MailView] = []
    for raw_path in result.mail_outbox:
        path = Path(raw_path)
        if not path.exists():
            continue
        message = message_from_string(path.read_text(encoding="utf-8"), policy=policy.default)
        transmitted, reason = verdicts.get(str(path), (False, "no delivery record"))
        views.append(
            MailView(
                audience=message["X-Onboarding-Audience"] or "team",  # type: ignore[arg-type]
                to=str(message["To"] or ""),
                subject=str(message["Subject"] or ""),
                body=message.get_content(),
                transmitted=transmitted,
                reason=reason,
            )
        )
    return views


def _delivery_verdicts(run_id: str) -> dict[str, tuple[bool, str]]:
    sink = default_sink(run_id, "", "")
    verdicts: dict[str, tuple[bool, str]] = {}
    for event in sink.events_for_run(run_id):
        if event.event_type != "mail_sent":
            continue
        path = str(event.payload.get("outbox_path", ""))
        verdicts[path] = (
            bool(event.payload.get("transmitted")),
            str(event.payload.get("reason", "")),
        )
    return verdicts


def stats() -> Stats:
    """Everything the page shows at a glance, counted rather than estimated."""
    from onboarding.core import qa
    from onboarding.core.registry import registry_path

    breakdown = qa.breakdown()
    summaries = all_summaries()
    # Known plans first and always present, so the row does not reshuffle as
    # customers arrive; anything unexpected in the CSV is appended rather than
    # silently dropped.
    plans = [*qa.KNOWN_PLANS, *sorted(set(breakdown.counts) - set(qa.KNOWN_PLANS))]
    return Stats(
        customers=breakdown.total,
        by_plan=[PlanCount(plan=p, count=breakdown.of(p)) for p in plans],
        tasks_total=sum(s.total for s in summaries),
        tasks_completed=sum(s.completed for s in summaries),
        registry_path=str(registry_path()),
    )


def page() -> str:
    """The single HTML page, read from disk so it can be edited without a restart."""
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


def runs_dir() -> Path:
    return paths().runs
