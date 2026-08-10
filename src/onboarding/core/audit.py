"""JSONL audit trail.

One line per event, append-only. Every framework writes the same event types in
the same order, which makes the traces directly diffable.

Nothing raw ever lands here: payloads pass through ``_scrub`` and a test sweeps
the whole file for the fixtures' real PII values.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths

EventType = Literal[
    "run_started",
    "record_validated",
    "pii_masked",
    "injection_scanned",
    "risk_assessed",
    "plan_created",
    "prompt_rendered",
    "llm_called",
    "email_drafted",
    "tasks_generated",
    "reflection_completed",
    "rule_violation",
    "approval_required",
    "approval_decided",
    "escalated",
    "run_finished",
    "run_failed",
]

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<![\w<])\+?\d[\d\s().-]{7,18}\d(?![\w>])")


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    record_id: str
    framework: str
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)


def _scrub(value: Any) -> Any:
    """Belt-and-braces redaction of anything that still looks like PII."""
    if isinstance(value, str):
        value = _EMAIL.sub("<EMAIL_REDACTED>", value)
        return _PHONE.sub("<PHONE_REDACTED>", value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


class JsonlAuditSink:
    """Append-only JSONL sink."""

    def __init__(self, path: Path | None = None, *, run_id: str = "", record_id: str = "", framework: str = ""):
        self.path = path or paths().audit_log
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.record_id = record_id
        self.framework = framework
        self.event_ids: list[str] = []

    @concept(Concept.AUDIT_LOGGING)
    def emit(self, event_type: EventType, **payload: Any) -> str:
        """Write one event and return its id."""
        event = AuditEvent(
            run_id=self.run_id,
            record_id=self.record_id,
            framework=self.framework,
            event_type=event_type,
            payload=_scrub(payload),
        )
        line = event.model_dump_json()
        # Single O_APPEND write: concurrent framework runs interleave safely.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.event_ids.append(event.event_id)
        return event.event_id

    def read_all(self) -> list[AuditEvent]:
        if not self.path.exists():
            return []
        events: list[AuditEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(AuditEvent.model_validate_json(line))
        return events

    def events_for_run(self, run_id: str) -> list[AuditEvent]:
        return [e for e in self.read_all() if e.run_id == run_id]


def default_sink(run_id: str, record_id: str, framework: str, path: Path | None = None) -> JsonlAuditSink:
    override = os.environ.get("ONBOARDING_AUDIT_LOG")
    return JsonlAuditSink(
        path or (Path(override) if override else None),
        run_id=run_id,
        record_id=record_id,
        framework=framework,
    )


def new_run_id(record_id: str, framework: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{record_id}-{framework}-{stamp}-{uuid.uuid4().hex[:6]}"


def json_default(obj: Any) -> str:
    """Serialiser for the odd Decimal/date that reaches a payload."""
    return str(obj)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=json_default, sort_keys=True)
