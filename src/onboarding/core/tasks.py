"""Per-customer task lists, one CSV file each.

The registry row for a customer carries a ``tasks_path`` pointing at
``.runs/tasks/<record_id>.csv``. That file is the customer's checklist, one row
per task, with a ``status`` of ``pending`` or ``completed``.

Same read/write split as the registry: the query functions here are what the
chat agent reaches, and ``mark`` — the only mutator — is called from the web
app and the CLI, never exposed as a tool.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from filelock import FileLock

from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.schemas import OnboardingTask

FIELDNAMES = (
    "task_id",
    "title",
    "owner_role",
    "due_offset_days",
    "priority",
    "origin",
    "status",
    "updated_at",
)

Status = Literal["pending", "completed"]
LOCK_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class TaskRow:
    task_id: str
    title: str
    owner_role: str
    due_offset_days: str
    priority: str
    origin: str
    status: str
    updated_at: str = ""

    @property
    def done(self) -> bool:
        return self.status.strip().lower() == "completed"

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in FIELDNAMES}


@dataclass(slots=True)
class TaskSummary:
    """What the agent needs to answer "how many tasks are done for X"."""

    record_id: str
    total: int = 0
    completed: int = 0

    @property
    def pending(self) -> int:
        return self.total - self.completed

    def describe(self, who: str) -> str:
        if self.total == 0:
            return f"{who} has no tasks recorded."
        return (
            f"{who} has {self.completed} of {self.total} tasks completed "
            f"({self.pending} still pending)."
        )


def tasks_dir() -> Path:
    override = os.environ.get("ONBOARDING_TASKS_DIR")
    return Path(override) if override else paths().runs / "tasks"


def tasks_path(record_id: str) -> Path:
    """Where this customer's checklist lives."""
    safe = "".join(c for c in record_id if c.isalnum() or c in "-_") or "unknown"
    return tasks_dir() / f"{safe}.csv"


# ---------------------------------------------------------------------------
# Queries — read-only, safe for the agent
# ---------------------------------------------------------------------------


def read_tasks(record_id: str) -> list[TaskRow]:
    path = tasks_path(record_id)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            TaskRow(**{name: (row.get(name) or "") for name in FIELDNAMES})
            for row in csv.DictReader(fh)
        ]


@concept(Concept.ACTION)
def summarise(record_id: str) -> TaskSummary:
    rows = read_tasks(record_id)
    return TaskSummary(
        record_id=record_id,
        total=len(rows),
        completed=sum(1 for r in rows if r.done),
    )


def pending_tasks(record_id: str) -> list[TaskRow]:
    return [r for r in read_tasks(record_id) if not r.done]


def completed_tasks(record_id: str) -> list[TaskRow]:
    return [r for r in read_tasks(record_id) if r.done]


# ---------------------------------------------------------------------------
# Writes — deterministic code only
# ---------------------------------------------------------------------------


@concept(Concept.ACTION)
def write_tasks(record_id: str, tasks: list[OnboardingTask]) -> Path:
    """Create the customer's checklist. Every task starts pending."""
    path = tasks_path(record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS), open(
        path, "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "owner_role": task.owner_role,
                    "due_offset_days": str(task.due_offset_days),
                    "priority": task.priority,
                    "origin": task.origin,
                    "status": "pending",
                    "updated_at": now,
                }
            )
    return path


def mark(record_id: str, task_id: str, status: Status = "completed") -> bool:
    """Flip one task. Returns False when the task is not in the list."""
    path = tasks_path(record_id)
    if not path.exists():
        return False
    with FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS):
        rows = read_tasks(record_id)
        found = False
        now = datetime.now(UTC).isoformat(timespec="seconds")
        updated: list[dict[str, str]] = []
        for row in rows:
            data = row.as_dict()
            if row.task_id == task_id:
                data["status"] = status
                data["updated_at"] = now
                found = True
            updated.append(data)
        if found:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(updated)
    return found


def all_summaries() -> list[TaskSummary]:
    """Every customer that has a checklist on disk."""
    directory = tasks_dir()
    if not directory.exists():
        return []
    return [summarise(p.stem) for p in sorted(directory.glob("*.csv"))]
