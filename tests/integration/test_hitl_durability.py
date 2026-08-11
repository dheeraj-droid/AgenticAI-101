"""Human-in-the-loop: pause, log, stop — and resume from a *different process*.

None of this needs a model. A high-risk record is blocked before any drafting
happens, and a rejected run escalates without one, so the whole durable-state
mechanism is verifiable offline.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from onboarding.adapters.base import get_adapter
from onboarding.core.audit import JsonlAuditSink, new_run_id
from onboarding.core.config import paths
from onboarding.core.errors import ResumeNotSupportedError
from onboarding.core.hitl import ResumeIndex
from onboarding.core.pii import contains_raw_pii
from onboarding.core.schemas import ApprovalDecision

REPO_ROOT = Path(__file__).resolve().parents[2]

STATEFUL = ["maf", "langgraph"]
ALL = ["maf", "langchain", "langgraph"]


async def _run(framework: str, record, record_path: Path):
    adapter = get_adapter(framework)
    return await adapter.run(
        record, run_id=new_run_id(record.record_id, framework), record_path=str(record_path)
    )


@pytest.mark.parametrize("framework", ALL)
async def test_high_risk_record_blocks_before_drafting(framework, enterprise_record, record_path) -> None:
    """Pause, log, stop — and crucially, no email is drafted."""
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    assert result.status == "blocked_awaiting_approval"
    assert result.welcome_email is None, "a blocked record must never produce customer-facing prose"
    assert result.risk.requires_human_approval
    assert result.llm_calls == 0, "a blocked record must not spend tokens"


@pytest.mark.parametrize("framework", ALL)
async def test_blocked_run_still_hands_ops_its_checklist(framework, enterprise_record, record_path) -> None:
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    task_ids = {t.task_id for t in result.tasks}
    assert "obtain-human-approval" in task_ids
    assert len(task_ids) > 1


@pytest.mark.parametrize("framework", ALL)
async def test_approval_required_is_audited_with_reasons(framework, enterprise_record, record_path) -> None:
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    events = JsonlAuditSink().events_for_run(result.run_id)
    approval = [e for e in events if e.event_type == "approval_required"]
    assert approval, "the pause was not written to the audit log"
    payload = approval[0].payload
    assert payload["reasons"], "the log does not say why a human is needed"
    assert payload["risk_band"] == "high"


@pytest.mark.parametrize("framework", STATEFUL)
async def test_paused_state_is_persisted_to_disk(framework, enterprise_record, record_path) -> None:
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    entry = ResumeIndex().get(result.run_id)
    assert entry.framework == framework
    assert entry.state_snapshot is not None

    if framework == "langgraph":
        assert paths().langgraph_db.exists()
    else:
        assert any(paths().maf_checkpoints.rglob("*")), "no MAF checkpoint was written"


@pytest.mark.parametrize("framework", STATEFUL)
async def test_reject_resumes_and_escalates(framework, enterprise_record, record_path) -> None:
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    resumed = await get_adapter(framework).resume(
        result.run_id,
        ApprovalDecision(decision="reject", decided_by="tester", note="needs VP sign-off"),
    )
    assert resumed.status == "rejected"
    assert resumed.welcome_email is None
    assert any("tester" in reason for reason in resumed.escalation_queue)


@pytest.mark.parametrize("framework", STATEFUL)
async def test_decision_maker_survives_the_process_boundary(framework, enterprise_record, record_path) -> None:
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    await get_adapter(framework).resume(
        result.run_id, ApprovalDecision(decision="reject", decided_by="alex", note="on hold")
    )
    events = JsonlAuditSink().events_for_run(result.run_id)
    decided = [e for e in events if e.event_type == "approval_decided"]
    assert decided, "no approval_decided event was logged"
    assert decided[-1].payload["decided_by"] == "alex"
    assert decided[-1].payload["note"] == "on hold"


@pytest.mark.parametrize("framework", STATEFUL)
async def test_resume_works_from_a_brand_new_process(
    framework, enterprise_record, record_path, isolated_runs, monkeypatch
) -> None:
    """The real durability proof: the resuming interpreter never saw the run."""
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    assert result.status == "blocked_awaiting_approval"

    env = dict(**{k: v for k, v in _env().items()})
    proc = subprocess.run(
        [
            sys.executable, "-m", "onboarding.cli.main", "resume",
            "--run-id", result.run_id, "--decision", "reject", "--by", "other-process",
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )
    assert proc.returncode == 0, f"resume subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    assert "rejected" in proc.stdout

    entry = ResumeIndex().get(result.run_id)
    assert entry.status == "rejected"


async def test_langchain_cannot_resume(enterprise_record, record_path) -> None:
    """The stateless contrast, demonstrated rather than asserted in prose."""
    result = await _run("langchain", enterprise_record, record_path("enterprise_high_value"))
    assert result.status == "blocked_awaiting_approval"
    assert result.resume_supported is False
    assert result.resume_token is None

    with pytest.raises(ResumeNotSupportedError, match="stateless by design"):
        await get_adapter("langchain").resume(result.run_id, ApprovalDecision(decision="approve"))


# --- audit hygiene ---------------------------------------------------------


@pytest.mark.parametrize("framework", ALL)
async def test_audit_log_never_contains_raw_pii(framework, pii_record, record_path) -> None:
    """The audit trail is the most likely place for PII to leak. It must not."""
    # A missing LLM endpoint is fine here: the log written before the drafting
    # step is exactly what we want to sweep for leaks.
    with contextlib.suppress(Exception):
        await _run(framework, pii_record, record_path("pii_heavy"))

    log = paths().audit_log
    assert log.exists()
    text = log.read_text(encoding="utf-8")
    assert contains_raw_pii(text, pii_record) == []
    for contact in pii_record.all_contacts:
        assert str(contact.email) not in text


@pytest.mark.parametrize("framework", ALL)
async def test_audit_events_are_well_formed(framework, enterprise_record, record_path) -> None:
    result = await _run(framework, enterprise_record, record_path("enterprise_high_value"))
    for line in paths().audit_log.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        assert {"event_id", "ts", "run_id", "record_id", "framework", "event_type"} <= payload.keys()

    types = [e.event_type for e in JsonlAuditSink().events_for_run(result.run_id)]
    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    for expected in ("record_validated", "pii_masked", "injection_scanned", "risk_assessed"):
        assert expected in types


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env
