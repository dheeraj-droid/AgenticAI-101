"""The headline claim: the same record produces the same decisions in all four.

Model-free. Every fixture used here reaches a terminal state without drafting, so
the deterministic half of the pipeline is compared exactly with no endpoint
required. A record that *can* be onboarded goes on to drafting, which needs a
model — those live in ``tests/llm``.
"""

from __future__ import annotations

import pytest

from onboarding.adapters.base import FRAMEWORKS, get_adapter
from onboarding.core.audit import new_run_id

# Fixtures that finish without needing a model.
MODEL_FREE_FIXTURES = ["invalid_missing_fields", "injection_attempt"]


async def _run_all(record, path):
    results = {}
    for framework in FRAMEWORKS:
        adapter = get_adapter(framework)
        results[framework] = await adapter.run(
            record, run_id=new_run_id(record.record_id, framework), record_path=str(path)
        )
    return results


@pytest.mark.parametrize("fixture", MODEL_FREE_FIXTURES)
async def test_deterministic_outcomes_are_identical(fixture, record, record_path) -> None:
    results = await _run_all(record(fixture), record_path(fixture))
    outcomes = {fw: r.deterministic() for fw, r in results.items()}

    reference_fw, reference = next(iter(outcomes.items()))
    for framework, outcome in outcomes.items():
        if outcome == reference:
            continue
        differences = {
            field: (getattr(reference, field), getattr(outcome, field))
            for field in reference.model_dump()
            if getattr(reference, field) != getattr(outcome, field)
        }
        pytest.fail(f"{framework} diverges from {reference_fw} on {fixture}: {differences}")


@pytest.mark.parametrize("fixture", MODEL_FREE_FIXTURES)
async def test_status_and_risk_agree(fixture, record, record_path) -> None:
    results = await _run_all(record(fixture), record_path(fixture))
    assert len({r.status for r in results.values()}) == 1
    assert len({r.risk.band for r in results.values()}) == 1
    assert len({tuple(r.risk.reasons) for r in results.values()}) == 1


@pytest.mark.parametrize("fixture", MODEL_FREE_FIXTURES)
async def test_task_lists_agree(fixture, record, record_path) -> None:
    results = await _run_all(record(fixture), record_path(fixture))
    task_sets = {fw: tuple(sorted(t.task_id for t in r.tasks if t.origin == "rule")) for fw, r in results.items()}
    assert len(set(task_sets.values())) == 1, f"task lists differ: {task_sets}"


@pytest.mark.parametrize("fixture", MODEL_FREE_FIXTURES)
async def test_pii_and_injection_detection_agree(fixture, record, record_path) -> None:
    results = await _run_all(record(fixture), record_path(fixture))
    assert len({tuple(sorted(r.pii_entity_types)) for r in results.values()}) == 1
    assert len({tuple(sorted({s.pattern_id for s in r.injection_signals})) for r in results.values()}) == 1


async def test_injection_record_never_reaches_a_model(record, record_path) -> None:
    """The safety property: poisoned free text must not be drafted from."""
    results = await _run_all(record("injection_attempt"), record_path("injection_attempt"))
    for framework, result in results.items():
        assert result.risk.injection_risk, f"{framework} did not flag the injection"
        assert result.status == "escalated", f"{framework} did not stop the run"
        assert result.welcome_email is None, f"{framework} drafted prose for a poisoned record"
        assert result.llm_calls == 0, f"{framework} spent a model call on a poisoned record"
        assert result.registered is False, f"{framework} registered a poisoned record"


async def test_invalid_record_never_reaches_a_model(record, record_path) -> None:
    results = await _run_all(record("invalid_missing_fields"), record_path("invalid_missing_fields"))
    for framework, result in results.items():
        assert result.status == "escalated", framework
        assert result.llm_calls == 0, f"{framework} spent tokens on an invalid record"
        assert "fix-record-data" in {t.task_id for t in result.tasks}


async def test_missing_llm_fails_loudly_rather_than_silently(record, record_path, monkeypatch) -> None:
    """No stub model exists, so a record that needs drafting must raise."""
    from onboarding.core.errors import LlmNotConfiguredError

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises((LlmNotConfiguredError, Exception)) as exc_info:
        adapter = get_adapter("langgraph")
        await adapter.run(
            record("valid_smb"),
            run_id=new_run_id("CUST-1001", "langgraph"),
            record_path=str(record_path("valid_smb")),
        )
    assert "LLM" in str(exc_info.value) or "llm" in str(exc_info.value).lower()
