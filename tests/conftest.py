"""Shared fixtures.

Every test runs against a temporary ``.runs`` directory so the suite never
touches the developer's real audit log or checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from onboarding.adapters.base import load_record
from onboarding.core.config import paths
from onboarding.core.llm import llm_spec
from onboarding.core.schemas import CustomerRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "fixtures" / "customers"
FIXTURE_NAMES = sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    """Point every run artefact at a per-test directory."""
    root = tmp_path / "workspace"
    (root / "prompts").mkdir(parents=True)
    for prompt in (REPO_ROOT / "prompts").glob("*.json"):
        (root / "prompts" / prompt.name).write_bytes(prompt.read_bytes())
    (root / "pyproject.toml").write_text("# test workspace\n")
    (root / "fixtures" / "customers").mkdir(parents=True)
    for record in FIXTURE_DIR.glob("*.json"):
        (root / "fixtures" / "customers" / record.name).write_bytes(record.read_bytes())

    monkeypatch.setenv("ONBOARDING_ROOT", str(root))
    monkeypatch.setenv("ONBOARDING_AUDIT_LOG", str(root / ".runs" / "audit.jsonl"))
    paths.cache_clear()
    from onboarding.core.prompts import library

    library.cache_clear()
    yield root
    paths.cache_clear()
    library.cache_clear()


@pytest.fixture
def record_path():
    def _get(name: str) -> Path:
        return FIXTURE_DIR / f"{name}.json"

    return _get


@pytest.fixture
def record(record_path):
    def _get(name: str) -> CustomerRecord:
        return load_record(record_path(name))

    return _get


@pytest.fixture
def valid_record(record) -> CustomerRecord:
    return record("valid_smb")


@pytest.fixture
def enterprise_record(record) -> CustomerRecord:
    return record("enterprise_high_value")


@pytest.fixture
def pii_record(record) -> CustomerRecord:
    return record("pii_heavy")


@pytest.fixture
def injection_record(record) -> CustomerRecord:
    return record("injection_attempt")


@pytest.fixture
def invalid_record(record) -> CustomerRecord:
    return record("invalid_missing_fields")


@pytest.fixture
def llm_configured() -> bool:
    """Skip a test when no live endpoint is available.

    Deliberately a skip and not a stub: there is no fake model anywhere in this
    project, so an LLM test without an endpoint is not runnable rather than
    quietly meaningless.
    """
    if not llm_spec().configured:
        pytest.skip("no LLM endpoint configured (set LLM_BASE_URL and LLM_MODEL)")
    return True


@pytest.fixture
def force_regex_pii(monkeypatch):
    """Force the regex PII engine, proving masking works without spaCy."""
    monkeypatch.setenv("ONBOARDING_FORCE_REGEX_PII", "1")
    from onboarding.core.pii import _spacy_model_available

    _spacy_model_available.cache_clear()
    yield
    monkeypatch.delenv("ONBOARDING_FORCE_REGEX_PII", raising=False)
    _spacy_model_available.cache_clear()


def repo_env() -> dict[str, str]:
    """Environment for a subprocess that must see the same workspace."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env
