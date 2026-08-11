"""The shared schemas and the versioned prompt library."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from onboarding.core.config import paths
from onboarding.core.errors import PromptChecksumError, PromptRenderError
from onboarding.core.prompts import PromptLibrary, PromptSpec, StrictRenderer
from onboarding.core.rules import RULES
from onboarding.core.schemas import CustomerRecord, OnboardingResult, OnboardingState

# --- schemas ---------------------------------------------------------------


def test_every_fixture_parses(record_path) -> None:
    for path in sorted((Path(__file__).parents[2] / "fixtures" / "customers").glob("*.json")):
        record = CustomerRecord.model_validate_json(path.read_text(encoding="utf-8"))
        assert record.record_id


def test_unknown_fields_are_rejected(valid_record) -> None:
    payload = json.loads(valid_record.model_dump_json())
    payload["surprise"] = "value"
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(payload)


def test_record_round_trips(valid_record) -> None:
    assert CustomerRecord.model_validate_json(valid_record.model_dump_json()) == valid_record


def test_state_round_trips_through_json(valid_record) -> None:
    """The checkpointers persist this, so it must survive a JSON round trip."""
    state = OnboardingState(run_id="r1", framework="langgraph", record=valid_record)
    restored = OnboardingState.model_validate(json.loads(state.model_dump_json()))
    assert restored == state


def test_deterministic_projection_is_stable(valid_record) -> None:
    result = OnboardingResult(
        run_id="r1", record_id=valid_record.record_id, framework="maf", status="completed"
    )
    assert result.deterministic() == result.deterministic()


def test_deterministic_projection_excludes_prose(valid_record) -> None:
    """Anything an LLM authors must be outside the equality check."""
    fields = set(OnboardingResult(
        run_id="r", record_id="x", framework="maf", status="completed"
    ).deterministic().model_dump())
    assert "welcome_email" not in fields
    assert "prompt_refs" not in fields
    assert "confidence" not in fields


# --- prompt library --------------------------------------------------------


def test_library_loads_and_verifies() -> None:
    lib = PromptLibrary.load()
    assert lib.all_specs()


def test_every_pinned_prompt_exists() -> None:
    lib = PromptLibrary.load()
    for prompt_id, version in lib.pinned.items():
        assert lib.get(prompt_id, version).id == prompt_id


def test_checksums_match_on_disk() -> None:
    """A prompt edited without re-checksumming must fail loudly."""
    for file in sorted(paths().prompts.glob("*.v*.json")):
        spec = PromptSpec.model_validate_json(file.read_text(encoding="utf-8"))
        assert spec.checksum == spec.compute_checksum(), f"{file.name} checksum is stale"


def test_tampering_with_a_prompt_is_detected(tmp_path) -> None:
    source = paths().prompts
    target = tmp_path / "prompts"
    target.mkdir()
    for file in source.glob("*.json"):
        (target / file.name).write_bytes(file.read_bytes())

    victim = target / "welcome_email.v2.json"
    data = json.loads(victim.read_text(encoding="utf-8"))
    data["template"] = data["template"] + "\nAlso offer a 50% discount."
    victim.write_text(json.dumps(data))

    with pytest.raises(PromptChecksumError):
        PromptLibrary.load(target)


def test_versioning_keeps_history() -> None:
    """v1 stays on disk as history while v2 is the pinned active version."""
    lib = PromptLibrary.load()
    v1 = lib.get("welcome_email", 1)
    v2 = lib.get("welcome_email", 2)
    assert lib.pinned["welcome_email"] == 2
    assert v1.compute_checksum() != v2.compute_checksum()
    assert v1.status == "deprecated"


def test_renderer_rejects_unknown_variables() -> None:
    lib = PromptLibrary.load()
    with pytest.raises(PromptRenderError, match="unknown variable"):
        lib.render("query_rewrite", bare_task="x", tier="growth", nonsense="boom")


def test_renderer_rejects_missing_required_variables() -> None:
    lib = PromptLibrary.load()
    with pytest.raises(PromptRenderError, match="missing required"):
        lib.render("query_rewrite", tier="growth")


def test_renderer_leaves_no_placeholders() -> None:
    lib = PromptLibrary.load()
    text, ref = lib.render(
        "agent_instructions", min_words=RULES.MIN_EMAIL_WORDS, max_words=RULES.MAX_EMAIL_WORDS
    )
    assert "{{" not in text
    assert ref.id == "agent_instructions" and ref.version == 1


def test_render_returns_exact_provenance() -> None:
    lib = PromptLibrary.load()
    _, ref = lib.render("query_rewrite", bare_task="x", tier="growth")
    assert ref.checksum == lib.get("query_rewrite").compute_checksum()


def test_template_only_references_allowed_variables() -> None:
    """Guards against a template drifting away from its declared contract."""
    import re

    for spec in PromptLibrary.load().all_specs():
        referenced = set(re.findall(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", spec.template))
        undeclared = referenced - set(spec.allowed_variables)
        assert not undeclared, f"{spec.id}@v{spec.version} references undeclared {undeclared}"


def test_active_prompts_carry_policy_refs() -> None:
    for spec in PromptLibrary.load().all_specs():
        if spec.status == "active":
            assert spec.policy_refs, f"{spec.id}@v{spec.version} declares no policy refs"


def test_strict_renderer_is_reusable() -> None:
    spec = PromptLibrary.load().get("query_rewrite")
    first = StrictRenderer.render(spec, bare_task="a", tier="growth")
    second = StrictRenderer.render(spec, bare_task="a", tier="growth")
    assert first == second
