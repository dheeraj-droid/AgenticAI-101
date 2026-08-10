"""PII masking must work with or without spaCy, and must never leak."""

from __future__ import annotations

import pytest

from onboarding.core.pii import (
    PresidioPiiEngine,
    RegexPiiEngine,
    contains_raw_pii,
    get_pii_engine,
    mask_text,
    redact_record,
)


@pytest.fixture(params=["auto", "regex"])
def engine(request, pii_record, monkeypatch):
    """Exercise every assertion against both engines."""
    if request.param == "regex":
        return RegexPiiEngine(extra_names=tuple(c.full_name for c in pii_record.all_contacts))
    return get_pii_engine(pii_record)


def test_pii_heavy_record_is_fully_masked(pii_record, engine) -> None:
    report = redact_record(pii_record, engine)
    blob = " ".join(report.masked_text_by_field.values())

    assert contains_raw_pii(blob, pii_record) == []
    for contact in pii_record.all_contacts:
        assert contact.full_name not in blob
        assert str(contact.email) not in blob


def test_expected_entity_types_are_detected(pii_record, engine) -> None:
    report = redact_record(pii_record, engine)
    found = set(report.entity_types)
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"} <= found


def test_placeholder_map_never_stores_raw_values(pii_record, engine) -> None:
    """The digest map goes into the audit log, so it must be hashes only."""
    report = redact_record(pii_record, engine)
    for digest in report.placeholder_digests.values():
        assert digest.startswith("sha256:")
    joined = " ".join(report.placeholder_digests.values())
    assert contains_raw_pii(joined, pii_record) == []


def test_repeated_values_mask_to_the_same_placeholder(engine) -> None:
    text = "Email a@b.com, or reach a@b.com again."
    masked, entities, _ = mask_text(text, "notes", engine)
    assert masked.count("<EMAIL_ADDRESS_1>") == 2
    assert "a@b.com" not in masked


def test_two_word_names_mask_as_one_placeholder(pii_record) -> None:
    """A name must not become '<PERSON_1> <PERSON_2>' — that reads badly in a greeting."""
    engine = get_pii_engine(pii_record)
    name = pii_record.primary_contact.full_name
    masked, _, _ = mask_text(name, "primary_contact.full_name", engine)
    assert masked == "<PERSON_1>", f"expected a single placeholder, got {masked!r}"


def test_regex_engine_needs_no_spacy(force_regex_pii, pii_record) -> None:
    """The fallback must be a real fallback: masking cannot silently switch off."""
    engine = get_pii_engine(pii_record)
    assert engine.name == "regex"
    report = redact_record(pii_record, engine)
    assert report.engine == "regex"
    assert contains_raw_pii(" ".join(report.masked_text_by_field.values()), pii_record) == []


def test_empty_text_is_handled(engine) -> None:
    masked, entities, digests = mask_text("", "notes", engine)
    assert (masked, entities, digests) == ("", [], {})


def test_record_without_pii_is_left_alone(engine) -> None:
    text = "The team is migrating from spreadsheets before the autumn push."
    masked, _, _ = mask_text(text, "notes", engine)
    assert masked == text


def test_presidio_engine_is_used_when_spacy_is_available(pii_record) -> None:
    engine = get_pii_engine(pii_record)
    if isinstance(engine, PresidioPiiEngine):
        assert engine.name == "presidio"
    else:  # environment without the model — the fallback must still be sound
        assert engine.name == "regex"
