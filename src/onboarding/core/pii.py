"""PII detection and masking with Microsoft Presidio (free, local, no cloud call).

Masking runs during Perception, *before* any text reaches the LLM. Two engines
share one interface:

``PresidioPiiEngine``
    Full Presidio with the spaCy NLP engine — adds NER-based PERSON/LOCATION/ORG
    detection in free text.
``RegexPiiEngine``
    Pattern recognizers only, plus a name recognizer seeded from the record's own
    contacts. Used when the spaCy model is unavailable.

The fallback degrades *detection breadth*, never masking itself, and the active
engine is recorded in the result and printed in the comparison report so the
degradation is never silent.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from functools import lru_cache
from typing import Literal, Protocol

from onboarding.core.concepts import Concept, concept
from onboarding.core.config import force_regex_pii
from onboarding.core.schemas import CustomerRecord, PiiEntity, RedactionReport

logger = logging.getLogger(__name__)

# Entities we mask. DATE_TIME is deliberately excluded: contract dates are
# business facts the model needs, and masking them costs more than it protects.
TARGET_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "UK_NINO",
    "IP_ADDRESS",
    "PERSON",
    "LOCATION",
    "URL",
)

URL_PATTERN = r"\bhttps?://[^\s,)]+"

EngineName = Literal["presidio", "regex"]


class PiiEngine(Protocol):
    name: EngineName

    def analyze(self, text: str, field: str) -> list[PiiEntity]: ...


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class PresidioPiiEngine:
    """Presidio with the spaCy NLP engine, plus two deliberate registry changes.

    * The built-in URL recognizer is replaced with a plain regex one. The stock
      recognizer pulls the public-suffix list over the network on first use,
      which is both slow and a hard dependency we do not want in a PII path.
    * A deny-list recognizer is seeded with the record's own contact names.
      NER on a bare "Annika Vogel" string is unreliable, but for this
      application we always know who the people are — so name masking is made
      deterministic rather than left to the model.
    """

    name: EngineName = "presidio"

    def __init__(self, known_names: tuple[str, ...] = ()) -> None:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
        for name in ("UrlRecognizer", "DomainRecognizer"):
            with contextlib.suppress(Exception):  # not present in this Presidio build
                registry.remove_recognizer(name)
        registry.add_recognizer(
            PatternRecognizer(
                supported_entity="URL",
                name="RegexUrlRecognizer",
                patterns=[Pattern(name="url", regex=URL_PATTERN, score=0.8)],
            )
        )

        # Presidio's deny_list matches token by token, so "Annika Vogel" would
        # come back as two adjacent PERSON spans and mask to
        # "<PERSON_1> <PERSON_2>". Full names therefore get an explicit pattern
        # (which wins the longest-span merge), and the deny_list catches the
        # individual parts wherever they appear alone.
        full_patterns: list[Pattern] = []
        deny: list[str] = []
        for i, raw_name in enumerate(known_names):
            full_name = raw_name.strip()
            if len(full_name) <= 2:
                continue
            full_patterns.append(
                Pattern(name=f"known-name-{i}", regex=rf"\b{re.escape(full_name)}\b", score=0.95)
            )
            deny.extend(part for part in full_name.split() if len(part) > 2)
        if full_patterns or deny:
            registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="PERSON",
                    name="KnownContactNameRecognizer",
                    patterns=full_patterns,
                    deny_list=sorted(set(deny)),
                    deny_list_score=0.92,
                )
            )

        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, registry=registry, supported_languages=["en"]
        )

    def analyze(self, text: str, field: str) -> list[PiiEntity]:
        if not text.strip():
            return []
        results = self._analyzer.analyze(text=text, entities=list(TARGET_ENTITIES), language="en")
        return [
            PiiEntity(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=round(float(r.score), 3),
                source_field=field,
                engine="presidio",
            )
            for r in results
            if r.score >= 0.4
        ]


class RegexPiiEngine:
    """Pattern-only recognizers — no spaCy, no model download.

    Names are still masked reliably because we seed a recognizer with the
    contact names from the record itself: for this application we always know
    who the people are, so name masking never depends on NER.
    """

    name: EngineName = "regex"

    PATTERNS: tuple[tuple[str, str], ...] = (
        ("EMAIL_ADDRESS", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        (
            "PHONE_NUMBER",
            r"(?:(?<=\s)|(?<=^)|(?<=[:(]))\+?\d[\d\s().-]{7,18}\d(?=\s|$|[.,;)])",
        ),
        ("CREDIT_CARD", r"\b(?:\d[ -]*?){13,19}\b"),
        ("IBAN_CODE", r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
        ("US_SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("UK_NINO", r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b"),
        ("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        ("URL", r"\bhttps?://[^\s,)]+"),
    )

    def __init__(self, extra_names: tuple[str, ...] = ()) -> None:
        self._compiled = [(name, re.compile(p, re.IGNORECASE)) for name, p in self.PATTERNS]
        self._names = tuple(n for n in extra_names if len(n.strip()) > 2)

    def analyze(self, text: str, field: str) -> list[PiiEntity]:
        if not text.strip():
            return []
        found: list[PiiEntity] = []
        for entity_type, pattern in self._compiled:
            for m in pattern.finditer(text):
                if not m.group().strip():
                    continue
                found.append(
                    PiiEntity(
                        entity_type=entity_type,
                        start=m.start(),
                        end=m.end(),
                        score=0.85,
                        source_field=field,
                        engine="regex",
                    )
                )
        for full_name in self._names:
            # Mask the full name and each of its parts.
            for token in {full_name, *full_name.split()}:
                if len(token) <= 2:
                    continue
                for m in re.finditer(rf"\b{re.escape(token)}\b", text, re.IGNORECASE):
                    found.append(
                        PiiEntity(
                            entity_type="PERSON",
                            start=m.start(),
                            end=m.end(),
                            score=0.9,
                            source_field=field,
                            engine="regex",
                        )
                    )
        return found


@lru_cache(maxsize=1)
def _spacy_model_available() -> bool:
    if force_regex_pii():
        return False
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.warning(
            "spaCy model 'en_core_web_sm' unavailable (%s); falling back to the regex PII engine. "
            "Install it with: uv pip install 'en_core_web_sm @ https://github.com/explosion/"
            "spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'",
            exc,
        )
        return False


def get_pii_engine(record: CustomerRecord | None = None) -> PiiEngine:
    """Return the best available engine, never ``None``."""
    names = tuple(c.full_name for c in record.all_contacts) if record else ()
    if _spacy_model_available():
        try:
            return PresidioPiiEngine(known_names=names)
        except Exception as exc:  # pragma: no cover - depends on the environment
            logger.warning("Presidio/spaCy engine failed to start (%s); using regex engine.", exc)
    return RegexPiiEngine(extra_names=names)


def _merge_spans(entities: list[PiiEntity]) -> list[PiiEntity]:
    """Collapse overlapping spans, keeping the longest (then highest-scoring)."""
    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start), -e.score))
    kept: list[PiiEntity] = []
    for e in ordered:
        if kept and e.start < kept[-1].end:
            continue
        kept.append(e)
    return kept


def mask_text(text: str, field: str, engine: PiiEngine) -> tuple[str, list[PiiEntity], dict[str, str]]:
    """Replace detected PII with stable ``<TYPE_n>`` placeholders.

    Returns the masked text, the entities found, and a placeholder -> digest map
    (a hash, never the raw value — this map is written to the audit log).
    """
    entities = _merge_spans(engine.analyze(text, field))
    if not entities:
        return text, [], {}

    counters: dict[str, int] = {}
    seen: dict[str, str] = {}  # raw value -> placeholder, so repeats mask identically
    digests: dict[str, str] = {}
    out = []
    cursor = 0
    for e in sorted(entities, key=lambda e: e.start):
        raw = text[e.start : e.end]
        placeholder = seen.get(raw)
        if placeholder is None:
            counters[e.entity_type] = counters.get(e.entity_type, 0) + 1
            placeholder = f"<{e.entity_type}_{counters[e.entity_type]}>"
            seen[raw] = placeholder
            digests[placeholder] = _digest(raw)
        out.append(text[cursor : e.start])
        out.append(placeholder)
        cursor = e.end
    out.append(text[cursor:])
    return "".join(out), entities, digests


@concept(Concept.PII_DETECTION, Concept.PERCEPTION)
def redact_record(record: CustomerRecord, engine: PiiEngine | None = None) -> RedactionReport:
    """Mask every free-text and contact field on the record.

    The result feeds ``steps.build_safe_context``; nothing else is ever put in
    front of the model.
    """
    engine = engine or get_pii_engine(record)
    fields: dict[str, str] = {
        "signup_notes": record.signup_notes,
        "primary_contact.full_name": record.primary_contact.full_name,
        "primary_contact.email": str(record.primary_contact.email),
    }
    if record.primary_contact.phone:
        fields["primary_contact.phone"] = record.primary_contact.phone
    for i, contact in enumerate(record.additional_contacts):
        fields[f"additional_contacts.{i}.full_name"] = contact.full_name
        fields[f"additional_contacts.{i}.email"] = str(contact.email)
        if contact.phone:
            fields[f"additional_contacts.{i}.phone"] = contact.phone

    masked: dict[str, str] = {}
    all_entities: list[PiiEntity] = []
    all_digests: dict[str, str] = {}
    for field, value in fields.items():
        text, entities, digests = mask_text(value, field, engine)
        masked[field] = text
        all_entities.extend(entities)
        for k, v in digests.items():
            all_digests.setdefault(f"{field}:{k}", v)

    return RedactionReport(
        entities=all_entities,
        masked_text_by_field=masked,
        placeholder_digests=all_digests,
        engine=engine.name,
    )


def contains_raw_pii(text: str, record: CustomerRecord) -> list[str]:
    """Return the raw PII values from ``record`` that leaked into ``text``.

    Used by Reflection to check generated prose, and by tests to prove the audit
    log never contains raw values.
    """
    needles: list[str] = []
    for contact in record.all_contacts:
        needles.append(str(contact.email))
        if contact.phone:
            needles.append(contact.phone)
            needles.append(re.sub(r"[\s().-]", "", contact.phone))
    haystack = text.lower()
    return sorted({n for n in needles if n and len(n) > 4 and n.lower() in haystack})
