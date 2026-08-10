"""Tone policy for customer-facing prose, plus slot extraction for comparison."""

from __future__ import annotations

import re

from onboarding.core.concepts import Concept, concept
from onboarding.core.rules import (
    BANNED_PHRASES,
    CTA_LEXICON,
    MAX_EXCLAMATIONS,
    RULES,
    WARMTH_LEXICON,
)
from onboarding.core.schemas import RuleViolation

RULE_ID = "TONE"


@concept(Concept.TONE_POLICY, Concept.REFLECTION)
def validate_tone(subject: str, body: str) -> list[RuleViolation]:
    """Check a draft against the house tone policy."""
    violations: list[RuleViolation] = []
    text = f"{subject}\n{body}"
    lowered = text.lower()

    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            violations.append(
                RuleViolation(
                    rule_id=RULE_ID,
                    detail=f"uses the banned phrase {phrase!r}",
                    span=phrase,
                )
            )

    if not any(w in lowered for w in WARMTH_LEXICON):
        violations.append(
            RuleViolation(rule_id=RULE_ID, detail="the draft reads cold: no welcoming language")
        )

    if not any(c in lowered for c in CTA_LEXICON):
        violations.append(
            RuleViolation(rule_id=RULE_ID, detail="the draft has no clear next step for the customer")
        )

    exclamations = body.count("!")
    if exclamations > MAX_EXCLAMATIONS:
        violations.append(
            RuleViolation(
                rule_id=RULE_ID,
                detail=f"{exclamations} exclamation marks (max {MAX_EXCLAMATIONS})",
            )
        )

    if text.isupper() or len(re.findall(r"\b[A-Z]{4,}\b", body)) > 3:
        violations.append(RuleViolation(rule_id=RULE_ID, detail="shouty capitalisation"))

    words = len(body.split())
    if words < RULES.MIN_EMAIL_WORDS:
        violations.append(
            RuleViolation(rule_id=RULE_ID, detail=f"too short: {words} words (min {RULES.MIN_EMAIL_WORDS})")
        )
    elif words > RULES.MAX_EMAIL_WORDS:
        violations.append(
            RuleViolation(rule_id=RULE_ID, detail=f"too long: {words} words (max {RULES.MAX_EMAIL_WORDS})")
        )

    if not subject.strip():
        violations.append(RuleViolation(rule_id=RULE_ID, detail="the subject line is empty"))

    return violations


_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "them", "they", "this", "to", "us", "was", "we", "were", "will", "with", "you", "your", "yours"]
)


@concept(Concept.REFLECTION)
def extract_slots(subject: str, body: str, company_name: str) -> dict[str, bool | str]:
    """Structural slots used for cross-framework prose agreement.

    Deliberately coarse: all three frameworks must agree on *these*, not on
    wording, so the comparison stays stable across models.
    """
    text = f"{subject}\n{body}".lower()
    return {
        "mentions_company": company_name.lower() in text,
        "has_greeting": bool(re.match(r"\s*(hi|hello|dear|greetings|welcome)\b", body.strip(), re.I)),
        "has_next_steps": any(c in text for c in CTA_LEXICON),
        "has_signoff": bool(
            re.search(r"\b(regards|sincerely|best|thanks|thank you|cheers|warmly)\b[,\s]*\S*\s*$", body, re.I)
        ),
        "has_warmth": any(w in text for w in WARMTH_LEXICON),
    }


def content_words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z][a-z'-]{2,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def jaccard(a: str, b: str) -> float:
    """Content-word overlap — a loose, model-stable prose agreement measure."""
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
