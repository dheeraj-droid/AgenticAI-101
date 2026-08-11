"""Confidence scoring and the threshold fallback to a human queue.

Below ``RULES.MIN_CONFIDENCE`` we do not ship the draft: the run is escalated
instead. Same function in all four frameworks.
"""

from __future__ import annotations

from onboarding.core.concepts import Concept, concept
from onboarding.core.rules import RULES
from onboarding.core.schemas import Finding, InjectionSignal, RuleViolation

# Each signal shaves confidence. Declarative so the report can explain a score.
PENALTIES: dict[str, float] = {
    "validation_error": 0.30,
    "validation_warning": 0.10,
    "injection_block": 0.35,
    "injection_flag": 0.10,
    "violation": 0.20,
    "redacted": 0.25,
    "repair": 0.10,
}


@concept(Concept.CONFIDENCE_FALLBACK, Concept.REFLECTION)
def score_confidence(
    *,
    findings: list[Finding],
    injection_signals: list[InjectionSignal],
    violations: list[RuleViolation],
    repair_attempts: int = 0,
) -> float:
    """Return a 0..1 confidence for the generated output."""
    score = 1.0
    for f in findings:
        if f.severity == "error":
            score -= PENALTIES["validation_error"]
        elif f.severity == "warning":
            score -= PENALTIES["validation_warning"]
    for s in injection_signals:
        score -= PENALTIES["injection_block" if s.severity == "block" else "injection_flag"]
    for v in violations:
        score -= PENALTIES["redacted"] if v.remediation == "redacted" else PENALTIES["violation"]
    score -= PENALTIES["repair"] * repair_attempts
    return round(max(0.0, min(1.0, score)), 3)


@concept(Concept.CONFIDENCE_FALLBACK)
def below_threshold(confidence: float) -> bool:
    return confidence < RULES.MIN_CONFIDENCE


@concept(Concept.CONFIDENCE_FALLBACK)
def escalation_reasons(confidence: float, violations: list[RuleViolation]) -> list[str]:
    """Human-readable reasons for routing to the escalation queue."""
    reasons: list[str] = []
    if below_threshold(confidence):
        reasons.append(
            f"confidence {confidence} is below the {RULES.MIN_CONFIDENCE} threshold; "
            "routing to the human review queue"
        )
    for v in violations:
        if v.remediation == "redacted":
            reasons.append(f"content was redacted to satisfy {v.rule_id}: {v.detail}")
        elif v.remediation is None:
            reasons.append(f"unresolved {v.rule_id} violation: {v.detail}")
    return reasons
