"""Risk assessment — decides autonomous vs assistive handling.

The score produced here drives the planning strategy and the confidence floor.
It is deliberately *not* the stop/go decision: whether a record can be onboarded
at all is the narrower question ``OnboardingState.must_escalate`` answers, and
both graphs route on that one predicate so they cannot drift apart.
"""

from __future__ import annotations

from onboarding.core.concepts import Concept, concept
from onboarding.core.injection import has_blocking_signal
from onboarding.core.rules import risk_triggers
from onboarding.core.schemas import (
    CustomerRecord,
    Finding,
    InjectionSignal,
    RiskAssessment,
)
from onboarding.core.validation import has_errors, has_warnings

# Weights are declarative so the score is explainable in the audit log.
_WEIGHTS = {"TIER": 0.30, "VALUE": 0.30, "INJECTION": 0.45, "VALIDATION_ERROR": 0.35,
            "VALIDATION_WARNING": 0.15}


@concept(Concept.PLANNING, Concept.AUTONOMOUS_VS_ASSISTIVE, Concept.POLICY_CONSTRAINED)
def assess_risk(
    record: CustomerRecord,
    findings: list[Finding],
    injection_signals: list[InjectionSignal],
) -> RiskAssessment:
    """Score the record so planning and the confidence threshold can use it."""
    blocking_injection = has_blocking_signal(injection_signals)
    errors = has_errors(findings)
    warnings = has_warnings(findings)

    triggers = risk_triggers(
        tier=record.tier,
        annual_contract_value_usd=record.commercial_terms.annual_contract_value_usd,
        has_injection_block=blocking_injection,
        has_validation_warning=warnings,
        has_validation_error=errors,
    )
    codes = {t.code for t in triggers}
    score = min(1.0, sum(_WEIGHTS.get(code, 0.1) for code in codes))
    band = "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"

    return RiskAssessment(
        tier_risk="TIER" in codes,
        value_risk="VALUE" in codes,
        injection_risk="INJECTION" in codes,
        validation_risk=bool({"VALIDATION_ERROR", "VALIDATION_WARNING"} & codes),
        score=round(score, 3),
        band=band,  # type: ignore[arg-type]
        reasons=[t.reason for t in triggers],
    )
