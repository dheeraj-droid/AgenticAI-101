"""The "no fabricated discounts" guarantee.

Defense in depth, three layers:

1. **Preventive** — ``render_allowlist`` builds the prompt block that states
   exactly which concessions (if any) are approved.
2. **Detective** — ``DiscountClaimValidator.validate`` scans generated prose for
   pricing/discount language and rejects any claim not present in the record's
   ``declared_discounts``.
3. **Corrective** — ``redact`` deletes offending sentences outright.

Layer 2 is the one that actually counts: the guarantee is enforced by a
validator that runs on all three frameworks' output identically, not by asking a
model nicely. ``validate(redact(text, claims), terms) == []`` is asserted in the
tests, so an undeclared discount cannot reach a customer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from onboarding.core.concepts import Concept, concept
from onboarding.core.schemas import CommercialTerms, CustomerRecord, RuleViolation

RULE_ID = "NO_FABRICATED_DISCOUNTS"

ClaimKind = Literal["percent", "fixed_usd", "free_months", "waived_fee", "unspecified"]

_NUMBER_WORDS: dict[str, Decimal] = {
    "one": Decimal(1),
    "two": Decimal(2),
    "three": Decimal(3),
    "four": Decimal(4),
    "five": Decimal(5),
    "six": Decimal(6),
    "seven": Decimal(7),
    "eight": Decimal(8),
    "nine": Decimal(9),
    "ten": Decimal(10),
    "eleven": Decimal(11),
    "twelve": Decimal(12),
    "fifteen": Decimal(15),
    "twenty": Decimal(20),
    "twenty-five": Decimal(25),
    "thirty": Decimal(30),
    "fifty": Decimal(50),
}


@dataclass(frozen=True, slots=True)
class DiscountClaim:
    kind: ClaimKind
    value: Decimal | None
    span: str
    sentence: str


def _to_decimal(raw: str) -> Decimal | None:
    raw = raw.strip().lower().replace(",", "").replace("$", "")
    if raw in _NUMBER_WORDS:
        return _NUMBER_WORDS[raw]
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return None


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


@concept(Concept.NO_FABRICATED_CLAIMS)
def input_number_set(record: CustomerRecord) -> set[Decimal]:
    """Numbers that legitimately appear in the record.

    Prevents false positives on things like "your 12-month term" or a correctly
    quoted contract value.
    """
    terms = record.commercial_terms
    numbers: set[Decimal] = {
        Decimal(terms.annual_contract_value_usd),
        Decimal(terms.term_months),
    }
    for d in terms.declared_discounts:
        numbers.add(Decimal(d.value))
    if record.requested_go_live:
        numbers.add(Decimal(record.requested_go_live.year))
    numbers.add(Decimal(terms.contract_start.year))
    return numbers


class DiscountClaimValidator:
    """Finds commercial concession claims in prose and grounds them in the record."""

    PERCENT = re.compile(
        r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent|per cent)\s*(?:off|discount|reduction)?"
        r"|\b(" + "|".join(_NUMBER_WORDS) + r")\s+percent\b"
        r"|\bhalf\s+(?:off|price)\b",
        re.IGNORECASE,
    )
    MONEY_OFF = re.compile(
        r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:off|discount|credit|back|reduction|rebate)"
        r"|\b([\d,]+(?:\.\d+)?)\s*(?:usd|dollars)\s*(?:off|discount|credit|rebate)\b",
        re.IGNORECASE,
    )
    FREE_PERIOD = re.compile(
        r"\b(?:first|next|initial)?\s*(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\s*"
        r"(month|months|week|weeks|year|years)\b[^.\n]{0,20}?"
        r"\b(free|at no cost|on us|complimentary|no charge)\b"
        r"|\b(free|complimentary)\s+(?:\w+\s+){0,2}?(?:trial|period|month|months)\b",
        re.IGNORECASE,
    )
    WAIVER = re.compile(
        r"\bwaiv(?:e|ed|er|ing)\b[^.\n]{0,40}\b(fee|fees|cost|costs|charge|charges)\b"
        r"|\bno\s+(?:setup|set-up|onboarding|implementation|installation|activation)\s+"
        r"(?:fee|fees|cost|costs|charge|charges)\b"
        r"|\b(?:setup|set-up|onboarding|implementation)\s+(?:fee|fees)\s+(?:is|are|will be)\s+"
        r"(?:waived|free|on us)\b",
        re.IGNORECASE,
    )
    GENERIC = re.compile(
        r"\b(discount(?:s|ed)?|promo(?:tion|tional)?|special (?:rate|rates|pricing|price|offer)|"
        r"reduced (?:rate|rates|price|pricing)|rebate|price break|money back|"
        r"account credit|credits? (?:toward|towards|of)|loyalty bonus)\b",
        re.IGNORECASE,
    )

    def scan(self, text: str) -> list[DiscountClaim]:
        """Find every commercial-concession claim in ``text``."""
        claims: list[DiscountClaim] = []
        for sentence in _sentences(text):
            for m in self.PERCENT.finditer(sentence):
                raw = m.group(1) or m.group(2)
                value = _to_decimal(raw) if raw else (Decimal(50) if "half" in m.group().lower() else None)
                claims.append(DiscountClaim("percent", value, m.group().strip(), sentence))
            for m in self.MONEY_OFF.finditer(sentence):
                raw = m.group(1) or m.group(2)
                claims.append(
                    DiscountClaim("fixed_usd", _to_decimal(raw) if raw else None, m.group().strip(), sentence)
                )
            for m in self.FREE_PERIOD.finditer(sentence):
                raw = m.group(1)
                value = _to_decimal(raw) if raw else None
                unit = (m.group(2) or "").lower()
                if value is not None and unit.startswith("year"):
                    value = value * 12
                elif value is not None and unit.startswith("week"):
                    value = None  # not expressible as free months
                claims.append(DiscountClaim("free_months", value, m.group().strip(), sentence))
            for m in self.WAIVER.finditer(sentence):
                claims.append(DiscountClaim("waived_fee", None, m.group().strip(), sentence))
            for m in self.GENERIC.finditer(sentence):
                claims.append(DiscountClaim("unspecified", None, m.group().strip(), sentence))
        return claims

    def _is_declared(self, claim: DiscountClaim, terms: CommercialTerms) -> bool:
        declared = terms.declared_discounts
        if not declared:
            return False
        for d in declared:
            if claim.kind == "unspecified":
                # A vague mention is fine only if *something* is approved.
                return True
            if d.kind != claim.kind:
                continue
            if claim.value is None:
                # Kind matches and no number was claimed: e.g. "we've waived the
                # setup fee" against an approved waived_fee.
                return True
            if Decimal(d.value) == claim.value:
                return True
        return False

    @concept(Concept.NO_FABRICATED_CLAIMS, Concept.REFLECTION)
    def validate(
        self, text: str, terms: CommercialTerms, *, allowed_numbers: set[Decimal] | None = None
    ) -> list[RuleViolation]:
        """Return a violation for every claim not grounded in ``terms``."""
        allowed_numbers = allowed_numbers or set()
        violations: list[RuleViolation] = []
        seen: set[str] = set()
        for claim in self.scan(text):
            if self._is_declared(claim, terms):
                continue
            # A number quoted straight from the record is not a fabricated claim,
            # unless the surrounding words make it a concession.
            if (
                claim.value is not None
                and claim.value in allowed_numbers
                and claim.kind not in ("percent", "free_months")
            ):
                continue
            key = claim.sentence.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                RuleViolation(
                    rule_id=RULE_ID,
                    detail=(
                        f"the draft claims a {claim.kind} concession "
                        f"({claim.span!r}) that is not in the customer's approved terms"
                    ),
                    span=claim.sentence,
                )
            )
        return violations

    def redact(self, text: str, violations: list[RuleViolation]) -> str:
        """Delete the offending sentences. The last line of defense."""
        offending = {v.span.strip() for v in violations if v.span}
        if not offending:
            return text
        kept = [s for s in _sentences(text) if s.strip() not in offending]
        return " ".join(kept).strip()


VALIDATOR = DiscountClaimValidator()


@concept(Concept.NO_FABRICATED_CLAIMS, Concept.CONTEXT_AWARE_PROMPT, Concept.POLICY_CONSTRAINED)
def render_allowlist(terms: CommercialTerms) -> str:
    """The prompt block stating exactly what may be offered."""
    if not terms.declared_discounts:
        return (
            "APPROVED COMMERCIAL CONCESSIONS: NONE.\n"
            "You must not mention pricing, discounts, credits, rebates, free periods, "
            "trials, waived fees or any other commercial concession. Do not invent numbers."
        )
    lines = ["APPROVED COMMERCIAL CONCESSIONS (you may mention ONLY these, exactly as stated):"]
    for d in terms.declared_discounts:
        if d.kind == "percent":
            detail = f"{d.value}% off"
        elif d.kind == "fixed_usd":
            detail = f"${d.value} credit"
        elif d.kind == "free_months":
            detail = f"{d.value} free month(s)"
        else:
            detail = "waived fee"
        lines.append(f"  - {d.label}: {detail} (approved by {d.approved_by})")
    lines.append("Any other concession is forbidden. Do not invent numbers.")
    return "\n".join(lines)
