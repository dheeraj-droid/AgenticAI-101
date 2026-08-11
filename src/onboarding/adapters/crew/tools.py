"""Tools bound to the CrewAI crew.

Identical bodies to the LangChain agent's tools — the same ``onboarding.core``
functions, the same already-masked ``safe_context``, the same server-side
re-check afterwards. Only the decorator differs, which is exactly what makes the
comparison meaningful: the crew's behaviour cannot be explained by it having
been given different information.

``_CURRENT`` is the same bind-for-one-invocation trick the LangChain adapter
uses. A CrewAI tool is a plain callable with no place to put per-run state, so
the adapter sets it before ``kickoff`` and clears it in a ``finally``.
"""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import tool

from onboarding.adapters.maf.tools import (
    check_region_compliance as _check_region_compliance,
)
from onboarding.adapters.maf.tools import (
    get_email_length_policy as _get_email_length_policy,
)
from onboarding.adapters.maf.tools import (
    lookup_product_catalog as _lookup_product_catalog,
)
from onboarding.adapters.maf.tools import (
    lookup_sla_matrix as _lookup_sla_matrix,
)
from onboarding.core.concepts import Concept, concept
from onboarding.core.discounts import render_allowlist
from onboarding.core.schemas import OnboardingState, WelcomeEmail
from onboarding.core.steps import review_output

_CURRENT: dict[str, Any] = {"state": None}


def set_current_state(state: OnboardingState | None) -> None:
    _CURRENT["state"] = state


def _state() -> OnboardingState:
    state = _CURRENT["state"]
    if state is None:
        raise RuntimeError("no onboarding state is bound; the adapter must call set_current_state")
    return state


@tool("get_customer_context")
@concept(Concept.CONTEXT_AWARE_PROMPT, Concept.PII_DETECTION)
def get_customer_context() -> str:
    """Get the masked facts about this customer. This is the only source of truth."""
    state = _state()
    perception = state.perception
    context = dict(perception.safe_context) if perception else {}
    context.pop("masked_notes", None)
    context["approved_concessions"] = render_allowlist(state.record.commercial_terms)
    context["task"] = state.plan.rewritten_query if state.plan else ""
    return json.dumps(context, indent=2)


@tool("check_business_rules")
@concept(Concept.REFLECTION, Concept.NO_FABRICATED_CLAIMS, Concept.TONE_POLICY)
def check_business_rules(subject: str, body: str) -> str:
    """Check a draft against tone, PII and approved-concession policy.

    Pass the draft subject line and body. Returns "PASS", or a numbered list of
    what must be fixed.
    """
    state = _state()
    probe = state.model_copy(update={"email": WelcomeEmail(subject=subject, body=body)})
    violations = review_output(probe)
    if not violations:
        return "PASS - this draft satisfies every house rule."
    lines = [f"{i}. [{v.rule_id}] {v.detail}" for i, v in enumerate(violations, 1)]
    return "FAIL - fix all of the following, then check again:\n" + "\n".join(lines)


@tool("lookup_product_catalog")
@concept(Concept.ACTION)
def lookup_product_catalog(product: str) -> str:
    """Describe what a product code gives the customer, e.g. 'analytics'."""
    return _lookup_product_catalog(product)


@tool("lookup_sla_matrix")
@concept(Concept.ACTION)
def lookup_sla_matrix(tier: str) -> str:
    """Return the support commitment for a tier: starter, growth or enterprise."""
    return _lookup_sla_matrix(tier)


@tool("check_region_compliance")
@concept(Concept.ACTION)
def check_region_compliance(region: str) -> str:
    """List the compliance steps onboarding must complete for a region code."""
    return _check_region_compliance(region)


@tool("get_email_length_policy")
@concept(Concept.TONE_POLICY)
def get_email_length_policy() -> str:
    """Return the required length band for a welcome email body."""
    return _get_email_length_policy()


WRITER_TOOLS = [
    get_customer_context,
    lookup_product_catalog,
    lookup_sla_matrix,
    check_region_compliance,
    get_email_length_policy,
]

# The reviewer gets the validator and nothing else. It is checking a draft, not
# gathering new facts, and a reviewer that can look things up tends to start
# rewriting instead of reviewing.
REVIEWER_TOOLS = [check_business_rules, get_email_length_policy]

CREW_TOOLS = [*WRITER_TOOLS, check_business_rules]
