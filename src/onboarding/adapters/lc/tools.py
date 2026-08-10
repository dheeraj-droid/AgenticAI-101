"""Tools bound to the single LangChain agent.

Each one is a thin wrapper over ``onboarding.core``. Note what the agent can and
cannot do: it can *read* context and *check* its own draft, but it cannot skip
validation, masking or rule enforcement — those run in the adapter, outside the
agent's control, and the adapter re-checks whatever the agent produces.

``get_customer_context`` returns the already-masked ``safe_context``, so the
agent never sees a raw email address or phone number even if it misuses a tool.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import Field

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

# Set by the adapter for the duration of one agent invocation.
_CURRENT: dict[str, Any] = {"state": None}


def set_current_state(state: OnboardingState | None) -> None:
    _CURRENT["state"] = state


def _state() -> OnboardingState:
    state = _CURRENT["state"]
    if state is None:
        raise RuntimeError("no onboarding state is bound; the adapter must call set_current_state")
    return state


@tool
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


@tool
@concept(Concept.REFLECTION, Concept.NO_FABRICATED_CLAIMS, Concept.TONE_POLICY)
def check_business_rules(
    subject: Annotated[str, Field(description="The draft subject line")],
    body: Annotated[str, Field(description="The draft email body")],
) -> str:
    """Check a draft against tone, PII and approved-concession policy before submitting it.

    Returns "PASS" or a numbered list of what must be fixed.
    """
    state = _state()
    probe = state.model_copy(update={"email": WelcomeEmail(subject=subject, body=body)})
    violations = review_output(probe)
    if not violations:
        return "PASS - this draft satisfies every house rule."
    lines = [f"{i}. [{v.rule_id}] {v.detail}" for i, v in enumerate(violations, 1)]
    return "FAIL - fix all of the following, then check again:\n" + "\n".join(lines)


@tool
@concept(Concept.ACTION)
def lookup_product_catalog(
    product: Annotated[str, Field(description="Product code, e.g. 'analytics'")],
) -> str:
    """Describe what a product code gives the customer."""
    return _lookup_product_catalog(product)


@tool
@concept(Concept.ACTION)
def lookup_sla_matrix(
    tier: Annotated[str, Field(description="Customer tier: starter, growth or enterprise")],
) -> str:
    """Return the support commitment for a tier."""
    return _lookup_sla_matrix(tier)


@tool
@concept(Concept.ACTION)
def check_region_compliance(
    region: Annotated[str, Field(description="Region code: us, eu, apac or other")],
) -> str:
    """List the compliance steps onboarding must complete for a region."""
    return _check_region_compliance(region)


@tool
@concept(Concept.TONE_POLICY)
def get_email_length_policy() -> str:
    """Return the required length band for a welcome email body."""
    return _get_email_length_policy()


AGENT_TOOLS = [
    get_customer_context,
    lookup_product_catalog,
    lookup_sla_matrix,
    check_region_compliance,
    get_email_length_policy,
    check_business_rules,
]
