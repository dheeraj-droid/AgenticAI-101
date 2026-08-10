"""Tools exposed to the Microsoft Agent Framework agent.

These are read-only lookups the drafting agent may call to ground its email.
They are deliberately *not* the pipeline: validation, masking and rule
enforcement all happen in the workflow's executors, where they cannot be skipped
by a model choosing not to call a tool.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from onboarding.core.rules import PRODUCT_TASKS, REGION_TASKS, RULES

# A small, static catalogue. Real deployments would query a product system.
_PRODUCT_BLURBS: dict[str, str] = {
    "core": "the core platform, including projects, roles and audit history",
    "analytics": "the analytics module, with dashboards and scheduled reports",
    "api": "the public REST API and webhooks",
    "sso": "single sign-on against your identity provider",
    "support-premium": "premium support with a named responder and faster targets",
    "reporting": "the reporting pack, including exports and shared views",
}

_SLA_MATRIX: dict[str, str] = {
    "starter": "next business day response, business hours",
    "growth": "four business hours response, extended hours",
    "enterprise": "one hour response, 24/7, with a named escalation contact",
}


def lookup_product_catalog(
    product: Annotated[str, Field(description="Product code, e.g. 'analytics' or 'sso'")],
) -> str:
    """Describe what a product code gives the customer."""
    return _PRODUCT_BLURBS.get(product.lower().strip(), f"an unrecognised product code ({product})")


def lookup_sla_matrix(
    tier: Annotated[str, Field(description="Customer tier: starter, growth or enterprise")],
) -> str:
    """Return the support commitment for a tier."""
    return _SLA_MATRIX.get(tier.lower().strip(), "no published support commitment")


def check_region_compliance(
    region: Annotated[str, Field(description="Region code: us, eu, apac or other")],
) -> str:
    """List the compliance steps onboarding must complete for a region."""
    tasks = REGION_TASKS.get(region.lower().strip(), ())
    if not tasks:
        return "no region-specific compliance steps are required"
    return "; ".join(t.title for t in tasks)


def get_email_length_policy() -> str:
    """Return the required length band for a welcome email body."""
    return f"between {RULES.MIN_EMAIL_WORDS} and {RULES.MAX_EMAIL_WORDS} words"


def list_onboarding_task_catalog(
    product: Annotated[str, Field(description="Product code to look up onboarding tasks for")],
) -> str:
    """Return the internal onboarding task a product triggers, if any."""
    template = PRODUCT_TASKS.get(product.lower().strip())
    return template.title if template else "no product-specific onboarding task"


AGENT_TOOLS = [
    lookup_product_catalog,
    lookup_sla_matrix,
    check_region_compliance,
    get_email_length_policy,
    list_onboarding_task_catalog,
]
