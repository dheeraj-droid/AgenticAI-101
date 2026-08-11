"""The tool surface the chat agent is given — read-only, in every framework.

Each tool is a thin wrapper over ``onboarding.core.qa``, which itself imports
only the registry's query functions. There is no path from any of these to
``registry.append_customer``, the mailer, or the approval store.

``tests/integration/test_chat_readonly.py`` enforces that by walking the import
graph: if anyone adds a tool here that can write, the build fails.
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field

from onboarding.core import qa
from onboarding.core.concepts import Concept, concept
from onboarding.core.registry import registry_path


@concept(Concept.ACTION, Concept.PII_DETECTION)
def count_customers_by_plan() -> str:
    """Count how many customers are on each plan (free, pro, pro+).

    Use this for any "how many" question about plans.
    """
    return qa.describe_breakdown()


@concept(Concept.ACTION)
def list_customers_on_plan(
    plan: Annotated[str, Field(description="One of: free, pro, pro+")],
) -> str:
    """List the customers on a given plan."""
    rows = qa.on_plan(plan)
    if not rows:
        return f"No customers are on the {plan!r} plan."
    return f"{len(rows)} customer(s) on {plan}:\n" + "\n".join(f"- {r.as_line()}" for r in rows)


@concept(Concept.ACTION)
def find_customer_details(
    query: Annotated[str, Field(description="A customer name, company, email or record id")],
) -> str:
    """Look up one customer. Email and phone come back masked, by policy."""
    rows = qa.lookup(query)
    if not rows:
        return f"No customer matches {query!r}."
    return "\n".join(f"- {r.as_line()}" for r in rows)


@concept(Concept.ACTION)
def list_all_customers() -> str:
    """List every registered customer."""
    rows = qa.all_customers()
    if not rows:
        return "The registry is empty."
    return "\n".join(f"- {r.as_line()}" for r in rows)


@concept(Concept.ACTION)
def describe_registry() -> str:
    """Where the registry lives and how many rows it has."""
    counts = qa.breakdown()
    return json.dumps(
        {"path": str(registry_path()), "customers": counts.total, "by_plan": counts.counts},
        indent=2,
    )


# The complete, read-only tool surface. Shared by all three chat agents so the
# comparison stays honest: same tools, different orchestration.
READ_ONLY_TOOLS = [
    count_customers_by_plan,
    list_customers_on_plan,
    find_customer_details,
    list_all_customers,
    describe_registry,
]

TOOL_NAMES = tuple(fn.__name__ for fn in READ_ONLY_TOOLS)
