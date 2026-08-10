"""The contract every framework adapter satisfies.

Adapters contain **no business logic**. Each node/executor/tool body unpacks
state, calls exactly one ``onboarding.core`` function, and packs the result back.
``tests/integration/test_adapter_thinness.py`` enforces that with an AST walk —
that is what makes the three implementations genuinely comparable rather than
three separate programs that happen to agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from onboarding.core.rules import Capabilities
from onboarding.core.schemas import (
    ApprovalDecision,
    CustomerRecord,
    Framework,
    OnboardingResult,
)


@runtime_checkable
class OnboardingAdapter(Protocol):
    """One framework's implementation of the onboarding assistant."""

    name: ClassVar[Framework]
    capabilities: ClassVar[Capabilities]

    async def run(self, record: CustomerRecord, *, run_id: str, record_path: str | None = None) -> OnboardingResult:
        """Run the assistant end to end for one record."""
        ...

    async def resume(self, run_id: str, decision: ApprovalDecision) -> OnboardingResult:
        """Resume a run that paused at the human-approval checkpoint."""
        ...


def load_record(path: str | Path) -> CustomerRecord:
    return CustomerRecord.model_validate_json(Path(path).read_text())


def get_adapter(framework: str) -> OnboardingAdapter:
    """Resolve a framework name to its adapter. Imports lazily so that a missing
    optional dependency in one framework never breaks the other two."""
    key = framework.lower()
    if key in ("lg", "langgraph"):
        from onboarding.adapters.lg.adapter import LangGraphAdapter

        return LangGraphAdapter()
    if key in ("lc", "langchain"):
        from onboarding.adapters.lc.adapter import LangChainAdapter

        return LangChainAdapter()
    if key in ("maf", "agent-framework", "agent_framework"):
        from onboarding.adapters.maf.adapter import MafAdapter

        return MafAdapter()
    raise ValueError(f"unknown framework {framework!r}; expected one of: maf, langchain, langgraph")


FRAMEWORKS: tuple[Framework, ...] = ("maf", "langchain", "langgraph")
