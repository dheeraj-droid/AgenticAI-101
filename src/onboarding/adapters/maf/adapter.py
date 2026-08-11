"""Microsoft Agent Framework adapter: a typed executor graph with tools."""

from __future__ import annotations

from typing import ClassVar

from onboarding.adapters.callers import MafLlmCaller
from onboarding.adapters.maf import executors
from onboarding.adapters.maf.workflow import build_workflow
from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.errors import OnboardingError
from onboarding.core.llm import llm_spec
from onboarding.core.rules import Capabilities
from onboarding.core.schemas import CustomerRecord, Framework, OnboardingResult


class MafAdapter:
    """Executor graph with switch-case routing."""

    name: ClassVar[Framework] = "maf"
    capabilities: ClassVar[Capabilities] = Capabilities(
        multi_step=True,
        conditional_branching=True,
        tools=True,
        agent_count="multi",
        notes=(
            "Typed executors and edges, checked when the workflow is built rather than "
            "when it runs. Switch-case edge groups make one-of-N routing explicit rather "
            "than a chain of ifs buried in a node."
        ),
        extras={"switch_case_groups": "2", "executors": "10", "agent_tools": "5"},
    )

    def __init__(self, *, allow_send: bool = False) -> None:
        self.allow_send = allow_send

    def _caller(self):
        if not llm_spec().configured:
            return None
        return MafLlmCaller()

    @concept(Concept.AGENTIC_FIRST, Concept.WORKFLOW_DECOMPOSITION)
    async def run(
        self, record: CustomerRecord, *, run_id: str, record_path: str | None = None
    ) -> OnboardingResult:
        executors.set_context(llm=self._caller(), allow_send=self.allow_send)
        state = steps.new_state(record, run_id, self.name)
        sink = default_sink(run_id, record.record_id, self.name)
        sink.emit("run_started", framework=self.name, company=record.company_name)

        outcome = await build_workflow().run(state.model_dump(mode="json"))
        outputs = [o for o in outcome.get_outputs() if isinstance(o, OnboardingResult)]
        if not outputs:
            raise OnboardingError(
                f"MAF run {run_id!r} produced no output "
                f"(final state: {outcome.get_final_state()})"
            )
        return outputs[-1]
