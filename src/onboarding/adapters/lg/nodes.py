"""LangGraph nodes and routing functions.

Every node is a thin wrapper: deserialise state, call one ``core.steps``
function, serialise back. The routing functions delegate to the same predicates
the MAF workflow uses, so the two graphs cannot diverge.
"""

from __future__ import annotations

from typing import Any

from onboarding.adapters.lg.state import GraphState
from onboarding.core import steps
from onboarding.core.audit import default_sink
from onboarding.core.concepts import Concept, concept
from onboarding.core.llm import LlmCaller
from onboarding.core.schemas import OnboardingState

# Set by the adapter before each run; the graph is otherwise pure.
_CONTEXT: dict[str, Any] = {"llm": None, "allow_send": False}


def set_context(*, llm: LlmCaller | None, allow_send: bool = False) -> None:
    _CONTEXT["llm"] = llm
    _CONTEXT["allow_send"] = allow_send


def _load(payload: GraphState) -> OnboardingState:
    return OnboardingState.model_validate(payload["state"])


def _dump(state: OnboardingState, **extra: Any) -> GraphState:
    return {"state": state.model_dump(mode="json"), **extra}


def _sink(state: OnboardingState):
    return default_sink(state.run_id, state.record.record_id, state.framework)


def _llm() -> LlmCaller:
    llm = _CONTEXT["llm"]
    if llm is None:
        from onboarding.core.errors import LlmNotConfiguredError

        raise LlmNotConfiguredError("(the LangGraph adapter reached a node that needs a model)")
    return llm


# --- nodes -----------------------------------------------------------------


@concept(Concept.PERCEPTION)
def ingest(payload: GraphState) -> GraphState:
    state = _load(payload)
    return _dump(steps.perceive(state, _sink(state)))


@concept(Concept.PLANNING, Concept.LEAST_TO_MOST)
def plan(payload: GraphState) -> GraphState:
    state = _load(payload)
    return _dump(steps.plan(state, _sink(state)))


@concept(Concept.QUERY_REWRITING)
def rewrite_query(payload: GraphState) -> GraphState:
    state = _load(payload)
    _sink(state).emit("plan_created", rewritten_query=state.plan.rewritten_query if state.plan else "")
    return _dump(state)


@concept(Concept.CONDITIONAL_BRANCHING)
def risk_gate(payload: GraphState) -> GraphState:
    # Risk is computed in plan(); this node exists so the branch has a named source.
    return payload


@concept(Concept.ACTION)
async def draft_email(payload: GraphState) -> GraphState:
    state = _load(payload)
    return _dump(await steps.act_draft_email(state, _llm(), _sink(state)))


@concept(Concept.ACTION)
async def build_tasks(payload: GraphState) -> GraphState:
    state = _load(payload)
    return _dump(await steps.act_build_tasks(state, _sink(state), llm=_CONTEXT["llm"]))


@concept(Concept.REFLECTION)
def reflect(payload: GraphState) -> GraphState:
    state = _load(payload)
    return _dump(steps.reflect(state, _sink(state)))


@concept(Concept.REFLECTION, Concept.NO_FABRICATED_CLAIMS)
async def repair(payload: GraphState) -> GraphState:
    state = _load(payload)
    return _dump(await steps.repair_email(state, _llm(), _sink(state)))


@concept(Concept.CONFIDENCE_FALLBACK)
def escalate(payload: GraphState) -> GraphState:
    state = _load(payload)
    sink = _sink(state)
    state = steps.escalate(state, sink)
    return _dump(steps.notify_already_registered(state, sink, allow_send=_CONTEXT["allow_send"]))


@concept(Concept.ACTION)
def deliver(payload: GraphState) -> GraphState:
    """Register the customer, then mail the team and the customer."""
    state = _load(payload)
    sink = _sink(state)
    state = steps.register_customer(state, sink)
    return _dump(steps.send_notifications(state, sink, allow_send=_CONTEXT["allow_send"]))


@concept(Concept.AUDIT_LOGGING)
def finalize(payload: GraphState) -> GraphState:
    return _dump(_load(payload))


NODES = (
    ingest,
    plan,
    rewrite_query,
    risk_gate,
    draft_email,
    build_tasks,
    reflect,
    repair,
    deliver,
    escalate,
    finalize,
)


# --- routing (shared predicates, not re-implemented) -----------------------


@concept(Concept.CONDITIONAL_BRANCHING)
def route_after_risk(payload: GraphState) -> str:
    """Can this record be onboarded? Same predicate as the MAF switch-case group."""
    return "escalate" if _load(payload).must_escalate() else "draft"


@concept(Concept.CONDITIONAL_BRANCHING, Concept.CONFIDENCE_FALLBACK)
def route_after_reflect(payload: GraphState) -> str:
    state = _load(payload)
    if steps.needs_repair(state):
        return "repair"
    if steps.should_escalate(state):
        return "escalate"
    return "finalize"
