"""All three implementations construct, and their routing really is shared.

None of these tests calls a model — building a graph must never require one.
"""

from __future__ import annotations

from onboarding.adapters.base import FRAMEWORKS, get_adapter
from onboarding.adapters.lg.graph import CONDITIONAL_BRANCH_POINTS, build_graph
from onboarding.adapters.maf.workflow import EXECUTOR_IDS, build_workflow

# --- Microsoft Agent Framework --------------------------------------------


def test_maf_workflow_builds() -> None:
    workflow = build_workflow()
    assert {e.id for e in workflow.get_executors_list()} == set(EXECUTOR_IDS)


def test_maf_start_and_terminal_executors() -> None:
    workflow = build_workflow()
    assert workflow.get_start_executor().id == "ingest"
    assert {e.id for e in workflow.get_output_executors()} == {"finalize", "escalate"}


def test_maf_builds_fresh_instances_each_time() -> None:
    """Two concurrent runs must not share executor objects."""
    first = {id(e) for e in build_workflow().get_executors_list()}
    second = {id(e) for e in build_workflow().get_executors_list()}
    assert not (first & second)


def test_maf_agent_tools_are_exposed() -> None:
    from onboarding.adapters.maf.tools import AGENT_TOOLS

    assert len(AGENT_TOOLS) >= 5
    assert all(callable(t) and t.__doc__ for t in AGENT_TOOLS)


# --- LangGraph -------------------------------------------------------------


def test_langgraph_compiles() -> None:
    compiled = build_graph().compile()
    nodes = set(compiled.get_graph().nodes)
    expected = {
        "ingest", "plan", "rewrite_query", "risk_gate", "human_approval",
        "draft_email", "build_tasks", "reflect", "repair", "escalate", "finalize",
    }
    assert expected <= nodes


def test_langgraph_has_the_expected_branch_points() -> None:
    graph = build_graph().compile().get_graph()
    edges_from = {}
    for edge in graph.edges:
        edges_from.setdefault(edge.source, set()).add(edge.target)
    for branch in CONDITIONAL_BRANCH_POINTS:
        assert len(edges_from.get(branch, set())) >= 2, f"{branch} is not actually a branch"


def test_langgraph_repair_loop_returns_to_reflect() -> None:
    graph = build_graph().compile().get_graph()
    assert any(e.source == "repair" and e.target == "reflect" for e in graph.edges)


# --- LangChain -------------------------------------------------------------


def test_langchain_agent_builds_without_network(monkeypatch) -> None:
    """Constructing ChatOpenAI makes no request, so this is safe offline."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "not-needed")
    from onboarding.adapters.lc.agent import build_agent

    agent, refs = build_agent()
    assert agent is not None
    assert {r.id for r in refs} == {"system_policy", "agent_instructions"}


def test_langchain_tool_surface_is_stable() -> None:
    from onboarding.adapters.lc.agent import AGENT_TOOL_NAMES

    assert set(AGENT_TOOL_NAMES) == {
        "get_customer_context",
        "lookup_product_catalog",
        "lookup_sla_matrix",
        "check_region_compliance",
        "get_email_length_policy",
        "check_business_rules",
    }


# --- shared routing --------------------------------------------------------


def test_both_graphs_route_on_the_same_core_predicates() -> None:
    """The whole comparison rests on this: MAF and LangGraph must delegate their
    routing decisions to ``core`` rather than each reimplementing them.

    Checked behaviourally — the same state must produce the same routing choice
    in both graphs — rather than by reading source text.
    """
    from onboarding.adapters.lg import nodes as lg_nodes
    from onboarding.adapters.maf import workflow as maf_workflow

    for name, needs_approval, blocking in [
        ("clean", False, False),
        ("approval", True, False),
        ("blocked", False, True),
        ("both", True, True),
    ]:
        payload = _state_payload(needs_approval=needs_approval, blocking=blocking)

        maf_choice = (
            "escalate"
            if maf_workflow.has_blocking_errors(payload)
            else "approve_needed"
            if maf_workflow.needs_approval(payload)
            else "draft"
        )
        assert lg_nodes.route_after_risk(_graph_payload(payload)) == maf_choice, (
            f"the two graphs disagree on routing for the {name!r} case"
        )


def test_both_graphs_share_the_reflection_predicates() -> None:
    from onboarding.adapters.lg import nodes as lg_nodes
    from onboarding.adapters.maf import workflow as maf_workflow
    from onboarding.core import steps
    from onboarding.core.schemas import OnboardingState, Reflection, RuleViolation

    payload = _state_payload()
    state = OnboardingState.model_validate(payload)
    state.reflection = Reflection(
        violations=[RuleViolation(rule_id="TONE", detail="cold")], confidence=0.4, passed=False
    )
    payload = state.model_dump(mode="json")

    assert maf_workflow.needs_repair(payload) is steps.needs_repair(state)
    assert maf_workflow.should_escalate(payload) is steps.should_escalate(state)
    assert lg_nodes.route_after_reflect(_graph_payload(payload)) == "repair"


def _state_payload(*, needs_approval: bool = False, blocking: bool = False) -> dict:
    """A minimal serialised state with the requested routing characteristics."""
    from datetime import date
    from decimal import Decimal

    from onboarding.core.schemas import (
        CommercialTerms,
        Contact,
        CustomerRecord,
        Finding,
        OnboardingState,
        Perception,
        RiskAssessment,
    )

    record = CustomerRecord(
        record_id="T-1",
        company_name="Test Co",
        tier="growth",
        region="us",
        primary_contact=Contact(full_name="Ada Lovelace", email="ada@test.co"),
        products=["core"],
        commercial_terms=CommercialTerms(
            annual_contract_value_usd=Decimal("1000"),
            contract_start=date(2026, 1, 1),
            term_months=12,
        ),
    )
    findings = (
        [Finding(code="NO_PRODUCTS", severity="error", field_path="products", message="none")]
        if blocking
        else []
    )
    state = OnboardingState(run_id="r1", framework="maf", record=record)
    state.perception = Perception(findings=findings)
    state.risk = RiskAssessment(requires_human_approval=needs_approval)
    return state.model_dump(mode="json")


def _graph_payload(state_payload: dict) -> dict:
    """LangGraph nodes receive the state under a ``state`` channel."""
    return {"state": state_payload}


# --- capabilities ----------------------------------------------------------


def test_every_adapter_declares_capabilities() -> None:
    for framework in FRAMEWORKS:
        caps = get_adapter(framework).capabilities
        assert caps.checkpoint_backend and caps.notes


def test_only_langchain_is_stateless() -> None:
    """The comparison's headline claim, asserted rather than merely documented."""
    assert get_adapter("langchain").capabilities.durable_resume is False
    assert get_adapter("maf").capabilities.durable_resume is True
    assert get_adapter("langgraph").capabilities.durable_resume is True
    assert get_adapter("langchain").capabilities.agent_count == "single"
