"""Structural guarantees that keep the three implementations comparable.

If these fail, the comparison stops meaning anything: an adapter has grown its
own business logic, or a fake model has crept into the product code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "onboarding"
ADAPTERS = SRC / "adapters"

# Statements allowed in an adapter function beyond the call into core.
MAX_ADAPTER_STATEMENTS = 12

# Graph builders are exempt from the statement count: they are declarative
# wiring, and their length scales with the number of nodes, which is exactly the
# thing we want to be free to grow. They are NOT exempt from
# `test_adapters_define_no_rules` below, so they still cannot contain a
# threshold, a regex or any other business rule.
GRAPH_BUILDERS = {"build_workflow", "build_graph", "build_agent"}


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node


@pytest.mark.parametrize("path", _python_files(ADAPTERS), ids=lambda p: p.name)
def test_adapters_stay_thin(path: Path) -> None:
    """Adapter functions wire frameworks together; they do not compute business rules."""
    tree = ast.parse(path.read_text())
    for func in _functions(tree):
        if func.name in GRAPH_BUILDERS:
            continue
        statements = [n for n in func.body if not isinstance(n, ast.Expr | ast.Pass)]
        assert len(statements) <= MAX_ADAPTER_STATEMENTS, (
            f"{path.name}:{func.name} has {len(statements)} statements — "
            "business logic belongs in onboarding.core"
        )


@pytest.mark.parametrize("path", _python_files(ADAPTERS), ids=lambda p: p.name)
def test_graph_builders_only_wire(path: Path) -> None:
    """A graph builder may be long, but every call it makes must be wiring.

    This is what keeps the statement-count exemption honest: the builder can add
    nodes and edges, and can name core predicates, but it cannot compute
    anything itself.
    """
    allowed_calls = {
        # framework wiring
        "add_node", "add_edge", "add_conditional_edges", "add_switch_case_edge_group",
        "add_chain", "add_fan_in_edges", "add_fan_out_edges", "build", "compile",
        "StateGraph", "WorkflowBuilder", "Case", "Default", "create_agent", "tool",
        # local construction
        "IngestExecutor", "PlanExecutor", "RiskGateExecutor", "ApprovalExecutor",
        "DraftEmailExecutor", "TaskListExecutor", "ReflectExecutor", "RepairExecutor",
        "DeliverExecutor", "EscalateExecutor", "FinalizeExecutor",
        # prompt/model plumbing
        "render", "render_system_prompt", "library", "make_langchain_model", "make_maf_client",
    }
    tree = ast.parse(path.read_text())
    for func in _functions(tree):
        if func.name not in GRAPH_BUILDERS:
            continue
        # Walk the body only — ast.walk(func) would also visit the @concept
        # decorator, which is documentation, not wiring.
        for statement in func.body:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name is None or name.startswith("_"):
                    continue
                assert name in allowed_calls, (
                    f"{path.name}:{func.name} calls {name!r} — a graph builder may only "
                    "wire nodes and edges together"
            )


@pytest.mark.parametrize("path", _python_files(ADAPTERS), ids=lambda p: p.name)
def test_adapters_define_no_rules(path: Path) -> None:
    """No thresholds, no regexes, no policy constants outside core."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            assert target != "compile" or "re" not in ast.dump(node.func), (
                f"{path.name} compiles a regex — detection rules belong in onboarding.core"
            )
        # Bare numeric literals are how thresholds sneak in.
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            pytest.fail(f"{path.name} contains the float literal {node.value} — put it in core.rules")


def test_adapters_do_not_import_the_openai_sdk() -> None:
    """LLM access goes through core.llm so every framework shares one endpoint."""
    for path in _python_files(ADAPTERS):
        if path.name == "callers.py":
            continue  # the one place framework-native clients are constructed
        source = path.read_text()
        assert "import openai" not in source, f"{path.name} imports the OpenAI SDK directly"


def test_no_fake_model_anywhere_in_src() -> None:
    """There is deliberately no stub LLM: a missing endpoint fails loudly.

    A fake model would quietly make the comparison meaningless, so its absence
    is enforced rather than merely intended.
    """
    banned = ("FakeChatModel", "FakeListChatModel", "FakeLLM", "MockChatModel", "StubChatModel")
    for path in _python_files(SRC):
        source = path.read_text()
        for name in banned:
            assert name not in source, f"{path.name} references {name}"
        assert "unittest.mock" not in source, f"{path.name} imports unittest.mock"


def test_core_never_imports_a_framework() -> None:
    """The pipeline must stay framework-agnostic, or the three runs are not
    really running the same code."""
    frameworks = ("langchain", "langgraph", "agent_framework")
    for path in _python_files(SRC / "core"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                # core.llm names the client classes only inside TYPE_CHECKING or
                # a function body, which is what keeps core importable alone.
                if any(name.startswith(f) for f in frameworks):
                    assert path.name == "llm.py", (
                        f"onboarding/core/{path.name} imports {name} at module scope"
                    )


def test_core_imports_without_any_framework_installed() -> None:
    """Importing the pipeline must not require LangChain or agent_framework."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "sys.modules['langchain'] = None;"
        "sys.modules['langgraph'] = None;"
        "sys.modules['agent_framework'] = None;"
        "import onboarding.core.steps;"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(SRC.parents[1])
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_every_concept_is_implemented_somewhere() -> None:
    """The documented principles must be bound to real code."""
    from onboarding.core.concepts import Concept, load_all_layers, registry

    load_all_layers()
    bound = {b.concept for b in registry()}
    missing = sorted(str(c) for c in Concept if c not in bound)
    assert not missing, f"concepts documented but not implemented: {missing}"


def test_each_framework_binds_the_agent_loop() -> None:
    """All four loop phases must appear in every framework, not just in core."""
    from onboarding.core.concepts import Concept, load_all_layers, registry

    load_all_layers()
    loop = {Concept.PERCEPTION, Concept.PLANNING, Concept.ACTION, Concept.REFLECTION}
    for layer in ("maf", "langgraph"):
        bound = {b.concept for b in registry() if b.layer == layer}
        assert loop <= bound, f"{layer} is missing {sorted(str(c) for c in loop - bound)}"
