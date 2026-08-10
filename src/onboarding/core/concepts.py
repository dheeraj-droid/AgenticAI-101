"""Catalogue of the agentic-AI concepts this project demonstrates.

Every concept is attached to the code that implements it via the ``@concept``
decorator, and the per-framework mapping tables in ``docs/`` are generated from
the resulting registry. That way the documentation cannot drift away from the
code: a test regenerates the tables and compares them to what is committed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar


class Concept(StrEnum):
    """The principles the three implementations are meant to show."""

    # --- agent loop -------------------------------------------------------
    PERCEPTION = "Agent loop: Perception"
    PLANNING = "Agent loop: Planning"
    ACTION = "Agent loop: Action"
    REFLECTION = "Agent loop: Reflection"

    # --- architecture -----------------------------------------------------
    AGENT_VS_LLM_APP = "AI agent vs LLM application"
    AGENTIC_FIRST = "Agentic-first vs traditional architecture"
    AUTONOMOUS_VS_ASSISTIVE = "Autonomous vs assistive agents"
    SINGLE_VS_MULTI_AGENT = "Single-agent vs multi-agent systems"
    STATELESS_VS_STATEFUL = "Stateless vs stateful agents"

    # --- planning & control ----------------------------------------------
    CHAIN_OF_THOUGHT = "Chain-of-Thought reasoning"
    LEAST_TO_MOST = "Least-to-most planning"
    POLICY_CONSTRAINED = "Policy-constrained reasoning"
    WORKFLOW_DECOMPOSITION = "Workflow decomposition"
    HUMAN_IN_THE_LOOP = "Human-in-the-loop checkpoint"
    CONDITIONAL_BRANCHING = "Conditional branching"

    # --- prompt engineering ----------------------------------------------
    QUERY_REWRITING = "Query rewriting & expansion"
    CONTEXT_AWARE_PROMPT = "Context-aware prompt enhancement"
    PROMPT_LIBRARY = "Prompt libraries & versioning"
    PROMPT_INJECTION_DEFENSE = "Prompt injection defenses"
    CHUNKING = "Chunking strategies"

    # --- safety & operations ---------------------------------------------
    PII_DETECTION = "PII detection & masking"
    NO_FABRICATED_CLAIMS = "No fabricated discounts / grounded claims"
    TONE_POLICY = "Tone policy"
    CONFIDENCE_FALLBACK = "Confidence-threshold fallback"
    DURABLE_STATE = "Durable state & resume"
    AUDIT_LOGGING = "Audit logging"


@dataclass(frozen=True, slots=True)
class ConceptBinding:
    """One place in the code where a concept is implemented."""

    concept: Concept
    module: str
    qualname: str
    layer: str  # "core" | "maf" | "langchain" | "langgraph"

    @property
    def location(self) -> str:
        return f"{self.module}:{self.qualname}"


_REGISTRY: list[ConceptBinding] = []

F = TypeVar("F", bound=Callable)


def _layer_for(module: str) -> str:
    if ".adapters.maf" in module:
        return "maf"
    if ".adapters.lc" in module:
        return "langchain"
    if ".adapters.lg" in module:
        return "langgraph"
    return "core"


def concept(*concepts: Concept) -> Callable[[F], F]:
    """Record that the decorated callable implements the given concepts.

    The decorator is transparent — it returns the function unchanged and adds no
    runtime cost beyond one registry append at import time.
    """

    def decorate(fn: F) -> F:
        module = getattr(fn, "__module__", "?")
        for c in concepts:
            _REGISTRY.append(
                ConceptBinding(
                    concept=c,
                    module=module,
                    qualname=getattr(fn, "__qualname__", str(fn)),
                    layer=_layer_for(module),
                )
            )
        existing = tuple(getattr(fn, "__concepts__", ()))
        fn.__concepts__ = existing + concepts  # type: ignore[attr-defined]
        return fn

    return decorate


def registry() -> list[ConceptBinding]:
    """All bindings recorded so far (import the adapters first for full coverage)."""
    return list(_REGISTRY)


def bindings_for(c: Concept) -> list[ConceptBinding]:
    return [b for b in _REGISTRY if b.concept is c]


def load_all_layers() -> None:
    """Import every adapter so the registry is complete.

    Importing an adapter must never require a configured LLM — construction is
    lazy — so this is safe to call from the CLI and from tests.
    """
    import importlib

    for mod in (
        "onboarding.core.steps",
        "onboarding.adapters.lg.graph",
        "onboarding.adapters.maf.workflow",
        "onboarding.adapters.lc.agent",
    ):
        importlib.import_module(mod)
