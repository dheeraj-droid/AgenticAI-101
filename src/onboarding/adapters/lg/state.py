"""LangGraph channel state.

The whole pipeline state travels as one serialised ``OnboardingState`` blob so
the SqliteSaver checkpoint is self-contained and a fresh process can resume from
it without needing the original record file.
"""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    """Channels for the onboarding graph."""

    state: dict[str, Any]  # serialised OnboardingState
    decision: str | None  # human approval decision, supplied on resume
    route: str  # last routing choice, for tracing
