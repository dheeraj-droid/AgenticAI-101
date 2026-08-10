"""The single LangChain agent.

One ``create_agent`` call, one tool-using loop, no graph and no checkpointer.
The agent chooses which tools to call and in what order — that freedom is the
point of the comparison against the two explicit graphs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from onboarding.adapters.lc.tools import AGENT_TOOLS
from onboarding.core.concepts import Concept, concept
from onboarding.core.llm import make_langchain_model
from onboarding.core.prompts import PromptLibrary, library
from onboarding.core.rules import RULES
from onboarding.core.schemas import PromptRef
from onboarding.core.steps import render_system_prompt


class EmailDraft(BaseModel):
    """Structured output contract for the agent."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="The email subject line")
    body: str = Field(description="The email body")


@concept(Concept.SINGLE_VS_MULTI_AGENT, Concept.AGENT_VS_LLM_APP, Concept.STATELESS_VS_STATEFUL)
def build_agent(
    model: Any | None = None, lib: PromptLibrary | None = None
) -> tuple[Any, list[PromptRef]]:
    """Build the agent and report which prompt versions went into it.

    ``checkpointer=None`` is deliberate, not an oversight: this adapter is the
    stateless point of comparison. It has no thread to resume, which is exactly
    what makes the stateful/stateless distinction demonstrable rather than
    merely asserted.
    """
    from langchain.agents import create_agent

    lib = lib or library()
    system, system_ref = render_system_prompt(lib)
    instructions, instructions_ref = lib.render(
        "agent_instructions",
        min_words=RULES.MIN_EMAIL_WORDS,
        max_words=RULES.MAX_EMAIL_WORDS,
    )
    agent = create_agent(
        model=model or make_langchain_model(),
        tools=AGENT_TOOLS,
        system_prompt=f"{system}\n\n{instructions}",
        response_format=EmailDraft,
        checkpointer=None,
    )
    return agent, [system_ref, instructions_ref]


AGENT_TOOL_NAMES = tuple(t.name for t in AGENT_TOOLS)
