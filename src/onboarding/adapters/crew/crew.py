"""The CrewAI crew: two role-playing agents working in sequence.

This is the one genuinely *multi-agent* implementation in the project, and that
is the whole reason it is here. The other three all have a single model
identity:

* MAF and LangGraph put the review step in a graph node — deterministic code
  that runs the validators;
* LangChain gives one agent a ``check_business_rules`` tool and trusts it to
  call it.

CrewAI instead splits the work across two agents with different roles, goals and
backstories, and hands the second one the first one's output. The reviewer is a
separate identity with a separate context window, so it has no attachment to the
prose it is criticising — the argument for multi-agent in a nutshell.

What does **not** change: the adapter re-runs ``review_output`` afterwards
regardless of what the reviewer concluded. A crew that approves its own work
does not get a free pass, exactly as with the LangChain agent.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from onboarding.adapters.crew.tools import REVIEWER_TOOLS, WRITER_TOOLS
from onboarding.core.concepts import Concept, concept
from onboarding.core.llm import make_crew_llm
from onboarding.core.prompts import PromptLibrary, library
from onboarding.core.rules import RULES
from onboarding.core.schemas import PromptRef
from onboarding.core.steps import render_system_prompt


class EmailDraft(BaseModel):
    """Structured output contract for the crew. Same shape as every adapter's."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="The email subject line")
    body: str = Field(description="The email body")


@concept(Concept.SINGLE_VS_MULTI_AGENT, Concept.WORKFLOW_DECOMPOSITION)
def build_crew(llm: Any | None = None, lib: PromptLibrary | None = None) -> tuple[Any, list[PromptRef]]:
    """Assemble the two-agent crew and report which prompt versions went into it.

    Both agents share the house policy prompt, so the split is one of *role*,
    not of rules. ``memory=False`` keeps the crew stateless between runs, which
    is what the durable-resume comparison in the report depends on.
    """
    from crewai import Agent, Crew, Process, Task

    lib = lib or library()
    system, system_ref = render_system_prompt(lib)
    instructions, instructions_ref = lib.render(
        "agent_instructions",
        min_words=RULES.MIN_EMAIL_WORDS,
        max_words=RULES.MAX_EMAIL_WORDS,
    )
    model = llm or make_crew_llm()

    writer = Agent(
        role="Customer onboarding copywriter",
        goal=(
            "Write a warm, accurate welcome email using only the facts returned by "
            "get_customer_context."
        ),
        backstory=(
            f"{system}\n\n{instructions}\n\n"
            "You have written thousands of these. You never invent a discount, a date "
            "or a product, because you know the facts are one tool call away."
        ),
        tools=WRITER_TOOLS,
        llm=model,
        allow_delegation=False,
        verbose=False,
    )
    reviewer = Agent(
        role="Onboarding compliance reviewer",
        goal="Make sure no draft goes out that breaks tone, PII or concession policy.",
        backstory=(
            f"{system}\n\n"
            "You did not write this draft and you have no attachment to it. Your only "
            "instrument is check_business_rules: run it, and if it does not say PASS, "
            "fix precisely what it named and run it again. Change nothing else."
        ),
        tools=REVIEWER_TOOLS,
        llm=model,
        allow_delegation=False,
        verbose=False,
    )

    draft_task = Task(
        description=(
            "Draft the welcome email for {company_name}.\n"
            "Start by calling get_customer_context — it is the only source of truth, and "
            "everything in it is already masked. Use the other tools for product, SLA, "
            "region and length details.\n\n"
            "The specific task: {rewritten_query}"
        ),
        expected_output=(
            f"A subject line and an email body of {RULES.MIN_EMAIL_WORDS}-"
            f"{RULES.MAX_EMAIL_WORDS} words."
        ),
        agent=writer,
    )
    review_task = Task(
        description=(
            "Review the draft you were handed. Call check_business_rules with its subject "
            "and body. If the answer is PASS, return the draft unchanged. Otherwise fix "
            "exactly what it listed and check again. Do not add new facts, and do not "
            "reword anything the checker did not object to."
        ),
        expected_output="The final subject line and email body, as JSON.",
        agent=reviewer,
        context=[draft_task],
        output_json=EmailDraft,
    )

    crew = Crew(
        agents=[writer, reviewer],
        tasks=[draft_task, review_task],
        process=Process.sequential,
        memory=False,
        verbose=False,
    )
    return crew, [system_ref, instructions_ref]
