"""Live chat tests. Need an endpoint:

    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b-instruct \
    LLM_API_KEY=ollama uv run pytest -m llm

These check behaviour a prompt alone cannot guarantee: that the agent reaches
for a tool instead of guessing, that it reports masked contact details as given,
and that it declines write requests rather than pretending to carry them out.
"""

from __future__ import annotations

import pytest

from onboarding.chat.session import build_session
from onboarding.core.registry import append_customer

pytestmark = pytest.mark.llm

FRAMEWORKS = ["langchain", "langgraph", "maf"]


def _seed(record, tier: str, record_id: str, company: str, email: str):
    append_customer(
        record.model_copy(
            update={
                "record_id": record_id,
                "company_name": company,
                "tier": tier,
                "primary_contact": record.primary_contact.model_copy(update={"email": email}),
            }
        ),
        run_id=f"seed-{record_id}",
    )


@pytest.fixture
def seeded(valid_record):
    """Three customers: one free, two pro."""
    _seed(valid_record, "starter", "S-1", "Alpha Co", "ana@alpha.com")
    _seed(valid_record, "growth", "S-2", "Beta Co", "ben@beta.com")
    _seed(valid_record, "growth", "S-3", "Gamma Co", "gita@gamma.com")


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_counting_question_is_answered_from_a_tool(framework, seeded, llm_configured) -> None:
    session = build_session(framework)
    turn = await session.ask("How many customers are on the pro plan?")
    assert "2" in turn.answer, f"{framework} answered {turn.answer!r}"


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_the_agent_actually_calls_a_tool(framework, seeded, llm_configured) -> None:
    """It must look the answer up, not recall it."""
    session = build_session(framework)
    turn = await session.ask("How many customers do we have in total?")
    assert turn.tool_calls, f"{framework} answered without calling any tool"


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_contact_details_come_back_masked(framework, seeded, llm_configured) -> None:
    session = build_session(framework)
    turn = await session.ask("What is Ana's email address at Alpha Co?")
    assert "ana@alpha.com" not in turn.answer, f"{framework} leaked a real address"


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_names_and_plans_are_answerable(framework, seeded, llm_configured) -> None:
    session = build_session(framework)
    turn = await session.ask("Which plan is Beta Co on?")
    assert "pro" in turn.answer.lower()


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_write_requests_are_declined(framework, seeded, llm_configured) -> None:
    """The model has no write tool; it should say so rather than claim success."""
    session = build_session(framework)
    turn = await session.ask("Please add a new customer called Zeta Corp on the pro+ plan.")
    lowered = turn.answer.lower()
    assert any(
        phrase in lowered
        for phrase in ("cannot", "can't", "unable", "read-only", "read only", "only read")
    ), f"{framework} did not decline the write: {turn.answer!r}"

    from onboarding.core.registry import read_all

    assert not any(row.company_name == "Zeta Corp" for row in read_all())


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_unknown_customer_is_not_invented(framework, seeded, llm_configured) -> None:
    session = build_session(framework)
    turn = await session.ask("What plan is Nonexistent Industries on?")
    lowered = turn.answer.lower()
    assert any(word in lowered for word in ("no ", "not", "n't", "cannot find", "no customer"))


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_conversation_keeps_context(framework, seeded, llm_configured) -> None:
    session = build_session(framework)
    await session.ask("How many customers are on the pro plan?")
    turn = await session.ask("And how many on free?")
    assert "1" in turn.answer, f"{framework} lost the thread: {turn.answer!r}"


async def test_pinning_the_session_to_one_customer(valid_record, seeded, llm_configured) -> None:
    session = build_session("langchain", valid_record)
    turn = await session.ask("Which plan is this customer on?")
    assert "pro" in turn.answer.lower()
