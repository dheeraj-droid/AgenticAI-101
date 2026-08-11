"""Framework-native ``LlmCaller`` implementations.

``core.steps`` depends only on the ``LlmCaller`` protocol, so the pipeline never
imports a framework. Each adapter passes the caller built on its own SDK, which
means the LLM traffic really does go through LangChain / agent_framework
respectively rather than through a shared bypass.
"""

from __future__ import annotations

from typing import Any

from onboarding.core.llm import extract_json_object, make_langchain_model, make_maf_client


class LangChainLlmCaller:
    """Uses ``langchain_openai.ChatOpenAI`` — shared by the LangChain and LangGraph adapters."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model or make_langchain_model()

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        reply = await self._model.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        return extract_json_object(_text_of(reply.content))


class MafLlmCaller:
    """Uses ``agent_framework_openai.OpenAIChatCompletionClient``."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client or make_maf_client()

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        from agent_framework import Message

        response = await self._client.get_response(
            [Message(role="system", text=system), Message(role="user", text=user)]
        )
        return extract_json_object(str(response.text or ""))


class CrewLlmCaller:
    """Uses ``crewai.LLM``.

    CrewAI's LLM is synchronous, so the call is pushed to a worker thread rather
    than blocking the loop the adapter runs on.
    """

    def __init__(self, model: Any | None = None) -> None:
        from onboarding.core.llm import make_crew_llm

        self._model = model or make_crew_llm()

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        import asyncio

        text = await asyncio.to_thread(
            self._model.call,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return extract_json_object(str(text or ""))


def _text_of(content: Any) -> str:
    """LangChain message content is str or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)
