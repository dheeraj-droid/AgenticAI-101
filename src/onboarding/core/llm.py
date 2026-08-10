"""LLM wiring.

One OpenAI-compatible endpoint drives all three frameworks:

* Microsoft Agent Framework -> ``OpenAIChatCompletionClient(base_url=..., model=...)``
* LangChain / LangGraph     -> ``ChatOpenAI(base_url=..., model=...)``

There is no fake, stub or mock model in this module or anywhere else in ``src``.
If the endpoint is not configured, ``require_llm`` raises immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from onboarding.core.config import LlmSpec, llm_spec
from onboarding.core.errors import LlmNotConfiguredError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_framework_openai import OpenAIChatCompletionClient
    from langchain_openai import ChatOpenAI


class LlmCaller(Protocol):
    """What ``core.steps`` needs from a model — deliberately tiny.

    Each adapter supplies its own framework-native implementation, so the core
    pipeline never imports a framework and the frameworks never re-implement the
    pipeline.
    """

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        """Return the model's reply parsed as a JSON object."""
        ...


def require_llm() -> LlmSpec:
    spec = llm_spec()
    if not spec.configured:
        raise LlmNotConfiguredError()
    return spec


def make_maf_client() -> OpenAIChatCompletionClient:
    """Chat-Completions client for Microsoft Agent Framework.

    Deliberately the Chat Completions client, not the Responses-API one: Ollama
    and most local servers only implement ``/v1/chat/completions``.
    """
    from agent_framework_openai import OpenAIChatCompletionClient

    spec = require_llm()
    return OpenAIChatCompletionClient(
        model=spec.model,
        api_key=spec.api_key or "not-needed",
        base_url=spec.base_url,
    )


def make_langchain_model(**kwargs: Any) -> ChatOpenAI:
    """Chat model for LangChain and LangGraph."""
    from langchain_openai import ChatOpenAI

    spec = require_llm()
    return ChatOpenAI(
        model=spec.model,
        api_key=spec.api_key or "not-needed",
        base_url=spec.base_url,
        temperature=kwargs.pop("temperature", 0.2),
        **kwargs,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Small local models routinely wrap JSON in prose or a ``json`` fence, so we
    scan for the first balanced ``{...}`` rather than trusting the whole string.
    """
    import json

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(parsed, dict):
                    return parsed
    raise ValueError(f"model reply contained no JSON object: {text[:200]!r}")
