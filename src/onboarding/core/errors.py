"""Errors shared by the core pipeline and all four framework adapters."""

from __future__ import annotations


class OnboardingError(Exception):
    """Base class for everything this package raises deliberately."""


class LlmNotConfiguredError(OnboardingError):
    """Raised when an LLM call is attempted without a configured endpoint.

    There is deliberately no stub/fake model anywhere in this package: a missing
    endpoint is a loud failure, never a silently degraded run.
    """

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "No LLM endpoint is configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY "
            "(see .env.example).\n"
            "  Free local profile:  LLM_BASE_URL=http://localhost:11434/v1  "
            "LLM_MODEL=qwen2.5:3b-instruct  LLM_API_KEY=ollama\n"
            "  Anthropic profile:   LLM_BASE_URL=https://api.anthropic.com/v1/  "
            "LLM_MODEL=claude-haiku-4-5-20251001  LLM_API_KEY=sk-ant-...\n"
            f"{detail}".rstrip()
        )


class PromptChecksumError(OnboardingError):
    """A prompt file's recorded checksum does not match its content."""


class PromptRenderError(OnboardingError):
    """A prompt was rendered with unknown, missing or unsubstituted variables."""


