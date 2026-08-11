"""Runtime configuration: paths and the single LLM endpoint spec."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def project_root() -> Path:
    """Repository root — the directory holding ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


@dataclass(frozen=True, slots=True)
class Paths:
    root: Path

    @property
    def prompts(self) -> Path:
        return self.root / "prompts"

    @property
    def fixtures(self) -> Path:
        return self.root / "fixtures" / "customers"

    @property
    def runs(self) -> Path:
        return self.root / ".runs"

    @property
    def audit_log(self) -> Path:
        return self.runs / "audit.jsonl"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    def ensure_runs(self) -> Path:
        self.runs.mkdir(parents=True, exist_ok=True)
        return self.runs


@lru_cache(maxsize=1)
def paths() -> Paths:
    override = os.environ.get("ONBOARDING_ROOT")
    return Paths(root=Path(override).resolve() if override else project_root())


@dataclass(frozen=True, slots=True)
class LlmSpec:
    """One OpenAI-compatible endpoint, shared by all four frameworks."""

    base_url: str | None
    model: str | None
    api_key: str | None
    profile: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def redacted(self) -> dict[str, str]:
        return {
            "profile": self.profile,
            "base_url": self.base_url or "(unset)",
            "model": self.model or "(unset)",
            "api_key": "(set)" if self.api_key else "(unset)",
        }


def llm_spec() -> LlmSpec:
    """Read the endpoint from the environment. Never raises — see ``llm.require_llm``."""
    return LlmSpec(
        base_url=os.environ.get("LLM_BASE_URL") or None,
        model=os.environ.get("LLM_MODEL") or None,
        api_key=os.environ.get("LLM_API_KEY") or None,
        profile=os.environ.get("LLM_PROFILE", "ollama"),
    )


def force_regex_pii() -> bool:
    return os.environ.get("ONBOARDING_FORCE_REGEX_PII", "").strip() not in ("", "0", "false")
