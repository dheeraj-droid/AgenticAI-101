"""Versioned prompt library.

Prompts live in ``prompts/<id>.v<N>.json`` and the active version of each is
pinned in ``prompts/index.json``. Every file carries a checksum over its
semantic content, so an accidental edit is a hard failure rather than a silent
behaviour change — the same discipline as the JSON prompt library in the
Copilot Studio banking lab, minus the cloud storage.

Rendering is strict: unknown variables, missing required variables, and leftover
placeholders all raise.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from onboarding.core.concepts import Concept, concept
from onboarding.core.config import paths
from onboarding.core.errors import PromptChecksumError, PromptRenderError
from onboarding.core.schemas import PromptRef

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptSpec(BaseModel):
    """One versioned prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    version: int = Field(ge=1)
    status: Literal["active", "deprecated"] = "active"
    role: Literal["system", "user"] = "user"
    description: str = ""
    template: str
    allowed_variables: list[str] = Field(default_factory=list)
    required_variables: list[str] = Field(default_factory=list)
    policy_refs: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    checksum: str = ""

    def semantic_payload(self) -> dict[str, Any]:
        """The fields the checksum covers — description and status may change freely."""
        return {
            "id": self.id,
            "version": self.version,
            "role": self.role,
            "template": self.template,
            "allowed_variables": sorted(self.allowed_variables),
            "required_variables": sorted(self.required_variables),
            "policy_refs": sorted(self.policy_refs),
        }

    def compute_checksum(self) -> str:
        blob = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def ref(self) -> PromptRef:
        return PromptRef(id=self.id, version=self.version, checksum=self.compute_checksum())


class StrictRenderer:
    """``{{var}}`` substitution that refuses to guess."""

    @staticmethod
    def render(spec: PromptSpec, **variables: Any) -> str:
        allowed = set(spec.allowed_variables)
        supplied = set(variables)

        unknown = supplied - allowed
        if unknown:
            raise PromptRenderError(
                f"prompt {spec.id}@v{spec.version}: unknown variable(s) {sorted(unknown)}; "
                f"allowed are {sorted(allowed)}"
            )
        missing = set(spec.required_variables) - supplied
        if missing:
            raise PromptRenderError(
                f"prompt {spec.id}@v{spec.version}: missing required variable(s) {sorted(missing)}"
            )

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                if name in allowed:
                    return ""
                raise PromptRenderError(
                    f"prompt {spec.id}@v{spec.version}: template references {name!r}, "
                    "which is not in allowed_variables"
                )
            return str(variables[name])

        rendered = _PLACEHOLDER.sub(substitute, spec.template)
        leftover = _PLACEHOLDER.findall(rendered)
        if leftover:
            raise PromptRenderError(
                f"prompt {spec.id}@v{spec.version}: unsubstituted placeholder(s) {leftover}"
            )
        return rendered


class PromptLibrary:
    """All prompts on disk, with the active version of each pinned by index.json."""

    def __init__(self, specs: dict[tuple[str, int], PromptSpec], index: dict[str, int], root: Path):
        self._specs = specs
        self._index = index
        self.root = root

    @classmethod
    @concept(Concept.PROMPT_LIBRARY)
    def load(cls, directory: Path | None = None, *, verify: bool = True) -> PromptLibrary:
        root = directory or paths().prompts
        index_path = root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"prompt index not found: {index_path}")
        index: dict[str, int] = json.loads(index_path.read_text())

        specs: dict[tuple[str, int], PromptSpec] = {}
        for file in sorted(root.glob("*.v*.json")):
            spec = PromptSpec.model_validate_json(file.read_text())
            if verify and spec.checksum:
                actual = spec.compute_checksum()
                if actual != spec.checksum:
                    raise PromptChecksumError(
                        f"{file.name}: checksum mismatch.\n  recorded: {spec.checksum}\n"
                        f"  actual:   {actual}\n"
                        "If the change was intentional run: onboarding prompts rechecksum"
                    )
            specs[(spec.id, spec.version)] = spec

        for prompt_id, version in index.items():
            if (prompt_id, version) not in specs:
                raise FileNotFoundError(
                    f"index.json pins {prompt_id}@v{version} but no such prompt file exists"
                )
        return cls(specs, index, root)

    def get(self, prompt_id: str, version: int | None = None) -> PromptSpec:
        if version is None:
            version = self._index.get(prompt_id)
            if version is None:
                raise KeyError(f"prompt {prompt_id!r} is not pinned in index.json")
        try:
            return self._specs[(prompt_id, version)]
        except KeyError as exc:
            raise KeyError(f"prompt {prompt_id}@v{version} not found") from exc

    @concept(Concept.PROMPT_LIBRARY, Concept.CONTEXT_AWARE_PROMPT)
    def render(self, prompt_id: str, *, version: int | None = None, **variables: Any) -> tuple[str, PromptRef]:
        """Render a pinned prompt and return the text plus its exact provenance."""
        spec = self.get(prompt_id, version)
        return StrictRenderer.render(spec, **variables), spec.ref()

    @property
    def pinned(self) -> dict[str, int]:
        return dict(self._index)

    def all_specs(self) -> list[PromptSpec]:
        return [self._specs[k] for k in sorted(self._specs)]


@lru_cache(maxsize=1)
def library() -> PromptLibrary:
    return PromptLibrary.load()
