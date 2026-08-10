"""Chunking for free-text notes.

Sentence-aware, fixed-budget chunks with overlap. Long ``signup_notes`` are split
before analysis so a note longer than the model's useful context does not push
the policy block out of the prompt, and so an injection attempt buried at the end
of a long note is still scanned.
"""

from __future__ import annotations

import re

from onboarding.core.concepts import Concept, concept
from onboarding.core.rules import RULES
from onboarding.core.schemas import NotesChunk


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token, min 1 per word)."""
    if not text.strip():
        return 0
    return max(len(text) // 4, len(text.split()))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


@concept(Concept.CHUNKING, Concept.PERCEPTION)
def chunk_notes(
    text: str,
    *,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[NotesChunk]:
    """Split ``text`` into overlapping, sentence-aligned chunks."""
    target = target_tokens or RULES.CHUNK_TOKEN_TARGET
    overlap = overlap_tokens if overlap_tokens is not None else RULES.CHUNK_TOKEN_OVERLAP
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[NotesChunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = " ".join(current)
        chunks.append(NotesChunk(index=len(chunks), text=body, token_estimate=estimate_tokens(body)))
        # Carry the tail of this chunk into the next one so a sentence straddling
        # a boundary is never seen only half.
        carried: list[str] = []
        carried_tokens = 0
        for sentence in reversed(current):
            t = estimate_tokens(sentence)
            if carried_tokens + t > overlap:
                break
            carried.insert(0, sentence)
            carried_tokens += t
        current = carried
        current_tokens = carried_tokens

    for sentence in sentences:
        tokens = estimate_tokens(sentence)
        if current and current_tokens + tokens > target:
            flush()
        current.append(sentence)
        current_tokens += tokens

    if current:
        body = " ".join(current)
        if not chunks or chunks[-1].text != body:
            chunks.append(NotesChunk(index=len(chunks), text=body, token_estimate=estimate_tokens(body)))

    return chunks
