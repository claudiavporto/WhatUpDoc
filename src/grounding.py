"""Grounding and faithfulness layer for WhatUpDoc.

Owner: Christopher Swartz (feature/llm-infrastructure)

A RAG answer is only trustworthy if two things hold:

  1. The model cited sources that were actually retrieved (not invented).
  2. The context handed to the model fit inside its window, so nothing
     was silently truncated by the runtime.

This module provides both guarantees so that src/llm.py can return an
answer *plus a measurable grounding score*, rather than raw text the
user has to trust blindly.

Nothing here touches the network — it is pure text analysis over the
model's output and the retrieved chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.vector_store import RetrievedChunk

# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------

# Matches [source, page N] and tolerant variants: [file.pdf, p. 3],
# [file.pdf, page 3], [file.pdf, pg 3]. Source is any text up to the comma.
_CITATION_RE = re.compile(
    r"\[\s*([^\[\],]+?)\s*,\s*(?:page|pg|p\.?)\s*(\d+)\s*\]",
    flags=re.IGNORECASE,
)


def _norm(source: str) -> str:
    """Normalize a source name for comparison (case/whitespace-insensitive)."""
    return source.strip().lower()


def parse_citations(answer: str) -> list[tuple[str, int]]:
    """Extract (source, page) pairs cited in the answer text."""
    return [(_norm(src), int(page)) for src, page in _CITATION_RE.findall(answer)]


# ---------------------------------------------------------------------------
# Faithfulness verification
# ---------------------------------------------------------------------------


@dataclass
class GroundingReport:
    """Measurable faithfulness summary for a single answer.

    grounding_score is the fraction of the answer's citations that point
    to a source/page actually present in the retrieved context. A score
    below 1.0 means the model fabricated at least one citation — the
    exact failure mode this project exists to prevent.
    """

    is_refusal: bool
    cited: list[tuple[str, int]]
    supported: list[tuple[str, int]] = field(default_factory=list)
    unsupported: list[tuple[str, int]] = field(default_factory=list)
    grounding_score: float | None = None  # None when the answer cites nothing

    def to_dict(self) -> dict:
        return {
            "is_refusal": self.is_refusal,
            "n_citations": len(self.cited),
            "n_supported": len(self.supported),
            "n_unsupported": len(self.unsupported),
            "grounding_score": self.grounding_score,
            "unsupported_citations": [f"{s}, page {p}" for s, p in self.unsupported],
        }

    def summary_line(self) -> str:
        if self.is_refusal:
            return "grounding: refusal (no answer claimed)"
        if self.grounding_score is None:
            return "grounding: WARNING — answer made claims with no citations"
        flag = "" if not self.unsupported else f" | {len(self.unsupported)} FABRICATED"
        return f"grounding: {self.grounding_score:.0%} of citations supported{flag}"


def verify_grounding(
    answer: str,
    retrieved: list[RetrievedChunk],
    refusal_marker: str,
) -> GroundingReport:
    """Score how faithfully an answer is grounded in the retrieved context.

    Args:
        answer: The model's generated answer.
        retrieved: The chunks passed to the model as context.
        refusal_marker: The exact refusal string the prompt mandates; if
            the answer contains it, the answer is treated as a (correct)
            refusal rather than a claim to be verified.

    Returns:
        A GroundingReport. Callers can log summary_line(), gate on
        grounding_score, or surface unsupported citations to the user.
    """
    is_refusal = refusal_marker.lower() in answer.lower()

    available = {(_norm(c.source), int(c.page_number)) for c in retrieved}
    cited = parse_citations(answer)

    supported = [c for c in cited if c in available]
    unsupported = [c for c in cited if c not in available]
    score = (len(supported) / len(cited)) if cited else None

    return GroundingReport(
        is_refusal=is_refusal,
        cited=cited,
        supported=supported,
        unsupported=unsupported,
        grounding_score=score,
    )


# ---------------------------------------------------------------------------
# Token-budgeted context packing
# ---------------------------------------------------------------------------


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Rough token estimate.

    LLaMA/Mistral use SentencePiece tokenizers we don't ship, so we
    approximate with a chars-per-token heuristic (~4 for English). This
    intentionally over-estimates slightly, which is the safe direction:
    better to leave a little window headroom than to overflow it.
    """
    return max(1, len(text) // chars_per_token)


def pack_context(
    chunks: list[RetrievedChunk],
    token_budget: int,
    reserved_tokens: int = 0,
    chars_per_token: int = 4,
) -> list[RetrievedChunk]:
    """Greedily select top-ranked chunks that fit within a token budget.

    Chunks are assumed to be ranked best-first (as the vector store
    returns them). We keep the most relevant ones and drop the tail if
    the full set would exceed the model's usable context window. This
    prevents the Ollama runtime from silently truncating the prompt,
    which would break citations for the dropped-but-unlabelled text.

    Args:
        chunks: Retrieval hits, best first.
        token_budget: Total tokens available for context.
        reserved_tokens: Tokens to hold back for the system prompt,
            question, and the answer itself.

    Returns:
        The prefix of chunks that fits; always at least one chunk if any
        were provided, so retrieval never returns empty-handed.
    """
    budget = max(0, token_budget - reserved_tokens)
    selected: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.text, chars_per_token)
        if selected and used + cost > budget:
            break
        selected.append(chunk)
        used += cost
    return selected
