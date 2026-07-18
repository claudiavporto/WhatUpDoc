"""Chunking strategies for WhatUpDoc.

Implements the three strategies compared in Research Question 1:

  1. fixed     — fixed character windows with overlap (baseline)
  2. sentence  — whole sentences packed up to a size limit (default)
  3. paragraph — paragraph-boundary chunks, merging short paragraphs

Sentence splitting uses spaCy when installed (higher quality on legal
text with abbreviations like "Sec." and "No."); otherwise it falls back
to a regex splitter so the pipeline still runs in minimal environments.

All strategies carry the Page metadata through to each chunk, which is
what lets the generator cite "source, page N" in its answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.data_loader import Page
from utils.helpers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentence splitting (spaCy preferred, regex fallback)
# ---------------------------------------------------------------------------

_NLP = None


def _get_spacy():
    """Load spaCy's small English model once, if available."""
    global _NLP
    if _NLP is None:
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except Exception:
            _NLP = False  # sentinel: tried and unavailable
            logger.warning("spaCy model unavailable; using regex sentence splitter.")
    return _NLP or None


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (spaCy if available, else regex)."""
    nlp = _get_spacy()
    if nlp is not None:
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A retrieval unit: text plus everything needed to cite it."""

    text: str
    source: str
    page_number: int
    chunk_id: str
    strategy: str
    metadata: dict = field(default_factory=dict)


def _make_chunk(text: str, page: Page, index: int, strategy: str) -> Chunk:
    return Chunk(
        text=text.strip(),
        source=page.source,
        page_number=page.page_number,
        chunk_id=f"{page.source}:p{page.page_number}:{strategy}:{index}",
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Strategy 1 — fixed-size windows with overlap
# ---------------------------------------------------------------------------


def chunk_fixed(pages: list[Page], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Fixed character windows with overlap.

    Simple and predictable, but can split mid-sentence, which hurts
    both embedding quality and answer readability.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        text, start, i = page.text, 0, 0
        while start < len(text):
            piece = text[start : start + chunk_size]
            if piece.strip():
                chunks.append(_make_chunk(piece, page, i, "fixed"))
                i += 1
            start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — sentence-boundary packing (default)
# ---------------------------------------------------------------------------


def chunk_sentences(pages: list[Page], max_chars: int = 1200, overlap_sentences: int = 1) -> list[Chunk]:
    """Pack whole sentences into chunks up to max_chars.

    Never splits mid-sentence. Consecutive chunks share
    `overlap_sentences` trailing sentences so context that straddles a
    boundary is retrievable from either side.
    """
    chunks: list[Chunk] = []
    for page in pages:
        sentences = split_sentences(page.text)
        current: list[str] = []
        length, i = 0, 0
        for sent in sentences:
            # flush when adding this sentence would exceed the limit
            if current and length + len(sent) + 1 > max_chars:
                chunks.append(_make_chunk(" ".join(current), page, i, "sentence"))
                i += 1
                current = current[-overlap_sentences:] if overlap_sentences else []
                length = sum(len(s) + 1 for s in current)
            current.append(sent)
            length += len(sent) + 1
        if current:
            chunks.append(_make_chunk(" ".join(current), page, i, "sentence"))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3 — paragraph boundaries
# ---------------------------------------------------------------------------


def chunk_paragraphs(pages: list[Page], max_chars: int = 1800) -> list[Chunk]:
    """Chunk on paragraph breaks, merging short paragraphs up to max_chars.

    Preserves the author's own topical structure — well suited to
    contracts and policies whose clauses are already paragraph-shaped.
    Oversized single paragraphs fall back to sentence packing.
    """
    chunks: list[Chunk] = []
    for page in pages:
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        current: list[str] = []
        length, i = 0, 0

        def flush():
            nonlocal current, length, i
            if current:
                chunks.append(_make_chunk("\n\n".join(current), page, i, "paragraph"))
                i += 1
                current, length = [], 0

        for para in paragraphs:
            if len(para) > max_chars:
                # paragraph alone exceeds limit -> sentence-pack it
                flush()
                sub = chunk_sentences([Page(para, page.source, page.page_number)], max_chars)
                for c in sub:
                    c.chunk_id = f"{page.source}:p{page.page_number}:paragraph:{i}"
                    c.strategy = "paragraph"
                    chunks.append(c)
                    i += 1
                continue
            if length + len(para) + 2 > max_chars:
                flush()
            current.append(para)
            length += len(para) + 2
        flush()
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentences,
    "paragraph": chunk_paragraphs,
}


def chunk_pages(pages: list[Page], strategy: str, **kwargs) -> list[Chunk]:
    """Chunk pages using the named strategy from configs/config.yaml."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES)}")
    chunks = STRATEGIES[strategy](pages, **kwargs)
    logger.info("Chunked %d page(s) -> %d chunk(s) using '%s'", len(pages), len(chunks), strategy)
    return chunks
