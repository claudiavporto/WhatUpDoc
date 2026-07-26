"""Regression tests for src/chunking.py.

Note: this is a separate, independent test module from
tests/test_doc_ingestion.py. There are two parallel chunking
implementations in this repo -- src/chunking.py (the live retrieval
pipeline) and utils/doc_ingestion.py (used for the RQ1 chunk-structure
comparison experiments; see Doc_Ingest_Instructions.md) -- and each
gets its own test file rather than one pretending to cover the other.

These tests exercise the three src.chunking strategies (fixed,
sentence, paragraph) directly against synthetic Page objects, so they
run with no Ollama server, no ChromaDB, and no sample corpus required.
"""

from __future__ import annotations

import pytest

from src.chunking import chunk_fixed, chunk_pages, chunk_paragraphs, chunk_sentences
from src.data_loader import Page


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_page() -> Page:
    text = (
        "This is the first sentence. This is the second sentence. "
        "Here comes a third one, slightly longer than the rest. "
        "And a fourth sentence to round things out."
    )
    return Page(text=text, source="sample.pdf", page_number=1)


@pytest.fixture
def paragraph_page() -> Page:
    text = (
        "Short para one.\n\n"
        "Short para two.\n\n"
        "Short para three that is still fairly brief.\n\n"
        + ("Long paragraph sentence. " * 200)  # forces the >max_chars fallback path
    )
    return Page(text=text, source="clauses.docx", page_number=2)


# ---------------------------------------------------------------------------
# chunk_fixed
# ---------------------------------------------------------------------------


def test_fixed_rejects_overlap_ge_chunk_size(sample_page):
    with pytest.raises(ValueError):
        chunk_fixed([sample_page], chunk_size=100, overlap=100)


def test_fixed_produces_chunks_with_citation_metadata(sample_page):
    chunks = chunk_fixed([sample_page], chunk_size=50, overlap=10)
    assert chunks, "fixed strategy produced no chunks for non-empty input"
    for c in chunks:
        assert c.source == sample_page.source
        assert c.page_number == sample_page.page_number
        assert c.strategy == "fixed"
        assert c.text.strip(), "fixed strategy emitted a blank/whitespace-only chunk"


# ---------------------------------------------------------------------------
# chunk_sentences
# ---------------------------------------------------------------------------


def test_sentences_never_split_mid_sentence(sample_page):
    chunks = chunk_sentences([sample_page], max_chars=60, overlap_sentences=0)
    assert chunks
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?")), (
            f"chunk does not end on a sentence boundary: {c.text!r}"
        )


def test_sentences_respects_overlap(sample_page):
    chunks = chunk_sentences([sample_page], max_chars=60, overlap_sentences=1)
    joined = " ".join(c.text for c in chunks)
    assert joined.count("second sentence") >= 1


def test_sentences_no_empty_chunks(sample_page):
    """Regression test for quick_debug_empty_chunks.py: the M3 debug
    script manually scanned legal/policy corpora for near-empty chunks
    (word_count <= 2). This exercises the same failure mode directly,
    without needing a real corpus on disk."""
    chunks = chunk_sentences([sample_page], max_chars=1200)
    for c in chunks:
        assert len(c.text.split()) > 2, f"near-empty chunk: {c.text!r}"


# ---------------------------------------------------------------------------
# chunk_paragraphs
# ---------------------------------------------------------------------------


def test_paragraphs_merges_short_paragraphs(paragraph_page):
    chunks = chunk_paragraphs([paragraph_page], max_chars=1800)
    assert chunks
    assert all(c.strategy == "paragraph" for c in chunks)


def test_paragraphs_falls_back_to_sentence_packing_for_oversized_paragraph(paragraph_page):
    chunks = chunk_paragraphs([paragraph_page], max_chars=1800)
    long_chunks = [c for c in chunks if "Long paragraph sentence." in c.text]
    assert len(long_chunks) > 1, "oversized paragraph was not split"
    for c in long_chunks:
        assert len(c.text) <= 1800 + 50  # small slack for sentence-boundary rounding


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["fixed", "sentence", "paragraph"])
def test_chunk_pages_dispatches_to_each_registered_strategy(sample_page, strategy):
    chunks = chunk_pages([sample_page], strategy)
    assert chunks
    assert all(c.strategy == strategy for c in chunks)


def test_chunk_pages_rejects_unknown_strategy(sample_page):
    with pytest.raises(ValueError):
        chunk_pages([sample_page], "not_a_real_strategy")