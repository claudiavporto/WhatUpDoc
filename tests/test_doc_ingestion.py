"""Regression tests for utils/doc_ingestion.py.

Replaces the print-based debug scripts from Milestone 3
(test_txt_ingestion.py, quick_debug_empty_chunks.py) with real
assertions. Those scripts were not broken (utils/doc_ingestion.py is a
real, working module -- see Doc_Ingest_Instructions.md for why it
exists in parallel with src/chunking.py), but they relied on a human
reading console output rather than a test that fails loudly and
specifically in CI, and they depended on real files under
data/corpus/{legal,medical,policy} being present and unchanged.

These tests build small synthetic .txt/.pdf/.docx fixtures instead, so
they exercise the same parsing and chunking logic without depending on
the real corpus.
"""

from __future__ import annotations

import pytest

from utils.doc_ingestion import (
    chunk_document,
    clean_text,
    fixed_chunking,
    ingest_document,
    paragraph_chunking,
    parse_document,
    parse_docx,
    parse_pdf,
    parse_txt,
    sentence_chunking,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_collapses_whitespace_and_strips():
    assert clean_text("  hello    world  \n") == "hello world"


# ---------------------------------------------------------------------------
# parse_txt: the two documented fixes (single-newline splitting,
# literal "\n" normalization) plus the encoding fallback
# ---------------------------------------------------------------------------


def test_parse_txt_splits_on_single_newlines_not_blank_lines(tmp_path):
    """This dataset separates paragraphs with a single newline; splitting
    on \\n\\n (the old bug this fix addresses) would collapse the whole
    note into one paragraph."""
    f = tmp_path / "clinical_note.txt"
    f.write_text("Chief complaint: cough.\nAssessment: viral URI.\nPlan: rest.")
    paragraphs, metadata = parse_txt(f)
    assert len(paragraphs) == 3
    assert paragraphs[0]["text"] == "Chief complaint: cough."
    assert metadata["file_type"] == "txt"
    assert metadata["total_paragraphs"] == 3


def test_parse_txt_normalizes_literal_backslash_n(tmp_path):
    """Some source records store paragraph breaks as a literal two-character
    "\\n" sequence rather than an actual newline byte."""
    f = tmp_path / "literal_backslash.txt"
    f.write_text("First paragraph.\\nSecond paragraph.")
    paragraphs, _ = parse_txt(f)
    assert len(paragraphs) == 2
    assert paragraphs[0]["text"] == "First paragraph."
    assert paragraphs[1]["text"] == "Second paragraph."


def test_parse_txt_skips_blank_paragraphs(tmp_path):
    f = tmp_path / "with_blanks.txt"
    f.write_text("First.\n\nSecond.")  # the middle "paragraph" is blank
    paragraphs, metadata = parse_txt(f)
    assert len(paragraphs) == 2
    assert metadata["total_paragraphs"] == 2  # updated to post-filter count


def test_parse_txt_falls_back_to_latin1_on_invalid_utf8(tmp_path):
    f = tmp_path / "latin1.txt"
    # 0xE9 is "é" in latin-1 but not valid standalone UTF-8
    f.write_bytes("Caf\xe9 note.".encode("latin-1"))
    paragraphs, _ = parse_txt(f)
    assert "Café" in paragraphs[0]["text"] or "Caf" in paragraphs[0]["text"]


def test_parse_txt_paragraph_metadata_shape(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Only paragraph.")
    paragraphs, _ = parse_txt(f)
    assert paragraphs[0]["metadata"]["filename"] == "note.txt"
    assert paragraphs[0]["metadata"]["paragraph_number"] == 1


# ---------------------------------------------------------------------------
# parse_document: dispatch + unsupported extension
# ---------------------------------------------------------------------------


def test_parse_document_rejects_unsupported_extension(tmp_path):
    f = tmp_path / "notes.rtf"
    f.write_text("not a supported type")
    with pytest.raises(ValueError):
        parse_document(f)


def test_parse_document_dispatches_txt(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Some clinical text.")
    paragraphs, metadata = parse_document(f)
    assert metadata["file_type"] == "txt"
    assert paragraphs


# ---------------------------------------------------------------------------
# fixed_chunking
# ---------------------------------------------------------------------------


def test_fixed_chunking_rejects_overlap_ge_chunk_size():
    records = [{"text": "word " * 50, "metadata": {"filename": "x.txt"}}]
    with pytest.raises(ValueError):
        fixed_chunking(records, chunk_size=10, overlap=10)


def test_fixed_chunking_produces_chunk_metadata():
    records = [{"text": "word " * 50, "metadata": {"filename": "x.txt"}}]
    chunks = fixed_chunking(records, chunk_size=20, overlap=5)
    assert chunks
    for c in chunks:
        assert c["metadata"]["chunking_strategy"] == "fixed"
        assert c["metadata"]["filename"] == "x.txt"
        assert c["metadata"]["token_count"] > 0
        assert c["text"].strip()


# ---------------------------------------------------------------------------
# sentence_chunking
# ---------------------------------------------------------------------------


def test_sentence_chunking_no_empty_chunks():
    """Regression test for quick_debug_empty_chunks.py's manual scan for
    near-empty chunks (word_count <= 2) across the legal/policy corpus."""
    records = [{
        "text": "This is sentence one. This is sentence two. "
                "This is a third, slightly longer sentence to round it out.",
        "metadata": {"filename": "clause.pdf"},
    }]
    chunks = sentence_chunking(records, chunk_size=15)
    for c in chunks:
        assert len(c["text"].split()) > 2, f"near-empty chunk: {c['text']!r}"
        assert c["metadata"]["chunking_strategy"] == "sentence"


def test_sentence_chunking_never_splits_mid_sentence():
    records = [{
        "text": "Short one. Short two. Short three. Short four.",
        "metadata": {"filename": "clause.pdf"},
    }]
    chunks = sentence_chunking(records, chunk_size=5)
    for c in chunks:
        assert c["text"].strip().endswith("."), f"chunk not sentence-terminated: {c['text']!r}"


# ---------------------------------------------------------------------------
# paragraph_chunking
# ---------------------------------------------------------------------------


def test_paragraph_chunking_inherits_metadata_from_first_paragraph_in_chunk():
    records = [
        {"text": "First paragraph.", "metadata": {"filename": "a.docx", "paragraph_number": 1}},
        {"text": "Second paragraph.", "metadata": {"filename": "a.docx", "paragraph_number": 2}},
    ]
    chunks = paragraph_chunking(records, chunk_size=100)
    assert len(chunks) == 1  # both fit in one chunk at this size
    assert chunks[0]["metadata"]["paragraph_number"] == 1  # from the FIRST paragraph
    assert chunks[0]["metadata"]["chunking_strategy"] == "paragraph"


# ---------------------------------------------------------------------------
# chunk_document / ingest_document dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["fixed", "sentence", "paragraph"])
def test_chunk_document_dispatches_to_each_strategy(strategy):
    records = [{"text": "Some text to chunk. More text here.", "metadata": {"filename": "x.txt"}}]
    chunks = chunk_document(records, strategy)
    assert chunks
    assert all(c["metadata"]["chunking_strategy"] == strategy for c in chunks)


def test_chunk_document_rejects_invalid_strategy():
    records = [{"text": "text", "metadata": {"filename": "x.txt"}}]
    with pytest.raises(ValueError):
        chunk_document(records, "not_a_real_strategy")


@pytest.mark.parametrize("strategy", ["fixed", "sentence", "paragraph"])
def test_ingest_document_end_to_end_txt(tmp_path, strategy):
    """Same check as test_txt_ingestion.py, across all three strategies,
    as assertions instead of printed output."""
    f = tmp_path / "clinical_note.txt"
    f.write_text(
        "Chief complaint: persistent cough for two weeks.\n"
        "Assessment: likely viral upper respiratory infection.\n"
        "Plan: supportive care, follow up if symptoms worsen."
    )
    chunks, metadata = ingest_document(str(f), strategy=strategy)
    assert metadata["file_type"] == "txt"
    assert chunks
    assert chunks[0]["text"]
    assert chunks[0]["metadata"]["chunking_strategy"] == strategy