"""Regression tests for src/grounding.py.

The Milestone 3 live run scored 0.0 grounding for both prompt profiles.
Root cause (see grounding.py's module and resolve_citations() docstrings):
the model cited sources in formats the parser didn't recognize as
equivalent to a real citation, so every citation was counted as a
fabrication rather than a format-compliance issue --

  * "[source: file.pdf, page 3]"  (echoed the excerpt-header phrasing)
  * "[excerpt 2, page 3]"         (cited by position instead of by name)
  * "[Excerpt 2]"                 (bare excerpt reference, no page at all)

These tests pin that fix down so a future prompt or parser change can't
silently regress it back to conflating format mismatches with
fabrications.
"""

from __future__ import annotations

from src.grounding import resolve_citations, verify_grounding
from src.llm import REFUSAL_MARKER
from src.vector_store import RetrievedChunk


def _chunk(source: str, page: int, text: str = "irrelevant body text") -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source=source, page_number=page,
        chunk_id=f"{source}:p{page}", distance=0.1,
    )


RETRIEVED = [
    _chunk("lease_agreement.pdf", 2),   # excerpt 1
    _chunk("lease_agreement.pdf", 3),   # excerpt 2
]


# ---------------------------------------------------------------------------
# resolve_citations: the three M3 nonstandard formats + the standard one
# ---------------------------------------------------------------------------


def test_standard_citation_is_not_flagged_nonstandard():
    answer = "Rent is $1,200/month [lease_agreement.pdf, page 2]."
    citations, nonstandard = resolve_citations(answer, RETRIEVED)
    assert citations == [("lease_agreement.pdf", 2)]
    assert nonstandard == []


def test_source_prefixed_citation_resolves_and_is_flagged():
    answer = "Rent is $1,200/month [source: lease_agreement.pdf, page 2]."
    citations, nonstandard = resolve_citations(answer, RETRIEVED)
    assert citations == [("lease_agreement.pdf", 2)]
    assert len(nonstandard) == 1


def test_excerpt_number_with_page_resolves_to_real_source():
    # "excerpt 2" is the 2nd retrieved chunk -> lease_agreement.pdf
    answer = "Late fees apply [excerpt 2, page 3]."
    citations, nonstandard = resolve_citations(answer, RETRIEVED)
    assert citations == [("lease_agreement.pdf", 3)]
    assert len(nonstandard) == 1


def test_bare_excerpt_reference_resolves_via_retrieved_metadata():
    # "[Excerpt 2]" has no page at all -> pulled straight from RETRIEVED[1]
    answer = "Late fees apply [Excerpt 2]."
    citations, nonstandard = resolve_citations(answer, RETRIEVED)
    assert citations == [("lease_agreement.pdf", 3)]
    assert len(nonstandard) == 1


def test_excerpt_number_out_of_range_is_a_genuine_fabrication():
    # only 2 excerpts were retrieved; "excerpt 9" was never shown to the model
    answer = "Rent is $1,200/month [excerpt 9, page 3]."
    citations, _ = resolve_citations(answer, RETRIEVED)
    available = {(c.source.lower(), c.page_number) for c in RETRIEVED}
    assert citations[0] not in available


# ---------------------------------------------------------------------------
# verify_grounding: end-to-end score + refusal handling
# ---------------------------------------------------------------------------


def test_grounding_score_is_perfect_when_all_citations_are_nonstandard_but_valid():
    """The exact M3 regression: before the fix, an answer that cited
    every claim correctly but in the wrong *format* scored 0.0. It
    should score 1.0, with the format issue reported separately via
    `nonstandard` / `format_compliance`, not folded into the score."""
    answer = (
        "Rent is $1,200/month [source: lease_agreement.pdf, page 2]. "
        "A late fee applies after 5 days [Excerpt 2]."
    )
    report = verify_grounding(answer, RETRIEVED, REFUSAL_MARKER)
    assert report.grounding_score == 1.0
    assert report.unsupported == []
    assert len(report.nonstandard) == 2
    assert report.format_compliance == 0.0  # zero citations used the instructed format


def test_grounding_score_penalizes_true_fabrication():
    answer = "The CEO's salary is $2M [made_up_source.pdf, page 1]."
    report = verify_grounding(answer, RETRIEVED, REFUSAL_MARKER)
    assert report.grounding_score == 0.0
    assert len(report.unsupported) == 1


def test_refusal_is_not_scored_as_a_claim():
    report = verify_grounding(REFUSAL_MARKER, RETRIEVED, REFUSAL_MARKER)
    assert report.is_refusal is True
    assert report.grounding_score is None