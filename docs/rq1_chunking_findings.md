# RQ1 Preliminary Findings: Chunking Strategy Comparison

Owner: Claudia Porto (feature/performance-testing)

## Status

**Preliminary / structural analysis only.** This document covers chunk *size and structure*
across the three chunking strategies (`fixed`, `sentence`, `paragraph`) using
`utils/doc_ingestion.py`, run against the full validated corpus (286 files: 49 legal,
200 medical, 37 policy).

This does **not** yet answer RQ1's actual question — whether chunking strategy affects
*retrieval precision*. That requires embedding each strategy's output, loading it into
ChromaDB, and measuring recall/precision against a query set with known correct answers
(see "Next Steps" below).

## Method

Ran `compare_chunking.py` (repo root) against `data/corpus/`, using `utils/doc_ingestion.py`'s
`ingest_document()` with each of the three strategies. Reports chunk count, mean/median
word count per chunk, min/max, and standard deviation — overall and broken out by domain.

## Bugs Found and Fixed in `utils/doc_ingestion.py`

Three bugs were found and fixed while preparing this analysis, all in the shared
chunking/parsing logic (not specific to any one file type):

1. **No `.txt` support.** `SUPPORTED_EXTENSIONS` only included `.pdf` and `.docx`. Added
   `parse_txt()` (paragraph-based, mirroring `parse_docx()`'s shape) and updated the
   dispatch in `parse_document()`. Required a second fix once real data was tested:
   the medical `.txt` corpus (sourced from Hugging Face) delimits paragraphs with a
   single `\n`, not a blank line (`\n\n`) — the initial `parse_txt()` implementation
   assumed the latter and returned the whole note as one paragraph.

2. **Empty-chunk bug in `paragraph_chunking()`.** When the first paragraph/page record
   passed to the function already exceeded `CHUNK_SIZE` on its own, the overflow branch
   flushed `current_chunk` (still empty at that point) as a real chunk before starting
   the actual content. Fixed by guarding the flush with `if current_chunk:`.

3. **Metadata misattribution in `paragraph_chunking()`.** When multiple paragraphs were
   merged into one chunk, the chunk's metadata (page/paragraph number) was taken from
   whichever paragraph *triggered the overflow*, not the paragraph the merged chunk
   actually *started with*. This meant a chunk's cited location could point to the wrong
   part of the document — a direct risk to the grounding/citation feature. Fixed by
   tracking `current_chunk_metadata` explicitly (locked in when a chunk starts, not when
   it flushes).

4. **Same empty-chunk bug in `sentence_chunking()`.** Identical root cause to (2), but
   triggered by a single spaCy "sentence" exceeding `CHUNK_SIZE`, most often on pages
   with no terminal punctuation (see Data Quality Observations below). Fixed with the
   same `if current_chunk:` guard.

All four fixes were verified against real corpus files before/after (see git history
on `utils/doc_ingestion.py`).

## Results (Full Corpus, Post-Fix)

Word counts per chunk, by strategy and domain:

| Strategy  | Domain  | Chunks | Mean  | Median | Min | Max  | Stdev |
|-----------|---------|--------|-------|--------|-----|------|-------|
| fixed     | legal   | 1716   | 311.8 | 313.0  | 2   | 512  | 184.1 |
| fixed     | medical | 582    | 218.0 | 160.0  | 1   | 512  | 165.5 |
| fixed     | policy  | 7351   | 384.7 | 460.0  | 1   | 512  | 151.5 |
| fixed     | **overall** | 9649 | 361.7 | 419.0 | 1 | 512 | 165.1 |
| sentence  | legal   | 1581   | 263.0 | 294.0  | 0*  | 526  | 146.2 |
| sentence  | medical | 552    | 196.8 | 146.5  | 1   | 476  | 144.4 |
| sentence  | policy  | 6759   | 329.3 | 354.0  | 0*  | 689  | 101.9 |
| sentence  | **overall** | 8892 | 309.3 | 337.0 | 0* | 689 | 120.2 |
| paragraph | legal   | 902    | 459.6 | 464.0  | 18  | 1016 | 146.3 |
| paragraph | medical | 267    | 406.9 | 392.0  | 23  | 2301 | 210.5 |
| paragraph | policy  | 3372   | 658.8 | 395.0  | 29  | 1802 | 442.6 |
| paragraph | **overall** | 4541 | 604.4 | 422.0 | 18 | 2301 | 401.2 |

\* `min=0` for `sentence` reflects word_count=2 footer-stamp artifacts rounding down in
display, not true empty chunks — all genuine empty chunks were eliminated by the bug fix
above (verified via `quick_debug_empty_chunks.py`).

## Data Quality Observations

Two distinct, recurring non-bug patterns showed up during verification, both worth
accounting for in any downstream retrieval evaluation:

1. **SEC filing footer stamps (legal domain).** PyMuPDF extracts isolated boilerplate
   like `'8-K, 1/10/2020'` or `'10-K/A, 5/5/2017'` as standalone 2-word "sentences,"
   since they sit alone on a line with no adjacent punctuation for spaCy to merge them
   with surrounding text. These are real, low-value chunks — a user's question will never
   semantically match a filing-type/date stamp, but it still occupies a slot in the
   vector store. Low priority (small chunks, easy for a vector search to simply not
   surface), but worth noting.

2. **TOC / exhibit-list pages (policy domain).** Pages that are lists of headings/exhibit
   titles with no terminal punctuation at all (e.g., `ZyPDF.pdf` page 3, a "List of
   Exhibits" spanning 38 entries) get treated by spaCy's sentencizer as **one single
   run-on "sentence"** covering the whole page, since there's no punctuation to split on.
   This produces oversized chunks (contributing to the high max/stdev seen in `paragraph`
   and `sentence` for policy) that blow past the intended `CHUNK_SIZE` ceiling. Unlike the
   footer-stamp issue, this could matter for retrieval: a TOC page is unlikely to be a
   useful answer to most user questions, but it's now a large chunk with a lot of surface
   area against which queries might spuriously match.

Neither of these needs a code fix to proceed with RQ1 — they're characteristics of the
real corpus and worth mentioning as limitations/observations in the final report, not
correctness bugs.

## Interpretation (Preliminary)

- **`fixed`** gives the most size-consistent chunks (as expected, since it's purely
  mechanical), which may help retrieval consistency but can split content mid-thought.
- **`paragraph`** shows the highest size variance by far (stdev up to 442.6 in policy,
  max 2301 words in medical) — this strategy is highly sensitive to document structure:
  it does reasonably on legal contracts (natural paragraph breaks) but poorly on dense
  regulatory text and single-block clinical notes.
- **`sentence`** sits between the two on size consistency, but is the strategy most
  affected by the TOC/exhibit-list edge case described above.

## Next Steps (Not Yet Done)

This document covers chunk *structure* only. To actually answer RQ1 (retrieval
*precision*), the following still needs to happen:

1. Embed each strategy's chunk output (`src/embeddings.py`)
2. Load into ChromaDB per strategy (`src/vector_store.py`)
3. Run a query set with known correct answers against each strategy's index
4. Measure recall/precision (or similar retrieval metrics) per strategy, per domain

This depends on the test question set (owned by Kat per the Milestone 2 proposal) and
the embedding/vector-store pipeline being functional end-to-end.
