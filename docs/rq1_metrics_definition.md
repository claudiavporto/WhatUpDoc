# RQ1 Evaluation Metrics Definition

Owner: Claudia Porto (feature/performance-testing)

## Research Question

**RQ1: Does chunking strategy affect retrieval precision on legal and medical documents?**

This document defines the metrics used to answer RQ1, what counts as a "correct"
retrieval, and why. It is the companion specification for
`experiments/04_precision_recall_eval.py`, which implements it, and
`experiments/results/rq1_summary.csv` / `rq1_detail_{strategy}.csv`, which report
the results.

## What Counts as a Correct Retrieval

A retrieval is scored **correct at the source-document level**: given a query with
a known-correct source file (from `data/eval/final_query_set.csv`), a hit is counted
if any chunk in the top-k results returned by the vector store has a `source` field
matching that file.

This is **not** an exact page/location match. That is a deliberate choice, not an
oversight:

- The evaluation query set's `location` field was assigned by
  `experiments/build_query_candidates.py`, which batches `.docx`/`.txt` medical notes
  into groups of 15 paragraphs to derive a "location" number.
- The live retrieval pipeline (`src/data_loader.py`) uses a *different* scheme:
  `.txt` files are loaded as a single Page (page_number=1) in their entirety, and
  `src/chunking.py` then subdivides that one Page's text into multiple chunks per
  strategy, each still carrying `page_number=1`.
- These two numbering schemes were built independently and are not reliably
  comparable chunk-for-chunk, particularly for `.txt` medical notes. Requiring an
  exact location match would produce false negatives caused by a bookkeeping
  mismatch between two scripts, not by the chunking strategy actually failing to
  retrieve the right content.

Source-document-level precision is the standard, defensible unit for this kind of
retrieval evaluation: it answers "did the system find the right document to answer
this question," which is the practically meaningful question for a RAG system's
retrieval stage, independent of exactly which chunk within that document was
returned.

## Metrics

For each chunking strategy, computed overall and broken out by category
(policy / legal / medical):

### Hit@k

The fraction of queries where the correct source document appears **anywhere**
in the top-k retrieved chunks (k is configurable via `--top-k`, default 4 —
matches `configs/config.yaml`'s production `retrieval.top_k`).

```
Hit@k = (number of queries with a correct-source hit in top k) / (total queries)
```

Range: 0.0 (never found) to 1.0 (always found). This is the primary headline
number for comparing strategies.

### Mean Reciprocal Rank (MRR)

For each query, the reciprocal of the rank position of the first correct-source
hit (1.0 if the correct source is the top result, 0.5 if it's second, 0.33 if
third, etc.), or 0 if no correct-source hit appears in the top k. MRR is the
average of this value across all queries.

```
MRR = (1/N) * sum(1/rank_i if hit else 0, for each query i)
```

MRR captures ranking quality that Hit@k alone misses: two strategies can have
identical Hit@5, but one might consistently rank the correct source 1st while
the other buries it at 5th. Hit@k would call these equivalent; MRR would not.

## Per-Query Detail

`experiments/results/rq1_detail_{strategy}.csv` records, for every query:
`category`, whether it was a hit, the rank of the first correct hit (blank if
none), the reciprocal rank, and the top-ranked result's actual source and
distance score. This is kept alongside the summary numbers specifically so that
individual misses can be inspected and explained in the final report, rather
than only reporting an aggregate score.

## Known Limitations

- **Sample scale.** Given embedding throughput constraints (~2.2–2.5 seconds
  per chunk observed locally with `nomic-embed-text` via Ollama on CPU), the
  results reported in this milestone reflect a **representative sample of 3
  files per category (9 files total: `00-3.pdf`, `2012-31205.pdf`,
  `2014-30382.pdf` for policy; the first 3 alphabetically for legal and
  medical)**, not the full 286-document corpus. This run took approximately
  5 hours end to end across all three strategies. A full-corpus run is
  estimated at 15+ hours of embedding time alone and is scheduled for
  Milestone 4, consistent with the scope already noted in
  `experiments/01_chunking_comparison.py` and
  `experiments/02_retrieval_smoke_test.py`.
- **Effective query coverage was much smaller than 30 per strategy.** Because
  only 3 files per category were embedded, most of the 30 queries in
  `final_query_set.csv` target documents that were never in the index for
  this run — those necessarily register as misses, not because retrieval
  failed, but because their target document was out of scope by construction.
  Only **7 of the 30 queries** (4 policy, 1 legal, 2 medical) had their
  target document actually embedded. See Results below for the corrected
  reading of this run's numbers.
- **Distance metric, not a calibrated relevance score.** ChromaDB's returned
  `distance` field (cosine distance) is reported per query for inspection but
  is not itself used as a pass/fail threshold; only rank position among the
  top-k results determines Hit@k and MRR.
- **Query set size.** 30 queries (10 per category), curated from an
  auto-generated candidate pool and manually reviewed for correctness. This is
  adequate to detect large differences between strategies but is a small
  sample for detecting subtle ones; per-category breakdowns in particular
  (10 queries each) should be read as indicative rather than statistically
  definitive.

## Results (Sample-Scale Run, 3 Files/Category)

Raw output (`experiments/results/rq1_summary.csv`):

| Strategy | Category | N | Hit@4 | MRR |
|---|---|---|---|---|
| fixed | ALL | 30 | 0.233 | 0.217 |
| fixed | legal | 10 | 0.1 | 0.1 |
| fixed | policy | 10 | 0.4 | 0.4 |
| fixed | medical | 10 | 0.2 | 0.15 |
| sentence | ALL | 30 | 0.233 | 0.211 |
| sentence | legal | 10 | 0.1 | 0.1 |
| sentence | policy | 10 | 0.4 | 0.4 |
| sentence | medical | 10 | 0.2 | 0.133 |
| paragraph | ALL | 30 | 0.233 | 0.233 |
| paragraph | legal | 10 | 0.1 | 0.1 |
| paragraph | policy | 10 | 0.4 | 0.4 |
| paragraph | medical | 10 | 0.2 | 0.2 |

**Read in raw form, these numbers understate retrieval quality**, since 23 of
30 queries were scored against documents that were never embedded in this
run (see the coverage limitation above). Correcting for the 7 queries that
actually had their target document in scope:

- **7 of 7 in-scope queries hit successfully, across all three strategies.**
  When the correct document was actually in the index, retrieval found it
  every time, regardless of chunking strategy, at this sample size.
- **The three strategies differ in ranking quality (MRR), not hit rate.**
  `paragraph` (0.233) ranked correct hits marginally higher on average than
  `fixed` (0.217) and `sentence` (0.211). This is directionally consistent
  with the structural findings in `docs/rq1_chunking_findings.md`, which
  predicted paragraph chunking would better preserve topical/clause
  coherence, but with only 7 effective data points this difference is
  suggestive, not statistically conclusive.
- **No strategy failed outright at this scale.** All three retrieved every
  in-scope target document; the open question a full-corpus run would answer
  is whether this holds once the vector store contains thousands of
  competing chunks rather than a few thousand, where retrieval precision has
  more opportunity to diverge between strategies.