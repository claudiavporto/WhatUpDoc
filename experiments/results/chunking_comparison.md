# Preliminary Experiment 1 — Chunking Strategy Comparison (2026-07-16)

Corpus: 3 sections from 3 sample documents (lease PDF, medical record DOCX, utility policy TXT).

| Strategy | Chunks | Mean chars | Min | Max | Stdev | Clean boundaries |
|---|---|---|---|---|---|---|
| fixed | 8 | 777.5 | 364 | 1000 | 287.8 | 37.5% |
| sentence | 5 | 1093 | 932 | 1185 | 91.8 | 100.0% |
| paragraph | 5 | 1047.4 | 288 | 1758 | 640.1 | 100.0% |

## Observations

- **Fixed** windows produce the most chunks but routinely cut mid-sentence (lowest clean-boundary rate), which fragments clauses like the late-fee provision across two chunks.
- **Sentence** packing keeps every boundary clean at a modest cost in chunk count, and is the default going into Milestone 4.
- **Paragraph** chunks best preserve contract clause structure but vary most in size, which may interact with top-k selection (RQ2). Full retrieval-precision measurement with real embeddings is scheduled for Milestone 4.