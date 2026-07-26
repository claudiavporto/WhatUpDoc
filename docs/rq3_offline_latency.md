# RQ3: Offline Operation and Latency Trade-offs

Owner: Claudia Porto (feature/performance-testing)

## Research Question

**RQ3: Can the pipeline run fully offline, and what are the latency trade-offs?**

## Offline Guarantee

WhatUpDoc's privacy claim rests on every stage of the pipeline running entirely
against local resources, with no data leaving the device. Verified per stage:

| Stage | Mechanism | Network exposure |
|---|---|---|
| Document parsing (`src/data_loader.py`) | PyMuPDF / python-docx, local file I/O only | None — no network code path exists |
| Chunking (`src/chunking.py`) | Pure in-process text processing | None |
| Embedding (`src/embeddings.py`) | HTTP request to Ollama's `/api/embeddings` endpoint | Local only, enforced (see below) |
| Vector store (`src/vector_store.py`) | ChromaDB `PersistentClient`, file-based local storage | None — no network code path exists |
| Generation (`src/llm.py`) | HTTP request to Ollama's `/api/generate` endpoint | Local only, enforced — same mechanism as embedding (see below) |

### Enforcement mechanism

Both `OllamaEmbedder.__init__` (`src/embeddings.py`) and `OllamaLLM.__init__`
(`src/llm.py`) call the identical `assert_local_host(self.host,
cfg["privacy"]["allowed_hosts"])` before any HTTP request is made, whenever
`cfg["privacy"]["enforce_offline"]` is true (the default in
`configs/config.yaml`). This is one shared enforcement pattern applied
consistently to both network-touching stages, not two independent
implementations that could drift out of sync — both stages fail closed at
construction time if a non-local host is configured, rather than allowing any
request to be attempted.

**Verified behavior — both directions confirmed:**

- With `configs/config.yaml`'s default `ollama.host: "http://localhost:11434"`
  and `privacy.allowed_hosts: ["localhost", "127.0.0.1"]`, `OllamaEmbedder()`
  constructs successfully and all embedding calls in the RQ1 evaluation run
  (`experiments/04_precision_recall_eval.py`) resolved against `localhost`
  only.
- **Rejection path actually executed and confirmed**, not just read from
  source: pointing `OllamaEmbedder` at a non-local host raises immediately,
  before any request is attempted:

  ```
  PASS: correctly rejected non-local host: Privacy guard: refusing to
  connect to non-local host 'example.com'. Allowed hosts: ['localhost',
  '127.0.0.1']. Check configs/config.yaml.
  ```

  Since `OllamaLLM` calls the identical `assert_local_host` function, this
  single test covers both the embedding and generation clients.

### Open items

- `OllamaLLM` falls back from the primary model (LLaMA 3 8B) to a secondary
  (Mistral 7B) on an HTTP 404 (model not pulled). This fallback path still
  goes through the same local-host-only client, so it doesn't weaken the
  offline guarantee, but it does mean a report claiming "LLaMA 3 8B" answers
  should confirm which model actually served each response if reproducibility
  matters (the fallback logs a warning when triggered).

## Latency

### Measured embedding throughput

All measurements below are from `nomic-embed-text` served locally via Ollama on
CPU (no GPU acceleration configured), observed during the smoke test and the
RQ1 precision/recall evaluation run:

| Source document | Pages | Strategy | Chunks | Wall time | Sec/chunk |
|---|---|---|---|---|---|
| `00-3.pdf` | 66 | paragraph | 352 | ~13.0 min | ~2.22 |
| `00-3.pdf` | 66 | fixed | 640 | ~26.6 min | ~2.49 |
| `2012-31205.pdf` | 96 | fixed | 834 | (in progress) | ~2.0–2.4 (partial) |

Throughput is **consistent across documents and strategies at roughly 2.2–2.5
seconds per chunk**, regardless of chunk size (chunk text lengths in this
sample ranged from ~350 to over 1000 characters). This stability suggests
per-request overhead (HTTP round-trip + model invocation setup) dominates over
the actual token-count-dependent compute time for chunks in this size range —
a hypothesis based on these measurements, not independently isolated or
confirmed.

### Full-corpus extrapolation

The full corpus (86 policy + legal PDFs, 200 medical DOCX/TXT files) produced
the following approximate chunk counts under a *different* chunker
(`utils/doc_ingestion.py`, token-based, 512/64) during the preliminary
structural analysis (see `docs/rq1_chunking_findings.md`):

| Strategy | Chunks (approx., preliminary chunker) |
|---|---|
| fixed | 9,649 |
| sentence | 8,892 |
| paragraph | 4,541 |

The live pipeline (`src/chunking.py`, character-based, 1000/1200/1800) will
produce different exact counts, but at a similar order of magnitude. At
~2.2–2.5 sec/chunk, embedding the full corpus once under one strategy would
take **roughly 3–7 hours depending on strategy**, and all three strategies
sequentially (as required for a full RQ1 comparison) would take **on the order
of 15+ hours of embedding time alone** — before any retrieval evaluation. This
is the direct, load-bearing answer to RQ3's "latency trade-offs" question: the
practical bottleneck for this pipeline at its current scale is embedding
throughput on CPU-only local hardware, not retrieval or generation latency.

### Retrieval and generation latency

Retrieval query latency (embedding one query + ChromaDB similarity search) and
generation latency (LLaMA 3 8B response time) were not separately measured as
part of this evaluation. Both are expected to be small relative to the
corpus-embedding bottleneck above (single-query embedding is one API call, not
thousands), but this should be measured directly rather than assumed for a
complete RQ3 answer.

## Summary

- **Offline: yes, by construction and now confirmed at runtime.** Parsing,
  chunking, and vector storage have no network code path at all. Both
  embedding (`src/embeddings.py`) and generation (`src/llm.py`) enforce a
  local-host check via the same shared `assert_local_host` call before any
  request is made — and that rejection path has been executed and confirmed
  to actually raise, not just assumed from reading the source.
- **Latency: the real cost is embedding, not inference.** Local CPU-based
  embedding throughput (~2.2–2.5 sec/chunk) makes full-corpus, multi-strategy
  evaluation a multi-hour undertaking, which directly shaped the evaluation
  scope for RQ1 in this milestone (see `docs/rq1_metrics_definition.md`,
  "Known Limitations").