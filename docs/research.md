# Research and Selection of Methods

**WhatUpDoc — IE 7374 Group 03** | Milestone 3
Claudia Porto, Christopher Swartz, Sean Costello

This document records the research, benchmarking, and preliminary experiments behind the technical selections in `configs/config.yaml`. It satisfies the "Research and Selection of Methods" component of Milestone 3.

---

## 1. Objectives

WhatUpDoc is a locally-hosted, privacy-first RAG application built for professionals who work with sensitive documents — use cases where uploading files to a cloud-based AI service is not an option due to regulatory requirements (HIPAA, NDAs) or organizational policy. All processing happens on the user's machine with zero external network calls.

The task is **retrieval-augmented question answering over private documents**, decomposed into four sub-tasks:

1. **Document parsing** — extract text from PDF, DOCX, and TXT while preserving location metadata (page numbers, or paragraph-batch numbers for unpaginated formats) for citation.
2. **Chunking and embedding** — split documents into retrieval units and encode them as dense vectors, entirely on-device.
3. **Retrieval** — return the most relevant chunks for a natural-language question via vector similarity.
4. **Grounded generation** — produce an answer from a locally hosted LLM that (a) uses only the retrieved context, (b) cites its sources, and (c) refuses rather than guesses when the answer is absent.

The binding constraint that differentiates this project from typical RAG builds: **no network egress**. Every component must run offline on consumer hardware (target: 16 GB RAM, no dedicated GPU required).

The target document types — legal contracts, medical records, and public policy documents — were chosen because they represent the highest-stakes privacy scenarios and the most demanding retrieval challenges in terms of vocabulary complexity and document length, giving RQ1's chunking-strategy comparison a genuinely hard, realistic testbed rather than a toy corpus.

## 2. Literature Review

**RAG foundations.** Lewis et al. (2020) introduced retrieval-augmented generation, showing that conditioning a generator on retrieved passages improves factuality on knowledge-intensive tasks relative to closed-book generation. The core architecture — dense retriever + sequence generator — remains the template for modern document Q&A. Subsequent practice has largely replaced end-to-end trained retrievers with off-the-shelf embedding models and approximate nearest-neighbor stores, which is the pattern we adopt: for a course-scale project, a frozen pretrained embedder plus a frozen pretrained generator is both feasible and well-supported.

**Grounding and hallucination control.** Instruction-following LLMs will substitute parametric knowledge for context when retrieval fails, producing confident but unsupported answers. Practical mitigations from the RAG deployment literature that we adopt: low decoding temperature, explicit "answer only from context" system instructions, a mandated refusal string for unanswerable questions, and inline source citation so users can verify claims. Our prompt design (see `src/llm.py`) implements all four, and our demo question set deliberately includes an unanswerable question to test refusal behavior.

**Domain benchmark.** Hendrycks et al. (2021) released CUAD, an expert-annotated dataset of 510 legal contracts with clause-level labels. CUAD demonstrates both that contract review is a high-value retrieval target and that clause-finding is hard for generic models — motivating our RQ1 focus on chunking strategies that respect clause boundaries. CUAD contracts (public, real, professionally annotated) are a candidate evaluation corpus for Milestones 4–5, complementing our synthetic samples.

**Grounding verification.** A recurring finding in the RAG literature is that retrieval alone does not guarantee faithful answers: instruction-tuned models will still cite sources they were not given, or answer from parametric memory when retrieval is weak. We therefore treat citation faithfulness as a measurable property rather than an assumption. After generation, every `[source, page N]` tag the model emits is checked against the set of chunks actually retrieved; citations with no matching chunk are flagged as fabricated and reduce a per-answer *grounding score* (fraction of citations that are supported). This converts the project's central claim — "answers come only from your documents" — into a number we can report and regression-test. See `src/grounding.py` and Experiment 3.

**Local inference.** The Ollama runtime packages quantized GGUF model weights behind a stable localhost HTTP API, making swap-in/swap-out comparison of 7–8B models straightforward. Quantized 7–8B instruction-tuned models are the established sweet spot for CPU/consumer-GPU inference: small enough to fit in 8–16 GB of RAM at 4-bit quantization, large enough to follow multi-rule system prompts reliably — which our citation-and-refusal prompt requires.

## 3. Benchmarking of Candidate Methods

### 3.1 Generator LLM

| Criterion | **LLaMA 3 8B Instruct (selected)** | Mistral 7B Instruct (fallback) | Phi-3 Mini 3.8B |
|---|---|---|---|
| Instruction following on multi-rule prompts | Strong | Good | Adequate |
| Approx. RAM at Q4 quantization | ~6 GB | ~5 GB | ~2.5 GB |
| Context window (Ollama default) | 8k tokens | 32k (sliding window) | 4k (128k variant exists) |
| License for coursework use | Llama Community License — permitted | Apache 2.0 | MIT |
| Ollama availability | `llama3:8b` | `mistral:7b` | `phi3:mini` |

**Decision:** LLaMA 3 8B primary, Mistral 7B automatic fallback (implemented in `src/llm.py`). Rationale: the strict citation/refusal prompt is a multi-rule instruction-following task, where 8B-class models are markedly more reliable than 4B-class; 8k context comfortably holds top-k=4 chunks of ~1.2k characters plus the question. Phi-3 Mini is retained as a low-resource option to test in Milestone 4 if teammate hardware requires it.

### 3.2 Embedding model

| Criterion | **nomic-embed-text (selected)** | all-MiniLM-L6-v2 | mxbai-embed-large |
|---|---|---|---|
| Dimensions | 768 | 384 | 1024 |
| Effective input length | long (8k-token training) | short (~256 tokens) | moderate |
| Retrieval quality on long-form text | Strong | Weaker on >1 paragraph | Strong |
| Serving path | Ollama (same runtime as LLM) | needs sentence-transformers stack | Ollama |
| Footprint | ~270 MB | ~90 MB | ~670 MB |

**Decision:** `nomic-embed-text`. Its long-input training matters because our chunks run ~1,000–1,200 characters, beyond MiniLM's effective window; serving it from the same Ollama runtime as the generator means one dependency, one privacy boundary, and no PyTorch install for end users. mxbai-embed-large is the upgrade path if Milestone 4 retrieval precision is insufficient.

### 3.3 Vector store

| Criterion | **ChromaDB (selected)** | FAISS | Qdrant |
|---|---|---|---|
| Deployment model | Embedded, in-process | Library only (no metadata store) | Separate server process |
| Metadata filtering / citation payloads | Built in | Must build ourselves | Built in |
| Local persistence | One-line `PersistentClient` | Manual index serialization | Volume config |
| Setup burden on graders/teammates | `pip install chromadb` | Low, but more glue code | Docker required |

**Decision:** ChromaDB. Cosine-distance HNSW retrieval with built-in metadata storage covers our needs at course scale (hundreds–thousands of chunks); an embedded database keeps the "clone → pip install → run" path short. One caution discovered during research: **Chroma's default embedding function downloads a model from the internet**, which would silently violate the privacy guarantee — so `src/vector_store.py` accepts only caller-supplied embeddings and never invokes Chroma's default embedder.

### 3.4 Document parsing and chunking

- **PDF:** PyMuPDF selected over pypdf/pdfminer for markedly better text-order fidelity on multi-column and table-heavy layouts, plus native page numbers for citation.
- **DOCX:** python-docx; paragraphs are grouped into numbered sections to give citations a stable location reference in an unpaginated format.
- **TXT:** loaded as plain text with UTF-8 decoding (latin-1 fallback for non-standard encodings); needed specifically for the medical corpus, roughly half of which (100 of 200 files) was sourced as `.txt` rather than `.docx`.
- **Chunking:** three strategies implemented for RQ1 — fixed windows with overlap (baseline common in RAG tutorials), sentence-boundary packing (spaCy when available, regex fallback), and paragraph-boundary chunks (motivated by CUAD's clause-level structure). See §4 for measured comparison.

### 3.5 UI

Gradio selected over Streamlit for its built-in `ChatInterface` and file-upload components; bound to `127.0.0.1` with `share=False` so the interface itself cannot be exposed off-machine.

## 4. Preliminary Experiments

Two small-scale experiments validate feasibility before full-scale implementation. Both run **fully offline with no Ollama dependency**, so any grader or teammate can reproduce them immediately after cloning.

### Experiment 1 — Chunking strategy comparison (`experiments/01_chunking_comparison.py`)

Corpus: three synthetic sample documents (lease PDF, medical-record DOCX, utility-policy TXT; see `data/make_samples.py`). Measured results:

| Strategy | Chunks | Mean chars | Stdev | Clean sentence boundaries |
|---|---|---|---|---|
| fixed (1000/200 overlap) | 8 | 777.5 | 287.8 | **37.5%** |
| sentence (≤1200) | 5 | 1093.0 | 91.8 | **100%** |
| paragraph (≤1800) | 5 | 1047.4 | 640.1 | **100%** |

**Findings:** fixed windows cut mid-sentence in 5 of 8 chunks — on the lease, the late-fee amount was separated from the clause defining it, exactly the failure mode that degrades clause retrieval. Sentence packing achieved clean boundaries with the most uniform chunk sizes. **Adjustment made:** default strategy changed from `fixed` (original plan) to `sentence` in `config.yaml` (confirmed current).

**Update — real RQ1 precision results now exist.** `experiments/04_precision_recall_eval.py` has since measured actual retrieval precision (Hit@k, MRR) across all three strategies with real embeddings against the 30-query evaluation set. See `docs/rq1_metrics_definition.md` for full results and methodology; summary: at a 9-file sample scale, all three strategies retrieved every in-scope query's correct source document (7/7), with `paragraph` showing a marginal MRR edge over `fixed` and `sentence`. Full-corpus evaluation remains scheduled for Milestone 4.

### Experiment 2 — Retrieval smoke test (`experiments/02_retrieval_smoke_test.py`)

Validates the storage/retrieval plumbing using a deterministic hash-based mock embedding (test double for nomic-embed-text). All assertions pass: 5/5 chunks indexed and retrievable; domain-routing queries about rent, allergies, and backflow testing each retrieved the correct source document first; source/page metadata survives the round trip, confirming the citation path from vector store to prompt.

**Limitation noted:** the mock embedding captures only lexical overlap; it validates plumbing, not semantic retrieval quality. Semantic evaluation with real embeddings is Milestone 4 work (see `experiments/04_precision_recall_eval.py`).

### Experiment 3 — Prompt-profile ablation and grounding evaluation (`experiments/03_prompt_ablation.py`)

Measures the behavior the whole project depends on: does the prompt keep the model grounded? A labeled question set of nine questions over the sample corpus — six answerable, three deliberately unanswerable (patient blood type, landlord phone number, authority budget; none present in the documents) — is run through each prompt profile and scored on three metrics: **refusal accuracy** on unanswerable questions, **citation rate** on answerable ones, and **mean grounding score** (fraction of emitted citations that map to a retrieved chunk).

The harness produces the report-ready numbers reported in Milestones 4–5 when run against a live local model (`python experiments/03_prompt_ablation.py`). Its `--selftest` mode validates the grounding verifier itself on fixed answer fixtures with **no Ollama dependency**, and passes 4/4 assertions here: a fully grounded answer scores 1.0, a fabricated citation is caught and scores 0.0, a correct refusal is recognized rather than scored as a claim, and a mixed answer scores 0.5 with the one fabricated citation flagged. This makes the *scoring logic* reproducible on any machine even before models are pulled, mirroring Experiment 2's offline approach.

**Design note:** the strict-cited profile is expected to show near-100% refusal accuracy and high grounding scores, while a hypothetical no-guardrail baseline should refuse ~0% of unanswerable questions — the ablation quantifies exactly how much the prompt engineering buys, which is the core evidence for RQ2.

**Update — live ablation run completed.** Both prompt profiles achieved perfect refusal accuracy (1.0/1.0) against the live model — the safety-critical behavior works as designed. Grounding scores came back at 0.0 for both profiles, which on inspection appears to be a citation-format mismatch (the model echoing the excerpt header's `"source: X"` phrasing rather than the prompt's instructed `[filename, page N]` format) rather than genuine fabrication, though this requires direct confirmation against raw answer text. Full results, methodology, and the format-mismatch analysis are in `docs/rq2_prompt_ablation.md`.

## 5. Selected Architecture (summary)

```
PDF/DOCX/TXT ─→ PyMuPDF / python-docx ─→ sentence chunking (spaCy)
        ─→ nomic-embed-text (Ollama) ─→ ChromaDB (cosine, local persist)

Question ─→ nomic-embed-text ─→ top-k=4 retrieval
        ─→ token-budget context packing ─→ strict-cited prompt
        ─→ LLaMA 3 8B (Ollama) ─→ citation faithfulness check ─→ scored answer
```

All parameters live in `configs/config.yaml`; the privacy guard in `utils/helpers.py` refuses any non-localhost endpoint before the first network call, applied consistently to both the embedding client (`src/embeddings.py`) and the generation client (`src/llm.py`).

## 6. References

- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems 33*, 9459–9474.
- Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. *arXiv:2103.06268*.
- Nussbaum, Z., Morris, J. X., Duderstadt, B., & Mulyar, A. (2024). Nomic Embed: Training a reproducible long context text embedder. *arXiv:2402.01613*.
- Ollama documentation — https://github.com/ollama/ollama
- ChromaDB documentation — https://docs.trychroma.com