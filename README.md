# WhatUpDoc

**A Privacy-First Local RAG Application**
IE 7374 – Group 03 | Claudia Porto, Christopher Swartz, Sean Costello

## Overview

WhatUpDoc is a fully offline Retrieval-Augmented Generation (RAG) application that allows users to securely query sensitive documents without sending any data to external servers. The system is designed for professionals who need the analytical power of generative AI but operate under strict data governance requirements, including HIPAA-regulated medical records, NDA-covered legal contracts, and confidential proprietary documents.

Unlike commercial tools such as ChatGPT Enterprise or Adobe AI Assistant, WhatUpDoc runs entirely on the user's local machine. Documents are parsed, embedded, and stored locally. Queries are answered by a locally hosted LLM. No data ever leaves the device.

## The Problem

Professionals handling sensitive information are currently unable to leverage modern AI tools due to the risk of data leakage, intellectual property theft, and regulatory violations associated with cloud-based processing. Uploading documents containing PII or PHI to public AI services directly violates laws like HIPAA and standard NDAs.

WhatUpDoc addresses this gap by combining semantic search, source-grounded generation, and strict data containment in a single local pipeline.

## Stack

- **LLM:** LLaMA 3 8B via Ollama (Mistral 7B as fallback)
- **Embeddings:** nomic-embed-text via Ollama (768-dimensional)
- **Vector store:** ChromaDB (local persistent storage)
- **Document parsing:** PyMuPDF, python-docx, plain-text (PDF, DOCX, and TXT corpora)
- **Chunking:** fixed-size, sentence-boundary (spaCy), paragraph-boundary
- **UI:** Gradio

Note: the rubric's suggested structure names a `models/` folder; this project's model implementation (data loading, chunking, embeddings, vector store, LLM client, grounding) lives in `src/`, the standard Python package convention, since these are pipeline components rather than trained model artifacts. `models/` is retained for lightweight sample fixtures used by offline experiments.

## Research Questions

| | |
|---|---|
| RQ1 | Does chunking strategy affect retrieval precision on legal and medical documents? |
| RQ2 | How does top-k context size affect LLaMA 3 response accuracy? |
| RQ3 | Can the pipeline run fully offline, and what are the latency trade-offs? |

## Privacy Guarantee

WhatUpDoc is designed so that no document content, query text, or generated answer ever leaves your machine.

Document parsing, chunking, and vector storage are 100% local. PDF/DOCX/TXT parsing (PyMuPDF, python-docx), chunking, and ChromaDB's persistent store all operate on local files and local disk — none of these stages contain a network code path at all.

Embedding and generation are enforced local-only, not just configured local-only. Both the embedding client (`src/embeddings.py`) and the generation client (`src/llm.py`) call a shared `assert_local_host()` guard before making any request. If `configs/config.yaml`'s `ollama.host` is ever set to anything other than `localhost`/`127.0.0.1`, the client refuses to construct and the pipeline stops rather than silently sending data off-machine.

ChromaDB's default embedding function is deliberately never used, since it downloads a model from the internet on first use. `src/vector_store.py` only accepts embeddings supplied by the caller (our local Ollama-served `nomic-embed-text`), so no implicit network call can occur through the vector store either.

The UI is bound to localhost. The Gradio interface runs with `share=False` on `127.0.0.1`, so the app itself is not exposed off-machine.

You can verify the local-host enforcement yourself (this has been run and confirmed to correctly raise a `RuntimeError` for a non-local host):

```powershell
python -c "
from src.config import get_config
from src.embeddings import OllamaEmbedder
cfg = dict(get_config())
cfg['ollama'] = dict(cfg['ollama'])
cfg['ollama']['host'] = 'http://example.com:11434'
try:
    OllamaEmbedder(cfg)
    print('FAIL: no exception raised for a non-local host')
except RuntimeError as e:
    print(f'PASS: correctly rejected non-local host: {e}')
"
```

See `docs/rq3_offline_latency.md` for the full analysis of offline enforcement and its latency trade-offs.

## Setup

### Prerequisites

- Conda (Miniconda or Anaconda)
- Ollama — the local LLM/embedding server

### 1. Clone and create the environment

```powershell
git clone https://github.com/claudiavporto/WhatUpDoc.git
cd WhatUpDoc
conda env create -f environment.yml
conda activate whatupdoc
python -m spacy download en_core_web_sm
```

### 2. Install and start Ollama

Install from [ollama.com/download](https://ollama.com/download), then pull the models this project uses:

```powershell
ollama pull nomic-embed-text
ollama pull llama3:8b
ollama pull mistral:7b
```

Confirm Ollama is running:

```powershell
curl -UseBasicParsing http://localhost:11434
```

You should see `Ollama is running`. If not, run `ollama serve` in a separate terminal and leave it open.

### 3. Verify the pipeline runs end to end

```powershell
python experiments/tooling/smoke_test_pipeline.py --sample-file data/corpus/policy/00-3.pdf
```

This runs a real document through parse → chunk → embed → store → query and reports which stage(s) succeed. A full pass confirms your environment is set up correctly before you run any of the actual experiments.

### 4. Run the demo pipeline

```powershell
python src/model_runner.py
```

This ingests two sources and runs 13 representative questions spanning both, through the full grounded RAG pipeline:

- **`data/raw/`** — three small, hand-authored sample fixtures (`sample_lease_agreement.pdf`, `sample_medical_record.docx`, `sample_utility_policy.txt`), one per domain. These are ingested in full (no cap) and every answer can be checked by eye against the source file in seconds.
- **`data/corpus/legal/`, `data/corpus/medical/`, and `data/corpus/policy/`** — a capped sample of the larger real-world corpus, so the demo also exercises retrieval at realistic document lengths and chunk counts.

Each source contributes 6 domain questions (2 per domain: legal, medical, policy), plus one shared question ("What is the CEO's salary?") that is deliberately unanswerable from either source, to confirm the refusal path works rather than the model guessing. The script prints each answer with its grounding score and saves everything to `outputs/samples.txt`.

> **Note on question count:** Milestone 4's spec calls for a demo of "5–10 samples." This run currently executes 13 (6 `data/raw` questions + 6 `data/corpus` questions + 1 shared refusal check), because both the original small-fixture question set and the larger-corpus question set were kept intentionally rather than choosing one. If 5–10 is a hard requirement rather than a guideline, trim `DEMO_QUESTIONS` in `src/model_runner.py` down to one source or fewer questions per domain before submitting.

Ingestion of `data/corpus/` is capped at `MAX_CHUNKS_PER_DOMAIN` (150 chunks per domain, defined in `src/model_runner.py`) rather than embedding the entire corpus — a full run of `legal/` alone produced 3,516 chunks, which at the measured ~2.5 sec/chunk local embedding rate takes ~2.5 hours for that one domain (see `docs/rq3_offline_latency.md`). The cap keeps this a fast, single-command demo; exhaustive full-corpus embedding across all three chunking strategies is handled separately by `experiments/04_precision_recall_eval.py`. Expect roughly 20–25 minutes for this step, including generation across all 13 questions.

### 5. Run the experiments

```powershell
# RQ1: chunk-structure comparison (fast, no Ollama needed)
python experiments/01_chunking_comparison.py

# RQ1: retrieval plumbing smoke test (fast, no Ollama needed)
python experiments/02_retrieval_smoke_test.py

# RQ1: real retrieval precision/recall across chunking strategies (slow — see note below)
python experiments/04_precision_recall_eval.py

# RQ2: prompt-profile ablation and grounding evaluation
python experiments/03_prompt_ablation.py --selftest   # offline check, no Ollama
python experiments/03_prompt_ablation.py              # full run, needs Ollama

# RQ2: top-k context-size sweep at a fixed prompt profile (needs Ollama)
python experiments/03_prompt_ablation.py --topk-sweep 1,2,4,6
```

Note on runtime: `04_precision_recall_eval.py` embeds the full document corpus from scratch for each chunking strategy. Measured throughput on local CPU inference is roughly 2.2–2.5 seconds per chunk; a full-corpus, all-strategy run can take several hours. Use `--max-files-per-category N` to run a fast sanity check on a handful of files first.

### 6. Launch the app

```powershell
python app.py
```

## Running the Demo Pipeline

See Setup step 4 above for the command. To summarize what it produces:

- **Input:**
  - `data/raw/sample_lease_agreement.pdf`, `data/raw/sample_medical_record.docx`, `data/raw/sample_utility_policy.txt` — ingested in full.
  - PDF/DOCX/TXT files under `data/corpus/legal/`, `data/corpus/medical/`, and `data/corpus/policy/`, processed in sorted order until each domain reaches its `MAX_CHUNKS_PER_DOMAIN` cap (150 chunks) or runs out of files — not the full corpus, by design (see Setup step 4 above).
- **Output:** `outputs/samples.txt`, containing 13 question/answer pairs, each with its retrieved sources (source, page, distance) and a grounding summary (e.g. `grounding: 100% of citations supported`, or `grounding: refusal (no answer claimed)` for the deliberately unanswerable question).
- **Reproducing a specific answer:** every citation in the output is traceable to a specific `(source, page)` pair from the retrieved chunks printed alongside it, so any answer can be manually checked against the source document.

## Preliminary Results

<!-- TODO: replace this section with 2–3 representative Q&A pairs copied
     directly from an actual run of outputs/samples.txt, e.g.:

Q: What is the monthly rent and when is it due?
A: Rent is $1,850/month, due on the 1st [sample_lease_agreement.pdf, page 1].
   [grounding: 100% of citations supported]

Q: What is the CEO's salary?
A: I can't find that in the provided documents.
   [grounding: refusal (no answer claimed)]

Then summarize across the full run, e.g.:
"Across N chunks ingested from data/raw and the legal, medical, and
policy corpora under data/corpus, grounded answers to in-corpus
questions consistently cite real retrieved sources (X/13 questions
scored 100% grounding), and the deliberately unanswerable question
correctly triggers the refusal path rather than a fabricated answer." -->

*(To be filled in after running `python src/model_runner.py` — see `outputs/samples.txt` for the full run and `docs/` for the RQ1–RQ3 experiment results.)*

## Known Issues & Limitations

- **Citation format sensitivity (resolved).** An earlier live run scored 0.0 grounding on both prompt profiles. Investigation showed the model was citing correctly-retrieved sources in nonstandard formats the parser didn't recognize as equivalent to the instructed `[source, page N]` form — e.g. echoing the excerpt-header phrasing (`[source: file.pdf, page 3]`) or citing by excerpt position (`[Excerpt 2]`) instead of by name. This conflated *format* mismatches with genuine *fabrications*. `src/grounding.py`'s `resolve_citations()` now resolves these nonstandard-but-valid formats and reports them separately (see `GroundingReport.nonstandard` / `format_compliance`), while the system prompt (`src/llm.py`) was updated with an explicit correct/incorrect example pair to reduce how often the model produces them in the first place.
- **Fabricated citations to un-ingested files are not always caught.** In one run against an earlier, `data/corpus`-only version of `model_runner.py`, the model produced fluent, specific-sounding answers (a rent figure, a named patient's medications, a testing interval) that matched what would be in `data/raw`'s sample files almost exactly — but those files had not actually been ingested in that run. The grounding checker scored several of these "100% of citations supported" despite the cited source never being loaded into the vector store. Restoring `data/raw/` ingestion in `model_runner.py` addresses the immediate case (the cited files are now actually ingested), but the underlying gap — the grounding checker not verifying that a cited source file was part of the current run's ingested set — has not been fixed and should be treated as a known limitation of `src/grounding.py`, not just a data-loading fix.
- **Token budget is approximate.** `src/grounding.py`'s `estimate_tokens()` uses a chars-per-token heuristic (~4) rather than the actual LLaMA/Mistral SentencePiece tokenizer, since we don't ship it. This intentionally over-estimates to avoid silent context truncation, at the cost of sometimes packing slightly fewer chunks than the true budget would allow.
- **DOCX pagination is approximate.** DOCX has no fixed page concept, so `src/data_loader.py` groups paragraphs into ~15-paragraph sections and cites those as "pages." Citations for DOCX sources are therefore section-accurate, not page-accurate in the PDF sense.
- **Re-ingestion re-embeds everything.** `RAGPipeline.ingest()` upserts by deterministic `chunk_id`, so re-running `model_runner.py` won't duplicate data, but it also has no "skip if already ingested" check — every run re-embeds its chunks from scratch through Ollama, regardless of prior runs.
- **The demo run samples `data/corpus`, not all of it — but ingests `data/raw` in full.** `src/model_runner.py` caps `data/corpus` ingestion at `MAX_CHUNKS_PER_DOMAIN` (150 chunks) per domain to keep the demo fast, while `data/raw`'s three small fixtures are always ingested completely (see Setup step 4). Preliminary Results below therefore reflect retrieval/grounding against `data/raw` in full plus a sample of `data/corpus`, not the complete legal/medical/policy corpora — full-corpus precision/recall is measured separately by `experiments/04_precision_recall_eval.py` and reported in `docs/rq1_metrics_definition.md`.
- **Demo question count exceeds the Milestone 4 guideline.** The spec calls for "5–10 samples"; the current demo runs 13 (see Setup step 4 for the breakdown and trimming instructions).

## Testing

```powershell
pytest
```

Unit tests cover chunking (all three strategies), document ingestion, and the grounding/citation-resolution logic — including a regression test that pins down the citation-format bug described above, so it can't silently reappear.

## References

### Methods

- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems 33*, 9459–9474.
- Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. *arXiv:2103.06268*.
- Nussbaum, Z., Morris, J. X., Duderstadt, B., & Mulyar, A. (2024). Nomic Embed: Training a reproducible long context text embedder. *arXiv:2402.01613*.
- Ollama documentation — https://github.com/ollama/ollama
- ChromaDB documentation — https://docs.trychroma.com

### Data Sources

- AGBonnet. (n.d.). augmented-clinical-notes [Dataset]. Hugging Face. https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes
- U.S. Environmental Protection Agency. (n.d.). EPA.gov. https://www.epa.gov/
- Office of the Federal Register, National Archives and Records Administration. (n.d.). Federal Register. https://www.federalregister.gov/
- Federal Energy Regulatory Commission. (n.d.). FERC.gov. https://www.ferc.gov/
- U.S. Government Publishing Office. (n.d.). GovInfo. https://www.govinfo.gov/