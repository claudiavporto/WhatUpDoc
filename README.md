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

## Research Questions

| | |
|--|--|
| RQ1 | Does chunking strategy affect retrieval precision on legal and medical documents? |
| RQ2 | How does top-k context size affect LLaMA 3 response accuracy? |
| RQ3 | Can the pipeline run fully offline, and what are the latency trade-offs? |

## Privacy Guarantee

WhatUpDoc is designed so that no document content, query text, or generated
answer ever leaves your machine.

- **Document parsing, chunking, and vector storage are 100% local.** PDF/DOCX/TXT
  parsing (PyMuPDF, python-docx), chunking, and ChromaDB's persistent store all
  operate on local files and local disk — none of these stages contain a
  network code path at all.
- **Embedding and generation are enforced local-only, not just configured
  local-only.** Both the embedding client (`src/embeddings.py`) and the
  generation client (`src/llm.py`) call a shared `assert_local_host()` guard
  before making any request. If `configs/config.yaml`'s `ollama.host` is ever
  set to anything other than `localhost`/`127.0.0.1`, the client refuses to
  construct and the pipeline stops rather than silently sending data
  off-machine.
- **ChromaDB's default embedding function is deliberately never used**, since
  it downloads a model from the internet on first use. `src/vector_store.py`
  only accepts embeddings supplied by the caller (our local Ollama-served
  `nomic-embed-text`), so no implicit network call can occur through the
  vector store either.
- **The UI is bound to localhost.** The Gradio interface runs with
  `share=False` on `127.0.0.1`, so the app itself is not exposed off-machine.

You can verify the local-host enforcement yourself (this has been run and
confirmed to correctly raise a `RuntimeError` for a non-local host):

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

See `docs/rq3_offline_latency.md` for the full analysis of offline enforcement
and its latency trade-offs.

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- [Ollama](https://ollama.com/download) — the local LLM/embedding server

### 1. Clone and create the environment

```powershell
git clone https://github.com/claudiavporto/WhatUpDoc.git
cd WhatUpDoc
conda env create -f environment.yml
conda activate whatupdoc
python -m spacy download en_core_web_sm
```

### 2. Install and start Ollama

Install from [ollama.com/download](https://ollama.com/download), then pull the
models this project uses:

```powershell
ollama pull nomic-embed-text
ollama pull llama3:8b
ollama pull mistral:7b
```

Confirm Ollama is running:

```powershell
curl -UseBasicParsing http://localhost:11434
```

You should see `Ollama is running`. If not, run `ollama serve` in a separate
terminal and leave it open.

### 3. Verify the pipeline runs end to end

```powershell
python experiments/tooling/smoke_test_pipeline.py --sample-file data/corpus/policy/00-3.pdf
```

This runs a real document through parse → chunk → embed → store → query and
reports which stage(s) succeed. A full pass confirms your environment is set
up correctly before you run any of the actual experiments.

### 4. Run the experiments

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
```

**Note on runtime:** `04_precision_recall_eval.py` embeds the full document
corpus from scratch for each chunking strategy. Measured throughput on local
CPU inference is roughly 2.2–2.5 seconds per chunk; a full-corpus, all-strategy
run can take several hours. Use `--max-files-per-category N` to run a fast
sanity check on a handful of files first.

### 5. Launch the app

```powershell
python app.py
```

## References

### Methods

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems 33*, 9459–9474.

Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. *arXiv:2103.06268*.

Nussbaum, Z., Morris, J. X., Duderstadt, B., & Mulyar, A. (2024). Nomic Embed: Training a reproducible long context text embedder. *arXiv:2402.01613*.

Ollama documentation — https://github.com/ollama/ollama

ChromaDB documentation — https://docs.trychroma.com

### Data Sources

AGBonnet. (n.d.). augmented-clinical-notes [Dataset]. Hugging Face. https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes

U.S. Environmental Protection Agency. (n.d.). EPA.gov. https://www.epa.gov/

Office of the Federal Register, National Archives and Records Administration. (n.d.). Federal Register. https://www.federalregister.gov/

Federal Energy Regulatory Commission. (n.d.). FERC.gov. https://www.ferc.gov/

U.S. Government Publishing Office. (n.d.). GovInfo. https://www.govinfo.gov/