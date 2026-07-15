# WhatUpDoc

**A Privacy-First Local RAG Application**

IE 7374 – Group 03 | Claudia Porto, Christopher Swartz, Sean Costello, Kat Fountain

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
- **Document parsing:** PyMuPDF, python-docx
- **Chunking:** fixed-size, sentence-boundary (spaCy), paragraph-boundary
- **UI:** Gradio

## Research Questions

| | |
|--|--|
| RQ1 | Does chunking strategy affect retrieval precision on legal and medical documents? |
| RQ2 | How does top-k context size affect LLaMA 3 response accuracy? |
| RQ3 | Can the pipeline run fully offline, and what are the latency trade-offs? |

## Privacy Guarantee



## References

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 33*, 9459–9474.

Hendrycks, D., et al. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. *arXiv:2103.06268.*