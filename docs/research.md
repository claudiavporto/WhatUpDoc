# Research and Method Selection

## 1. Objectives

WhatUpDoc performs three core tasks:

- **Document retrieval:** Given a user query, identify the most semantically relevant passages from a corpus of uploaded documents.
- **Grounded answer generation:** Generate a natural language answer that is strictly grounded in the retrieved passages, with source citations.
- **Privacy preservation:** Execute the entire pipeline offline with zero external network calls.

These tasks place WhatUpDoc in the domain of **retrieval-augmented generation (RAG)** — specifically a locally-hosted, privacy-first variant targeting dense document types (legal contracts, medical records, policy documents).