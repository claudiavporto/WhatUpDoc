# Research and Method Selection

## 1. Objectives

WhatUpDoc is a locally-hosted, privacy-first RAG application built for professionals who work with sensitive documents. The system is designed for use cases where uploading files to a cloud-based AI service is not an option due to regulatory requirements (HIPAA, NDAs) or organizational policy. All processing happens on the user's machine with zero external network calls.

The system performs three core tasks:

- **Document retrieval:** Given a user query, identify the most semantically relevant passages from a corpus of uploaded PDF and DOCX files using dense vector similarity search.
- **Grounded answer generation:** Generate a natural language answer that is strictly grounded in the retrieved passages, with the source document and page number cited in every response.
- **Privacy preservation:** Execute the entire pipeline, including embedding, retrieval, and generation, entirely offline using locally served models with no data transmitted to external servers.

These tasks map directly onto the RAG framework introduced by Lewis et al. (2020), with the key distinction that every component (the embedding model, the vector store, and the language model) runs locally rather than through a third-party API. The target document types are dense and domain-specific: legal contracts, medical records, and public policy documents. These document types were chosen because they represent the highest-stakes privacy scenarios and the most demanding retrieval challenges in terms of vocabulary complexity and document length.