# Dataset Documentation

## Overview

WhatUpDoc uses a curated synthetic test corpus designed to simulate the types of sensitive documents the system is built for. The corpus spans three domains chosen to stress-test retrieval across different vocabulary complexity, document length, and structural patterns.

No real PII or PHI is present in any document in this corpus.

## Corpus Summary

| Domain | Source | Format | Count | Notes |
|--------|--------|--------|-------|-------|
| Medical records | Synthetically generated | PDF, TXT | ~50 | No real PII |
| Legal contracts | CUAD (Hendrycks et al., 2021) | PDF | ~50 | Creative Commons licensed |
| Policy documents | Government open-data portals | PDF | ~30 | Public domain |

Total: approximately 130 documents, estimated 800 to 1,200 pages.

## Domain Descriptions

### Medical Records

Synthetic patient summaries and discharge notes generated to mimic HIPAA-relevant document structure. Each document includes sections typical of real medical records such as patient history, diagnosis, medications, and discharge instructions. All names, dates, and identifying details are fabricated.

Files are stored in `data/sample/medical/`.

### Legal Contracts

Commercial contracts sourced from the Contract Understanding Atticus Dataset (CUAD), which contains 510 contracts with expert annotations across 41 legal question categories. CUAD is available under a Creative Commons license and is publicly accessible at https://huggingface.co/datasets/theatticusproject/cuad.

A subset of 50 contracts was selected to represent a range of contract types including licensing agreements, service agreements, and employment contracts.

Files are stored in `data/sample/legal/`.

### Policy Documents

Municipal utility policies and infrastructure guidelines sourced from government open-data portals. These documents are public domain and represent the kind of internal policy documentation common in public agencies and municipal organizations.

Files are stored in `data/sample/policy/`.

## Directory Structure

```
data/
├── sample/
│   ├── medical/       # Synthetic patient records and discharge notes
│   ├── legal/         # CUAD contract subset
│   └── policy/        # Public domain policy documents
├── raw/               # Original unprocessed files (gitignored)
├── processed/         # Chunked outputs after ingestion (gitignored)
├── ingest.py          # Ingestion pipeline
└── chunking.py        # Chunking strategy implementations
```

## Data Loading and Exploration

To load and inspect the corpus before ingestion:

```bash
python data/explore.py --input data/sample/
```

This script prints a summary of all documents in the corpus including file name, format, estimated page count, and word count.

## Ingestion

To ingest the full corpus using the default sentence-boundary chunking strategy:

```bash
python data/ingest.py --input data/sample/ --strategy sentence
```

To run ingestion across all three strategies for RQ1 comparison:

```bash
python data/ingest.py --input data/sample/ --strategy fixed
python data/ingest.py --input data/sample/ --strategy sentence
python data/ingest.py --input data/sample/ --strategy paragraph
```

## Preprocessing Notes

- PDF files are parsed page by page using PyMuPDF. Headers, footers, and page numbers are stripped where detectable.
- DOCX files are parsed paragraph by paragraph using python-docx and rejoined as full document text before chunking.
- Each chunk is tagged with source filename, page number, chunk index, and chunking strategy as metadata in ChromaDB.
- Empty chunks and chunks under 20 words are filtered out before embedding.

## References

Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. arXiv:2103.06268.
