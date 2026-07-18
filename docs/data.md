# Dataset Documentation

## Overview

WhatUpDoc uses a curated test corpus designed to simulate the types of sensitive documents the system is built for. The corpus spans three domains chosen to stress-test retrieval across different vocabulary complexity, document length, and structural patterns.

No real PII or PHI is present in any document in this corpus.

**Note on scope:** the corpus grew significantly beyond initial estimates during dataset curation (see Known Deviations below). Numbers in this document reflect the actual corpus as validated, not the original proposal targets.

## Corpus Summary

| Domain | Source | Format | Count | Notes |
|--------|--------|--------|-------|-------|
| Medical records | Synthetic clinical notes ([AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes), Hugging Face) | DOCX, TXT | 200 (100 DOCX + 100 TXT) | No real PII; deduplicated by content hash |
| Legal contracts | CUAD (Hendrycks et al., 2021) | PDF | 49 | Creative Commons licensed |
| Policy documents | Federal Register, EPA, FERC (government open-data portals) | PDF | 37 | Public domain |

**Total: 286 documents, approximately 4,850 pages** (see Known Deviations — this exceeds the originally proposed ~130 documents / 800–1,200 pages).

## Known Deviations from Original Proposal

- **Document count and page count are substantially higher than proposed.** The policy category in particular expanded from covering primarily drinking water regulation to also include Clean Water Act/NPDES, FERC energy transmission orders, and RCRA solid waste rules, adding both document count and, in a few cases, very long individual filings (e.g., one FERC order alone runs 1,255 pages).
- **Category page counts are imbalanced.** Policy accounts for roughly 3,600 of the corpus's ~4,850 pages (≈74%), with legal at ~1,035 and medical at ~225. This means the vector store will be policy-dominated relative to the other two domains, which should be accounted for when interpreting RQ1 (chunking strategy) results across domains — retrieval behavior comparisons between domains will not be working from comparably-sized pools.
- **Medical format changed** from the originally proposed PDF/TXT split to DOCX/TXT (no PDF), since clinical notes are generated directly as text and saved in these formats rather than rendered to PDF.

All PDF, DOCX, and TXT files have been validated for corruption, text-extractability, and exact duplication using `validate_corpus.py` (see Validation section below). As of the most recent run: 0 corrupted files, 0 non-extractable/scanned files, 0 duplicate sets.

## Domain Descriptions

### Medical Records

Synthetic clinical notes sourced from the [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) dataset on Hugging Face, which contains structured clinical note text (patient history, diagnosis, medications, discharge-style content). 100 unique records are saved as DOCX and a further 100 unique records (non-overlapping with the DOCX set) are saved as TXT, giving the corpus two formats to test ingestion against. All content is synthetic; no real patient names, dates, or identifying details are present.

Files are stored in `data/corpus/medical/` and generated via `data/fetch_medical.py`, which streams the source dataset, deduplicates by content hash, and filters out near-empty records (under 50 characters).

### Legal Contracts

Commercial contracts sourced from the Contract Understanding Atticus Dataset (CUAD), which contains 510 contracts with expert annotations across 41 legal question categories. CUAD is available under a Creative Commons license and is publicly accessible at https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1/full_contract_pdf/Part_I.

A subset of 49 contracts was selected to represent a range of contract types including licensing agreements, service agreements, development agreements, and non-competition agreements.

Files are stored in `data/corpus/legal/`.

### Policy Documents

Federal regulatory and infrastructure documents sourced from government open-data portals:

- [Federal Register](https://www.federalregister.gov/) — primary source for final rules (e.g., Lead and Copper Rule Minor Revisions, NPDES rules, RCRA solid waste rules)
- [EPA](https://www.epa.gov/) — program pages and supporting documents (fact sheets, guidance, rule summaries) for drinking water, Clean Water Act, and solid waste regulations
- [FERC](https://www.ferc.gov/) — energy transmission and grid infrastructure orders
- [GovInfo](https://www.govinfo.gov/) — archival source for select Federal Register PDFs and CFR text

The category originally focused on drinking water regulation (Safe Drinking Water Act / Lead and Copper Rule) and has since been broadened to include:

- Safe Drinking Water Act rules (Lead and Copper Rule, Total Coliform Rule)
- Clean Water Act / NPDES wastewater rules
- FERC energy transmission and grid infrastructure orders
- RCRA solid waste / municipal landfill rules

These documents are public domain and represent the kind of regulatory and policy documentation common to public agencies and utilities. Document length varies widely, from 2-page fact sheets to multi-hundred-page final rules — see Known Deviations above for how this affects corpus balance.

Files are stored in `data/corpus/policy/`.

## Directory Structure

```
data/
├── corpus/
│   ├── medical/        # Synthetic clinical notes (DOCX + TXT)
│   ├── legal/           # CUAD contract subset (PDF)
│   └── policy/          # Public domain regulatory documents (PDF)
├── raw/                 # Original unprocessed files (gitignored)
├── processed/           # Chunked outputs after ingestion (gitignored)
├── fetch_medical.py     # Pulls and dedupes synthetic medical notes from Hugging Face
├── validate_corpus.py   # Corpus validation: corruption, extractability, duplicates, page counts
├── ingest.py            # Ingestion pipeline
└── chunking.py          # Chunking strategy implementations
```

## Validation

Before ingestion, run the validation script to confirm the corpus is free of corrupted files, scanned/non-extractable PDFs, and exact duplicates:

```bash
python validate_corpus.py --data-dir data/corpus
```

This reports, per category and in total: file counts, page counts (actual for PDF, estimated from word count for DOCX/TXT), corrupted files, likely-scanned or empty files, and duplicate sets detected via content hash. Re-run this any time new documents are added to the corpus.

## Data Loading and Exploration

To load and inspect the corpus before ingestion:

```bash
python data/explore.py --input data/corpus/
```

This script prints a summary of all documents in the corpus including file name, format, estimated page count, and word count.

## Ingestion

To ingest the full corpus using the default sentence-boundary chunking strategy:

```bash
python data/ingest.py --input data/corpus/ --strategy sentence
```

To run ingestion across all three strategies for RQ1 comparison:

```bash
python data/ingest.py --input data/corpus/ --strategy fixed
python data/ingest.py --input data/corpus/ --strategy sentence
python data/ingest.py --input data/corpus/ --strategy paragraph
```

## Preprocessing Notes

- PDF files are parsed page by page using PyMuPDF. Headers, footers, and page numbers are stripped where detectable.
- DOCX files are parsed paragraph by paragraph using python-docx (including table cell text) and rejoined as full document text before chunking.
- TXT files are read directly as plain text (UTF-8, with a Latin-1 fallback for encoding edge cases).
- Each chunk is tagged with source filename, page number (or estimated page for DOCX/TXT), chunk index, and chunking strategy as metadata in ChromaDB.
- Empty chunks and chunks under 20 words are filtered out before embedding.

## References

Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. arXiv:2103.06268.

AGBonnet. (n.d.). augmented-clinical-notes [Dataset]. Hugging Face. https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes

U.S. Environmental Protection Agency. (n.d.). EPA.gov. https://www.epa.gov/

Office of the Federal Register, National Archives and Records Administration. (n.d.). Federal Register. https://www.federalregister.gov/

Federal Energy Regulatory Commission. (n.d.). FERC.gov. https://www.ferc.gov/

U.S. Government Publishing Office. (n.d.). GovInfo. https://www.govinfo.gov/
