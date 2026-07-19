# Document Ingestion (utils/doc_ingestion.py)

> **Note:** This module is used for the RQ1 chunk-structure comparison
> experiments (`experiments/compare_chunking.py`, `experiments/01_chunking_comparison.py`)
> and their tests. It is **not** part of the live retrieval pipeline —
> that path is `src/data_loader.py` → `src/chunking.py`. The two exist
> in parallel because this one preserves the original token-based
> chunking analysis from earlier milestones.

Use the `ingest_document()` function to parse a document and apply the desired chunking strategy.

```python
from utils.doc_ingestion import ingest_document

chunks, metadata = ingest_document(
    "data/corpus/policy/example.pdf",
    strategy="fixed"
)
```

## Parameters

- `file_path` – Path to the source document. Supports `.pdf`, `.docx`, and `.txt`.
- `strategy` – Chunking strategy to apply. One of `"fixed"`, `"sentence"`, `"paragraph"`.

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `"fixed"` | Fixed-size token chunking with overlap. Default `chunk_size=512`, `overlap=64`. Slides forward by `chunk_size - overlap` tokens per chunk. |
| `"sentence"` | Groups whole sentences (via spaCy's sentencizer) into chunks up to `chunk_size` tokens. No overlap; a sentence is never split across chunks. |
| `"paragraph"` | Groups whole paragraphs into chunks up to `chunk_size` tokens. No overlap; a paragraph is never split across chunks. |

Token counts for all three strategies are computed with spaCy's blank English tokenizer (not a whitespace split), so chunk boundaries reflect spaCy's tokenization rather than raw word counts.

## Parsing by file type

- **PDF** – parsed page by page via PyMuPDF. Metadata includes `page_number`. Blank pages are skipped.
- **DOCX** – parsed paragraph by paragraph via python-docx. Metadata includes `paragraph_number` and `paragraph_style`. Blank paragraphs are skipped.
- **TXT** – parsed paragraph by paragraph, splitting on single newlines (not blank-line breaks). Metadata includes `paragraph_number`. Reads as UTF-8 with a latin-1 fallback. Literal two-character `\n` sequences in the source are normalized to real newlines before splitting.

For `"paragraph"` chunking, a chunk's metadata is inherited from the *first* paragraph in that chunk, since a chunk can span multiple original paragraphs.

## Returns

- `chunks` – List of `{"text": ..., "metadata": {...}}`. Metadata always includes `filename`, `chunk_index`, `chunking_strategy`, and `token_count`, plus the file-type-specific fields above.
- `metadata` – Document-level metadata: `filename`, `title`, `author`, `file_type`, and either `total_pages` (PDF) or `total_paragraphs` (DOCX/TXT).

## Errors

- Unsupported file extension raises `ValueError`.
- An unrecognized `strategy` value raises `ValueError` (note: the actual error message has a typo — "Not a valdi chunking strategy" — harmless but worth fixing if you touch this file again).
- `"fixed"` chunking raises `ValueError` if `overlap >= chunk_size`.

## Note

Parsing prints a page/paragraph-count summary to console as a side effect on every call. Useful for interactive debugging, noisy if called from `experiments/compare_chunking.py` in a loop over many files.