# Document Ingestion

Use the `ingest_document()` function to parse a document and apply the desired chunking strategy.

```python
from doc_ingestion import ingest_document

chunks, metadata = ingest_document(
    "docs/example.pdf",
    strategy="fixed"
)
```

## Parameters

- `file_path` – Path to the PDF or DOCX document.
- `strategy` – Chunking strategy to apply.

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `"fixed"` | Fixed-size token chunking with overlap. |
| `"sentence"` | Groups complete sentences into chunks. |
| `"paragraph"` | Groups complete paragraphs into chunks. |

## Returns

- `chunks` – List of document chunks with metadata.
- `metadata` – Document-level metadata (filename, title, author, page count, etc.).
