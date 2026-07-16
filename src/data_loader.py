"""Document ingestion for WhatUpDoc.

Extracts plain text from PDF, DOCX, and TXT files while preserving the
source metadata (filename, page/paragraph number) needed for citation.
Nothing in this module touches the network: parsing is fully local.

Each document becomes a list of Page records; downstream chunking
(src/chunking.py) splits pages into retrieval-sized pieces but carries
the page metadata through, so every answer can point back to
"contract.pdf, page 3".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from utils.helpers import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Page:
    """One unit of source text with the metadata needed for citation."""

    text: str
    source: str          # filename, e.g. "lease_agreement.pdf"
    page_number: int     # 1-based page (PDF) or section index (DOCX/TXT)
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Normalize extracted text.

    - collapse runs of spaces/tabs
    - collapse 3+ newlines to paragraph breaks
    - strip common PDF artifacts (soft hyphens, form feeds)
    """
    text = text.replace("\u00ad", "").replace("\f", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Page]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    pages: list[Page] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:  # skip blank/image-only pages
                pages.append(Page(text=text, source=path.name, page_number=i))
    return pages


def load_docx(path: Path) -> list[Page]:
    """Extract text from a DOCX.

    DOCX has no fixed pagination, so paragraphs are grouped into
    numbered sections (~15 paragraphs each) to give citations a stable
    location reference.
    """
    doc = DocxDocument(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    pages: list[Page] = []
    section_size = 15
    for i in range(0, len(paragraphs), section_size):
        text = clean_text("\n\n".join(paragraphs[i : i + section_size]))
        pages.append(
            Page(text=text, source=path.name, page_number=i // section_size + 1)
        )
    return pages


def load_txt(path: Path) -> list[Page]:
    """Load a plain-text or markdown file as a single section."""
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [Page(text=text, source=path.name, page_number=1)]


def load_document(path: str | Path) -> list[Page]:
    """Dispatch to the correct parser based on file extension.

    Args:
        path: Path to a .pdf, .docx, .txt, or .md file.

    Returns:
        List of Page records (empty if the file had no extractable text).

    Raises:
        ValueError: for unsupported file types.
        FileNotFoundError: if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    loader = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt, ".md": load_txt}[ext]
    pages = loader(path)
    logger.info("Loaded %s: %d section(s), %d chars",
                path.name, len(pages), sum(len(p.text) for p in pages))
    return pages


def load_directory(directory: str | Path) -> list[Page]:
    """Load every supported document in a directory (non-recursive)."""
    directory = Path(directory)
    pages: list[Page] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            try:
                pages.extend(load_document(path))
            except Exception as exc:  # keep ingesting other files
                logger.error("Failed to load %s: %s", path.name, exc)
    return pages
