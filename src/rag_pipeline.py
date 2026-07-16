"""End-to-end RAG orchestration for WhatUpDoc.

Ties the pipeline together:

    documents -> data_loader -> chunking -> embeddings -> vector_store
                                                              |
    question  -> embeddings -> retrieval -> llm (grounded) <--+

Used by both the Gradio app (app.py) and the command-line runner
(src/model_runner.py). All parameters come from configs/config.yaml.
"""

from __future__ import annotations

from pathlib import Path

from src.chunking import chunk_pages
from src.config import get_config
from src.data_loader import load_directory, load_document
from src.embeddings import OllamaEmbedder
from src.llm import OllamaLLM
from src.vector_store import RetrievedChunk, VectorStore
from utils.helpers import get_logger, timed

logger = get_logger(__name__)


class RAGPipeline:
    """Ingest documents and answer grounded questions about them."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or get_config()
        self.embedder = OllamaEmbedder(self.cfg)
        self.store = VectorStore(self.cfg)
        self.llm = OllamaLLM(self.cfg)

    # -- ingestion -----------------------------------------------------------

    def _strategy_kwargs(self) -> tuple[str, dict]:
        """Pull the active chunking strategy and its parameters from config."""
        c = self.cfg["chunking"]
        strategy = c["strategy"]
        kwargs = {
            "fixed": {"chunk_size": c["fixed"]["chunk_size_chars"],
                      "overlap": c["fixed"]["overlap_chars"]},
            "sentence": {"max_chars": c["sentence"]["max_chars"],
                         "overlap_sentences": c["sentence"]["overlap_sentences"]},
            "paragraph": {"max_chars": c["paragraph"]["max_chars"]},
        }[strategy]
        return strategy, kwargs

    def ingest(self, path: str | Path) -> int:
        """Ingest one document or every document in a directory.

        Returns:
            Number of chunks added to the vector store.
        """
        path = Path(path)
        pages = load_directory(path) if path.is_dir() else load_document(path)
        if not pages:
            logger.warning("No extractable text found in %s", path)
            return 0

        strategy, kwargs = self._strategy_kwargs()
        chunks = chunk_pages(pages, strategy, **kwargs)

        with timed(f"Embedding {len(chunks)} chunks", logger):
            embeddings = self.embedder.embed_batch([c.text for c in chunks])
        self.store.add_chunks(chunks, embeddings)
        return len(chunks)

    # -- question answering ----------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed the question and return the nearest chunks."""
        q_emb = self.embedder.embed_one(question)
        return self.store.query(q_emb, top_k=top_k)

    def ask(self, question: str) -> dict:
        """Answer a question with citations.

        Returns:
            dict with keys:
              answer   — the grounded model response
              sources  — list of {source, page_number, distance, preview}
        """
        with timed("Retrieval", logger):
            hits = self.retrieve(question)
        if not hits:
            return {"answer": "No documents have been ingested yet.", "sources": []}

        with timed("Generation", logger):
            result = self.llm.generate_grounded(question, hits)

        return {
            "answer": result.text,
            "grounding": result.grounding.to_dict(),
            "grounding_summary": result.grounding.summary_line(),
            "sources": [
                {
                    "source": h.source,
                    "page_number": h.page_number,
                    "distance": round(h.distance, 4),
                    "preview": h.text[:200],
                }
                for h in result.used_chunks
            ],
        }
