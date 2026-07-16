"""ChromaDB vector store wrapper for WhatUpDoc.

Stores chunk embeddings in a local persistent ChromaDB collection and
performs cosine-similarity retrieval. Embeddings are always supplied by
the caller (src/embeddings.py) rather than Chroma's built-in embedding
function — Chroma's default downloads a model from the internet, which
would violate the offline guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src.chunking import Chunk
from src.config import get_config, resolve_path
from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieval hit: chunk content, citation metadata, and distance."""

    text: str
    source: str
    page_number: int
    chunk_id: str
    distance: float


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, config: dict | None = None):
        cfg = config or get_config()
        r = cfg["retrieval"]
        persist_dir = resolve_path(r["persist_dir"])
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=r["collection_name"],
            metadata={"hnsw:space": r["distance_metric"]},
        )
        self.top_k = r["top_k"]

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert (or upsert) chunks with caller-supplied embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "page_number": c.page_number, "strategy": c.strategy}
                for c in chunks
            ],
        )
        logger.info("Upserted %d chunk(s); collection size now %d",
                    len(chunks), self.collection.count())

    def query(self, query_embedding: list[float], top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k nearest chunks for a query embedding."""
        k = min(top_k or self.top_k, max(self.collection.count(), 1))
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        hits: list[RetrievedChunk] = []
        for doc, meta, dist, cid in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]
        ):
            hits.append(
                RetrievedChunk(
                    text=doc,
                    source=meta["source"],
                    page_number=meta["page_number"],
                    chunk_id=cid,
                    distance=dist,
                )
            )
        return hits

    def reset(self) -> None:
        """Delete and recreate the collection (used between experiments)."""
        name = self.collection.name
        meta = self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name, metadata=meta)
        logger.info("Collection '%s' reset", name)
