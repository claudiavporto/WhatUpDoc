"""Preliminary Experiment 2 — vector store retrieval smoke test.

    python experiments/02_retrieval_smoke_test.py

Validates the ChromaDB wrapper (insert, cosine retrieval, metadata
round-trip) WITHOUT requiring Ollama, by substituting a deterministic
bag-of-words hash embedding for the real nomic-embed-text model. This
lets every teammate verify the storage/retrieval layer on any machine.

The mock embedding is intentionally crude — it only captures lexical
overlap — so this test checks *plumbing*, not semantic quality.
Semantic retrieval precision with real embeddings is measured in
Milestone 4 against Kat's evaluation question set.

Pass criteria (asserted):
  1. Every chunk inserted is retrievable.
  2. A query about rent retrieves the lease chunk first.
  3. A query about allergies retrieves the medical record chunk first.
  4. Source/page metadata survives the round trip intact.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_pages       # noqa: E402
from src.config import get_config          # noqa: E402
from src.data_loader import load_directory  # noqa: E402
from src.vector_store import VectorStore    # noqa: E402

DIM = 256


def mock_embed(text: str) -> list[float]:
    """Deterministic bag-of-words hash embedding (test double).

    Each token is hashed into one of DIM buckets; the vector is the
    L2-normalized bucket count. Identical wording -> identical vector,
    shared vocabulary -> nonzero cosine similarity.
    """
    vec = [0.0] * DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def main() -> None:
    repo = Path(__file__).resolve().parent.parent

    # isolated throwaway store so we never pollute the real index
    cfg = get_config()
    cfg = {**cfg, "retrieval": {**cfg["retrieval"],
                                "persist_dir": tempfile.mkdtemp(prefix="whatupdoc_test_"),
                                "collection_name": "smoke_test"}}

    pages = load_directory(repo / "data" / "raw")
    chunks = chunk_pages(pages, "sentence")
    store = VectorStore(cfg)
    store.reset()
    store.add_chunks(chunks, [mock_embed(c.text) for c in chunks])

    # 1. everything inserted is retrievable
    assert store.collection.count() == len(chunks), "insert count mismatch"

    # 2 & 3. lexical routing sanity checks
    checks = {
        "monthly rent late fee tenant": "sample_lease_agreement.pdf",
        "patient allergies penicillin": "sample_medical_record.docx",
        "backflow prevention assembly testing": "sample_utility_policy.txt",
    }
    for query, expected_source in checks.items():
        hits = store.query(mock_embed(query), top_k=3)
        top = hits[0]
        assert top.source == expected_source, (
            f"query '{query}' routed to {top.source}, expected {expected_source}"
        )
        # 4. metadata round-trip
        assert top.page_number >= 1 and top.chunk_id
        print(f"PASS  '{query}' -> {top.source} p.{top.page_number} "
              f"(distance {top.distance:.3f})")

    print(f"\nAll retrieval smoke tests passed ({len(chunks)} chunks indexed).")


if __name__ == "__main__":
    main()
