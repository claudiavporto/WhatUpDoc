#!/usr/bin/env python3
"""
smoke_test_pipeline.py

Owner: Claudia Porto (feature/performance-testing)

Runs the real WhatUpDoc pipeline end to end on one sample file:

    src.data_loader.load_document
        -> src.chunking.chunk_pages
        -> src.embeddings.OllamaEmbedder
        -> src.vector_store.VectorStore.add_chunks
        -> src.vector_store.VectorStore.query

Run from the repo root (so `src` and `utils` are importable as packages
and configs/config.yaml resolves correctly):

    python smoke_test_pipeline.py --sample-file data/corpus/policy/00-3.pdf

Optional:
    --strategy fixed|sentence|paragraph   (default: paragraph)
    --test-query "some question"          (default: a generic one)
    --collection-name smoke-test          (keeps test data out of your
                                            real collection; recommended)

Each stage is wrapped separately so a failure tells you exactly which
piece broke and why, instead of one raw traceback.
"""

import argparse
import sys
import traceback
from pathlib import Path


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_summary(results):
    section("SUMMARY")
    for stage, ok in results.items():
        print(f"  {stage:15s} {'OK' if ok else 'NOT CONFIRMED'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-file", required=True, help="A real corpus file, e.g. data/corpus/policy/00-3.pdf")
    parser.add_argument("--strategy", default="paragraph", choices=["fixed", "sentence", "paragraph"])
    parser.add_argument("--test-query", default="What is the standard or limit described in this document?")
    parser.add_argument("--collection-name", default="smoke_test_collection",
                         help="Chroma collection to use for this test, so it doesn't pollute your real one")
    args = parser.parse_args()

    results = {"load": False, "chunk": False, "embed": False, "store": False, "query": False}
    sample_file = Path(args.sample_file)

    if not sample_file.exists():
        print(f"ERROR: sample file not found: {sample_file}")
        print("Run this from the repo root, and check the path is correct.")
        sys.exit(1)

    if not Path("src").is_dir() or not Path("src/__init__.py").exists():
        print("WARNING: no src/__init__.py found in the current directory.")
        print("This script must be run from the repo root so `src` and `utils`")
        print("import as proper packages (matching their `from src.x import y` style).")

    # ------------------------------------------------------------------
    # Stage 1: load_document -> list[Page]
    # ------------------------------------------------------------------
    section("STAGE 1: src.data_loader.load_document")
    try:
        from src.data_loader import load_document
    except Exception:
        print("FAILED to import src.data_loader. Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    try:
        pages = load_document(str(sample_file))
        print(f"OK: loaded {sample_file.name} -> {len(pages)} Page section(s)")
        if pages:
            print(f"    first page: source={pages[0].source!r} page_number={pages[0].page_number} chars={len(pages[0].text)}")
        results["load"] = True
    except Exception:
        print(f"FAILED calling load_document({sample_file}). Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    if not pages:
        print("WARNING: 0 pages extracted (blank/image-only file?). Stopping here.")
        print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 2: chunk_pages -> list[Chunk]
    # ------------------------------------------------------------------
    section(f"STAGE 2: src.chunking.chunk_pages (strategy='{args.strategy}')")
    try:
        from src.chunking import chunk_pages
    except Exception:
        print("FAILED to import src.chunking. Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    try:
        chunks = chunk_pages(pages, args.strategy)
        print(f"OK: {len(pages)} page(s) -> {len(chunks)} chunk(s)")
        if chunks:
            c = chunks[0]
            print(f"    first chunk_id={c.chunk_id!r} strategy={c.strategy!r} chars={len(c.text)}")
        results["chunk"] = True
    except Exception:
        print(f"FAILED calling chunk_pages(pages, '{args.strategy}'). Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    if not chunks:
        print("WARNING: 0 chunks produced. Stopping here.")
        print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 3: OllamaEmbedder
    # ------------------------------------------------------------------
    section("STAGE 3: src.embeddings.OllamaEmbedder")
    try:
        from src.embeddings import OllamaEmbedder
    except Exception:
        print("FAILED to import src.embeddings. Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    try:
        embedder = OllamaEmbedder()
        print(f"OK: OllamaEmbedder initialized (host={embedder.host}, model={embedder.model})")
    except Exception:
        print("FAILED to initialize OllamaEmbedder. This usually means either")
        print("configs/config.yaml is missing/malformed, or the configured host")
        print("failed the offline/privacy check. Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunk(s) via Ollama...")
    try:
        embeddings = embedder.embed_batch(texts)
        dim = len(embeddings[0]) if embeddings and embeddings[0] else "unknown"
        print(f"OK: got {len(embeddings)} embedding(s), dimension={dim}")
        results["embed"] = True
    except RuntimeError as e:
        # embeddings.py already raises a clear, actionable RuntimeError
        # when Ollama is unreachable -- surface it as-is.
        print(f"FAILED: {e}")
        print_summary(results)
        sys.exit(1)
    except Exception:
        print("FAILED calling embed_batch(). Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 4: VectorStore.add_chunks + query
    # ------------------------------------------------------------------
    section("STAGE 4: src.vector_store.VectorStore")
    try:
        from src.vector_store import VectorStore
        from src.config import get_config
    except Exception:
        print("FAILED to import src.vector_store / src.config. Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    try:
        cfg = get_config()
        cfg = dict(cfg)  # shallow copy so we don't mutate the cached config
        cfg["retrieval"] = dict(cfg["retrieval"])
        cfg["retrieval"]["collection_name"] = args.collection_name
        store = VectorStore(cfg)
        print(f"OK: VectorStore initialized (collection='{args.collection_name}')")
    except Exception:
        print("FAILED to initialize VectorStore. Check configs/config.yaml has a")
        print("[retrieval] section with persist_dir, collection_name, distance_metric, top_k.")
        print("Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    try:
        store.add_chunks(chunks, embeddings)
        print(f"OK: upserted {len(chunks)} chunk(s) into '{args.collection_name}'")
        results["store"] = True
    except Exception:
        print("FAILED calling add_chunks(chunks, embeddings). Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    print()
    print(f"Querying with: {args.test_query!r}")
    try:
        query_embedding = embedder.embed_one(args.test_query)
        hits = store.query(query_embedding, top_k=3)
        print(f"OK: retrieved {len(hits)} hit(s)")
        for i, hit in enumerate(hits, start=1):
            preview = hit.text[:80].replace("\n", " ")
            print(f"    #{i} [{hit.source} p{hit.page_number}] dist={hit.distance:.4f}  {preview}...")
        results["query"] = True
    except Exception:
        print("FAILED during query. Full traceback:")
        traceback.print_exc()
        print_summary(results)
        sys.exit(1)

    print()
    print(f"NOTE: test data was written to the '{args.collection_name}' collection,")
    print("separate from your real one. Call store.reset() or delete that")
    print("collection when you're done testing, if you want to clean up.")

    print_summary(results)


if __name__ == "__main__":
    main()
