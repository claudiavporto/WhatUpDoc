#!/usr/bin/env python3
"""RQ1: Does chunking strategy affect retrieval precision on legal and medical documents?

Owner: Claudia Porto (feature/performance-testing)

For each chunking strategy (fixed, sentence, paragraph):
  1. Build a FRESH collection from the entire corpus (data/corpus/{policy,legal,medical})
  2. Run every query in the final evaluation set (data/eval/final_query_set.csv)
     against that collection
  3. Check whether the query's known-correct source document appears in the
     top-k retrieved chunks

Metrics (per strategy, and per category within each strategy):
  - Hit@k   : fraction of queries where the correct source document appears
              anywhere in the top-k results (k configurable, default 5)
  - MRR     : Mean Reciprocal Rank -- 1/rank of the first correct hit,
              averaged across queries (0 if no hit in top-k)

"Correct" is defined at the SOURCE DOCUMENT level (does the retrieved chunk
come from the right file), not exact page/location, since the evaluation
CSV's `location` field was assigned by a different batching scheme than the
live chunking pipeline uses (particularly for .txt files) and isn't reliably
comparable chunk-for-chunk. Source-document-level precision is the standard
and defensible unit for this kind of retrieval evaluation.

WARNING ON RUNTIME: embedding a single 66-page PDF was measured at ~13
minutes in prior testing. This script embeds the ENTIRE corpus (all PDFs +
DOCX + TXT) THREE TIMES (once per strategy), from scratch, every run. That
could take multiple hours depending on corpus size. Use
--max-files-per-category during development to sanity-check the script on a
handful of files before committing to a full run.

Usage:
    python experiments/04_precision_recall_eval.py
    python experiments/04_precision_recall_eval.py --max-files-per-category 3
    python experiments/04_precision_recall_eval.py --strategies fixed sentence
    python experiments/04_precision_recall_eval.py --top-k 3
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

# This script lives in experiments/, one level below the repo root, but
# imports `src.*` as if run from the repo root. Python only auto-adds the
# invoked script's own directory to sys.path, not the current working
# directory, so without this the import fails when run as
# `python experiments/04_precision_recall_eval.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_query_set(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_corpus_files(corpus_dir, categories, max_per_category=None):
    """Return list of (path, category) tuples for every corpus file."""
    files = []
    for category in categories:
        cat_dir = Path(corpus_dir) / category
        if not cat_dir.is_dir():
            print(f"  WARNING: {cat_dir} not found, skipping category '{category}'")
            continue
        cat_files = sorted(
            p for p in cat_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
        )
        if max_per_category:
            cat_files = cat_files[:max_per_category]
        for p in cat_files:
            files.append((p, category))
    return files


def build_collection(strategy, corpus_files, embedder, store_cls, get_config,
                      load_document, chunk_pages, collection_name):
    """Build a fresh collection for one strategy from the given corpus files."""
    cfg = dict(get_config())
    cfg["retrieval"] = dict(cfg["retrieval"])
    cfg["retrieval"]["collection_name"] = collection_name
    store = store_cls(cfg)
    store.reset()  # guarantee "from scratch" even if this collection existed before

    total_chunks = 0
    t0 = time.time()
    for i, (path, category) in enumerate(corpus_files, start=1):
        try:
            pages = load_document(str(path))
            chunks = chunk_pages(pages, strategy)
        except Exception as e:
            print(f"  [SKIP] {path.name}: {e}")
            continue
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)
        store.add_chunks(chunks, embeddings)
        total_chunks += len(chunks)
        elapsed = time.time() - t0
        print(f"  [{i}/{len(corpus_files)}] {path.name} ({category}) "
              f"-> {len(chunks)} chunks | total so far: {total_chunks} | "
              f"elapsed: {elapsed/60:.1f} min")

    return store, total_chunks


def evaluate_queries(store, embedder, queries, top_k):
    """Run every query against the collection; return per-query result rows."""
    results = []
    for row in queries:
        query_text = row["query"]
        expected_source = row["source_file"]
        category = row["category"]

        query_embedding = embedder.embed_one(query_text)
        hits = store.query(query_embedding, top_k=top_k)

        rank_of_hit = None
        for rank, hit in enumerate(hits, start=1):
            if hit.source == expected_source:
                rank_of_hit = rank
                break

        results.append({
            "query_id": row.get("id", ""),
            "category": category,
            "query": query_text,
            "expected_source": expected_source,
            "hit_at_k": rank_of_hit is not None,
            "rank_of_hit": rank_of_hit if rank_of_hit else "",
            "reciprocal_rank": (1.0 / rank_of_hit) if rank_of_hit else 0.0,
            "top_result_source": hits[0].source if hits else "",
            "top_result_distance": f"{hits[0].distance:.4f}" if hits else "",
        })
    return results


def summarize(results, strategy):
    """Aggregate hit@k and MRR overall and per category."""
    by_category = defaultdict(list)
    for r in results:
        by_category["ALL"].append(r)
        by_category[r["category"]].append(r)

    summary_rows = []
    for category, rows in by_category.items():
        n = len(rows)
        hit_rate = sum(1 for r in rows if r["hit_at_k"]) / n if n else 0.0
        mrr = sum(r["reciprocal_rank"] for r in rows) / n if n else 0.0
        summary_rows.append({
            "strategy": strategy,
            "category": category,
            "n_queries": n,
            "hit_at_k": round(hit_rate, 3),
            "mrr": round(mrr, 3),
        })
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--query-set", default="data/eval/final_query_set.csv")
    parser.add_argument("--strategies", nargs="+", default=["fixed", "sentence", "paragraph"],
                         choices=["fixed", "sentence", "paragraph"])
    parser.add_argument("--top-k", type=int, default=4,
                         help="Chunks retrieved per query. Default 4 matches "
                              "configs/config.yaml's retrieval.top_k (production value), "
                              "so RQ1 evaluation reflects the same k the live app actually uses.")
    parser.add_argument("--max-files-per-category", type=int, default=None,
                         help="Limit corpus files per category, for a fast sanity check "
                              "before committing to a full run (full corpus embeds "
                              "3x and can take hours).")
    parser.add_argument("--categories", nargs="+", default=["policy", "legal", "medical"])
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    try:
        from src.data_loader import load_document
        from src.chunking import chunk_pages
        from src.embeddings import OllamaEmbedder
        from src.vector_store import VectorStore
        from src.config import get_config
    except Exception:
        print("FAILED to import pipeline modules. Run this from the repo root")
        print("with your conda env activated.")
        raise

    queries = load_query_set(args.query_set)
    print(f"Loaded {len(queries)} queries from {args.query_set}")

    corpus_files = collect_corpus_files(args.corpus_dir, args.categories, args.max_files_per_category)
    print(f"Found {len(corpus_files)} corpus file(s) across {args.categories}")
    if args.max_files_per_category:
        print(f"NOTE: limited to {args.max_files_per_category} file(s) per category "
              f"(--max-files-per-category). This is a SANITY CHECK run, not a full result.")

    embedder = OllamaEmbedder()

    all_summary_rows = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy in args.strategies:
        section(f"STRATEGY: {strategy}")
        collection_name = f"rq1_eval_{strategy}"

        print(f"Building collection '{collection_name}' from scratch...")
        store, total_chunks = build_collection(
            strategy, corpus_files, embedder, VectorStore, get_config,
            load_document, chunk_pages, collection_name,
        )
        print(f"Collection built: {total_chunks} total chunks")

        print(f"\nEvaluating {len(queries)} queries at top-{args.top_k}...")
        results = evaluate_queries(store, embedder, queries, args.top_k)

        detail_path = output_dir / f"rq1_detail_{strategy}.csv"
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Per-query detail written to {detail_path}")

        summary_rows = summarize(results, strategy)
        all_summary_rows.extend(summary_rows)

        print(f"\n{'Category':<10} {'N':>4} {'Hit@' + str(args.top_k):>8} {'MRR':>8}")
        for row in summary_rows:
            print(f"{row['category']:<10} {row['n_queries']:>4} {row['hit_at_k']:>8} {row['mrr']:>8}")

    summary_path = output_dir / "rq1_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_rows)

    section("FINAL SUMMARY (all strategies)")
    print(f"{'Strategy':<12} {'Category':<10} {'N':>4} {'Hit@' + str(args.top_k):>8} {'MRR':>8}")
    for row in all_summary_rows:
        print(f"{row['strategy']:<12} {row['category']:<10} {row['n_queries']:>4} {row['hit_at_k']:>8} {row['mrr']:>8}")
    print(f"\nFull summary written to {summary_path}")
    print("Per-query detail written to experiments/results/rq1_detail_<strategy>.csv")


if __name__ == "__main__":
    main()