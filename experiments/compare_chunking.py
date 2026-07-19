#!/usr/bin/env python3
"""
compare_chunking.py

RQ1 support script: runs all three chunking strategies (fixed, sentence, paragraph)
from Sean's utils/doc_ingestion.py across the validated corpus and reports comparative
stats, both overall and broken down by domain (medical/legal/policy).

utils/doc_ingestion.py now supports .pdf, .docx, and .txt (as of the fix adding
parse_txt() and updating SUPPORTED_EXTENSIONS). All three are treated as ingestable
below. If any other unsupported extension shows up in the corpus later, it will be
reported as skipped rather than silently ignored or crashing the run.

This does NOT test retrieval precision directly (that requires the embedding +
vector store + a query set) — it's the first step: understanding how each strategy
actually splits up the real corpus before you build the retrieval evaluation on top.

Run this script from the repo root (so the utils/ package can be found):
    python compare_chunking.py --data-dir data/corpus
    python compare_chunking.py --data-dir data/corpus --strategies fixed sentence
"""

import argparse
import contextlib
import io
import statistics
import sys
from pathlib import Path

# Make sure the repo root is on sys.path so `utils` resolves as a package,
# regardless of where this script is actually invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from utils.doc_ingestion import ingest_document
except ImportError as e:
    print("ERROR: could not import ingest_document from utils/doc_ingestion.py")
    print(f"  ({e})")
    print()
    print("Make sure this script is run from the repo root (WhatUpDoc/), and that")
    print("utils/doc_ingestion.py and utils/__init__.py both exist.")
    sys.exit(1)

STRATEGIES = ["fixed", "sentence", "paragraph"]
# utils/doc_ingestion.py now supports all three of these (post .txt fix).
INGESTABLE_EXTENSIONS = {".pdf", ".docx", ".txt"}
# Anything else found in the corpus gets reported as skipped rather than crashing.
SKIPPED_EXTENSIONS = set()


def find_corpus_files(data_dir: Path) -> tuple:
    """
    Group files by domain subfolder (medical/legal/policy), matching validate_corpus.py's
    approach. Returns (domains, skipped) where domains maps domain -> list of ingestable
    files, and skipped maps domain -> list of files with unsupported extensions (.txt).
    """
    domains = {}
    skipped = {}
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]

    def collect(folder: Path):
        all_files = sorted(folder.rglob("*"))
        ingestable = [f for f in all_files if f.suffix.lower() in INGESTABLE_EXTENSIONS]
        skip = [f for f in all_files if f.suffix.lower() in SKIPPED_EXTENSIONS]
        return ingestable, skip

    if subdirs:
        for d in subdirs:
            ingestable, skip = collect(d)
            if ingestable:
                domains[d.name] = ingestable
            if skip:
                skipped[d.name] = skip
    else:
        ingestable, skip = collect(data_dir)
        if ingestable:
            domains["all"] = ingestable
        if skip:
            skipped["all"] = skip

    return domains, skipped


def chunk_lengths_words(chunks) -> list:
    """Return word count per chunk, tolerant of chunks being strings or dicts with a 'text' key."""
    lengths = []
    for c in chunks:
        if isinstance(c, str):
            text = c
        elif isinstance(c, dict):
            text = c.get("text", "")
        else:
            text = str(c)
        lengths.append(len(text.split()))
    return lengths


def summarize(lengths: list) -> dict:
    if not lengths:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "stdev": 0}
    return {
        "count": len(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "stdev": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare chunking strategies across the WhatUpDoc corpus (RQ1).")
    parser.add_argument("--data-dir", type=str, default="data/corpus",
                         help="Path to the corpus directory (default: data/corpus)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES,
                         help=f"Which strategies to run (default: all of {STRATEGIES})")
    parser.add_argument("--limit-per-domain", type=int, default=None,
                         help="Optional: only process the first N files per domain (useful for a quick test run)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    domains, skipped = find_corpus_files(data_dir)
    if not domains:
        print(f"No .pdf or .docx files found under {data_dir} "
              f"(utils/doc_ingestion.py doesn't support .txt)")
        sys.exit(1)

    total_skipped = sum(len(files) for files in skipped.values())
    if total_skipped:
        print(f"NOTE: skipping {total_skipped} file(s) with unsupported extensions "
              f"(see SKIPPED_EXTENSIONS). See breakdown at the end.\n")

    # results[strategy][domain] = list of word-count-per-chunk across all files in that domain
    results = {s: {d: [] for d in domains} for s in args.strategies}
    file_errors = []

    for domain, files in domains.items():
        files_to_process = files[: args.limit_per_domain] if args.limit_per_domain else files
        print(f"\n[{domain}] processing {len(files_to_process)} file(s)...")

        for file_path in files_to_process:
            for strategy in args.strategies:
                try:
                    # doc_ingestion.py prints "Document Information" on every call —
                    # suppress it here so the actual comparison report stays readable.
                    with contextlib.redirect_stdout(io.StringIO()):
                        chunks, metadata = ingest_document(str(file_path), strategy=strategy)
                    lengths = chunk_lengths_words(chunks)
                    results[strategy][domain].extend(lengths)
                except Exception as e:
                    file_errors.append((file_path, strategy, str(e)))
                    print(f"  [ERROR] {file_path.name} ({strategy}): {e}")

    # ---- Report ----
    print("\n" + "=" * 78)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 78)

    for strategy in args.strategies:
        print(f"\n### Strategy: {strategy}")
        all_lengths_this_strategy = []

        for domain in domains:
            lengths = results[strategy][domain]
            all_lengths_this_strategy.extend(lengths)
            s = summarize(lengths)
            print(f"  [{domain:10s}] chunks={s['count']:6d}  "
                  f"mean={s['mean']:6.1f}w  median={s['median']:6.1f}w  "
                  f"min={s['min']:4d}  max={s['max']:5d}  stdev={s['stdev']:6.1f}")

        overall = summarize(all_lengths_this_strategy)
        print(f"  [{'OVERALL':10s}] chunks={overall['count']:6d}  "
              f"mean={overall['mean']:6.1f}w  median={overall['median']:6.1f}w  "
              f"min={overall['min']:4d}  max={overall['max']:5d}  stdev={overall['stdev']:6.1f}")

    if file_errors:
        print("\n" + "=" * 78)
        print(f"ERRORS ({len(file_errors)} file/strategy combinations failed)")
        print("=" * 78)
        for path, strategy, err in file_errors:
            print(f"  {path} [{strategy}]: {err}")

    if skipped:
        print("\n" + "=" * 78)
        print(f"SKIPPED — UNSUPPORTED FILE TYPES ({total_skipped} file(s))")
        print("=" * 78)
        for domain, files in skipped.items():
            print(f"  [{domain}] {len(files)} file(s) skipped: "
                  f"{sorted(set(f.suffix.lower() for f in files))}")
        print("\n  These files were NOT included in the chunking comparison above.")

    print("\nDone.")
    print("\nNote: this compares chunk SIZE distributions only. To evaluate retrieval")
    print("PRECISION (RQ1's actual question), you'll need to run each strategy through")
    print("embedding + ChromaDB + a query set, then measure recall/precision per strategy.")


if __name__ == "__main__":
    main()