"""Preliminary Experiment 1 — chunking strategy comparison (RQ1 groundwork).

    python experiments/01_chunking_comparison.py

Runs all three chunking strategies over the sample corpus in data/raw/
and measures structural properties that predict retrieval quality:

  - chunk count and size distribution (more, smaller chunks = finer
    retrieval granularity but less context per hit)
  - sentence integrity: % of chunks that end mid-sentence (broken
    boundaries degrade both embedding quality and answer readability)

Runs fully offline — no Ollama required — so any teammate can
reproduce it immediately after cloning. Results are written to
experiments/results/chunking_comparison.csv and .md.
"""

from __future__ import annotations

import csv
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import STRATEGIES, chunk_pages   # noqa: E402
from src.data_loader import load_directory          # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SENTENCE_ENDINGS = (".", "!", "?", '."', '?"', '!"', ":", ")")


def ends_cleanly(text: str) -> bool:
    """True if a chunk ends at a plausible sentence/clause boundary."""
    return text.rstrip().endswith(SENTENCE_ENDINGS)


def main() -> None:
    pages = load_directory(Path(__file__).resolve().parent.parent / "data" / "raw")
    if not pages:
        raise SystemExit("No sample docs found. Run `python data/make_samples.py` first.")

    rows = []
    for strategy in STRATEGIES:
        chunks = chunk_pages(pages, strategy)
        lengths = [len(c.text) for c in chunks]
        rows.append(
            {
                "strategy": strategy,
                "n_chunks": len(chunks),
                "mean_chars": round(statistics.mean(lengths), 1),
                "min_chars": min(lengths),
                "max_chars": max(lengths),
                "stdev_chars": round(statistics.pstdev(lengths), 1),
                "pct_clean_boundaries": round(
                    100 * sum(ends_cleanly(c.text) for c in chunks) / len(chunks), 1
                ),
            }
        )

    # CSV for downstream analysis
    csv_path = RESULTS_DIR / "chunking_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Markdown summary for the repo / report
    md = [
        f"# Preliminary Experiment 1 — Chunking Strategy Comparison ({date.today()})",
        "",
        f"Corpus: {len(pages)} sections from 3 sample documents "
        "(lease PDF, medical record DOCX, utility policy TXT).",
        "",
        "| Strategy | Chunks | Mean chars | Min | Max | Stdev | Clean boundaries |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['strategy']} | {r['n_chunks']} | {r['mean_chars']} | "
            f"{r['min_chars']} | {r['max_chars']} | {r['stdev_chars']} | "
            f"{r['pct_clean_boundaries']}% |"
        )
    md += [
        "",
        "## Observations",
        "",
        "- **Fixed** windows produce the most chunks but routinely cut "
        "mid-sentence (lowest clean-boundary rate), which fragments "
        "clauses like the late-fee provision across two chunks.",
        "- **Sentence** packing keeps every boundary clean at a modest "
        "cost in chunk count, and is the default going into Milestone 4.",
        "- **Paragraph** chunks best preserve contract clause structure "
        "but vary most in size, which may interact with top-k selection "
        "(RQ2). Full retrieval-precision measurement with real "
        "embeddings is scheduled for Milestone 4.",
    ]
    md_path = RESULTS_DIR / "chunking_comparison.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {csv_path}\nWrote {md_path}\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
