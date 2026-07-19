"""Experiment 4 — Retrieval + grounding evaluation against the labeled query set.

    python experiments/04_retrieval_grounding_eval.py            # needs Ollama
    python experiments/04_retrieval_grounding_eval.py --selftest  # offline, no Ollama

Owner: Christopher Swartz (feature/llm-infrastructure)
Consumes the evaluation query set (data/eval/final_query_set.csv), which is
owned by the evaluation-framework workstream. This harness is the join point
between that answer key and the grounding layer (src/grounding.py): it turns
30 queries with known correct sources and answers into hard numbers.

For each query it measures four things a private-document assistant must get
right, and reports them overall and broken out by domain (legal/medical/policy):

  1. retrieval hit@k — did the chunk from the CORRECT source_file make it into
     the top-k retrieved chunks? This is the retrieval-precision signal RQ1 and
     RQ2 need, measured against ground truth rather than eyeballing.
  2. citation correctness — did the model cite that correct source in its answer?
     (uses the citation parser in src/grounding.py)
  3. answer match — does the generated answer contain the expected_answer text?
     A first, lenient faithfulness signal (substring / token-overlap).
  4. mean grounding score — of the citations the model produced, what fraction
     point to a chunk that was actually retrieved (vs. fabricated)?

Two modes, mirroring experiments 02 and 03:

  * Default: runs the real pipeline against a local Ollama server over the whole
    query set and writes per-query and summary results to experiments/results/.
    These are the numbers reported in the Milestone 4/5 technical report.

  * --selftest: validates the SCORING functions on fixed fixtures with no Ollama
    and no models, so the metric logic is reproducible on any machine.

Note on scope: this harness evaluates retrieval and grounding. It deliberately
does NOT touch the chunking or ingestion modules — it consumes whatever the live
pipeline (src/data_loader.py -> src/chunking.py) produces, so its numbers always
describe the real system.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_CSV = Path(__file__).resolve().parent.parent / "data" / "eval" / "final_query_set.csv"


# ---------------------------------------------------------------------------
# Scoring helpers (pure functions — unit-tested by --selftest, no Ollama)
# ---------------------------------------------------------------------------

def normalize_source(name: str) -> str:
    """Normalize a source filename for comparison.

    The eval set stores URL-encoded filenames (e.g. '...Content%20License...')
    to match the on-disk corpus names, while the pipeline reports the decoded
    name from the filesystem. Decode and lowercase both so they compare equal.
    """
    return urllib.parse.unquote(name).strip().lower()


def retrieval_hit(expected_source: str, retrieved_sources: list[str]) -> bool:
    """True if the expected source document appears among retrieved chunks."""
    want = normalize_source(expected_source)
    return any(normalize_source(s) == want for s in retrieved_sources)


def cited_correct_source(answer: str, expected_source: str) -> bool:
    """True if the model's answer cites the expected source document.

    Uses the same citation parser the grounding layer uses, then compares
    the cited source name (decoded, lowercased) to the expected one.
    """
    from src.grounding import parse_citations

    want = normalize_source(expected_source)
    # parse_citations returns (source, page); source is already lowercased there
    for cited_source, _page in parse_citations(answer):
        if normalize_source(cited_source) in want or want.startswith(normalize_source(cited_source)):
            return True
    return False


def answer_matches(answer: str, expected: str, min_overlap: float = 0.6) -> bool:
    """Lenient check that the answer captures the expected answer.

    Passes if the expected answer appears as a substring (case-insensitive)
    OR if at least `min_overlap` of the expected answer's content words appear
    in the model's answer. Ground-truth answers are short factual spans
    (defined terms, dollar amounts, jurisdictions), so token overlap is a
    reasonable proxy before a stricter judged metric is added in M5.
    """
    a, e = answer.lower(), expected.lower().strip()
    if not e:
        return False
    if e in a:
        return True
    expected_words = [w for w in re.findall(r"[a-z0-9$%.]+", e) if len(w) > 2]
    if not expected_words:
        return False
    hit = sum(1 for w in expected_words if w in a)
    return (hit / len(expected_words)) >= min_overlap


# ---------------------------------------------------------------------------
# Offline self-test of the scoring functions
# ---------------------------------------------------------------------------

def run_selftest() -> None:
    """Validate the scoring functions on fixtures (no Ollama, no models)."""
    # 1. retrieval_hit: URL-encoded vs decoded names must compare equal
    assert retrieval_hit(
        "GopageCorp%20Content%20License.pdf",
        ["gopagecorp content license.pdf", "other.pdf"],
    )
    assert not retrieval_hit("missing.pdf", ["a.pdf", "b.pdf"])

    # 2. cited_correct_source: parses [source, page] and matches decoded name
    ans_good = "Territory is Canada, US and Mexico [GopageCorp Content License.pdf, page 2]."
    assert cited_correct_source(ans_good, "GopageCorp%20Content%20License.pdf")
    ans_wrong = "Territory is Canada [SomeOtherContract.pdf, page 2]."
    assert not cited_correct_source(ans_wrong, "GopageCorp%20Content%20License.pdf")

    # 3. answer_matches: substring and token-overlap
    assert answer_matches("The governing law is Delaware.", "Delaware")
    assert answer_matches(
        "Covers Canada, the United States, and Mexico.",
        "Canada, United States and Mexico",
    )
    assert not answer_matches("The governing law is California.", "Delaware")

    print("PASS  retrieval_hit matches URL-encoded and decoded source names")
    print("PASS  cited_correct_source parses citations and matches the right source")
    print("PASS  answer_matches handles substring and token-overlap cases")
    print("PASS  wrong source / wrong answer correctly score as misses")
    print("\nScoring self-test passed (all fixtures). No Ollama required.")


# ---------------------------------------------------------------------------
# Full evaluation against a live local model
# ---------------------------------------------------------------------------

def load_eval_set() -> list[dict]:
    if not EVAL_CSV.exists():
        raise SystemExit(f"Eval set not found at {EVAL_CSV}.")
    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_eval() -> None:
    """Run the whole query set through the live pipeline and score it."""
    from src.config import get_config
    from src.rag_pipeline import RAGPipeline

    cfg = get_config()
    repo = Path(__file__).resolve().parent.parent
    corpus_dir = repo / "data" / "corpus"

    pipeline = RAGPipeline(cfg)
    print(f"Ingesting corpus from {corpus_dir} (this can take a while)...")
    n_chunks = 0
    for domain_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        n_chunks += pipeline.ingest(domain_dir)
    if n_chunks == 0:
        raise SystemExit("No chunks ingested — is the corpus present under data/corpus/?")
    print(f"Ingested {n_chunks} chunks.\n")

    eval_rows = load_eval_set()
    per_query = []
    by_domain: dict[str, list[dict]] = defaultdict(list)

    for row in eval_rows:
        result = pipeline.ask(row["query"])
        retrieved_sources = [s["source"] for s in result["sources"]]

        rec = {
            "id": row["id"],
            "category": row["category"],
            "match_type": row["match_type"],
            "retrieval_hit": retrieval_hit(row["source_file"], retrieved_sources),
            "cited_correct": cited_correct_source(result["answer"], row["source_file"]),
            "answer_match": answer_matches(result["answer"], row["expected_answer"]),
            "grounding_score": result["grounding"].get("grounding_score"),
        }
        per_query.append(rec)
        by_domain[row["category"]].append(rec)
        mark = "OK " if rec["retrieval_hit"] else "MISS"
        print(f"[{mark}] {row['id']:<4} {row['category']:<8} "
              f"hit={rec['retrieval_hit']} cite={rec['cited_correct']} "
              f"ans={rec['answer_match']}")

    # write per-query CSV
    pq_path = RESULTS_DIR / "eval_per_query.csv"
    with open(pq_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_query[0].keys())
        w.writeheader()
        w.writerows(per_query)

    # aggregate
    def rate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None

    def mean_grounding(rows):
        vals = [r["grounding_score"] for r in rows if r["grounding_score"] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = []
    for domain in ["legal", "medical", "policy"]:
        rows = by_domain[domain]
        if rows:
            summary.append({
                "domain": domain,
                "n": len(rows),
                "retrieval_hit_rate": rate(rows, "retrieval_hit"),
                "citation_correct_rate": rate(rows, "cited_correct"),
                "answer_match_rate": rate(rows, "answer_match"),
                "mean_grounding_score": mean_grounding(rows),
            })
    summary.append({
        "domain": "OVERALL",
        "n": len(per_query),
        "retrieval_hit_rate": rate(per_query, "retrieval_hit"),
        "citation_correct_rate": rate(per_query, "cited_correct"),
        "answer_match_rate": rate(per_query, "answer_match"),
        "mean_grounding_score": mean_grounding(per_query),
    })

    sum_path = RESULTS_DIR / "eval_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    print("\n" + "=" * 78)
    print(f"{'domain':<10}{'n':<4}{'retr.hit':<10}{'cite.correct':<14}"
          f"{'ans.match':<11}{'grounding':<10}")
    for s in summary:
        print(f"{s['domain']:<10}{s['n']:<4}{str(s['retrieval_hit_rate']):<10}"
              f"{str(s['citation_correct_rate']):<14}{str(s['answer_match_rate']):<11}"
              f"{str(s['mean_grounding_score']):<10}")
    print(f"\nWrote {pq_path}\nWrote {sum_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval + grounding evaluation.")
    parser.add_argument("--selftest", action="store_true",
                        help="Validate scoring functions offline (no Ollama).")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        run_eval()


if __name__ == "__main__":
    main()
