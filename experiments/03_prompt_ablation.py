"""Experiment 3 — Prompt-profile ablation, grounding evaluation, and top-k sweep.

    python experiments/03_prompt_ablation.py                 # needs Ollama
    python experiments/03_prompt_ablation.py --selftest      # offline, no Ollama
    python experiments/03_prompt_ablation.py --topk-sweep 1,2,4,6   # needs Ollama

Owner: Christopher Swartz (feature/llm-infrastructure)

This experiment measures the thing the whole project rests on: does the
prompt actually keep the model grounded? It compares prompt profiles on
a small labeled question set and scores four behaviors that a good
private-document assistant must get right:

  - refusal accuracy: on questions the documents DON'T answer, does the
    model correctly refuse instead of hallucinating? (should be high)
  - citation rate: on answerable questions, does the model cite sources?
  - grounding score: of the citations produced, what fraction point to
    a chunk that was actually retrieved (vs. fabricated)?
  - format compliance: of the citations produced, what fraction follow
    the exact "[source, page N]" format the prompt instructs? (Milestone
    4 addition — the M3 run showed grounded-but-misformatted citations
    being conflated with fabrications.)

Three modes:

  * Default: runs each prompt profile through the real pipeline against
    a local Ollama server and writes results to experiments/results/,
    including a raw-answer dump (prompt_ablation_raw_answers.md) so
    citation behavior can be inspected directly rather than inferred
    from parsed tuples. These are the numbers reported in the
    Milestone 4/5 technical report.

  * --topk-sweep: holds the prompt profile fixed (default strict_cited)
    and varies retrieval top_k, completing the second half of RQ2
    ("How does top-k context size affect response accuracy?").

  * --selftest: validates the grounding VERIFIER (src/grounding.py) on
    fixed, hand-written answer fixtures — no model, no Ollama — so the
    scoring logic is reproducible on any machine. This mirrors how
    experiment 02 uses a mock embedding to test retrieval plumbing.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grounding import verify_grounding                     # noqa: E402
from src.vector_store import RetrievedChunk                    # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REFUSAL = "I can't find that in the provided documents."

# Labeled evaluation set over the sample corpus. `answerable=False` marks
# questions whose answer is deliberately absent from the documents, so a
# correct system must refuse.
EVAL_SET = [
    ("What is the monthly rent and when is it due?", True),
    ("What is the late fee for overdue rent?", True),
    ("What medications is the patient currently taking?", True),
    ("Does the patient have any allergies?", True),
    ("How often must backflow prevention assemblies be tested?", True),
    ("Who owns the service lateral past the meter?", True),
    ("What is the patient's blood type?", False),          # not in the record
    ("What is the landlord's home phone number?", False),  # not in the lease
    ("What is the water authority's annual budget?", False),  # not in the policy
]


# ---------------------------------------------------------------------------
# Offline self-test of the grounding verifier
# ---------------------------------------------------------------------------

def run_selftest() -> None:
    """Validate the verifier on fixed fixtures (no model required)."""
    ctx = [
        RetrievedChunk("Rent is $1,850.00 per month, due on the 1st.",
                       "sample_lease_agreement.pdf", 1, "id1", 0.1),
        RetrievedChunk("Allergies: Penicillin - hives. Sulfa - rash.",
                       "sample_medical_record.docx", 1, "id2", 0.2),
    ]

    # 1. Fully grounded answer -> score 1.0, no fabrications
    good = "Rent is $1,850.00, due on the 1st [sample_lease_agreement.pdf, page 1]."
    r = verify_grounding(good, ctx, REFUSAL)
    assert r.grounding_score == 1.0 and not r.unsupported, r.to_dict()
    assert r.format_compliance == 1.0, r.to_dict()

    # 2. Fabricated citation -> caught as unsupported, score < 1.0
    bad = ("The tenant may sublet freely "
           "[sample_lease_agreement.pdf, page 9].")  # page 9 was never retrieved
    r = verify_grounding(bad, ctx, REFUSAL)
    assert r.unsupported and r.grounding_score == 0.0, r.to_dict()

    # 3. Correct refusal -> flagged as refusal, not scored as a claim
    r = verify_grounding(REFUSAL, ctx, REFUSAL)
    assert r.is_refusal and not r.cited, r.to_dict()

    # 4. Mixed answer -> partial score
    mixed = ("Rent is $1,850 [sample_lease_agreement.pdf, page 1] and pets "
             "are banned [sample_lease_agreement.pdf, page 12].")  # p.12 fabricated
    r = verify_grounding(mixed, ctx, REFUSAL)
    assert r.grounding_score == 0.5 and len(r.unsupported) == 1, r.to_dict()

    # --- Milestone 4 fixtures: the exact formats the M3 live run produced ---

    # 5. "source:"-prefixed citation (echoes the excerpt header) -> resolved
    #    as grounded, but flagged as a format-compliance failure.
    echoed = "Testing is annual [source: sample_lease_agreement.pdf, page 1]."
    r = verify_grounding(echoed, ctx, REFUSAL)
    assert r.grounding_score == 1.0 and not r.unsupported, r.to_dict()
    assert len(r.nonstandard) == 1 and r.format_compliance == 0.0, r.to_dict()

    # 6. Excerpt-number citation with a page -> resolved to that excerpt's
    #    source, grounded, flagged nonstandard.
    by_number = "The patient is allergic to penicillin [Excerpt 2, page 1]."
    r = verify_grounding(by_number, ctx, REFUSAL)
    assert r.grounding_score == 1.0 and len(r.nonstandard) == 1, r.to_dict()

    # 7. Bare excerpt citation, no page -> resolved to the excerpt's own
    #    (source, page), grounded, flagged nonstandard.
    bare = "The patient is allergic to penicillin [Excerpt 2]."
    r = verify_grounding(bare, ctx, REFUSAL)
    assert r.grounding_score == 1.0 and len(r.nonstandard) == 1, r.to_dict()

    # 8. Excerpt number that was never shown -> still a fabrication.
    ghost = "Utilities are included [Excerpt 7]."
    r = verify_grounding(ghost, ctx, REFUSAL)
    assert r.grounding_score == 0.0 and r.unsupported, r.to_dict()

    print("PASS  fully grounded answer scores 1.0")
    print("PASS  fabricated citation detected and scored 0.0")
    print("PASS  refusal recognized, not scored as a claim")
    print("PASS  mixed answer scores 0.5 with one fabrication flagged")
    print("PASS  'source:'-prefixed citation resolved, flagged nonstandard")
    print("PASS  excerpt-number citation resolved to its source, flagged nonstandard")
    print("PASS  bare [Excerpt N] resolved, flagged nonstandard")
    print("PASS  out-of-range excerpt number still counted as fabrication")
    print("\nGrounding verifier self-test passed (8/8). No Ollama required.")


# ---------------------------------------------------------------------------
# Shared scoring loop
# ---------------------------------------------------------------------------

def score_eval_set(pipeline, profile: str, top_k: int | None,
                   raw_dump: list[str] | None) -> dict:
    """Run EVAL_SET once under (profile, top_k) and aggregate metrics."""
    pipeline.llm.profile = profile
    n_ans = n_unans = 0
    correct_refusals = cited_answerable = 0
    grounding_scores: list[float] = []
    format_scores: list[float] = []
    latencies: list[float] = []
    models_used: set[str] = set()

    for question, answerable in EVAL_SET:
        hits = pipeline.retrieve(question, top_k=top_k)
        t0 = time.perf_counter()
        result = pipeline.llm.generate_grounded(question, hits)
        latencies.append(time.perf_counter() - t0)
        report = result.grounding
        if pipeline.llm.last_model_used:
            models_used.add(pipeline.llm.last_model_used)

        if raw_dump is not None:
            headers = ", ".join(
                f"{c.source} p.{c.page_number}" for c in result.used_chunks
            )
            raw_dump.append(
                f"### {question}\n\n"
                f"*profile:* `{profile}` | *top_k:* {top_k or 'config'} | "
                f"*model:* `{pipeline.llm.last_model_used}` | "
                f"*answerable:* {answerable}\n\n"
                f"*context shown:* {headers}\n\n"
                f"```\n{result.text}\n```\n\n"
                f"*grounding:* {report.summary_line()}\n"
            )

        if answerable:
            n_ans += 1
            if report.cited:
                cited_answerable += 1
            if report.grounding_score is not None:
                grounding_scores.append(report.grounding_score)
            if report.format_compliance is not None:
                format_scores.append(report.format_compliance)
        else:
            n_unans += 1
            if report.is_refusal:
                correct_refusals += 1

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "profile": profile,
        "top_k": top_k if top_k is not None else "config",
        "model_used": "+".join(sorted(models_used)) or "unknown",
        "refusal_accuracy": round(correct_refusals / n_unans, 2) if n_unans else None,
        "citation_rate": round(cited_answerable / n_ans, 2) if n_ans else None,
        "mean_grounding_score": mean(grounding_scores),
        "mean_format_compliance": mean(format_scores),
        "mean_latency_s": mean(latencies),
    }


def _build_pipeline():
    from src.config import get_config
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(get_config())
    repo = Path(__file__).resolve().parent.parent
    if pipeline.ingest(repo / "data" / "raw") == 0:
        raise SystemExit("No sample docs. Run `python data/make_samples.py` first.")
    return pipeline


def _write_results(rows: list[dict], csv_name: str, raw_dump: list[str],
                   raw_name: str) -> None:
    csv_path = RESULTS_DIR / csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    raw_path = RESULTS_DIR / raw_name
    raw_path.write_text(
        "# Raw generated answers\n\n"
        "Dumped so citation behavior can be inspected directly (Milestone 4, "
        "RQ2 next step 1) instead of inferred from parsed citation tuples.\n\n"
        + "\n---\n\n".join(raw_dump),
        encoding="utf-8",
    )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {raw_path}\n")
    cols = list(rows[0].keys())
    print("  ".join(f"{c:<22}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):<22}" for c in cols))


# ---------------------------------------------------------------------------
# Mode 1: prompt-profile ablation against a live local model
# ---------------------------------------------------------------------------

def run_ablation() -> None:
    """Run each prompt profile through the real pipeline and score it."""
    from src.llm import PROMPT_PROFILES

    pipeline = _build_pipeline()
    raw_dump: list[str] = []
    rows = [score_eval_set(pipeline, profile, None, raw_dump)
            for profile in PROMPT_PROFILES]
    _write_results(rows, "prompt_ablation.csv", raw_dump,
                   "prompt_ablation_raw_answers.md")


# ---------------------------------------------------------------------------
# Mode 2: top-k sweep at a fixed prompt profile (RQ2, second factor)
# ---------------------------------------------------------------------------

def run_topk_sweep(top_ks: list[int], profile: str) -> None:
    """Vary retrieval top_k at a fixed profile and score each setting."""
    pipeline = _build_pipeline()
    raw_dump: list[str] = []
    rows = [score_eval_set(pipeline, profile, k, raw_dump) for k in top_ks]
    _write_results(rows, "topk_sweep.csv", raw_dump,
                   "topk_sweep_raw_answers.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt-profile ablation and top-k sweep.")
    parser.add_argument("--selftest", action="store_true",
                        help="Validate the grounding verifier offline (no Ollama).")
    parser.add_argument("--topk-sweep", metavar="K1,K2,...",
                        help="Comma-separated top_k values to sweep at a fixed "
                             "profile (e.g. 1,2,4,6). Needs Ollama.")
    parser.add_argument("--profile", default="strict_cited",
                        help="Prompt profile for --topk-sweep (default: strict_cited).")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.topk_sweep:
        run_topk_sweep([int(k) for k in args.topk_sweep.split(",")], args.profile)
    else:
        run_ablation()


if __name__ == "__main__":
    main()
