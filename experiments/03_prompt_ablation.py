"""Experiment 3 — Prompt-profile ablation and grounding evaluation.

    python experiments/03_prompt_ablation.py            # needs Ollama
    python experiments/03_prompt_ablation.py --selftest  # offline, no Ollama

Owner: Christopher Swartz (feature/llm-infrastructure)

This experiment measures the thing the whole project rests on: does the
prompt actually keep the model grounded? It compares prompt profiles on
a small labeled question set and scores three behaviors that a good
private-document assistant must get right:

  - refusal accuracy: on questions the documents DON'T answer, does the
    model correctly refuse instead of hallucinating? (should be high)
  - citation rate: on answerable questions, does the model cite sources?
  - grounding score: of the citations produced, what fraction point to
    a chunk that was actually retrieved (vs. fabricated)?

Two modes:

  * Default: runs each prompt profile through the real pipeline against
    a local Ollama server and writes results to experiments/results/.
    These are the numbers reported in the Milestone 4/5 technical report.

  * --selftest: validates the grounding VERIFIER (src/grounding.py) on
    fixed, hand-written answer fixtures — no model, no Ollama — so the
    scoring logic is reproducible on any machine. This mirrors how
    experiment 02 uses a mock embedding to test retrieval plumbing.
"""

from __future__ import annotations

import argparse
import csv
import sys
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

    print("PASS  fully grounded answer scores 1.0")
    print("PASS  fabricated citation detected and scored 0.0")
    print("PASS  refusal recognized, not scored as a claim")
    print("PASS  mixed answer scores 0.5 with one fabrication flagged")
    print("\nGrounding verifier self-test passed (4/4). No Ollama required.")


# ---------------------------------------------------------------------------
# Full ablation against a live local model
# ---------------------------------------------------------------------------

def run_ablation() -> None:
    """Run each prompt profile through the real pipeline and score it."""
    from src.config import get_config
    from src.llm import PROMPT_PROFILES
    from src.rag_pipeline import RAGPipeline

    cfg = get_config()
    repo = Path(__file__).resolve().parent.parent

    pipeline = RAGPipeline(cfg)
    if pipeline.ingest(repo / "data" / "raw") == 0:
        raise SystemExit("No sample docs. Run `python data/make_samples.py` first.")

    rows = []
    for profile in PROMPT_PROFILES:
        pipeline.llm.profile = profile
        n_ans = n_unans = 0
        correct_refusals = cited_answerable = 0
        grounding_scores = []

        for question, answerable in EVAL_SET:
            hits = pipeline.retrieve(question)
            result = pipeline.llm.generate_grounded(question, hits)
            report = result.grounding

            if answerable:
                n_ans += 1
                if report.cited:
                    cited_answerable += 1
                if report.grounding_score is not None:
                    grounding_scores.append(report.grounding_score)
            else:
                n_unans += 1
                if report.is_refusal:
                    correct_refusals += 1

        rows.append({
            "profile": profile,
            "refusal_accuracy": round(correct_refusals / n_unans, 2) if n_unans else None,
            "citation_rate": round(cited_answerable / n_ans, 2) if n_ans else None,
            "mean_grounding_score": (
                round(sum(grounding_scores) / len(grounding_scores), 2)
                if grounding_scores else None
            ),
        })

    csv_path = RESULTS_DIR / "prompt_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {csv_path}\n")
    print(f"{'profile':<14}{'refusal_acc':<13}{'citation_rate':<15}{'grounding':<10}")
    for r in rows:
        print(f"{r['profile']:<14}{str(r['refusal_accuracy']):<13}"
              f"{str(r['citation_rate']):<15}{str(r['mean_grounding_score']):<10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-profile ablation.")
    parser.add_argument("--selftest", action="store_true",
                        help="Validate the grounding verifier offline (no Ollama).")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        run_ablation()


if __name__ == "__main__":
    main()
