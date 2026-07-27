"""Single-command demo for WhatUpDoc.

    python src/model_runner.py

Ingests two sources and runs a batch of representative questions
through the full RAG pipeline:

  1. data/raw/ -- the small, hand-authored sample fixtures (one lease,
     one medical record, one utility policy). Fast to ingest, and every
     answer can be verified by eye against the source file.
  2. data/corpus/{legal,medical,policy}/ -- a capped sample of the
     larger real-world corpus, so the demo also exercises retrieval at
     realistic chunk counts and document lengths.

Prints each grounded answer and saves everything to outputs/samples.txt.

Requires a running local Ollama server (see README setup). If Ollama
is unreachable, the script exits with instructions instead of a stack
trace.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# allow `python src/model_runner.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, get_config          # noqa: E402
from src.rag_pipeline import RAGPipeline              # noqa: E402
from utils.helpers import get_logger                  # noqa: E402

logger = get_logger(__name__)

# Real dataset: three domain subfolders under data/corpus/, matching the
# three sample-question domains below (legal, medical, policy).
CORPUS_DOMAINS = ["legal", "medical", "policy"]

# This is a fast, single-command DEMO -- not a full-corpus embedding job.
# Note: Milestone 4's spec calls for "5-10 samples"; this demo currently
# runs 13 (6 data/raw questions + 6 data/corpus questions + 1 refusal
# check) because both full question sets were kept intentionally. Trim
# back to one set, or fewer questions per domain, if the 5-10 count is
# a hard requirement rather than a guideline.
#
# A full run of legal/ alone produced 3,516 chunks -- at the measured
# ~2.5s/chunk local embedding rate, that's ~2.5 hours for one domain.
# Capping data/corpus/ ingestion here keeps the demo runnable in
# minutes; exhaustive full-corpus embedding belongs in
# experiments/04_precision_recall_eval.py (--max-files-per-category).
# 150 chunks/domain -> ~450 chunks total -> roughly 19 minutes at 2.5s/chunk.
MAX_CHUNKS_PER_DOMAIN = 150

_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

# All six original questions, grounded in the small hand-authored
# fixtures under data/raw/ (sample_lease_agreement.pdf,
# sample_medical_record.docx, sample_utility_policy.txt) -- answers are
# checkable by eye against the source file in seconds.
RAW_QUESTIONS = [
    "What is the monthly rent and when is it due?",
    "What happens if the tenant pays rent late?",
    "What medications is the patient currently taking?",
    "Does the patient have any allergies?",
    "How often must backflow prevention assemblies be tested?",
    "Who owns the service lateral?",
]

# All six questions grounded in the larger real corpus under
# data/corpus/{legal,medical,policy}/ (the Xencor-Aimmune License,
# Development and Commercialization Agreement; clinical_note_001.docx
# and clinical_note_005.docx; the EPA Lead and Copper Rule Minor
# Revisions final rule, Federal Register Vol. 65 No. 8).
CORPUS_QUESTIONS = [
    "What is the upfront cash payment Aimmune must pay Xencor under the license agreement?",
    "How many days does the Breaching Party have to cure a payment default before the agreement can be terminated?",
    "What dose of tetrabenazine was the patient started on for tardive dystonia?",
    "What was the patient's vitamin D level and what treatment was prescribed?",
    "When does the Lead and Copper Rule Minor Revisions final rule take effect?",
    "By what date must states adopt the LCRMR to maintain primacy?",
]

DEMO_QUESTIONS = RAW_QUESTIONS + CORPUS_QUESTIONS + [
    "What is the CEO's salary?",  # deliberately unanswerable -> tests refusal
]


def _ingest_raw(pipeline: RAGPipeline) -> int:
    """Ingest the small hand-authored fixtures under data/raw/, if present.

    Unlike the capped data/corpus/ ingestion below, this ingests the
    whole folder unconditionally -- it's three short files, not a
    multi-thousand-chunk corpus.

    Returns:
        Total chunks ingested from data/raw/, or 0 if the folder is
        missing/empty.
    """
    raw_dir = REPO_ROOT / "data" / "raw"
    if not raw_dir.is_dir():
        logger.warning("Skipping missing data/raw/ (expected sample fixtures)")
        return 0
    n = pipeline.ingest(raw_dir)
    logger.info("Ingested %d chunk(s) from data/raw/", n)
    return n


def _ingest_corpus(pipeline: RAGPipeline) -> int:
    """Ingest a bounded sample of each domain folder under data/corpus/.

    Skips any domain folder that doesn't exist (logging a warning)
    rather than failing the whole run over one missing/misnamed folder.
    Stops adding new files to a domain once MAX_CHUNKS_PER_DOMAIN chunks
    have been ingested from it (a file already in progress still
    finishes, so the true total may exceed the cap slightly).

    Returns:
        Total chunks ingested across all domains found.
    """
    corpus_root = REPO_ROOT / "data" / "corpus"
    total = 0
    for domain in CORPUS_DOMAINS:
        domain_dir = corpus_root / domain
        if not domain_dir.is_dir():
            logger.warning("Skipping missing domain folder: %s", domain_dir)
            continue

        files = sorted(
            f for f in domain_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if not files:
            logger.warning("No supported documents found in %s", domain_dir)
            continue

        domain_total = 0
        files_used = 0
        for f in files:
            if domain_total >= MAX_CHUNKS_PER_DOMAIN:
                break
            domain_total += pipeline.ingest(f)
            files_used += 1

        logger.info(
            "Ingested %d chunk(s) from %d/%d file(s) in %s/ (demo cap: %d chunks/domain)",
            domain_total, files_used, len(files), domain, MAX_CHUNKS_PER_DOMAIN,
        )
        total += domain_total
    return total


def main() -> int:
    cfg = get_config()
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    try:
        pipeline = RAGPipeline(cfg)
        n_raw = _ingest_raw(pipeline)
        n_corpus = _ingest_corpus(pipeline)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    n = n_raw + n_corpus
    if n == 0:
        logger.error(
            "No documents found under %s or %s — add sample files and rerun.",
            REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "corpus",
        )
        return 1

    lines = [
        f"WhatUpDoc sample outputs — {datetime.now():%Y-%m-%d %H:%M}",
        f"Model: {cfg['ollama']['llm_model']} | Chunking: {cfg['chunking']['strategy']} "
        f"| top_k: {cfg['retrieval']['top_k']}",
        f"data/raw: {n_raw} chunk(s) | "
        f"data/corpus/{{{','.join(CORPUS_DOMAINS)}}}: {n_corpus} chunk(s)",
        "=" * 72,
    ]

    for q in DEMO_QUESTIONS:
        print(f"\nQ: {q}")
        result = pipeline.ask(q)
        print(f"A: {result['answer']}")
        print(f"   [{result['grounding_summary']}]")
        lines += [
            f"\nQ: {q}",
            f"A: {result['answer']}",
            f"Grounding: {result['grounding_summary']}",
            "Retrieved sources:",
        ]
        lines += [
            f"  - {s['source']} p.{s['page_number']} (distance {s['distance']})"
            for s in result["sources"]
        ]

    out_path = out_dir / "samples.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %d sample outputs to %s", len(DEMO_QUESTIONS), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())