"""Single-command demo for WhatUpDoc.

    python src/model_runner.py

Ingests the sample documents in data/raw/, runs a batch of
representative questions through the full RAG pipeline, prints each
grounded answer, and saves everything to outputs/samples.txt.

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

# Representative questions spanning the three sample domains
DEMO_QUESTIONS = [
    "What is the monthly rent and when is it due?",
    "What happens if the tenant pays rent late?",
    "What medications is the patient currently taking?",
    "Does the patient have any allergies?",
    "How often must backflow prevention assemblies be tested?",
    "Who owns the service lateral according to the utility policy?",
    "What is the CEO's salary?",  # deliberately unanswerable -> tests refusal
]


def main() -> int:
    cfg = get_config()
    data_dir = REPO_ROOT / "data" / "raw"
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    try:
        pipeline = RAGPipeline(cfg)
        n = pipeline.ingest(data_dir)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    if n == 0:
        logger.error("No documents found in %s — add PDFs/DOCX/TXT and rerun.", data_dir)
        return 1

    lines = [
        f"WhatUpDoc sample outputs — {datetime.now():%Y-%m-%d %H:%M}",
        f"Model: {cfg['ollama']['llm_model']} | Chunking: {cfg['chunking']['strategy']} "
        f"| top_k: {cfg['retrieval']['top_k']}",
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
