"""Local LLM client and prompt engineering for WhatUpDoc.

Owner: Christopher Swartz (feature/llm-infrastructure)

Three responsibilities:

1. LLM infrastructure — talk to a locally hosted model through Ollama's
   /api/generate endpoint, with automatic fallback from LLaMA 3 8B to
   Mistral 7B if the primary model has not been pulled, and the same
   localhost-only privacy guard used by the embedding client.

2. Prompt engineering — build the grounded prompt that turns retrieved
   chunks into a citable answer. The system prompt enforces three rules:
     - answer ONLY from the provided context,
     - cite every claim as [source, page N],
     - refuse with a fixed marker rather than guess when the context
       does not contain the answer.

3. Grounding verification — before generating, pack retrieved chunks to
   fit the model's context window (src/grounding.py); after generating,
   verify that every citation the model produced points to a chunk that
   was actually retrieved, and attach a measurable grounding score to
   the result. This turns "trust the model" into "verify the model."

Prompt profiles are named so experiments can A/B them from config
without touching code (generation.prompt_profile in config.yaml).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from dataclasses import dataclass

from src.config import get_config
from src.grounding import GroundingReport, pack_context, verify_grounding
from src.vector_store import RetrievedChunk
from utils.helpers import assert_local_host, get_logger

logger = get_logger(__name__)

# The exact string the model must emit when the answer is not in context.
# Centralized here so the prompt, the verifier, and the experiments all
# agree on one canonical refusal marker.
REFUSAL_MARKER = "I can't find that in the provided documents."

# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

# Profile 1 — default. Optimized for verifiability: every sentence in the
# answer must be traceable to a bracketed citation the user can check.
STRICT_CITED_SYSTEM_PROMPT = """\
You are WhatUpDoc, a document analysis assistant that answers questions \
using ONLY the context excerpts provided below. You never use outside \
knowledge, and you never guess.

Rules:
1. Base every statement strictly on the CONTEXT section.
2. After each claim, cite its origin in brackets: [source, page N]. \
Use the exact source name and page number shown in the excerpt header.
3. If the context does not contain the information needed to answer, \
reply exactly: "I can't find that in the provided documents." (verbatim, \
nothing else). Do not speculate or fill gaps with general knowledge.
4. If excerpts conflict, present both versions with their citations \
and note the conflict.
5. Quote critical language (dollar amounts, dates, defined terms, \
dosages) verbatim inside quotation marks.
6. Keep answers concise and factual. Do not add disclaimers, opinions, \
or advice beyond what the documents state.
"""

# Profile 2 — shorter answers for quick lookups; same grounding rules.
CONCISE_SYSTEM_PROMPT = """\
You answer questions using ONLY the context excerpts below. Answer in \
1-3 sentences, cite as [source, page N], and if the answer is not in \
the context reply exactly: "I can't find that in the provided documents."
"""

PROMPT_PROFILES = {
    "strict_cited": STRICT_CITED_SYSTEM_PROMPT,
    "concise": CONCISE_SYSTEM_PROMPT,
}


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, source-labelled excerpts.

    The header line of each excerpt is what the model is instructed to
    cite, so the format here and the citation rule in the system prompt
    must stay in sync.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"--- Excerpt {i} (source: {c.source}, page {c.page_number}) ---\n{c.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk], profile: str = "strict_cited") -> tuple[str, str]:
    """Assemble (system_prompt, user_prompt) for a grounded answer.

    Args:
        question: The user's natural-language question.
        chunks: Retrieval hits from the vector store, best first.
        profile: Key into PROMPT_PROFILES.

    Returns:
        Tuple of (system prompt, user prompt) ready for the LLM.
    """
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown prompt profile '{profile}'. Choose from {list(PROMPT_PROFILES)}")

    user_prompt = (
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (with [source, page N] citations):"
    )
    return PROMPT_PROFILES[profile], user_prompt


# ---------------------------------------------------------------------------
# LLM infrastructure
# ---------------------------------------------------------------------------


class OllamaLLM:
    """Client for a locally hosted generative model via Ollama."""

    def __init__(self, config: dict | None = None):
        cfg = config or get_config()
        o, g = cfg["ollama"], cfg["generation"]

        self.host = o["host"].rstrip("/")
        self.model = o["llm_model"]
        self.fallback_model = o.get("fallback_llm_model")
        self.timeout = o["request_timeout_s"]
        self.temperature = g["temperature"]
        self.max_tokens = g["max_tokens"]
        self.profile = g["prompt_profile"]
        self.context_token_budget = g.get("context_token_budget", 6000)
        self.chars_per_token = g.get("chars_per_token_estimate", 4)

        if cfg["privacy"]["enforce_offline"]:
            assert_local_host(self.host, cfg["privacy"]["allowed_hosts"])

    # -- low-level call ----------------------------------------------------

    def _generate(self, model: str, system: str, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["response"].strip()

    # -- public API ---------------------------------------------------------

    def _generate_with_fallback(self, system: str, prompt: str) -> str:
        """Generate, falling back to the secondary model on HTTP 404."""
        try:
            return self._generate(self.model, system, prompt)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and self.fallback_model:
                logger.warning(
                    "Model '%s' not found; falling back to '%s'. "
                    "Pull the primary with `ollama pull %s`.",
                    self.model, self.fallback_model, self.model,
                )
                return self._generate(self.fallback_model, system, prompt)
            raise
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Start it with "
                f"`ollama serve` and pull a model with `ollama pull {self.model}`."
            ) from exc

    def generate_grounded(self, question: str, chunks: list[RetrievedChunk]) -> "GroundedAnswer":
        """Generate a cited answer and verify it against the context.

        Pipeline:
          1. Pack retrieved chunks to fit the model's context window,
             reserving room for the system prompt, question, and answer.
          2. Generate (with model fallback).
          3. Verify every citation the model produced against the chunks
             actually used, producing a measurable grounding score.

        Returns:
            A GroundedAnswer bundling the text, the grounding report, and
            the chunks the answer was built from.
        """
        system_prompt = PROMPT_PROFILES[self.profile]
        reserved = estimate_reserved(system_prompt, question, self.max_tokens, self.chars_per_token)
        used_chunks = pack_context(
            chunks,
            token_budget=self.context_token_budget,
            reserved_tokens=reserved,
            chars_per_token=self.chars_per_token,
        )
        if len(used_chunks) < len(chunks):
            logger.info("Context budget: using %d of %d retrieved chunks",
                        len(used_chunks), len(chunks))

        system, prompt = build_prompt(question, used_chunks, self.profile)
        text = self._generate_with_fallback(system, prompt)
        report = verify_grounding(text, used_chunks, REFUSAL_MARKER)

        if report.unsupported:
            logger.warning("Answer contains %d fabricated citation(s): %s",
                           len(report.unsupported), report.unsupported)

        return GroundedAnswer(text=text, grounding=report, used_chunks=used_chunks)

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Backward-compatible wrapper returning just the answer text."""
        return self.generate_grounded(question, chunks).text


def estimate_reserved(system: str, question: str, max_answer_tokens: int, chars_per_token: int) -> int:
    """Tokens to hold back from the context budget for non-context content."""
    from src.grounding import estimate_tokens

    overhead = estimate_tokens(system, chars_per_token) + estimate_tokens(question, chars_per_token)
    return overhead + max_answer_tokens + 64  # +64 for prompt scaffolding


@dataclass
class GroundedAnswer:
    """An answer plus everything needed to trust it."""

    text: str
    grounding: GroundingReport
    used_chunks: list[RetrievedChunk]
