"""Local embedding client for WhatUpDoc.

Wraps Ollama's /api/embeddings endpoint (nomic-embed-text, 768-dim).
The privacy guard in utils.helpers refuses any non-localhost host
before the first request is made.

Only the Python standard library is used for HTTP so the dependency
footprint stays small and auditable.
"""

from __future__ import annotations

import json
import urllib.request

from src.config import get_config
from utils.helpers import assert_local_host, get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    """Batch embedding via a locally hosted Ollama server."""

    def __init__(self, config: dict | None = None):
        cfg = config or get_config()
        self.host = cfg["ollama"]["host"].rstrip("/")
        self.model = cfg["ollama"]["embedding_model"]
        self.timeout = cfg["ollama"]["request_timeout_s"]

        if cfg["privacy"]["enforce_offline"]:
            assert_local_host(self.host, cfg["privacy"]["allowed_hosts"])

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Raises RuntimeError if Ollama is down."""
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["embedding"]
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. "
                "Is it running? Start it with `ollama serve` and pull the "
                f"embedding model with `ollama pull {self.model}`."
            ) from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings (sequential; Ollama queues internally)."""
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self.embed_one(text))
            if (i + 1) % 25 == 0:
                logger.info("Embedded %d/%d chunks", i + 1, len(texts))
        return embeddings
