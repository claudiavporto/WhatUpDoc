"""Shared utilities for WhatUpDoc.

Includes the privacy enforcement check that backs the project's core
guarantee: the pipeline only ever talks to a local Ollama server.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from urllib.parse import urlparse


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def assert_local_host(url: str, allowed_hosts: list[str]) -> None:
    """Raise if a configured endpoint is not a local address.

    This is the enforcement point for the privacy guarantee. Every
    component that opens a network connection (embeddings, LLM) calls
    this before its first request, so a misconfigured remote URL fails
    loudly instead of silently sending document text off-machine.

    Args:
        url: The endpoint the component intends to call.
        allowed_hosts: Hostnames permitted by configs/config.yaml.

    Raises:
        RuntimeError: if the hostname is not in the allow-list.
    """
    host = urlparse(url).hostname
    if host not in allowed_hosts:
        raise RuntimeError(
            f"Privacy guard: refusing to connect to non-local host '{host}'. "
            f"Allowed hosts: {allowed_hosts}. Check configs/config.yaml."
        )


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    """Context manager that logs wall-clock time for a block.

    Used in experiments to measure ingestion, retrieval, and
    generation latency (Research Question 3).
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"{label}: {elapsed:.2f}s"
    (logger.info if logger else print)(msg)


def truncate(text: str, limit: int = 300) -> str:
    """Shorten text for logging/preview without breaking mid-word."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"
