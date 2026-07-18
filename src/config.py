"""Configuration loader for WhatUpDoc.

Reads configs/config.yaml once and exposes it as a nested dict.
Every other module imports get_config() instead of hard-coding values,
so hyperparameters and model choices live in exactly one place.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

# Repo root = one directory above src/
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


@functools.lru_cache(maxsize=1)
def get_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load and cache the YAML config.

    Args:
        path: Optional alternate config path (used by tests/experiments).

    Returns:
        Nested dict mirroring configs/config.yaml.

    Raises:
        FileNotFoundError: if the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            "Run from the repo root or pass an explicit path."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a repo-relative path (e.g. data/processed/chroma) to absolute."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
