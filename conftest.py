"""Shared pytest fixtures/setup for the WhatUpDoc test suite.

Mirrors the sys.path handling in src/model_runner.py so `from src...`
and `from utils...` imports resolve the same way whether you run
`pytest` from the repo root, from tests/, or via an IDE test runner.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))