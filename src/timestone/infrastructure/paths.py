"""Centralized filesystem path resolution."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
TWINS_DIR = DATA_DIR / "twins"
CASE_LIBRARY_PATH = DATA_DIR / "case_library.json"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
OUTCOMES_DIR = RESULTS_DIR / "outcomes"


def ensure_dirs() -> None:
    """Create all results directories on demand (idempotent)."""
    for d in (RESULTS_DIR, RUNS_DIR, OUTCOMES_DIR):
        d.mkdir(parents=True, exist_ok=True)
