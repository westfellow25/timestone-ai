"""Infrastructure - cross-cutting tech concerns (config, logging, paths, LLM)."""
from .paths import REPO_ROOT, DATA_DIR, TWINS_DIR, CASE_LIBRARY_PATH, RESULTS_DIR, RUNS_DIR, OUTCOMES_DIR, ensure_dirs

__all__ = ["REPO_ROOT", "DATA_DIR", "TWINS_DIR", "CASE_LIBRARY_PATH",
           "RESULTS_DIR", "RUNS_DIR", "OUTCOMES_DIR", "ensure_dirs"]
