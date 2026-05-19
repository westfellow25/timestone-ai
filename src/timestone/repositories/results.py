"""Stores simulation outputs as versioned runs."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..infrastructure.paths import RUNS_DIR, ensure_dirs


class ResultsRepository:
    def __init__(self, runs_dir: Optional[Path] = None):
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def new_run(self, company_name: str) -> Path:
        """Create a fresh run directory and return its path."""
        ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        safe = company_name.lower().replace(" ", "_").replace("/", "_")[:30]
        nonce = uuid.uuid4().hex[:6]
        run_dir = self.runs_dir / f"{ts}_{safe}_{nonce}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_scenarios(self, run_dir: Path, scenarios_payload: Dict) -> None:
        (run_dir / "scenarios.json").write_text(
            json.dumps(scenarios_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_simulation(self, run_dir: Path, simulation_payload: Dict) -> None:
        (run_dir / "simulation.json").write_text(
            json.dumps(simulation_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_report(self, run_dir: Path, report_payload: Dict) -> None:
        (run_dir / "report.json").write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_runs(self) -> List[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted([p for p in self.runs_dir.iterdir() if p.is_dir()], reverse=True)

    def load_run(self, run_dir: Path) -> Dict:
        out = {}
        for name in ("scenarios", "simulation", "report"):
            f = run_dir / f"{name}.json"
            if f.exists():
                out[name] = json.loads(f.read_text())
        out["run_id"] = run_dir.name
        return out
