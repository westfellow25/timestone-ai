"""Append-only repository of realised outcomes from past recommendations.

This is the most valuable persistence in TimeStone. Every record links
a prediction to a real measured outcome and becomes training data for
Bayesian recalibration of priors. Do not allow deletion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..domain.outcome import OutcomeRecord
from ..infrastructure.paths import OUTCOMES_DIR, ensure_dirs


class OutcomesRepository:
    def __init__(self, outcomes_dir: Optional[Path] = None):
        self.dir = Path(outcomes_dir) if outcomes_dir else OUTCOMES_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: OutcomeRecord) -> Path:
        path = self.dir / f"{record.id}.json"
        if path.exists():
            raise FileExistsError(
                f"Outcome {record.id} already recorded; outcomes are append-only. "
                f"To correct, append a new record citing this one in deviation_notes."
            )
        path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def list_all(self) -> List[OutcomeRecord]:
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("*.json")):
            out.append(OutcomeRecord.from_dict(json.loads(p.read_text())))
        return out

    def by_company(self, company_name: str) -> List[OutcomeRecord]:
        return [o for o in self.list_all() if o.company_name == company_name]
