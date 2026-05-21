"""Per-tenant run log — minimal scaffolding for the SaaS layer.

Appends every successful /assess call to ``results/tenant_runs/{tenant}.json``
so the customer dashboard can list "your last 10 assessments" without
giving them access to runs belonging to other tenants.

Production would back this with Postgres + row-level security.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ...infrastructure.paths import RESULTS_DIR


@dataclass
class RunSummary:
    run_id: str
    tenant_id: str
    company_name: str
    generated_at: str
    top_scenario: str
    top_success_p: float
    top_npv_musd: float
    headline: str

    def to_dict(self) -> Dict:
        return asdict(self)


class TenantRunLog:
    """Append-only per-tenant run log, JSON on disk.

    Reads are cheap (whole file); writes are append+rewrite. Fine up to
    a few hundred runs per tenant — beyond that we'd swap for SQLite.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or (RESULTS_DIR / "tenant_runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, tenant_id: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in tenant_id)
        return self.base_dir / f"{safe}.json"

    def append(self, summary: RunSummary) -> None:
        p = self._path(summary.tenant_id)
        runs: List[Dict] = []
        if p.exists():
            try:
                runs = json.loads(p.read_text()).get("runs", [])
            except json.JSONDecodeError:
                runs = []
        runs.append(summary.to_dict())
        # Cap at 200 most recent per tenant.
        runs = runs[-200:]
        p.write_text(json.dumps({
            "tenant_id": summary.tenant_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runs": runs,
        }, indent=2))

    def list(self, tenant_id: str, limit: int = 20) -> List[Dict]:
        p = self._path(tenant_id)
        if not p.exists():
            return []
        try:
            runs = json.loads(p.read_text()).get("runs", [])
        except json.JSONDecodeError:
            return []
        return list(reversed(runs))[:limit]
