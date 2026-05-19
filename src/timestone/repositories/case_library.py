"""Loads and saves the TransformationCase corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..domain.case import TransformationCase
from ..infrastructure.paths import CASE_LIBRARY_PATH


class CaseLibraryRepository:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else CASE_LIBRARY_PATH

    def load_all(self) -> List[TransformationCase]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [TransformationCase.from_dict(c) for c in data["cases"]]

    def save_all(self, cases: List[TransformationCase], description: str = "") -> None:
        # We don't have a clean to_dict on TransformationCase; serialize manually
        payload = {
            "schema_version": "1.0",
            "description": description or "Case library of real transformations",
            "cases": [self._case_to_dict(c) for c in cases],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _case_to_dict(c: TransformationCase) -> dict:
        return {
            "id": c.id,
            "company": c.company,
            "industry": c.industry,
            "industry_tags": c.industry_tags,
            "geography": c.geography,
            "company_size": {
                "revenue_usd": c.revenue_usd, "employees": c.employees,
                "revenue_bucket": c.revenue_bucket,
            },
            "transformation": {
                "type": c.transformation_type, "subtype": c.transformation_subtype,
                "description": c.description, "start_year": c.start_year,
                "planned_duration_months": c.planned_duration_months,
                "actual_duration_months": c.actual_duration_months,
                "status": c.status, "vendor": c.vendor,
            },
            "financials": {
                "promised_investment_usd": c.promised_investment_usd,
                "actual_investment_usd": c.actual_investment_usd,
                "writeoff_usd": c.writeoff_usd,
                "promised_revenue_uplift_pct": c.promised_revenue_uplift_pct,
                "actual_revenue_uplift_pct": c.actual_revenue_uplift_pct,
                "promised_cost_reduction_pct": c.promised_cost_reduction_pct,
                "actual_cost_reduction_pct": c.actual_cost_reduction_pct,
            },
            "failure_modes": c.failure_modes,
            "success_factors": c.success_factors,
            "sources": c.sources,
            "tacit_notes": c.tacit_notes,
        }
