"""Recommendation and AssessmentReport - the user-facing output of TimeStone."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Recommendation:
    """One ranked recommendation, packaged for a CFO/CEO."""
    rank: int
    scenario_id: int
    scenario_name: str
    headline: str                       # e.g. "STRONG RECOMMENDATION"
    success_probability: float
    mean_npv: float
    mean_roi: float
    payback_years: float
    based_on_cases: List[str]
    empirical_failure_rate: Optional[float] = None
    p10_revenue_uplift: Optional[float] = None
    p90_revenue_uplift: Optional[float] = None
    risk_level: str = "medium"
    description: str = ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class AssessmentReport:
    """Output of assess_company use case."""
    run_id: str
    company_name: str
    generated_at: str
    config_summary: Dict
    top_recommendations: List[Recommendation]
    total_scenarios: int
    failure_rate_among_scenarios: float
    case_library_size: int
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "company_name": self.company_name,
            "generated_at": self.generated_at,
            "config_summary": self.config_summary,
            "top_recommendations": [r.to_dict() for r in self.top_recommendations],
            "total_scenarios": self.total_scenarios,
            "failure_rate_among_scenarios": self.failure_rate_among_scenarios,
            "case_library_size": self.case_library_size,
            "notes": self.notes,
        }
