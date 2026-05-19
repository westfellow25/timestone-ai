"""OutcomeRecord - realised result of a TimeStone recommendation.

This is the most valuable data the company can accumulate. Every record
ties a prediction (made by TimeStone at engagement time) to a realised
outcome measured 12-24 months later. Over time this becomes a
proprietary "predicted vs actual" dataset that feeds Bayesian
recalibration of the priors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class OutcomeRecord:
    """One realised outcome paired with the original prediction."""
    id: str
    run_id: str                       # which assessment run produced the prediction
    company_name: str
    scenario_id: int
    scenario_name: str
    prediction_date: str              # ISO date of original recommendation
    measurement_date: str             # ISO date of outcome measurement
    months_elapsed: int

    # Prediction (snapshot at recommendation time)
    predicted_mean_npv: float
    predicted_success_probability: float
    predicted_revenue_uplift_pct: float
    predicted_cost_reduction_pct: float
    predicted_investment_usd: float
    predicted_payback_years: float

    # Actual
    actual_revenue_uplift_pct: Optional[float] = None
    actual_cost_reduction_pct: Optional[float] = None
    actual_investment_usd: Optional[float] = None
    actual_status: str = "in_progress"  # in_progress, success, partial, failed, abandoned

    # Context
    decision_taken: str = ""           # what the client actually did
    deviation_notes: str = ""          # why actual differs from predicted
    sources: list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict) -> "OutcomeRecord":
        return cls(**data)
