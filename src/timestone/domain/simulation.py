"""Simulation domain models - config and results, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimulationConfig:
    """Configurable financial assumptions for Monte Carlo simulation."""
    iterations: int = 1000
    horizon_years: int = 5
    discount_rate: float = 0.12
    adoption_curve: tuple = (0.40, 0.70, 0.95, 1.0, 1.0)

    revenue_variance: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.15, "medium": 0.25, "high": 0.40})
    cost_variance: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.10, "medium": 0.20, "high": 0.35})
    cost_overrun: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.10, "medium": 0.25, "high": 0.50})
    delay_factor: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.20, "medium": 0.40, "high": 0.80})

    market_downturn_prob: float = 0.08
    market_downturn_impact: float = -0.30
    competitive_response_prob: float = 0.15
    competitive_response_impact: float = -0.20
    execution_failure_prob: float = 0.05


@dataclass
class SimulationResult:
    """Aggregate result of Monte Carlo simulation for one scenario."""
    scenario_id: int
    scenario_name: str
    mean_npv: float
    median_npv: float
    mean_roi: float
    median_roi: float
    std_dev_roi: float
    confidence_90_lower: float
    confidence_90_upper: float
    success_probability: float
    high_success_probability: float
    risk_score: float
    payback_years_median: float
    iterations: int

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "mean_npv": self.mean_npv,
            "median_npv": self.median_npv,
            "mean_roi": self.mean_roi,
            "median_roi": self.median_roi,
            "std_dev_roi": self.std_dev_roi,
            "confidence_90_lower": self.confidence_90_lower,
            "confidence_90_upper": self.confidence_90_upper,
            "success_probability": self.success_probability,
            "high_success_probability": self.high_success_probability,
            "risk_score": self.risk_score,
            "payback_years_median": self.payback_years_median,
            "iterations": self.iterations,
        }
