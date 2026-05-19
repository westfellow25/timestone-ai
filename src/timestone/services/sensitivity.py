"""Sensitivity (tornado) analysis on a single scenario."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..domain.simulation import SimulationConfig
from .monte_carlo import MonteCarloSimulator


@dataclass
class SensitivityRow:
    parameter: str
    baseline_npv: float
    low_npv: float
    high_npv: float
    swing: float


def _evaluate(scenario: Dict, baseline_revenue: float, baseline_op_costs: float,
              iterations: int = 400, config: Optional[SimulationConfig] = None) -> float:
    sim = MonteCarloSimulator(
        config or SimulationConfig(iterations=iterations), random_seed=42)
    r = sim.simulate_scenario(scenario, baseline_revenue, baseline_op_costs)
    return r.mean_npv


def sensitivity_analysis(scenario: Dict, baseline_revenue: float,
                         baseline_op_costs: float, iterations: int = 400,
                         band: float = 0.30) -> List[SensitivityRow]:
    """Vary each parameter +/- band around baseline; report NPV swing."""
    baseline = _evaluate(scenario, baseline_revenue, baseline_op_costs, iterations)

    rows: List[SensitivityRow] = []

    def vary(name: str, low_mult: float, high_mult: float):
        low = copy.deepcopy(scenario)
        high = copy.deepcopy(scenario)
        if name in ("revenue_increase", "cost_reduction"):
            low["expected_impact"][name] = low["expected_impact"][name] * low_mult
            high["expected_impact"][name] = high["expected_impact"][name] * high_mult
        elif name == "investment_required":
            low["investment_required"] *= low_mult
            high["investment_required"] *= high_mult
        elif name == "implementation_time_months":
            low["implementation_time_months"] = max(1, int(low["implementation_time_months"] * low_mult))
            high["implementation_time_months"] = int(high["implementation_time_months"] * high_mult)
        else:
            return
        low_npv = _evaluate(low, baseline_revenue, baseline_op_costs, iterations)
        high_npv = _evaluate(high, baseline_revenue, baseline_op_costs, iterations)
        rows.append(SensitivityRow(
            parameter=name, baseline_npv=baseline,
            low_npv=low_npv, high_npv=high_npv,
            swing=abs(high_npv - low_npv)))

    vary("revenue_increase", 1 - band, 1 + band)
    vary("cost_reduction", 1 - band, 1 + band)
    vary("investment_required", 1 + band, 1 - band)   # inverted: lower invest = higher NPV
    vary("implementation_time_months", 1 + band, 1 - band)  # inverted

    # Discount rate variation
    cfg_low = SimulationConfig(iterations=iterations, discount_rate=0.08)
    cfg_high = SimulationConfig(iterations=iterations, discount_rate=0.16)
    low_npv = _evaluate(scenario, baseline_revenue, baseline_op_costs, iterations, cfg_high)
    high_npv = _evaluate(scenario, baseline_revenue, baseline_op_costs, iterations, cfg_low)
    rows.append(SensitivityRow(
        parameter="discount_rate", baseline_npv=baseline,
        low_npv=low_npv, high_npv=high_npv, swing=abs(high_npv - low_npv)))

    rows.sort(key=lambda r: r.swing, reverse=True)
    return rows
