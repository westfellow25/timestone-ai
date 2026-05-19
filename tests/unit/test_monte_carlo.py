"""Unit tests for the Monte Carlo simulator service."""
import math
import pytest

from timestone.domain.simulation import SimulationConfig, SimulationResult
from timestone.services.monte_carlo import MonteCarloSimulator


def test_simulate_scenario_returns_result_type(ktz_baseline, low_risk_scenario):
    sim = MonteCarloSimulator(SimulationConfig(iterations=200), random_seed=42)
    result = sim.simulate_scenario(low_risk_scenario,
                                    ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    assert isinstance(result, SimulationResult)
    assert result.scenario_id == 1
    assert result.iterations == 200


def test_success_probability_in_unit_interval(ktz_baseline, realistic_scenarios):
    sim = MonteCarloSimulator(SimulationConfig(iterations=500), random_seed=42)
    for s in realistic_scenarios:
        r = sim.simulate_scenario(s, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
        assert 0.0 <= r.success_probability <= 1.0
        assert 0.0 <= r.high_success_probability <= 1.0


def test_high_risk_has_more_variance_than_low_risk(ktz_baseline):
    scenario_low = {
        "id": 1, "name": "Test",
        "expected_impact": {"revenue_increase": 0.02, "cost_reduction": 0.02},
        "investment_required": 2_000_000.0, "implementation_time_months": 6,
        "risk_level": "low",
    }
    scenario_high = {**scenario_low, "id": 2, "risk_level": "high"}
    sim_low = MonteCarloSimulator(SimulationConfig(iterations=3000), random_seed=42)
    sim_high = MonteCarloSimulator(SimulationConfig(iterations=3000), random_seed=42)
    low = sim_low.simulate_scenario(scenario_low, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    high = sim_high.simulate_scenario(scenario_high, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    cv_low = low.std_dev_roi / abs(low.mean_roi) if low.mean_roi != 0 else 0
    cv_high = high.std_dev_roi / abs(high.mean_roi) if high.mean_roi != 0 else 0
    assert cv_high > cv_low


def test_not_every_scenario_is_perfect(ktz_baseline):
    bad_scenario = {
        "id": 99, "name": "Risky moonshot",
        "expected_impact": {"revenue_increase": 0.005, "cost_reduction": 0.005},
        "investment_required": 50_000_000.0, "implementation_time_months": 24,
        "risk_level": "high",
    }
    sim = MonteCarloSimulator(SimulationConfig(iterations=2000), random_seed=42)
    r = sim.simulate_scenario(bad_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    assert r.success_probability < 0.99


def test_cost_reduction_applies_to_operating_costs_not_revenue():
    scenario = {
        "id": 100, "name": "Pure cost play",
        "expected_impact": {"revenue_increase": 0.0, "cost_reduction": 0.05},
        "investment_required": 1.0, "implementation_time_months": 1, "risk_level": "low",
    }
    cfg = SimulationConfig(
        iterations=3000, execution_failure_prob=0.0,
        market_downturn_prob=0.0, competitive_response_prob=0.0,
        cost_variance={"low": 0.0, "medium": 0.0, "high": 0.0},
        revenue_variance={"low": 0.0, "medium": 0.0, "high": 0.0},
        cost_overrun={"low": 0.0, "medium": 0.0, "high": 0.0},
        delay_factor={"low": 0.0, "medium": 0.0, "high": 0.0},
    )
    big = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(scenario, 500_000_000.0, 100_000_000.0)
    small = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(scenario, 500_000_000.0, 10_000_000.0)
    ratio = big.mean_npv / small.mean_npv
    assert 9.5 < ratio < 10.5


def test_npv_falls_with_higher_discount_rate(ktz_baseline, low_risk_scenario):
    cfg_low = SimulationConfig(iterations=1000, discount_rate=0.05,
                               execution_failure_prob=0.0, market_downturn_prob=0.0,
                               competitive_response_prob=0.0)
    cfg_high = SimulationConfig(iterations=1000, discount_rate=0.25,
                                execution_failure_prob=0.0, market_downturn_prob=0.0,
                                competitive_response_prob=0.0)
    low = MonteCarloSimulator(cfg_low, random_seed=42).simulate_scenario(
        low_risk_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    high = MonteCarloSimulator(cfg_high, random_seed=42).simulate_scenario(
        low_risk_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    assert high.mean_npv < low.mean_npv


def test_deterministic_with_same_seed(ktz_baseline, medium_risk_scenario):
    sim_a = MonteCarloSimulator(SimulationConfig(iterations=500), random_seed=42)
    sim_b = MonteCarloSimulator(SimulationConfig(iterations=500), random_seed=42)
    a = sim_a.simulate_scenario(medium_risk_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    b = sim_b.simulate_scenario(medium_risk_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    assert a.mean_npv == pytest.approx(b.mean_npv)


def test_npv_computation_pure():
    import numpy as np
    npv = MonteCarloSimulator._compute_npv(
        year_zero_cost=-100.0,
        annual_benefits=np.array([50.0, 50.0, 50.0]), discount_rate=0.10)
    expected = -100.0 + 50.0 / 1.10 + 50.0 / 1.21 + 50.0 / 1.331
    assert npv == pytest.approx(expected, rel=1e-6)


def test_empirical_failure_rate_overrides_default(ktz_baseline):
    base = {
        "id": 1, "name": "Test",
        "expected_impact": {"revenue_increase": 0.04, "cost_reduction": 0.04},
        "investment_required": 5_000_000.0, "implementation_time_months": 12,
        "risk_level": "medium",
    }
    risky = {**base, "empirical_prior": {
        "revenue_uplift": {"n": 10, "mean": 0.04, "std": 0.05,
                           "p10": -0.02, "p50": 0.04, "p90": 0.10},
        "failure_rate": 0.55,
    }}
    cfg = SimulationConfig(iterations=2000)
    no_prior = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        base, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    with_prior = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        risky, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    assert with_prior.success_probability < no_prior.success_probability


def test_empirical_failure_rate_clipped():
    base = {
        "id": 1, "name": "Test",
        "expected_impact": {"revenue_increase": 0.04, "cost_reduction": 0.04},
        "investment_required": 5_000_000.0, "implementation_time_months": 12,
        "risk_level": "medium",
        "empirical_prior": {
            "revenue_uplift": {"n": 10, "mean": 0.04, "std": 0.05,
                               "p10": -0.02, "p50": 0.04, "p90": 0.10},
            "failure_rate": 0.99,
        },
    }
    cfg = SimulationConfig(iterations=2000)
    res = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        base, 500_000_000, 450_000_000)
    assert res.success_probability > 0.20
