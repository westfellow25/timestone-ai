"""
TimeStone AI — Simulation Tests

Tests for advanced Monte Carlo, regime detection, EVT, and sensitivity analysis.
"""

import numpy as np
import pytest

from src.simulation.advanced_monte_carlo import (
    AdvancedMonteCarloEngine,
    SimulationConfig,
    SamplingMethod,
)
from src.simulation.regime_detector import (
    ExtremeValueAnalyzer,
    MarketRegime,
    RegimeDetector,
)
from src.simulation.sensitivity_analyzer import SensitivityAnalyzer


# ---- Fixtures ----

@pytest.fixture
def sample_scenario():
    return {
        "id": "test-001",
        "name": "Test Dynamic Pricing",
        "type": "pricing_optimization",
        "expected_impact": {
            "revenue_increase": 0.15,
            "cost_reduction": 0.05,
        },
        "investment_required": 3_000_000,
        "implementation_time_months": 9,
        "risk_level": "medium",
    }


@pytest.fixture
def sample_scenarios():
    return [
        {
            "id": f"test-{i:03d}",
            "name": f"Scenario {i}",
            "type": "digital_transformation",
            "expected_impact": {
                "revenue_increase": 0.10 + 0.02 * i,
                "cost_reduction": 0.03 + 0.01 * i,
            },
            "investment_required": 1_000_000 + 500_000 * i,
            "implementation_time_months": 6 + i * 2,
            "risk_level": ["low", "medium", "high"][i % 3],
        }
        for i in range(5)
    ]


# ---- Advanced Monte Carlo Tests ----

class TestAdvancedMonteCarlo:
    def test_basic_simulation(self, sample_scenario):
        config = SimulationConfig(iterations=1000, seed=42)
        engine = AdvancedMonteCarloEngine(config)
        result = engine.simulate_scenario(sample_scenario, baseline_revenue=500e6)

        assert result.scenario_name == "Test Dynamic Pricing"
        assert 0 <= result.success_probability <= 1
        assert result.ci_lower < result.ci_upper
        assert result.diagnostics is not None

    def test_latin_hypercube(self, sample_scenario):
        config = SimulationConfig(
            iterations=2000, method=SamplingMethod.LATIN_HYPERCUBE, seed=42,
        )
        engine = AdvancedMonteCarloEngine(config)
        result = engine.simulate_scenario(sample_scenario, 500e6)
        assert result.diagnostics.iterations_run > 0

    def test_stratified_sampling(self, sample_scenario):
        config = SimulationConfig(
            iterations=2000, method=SamplingMethod.STRATIFIED, seed=42,
        )
        engine = AdvancedMonteCarloEngine(config)
        result = engine.simulate_scenario(sample_scenario, 500e6)
        assert result.diagnostics.iterations_run > 0

    def test_antithetic_variates(self, sample_scenario):
        config = SimulationConfig(
            iterations=1000, antithetic=True, seed=42,
        )
        engine = AdvancedMonteCarloEngine(config)
        result = engine.simulate_scenario(sample_scenario, 500e6)
        # Antithetic doubles the effective samples
        assert result.diagnostics.iterations_run == 2000

    def test_risk_metrics(self, sample_scenario):
        config = SimulationConfig(iterations=5000, seed=42)
        engine = AdvancedMonteCarloEngine(config)
        result = engine.simulate_scenario(sample_scenario, 500e6)

        assert result.value_at_risk_95 <= result.mean_roi
        assert result.conditional_var_95 <= result.value_at_risk_95
        assert result.max_drawdown >= 0
        assert len(result.percentiles) == 7

    def test_portfolio_simulation(self, sample_scenarios):
        config = SimulationConfig(iterations=1000, seed=42)
        engine = AdvancedMonteCarloEngine(config)
        portfolio = engine.simulate_portfolio(sample_scenarios, 500e6)

        assert len(portfolio["individual_results"]) == 5
        assert len(portfolio["ranked_by_sharpe"]) == 5
        if portfolio["portfolio_metrics"]:
            assert "diversification_benefit" in portfolio["portfolio_metrics"]

    def test_high_risk_scenario(self):
        scenario = {
            "id": "risk-test",
            "name": "High Risk Test",
            "expected_impact": {"revenue_increase": 0.30, "cost_reduction": 0.10},
            "investment_required": 10_000_000,
            "implementation_time_months": 24,
            "risk_level": "high",
        }
        config = SimulationConfig(iterations=5000, seed=42)
        engine = AdvancedMonteCarloEngine(config)
        result = engine.simulate_scenario(scenario, 500e6)

        # High risk should have wider confidence interval
        assert result.ci_upper - result.ci_lower > 0.3

    def test_convergence_diagnostics(self, sample_scenario):
        config = SimulationConfig(iterations=10000, seed=42)
        engine = AdvancedMonteCarloEngine(config)
        result = engine.simulate_scenario(sample_scenario, 500e6)

        diag = result.diagnostics
        assert diag.std_error > 0
        assert diag.effective_sample_size > 0


# ---- Regime Detection Tests ----

class TestRegimeDetector:
    def test_default_initialization(self):
        detector = RegimeDetector()
        assert detector.state.current_regime == MarketRegime.STABLE

    def test_detect_growth_regime(self):
        detector = RegimeDetector()
        returns = np.random.normal(0.08, 0.10, 24)  # growth-like returns
        state = detector.detect_regime(returns, window=12)
        assert state.confidence > 0

    def test_detect_crisis_regime(self):
        detector = RegimeDetector()
        returns = np.random.normal(-0.15, 0.35, 24)  # crisis-like returns
        state = detector.detect_regime(returns, window=12)
        assert state.current_regime in list(MarketRegime)

    def test_generate_regime_sequence(self):
        detector = RegimeDetector()
        sequence = detector.generate_regime_sequence(36, seed=42)
        assert len(sequence) == 36
        assert all(r in [m.value for m in MarketRegime] for r in sequence)

    def test_regime_adjusted_parameters(self):
        detector = RegimeDetector()
        params = detector.get_regime_adjusted_parameters(0.10, 0.05, MarketRegime.CRISIS)
        assert params["adjusted_volatility"] > 0.10  # crisis increases volatility
        assert params["correlation_shift"] > 0


# ---- Extreme Value Theory Tests ----

class TestExtremeValueAnalyzer:
    def test_fit_gpd(self):
        rng = np.random.default_rng(42)
        # Generate heavy-tailed losses
        losses = np.abs(rng.standard_t(df=3, size=1000))

        analyzer = ExtremeValueAnalyzer(threshold_percentile=95)
        result = analyzer.fit(losses)

        assert result["n_exceedances"] > 0
        assert result["gpd_shape"] is not None
        assert result["gpd_scale"] > 0

    def test_var_evt(self):
        rng = np.random.default_rng(42)
        losses = np.abs(rng.standard_t(df=3, size=1000))

        analyzer = ExtremeValueAnalyzer(threshold_percentile=90)
        analyzer.fit(losses)

        var_99 = analyzer.var_evt(0.99)
        var_95 = analyzer.var_evt(0.95)
        assert var_99 > var_95  # 99% VaR should be more extreme

    def test_cvar_evt(self):
        rng = np.random.default_rng(42)
        losses = np.abs(rng.standard_t(df=3, size=1000))

        analyzer = ExtremeValueAnalyzer(threshold_percentile=90)
        analyzer.fit(losses)

        cvar = analyzer.cvar_evt(0.99)
        var = analyzer.var_evt(0.99)
        assert cvar >= var  # CVaR >= VaR by definition

    def test_tail_probability(self):
        rng = np.random.default_rng(42)
        losses = np.abs(rng.standard_t(df=3, size=1000))

        analyzer = ExtremeValueAnalyzer(threshold_percentile=90)
        analyzer.fit(losses)

        p_low = analyzer.tail_probability(1.0)
        p_high = analyzer.tail_probability(5.0)
        assert p_low > p_high  # higher loss = lower probability

    def test_stress_test(self):
        rng = np.random.default_rng(42)
        losses = np.abs(rng.standard_t(df=3, size=1000))

        analyzer = ExtremeValueAnalyzer(threshold_percentile=90)
        analyzer.fit(losses)

        results = analyzer.stress_test([
            {"market_crash": 3.0},
            {"market_crash": 5.0, "liquidity_crisis": 2.0},
        ])
        assert len(results) == 2
        assert results[1]["probability"] < results[0]["probability"]


# ---- Sensitivity Analysis Tests ----

class TestSensitivityAnalyzer:
    def test_sobol_analysis(self):
        analyzer = SensitivityAnalyzer(seed=42)

        def model(x):
            return x[0] * 2 + x[1] * 0.5 + x[0] * x[1] * 0.3

        result = analyzer.sobol_analysis(
            model_fn=model,
            param_bounds={"a": (0, 1), "b": (0, 1)},
            n_samples=512,
        )

        assert "a" in result.first_order_indices
        assert result.first_order_indices["a"] > result.first_order_indices["b"]
        assert len(result.rankings) == 2

    def test_morris_screening(self):
        analyzer = SensitivityAnalyzer(seed=42)

        def model(x):
            return x[0] ** 2 + 0.1 * x[1] + x[2] * 0.5

        result = analyzer.morris_screening(
            model_fn=model,
            param_bounds={"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)},
            n_trajectories=30,
        )

        assert len(result.first_order_indices) == 3
        assert len(result.rankings) == 3

    def test_tornado_analysis(self):
        analyzer = SensitivityAnalyzer(seed=42)

        def model(params):
            return params["price"] * params["volume"] - params["cost"]

        result = analyzer.tornado_analysis(
            model_fn=model,
            base_params={"price": 100, "volume": 1000, "cost": 50000},
            param_ranges={
                "price": (80, 120),
                "volume": (800, 1200),
                "cost": (40000, 60000),
            },
        )

        assert len(result.factors) == 3
        assert result.base_value == 50000  # 100*1000 - 50000
        assert result.swing[0] >= result.swing[-1]  # sorted by swing
