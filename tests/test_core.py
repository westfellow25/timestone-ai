"""
TimeStone AI — Core Engine Tests

Comprehensive tests for the causal graph, temporal engine,
company genome, and Bayesian calibration systems.
"""

import math

import numpy as np
import pytest

from src.core.causal_graph import (
    CausalEdge,
    CausalGraph,
    CausalVariable,
    EdgeType,
    VariableType,
)
from src.core.temporal_engine import (
    SeasonalPattern,
    TemporalEngine,
    TimeResolution,
)
from src.core.company_genome import (
    CompanyGenome,
    GenomeDimension,
    GENOME_FACTORS,
)
from src.core.bayesian_calibration import (
    BayesianCalibrator,
    ObservedOutcome,
)


# ---- Fixtures ----

@pytest.fixture
def simple_graph() -> CausalGraph:
    """A small 4-node causal graph for testing."""
    g = CausalGraph()
    g.add_variable(CausalVariable("revenue", VariableType.FINANCIAL, 100.0, "M$", min_value=0))
    g.add_variable(CausalVariable("marketing_spend", VariableType.FINANCIAL, 10.0, "M$", min_value=0, volatility=0.0))
    g.add_variable(CausalVariable("customer_count", VariableType.CUSTOMER, 1000.0, "count", min_value=0, volatility=0.0))
    g.add_variable(CausalVariable("satisfaction", VariableType.CUSTOMER, 0.7, "score", 0, 1.0, volatility=0.0))

    g.add_edge(CausalEdge("marketing_spend", "customer_count", strength=50.0, confidence=0.9))
    g.add_edge(CausalEdge("customer_count", "revenue", strength=0.1, confidence=0.95))
    g.add_edge(CausalEdge("satisfaction", "customer_count", strength=200.0, lag_periods=1, confidence=0.8))
    return g


@pytest.fixture
def sample_genome() -> CompanyGenome:
    """A sample company genome with several factors set."""
    genome = CompanyGenome("TestCorp", "technology")
    genome.set_factor("revenue_growth_rate", 0.75)
    genome.set_factor("profit_margin", 0.60)
    genome.set_factor("cash_flow_stability", 0.80)
    genome.set_factor("digital_infrastructure_score", 0.85)
    genome.set_factor("data_readiness", 0.70)
    genome.set_factor("talent_density", 0.65)
    genome.set_factor("change_management_capability", 0.55)
    genome.set_factor("leadership_quality", 0.70)
    return genome


# ---- Causal Graph Tests ----

class TestCausalGraph:
    def test_add_variable(self, simple_graph):
        assert "revenue" in simple_graph.variables
        assert len(simple_graph.variables) == 4

    def test_topological_sort(self, simple_graph):
        order = simple_graph.topological_sort()
        assert len(order) == 4
        # marketing_spend and satisfaction should come before customer_count
        assert order.index("marketing_spend") < order.index("customer_count")
        assert order.index("customer_count") < order.index("revenue")

    def test_cycle_detection(self, simple_graph):
        with pytest.raises(ValueError, match="cycle"):
            simple_graph.add_edge(CausalEdge("revenue", "marketing_spend", strength=0.1))
            simple_graph.add_edge(CausalEdge("marketing_spend", "revenue", strength=0.1))

    def test_do_intervention_deterministic(self, simple_graph):
        # Increase marketing spend from 10 to 20
        trajectories = simple_graph.do_intervention(
            {"marketing_spend": 20.0},
            time_horizon=6,
            stochastic=False,
            seed=42,
        )
        # Marketing spend should be fixed at 20
        np.testing.assert_allclose(trajectories["marketing_spend"], 20.0)
        # Customer count should increase due to marketing
        assert trajectories["customer_count"][-1] > 1000.0

    def test_causal_path_finding(self, simple_graph):
        paths = simple_graph.find_all_causal_paths("marketing_spend", "revenue")
        assert len(paths) >= 1
        assert paths[0] == ["marketing_spend", "customer_count", "revenue"]

    def test_total_causal_effect(self, simple_graph):
        effect = simple_graph.total_causal_effect("marketing_spend", "revenue")
        # marketing → customer_count (50 * 0.9) → revenue (0.1 * 0.95)
        expected = 50.0 * 0.9 * 0.1 * 0.95
        assert abs(effect - expected) < 0.5

    def test_influence_scores(self, simple_graph):
        scores = simple_graph.get_influence_scores()
        # marketing_spend and satisfaction should have non-zero influence
        assert scores["marketing_spend"] > 0
        assert scores["revenue"] == 0  # revenue has no outgoing edges

    def test_vulnerability_scores(self, simple_graph):
        scores = simple_graph.get_vulnerability_scores()
        assert scores["revenue"] > 0
        assert scores["marketing_spend"] == 0  # no incoming edges

    def test_edge_types(self):
        g = CausalGraph()
        g.add_variable(CausalVariable("x", VariableType.FINANCIAL, 100.0))
        g.add_variable(CausalVariable("y", VariableType.FINANCIAL, 50.0))

        edge = CausalEdge("x", "y", strength=0.5, edge_type=EdgeType.LINEAR)
        assert edge.compute_effect(10.0, 100.0) == 5.0

        edge_log = CausalEdge("x", "y", strength=10.0, edge_type=EdgeType.LOGARITHMIC)
        effect = edge_log.compute_effect(10.0, 100.0)
        assert effect > 0

        edge_thresh = CausalEdge("x", "y", strength=100.0, edge_type=EdgeType.THRESHOLD, threshold_value=110.0)
        assert edge_thresh.compute_effect(15.0, 100.0) == 100.0
        assert edge_thresh.compute_effect(5.0, 100.0) == 0.0

    def test_serialization_roundtrip(self, simple_graph):
        data = simple_graph.to_dict()
        restored = CausalGraph.from_dict(data)
        assert len(restored.variables) == len(simple_graph.variables)
        assert restored.topological_sort() == simple_graph.topological_sort()

    def test_counterfactual(self, simple_graph):
        # Create factual history
        trajectories = simple_graph.do_intervention(
            {}, time_horizon=12, stochastic=False, seed=42,
        )
        factual = {name: arr for name, arr in trajectories.items()}

        # Counterfactual: what if we had doubled marketing at t=3?
        cf = simple_graph.counterfactual(
            factual_history=factual,
            counterfactual_interventions={"marketing_spend": 20.0},
            intervention_time=3,
            time_horizon=12,
            seed=42,
        )
        # Counterfactual marketing should be 20 from t=3 onward
        np.testing.assert_allclose(cf["marketing_spend"][3:], 20.0)


# ---- Temporal Engine Tests ----

class TestTemporalEngine:
    def test_basic_simulation(self, simple_graph):
        engine = TemporalEngine(simple_graph)
        results = engine.simulate(
            time_horizon=24,
            num_paths=10,
            seed=42,
        )
        assert "revenue" in results
        assert results["revenue"].shape == (10, 24)

    def test_seasonal_pattern(self, simple_graph):
        engine = TemporalEngine(simple_graph)
        engine.add_seasonal_pattern(SeasonalPattern(
            variable_name="revenue",
            period=12,
            amplitude=5.0,
        ))
        results = engine.simulate(time_horizon=24, num_paths=1, seed=42)
        # Revenue should have periodic variation
        rev = results["revenue"][0]
        assert np.std(rev) > 0

    def test_impulse_response(self, simple_graph):
        engine = TemporalEngine(simple_graph)
        irf = engine.compute_impulse_response(
            "marketing_spend", shock_magnitude=5.0,
            time_horizon=12, num_paths=100, seed=42,
        )
        # Marketing spend IRF should show the shock
        assert abs(irf["marketing_spend"][1]) > 0

    def test_aggregation(self, simple_graph):
        engine = TemporalEngine(simple_graph, base_resolution=TimeResolution.MONTHLY)
        results = engine.simulate(time_horizon=24, num_paths=5, seed=42)
        quarterly = engine.aggregate_to_resolution(results, TimeResolution.QUARTERLY)
        for name, data in quarterly.items():
            assert data.shape[1] == 8  # 24 months / 3


# ---- Company Genome Tests ----

class TestCompanyGenome:
    def test_set_and_get_factor(self, sample_genome):
        assert sample_genome.factors["revenue_growth_rate"].value == 0.75

    def test_invalid_factor(self):
        genome = CompanyGenome("Test", "test")
        with pytest.raises(ValueError, match="Unknown"):
            genome.set_factor("nonexistent_factor", 0.5)

    def test_dimension_score(self, sample_genome):
        score = sample_genome.get_dimension_score(GenomeDimension.FINANCIAL_HEALTH)
        assert 0.0 <= score.score <= 1.0

    def test_overall_score(self, sample_genome):
        score = sample_genome.get_overall_score()
        assert 0.0 <= score <= 1.0

    def test_genome_vector(self, sample_genome):
        vec = sample_genome.get_genome_vector()
        assert vec.shape == (48,)
        assert vec[0] == 0.75  # revenue_growth_rate

    def test_transformation_readiness(self, sample_genome):
        readiness = sample_genome.transformation_readiness("digital_transformation")
        assert "readiness_score" in readiness
        assert "readiness_grade" in readiness
        assert readiness["readiness_grade"] in ("A", "B", "C", "D", "F")

    def test_genome_distance(self, sample_genome):
        other = CompanyGenome("OtherCorp", "technology")
        other.set_factor("revenue_growth_rate", 0.30)
        other.set_factor("profit_margin", 0.40)

        dist_cosine = sample_genome.genome_distance(other, "cosine")
        dist_euclidean = sample_genome.genome_distance(other, "euclidean")
        assert dist_cosine >= 0
        assert dist_euclidean >= 0

    def test_capability_gap_analysis(self, sample_genome):
        target = CompanyGenome("TargetCorp", "technology")
        for dim in GenomeDimension:
            for factor in GENOME_FACTORS[dim]:
                target.set_factor(factor, 0.90)

        gaps = sample_genome.capability_gap_analysis(target)
        assert gaps["overall_gap_score"] >= 0
        assert len(gaps["critical_gaps"]) > 0

    def test_serialization(self, sample_genome):
        data = sample_genome.to_dict()
        restored = CompanyGenome.from_dict(data)
        assert restored.company_name == sample_genome.company_name
        np.testing.assert_array_almost_equal(
            restored.get_genome_vector(),
            sample_genome.get_genome_vector(),
        )


# ---- Bayesian Calibration Tests ----

class TestBayesianCalibration:
    def test_initialization(self, simple_graph):
        calibrator = BayesianCalibrator(simple_graph)
        assert len(calibrator.priors) > 0

    def test_bayesian_update(self, simple_graph):
        calibrator = BayesianCalibrator(simple_graph)
        key = "marketing_spend|customer_count"
        prior_mu = calibrator.priors[key].mu

        # Observe that the true strength is slightly higher
        calibrator.bayesian_update(key, 55.0, observation_noise=5.0)
        posterior_mu = calibrator.priors[key].mu

        # Posterior should move toward observation
        assert abs(posterior_mu - 55.0) < abs(prior_mu - 55.0)
        assert calibrator.priors[key].n_observations == 1

    def test_calibration_metrics(self, simple_graph):
        calibrator = BayesianCalibrator(simple_graph)

        for i in range(20):
            calibrator.record_outcome(ObservedOutcome(
                prediction_id=f"pred-{i}",
                variable_name="revenue",
                predicted_value=100.0 + i,
                actual_value=100.0 + i + np.random.normal(0, 2),
                predicted_confidence_lower=95.0 + i,
                predicted_confidence_upper=105.0 + i,
                timestamp="2026-01-01",
            ))

        metrics = calibrator.calibrate_from_outcomes()
        assert metrics.total_predictions == 20
        assert metrics.mean_absolute_error >= 0

    def test_edge_importance(self, simple_graph):
        calibrator = BayesianCalibrator(simple_graph)
        importance = calibrator.compute_edge_importance()
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())

    def test_suggest_data_collection(self, simple_graph):
        calibrator = BayesianCalibrator(simple_graph)
        suggestions = calibrator.suggest_data_collection(top_n=3)
        assert len(suggestions) <= 3
        assert all("value_of_information" in s for s in suggestions)

    def test_confidence_adjustment(self, simple_graph):
        calibrator = BayesianCalibrator(simple_graph)
        adjustment = calibrator.adaptive_confidence_adjustment()
        assert adjustment == 1.0  # not enough data yet
