"""
TimeStone AI — Bayesian Calibration System

Self-improving prediction loop that learns from the gap between
predicted and actual outcomes. Every simulation creates a feedback
signal that makes future simulations more accurate.

This creates a data moat: the more you use TimeStone, the better
it gets — for YOU and for similar companies (via federated priors).

Key concepts:
- Prior beliefs about causal edge strengths (from industry knowledge)
- Likelihood from observed prediction errors
- Posterior = updated beliefs (Bayesian updating)
- Calibration score: how well-calibrated are our confidence intervals?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.causal_graph import CausalEdge, CausalGraph


@dataclass
class PriorBelief:
    """
    Prior distribution over a causal edge's strength.
    Modeled as Normal(mu, sigma^2) for conjugate updating.
    """
    edge_key: str            # "source|target"
    mu: float                # prior mean
    sigma: float             # prior standard deviation
    n_observations: int = 0  # number of updates so far
    last_updated: str = ""

    @property
    def precision(self) -> float:
        """Precision = 1/variance."""
        return 1.0 / (self.sigma ** 2) if self.sigma > 0 else float("inf")


@dataclass
class ObservedOutcome:
    """A single observation: prediction vs. reality."""
    prediction_id: str
    variable_name: str
    predicted_value: float
    actual_value: float
    predicted_confidence_lower: float
    predicted_confidence_upper: float
    timestamp: str
    context: Dict = field(default_factory=dict)

    @property
    def error(self) -> float:
        return self.actual_value - self.predicted_value

    @property
    def absolute_error(self) -> float:
        return abs(self.error)

    @property
    def percentage_error(self) -> float:
        if self.predicted_value == 0:
            return float("inf") if self.actual_value != 0 else 0.0
        return abs(self.error / self.predicted_value) * 100

    @property
    def within_confidence(self) -> bool:
        return self.predicted_confidence_lower <= self.actual_value <= self.predicted_confidence_upper


@dataclass
class CalibrationMetrics:
    """Metrics on how well-calibrated our predictions are."""
    total_predictions: int
    mean_absolute_error: float
    mean_percentage_error: float
    confidence_coverage: float     # % of actuals within predicted CI
    expected_coverage: float       # target (e.g. 0.90 for 90% CI)
    calibration_score: float       # 1.0 = perfectly calibrated
    sharpness: float               # average CI width (narrower = better)
    bias: float                    # systematic over/under-prediction
    brier_score: float             # probability calibration (lower = better)


class BayesianCalibrator:
    """
    Bayesian calibration engine for the TimeStone prediction system.

    Maintains prior beliefs about causal edge strengths, updates them
    as prediction outcomes are observed, and tracks calibration metrics
    to ensure predictions are trustworthy.
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.priors: Dict[str, PriorBelief] = {}
        self.observations: List[ObservedOutcome] = []
        self._initialize_priors()

    def _initialize_priors(self) -> None:
        """Initialize priors from current graph edge strengths."""
        for source, edges in self.graph.edges.items():
            for edge in edges:
                key = f"{edge.source}|{edge.target}"
                self.priors[key] = PriorBelief(
                    edge_key=key,
                    mu=edge.strength,
                    sigma=abs(edge.strength) * 0.3 + 0.1,  # 30% uncertainty + floor
                    n_observations=0,
                )

    def record_outcome(self, outcome: ObservedOutcome) -> None:
        """Record an observed outcome for calibration."""
        self.observations.append(outcome)

    def bayesian_update(
        self,
        edge_key: str,
        observed_strength: float,
        observation_noise: float = 0.1,
    ) -> PriorBelief:
        """
        Perform Bayesian update on a causal edge's strength.

        Uses Normal-Normal conjugate update:
        posterior_mu = (prior_precision * prior_mu + obs_precision * obs) /
                       (prior_precision + obs_precision)
        posterior_precision = prior_precision + obs_precision
        """
        if edge_key not in self.priors:
            self.priors[edge_key] = PriorBelief(
                edge_key=edge_key,
                mu=observed_strength,
                sigma=observation_noise * 2,
            )
            return self.priors[edge_key]

        prior = self.priors[edge_key]

        prior_precision = prior.precision
        obs_precision = 1.0 / (observation_noise ** 2) if observation_noise > 0 else 100.0

        posterior_precision = prior_precision + obs_precision
        posterior_mu = (prior_precision * prior.mu + obs_precision * observed_strength) / posterior_precision
        posterior_sigma = math.sqrt(1.0 / posterior_precision)

        prior.mu = posterior_mu
        prior.sigma = posterior_sigma
        prior.n_observations += 1

        return prior

    def update_graph_from_posteriors(self) -> Dict[str, Tuple[float, float]]:
        """
        Update the causal graph's edge strengths from posterior beliefs.
        Returns {edge_key: (old_strength, new_strength)}.
        """
        changes: Dict[str, Tuple[float, float]] = {}

        for source, edges in self.graph.edges.items():
            for edge in edges:
                key = f"{edge.source}|{edge.target}"
                if key in self.priors:
                    prior = self.priors[key]
                    if prior.n_observations > 0:
                        old = edge.strength
                        edge.strength = prior.mu
                        # Update confidence based on posterior uncertainty
                        edge.confidence = max(0.1, min(1.0,
                            1.0 - prior.sigma / (abs(prior.mu) + 0.01)
                        ))
                        changes[key] = (old, edge.strength)

        return changes

    def calibrate_from_outcomes(
        self,
        outcomes: Optional[List[ObservedOutcome]] = None,
    ) -> CalibrationMetrics:
        """
        Compute calibration metrics from observed outcomes.
        Used to assess and improve prediction quality.
        """
        obs = outcomes or self.observations
        if not obs:
            return CalibrationMetrics(
                total_predictions=0,
                mean_absolute_error=0.0,
                mean_percentage_error=0.0,
                confidence_coverage=0.0,
                expected_coverage=0.90,
                calibration_score=0.0,
                sharpness=0.0,
                bias=0.0,
                brier_score=0.0,
            )

        errors = [o.error for o in obs]
        abs_errors = [o.absolute_error for o in obs]
        pct_errors = [o.percentage_error for o in obs if o.percentage_error != float("inf")]
        within_ci = [1 if o.within_confidence else 0 for o in obs]
        ci_widths = [o.predicted_confidence_upper - o.predicted_confidence_lower for o in obs]

        coverage = sum(within_ci) / len(within_ci)
        expected = 0.90

        # Calibration score: 1 - |coverage - expected| / expected
        calibration_score = max(0.0, 1.0 - abs(coverage - expected) / expected)

        # Brier score for binary calibration (within CI or not)
        brier = sum((expected - w) ** 2 for w in within_ci) / len(within_ci)

        return CalibrationMetrics(
            total_predictions=len(obs),
            mean_absolute_error=float(np.mean(abs_errors)),
            mean_percentage_error=float(np.mean(pct_errors)) if pct_errors else 0.0,
            confidence_coverage=coverage,
            expected_coverage=expected,
            calibration_score=calibration_score,
            sharpness=float(np.mean(ci_widths)),
            bias=float(np.mean(errors)),
            brier_score=brier,
        )

    def adaptive_confidence_adjustment(self) -> float:
        """
        If our confidence intervals are too narrow (coverage < expected),
        widen them. If too wide, narrow them. Returns the adjustment factor.
        """
        metrics = self.calibrate_from_outcomes()
        if metrics.total_predictions < 10:
            return 1.0  # not enough data

        coverage_ratio = metrics.confidence_coverage / metrics.expected_coverage

        if coverage_ratio < 0.9:
            # Under-coverage: widen intervals
            return 1.0 + (1.0 - coverage_ratio) * 0.5
        if coverage_ratio > 1.1:
            # Over-coverage: narrow intervals (more precise)
            return 1.0 - (coverage_ratio - 1.0) * 0.3

        return 1.0

    def compute_edge_importance(self) -> Dict[str, float]:
        """
        Rank edges by how much they contribute to prediction accuracy.
        Edges with high strength and low posterior uncertainty are most important.
        """
        importance: Dict[str, float] = {}
        for key, prior in self.priors.items():
            # Importance = |strength| * confidence (inverse of relative uncertainty)
            relative_certainty = 1.0 - min(1.0, prior.sigma / (abs(prior.mu) + 0.01))
            importance[key] = abs(prior.mu) * relative_certainty * (1 + math.log1p(prior.n_observations))
        return importance

    def suggest_data_collection(self, top_n: int = 5) -> List[Dict]:
        """
        Suggest which measurements would most improve prediction accuracy.
        Prioritizes edges with high strength but high uncertainty.
        """
        suggestions = []
        for key, prior in self.priors.items():
            value_of_information = abs(prior.mu) * prior.sigma
            if prior.n_observations < 3:
                value_of_information *= 2.0  # bonus for under-observed edges

            suggestions.append({
                "edge": key,
                "current_strength": prior.mu,
                "uncertainty": prior.sigma,
                "observations": prior.n_observations,
                "value_of_information": value_of_information,
                "recommendation": f"Collect data on relationship {key.replace('|', ' → ')}",
            })

        suggestions.sort(key=lambda x: x["value_of_information"], reverse=True)
        return suggestions[:top_n]

    # ---- federated prior aggregation ----

    def aggregate_federated_priors(
        self,
        external_priors: List[Dict[str, PriorBelief]],
        privacy_noise: float = 0.05,
    ) -> None:
        """
        Aggregate priors from multiple companies (federated learning).
        Adds differential privacy noise to protect individual company data.

        This is a key network effect: more companies using TimeStone
        means better priors for everyone.
        """
        rng = np.random.default_rng()

        for key in self.priors:
            matching_priors = [
                ep[key] for ep in external_priors
                if key in ep and ep[key].n_observations > 0
            ]

            if not matching_priors:
                continue

            # Precision-weighted average of means
            total_precision = self.priors[key].precision
            weighted_mu = self.priors[key].precision * self.priors[key].mu

            for ext_prior in matching_priors:
                noisy_mu = ext_prior.mu + rng.normal(0, privacy_noise)
                ext_precision = ext_prior.precision * 0.5  # discount external data
                total_precision += ext_precision
                weighted_mu += ext_precision * noisy_mu

            self.priors[key].mu = weighted_mu / total_precision
            self.priors[key].sigma = math.sqrt(1.0 / total_precision)

    # ---- serialization ----

    def to_dict(self) -> Dict:
        return {
            "priors": {
                key: {
                    "mu": p.mu,
                    "sigma": p.sigma,
                    "n_observations": p.n_observations,
                }
                for key, p in self.priors.items()
            },
            "observations": [
                {
                    "prediction_id": o.prediction_id,
                    "variable_name": o.variable_name,
                    "predicted_value": o.predicted_value,
                    "actual_value": o.actual_value,
                    "predicted_confidence_lower": o.predicted_confidence_lower,
                    "predicted_confidence_upper": o.predicted_confidence_upper,
                    "error": o.error,
                    "within_confidence": o.within_confidence,
                }
                for o in self.observations
            ],
            "calibration": self.calibrate_from_outcomes().__dict__
                if self.observations else None,
        }
