"""
TimeStone AI — Regime Detection & Extreme Value Theory

Markets don't stay in one state. This module detects regime changes
(growth → stagnation → crisis) and models tail risks using Extreme
Value Theory for black swan event quantification.

Key capabilities:
1. Hidden Markov Model for regime detection (growth/stagnation/crisis)
2. Generalized Pareto Distribution for tail risk modeling
3. Regime-conditional simulation parameters
4. Crisis propagation modeling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class MarketRegime(Enum):
    BOOM = "boom"
    GROWTH = "growth"
    STABLE = "stable"
    STAGNATION = "stagnation"
    CRISIS = "crisis"


@dataclass
class RegimeParameters:
    """Parameters characterizing a market regime."""
    regime: MarketRegime
    mean_return: float
    volatility: float
    correlation_shift: float    # how much correlations increase (crisis = +0.3)
    duration_mean: int          # expected duration in periods
    duration_std: int
    transition_probs: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeState:
    """Current regime and its history."""
    current_regime: MarketRegime
    regime_start_period: int
    regime_duration: int
    confidence: float
    history: List[Tuple[int, MarketRegime]] = field(default_factory=list)


# Default regime parameters calibrated from historical market data
DEFAULT_REGIME_PARAMS = {
    MarketRegime.BOOM: RegimeParameters(
        regime=MarketRegime.BOOM,
        mean_return=0.15,
        volatility=0.12,
        correlation_shift=-0.1,
        duration_mean=8,
        duration_std=4,
        transition_probs={"growth": 0.50, "stable": 0.30, "stagnation": 0.15, "crisis": 0.05},
    ),
    MarketRegime.GROWTH: RegimeParameters(
        regime=MarketRegime.GROWTH,
        mean_return=0.08,
        volatility=0.10,
        correlation_shift=-0.05,
        duration_mean=12,
        duration_std=6,
        transition_probs={"boom": 0.15, "stable": 0.45, "stagnation": 0.30, "crisis": 0.10},
    ),
    MarketRegime.STABLE: RegimeParameters(
        regime=MarketRegime.STABLE,
        mean_return=0.04,
        volatility=0.08,
        correlation_shift=0.0,
        duration_mean=18,
        duration_std=8,
        transition_probs={"boom": 0.05, "growth": 0.30, "stagnation": 0.45, "crisis": 0.20},
    ),
    MarketRegime.STAGNATION: RegimeParameters(
        regime=MarketRegime.STAGNATION,
        mean_return=-0.02,
        volatility=0.15,
        correlation_shift=0.15,
        duration_mean=10,
        duration_std=5,
        transition_probs={"boom": 0.02, "growth": 0.18, "stable": 0.30, "crisis": 0.50},
    ),
    MarketRegime.CRISIS: RegimeParameters(
        regime=MarketRegime.CRISIS,
        mean_return=-0.15,
        volatility=0.35,
        correlation_shift=0.40,
        duration_mean=4,
        duration_std=3,
        transition_probs={"boom": 0.01, "growth": 0.09, "stable": 0.20, "stagnation": 0.70},
    ),
}


class RegimeDetector:
    """
    Hidden Markov Model-based regime detection.

    Uses observed volatility, returns, and correlation patterns
    to infer the current market regime and predict transitions.
    """

    def __init__(
        self,
        regime_params: Optional[Dict[MarketRegime, RegimeParameters]] = None,
    ):
        self.regime_params = regime_params or DEFAULT_REGIME_PARAMS
        self.state = RegimeState(
            current_regime=MarketRegime.STABLE,
            regime_start_period=0,
            regime_duration=0,
            confidence=0.5,
        )
        self._regime_list = list(MarketRegime)

    def detect_regime(
        self,
        returns: np.ndarray,
        window: int = 12,
    ) -> RegimeState:
        """
        Detect current regime from observed return data.

        Uses a simplified HMM with emission probabilities based on
        rolling volatility and return levels.
        """
        if len(returns) < window:
            return self.state

        recent = returns[-window:]
        mean_ret = float(np.mean(recent))
        vol = float(np.std(recent))

        # Compute log-likelihoods for each regime
        log_likes: Dict[MarketRegime, float] = {}
        for regime, params in self.regime_params.items():
            # Gaussian emission probability
            z = (mean_ret - params.mean_return) / (params.volatility + 1e-9)
            ret_ll = -0.5 * z ** 2 - np.log(params.volatility + 1e-9)

            z_vol = (vol - params.volatility) / (params.volatility * 0.3 + 1e-9)
            vol_ll = -0.5 * z_vol ** 2

            log_likes[regime] = ret_ll + vol_ll

        # Posterior probabilities (with uniform prior + transition)
        transition_key = self.state.current_regime
        tp = self.regime_params[transition_key].transition_probs

        max_ll = max(log_likes.values())
        posteriors = {}
        for regime in MarketRegime:
            prior = tp.get(regime.value, 0.1)
            likelihood = np.exp(log_likes[regime] - max_ll)
            posteriors[regime] = prior * likelihood

        total = sum(posteriors.values())
        for regime in posteriors:
            posteriors[regime] /= total

        # Most likely regime
        best_regime = max(posteriors, key=posteriors.get)
        confidence = posteriors[best_regime]

        # Update state
        if best_regime != self.state.current_regime:
            self.state.history.append(
                (self.state.regime_start_period, self.state.current_regime)
            )
            self.state.current_regime = best_regime
            self.state.regime_start_period = len(returns)
            self.state.regime_duration = 0
        else:
            self.state.regime_duration += 1

        self.state.confidence = confidence
        return self.state

    def generate_regime_sequence(
        self,
        periods: int,
        start_regime: Optional[MarketRegime] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate a plausible regime sequence for simulation."""
        rng = np.random.default_rng(seed)
        current = start_regime or self.state.current_regime
        sequence = []
        remaining_duration = 0

        for _ in range(periods):
            if remaining_duration <= 0:
                # Transition
                params = self.regime_params[current]
                tp = params.transition_probs
                regimes = [MarketRegime(r) for r in tp.keys()]
                probs = [tp[r.value] for r in regimes]
                prob_sum = sum(probs)
                probs = [p / prob_sum for p in probs]

                current = rng.choice(regimes, p=probs)
                current_params = self.regime_params[current]
                remaining_duration = max(1, int(rng.normal(
                    current_params.duration_mean, current_params.duration_std
                )))

            sequence.append(current.value)
            remaining_duration -= 1

        return sequence

    def get_regime_adjusted_parameters(
        self,
        base_volatility: float,
        base_mean: float,
        regime: Optional[MarketRegime] = None,
    ) -> Dict[str, float]:
        """Get simulation parameters adjusted for current regime."""
        r = regime or self.state.current_regime
        params = self.regime_params[r]

        return {
            "adjusted_mean": base_mean + params.mean_return,
            "adjusted_volatility": base_volatility * (1 + params.volatility),
            "correlation_shift": params.correlation_shift,
            "regime": r.value,
        }


class ExtremeValueAnalyzer:
    """
    Extreme Value Theory (EVT) for tail risk quantification.

    Uses the Peaks-Over-Threshold (POT) method with Generalized Pareto
    Distribution (GPD) to model extreme losses that normal distributions
    systematically underestimate.
    """

    def __init__(self, threshold_percentile: float = 95.0):
        self.threshold_percentile = threshold_percentile
        self.gpd_shape: Optional[float] = None   # ξ (xi)
        self.gpd_scale: Optional[float] = None   # σ
        self.threshold: Optional[float] = None
        self.n_exceedances: int = 0

    def fit(self, losses: np.ndarray) -> Dict:
        """
        Fit GPD to exceedances over threshold.

        Uses the Probability-Weighted Moments (PWM) estimator
        which is more robust than MLE for small samples.
        """
        self.threshold = float(np.percentile(losses, self.threshold_percentile))
        exceedances = losses[losses > self.threshold] - self.threshold
        self.n_exceedances = len(exceedances)

        if self.n_exceedances < 5:
            # Not enough exceedances — fall back to exponential
            self.gpd_shape = 0.0
            self.gpd_scale = float(np.mean(exceedances)) if len(exceedances) > 0 else 1.0
            return self._fit_summary()

        # PWM estimation for GPD
        sorted_exc = np.sort(exceedances)
        n = len(sorted_exc)

        # First two probability-weighted moments
        b0 = np.mean(sorted_exc)
        b1 = np.sum(np.arange(1, n) / (n - 1) * sorted_exc[1:]) / n

        # GPD parameter estimates from PWM
        self.gpd_shape = float(2.0 - b0 / (b0 - 2 * b1))
        self.gpd_scale = float(2 * b0 * b1 / (b0 - 2 * b1))

        # Bound shape parameter for stability
        self.gpd_shape = max(-0.5, min(0.5, self.gpd_shape))
        self.gpd_scale = max(0.001, self.gpd_scale)

        return self._fit_summary()

    def _fit_summary(self) -> Dict:
        return {
            "threshold": self.threshold,
            "n_exceedances": self.n_exceedances,
            "gpd_shape": self.gpd_shape,
            "gpd_scale": self.gpd_scale,
            "tail_type": (
                "heavy_tail" if self.gpd_shape > 0.1
                else "thin_tail" if self.gpd_shape < -0.1
                else "exponential_tail"
            ),
        }

    def tail_probability(self, loss_level: float) -> float:
        """P(Loss > loss_level) using the fitted GPD."""
        if self.gpd_shape is None or self.threshold is None:
            raise RuntimeError("Must call fit() first")

        if loss_level <= self.threshold:
            return 1.0 - self.threshold_percentile / 100.0

        y = loss_level - self.threshold
        xi = self.gpd_shape
        sigma = self.gpd_scale

        exceedance_rate = 1.0 - self.threshold_percentile / 100.0

        if abs(xi) < 1e-10:
            # Exponential case
            gpd_survival = np.exp(-y / sigma)
        else:
            term = 1 + xi * y / sigma
            if term <= 0:
                return 0.0
            gpd_survival = term ** (-1.0 / xi)

        return exceedance_rate * gpd_survival

    def var_evt(self, confidence: float = 0.99) -> float:
        """
        Value at Risk using EVT (more accurate than empirical for extreme quantiles).
        """
        if self.gpd_shape is None or self.threshold is None:
            raise RuntimeError("Must call fit() first")

        p = 1 - confidence
        exceedance_rate = 1.0 - self.threshold_percentile / 100.0
        xi = self.gpd_shape
        sigma = self.gpd_scale

        if abs(xi) < 1e-10:
            return self.threshold + sigma * np.log(exceedance_rate / p)

        return self.threshold + (sigma / xi) * ((exceedance_rate / p) ** xi - 1)

    def cvar_evt(self, confidence: float = 0.99) -> float:
        """
        Conditional VaR (Expected Shortfall) using EVT.
        CVaR = E[Loss | Loss > VaR]
        """
        var = self.var_evt(confidence)
        xi = self.gpd_shape
        sigma = self.gpd_scale

        if xi >= 1.0:
            return float("inf")

        cvar = var / (1 - xi) + (sigma - xi * self.threshold) / (1 - xi)
        return max(var, cvar)

    def stress_test(
        self,
        scenarios: List[Dict[str, float]],
    ) -> List[Dict]:
        """
        Run stress tests for extreme scenarios.
        Each scenario specifies {variable: shock_magnitude}.
        """
        results = []
        for scenario in scenarios:
            total_loss = sum(scenario.values())
            prob = self.tail_probability(total_loss)
            return_period = 1.0 / prob if prob > 0 else float("inf")

            results.append({
                "scenario": scenario,
                "total_loss": total_loss,
                "probability": prob,
                "return_period_years": return_period,
                "severity": (
                    "catastrophic" if prob < 0.001
                    else "severe" if prob < 0.01
                    else "significant" if prob < 0.05
                    else "moderate"
                ),
            })

        return results
