"""
TimeStone AI — Sensitivity Analyzer

Answers the critical question: "What drives outcomes most?"
Uses Sobol sensitivity indices and Morris screening to decompose
output variance into contributions from each input factor.

This is how TimeStone goes beyond "what might happen" to
"what should you focus on to control outcomes."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SensitivityResult:
    """Sensitivity analysis results for one output variable."""
    output_name: str
    first_order_indices: Dict[str, float]   # S_i: direct effect
    total_order_indices: Dict[str, float]    # S_Ti: total (incl. interactions)
    interaction_indices: Dict[str, float]    # S_ij: pairwise interactions
    rankings: List[Tuple[str, float]]        # sorted by total order
    total_variance: float
    method: str


@dataclass
class TornadoChartData:
    """Data for tornado sensitivity chart."""
    output_name: str
    factors: List[str]
    low_values: List[float]    # output when factor is at low end
    high_values: List[float]   # output when factor is at high end
    base_value: float          # output at base case
    swing: List[float]         # high - low


class SensitivityAnalyzer:
    """
    Multi-method sensitivity analysis engine.

    Methods:
    1. Sobol indices — variance-based, gold standard
    2. Morris screening — efficient for many factors
    3. Tornado analysis — one-at-a-time for communication
    4. Correlation-based — fast but less rigorous
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def sobol_analysis(
        self,
        model_fn: Callable[[np.ndarray], float],
        param_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 4096,
    ) -> SensitivityResult:
        """
        Compute Sobol sensitivity indices using Saltelli's sampling scheme.

        Args:
            model_fn: function mapping parameter vector → scalar output
            param_bounds: {param_name: (low, high)}
            n_samples: base sample size (total evals = N * (2D + 2))
        """
        param_names = list(param_bounds.keys())
        d = len(param_names)
        bounds = np.array([param_bounds[p] for p in param_names])

        # Generate two independent quasi-random matrices
        A = self._sobol_sequence(n_samples, d)
        B = self._sobol_sequence(n_samples, d)

        # Scale to bounds
        A_scaled = bounds[:, 0] + A * (bounds[:, 1] - bounds[:, 0])
        B_scaled = bounds[:, 0] + B * (bounds[:, 1] - bounds[:, 0])

        # Evaluate base matrices
        f_A = np.array([model_fn(A_scaled[i]) for i in range(n_samples)])
        f_B = np.array([model_fn(B_scaled[i]) for i in range(n_samples)])

        total_variance = np.var(np.concatenate([f_A, f_B]))
        if total_variance == 0:
            return SensitivityResult(
                output_name="output",
                first_order_indices={p: 0.0 for p in param_names},
                total_order_indices={p: 0.0 for p in param_names},
                interaction_indices={},
                rankings=[(p, 0.0) for p in param_names],
                total_variance=0.0,
                method="sobol",
            )

        first_order = {}
        total_order = {}

        for i, name in enumerate(param_names):
            # AB_i: A with column i replaced by B
            AB_i = A_scaled.copy()
            AB_i[:, i] = B_scaled[:, i]
            f_AB_i = np.array([model_fn(AB_i[j]) for j in range(n_samples)])

            # First-order: S_i = V[E[Y|X_i]] / V[Y]
            # Jansen estimator
            first_order[name] = float(
                np.mean(f_B * (f_AB_i - f_A)) / total_variance
            )
            first_order[name] = max(0.0, min(1.0, first_order[name]))

            # Total order: S_Ti = E[V[Y|X_~i]] / V[Y]
            total_order[name] = float(
                0.5 * np.mean((f_A - f_AB_i) ** 2) / total_variance
            )
            total_order[name] = max(0.0, min(1.0, total_order[name]))

        # Interaction indices (pairwise)
        interaction = {}
        for i, name_i in enumerate(param_names):
            for j, name_j in enumerate(param_names):
                if j <= i:
                    continue
                # S_ij ≈ S_Ti + S_Tj - S_ij_joint - S_i - S_j
                # Simplified: interaction ≈ total - first order contributions
                s_ij = max(0.0, (total_order[name_i] - first_order[name_i] +
                                  total_order[name_j] - first_order[name_j]) / 2)
                interaction[f"{name_i}:{name_j}"] = float(s_ij)

        rankings = sorted(total_order.items(), key=lambda x: x[1], reverse=True)

        return SensitivityResult(
            output_name="output",
            first_order_indices=first_order,
            total_order_indices=total_order,
            interaction_indices=interaction,
            rankings=rankings,
            total_variance=float(total_variance),
            method="sobol",
        )

    def morris_screening(
        self,
        model_fn: Callable[[np.ndarray], float],
        param_bounds: Dict[str, Tuple[float, float]],
        n_trajectories: int = 50,
        n_levels: int = 8,
    ) -> SensitivityResult:
        """
        Morris method (Elementary Effects) for efficient factor screening.
        Good when you have many factors and want to quickly identify
        the important ones before running full Sobol.
        """
        param_names = list(param_bounds.keys())
        d = len(param_names)
        bounds = np.array([param_bounds[p] for p in param_names])

        delta = n_levels / (2 * (n_levels - 1))

        # Generate trajectories
        elementary_effects: Dict[str, List[float]] = {name: [] for name in param_names}

        for _ in range(n_trajectories):
            # Random base point on grid
            base = np.array([
                self.rng.integers(0, n_levels) / (n_levels - 1)
                for _ in range(d)
            ])

            # Scale to bounds
            x_base = bounds[:, 0] + base * (bounds[:, 1] - bounds[:, 0])
            f_base = model_fn(x_base)

            # Perturb each factor
            order = self.rng.permutation(d)
            current = base.copy()

            for idx in order:
                perturbed = current.copy()
                if current[idx] + delta <= 1.0:
                    perturbed[idx] = current[idx] + delta
                else:
                    perturbed[idx] = current[idx] - delta

                x_pert = bounds[:, 0] + perturbed * (bounds[:, 1] - bounds[:, 0])
                f_pert = model_fn(x_pert)

                ee = (f_pert - f_base) / delta
                elementary_effects[param_names[idx]].append(ee)

                current = perturbed
                f_base = f_pert

        # Compute Morris statistics
        first_order = {}
        total_order = {}

        for name in param_names:
            ees = np.array(elementary_effects[name])
            if len(ees) == 0:
                first_order[name] = 0.0
                total_order[name] = 0.0
                continue

            mu_star = float(np.mean(np.abs(ees)))  # Morris μ*
            sigma = float(np.std(ees))              # Morris σ

            first_order[name] = mu_star
            total_order[name] = mu_star + sigma  # higher σ = more interactions

        # Normalize to sum to ~1
        total_sum = sum(total_order.values()) or 1.0
        total_order = {k: v / total_sum for k, v in total_order.items()}
        first_sum = sum(first_order.values()) or 1.0
        first_order = {k: v / first_sum for k, v in first_order.items()}

        rankings = sorted(total_order.items(), key=lambda x: x[1], reverse=True)

        return SensitivityResult(
            output_name="output",
            first_order_indices=first_order,
            total_order_indices=total_order,
            interaction_indices={},
            rankings=rankings,
            total_variance=0.0,
            method="morris",
        )

    def tornado_analysis(
        self,
        model_fn: Callable[[Dict[str, float]], float],
        base_params: Dict[str, float],
        param_ranges: Dict[str, Tuple[float, float]],
    ) -> TornadoChartData:
        """
        One-at-a-time tornado analysis for executive communication.
        Shows impact of moving each factor to its low/high bound.
        """
        base_output = model_fn(base_params)

        factors = []
        lows = []
        highs = []
        swings = []

        for name, (low, high) in param_ranges.items():
            # Low scenario
            params_low = {**base_params, name: low}
            out_low = model_fn(params_low)

            # High scenario
            params_high = {**base_params, name: high}
            out_high = model_fn(params_high)

            factors.append(name)
            lows.append(out_low)
            highs.append(out_high)
            swings.append(abs(out_high - out_low))

        # Sort by swing (largest first)
        indices = np.argsort(swings)[::-1]

        return TornadoChartData(
            output_name="output",
            factors=[factors[i] for i in indices],
            low_values=[lows[i] for i in indices],
            high_values=[highs[i] for i in indices],
            base_value=base_output,
            swing=[swings[i] for i in indices],
        )

    def _sobol_sequence(self, n: int, d: int) -> np.ndarray:
        """Generate quasi-random Sobol sequence (using random as fallback)."""
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d, scramble=True, seed=self.rng)
            # Round up to power of 2
            m = int(np.ceil(np.log2(max(n, 2))))
            samples = sampler.random_base2(m)
            return samples[:n]
        except ImportError:
            # Fallback to LHS
            result = np.zeros((n, d))
            for dim in range(d):
                perm = self.rng.permutation(n)
                result[:, dim] = (perm + self.rng.random(n)) / n
            return result
