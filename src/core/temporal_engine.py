"""
TimeStone AI — Temporal Propagation Engine

Multi-resolution temporal modeling from daily operations to decade-scale
strategy. Combines the Causal Graph with time-aware state transitions,
seasonal patterns, and regime-dependent dynamics.

Key innovation: temporal effects don't just propagate forward — they
operate at different frequencies. A pricing change affects revenue in weeks,
market share in months, and competitive dynamics in years. This engine
models all resolutions simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.causal_graph import CausalGraph, CausalVariable


class TimeResolution(Enum):
    DAILY = ("daily", 365)
    WEEKLY = ("weekly", 52)
    MONTHLY = ("monthly", 12)
    QUARTERLY = ("quarterly", 4)
    ANNUAL = ("annual", 1)

    def __init__(self, label: str, periods_per_year: int):
        self.label = label
        self.periods_per_year = periods_per_year


@dataclass
class SeasonalPattern:
    """Encodes seasonal/cyclical behavior of a variable."""
    variable_name: str
    period: int              # cycle length in base time units
    amplitude: float         # peak-to-trough / 2
    phase_shift: float = 0.0 # offset in radians
    harmonics: int = 1       # number of Fourier harmonics

    def evaluate(self, t: int) -> float:
        total = 0.0
        for h in range(1, self.harmonics + 1):
            total += (self.amplitude / h) * np.sin(
                2 * np.pi * h * t / self.period + self.phase_shift
            )
        return total


@dataclass
class RegimeDependentDynamics:
    """Different causal strengths under different market regimes."""
    variable_name: str
    regime_multipliers: Dict[str, float] = field(default_factory=dict)
    # e.g. {"growth": 1.2, "stagnation": 0.7, "crisis": 0.3}


@dataclass
class TemporalState:
    """Snapshot of the system state at a point in time."""
    period: int
    resolution: TimeResolution
    values: Dict[str, float]
    regime: str = "normal"
    metadata: Dict = field(default_factory=dict)


class TemporalEngine:
    """
    Multi-resolution temporal simulation engine.

    Runs the causal graph forward in time with:
    - Multiple time resolutions (daily → annual)
    - Seasonal patterns overlaid on causal dynamics
    - Regime-dependent edge strengths
    - Momentum and autoregressive effects
    - Event-driven shocks at specified periods
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        base_resolution: TimeResolution = TimeResolution.MONTHLY,
    ):
        self.graph = causal_graph
        self.base_resolution = base_resolution
        self.seasonal_patterns: Dict[str, SeasonalPattern] = {}
        self.regime_dynamics: Dict[str, RegimeDependentDynamics] = {}
        self.scheduled_events: List[Tuple[int, Dict[str, float]]] = []
        self.ar_coefficients: Dict[str, List[float]] = {}  # autoregressive
        self.history: List[TemporalState] = []

    def add_seasonal_pattern(self, pattern: SeasonalPattern) -> None:
        self.seasonal_patterns[pattern.variable_name] = pattern

    def add_regime_dynamics(self, dynamics: RegimeDependentDynamics) -> None:
        self.regime_dynamics[dynamics.variable_name] = dynamics

    def set_autoregressive(self, variable_name: str, coefficients: List[float]) -> None:
        """Set AR(p) coefficients for a variable. coefficients[0] = AR(1), etc."""
        self.ar_coefficients[variable_name] = coefficients

    def schedule_event(self, period: int, interventions: Dict[str, float]) -> None:
        """Schedule a one-time shock/event at a specific period."""
        self.scheduled_events.append((period, interventions))

    def simulate(
        self,
        time_horizon: int,
        initial_interventions: Optional[Dict[str, float]] = None,
        regime_sequence: Optional[List[str]] = None,
        num_paths: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run multi-resolution temporal simulation.

        Args:
            time_horizon: number of periods to simulate
            initial_interventions: do-interventions applied at t=0
            regime_sequence: regime label for each period (or None for "normal")
            num_paths: number of stochastic paths to simulate
            seed: random seed

        Returns:
            {variable_name: np.array shape (num_paths, time_horizon)}
        """
        rng = np.random.default_rng(seed)
        topo_order = self.graph.topological_sort()

        intervened_set = set(initial_interventions.keys()) if initial_interventions else set()

        # Initialize output arrays
        results: Dict[str, np.ndarray] = {}
        for name in self.graph.variables:
            results[name] = np.zeros((num_paths, time_horizon))
            results[name][:, 0] = self.graph.variables[name].current_value

        # Apply initial interventions
        if initial_interventions:
            for var_name, val in initial_interventions.items():
                if var_name in results:
                    results[var_name][:, 0] = val

        # Build scheduled events lookup
        event_map: Dict[int, Dict[str, float]] = {}
        for period, interventions in self.scheduled_events:
            event_map.setdefault(period, {}).update(interventions)

        for path in range(num_paths):
            # Pending lagged effects
            pending: List[Tuple[str, int, float]] = []

            for t in range(1, time_horizon):
                regime = "normal"
                if regime_sequence and t < len(regime_sequence):
                    regime = regime_sequence[t]

                # Process pending effects
                period_deltas: Dict[str, float] = {}
                still_pending = []
                for target, arrival_t, delta in pending:
                    if arrival_t == t:
                        period_deltas[target] = period_deltas.get(target, 0.0) + delta
                    else:
                        still_pending.append((target, arrival_t, delta))
                pending = still_pending

                # Apply scheduled events
                if t in event_map:
                    for var_name, shock in event_map[t].items():
                        period_deltas[var_name] = period_deltas.get(var_name, 0.0) + shock

                for var_name in topo_order:
                    var = self.graph.variables[var_name]

                    if var_name in intervened_set:
                        results[var_name][path, t] = initial_interventions[var_name]
                        continue

                    base = results[var_name][path, t - 1]
                    delta = period_deltas.get(var_name, 0.0)

                    # Seasonal component
                    if var_name in self.seasonal_patterns:
                        delta += self.seasonal_patterns[var_name].evaluate(t)

                    # Autoregressive component
                    if var_name in self.ar_coefficients:
                        ar_coeffs = self.ar_coefficients[var_name]
                        for lag_idx, coeff in enumerate(ar_coeffs):
                            lag = lag_idx + 1
                            if t - lag >= 1:
                                prev_change = (
                                    results[var_name][path, t - lag]
                                    - results[var_name][path, t - lag - 1]
                                    if t - lag - 1 >= 0 else 0.0
                                )
                                delta += coeff * prev_change

                    # Mean reversion
                    if var.mean_reversion_rate > 0 and var.long_run_mean is not None:
                        delta += var.mean_reversion_rate * (var.long_run_mean - base)

                    # Stochastic noise
                    if var.volatility > 0:
                        scale = var.volatility * abs(base) if base != 0 else var.volatility
                        delta += rng.normal(0, scale)

                    new_value = var.clamp(base + delta)
                    results[var_name][path, t] = new_value

                    # Propagate to children with regime-dependent strength
                    delta_from_prev = new_value - results[var_name][path, t - 1]
                    for edge in self.graph.edges.get(var_name, []):
                        if edge.target in intervened_set:
                            continue

                        effect = edge.compute_effect(delta_from_prev, base)
                        effect *= edge.confidence

                        # Regime-dependent multiplier
                        if edge.target in self.regime_dynamics:
                            rd = self.regime_dynamics[edge.target]
                            effect *= rd.regime_multipliers.get(regime, 1.0)

                        if edge.decay_rate < 1.0:
                            effect *= edge.decay_rate ** t

                        if edge.lag_periods > 0:
                            arrival = t + edge.lag_periods
                            if arrival < time_horizon:
                                pending.append((edge.target, arrival, effect))
                        else:
                            period_deltas[edge.target] = (
                                period_deltas.get(edge.target, 0.0) + effect
                            )

        self.history = [
            TemporalState(
                period=t,
                resolution=self.base_resolution,
                values={name: np.mean(results[name][:, t]) for name in results},
                regime=regime_sequence[t] if regime_sequence and t < len(regime_sequence) else "normal",
            )
            for t in range(time_horizon)
        ]

        return results

    def aggregate_to_resolution(
        self,
        trajectories: Dict[str, np.ndarray],
        target_resolution: TimeResolution,
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate base-resolution trajectories to a coarser resolution.
        E.g., monthly → quarterly by averaging every 3 periods.
        """
        if target_resolution.periods_per_year >= self.base_resolution.periods_per_year:
            return trajectories

        ratio = self.base_resolution.periods_per_year // target_resolution.periods_per_year
        aggregated: Dict[str, np.ndarray] = {}

        for name, data in trajectories.items():
            num_paths, T = data.shape
            new_T = T // ratio
            reshaped = data[:, :new_T * ratio].reshape(num_paths, new_T, ratio)
            aggregated[name] = reshaped.mean(axis=2)

        return aggregated

    def compute_impulse_response(
        self,
        variable: str,
        shock_magnitude: float = 1.0,
        time_horizon: int = 36,
        num_paths: int = 500,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute impulse response functions: how does a one-time shock to
        `variable` propagate through the system over time?

        Returns difference between shocked and baseline trajectories,
        averaged over stochastic paths.
        """
        base_seed = seed or 42

        # Baseline (no shock)
        baseline = self.simulate(
            time_horizon=time_horizon,
            num_paths=num_paths,
            seed=base_seed,
        )

        # Shocked
        shock_interventions = {
            variable: self.graph.variables[variable].current_value + shock_magnitude
        }
        self.schedule_event(1, shock_interventions)

        shocked = self.simulate(
            time_horizon=time_horizon,
            num_paths=num_paths,
            seed=base_seed,
        )

        # Remove the temporary event
        self.scheduled_events = [
            (p, i) for p, i in self.scheduled_events
            if not (p == 1 and variable in i)
        ]

        # Compute impulse response (mean across paths)
        irf: Dict[str, np.ndarray] = {}
        for name in self.graph.variables:
            irf[name] = shocked[name].mean(axis=0) - baseline[name].mean(axis=0)

        return irf
