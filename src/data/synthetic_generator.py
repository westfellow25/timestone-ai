"""
TimeStone AI — Synthetic Data Generator

Generates realistic synthetic time series data for companies
to enable cold-start scenarios when historical data is unavailable.
Uses industry benchmarks, trend extrapolation, and correlated noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SyntheticSeries:
    """A synthetic time series with metadata."""
    name: str
    values: np.ndarray
    unit: str
    trend: float
    volatility: float
    seasonality_amplitude: float


class SyntheticDataGenerator:
    """
    Generate synthetic but realistic business time series.

    Uses a multi-component model:
    - Trend (linear or exponential)
    - Seasonality (Fourier series)
    - AR(1) noise (persistence)
    - Regime switches
    - Outliers / shocks
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_series(
        self,
        name: str,
        initial_value: float,
        periods: int = 36,
        annual_growth_rate: float = 0.05,
        volatility: float = 0.10,
        seasonality_amplitude: float = 0.05,
        seasonal_period: int = 12,
        ar1_persistence: float = 0.3,
        shock_probability: float = 0.02,
        shock_magnitude: float = 0.15,
        unit: str = "",
    ) -> SyntheticSeries:
        """Generate a single realistic time series."""
        values = np.zeros(periods)
        values[0] = initial_value

        monthly_growth = (1 + annual_growth_rate) ** (1 / 12) - 1
        monthly_vol = volatility / np.sqrt(12)

        noise_prev = 0.0
        for t in range(1, periods):
            trend = values[t - 1] * (1 + monthly_growth)

            seasonal = initial_value * seasonality_amplitude * np.sin(
                2 * np.pi * t / seasonal_period
            )

            white_noise = self.rng.normal(0, monthly_vol * initial_value)
            ar_noise = ar1_persistence * noise_prev + white_noise
            noise_prev = ar_noise

            shock = 0.0
            if self.rng.random() < shock_probability:
                shock = self.rng.choice([-1, 1]) * shock_magnitude * initial_value

            values[t] = trend + seasonal + ar_noise + shock

        return SyntheticSeries(
            name=name,
            values=values,
            unit=unit,
            trend=annual_growth_rate,
            volatility=volatility,
            seasonality_amplitude=seasonality_amplitude,
        )

    def generate_correlated_series(
        self,
        specs: List[Dict],
        correlation_matrix: np.ndarray,
        periods: int = 36,
    ) -> Dict[str, SyntheticSeries]:
        """
        Generate multiple series with a specified correlation structure.
        Uses Cholesky decomposition to inject correlations.
        """
        n = len(specs)
        assert correlation_matrix.shape == (n, n)

        # Cholesky factorization
        L = np.linalg.cholesky(correlation_matrix + 1e-6 * np.eye(n))

        # Generate correlated shocks
        independent_shocks = self.rng.standard_normal((periods, n))
        correlated_shocks = independent_shocks @ L.T

        results = {}
        for i, spec in enumerate(specs):
            values = np.zeros(periods)
            values[0] = spec["initial_value"]

            monthly_growth = (1 + spec.get("annual_growth", 0.05)) ** (1 / 12) - 1
            monthly_vol = spec.get("volatility", 0.10) / np.sqrt(12)

            for t in range(1, periods):
                trend = values[t - 1] * (1 + monthly_growth)
                noise = correlated_shocks[t, i] * monthly_vol * spec["initial_value"]
                seasonal = spec["initial_value"] * spec.get("seasonality", 0.0) * np.sin(
                    2 * np.pi * t / 12
                )
                values[t] = trend + seasonal + noise

            results[spec["name"]] = SyntheticSeries(
                name=spec["name"],
                values=values,
                unit=spec.get("unit", ""),
                trend=spec.get("annual_growth", 0.05),
                volatility=spec.get("volatility", 0.10),
                seasonality_amplitude=spec.get("seasonality", 0.0),
            )

        return results

    def generate_company_history(
        self,
        industry: str,
        scale: str = "medium",
        periods: int = 36,
    ) -> Dict[str, SyntheticSeries]:
        """Generate a full synthetic company history based on industry template."""
        scale_multiplier = {"small": 0.1, "medium": 1.0, "large": 10.0}[scale]
        base_revenue = 100e6 * scale_multiplier

        industry_profiles = {
            "transportation": {
                "revenue": {"growth": 0.05, "vol": 0.08, "seasonality": 0.10},
                "operating_cost": {"growth": 0.04, "vol": 0.05, "seasonality": 0.05},
                "capacity_utilization": {"growth": 0.01, "vol": 0.03, "seasonality": 0.15},
                "fuel_cost": {"growth": 0.03, "vol": 0.20, "seasonality": 0.08},
            },
            "fintech": {
                "revenue": {"growth": 0.25, "vol": 0.15, "seasonality": 0.03},
                "active_users": {"growth": 0.30, "vol": 0.08, "seasonality": 0.02},
                "transaction_volume": {"growth": 0.28, "vol": 0.12, "seasonality": 0.05},
                "fraud_rate": {"growth": -0.02, "vol": 0.30, "seasonality": 0.0},
            },
            "energy": {
                "revenue": {"growth": 0.03, "vol": 0.10, "seasonality": 0.15},
                "generation_capacity": {"growth": 0.02, "vol": 0.01, "seasonality": 0.0},
                "grid_reliability": {"growth": 0.001, "vol": 0.002, "seasonality": 0.01},
                "renewable_share": {"growth": 0.15, "vol": 0.03, "seasonality": 0.0},
            },
        }

        profile = industry_profiles.get(industry.lower(), industry_profiles["transportation"])
        results = {}

        for var_name, params in profile.items():
            initial = base_revenue if "revenue" in var_name else base_revenue * 0.5
            if "rate" in var_name or "ratio" in var_name or "utilization" in var_name or "reliability" in var_name or "share" in var_name:
                initial = 0.75
            elif "users" in var_name:
                initial = 1_000_000 * scale_multiplier
            elif "volume" in var_name:
                initial = 10_000_000 * scale_multiplier
            elif "capacity" in var_name:
                initial = 1000 * scale_multiplier

            results[var_name] = self.generate_series(
                name=var_name,
                initial_value=initial,
                periods=periods,
                annual_growth_rate=params["growth"],
                volatility=params["vol"],
                seasonality_amplitude=params["seasonality"],
            )

        return results
