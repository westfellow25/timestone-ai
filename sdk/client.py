"""
TimeStone AI — Python SDK

Official Python client for the TimeStone AI API.

Usage:
    from timestone_sdk import TimeStoneClient

    client = TimeStoneClient(api_key="your-key")

    company = client.create_company(
        name="My Corp",
        industry="fintech",
        revenue=100e6,
        operating_costs=80e6,
        employee_count=500,
        market_share=0.05,
    )

    result = client.simulate(
        company_id=company["company_id"],
        scenarios=[{...}],
        iterations=10000,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class TimeStoneConfig:
    api_key: str
    base_url: str = "http://localhost:8000"
    timeout: float = 120.0
    max_retries: int = 3


class TimeStoneError(Exception):
    """Base exception for TimeStone SDK errors."""


class AuthenticationError(TimeStoneError):
    """Raised when authentication fails."""


class RateLimitError(TimeStoneError):
    """Raised when rate limit is exceeded."""


class TimeStoneClient:
    """
    Official Python client for TimeStone AI.

    Thread-safe. Supports both sync and async (via `aclient` attribute).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ):
        self.config = TimeStoneConfig(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

    def _request(self, method: str, path: str, **kwargs) -> Dict:
        response = self._client.request(method, path, **kwargs)
        if response.status_code == 401:
            raise AuthenticationError("Invalid or missing API key")
        if response.status_code == 403:
            raise AuthenticationError("API key not authorized for this operation")
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        if response.status_code >= 400:
            raise TimeStoneError(
                f"API error {response.status_code}: {response.text}"
            )
        return response.json()

    # ---- Health ----

    def health(self) -> Dict:
        return self._request("GET", "/health")

    def info(self) -> Dict:
        return self._request("GET", "/v1/info")

    # ---- Companies ----

    def create_company(
        self,
        name: str,
        industry: str,
        revenue: float,
        operating_costs: float,
        employee_count: int,
        market_share: float,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        return self._request(
            "POST",
            "/v1/companies",
            json={
                "name": name,
                "industry": industry,
                "revenue": revenue,
                "operating_costs": operating_costs,
                "employee_count": employee_count,
                "market_share": market_share,
                "metadata": metadata or {},
            },
        )

    def get_company(self, company_id: str) -> Dict:
        return self._request("GET", f"/v1/companies/{company_id}")

    def list_companies(self) -> Dict:
        return self._request("GET", "/v1/companies")

    # ---- Simulation ----

    def simulate(
        self,
        company_id: str,
        scenarios: List[Dict],
        iterations: int = 10_000,
        confidence_level: float = 0.90,
        sampling_method: str = "latin_hypercube",
        include_sensitivity: bool = False,
        include_regime_analysis: bool = False,
    ) -> Dict:
        """
        Run Monte Carlo simulation for a list of scenarios.

        Each scenario dict should have:
        - name, type, expected_revenue_increase, expected_cost_reduction,
          investment_required, implementation_time_months, risk_level
        """
        return self._request(
            "POST",
            "/v1/simulate",
            json={
                "company_id": company_id,
                "scenarios": scenarios,
                "iterations": iterations,
                "confidence_level": confidence_level,
                "sampling_method": sampling_method,
                "include_sensitivity": include_sensitivity,
                "include_regime_analysis": include_regime_analysis,
            },
        )

    # ---- Causal Graph ----

    def analyze_intervention(
        self,
        company_id: str,
        interventions: Dict[str, float],
        time_horizon: int = 24,
        num_paths: int = 500,
        include_counterfactual: bool = False,
    ) -> Dict:
        """Analyze a do-intervention on the company's causal graph."""
        return self._request(
            "POST",
            "/v1/interventions",
            json={
                "company_id": company_id,
                "interventions": interventions,
                "time_horizon": time_horizon,
                "num_paths": num_paths,
                "include_counterfactual": include_counterfactual,
            },
        )

    def get_causal_graph(self, company_id: str) -> Dict:
        return self._request("GET", f"/v1/graph/{company_id}")

    # ---- Genome ----

    def create_genome(
        self,
        company_name: str,
        industry: str,
        factors: List[Dict],
    ) -> Dict:
        return self._request(
            "POST",
            "/v1/genome",
            json={
                "company_name": company_name,
                "industry": industry,
                "factors": factors,
            },
        )

    def get_transformation_readiness(
        self,
        genome_id: str,
        transformation_type: str = "digital_transformation",
    ) -> Dict:
        return self._request(
            "GET",
            f"/v1/genome/{genome_id}/readiness",
            params={"transformation_type": transformation_type},
        )

    # ---- Industries ----

    def list_industries(self) -> Dict:
        return self._request("GET", "/v1/industries")

    # ---- Regime Detection ----

    def detect_regime(self, returns: List[float], window: int = 12) -> Dict:
        return self._request(
            "POST",
            "/v1/regime/detect",
            json=returns,
            params={"window": window},
        )

    # ---- Calibration ----

    def record_outcome(
        self,
        prediction_id: str,
        variable_name: str,
        predicted_value: float,
        actual_value: float,
        predicted_ci_lower: float,
        predicted_ci_upper: float,
    ) -> Dict:
        """Record an actual outcome for Bayesian calibration."""
        return self._request(
            "POST",
            "/v1/calibrate",
            json={
                "prediction_id": prediction_id,
                "variable_name": variable_name,
                "predicted_value": predicted_value,
                "actual_value": actual_value,
                "predicted_ci_lower": predicted_ci_lower,
                "predicted_ci_upper": predicted_ci_upper,
            },
        )


__all__ = [
    "TimeStoneClient",
    "TimeStoneConfig",
    "TimeStoneError",
    "AuthenticationError",
    "RateLimitError",
]
