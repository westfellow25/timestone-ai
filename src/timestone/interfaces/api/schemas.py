"""Pydantic schemas for the TimeStone REST API.

Kept deliberately thin: API models mirror the domain objects but do not
import them, so wire format and domain can evolve independently.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ----- Library -----

class CaseSummary(BaseModel):
    case_id: str
    company: str
    industry: str
    year: int
    status: str  # success | failed | partial | cancelled
    transformation_type: str = ""
    geography: str = ""
    promised_investment_usd: Optional[float] = None
    actual_investment_usd: Optional[float] = None
    writeoff_usd: Optional[float] = None
    failure_modes: List[str] = Field(default_factory=list)
    sources: List[Dict] = Field(default_factory=list)


class LibraryResponse(BaseModel):
    total: int
    cases: List[CaseSummary]


# ----- Assess -----

class AssessOptionsIn(BaseModel):
    scenario_count: int = Field(default=200, ge=10, le=2000)
    iterations: int = Field(default=400, ge=100, le=2000)
    discount_rate: float = Field(default=0.12, ge=0.0, le=0.5)
    horizon_years: int = Field(default=5, ge=1, le=15)
    random_seed: int = 42
    use_case_library: bool = True


class AssessRequest(BaseModel):
    company_name: str
    options: AssessOptionsIn = Field(default_factory=AssessOptionsIn)


class RecommendationOut(BaseModel):
    rank: int
    scenario_id: int
    scenario_name: str
    headline: str
    success_probability: float
    mean_npv: float
    mean_roi: float
    payback_years: float
    based_on_cases: List[str]
    empirical_failure_rate: Optional[float] = None
    risk_level: str = "medium"
    description: str = ""


class AssessResponse(BaseModel):
    run_id: str
    company_name: str
    generated_at: str
    total_scenarios: int
    failure_rate_among_scenarios: float
    case_library_size: int
    top_recommendations: List[RecommendationOut]


# ----- Predictions (pre-registered) -----

class PredictionOut(BaseModel):
    id: str
    company: str
    project: str
    region: str
    invest_musd: float
    p_npv_positive: float
    mean_npv_musd: float
    p10_musd: float
    p90_musd: float
    payback_years: float
    recommendation: str
    verification_window: str
    failure_modes: List[str] = Field(default_factory=list)
    committed_at: str  # e.g. "2026-05-20"


# ----- Auth / tenants -----

class TenantInfo(BaseModel):
    tenant_id: str
    name: str
    plan: str
    quota_assess_per_day: int
    used_today: int


# ----- Errors -----

class ErrorOut(BaseModel):
    error: str
    detail: Optional[str] = None


# ----- Health -----

class HealthOut(BaseModel):
    status: str
    version: str
    case_library_size: int
    twins_loaded: int


# ----- Listings -----

class CompanyOut(BaseModel):
    name: str
    industry: str
    region: Optional[str] = None
    annual_revenue_usd: Optional[float] = None
    annual_op_costs_usd: Optional[float] = None
    employees: Optional[int] = None
    prior_transformation_count: int = 0


class CompanyListResponse(BaseModel):
    total: int
    companies: List[CompanyOut]
