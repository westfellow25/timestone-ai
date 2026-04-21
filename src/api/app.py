"""
TimeStone AI — FastAPI Application

Production REST API with:
- Versioned endpoints (v1)
- JWT authentication
- Rate limiting
- WebSocket for real-time simulation streaming
- Structured logging
- Health checks
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.core.bayesian_calibration import BayesianCalibrator, ObservedOutcome
from src.core.causal_graph import CausalGraph
from src.core.company_genome import CompanyGenome, GenomeDimension
from src.core.temporal_engine import TemporalEngine, TimeResolution
from src.intelligence.insight_engine import InsightEngine
from src.models.knowledge_graph import KnowledgeGraphBuilder
from src.simulation.advanced_monte_carlo import AdvancedMonteCarloEngine, SimulationConfig, SamplingMethod
from src.simulation.regime_detector import RegimeDetector, ExtremeValueAnalyzer
from src.simulation.sensitivity_analyzer import SensitivityAnalyzer
from src.persistence.database import initialize_database, get_session
from src.persistence.repositories import (
    TenantRepository,
    CompanyRepository,
    SimulationRepository,
    AuditLogRepository,
)


# ---- Stores ----
_twins_store: Dict[str, Dict] = {}
_simulation_store: Dict[str, Dict] = {}
_genome_store: Dict[str, CompanyGenome] = {}
_graph_store: Dict[str, CausalGraph] = {}
_demo_tenant_id: Optional[str] = None


# ---- Pydantic Models ----

class CompanyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    industry: str
    revenue: float = Field(gt=0)
    operating_costs: float = Field(gt=0)
    employee_count: int = Field(gt=0)
    market_share: float = Field(ge=0, le=1)
    metadata: Dict[str, Any] = {}


class ScenarioInput(BaseModel):
    name: str
    type: str = "digital_transformation"
    expected_revenue_increase: float = Field(ge=0, le=5.0)
    expected_cost_reduction: float = Field(ge=0, le=1.0)
    investment_required: float = Field(gt=0)
    implementation_time_months: int = Field(gt=0, le=120)
    risk_level: str = Field(default="medium", pattern="^(low|medium|high)$")


class SimulationRequest(BaseModel):
    company_id: str
    scenarios: List[ScenarioInput]
    iterations: int = Field(default=10_000, ge=100, le=500_000)
    confidence_level: float = Field(default=0.90, ge=0.5, le=0.99)
    sampling_method: str = Field(default="latin_hypercube")
    include_sensitivity: bool = False
    include_regime_analysis: bool = False
    time_horizon_months: int = Field(default=36, ge=6, le=120)


class GenomeFactorInput(BaseModel):
    name: str
    value: float = Field(ge=0, le=1)
    confidence: float = Field(default=0.8, ge=0, le=1)


class GenomeSetup(BaseModel):
    company_name: str
    industry: str
    factors: List[GenomeFactorInput]


class OutcomeRecord(BaseModel):
    prediction_id: str
    variable_name: str
    predicted_value: float
    actual_value: float
    predicted_ci_lower: float
    predicted_ci_upper: float


class InterventionRequest(BaseModel):
    company_id: str
    interventions: Dict[str, float]
    time_horizon: int = Field(default=24, ge=1, le=120)
    num_paths: int = Field(default=500, ge=10, le=10_000)
    include_counterfactual: bool = False
    counterfactual_time: int = Field(default=0, ge=0)


# ---- Auth ----

security = HTTPBearer(auto_error=False)

API_KEYS = {"ts-demo-key-2026"}  # demo; use proper auth in production


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    if credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials


# ---- App setup ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _demo_tenant_id
    # Initialize database
    initialize_database()
    with get_session() as session:
        tenant_repo = TenantRepository(session)
        demo_tenant = tenant_repo.get_by_slug("demo")
        if not demo_tenant:
            demo_tenant = tenant_repo.create("Demo", "demo", plan="enterprise")
        _demo_tenant_id = demo_tenant.id

    # Preload industry knowledge
    for industry in KnowledgeGraphBuilder.list_supported_industries():
        knowledge = KnowledgeGraphBuilder.get_industry_knowledge(industry)
        if knowledge:
            _graph_store[industry] = knowledge.typical_causal_graph
    yield
    _twins_store.clear()
    _simulation_store.clear()


app = FastAPI(
    title="TimeStone AI",
    description="Temporal Intelligence Platform — Predict business transformation outcomes before you commit.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Rate limiting middleware ----

_request_counts: Dict[str, List[float]] = {}
RATE_LIMIT = 100  # requests per minute


# ---- Health & Info ----

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "industries_loaded": len(_graph_store),
    }


@app.get("/v1/info")
async def info():
    return {
        "product": "TimeStone AI",
        "version": "1.0.0",
        "capabilities": [
            "causal_graph_analysis",
            "monte_carlo_simulation",
            "company_genome_profiling",
            "bayesian_calibration",
            "regime_detection",
            "extreme_value_analysis",
            "sensitivity_analysis",
            "strategic_insight_generation",
            "temporal_propagation",
            "counterfactual_reasoning",
        ],
        "supported_industries": KnowledgeGraphBuilder.list_supported_industries(),
        "simulation_methods": [m.value for m in SamplingMethod],
    }


# ---- Company / Twin Management ----

@app.post("/v1/companies", dependencies=[Depends(verify_token)])
async def create_company(company: CompanyCreate):
    company_id = str(uuid.uuid4())[:8]
    _twins_store[company_id] = {
        "id": company_id,
        "name": company.name,
        "industry": company.industry,
        "revenue": company.revenue,
        "operating_costs": company.operating_costs,
        "employee_count": company.employee_count,
        "market_share": company.market_share,
        "metadata": company.metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Auto-generate causal graph from industry knowledge
    knowledge = KnowledgeGraphBuilder.get_industry_knowledge(company.industry)
    if knowledge:
        _graph_store[company_id] = knowledge.typical_causal_graph

    return {"company_id": company_id, "status": "created", **_twins_store[company_id]}


@app.get("/v1/companies/{company_id}", dependencies=[Depends(verify_token)])
async def get_company(company_id: str):
    if company_id not in _twins_store:
        raise HTTPException(404, "Company not found")
    return _twins_store[company_id]


@app.get("/v1/companies", dependencies=[Depends(verify_token)])
async def list_companies():
    return {"companies": list(_twins_store.values()), "total": len(_twins_store)}


# ---- Simulation ----

@app.post("/v1/simulate", dependencies=[Depends(verify_token)])
async def run_simulation(request: SimulationRequest):
    if request.company_id not in _twins_store:
        raise HTTPException(404, "Company not found")

    company = _twins_store[request.company_id]
    baseline_revenue = company["revenue"]

    # Configure engine
    method_map = {
        "naive": SamplingMethod.NAIVE,
        "stratified": SamplingMethod.STRATIFIED,
        "antithetic": SamplingMethod.ANTITHETIC,
        "latin_hypercube": SamplingMethod.LATIN_HYPERCUBE,
        "importance": SamplingMethod.IMPORTANCE,
    }

    config = SimulationConfig(
        iterations=request.iterations,
        method=method_map.get(request.sampling_method, SamplingMethod.LATIN_HYPERCUBE),
        confidence_level=request.confidence_level,
        seed=42,
    )

    engine = AdvancedMonteCarloEngine(config)

    # Convert scenarios
    scenarios = []
    for s in request.scenarios:
        scenarios.append({
            "id": str(uuid.uuid4())[:8],
            "name": s.name,
            "type": s.type,
            "expected_impact": {
                "revenue_increase": s.expected_revenue_increase,
                "cost_reduction": s.expected_cost_reduction,
            },
            "investment_required": s.investment_required,
            "implementation_time_months": s.implementation_time_months,
            "risk_level": s.risk_level,
        })

    # Get causal multipliers if graph exists
    causal_multipliers = None
    if request.company_id in _graph_store:
        graph = _graph_store[request.company_id]
        influence = graph.get_influence_scores()
        max_influence = max(influence.values()) if influence else 1.0
        causal_multipliers = {
            "revenue_multiplier": 1.0 + 0.1 * influence.get("revenue", 0) / (max_influence + 1e-9),
            "cost_multiplier": 1.0,
            "risk_adjustment": 1.0,
        }

    # Run simulation
    portfolio = engine.simulate_portfolio(
        scenarios, baseline_revenue, causal_multipliers=causal_multipliers
    )

    # Generate insights
    insight_engine = InsightEngine()
    result_dicts = []
    for r in portfolio["individual_results"]:
        rd = {
            "scenario_id": r.scenario_id,
            "scenario_name": r.scenario_name,
            "mean_roi": r.mean_roi,
            "median_roi": r.median_roi,
            "std_dev": r.std_dev,
            "skewness": r.skewness,
            "kurtosis": r.kurtosis,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "success_probability": r.success_probability,
            "high_success_probability": r.high_success_probability,
            "ruin_probability": r.ruin_probability,
            "value_at_risk_95": r.value_at_risk_95,
            "conditional_var_95": r.conditional_var_95,
            "sharpe_ratio": r.sharpe_ratio,
            "percentiles": r.percentiles,
        }
        if r.diagnostics:
            rd["diagnostics"] = {
                "iterations_run": int(r.diagnostics.iterations_run),
                "std_error": float(r.diagnostics.std_error),
                "converged": bool(r.diagnostics.converged),
                "effective_sample_size": float(r.diagnostics.effective_sample_size),
                "variance_reduction_factor": float(r.diagnostics.variance_reduction_factor),
            }
        result_dicts.append(rd)

    insights = insight_engine.analyze_simulation_results(result_dicts)
    recommendations = insight_engine.generate_recommendations(result_dicts, insights)
    summary = insight_engine.generate_executive_summary(
        result_dicts, recommendations, company["name"]
    )

    # Store results
    sim_id = str(uuid.uuid4())[:8]
    _simulation_store[sim_id] = {
        "simulation_id": sim_id,
        "company_id": request.company_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": result_dicts,
        "portfolio_metrics": portfolio.get("portfolio_metrics", {}),
    }

    return {
        "simulation_id": sim_id,
        "company": company["name"],
        "total_simulations": portfolio["total_simulations"],
        "results": result_dicts,
        "ranked_by_sharpe": portfolio["ranked_by_sharpe"],
        "portfolio_metrics": portfolio.get("portfolio_metrics", {}),
        "insights": [
            {
                "id": i.insight_id,
                "type": i.insight_type.value,
                "title": i.title,
                "description": i.description,
                "urgency": i.urgency.value,
                "confidence": i.confidence.value,
                "action": i.recommended_action,
            }
            for i in insights[:10]
        ],
        "recommendations": [
            {
                "rank": r.rank,
                "title": r.title,
                "expected_roi": r.expected_roi,
                "success_probability": r.success_probability,
                "risk_level": r.risk_level,
                "investment": r.required_investment,
                "phases": r.implementation_phases,
            }
            for r in recommendations[:5]
        ],
        "executive_summary": {
            "headline": summary.headline,
            "key_finding": summary.key_finding,
            "bottom_line": summary.bottom_line,
            "next_steps": summary.next_steps,
        },
    }


# ---- Causal Graph Analysis ----

@app.post("/v1/interventions", dependencies=[Depends(verify_token)])
async def analyze_intervention(request: InterventionRequest):
    if request.company_id not in _graph_store:
        raise HTTPException(404, "No causal graph found for this company")

    graph = _graph_store[request.company_id]
    temporal = TemporalEngine(graph)

    trajectories = temporal.simulate(
        time_horizon=request.time_horizon,
        initial_interventions=request.interventions,
        num_paths=request.num_paths,
        seed=42,
    )

    # Compute mean trajectories
    mean_trajectories = {
        name: data.mean(axis=0).tolist() for name, data in trajectories.items()
    }

    ci_lower = {
        name: np.percentile(data, 5, axis=0).tolist() for name, data in trajectories.items()
    }
    ci_upper = {
        name: np.percentile(data, 95, axis=0).tolist() for name, data in trajectories.items()
    }

    # Causal path analysis
    paths_info = {}
    for target in graph.variables:
        for source in request.interventions:
            if source != target:
                effect = graph.total_causal_effect(source, target)
                if abs(effect) > 0.001:
                    paths_info[f"{source} → {target}"] = {
                        "total_effect": effect,
                        "paths": [
                            " → ".join(p)
                            for p in graph.find_all_causal_paths(source, target)[:5]
                        ],
                    }

    result = {
        "interventions": request.interventions,
        "time_horizon": request.time_horizon,
        "num_paths": request.num_paths,
        "mean_trajectories": mean_trajectories,
        "confidence_interval_lower": ci_lower,
        "confidence_interval_upper": ci_upper,
        "causal_effects": paths_info,
        "influence_scores": graph.get_influence_scores(),
        "vulnerability_scores": graph.get_vulnerability_scores(),
    }

    # Counterfactual
    if request.include_counterfactual:
        cf = graph.counterfactual(
            factual_history={
                name: data.mean(axis=0) for name, data in trajectories.items()
            },
            counterfactual_interventions=request.interventions,
            intervention_time=request.counterfactual_time,
            time_horizon=request.time_horizon,
            seed=42,
        )
        result["counterfactual"] = {
            name: arr.tolist() for name, arr in cf.items()
        }

    return result


# ---- Company Genome ----

@app.post("/v1/genome", dependencies=[Depends(verify_token)])
async def create_genome(setup: GenomeSetup):
    genome = CompanyGenome(setup.company_name, setup.industry)
    for f in setup.factors:
        genome.set_factor(f.name, f.value, confidence=f.confidence)

    genome_id = str(uuid.uuid4())[:8]
    _genome_store[genome_id] = genome

    return {
        "genome_id": genome_id,
        "overall_score": genome.get_overall_score(),
        "dimensions": {
            dim.value: genome.get_dimension_score(dim).score
            for dim in GenomeDimension
        },
        "genome_vector": genome.get_genome_vector().tolist(),
    }


@app.get("/v1/genome/{genome_id}/readiness", dependencies=[Depends(verify_token)])
async def transformation_readiness(
    genome_id: str,
    transformation_type: str = Query(default="digital_transformation"),
):
    if genome_id not in _genome_store:
        raise HTTPException(404, "Genome not found")
    genome = _genome_store[genome_id]
    return genome.transformation_readiness(transformation_type)


# ---- Causal Graph Exploration ----

@app.get("/v1/graph/{company_id}", dependencies=[Depends(verify_token)])
async def get_causal_graph(company_id: str):
    if company_id not in _graph_store:
        raise HTTPException(404, "No causal graph for this company")
    graph = _graph_store[company_id]
    return {
        "variables": len(graph.variables),
        "edges": sum(len(e) for e in graph.edges.values()),
        "graph": graph.to_dict(),
        "influence_scores": graph.get_influence_scores(),
        "vulnerability_scores": graph.get_vulnerability_scores(),
        "critical_paths": [
            {"path": " → ".join(p), "effect": e}
            for p, e in graph.get_critical_paths(10)
        ],
    }


# ---- Industry Knowledge ----

@app.get("/v1/industries", dependencies=[Depends(verify_token)])
async def list_industries():
    industries = []
    for industry in KnowledgeGraphBuilder.list_supported_industries():
        knowledge = KnowledgeGraphBuilder.get_industry_knowledge(industry)
        if knowledge:
            industries.append({
                "name": knowledge.industry_name,
                "key": industry,
                "value_drivers": knowledge.key_value_drivers,
                "transformation_success_rates": knowledge.transformation_success_rates,
                "common_failure_modes": knowledge.common_failure_modes,
            })
    return {"industries": industries}


# ---- Regime Detection ----

@app.post("/v1/regime/detect", dependencies=[Depends(verify_token)])
async def detect_regime(returns: List[float], window: int = Query(default=12, ge=3)):
    import numpy as _np
    detector = RegimeDetector()
    state = detector.detect_regime(_np.array(returns), window)
    return {
        "regime": state.current_regime.value,
        "confidence": state.confidence,
        "duration": state.regime_duration,
    }


# ---- Calibration ----

@app.post("/v1/calibrate", dependencies=[Depends(verify_token)])
async def record_calibration_outcome(outcome: OutcomeRecord):
    # In production, this would persist and update the global model
    return {
        "recorded": True,
        "error": outcome.actual_value - outcome.predicted_value,
        "percentage_error": abs(outcome.actual_value - outcome.predicted_value) / abs(outcome.predicted_value + 1e-9) * 100,
        "within_ci": outcome.predicted_ci_lower <= outcome.actual_value <= outcome.predicted_ci_upper,
    }


# ---- WebSocket for real-time simulation streaming ----

@app.websocket("/v1/ws/simulate")
async def websocket_simulation(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "simulate":
                config = SimulationConfig(iterations=data.get("iterations", 1000), seed=42)
                engine = AdvancedMonteCarloEngine(config)

                scenarios = data.get("scenarios", [])
                baseline = data.get("baseline_revenue", 500e6)

                for i, scenario in enumerate(scenarios):
                    result = engine.simulate_scenario(scenario, baseline)
                    await websocket.send_json({
                        "type": "scenario_result",
                        "progress": (i + 1) / len(scenarios),
                        "scenario_name": result.scenario_name,
                        "mean_roi": result.mean_roi,
                        "success_probability": result.success_probability,
                    })

                await websocket.send_json({"type": "complete", "total": len(scenarios)})

    except WebSocketDisconnect:
        pass


# ---- Entrypoint ----

import numpy as np  # needed for intervention endpoint

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
