"""Build the final AssessmentReport from scenarios + simulation results."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from ..domain.report import AssessmentReport, Recommendation
from ..domain.scenario import Scenario
from ..domain.simulation import SimulationResult


def _headline(success_probability: float) -> str:
    if success_probability >= 0.80:
        return "STRONG RECOMMENDATION"
    if success_probability >= 0.60:
        return "PROCEED WITH CAUTION"
    return "PILOT FIRST"


def build_report(run_id: str, company_name: str,
                 scenarios: List[Scenario],
                 results: List[SimulationResult],
                 case_library_size: int,
                 config_summary: Dict,
                 top_n: int = 3) -> AssessmentReport:
    by_id = {s.id: s for s in scenarios}
    sorted_results = sorted(results, key=lambda r: r.success_probability, reverse=True)
    top = sorted_results[:top_n]
    recs: List[Recommendation] = []
    for rank, r in enumerate(top, 1):
        scen = by_id.get(r.scenario_id)
        based_on = scen.based_on_cases if scen else []
        prior = (scen.empirical_prior if scen else None) or {}
        rev_prior = prior.get("revenue_uplift") or {}
        recs.append(Recommendation(
            rank=rank,
            scenario_id=r.scenario_id, scenario_name=r.scenario_name,
            headline=_headline(r.success_probability),
            success_probability=r.success_probability,
            mean_npv=r.mean_npv, mean_roi=r.mean_roi,
            payback_years=r.payback_years_median,
            based_on_cases=based_on,
            empirical_failure_rate=prior.get("failure_rate"),
            p10_revenue_uplift=rev_prior.get("p10"),
            p90_revenue_uplift=rev_prior.get("p90"),
            risk_level=scen.risk_level if scen else "medium",
            description=scen.description if scen else "",
        ))
    failure_rate = sum(1 for r in results if r.success_probability < 0.50) / max(1, len(results))
    return AssessmentReport(
        run_id=run_id, company_name=company_name,
        generated_at=datetime.utcnow().isoformat(),
        config_summary=config_summary,
        top_recommendations=recs,
        total_scenarios=len(scenarios),
        failure_rate_among_scenarios=failure_rate,
        case_library_size=case_library_size,
    )
