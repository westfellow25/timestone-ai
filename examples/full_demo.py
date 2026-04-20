"""
TimeStone AI — End-to-end demo

Runs a complete analysis for a sample transportation company to showcase
the full capabilities of the platform. Use this as a reference for
building applications with TimeStone.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from src.core.bayesian_calibration import BayesianCalibrator, ObservedOutcome
from src.core.causal_graph import CausalGraph
from src.core.company_genome import CompanyGenome, GENOME_FACTORS, GenomeDimension
from src.core.temporal_engine import TemporalEngine
from src.intelligence.insight_engine import InsightEngine
from src.models.knowledge_graph import KnowledgeGraphBuilder
from src.simulation.advanced_monte_carlo import (
    AdvancedMonteCarloEngine,
    SamplingMethod,
    SimulationConfig,
)
from src.simulation.regime_detector import ExtremeValueAnalyzer, RegimeDetector


def banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_demo():
    banner("TIMESTONE AI — FULL PIPELINE DEMO")
    print("Scenario: Kazakhstan Temir Zholy (KTZ) — strategic transformation analysis")

    # -----------------------------------------------------------------------
    # 1. Load industry knowledge graph
    # -----------------------------------------------------------------------
    banner("1. INDUSTRY KNOWLEDGE GRAPH")
    knowledge = KnowledgeGraphBuilder.get_industry_knowledge("transportation")
    graph = knowledge.typical_causal_graph

    print(f"Industry: {knowledge.industry_name}")
    print(f"Variables: {len(graph.variables)}")
    print(f"Causal edges: {sum(len(e) for e in graph.edges.values())}")
    print(f"Key value drivers: {', '.join(knowledge.key_value_drivers)}")

    # -----------------------------------------------------------------------
    # 2. Causal graph intervention analysis
    # -----------------------------------------------------------------------
    banner("2. CAUSAL INTERVENTION ANALYSIS — do(digital_maturity = 0.7)")

    interventions = {"digital_maturity": 0.7}
    trajectories = graph.do_intervention(
        interventions, time_horizon=24, stochastic=True, seed=42
    )

    print(f"{'Variable':<30} {'Baseline':>12} {'End State':>12} {'Change':>10}")
    print("-" * 66)
    for name in ["revenue", "route_efficiency", "fuel_cost", "customer_satisfaction"]:
        if name in trajectories:
            baseline = graph.variables[name].current_value
            end = trajectories[name][-1]
            change_pct = (end - baseline) / baseline * 100 if baseline != 0 else 0
            print(f"{name:<30} {baseline:>12.3f} {end:>12.3f} {change_pct:>9.1f}%")

    # Causal paths
    paths = graph.find_all_causal_paths("digital_maturity", "revenue")
    print(f"\nCausal paths from digital_maturity to revenue: {len(paths)}")
    for p in paths[:3]:
        print(f"  {' → '.join(p)}")

    # -----------------------------------------------------------------------
    # 3. Company Genome
    # -----------------------------------------------------------------------
    banner("3. COMPANY GENOME — 48-factor capability fingerprint")

    genome = CompanyGenome("KTZ", "transportation")
    genome.set_factor("revenue_growth_rate", 0.55)
    genome.set_factor("profit_margin", 0.35)
    genome.set_factor("cash_flow_stability", 0.70)
    genome.set_factor("capacity_utilization", 0.72)
    genome.set_factor("digital_infrastructure_score", 0.35)
    genome.set_factor("data_readiness", 0.40)
    genome.set_factor("ai_ml_capability", 0.25)
    genome.set_factor("change_management_capability", 0.50)
    genome.set_factor("talent_density", 0.45)
    genome.set_factor("leadership_quality", 0.65)
    genome.set_factor("market_share", 0.85)
    genome.set_factor("customer_retention", 0.88)
    genome.set_factor("competitive_moat_score", 0.75)

    print(f"Overall genome score: {genome.get_overall_score():.3f}")
    print(f"\nDimension scores:")
    for dim in GenomeDimension:
        ds = genome.get_dimension_score(dim)
        print(f"  {dim.value:<28} {ds.score:.3f}")

    # Transformation readiness
    readiness = genome.transformation_readiness("digital_transformation")
    print(f"\nDigital transformation readiness: {readiness['readiness_score']:.3f} (Grade: {readiness['readiness_grade']})")
    print(f"Recommendation: {readiness['recommendation']}")
    print(f"\nTop capability gaps:")
    for gap in readiness["critical_gaps"][:3]:
        current = gap["current"] if gap["current"] is not None else "not measured"
        print(f"  - {gap['factor']}: current={current}, required>=0.5")

    # -----------------------------------------------------------------------
    # 4. Advanced Monte Carlo Simulation
    # -----------------------------------------------------------------------
    banner("4. ADVANCED MONTE CARLO — 10,000 iterations × Latin Hypercube")

    scenarios = [
        {
            "id": "S1",
            "name": "Dynamic Pricing Implementation",
            "type": "pricing_optimization",
            "expected_impact": {"revenue_increase": 0.18, "cost_reduction": 0.03},
            "investment_required": 3_000_000,
            "implementation_time_months": 9,
            "risk_level": "medium",
        },
        {
            "id": "S2",
            "name": "AI Route Optimization",
            "type": "operational_efficiency",
            "expected_impact": {"revenue_increase": 0.08, "cost_reduction": 0.15},
            "investment_required": 6_000_000,
            "implementation_time_months": 15,
            "risk_level": "medium",
        },
        {
            "id": "S3",
            "name": "Predictive Maintenance Platform",
            "type": "operational_efficiency",
            "expected_impact": {"revenue_increase": 0.05, "cost_reduction": 0.22},
            "investment_required": 12_000_000,
            "implementation_time_months": 24,
            "risk_level": "high",
        },
        {
            "id": "S4",
            "name": "Digital Booking Platform",
            "type": "customer_experience",
            "expected_impact": {"revenue_increase": 0.12, "cost_reduction": 0.06},
            "investment_required": 2_000_000,
            "implementation_time_months": 6,
            "risk_level": "low",
        },
        {
            "id": "S5",
            "name": "Full Digital Twin (Fleet)",
            "type": "digital_transformation",
            "expected_impact": {"revenue_increase": 0.10, "cost_reduction": 0.18},
            "investment_required": 25_000_000,
            "implementation_time_months": 36,
            "risk_level": "high",
        },
    ]

    config = SimulationConfig(
        iterations=10_000,
        method=SamplingMethod.LATIN_HYPERCUBE,
        antithetic=True,
        seed=42,
    )
    engine = AdvancedMonteCarloEngine(config)

    portfolio = engine.simulate_portfolio(scenarios, baseline_revenue=500e6)

    print(f"Total simulations: {portfolio['total_simulations']:,}")
    print(f"\n{'Scenario':<35} {'Mean ROI':>10} {'Success':>10} {'VaR-95':>10} {'Sharpe':>8}")
    print("-" * 75)
    for r in portfolio["individual_results"]:
        name = r.scenario_name[:33]
        print(f"{name:<35} {r.mean_roi:>9.1%} {r.success_probability:>9.1%} {r.value_at_risk_95:>9.1%} {r.sharpe_ratio:>8.2f}")

    if portfolio["portfolio_metrics"]:
        pm = portfolio["portfolio_metrics"]
        print(f"\nPortfolio (equal weight):")
        print(f"  Mean ROI: {pm.get('mean_portfolio_roi', 0):.1%}")
        print(f"  Portfolio std: {pm.get('portfolio_std', 0):.3f}")
        print(f"  Diversification benefit: {pm.get('diversification_benefit', 0):.3f}")

    # -----------------------------------------------------------------------
    # 5. Regime Detection
    # -----------------------------------------------------------------------
    banner("5. MARKET REGIME DETECTION")

    rng = np.random.default_rng(42)
    returns = np.concatenate([
        rng.normal(0.08, 0.10, 12),
        rng.normal(0.04, 0.08, 12),
        rng.normal(-0.05, 0.18, 6),
    ])

    detector = RegimeDetector()
    state = detector.detect_regime(returns)
    print(f"Current regime: {state.current_regime.value}")
    print(f"Confidence: {state.confidence:.2%}")

    regime_seq = detector.generate_regime_sequence(24, seed=42)
    print(f"Projected 24-month regime sequence (sample):")
    print(f"  {' → '.join(regime_seq[:8])}")

    # -----------------------------------------------------------------------
    # 6. Extreme Value Analysis
    # -----------------------------------------------------------------------
    banner("6. EXTREME VALUE ANALYSIS — tail risk via GPD")

    all_rois = np.concatenate([r.roi_samples for r in portfolio["individual_results"]])
    losses = -all_rois[all_rois < 0]

    if len(losses) > 50:
        evt = ExtremeValueAnalyzer(threshold_percentile=90)
        fit = evt.fit(losses)
        print(f"Fitted GPD: shape={fit['gpd_shape']:.3f}, scale={fit['gpd_scale']:.3f}")
        print(f"Tail type: {fit['tail_type']}")
        print(f"\nVaR-99 (EVT): {evt.var_evt(0.99):.1%}")
        print(f"CVaR-99 (EVT): {evt.cvar_evt(0.99):.1%}")

    # -----------------------------------------------------------------------
    # 7. Strategic Insights
    # -----------------------------------------------------------------------
    banner("7. STRATEGIC INSIGHTS & RECOMMENDATIONS")

    result_dicts = []
    for r in portfolio["individual_results"]:
        result_dicts.append({
            "scenario_id": r.scenario_id,
            "scenario_name": r.scenario_name,
            "mean_roi": r.mean_roi,
            "median_roi": r.median_roi,
            "std_dev": r.std_dev,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "success_probability": r.success_probability,
            "ruin_probability": r.ruin_probability,
            "value_at_risk_95": r.value_at_risk_95,
            "sharpe_ratio": r.sharpe_ratio,
            "kurtosis": r.kurtosis,
            "investment_required": 0,
            "implementation_time_months": 12,
            "risk_level": "medium",
        })

    insight_engine = InsightEngine()
    insights = insight_engine.analyze_simulation_results(result_dicts)
    recommendations = insight_engine.generate_recommendations(result_dicts, insights)
    summary = insight_engine.generate_executive_summary(result_dicts, recommendations, "KTZ")

    print(f"\nGENERATED INSIGHTS ({len(insights)}):")
    for i in insights[:5]:
        print(f"\n[{i.urgency.value.upper()}] {i.title}")
        print(f"  {i.description}")
        print(f"  Action: {i.recommended_action}")

    print(f"\n\nTOP RECOMMENDATIONS:")
    for r in recommendations[:3]:
        print(f"\n#{r.rank}. {r.title}")
        print(f"  Expected ROI: {r.expected_roi:.1%}")
        print(f"  Success probability: {r.success_probability:.1%}")
        print(f"  Risk level: {r.risk_level}")
        print(f"  CI: [{r.confidence_interval[0]:.1%}, {r.confidence_interval[1]:.1%}]")

    # -----------------------------------------------------------------------
    # 8. Executive Summary
    # -----------------------------------------------------------------------
    banner("8. EXECUTIVE SUMMARY")

    print(f"\nHEADLINE: {summary.headline}")
    print(f"\nKEY FINDING:\n  {summary.key_finding}")
    print(f"\nBOTTOM LINE:\n  {summary.bottom_line}")
    print(f"\nNEXT STEPS:")
    for i, step in enumerate(summary.next_steps, 1):
        print(f"  {i}. {step}")

    # -----------------------------------------------------------------------
    # 9. Bayesian Calibration
    # -----------------------------------------------------------------------
    banner("9. BAYESIAN CALIBRATION — self-improving prediction loop")

    calibrator = BayesianCalibrator(graph)
    print(f"Initial priors: {len(calibrator.priors)} edges")

    # Simulate some observed outcomes
    for i in range(15):
        calibrator.record_outcome(ObservedOutcome(
            prediction_id=f"pred-{i}",
            variable_name="revenue",
            predicted_value=500e6 * (1 + 0.05 * i),
            actual_value=500e6 * (1 + 0.05 * i) + rng.normal(0, 5e6),
            predicted_confidence_lower=500e6 * (1 + 0.05 * i) * 0.95,
            predicted_confidence_upper=500e6 * (1 + 0.05 * i) * 1.05,
            timestamp=f"2026-{i+1:02d}-01",
        ))

    metrics = calibrator.calibrate_from_outcomes()
    print(f"\nCalibration metrics:")
    print(f"  Total observations: {metrics.total_predictions}")
    print(f"  Mean absolute error: {metrics.mean_absolute_error:,.0f}")
    print(f"  Confidence coverage: {metrics.confidence_coverage:.1%}")
    print(f"  Calibration score: {metrics.calibration_score:.3f}")
    print(f"  Bias: {metrics.bias:,.0f}")

    suggestions = calibrator.suggest_data_collection(top_n=3)
    print(f"\nTop data collection recommendations:")
    for s in suggestions:
        print(f"  - {s['recommendation']} (VoI: {s['value_of_information']:.2f})")

    banner("DEMO COMPLETE")
    print(f"\nTotal analysis: {portfolio['total_simulations']:,} Monte Carlo iterations")
    print(f"Causal paths analyzed: {len(graph.variables) ** 2}")
    print(f"Genome dimensions: {len(GenomeDimension)} × 8 factors = {sum(len(f) for f in GENOME_FACTORS.values())}")
    print(f"\nTimeStone AI: See 10,000 futures. Choose the one truth.\n")


if __name__ == "__main__":
    run_demo()
