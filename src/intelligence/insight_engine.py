"""
TimeStone AI — Insight Engine

Automated strategic insight generation from simulation results.
Transforms raw Monte Carlo outputs into actionable executive intelligence:

1. Pattern Detection — finds non-obvious relationships in results
2. Risk Narrative — translates statistical risk into business language
3. Recommendation Synthesis — combines multiple signals into ranked actions
4. Scenario Comparison — identifies trade-offs and synergies
5. Executive Summary — generates board-ready narrative
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class InsightType(Enum):
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    TRADE_OFF = "trade_off"
    SYNERGY = "synergy"
    ANOMALY = "anomaly"
    THRESHOLD = "threshold"
    DEPENDENCY = "dependency"


class Urgency(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"   # >90% statistical confidence
    HIGH = "high"             # 75-90%
    MODERATE = "moderate"     # 60-75%
    LOW = "low"               # <60%


@dataclass
class Insight:
    """A single actionable insight generated from analysis."""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    evidence: List[str]
    urgency: Urgency
    confidence: ConfidenceLevel
    affected_variables: List[str]
    quantitative_impact: Dict[str, float]
    recommended_action: str
    risk_if_ignored: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategicRecommendation:
    """A synthesized strategic recommendation from multiple insights."""
    rank: int
    title: str
    description: str
    expected_roi: float
    success_probability: float
    risk_level: str
    time_to_value_months: int
    required_investment: float
    prerequisites: List[str]
    supporting_insights: List[str]
    implementation_phases: List[Dict]
    risk_mitigation: List[str]
    confidence_interval: Tuple[float, float]
    go_no_go_criteria: List[str]


@dataclass
class ExecutiveSummary:
    """Board-ready executive summary."""
    headline: str
    key_finding: str
    top_recommendations: List[StrategicRecommendation]
    critical_risks: List[Insight]
    bottom_line: str
    next_steps: List[str]


class InsightEngine:
    """
    Generates strategic insights from simulation results.

    Combines statistical analysis with business logic to produce
    insights that are:
    - Specific (not "things might change")
    - Actionable (clear what to do)
    - Quantified (by how much)
    - Ranked (most important first)
    """

    def __init__(self):
        self.insights: List[Insight] = []
        self._insight_counter = 0

    def analyze_simulation_results(
        self,
        results: List[Dict],
        company_context: Optional[Dict] = None,
    ) -> List[Insight]:
        """
        Generate insights from a set of simulation results.

        Args:
            results: list of ScenarioSimulationResult dicts
            company_context: optional context about the company
        """
        self.insights = []

        self._detect_high_confidence_opportunities(results)
        self._detect_risk_clusters(results)
        self._detect_trade_offs(results)
        self._detect_synergies(results)
        self._detect_anomalies(results)
        self._detect_threshold_effects(results)
        self._assess_portfolio_concentration(results)

        # Sort by urgency then confidence
        urgency_order = {Urgency.CRITICAL: 0, Urgency.HIGH: 1, Urgency.MEDIUM: 2,
                         Urgency.LOW: 3, Urgency.INFORMATIONAL: 4}
        self.insights.sort(key=lambda i: (urgency_order[i.urgency], i.confidence.value))

        return self.insights

    def generate_recommendations(
        self,
        results: List[Dict],
        insights: Optional[List[Insight]] = None,
        budget_constraint: Optional[float] = None,
        max_recommendations: int = 5,
    ) -> List[StrategicRecommendation]:
        """
        Synthesize ranked strategic recommendations from results + insights.
        """
        if insights is None:
            insights = self.insights

        # Score scenarios
        scored = []
        for r in results:
            score = self._compute_recommendation_score(r)
            scored.append((r, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        total_investment = 0.0

        for rank, (r, score) in enumerate(scored[:max_recommendations], 1):
            investment = r.get("investment_required", 0)

            if budget_constraint and total_investment + investment > budget_constraint:
                continue

            total_investment += investment

            # Find supporting insights
            supporting = [
                i.insight_id for i in insights
                if any(v in i.affected_variables for v in [r.get("scenario_name", "")])
            ]

            mean_roi = r.get("mean_roi", 0)
            ci_lower = r.get("ci_lower", mean_roi - 0.1)
            ci_upper = r.get("ci_upper", mean_roi + 0.1)

            rec = StrategicRecommendation(
                rank=rank,
                title=r.get("scenario_name", f"Recommendation #{rank}"),
                description=self._generate_recommendation_description(r),
                expected_roi=mean_roi,
                success_probability=r.get("success_probability", 0),
                risk_level=self._categorize_risk(r),
                time_to_value_months=r.get("implementation_time_months", 12),
                required_investment=investment,
                prerequisites=self._identify_prerequisites(r),
                supporting_insights=supporting,
                implementation_phases=self._generate_phases(r),
                risk_mitigation=self._generate_mitigations(r),
                confidence_interval=(ci_lower, ci_upper),
                go_no_go_criteria=self._generate_go_no_go(r),
            )
            recommendations.append(rec)

        return recommendations

    def generate_executive_summary(
        self,
        results: List[Dict],
        recommendations: List[StrategicRecommendation],
        company_name: str = "the company",
    ) -> ExecutiveSummary:
        """Generate a board-ready executive summary."""
        insights = self.insights

        top_rec = recommendations[0] if recommendations else None
        critical_risks = [i for i in insights if i.urgency == Urgency.CRITICAL]

        if top_rec:
            headline = (
                f"{company_name}: {top_rec.title} offers "
                f"{top_rec.expected_roi:.0%} expected ROI with "
                f"{top_rec.success_probability:.0%} success probability"
            )
        else:
            headline = f"{company_name}: Transformation Analysis Complete"

        avg_success = np.mean([r.get("success_probability", 0) for r in results]) if results else 0
        n_viable = sum(1 for r in results if r.get("success_probability", 0) > 0.7)

        key_finding = (
            f"Analysis of {len(results)} transformation scenarios reveals "
            f"{n_viable} viable strategies (>70% success probability) "
            f"with an average success rate of {avg_success:.0%}. "
        )

        if critical_risks:
            key_finding += f"However, {len(critical_risks)} critical risk(s) require immediate attention."
        else:
            key_finding += "No critical risks were identified."

        total_potential_roi = sum(r.expected_roi for r in recommendations[:3])
        total_investment = sum(r.required_investment for r in recommendations[:3])

        bottom_line = (
            f"Implementing the top {min(3, len(recommendations))} recommendations "
            f"requires ${total_investment:,.0f} in investment and could yield "
            f"a combined {total_potential_roi:.0%} ROI over 3 years."
        )

        next_steps = [
            f"Approve Phase 1 of '{recommendations[0].title}'" if recommendations else "Refine scenarios",
            "Establish transformation governance committee",
            "Set up KPI dashboards for tracking progress",
            "Schedule 90-day checkpoint review",
            "Begin data collection for Bayesian calibration feedback loop",
        ]

        return ExecutiveSummary(
            headline=headline,
            key_finding=key_finding,
            top_recommendations=recommendations[:3],
            critical_risks=critical_risks,
            bottom_line=bottom_line,
            next_steps=next_steps,
        )

    # ---- Pattern Detection Methods ----

    def _detect_high_confidence_opportunities(self, results: List[Dict]) -> None:
        """Find scenarios with both high ROI and high confidence."""
        for r in results:
            sp = r.get("success_probability", 0)
            roi = r.get("mean_roi", 0)
            sharpe = r.get("sharpe_ratio", 0)

            if sp > 0.85 and roi > 0.15 and sharpe > 1.0:
                self._add_insight(
                    InsightType.OPPORTUNITY,
                    f"High-conviction opportunity: {r.get('scenario_name', 'Unknown')}",
                    f"This scenario shows {sp:.0%} success probability with {roi:.0%} expected ROI "
                    f"and a Sharpe ratio of {sharpe:.1f}, placing it in the top tier of risk-adjusted returns.",
                    [f"Success probability: {sp:.1%}", f"Mean ROI: {roi:.1%}", f"Sharpe: {sharpe:.2f}"],
                    Urgency.HIGH,
                    ConfidenceLevel.VERY_HIGH if sp > 0.9 else ConfidenceLevel.HIGH,
                    [r.get("scenario_name", "")],
                    {"expected_roi": roi, "success_probability": sp},
                    f"Prioritize '{r.get('scenario_name', '')}' for immediate Phase 1 implementation.",
                    "Delayed action risks competitive preemption and diminishing first-mover advantage.",
                )

    def _detect_risk_clusters(self, results: List[Dict]) -> None:
        """Identify clusters of scenarios with correlated risks."""
        high_risk = [r for r in results if r.get("ruin_probability", 0) > 0.05]

        if len(high_risk) >= 3:
            names = [r.get("scenario_name", "")[:30] for r in high_risk[:5]]
            avg_ruin = np.mean([r.get("ruin_probability", 0) for r in high_risk])

            self._add_insight(
                InsightType.RISK,
                f"{len(high_risk)} scenarios carry significant ruin risk",
                f"A cluster of {len(high_risk)} scenarios shows >5% probability of "
                f"catastrophic loss (ROI < -50%). Average ruin probability: {avg_ruin:.1%}. "
                f"These should be avoided or heavily risk-mitigated.",
                [f"Affected: {', '.join(names)}"],
                Urgency.CRITICAL if avg_ruin > 0.10 else Urgency.HIGH,
                ConfidenceLevel.HIGH,
                names,
                {"average_ruin_probability": avg_ruin, "count": len(high_risk)},
                "Exclude or restructure these scenarios with additional risk controls.",
                "Pursuing these without mitigation exposes the organization to potential catastrophic loss.",
            )

    def _detect_trade_offs(self, results: List[Dict]) -> None:
        """Identify scenarios where high reward comes with high risk."""
        for r in results:
            roi = r.get("mean_roi", 0)
            var = abs(r.get("value_at_risk_95", 0))
            spread = r.get("ci_upper", 0) - r.get("ci_lower", 0)

            if roi > 0.20 and var > 0.15 and spread > 0.5:
                self._add_insight(
                    InsightType.TRADE_OFF,
                    f"High reward / high risk: {r.get('scenario_name', '')}",
                    f"Expected ROI of {roi:.0%} but with VaR-95 of {var:.0%} and "
                    f"a confidence interval spread of {spread:.0%}. Consider phased implementation.",
                    [f"ROI: {roi:.1%}", f"VaR-95: {var:.1%}", f"CI spread: {spread:.1%}"],
                    Urgency.MEDIUM,
                    ConfidenceLevel.HIGH,
                    [r.get("scenario_name", "")],
                    {"expected_roi": roi, "var_95": var},
                    "Consider pilot phase before full commitment.",
                    "Full-scale implementation without piloting carries downside risk.",
                )

    def _detect_synergies(self, results: List[Dict]) -> None:
        """Identify pairs of scenarios that could amplify each other."""
        for i, r1 in enumerate(results):
            for r2 in results[i + 1:]:
                # Simplified synergy detection: complementary focus areas
                t1 = r1.get("type", "")
                t2 = r2.get("type", "")
                if t1 != t2 and r1.get("success_probability", 0) > 0.6 and r2.get("success_probability", 0) > 0.6:
                    combined_roi = r1.get("mean_roi", 0) + r2.get("mean_roi", 0)
                    if combined_roi > 0.30:
                        self._add_insight(
                            InsightType.SYNERGY,
                            f"Potential synergy: {r1.get('scenario_name', '')[:25]} + {r2.get('scenario_name', '')[:25]}",
                            f"Combining these two different transformation types could yield "
                            f"combined ROI of {combined_roi:.0%}.",
                            [],
                            Urgency.MEDIUM,
                            ConfidenceLevel.MODERATE,
                            [r1.get("scenario_name", ""), r2.get("scenario_name", "")],
                            {"combined_roi": combined_roi},
                            "Evaluate joint implementation roadmap.",
                            "Pursuing in isolation may miss compounding benefits.",
                        )
                        break  # one synergy per scenario
            else:
                continue

    def _detect_anomalies(self, results: List[Dict]) -> None:
        """Find scenarios with unexpected statistical properties."""
        rois = [r.get("mean_roi", 0) for r in results]
        if len(rois) < 3:
            return

        mean_roi = np.mean(rois)
        std_roi = np.std(rois)

        for r in results:
            roi = r.get("mean_roi", 0)
            z = (roi - mean_roi) / (std_roi + 1e-9)
            kurtosis = r.get("kurtosis", 0)

            if abs(z) > 2.5:
                self._add_insight(
                    InsightType.ANOMALY,
                    f"Statistical outlier: {r.get('scenario_name', '')}",
                    f"This scenario's ROI ({roi:.1%}) is {z:.1f} standard deviations "
                    f"from the mean. Verify assumptions and data inputs.",
                    [f"Z-score: {z:.2f}", f"Kurtosis: {kurtosis:.2f}"],
                    Urgency.MEDIUM,
                    ConfidenceLevel.MODERATE,
                    [r.get("scenario_name", "")],
                    {"z_score": float(z)},
                    "Deep-dive into this scenario's assumptions.",
                    "May indicate model mis-specification or a genuine outlier opportunity/risk.",
                )

    def _detect_threshold_effects(self, results: List[Dict]) -> None:
        """Identify scenarios near critical decision thresholds."""
        for r in results:
            sp = r.get("success_probability", 0)
            if 0.48 <= sp <= 0.52:
                self._add_insight(
                    InsightType.THRESHOLD,
                    f"Coin-flip scenario: {r.get('scenario_name', '')}",
                    f"Success probability of {sp:.1%} is near 50/50. "
                    f"Small changes in assumptions could tip this either way. "
                    f"Conduct sensitivity analysis to identify the swing factors.",
                    [f"Success: {sp:.1%}"],
                    Urgency.HIGH,
                    ConfidenceLevel.LOW,
                    [r.get("scenario_name", "")],
                    {"success_probability": sp},
                    "Run focused sensitivity analysis before deciding.",
                    "Decision without understanding the swing factors is essentially random.",
                )

    def _assess_portfolio_concentration(self, results: List[Dict]) -> None:
        """Check if recommended portfolio is too concentrated."""
        types = [r.get("type", "unknown") for r in results if r.get("success_probability", 0) > 0.7]
        if not types:
            return

        from collections import Counter
        type_counts = Counter(types)
        dominant = type_counts.most_common(1)[0]

        if dominant[1] / len(types) > 0.6:
            self._add_insight(
                InsightType.RISK,
                f"Portfolio concentration risk: {dominant[1]}/{len(types)} viable scenarios are '{dominant[0]}'",
                "Over-reliance on one transformation type. "
                "Diversify the portfolio across different transformation categories.",
                [],
                Urgency.MEDIUM,
                ConfidenceLevel.HIGH,
                [],
                {"concentration_ratio": dominant[1] / len(types)},
                "Actively seek opportunities in under-represented transformation types.",
                "Concentration increases systemic risk if that transformation type faces headwinds.",
            )

    # ---- Helper methods ----

    def _add_insight(self, itype, title, desc, evidence, urgency, confidence,
                     affected, impact, action, risk):
        self._insight_counter += 1
        self.insights.append(Insight(
            insight_id=f"INS-{self._insight_counter:04d}",
            insight_type=itype,
            title=title,
            description=desc,
            evidence=evidence,
            urgency=urgency,
            confidence=confidence,
            affected_variables=affected,
            quantitative_impact=impact,
            recommended_action=action,
            risk_if_ignored=risk,
        ))

    def _compute_recommendation_score(self, result: Dict) -> float:
        sp = result.get("success_probability", 0)
        roi = result.get("mean_roi", 0)
        sharpe = result.get("sharpe_ratio", 0)
        ruin = result.get("ruin_probability", 0)

        return (0.35 * sp + 0.25 * min(roi, 1.0) + 0.25 * min(sharpe / 3, 1.0)
                - 0.15 * ruin)

    def _categorize_risk(self, result: Dict) -> str:
        ruin = result.get("ruin_probability", 0)
        std = result.get("std_dev", 0)
        if ruin > 0.10 or std > 0.50:
            return "high"
        if ruin > 0.03 or std > 0.25:
            return "medium"
        return "low"

    def _generate_recommendation_description(self, result: Dict) -> str:
        name = result.get("scenario_name", "This transformation")
        roi = result.get("mean_roi", 0)
        sp = result.get("success_probability", 0)
        return (
            f"{name} is projected to deliver {roi:.0%} ROI with {sp:.0%} success probability. "
            f"Implementation should follow a phased approach with clear go/no-go gates."
        )

    def _identify_prerequisites(self, result: Dict) -> List[str]:
        prereqs = ["Secure executive sponsorship", "Allocate dedicated transformation team"]
        risk = result.get("risk_level", "medium")
        if risk == "high":
            prereqs.append("Establish risk mitigation framework")
            prereqs.append("Complete pilot validation phase")
        return prereqs

    def _generate_phases(self, result: Dict) -> List[Dict]:
        months = result.get("implementation_time_months", 12)
        return [
            {"phase": "Discovery & Planning", "duration_months": max(1, months // 6),
             "deliverables": ["Detailed requirements", "Risk assessment", "Pilot design"]},
            {"phase": "Pilot", "duration_months": max(2, months // 4),
             "deliverables": ["Pilot results", "Go/no-go decision", "Scaled plan"]},
            {"phase": "Scale", "duration_months": max(3, months // 2),
             "deliverables": ["Full deployment", "Performance metrics", "Team training"]},
            {"phase": "Optimize", "duration_months": max(2, months // 4),
             "deliverables": ["Optimization report", "ROI validation", "Lessons learned"]},
        ]

    def _generate_mitigations(self, result: Dict) -> List[str]:
        mitigations = ["Implement staged rollout with rollback capability"]
        risk = result.get("risk_level", "medium")
        if risk in ("high", "medium"):
            mitigations.extend([
                "Set automated circuit breakers on key metrics",
                "Maintain parallel operations during transition",
                "Monthly executive review of progress vs. plan",
            ])
        return mitigations

    def _generate_go_no_go(self, result: Dict) -> List[str]:
        return [
            "Pilot achieves >80% of projected impact metrics",
            "No critical issues identified during pilot phase",
            "Team readiness assessment score > 70%",
            "Budget variance within 15% of plan",
            "Key stakeholder alignment confirmed",
        ]
