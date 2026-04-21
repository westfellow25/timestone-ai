"""
TimeStone AI — Case Study & Report Generator

Auto-generates investor-ready documents from simulation results:
1. One-page executive brief
2. Detailed case study with charts
3. Technical appendix with methodology
4. Benchmark validation summary

Output format: structured dict (render to PDF/HTML via templates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReportSection:
    title: str
    content: str
    subsections: List["ReportSection"] = field(default_factory=list)
    metrics: Dict[str, str] = field(default_factory=dict)
    chart_data: Optional[Dict] = None


@dataclass
class CaseStudyReport:
    company_name: str
    industry: str
    generated_at: str
    sections: List[ReportSection]
    key_metrics: Dict[str, str]
    bottom_line: str
    metadata: Dict = field(default_factory=dict)


class ReportGenerator:
    """Generates structured reports from TimeStone analysis results."""

    def generate_executive_brief(
        self,
        company_name: str,
        industry: str,
        simulation_results: List[Dict],
        recommendations: List[Dict],
        genome_score: Optional[float] = None,
        readiness_grade: Optional[str] = None,
    ) -> CaseStudyReport:
        """Generate a one-page executive brief."""
        now = datetime.now(timezone.utc).isoformat()

        n_scenarios = len(simulation_results)
        viable = [r for r in simulation_results if r.get("success_probability", 0) > 0.7]
        top = recommendations[0] if recommendations else {}

        avg_roi = np.mean([r.get("mean_roi", 0) for r in simulation_results])
        avg_success = np.mean([r.get("success_probability", 0) for r in simulation_results])
        total_investment = sum(r.get("required_investment", 0) for r in recommendations[:3])

        sections = [
            ReportSection(
                title="Situation",
                content=(
                    f"{company_name} ({industry}) is evaluating strategic transformation options. "
                    f"TimeStone AI analyzed {n_scenarios} scenarios using Monte Carlo simulation "
                    f"with causal graph reasoning to quantify outcomes."
                ),
                metrics={
                    "Scenarios Analyzed": str(n_scenarios),
                    "Viable (>70% success)": str(len(viable)),
                    "Genome Score": f"{genome_score:.2f}" if genome_score else "N/A",
                    "Readiness Grade": readiness_grade or "N/A",
                },
            ),
            ReportSection(
                title="Key Findings",
                content=(
                    f"Average success probability across all scenarios is {avg_success:.0%}. "
                    f"{len(viable)} out of {n_scenarios} scenarios show >70% probability of positive ROI."
                ),
                subsections=[
                    ReportSection(
                        title=f"#{i+1}: {r.get('title', 'Recommendation')}",
                        content=(
                            f"Expected ROI: {r.get('expected_roi', 0):.0%} | "
                            f"Success: {r.get('success_probability', 0):.0%} | "
                            f"Risk: {r.get('risk_level', 'unknown')}"
                        ),
                    )
                    for i, r in enumerate(recommendations[:3])
                ],
            ),
            ReportSection(
                title="Risk Assessment",
                content=self._risk_narrative(simulation_results),
                metrics={
                    "Avg VaR-95": f"{np.mean([r.get('value_at_risk_95', 0) for r in simulation_results]):.1%}",
                    "Max ruin probability": f"{max(r.get('ruin_probability', 0) for r in simulation_results):.1%}",
                },
            ),
            ReportSection(
                title="Recommended Action",
                content=(
                    f"Proceed with '{top.get('title', 'top recommendation')}' as Phase 1. "
                    f"Estimated investment: ${total_investment:,.0f}. "
                    f"Expected combined ROI from top 3 strategies: "
                    f"{sum(r.get('expected_roi', 0) for r in recommendations[:3]):.0%}."
                ),
            ),
        ]

        return CaseStudyReport(
            company_name=company_name,
            industry=industry,
            generated_at=now,
            sections=sections,
            key_metrics={
                "Scenarios": str(n_scenarios),
                "Avg Success Probability": f"{avg_success:.0%}",
                "Avg ROI": f"{avg_roi:.0%}",
                "Top Recommendation": top.get("title", "N/A"),
                "Required Investment": f"${total_investment:,.0f}",
            },
            bottom_line=(
                f"{company_name} should prioritize '{top.get('title', 'the top strategy')}' "
                f"with {top.get('success_probability', 0):.0%} success probability and "
                f"{top.get('expected_roi', 0):.0%} expected ROI."
            ),
        )

    def generate_investor_one_pager(
        self,
        company_name: str,
        industry: str,
        before_metrics: Dict[str, float],
        simulation_results: List[Dict],
        benchmark_accuracy: Optional[float] = None,
    ) -> CaseStudyReport:
        """Generate a one-pager for investor due diligence."""
        now = datetime.now(timezone.utc).isoformat()

        viable = [r for r in simulation_results if r.get("success_probability", 0) > 0.7]
        best = max(simulation_results, key=lambda r: r.get("sharpe_ratio", 0)) if simulation_results else {}

        sections = [
            ReportSection(
                title="Client Profile",
                content=f"{company_name} — {industry}",
                metrics={k: f"{v:,.0f}" if isinstance(v, (int, float)) and v > 100 else f"{v}" for k, v in before_metrics.items()},
            ),
            ReportSection(
                title="TimeStone Analysis",
                content=(
                    f"Analyzed {len(simulation_results)} transformation scenarios. "
                    f"Identified {len(viable)} high-confidence opportunities."
                ),
                metrics={
                    "Scenarios": str(len(simulation_results)),
                    "Viable (>70%)": str(len(viable)),
                    "Best Sharpe Ratio": f"{best.get('sharpe_ratio', 0):.2f}",
                },
            ),
            ReportSection(
                title="Predictive Accuracy",
                content=(
                    f"Validated against historical benchmark cases. "
                    f"Prediction accuracy: {benchmark_accuracy:.0%}" if benchmark_accuracy
                    else "Accuracy benchmarks in progress."
                ),
            ),
            ReportSection(
                title="Value Delivered",
                content=(
                    f"Best scenario: '{best.get('scenario_name', 'N/A')}' with "
                    f"{best.get('mean_roi', 0):.0%} expected ROI and "
                    f"{best.get('success_probability', 0):.0%} success probability."
                ),
            ),
        ]

        return CaseStudyReport(
            company_name=company_name,
            industry=industry,
            generated_at=now,
            sections=sections,
            key_metrics={
                "Client": company_name,
                "Industry": industry,
                "Scenarios Analyzed": str(len(simulation_results)),
                "Best ROI": f"{best.get('mean_roi', 0):.0%}",
                "Accuracy": f"{benchmark_accuracy:.0%}" if benchmark_accuracy else "Pending",
            },
            bottom_line=(
                f"TimeStone identified {len(viable)} actionable strategies for {company_name}, "
                f"with the top opportunity offering {best.get('mean_roi', 0):.0%} ROI "
                f"at {best.get('success_probability', 0):.0%} confidence."
            ),
        )

    def generate_validation_report(
        self,
        validation_results: Dict,
    ) -> CaseStudyReport:
        """Generate a validation report for due diligence."""
        now = datetime.now(timezone.utc).isoformat()

        sections = [
            ReportSection(
                title="Methodology",
                content=(
                    "TimeStone predictions were validated against historical transformation cases "
                    "with known outcomes. Each case was run through the simulation engine blind "
                    "(without knowledge of actual outcomes) and compared post-hoc."
                ),
            ),
            ReportSection(
                title="Results",
                content="",
                metrics={
                    "Total Cases": str(validation_results.get("n_cases", 0)),
                    "MAPE": f"{validation_results.get('overall_mape', 0):.1%}",
                    "Coverage": f"{validation_results.get('coverage', 0):.0%}",
                    "Calibration Score": f"{validation_results.get('calibration_score', 0):.2f}",
                    "Bias": f"{validation_results.get('bias', 0):+.2%}",
                },
            ),
            ReportSection(
                title="Supported Claims",
                content="\n".join(
                    f"- {claim}" for claim in validation_results.get("accuracy_claims_supported", [])
                ) or "No claims met threshold criteria.",
            ),
        ]

        return CaseStudyReport(
            company_name="TimeStone AI",
            industry="Platform Validation",
            generated_at=now,
            sections=sections,
            key_metrics={
                "Cases": str(validation_results.get("n_cases", 0)),
                "MAPE": f"{validation_results.get('overall_mape', 0):.1%}",
                "Coverage": f"{validation_results.get('coverage', 0):.0%}",
            },
            bottom_line="See Supported Claims section for investor-facing accuracy statements.",
        )

    @staticmethod
    def _risk_narrative(results: List[Dict]) -> str:
        high_risk = [r for r in results if r.get("ruin_probability", 0) > 0.05]
        if not high_risk:
            return "No scenarios carry significant tail risk (ruin probability < 5% across all)."

        names = [r.get("scenario_name", "Unknown")[:30] for r in high_risk[:3]]
        return (
            f"{len(high_risk)} scenario(s) show ruin probability above 5%: "
            f"{', '.join(names)}. "
            f"These should only be pursued with explicit risk mitigation plans."
        )

    def to_markdown(self, report: CaseStudyReport) -> str:
        """Render a report to Markdown."""
        lines = [
            f"# {report.company_name} — TimeStone AI Analysis",
            f"*{report.industry} | Generated {report.generated_at}*",
            "",
        ]

        if report.key_metrics:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in report.key_metrics.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append(section.content)

            if section.metrics:
                lines.append("")
                for k, v in section.metrics.items():
                    lines.append(f"- **{k}**: {v}")

            for sub in section.subsections:
                lines.append(f"### {sub.title}")
                lines.append(sub.content)

            lines.append("")

        lines.append("---")
        lines.append(f"**Bottom Line:** {report.bottom_line}")

        return "\n".join(lines)
