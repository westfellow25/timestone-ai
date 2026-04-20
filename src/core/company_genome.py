"""
TimeStone AI — Company Genome

Multi-dimensional DNA encoding of a company's characteristics.
The Genome captures not just financial metrics but organizational
capability, technology maturity, market position, culture, and
transformation readiness across 6 dimensions and 48 sub-factors.

This enables:
- Cross-company comparison regardless of industry
- Transformation readiness scoring
- Identification of capability gaps
- Transfer learning of transformation outcomes across similar genomes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class GenomeDimension(Enum):
    FINANCIAL_HEALTH = "financial_health"
    OPERATIONAL_EXCELLENCE = "operational_excellence"
    MARKET_POSITION = "market_position"
    TECHNOLOGY_MATURITY = "technology_maturity"
    ORGANIZATIONAL_CAPABILITY = "organizational_capability"
    INNOVATION_CULTURE = "innovation_culture"


# 8 sub-factors per dimension = 48 total genome factors
GENOME_FACTORS: Dict[GenomeDimension, List[str]] = {
    GenomeDimension.FINANCIAL_HEALTH: [
        "revenue_growth_rate",
        "profit_margin",
        "cash_flow_stability",
        "debt_to_equity",
        "return_on_invested_capital",
        "revenue_concentration",
        "working_capital_efficiency",
        "capex_to_revenue",
    ],
    GenomeDimension.OPERATIONAL_EXCELLENCE: [
        "process_automation_level",
        "supply_chain_resilience",
        "quality_defect_rate",
        "capacity_utilization",
        "cycle_time_efficiency",
        "asset_turnover",
        "operational_downtime",
        "cost_per_unit",
    ],
    GenomeDimension.MARKET_POSITION: [
        "market_share",
        "brand_strength",
        "customer_retention",
        "pricing_power",
        "geographic_diversification",
        "customer_concentration",
        "competitive_moat_score",
        "market_growth_rate",
    ],
    GenomeDimension.TECHNOLOGY_MATURITY: [
        "digital_infrastructure_score",
        "data_readiness",
        "cloud_adoption",
        "cybersecurity_posture",
        "api_ecosystem_maturity",
        "ai_ml_capability",
        "legacy_system_burden",
        "tech_debt_ratio",
    ],
    GenomeDimension.ORGANIZATIONAL_CAPABILITY: [
        "talent_density",
        "leadership_quality",
        "change_management_capability",
        "cross_functional_collaboration",
        "decision_making_speed",
        "knowledge_management",
        "succession_planning",
        "employee_engagement",
    ],
    GenomeDimension.INNOVATION_CULTURE: [
        "r_and_d_intensity",
        "experimentation_velocity",
        "failure_tolerance",
        "external_partnership_density",
        "patent_portfolio_strength",
        "time_to_market",
        "intrapreneurship_score",
        "innovation_pipeline_depth",
    ],
}


@dataclass
class GenomeFactor:
    """A single measurable factor in the company genome."""
    name: str
    dimension: GenomeDimension
    value: float              # normalized to [0, 1]
    raw_value: float = 0.0    # original scale
    weight: float = 1.0       # importance weight
    benchmark_median: float = 0.5
    benchmark_p90: float = 0.8
    confidence: float = 0.8   # data quality / measurement confidence
    source: str = ""          # where this data comes from

    @property
    def percentile_rank(self) -> float:
        """Approximate percentile rank assuming beta distribution around benchmark."""
        if self.value >= self.benchmark_p90:
            return 0.9 + 0.1 * (self.value - self.benchmark_p90) / (1.0 - self.benchmark_p90 + 1e-9)
        if self.value >= self.benchmark_median:
            return 0.5 + 0.4 * (self.value - self.benchmark_median) / (self.benchmark_p90 - self.benchmark_median + 1e-9)
        return 0.5 * self.value / (self.benchmark_median + 1e-9)


@dataclass
class DimensionScore:
    """Aggregate score for one genome dimension."""
    dimension: GenomeDimension
    score: float
    factors: List[GenomeFactor]
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


class CompanyGenome:
    """
    48-factor DNA encoding of a company.

    The genome enables:
    1. Transformation Readiness Assessment
    2. Cross-company similarity (genome distance)
    3. Capability gap identification
    4. Outcome prediction based on similar genomes
    """

    def __init__(self, company_name: str, industry: str):
        self.company_name = company_name
        self.industry = industry
        self.factors: Dict[str, GenomeFactor] = {}
        self._dimension_weights: Dict[GenomeDimension, float] = {
            d: 1.0 for d in GenomeDimension
        }

    def set_factor(
        self,
        name: str,
        value: float,
        raw_value: float = 0.0,
        confidence: float = 0.8,
        source: str = "",
    ) -> None:
        """Set a genome factor value (automatically finds its dimension)."""
        dimension = self._find_dimension(name)
        if dimension is None:
            raise ValueError(f"Unknown genome factor: {name}")

        self.factors[name] = GenomeFactor(
            name=name,
            dimension=dimension,
            value=max(0.0, min(1.0, value)),
            raw_value=raw_value,
            confidence=confidence,
            source=source,
        )

    def set_dimension_weight(self, dimension: GenomeDimension, weight: float) -> None:
        self._dimension_weights[dimension] = weight

    def _find_dimension(self, factor_name: str) -> Optional[GenomeDimension]:
        for dim, factors in GENOME_FACTORS.items():
            if factor_name in factors:
                return dim
        return None

    # ---- scoring ----

    def get_dimension_score(self, dimension: GenomeDimension) -> DimensionScore:
        """Compute aggregate score for a dimension."""
        dim_factors = [
            self.factors[f]
            for f in GENOME_FACTORS[dimension]
            if f in self.factors
        ]

        if not dim_factors:
            return DimensionScore(dimension=dimension, score=0.0, factors=[])

        weighted_sum = sum(f.value * f.weight * f.confidence for f in dim_factors)
        total_weight = sum(f.weight * f.confidence for f in dim_factors)
        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        strengths = [f.name for f in dim_factors if f.value >= 0.7]
        weaknesses = [f.name for f in dim_factors if f.value < 0.4]

        return DimensionScore(
            dimension=dimension,
            score=score,
            factors=dim_factors,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def get_overall_score(self) -> float:
        """Weighted average across all dimensions."""
        total = 0.0
        weight_sum = 0.0
        for dim in GenomeDimension:
            ds = self.get_dimension_score(dim)
            w = self._dimension_weights[dim]
            total += ds.score * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.0

    def get_genome_vector(self) -> np.ndarray:
        """Return the 48-dimensional genome vector (for ML / similarity)."""
        vector = []
        for dim in GenomeDimension:
            for factor_name in GENOME_FACTORS[dim]:
                if factor_name in self.factors:
                    vector.append(self.factors[factor_name].value)
                else:
                    vector.append(0.0)
        return np.array(vector)

    # ---- transformation readiness ----

    def transformation_readiness(self, transformation_type: str) -> Dict:
        """
        Assess readiness for a specific type of transformation.

        Different transformations require different capability profiles:
        - digital_transformation: high tech maturity + change mgmt
        - pricing_optimization: data readiness + decision speed
        - process_automation: process maturity + tech infrastructure
        - market_expansion: financial health + brand + talent
        """
        readiness_profiles: Dict[str, Dict[str, float]] = {
            "digital_transformation": {
                "digital_infrastructure_score": 0.20,
                "data_readiness": 0.15,
                "cloud_adoption": 0.10,
                "change_management_capability": 0.15,
                "talent_density": 0.10,
                "ai_ml_capability": 0.10,
                "leadership_quality": 0.10,
                "legacy_system_burden": 0.10,
            },
            "pricing_optimization": {
                "data_readiness": 0.20,
                "decision_making_speed": 0.15,
                "pricing_power": 0.15,
                "ai_ml_capability": 0.15,
                "customer_retention": 0.10,
                "digital_infrastructure_score": 0.10,
                "competitive_moat_score": 0.10,
                "experimentation_velocity": 0.05,
            },
            "process_automation": {
                "process_automation_level": 0.15,
                "digital_infrastructure_score": 0.15,
                "data_readiness": 0.10,
                "cycle_time_efficiency": 0.10,
                "change_management_capability": 0.10,
                "talent_density": 0.10,
                "tech_debt_ratio": 0.10,
                "api_ecosystem_maturity": 0.10,
                "cost_per_unit": 0.10,
            },
            "market_expansion": {
                "cash_flow_stability": 0.15,
                "brand_strength": 0.15,
                "talent_density": 0.10,
                "geographic_diversification": 0.10,
                "revenue_growth_rate": 0.10,
                "leadership_quality": 0.10,
                "market_growth_rate": 0.10,
                "customer_retention": 0.10,
                "competitive_moat_score": 0.10,
            },
        }

        profile = readiness_profiles.get(transformation_type)
        if profile is None:
            profile = {f: 1.0 / 48 for dim in GENOME_FACTORS.values() for f in dim}

        readiness_score = 0.0
        gaps: List[Dict] = []
        enablers: List[Dict] = []

        for factor_name, weight in profile.items():
            if factor_name in self.factors:
                f = self.factors[factor_name]
                readiness_score += f.value * weight

                if f.value < 0.4:
                    gaps.append({
                        "factor": factor_name,
                        "current": f.value,
                        "required_minimum": 0.5,
                        "impact_weight": weight,
                        "gap_severity": (0.5 - f.value) * weight,
                    })
                elif f.value >= 0.7:
                    enablers.append({
                        "factor": factor_name,
                        "current": f.value,
                        "impact_weight": weight,
                    })
            else:
                gaps.append({
                    "factor": factor_name,
                    "current": None,
                    "required_minimum": 0.5,
                    "impact_weight": weight,
                    "gap_severity": 0.5 * weight,
                })

        gaps.sort(key=lambda x: x["gap_severity"], reverse=True)
        enablers.sort(key=lambda x: x["impact_weight"], reverse=True)

        total_weight = sum(profile.values())
        normalized_score = readiness_score / total_weight if total_weight > 0 else 0.0

        return {
            "transformation_type": transformation_type,
            "readiness_score": normalized_score,
            "readiness_grade": self._grade(normalized_score),
            "critical_gaps": gaps[:5],
            "key_enablers": enablers[:5],
            "recommendation": self._readiness_recommendation(normalized_score, gaps),
        }

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.8:
            return "A"
        if score >= 0.65:
            return "B"
        if score >= 0.5:
            return "C"
        if score >= 0.35:
            return "D"
        return "F"

    @staticmethod
    def _readiness_recommendation(score: float, gaps: List[Dict]) -> str:
        if score >= 0.75:
            return "PROCEED — Organization is well-positioned for this transformation."
        if score >= 0.55:
            critical = [g["factor"] for g in gaps[:3] if g.get("gap_severity", 0) > 0.05]
            gap_str = ", ".join(critical) if critical else "general capability"
            return f"PROCEED WITH PREPARATION — Address gaps in {gap_str} before scaling."
        if score >= 0.35:
            return "PILOT ONLY — Significant capability gaps. Run small-scale proof of concept first."
        return "NOT READY — Fundamental prerequisites missing. Build foundational capabilities first."

    # ---- similarity ----

    def genome_distance(self, other: "CompanyGenome", metric: str = "cosine") -> float:
        """
        Compute distance between two company genomes.

        Supports: cosine, euclidean, manhattan, weighted_euclidean
        """
        v1 = self.get_genome_vector()
        v2 = other.get_genome_vector()

        if metric == "cosine":
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            return 1.0 - (dot / (norm1 * norm2))

        if metric == "euclidean":
            return float(np.linalg.norm(v1 - v2))

        if metric == "manhattan":
            return float(np.sum(np.abs(v1 - v2)))

        if metric == "weighted_euclidean":
            weights = self._get_weight_vector()
            return float(np.sqrt(np.sum(weights * (v1 - v2) ** 2)))

        return float(np.linalg.norm(v1 - v2))

    def find_similar_companies(
        self,
        company_pool: List["CompanyGenome"],
        top_n: int = 5,
        metric: str = "cosine",
    ) -> List[Tuple["CompanyGenome", float]]:
        """Find the most similar companies from a pool."""
        distances = [(c, self.genome_distance(c, metric)) for c in company_pool]
        distances.sort(key=lambda x: x[1])
        return distances[:top_n]

    def _get_weight_vector(self) -> np.ndarray:
        weights = []
        for dim in GenomeDimension:
            dim_weight = self._dimension_weights[dim]
            for _ in GENOME_FACTORS[dim]:
                weights.append(dim_weight)
        return np.array(weights)

    # ---- capability gap analysis ----

    def capability_gap_analysis(self, target_genome: "CompanyGenome") -> Dict:
        """
        Compare this genome against a target (best-in-class or desired state).
        Identifies specific gaps and their business impact.
        """
        v_current = self.get_genome_vector()
        v_target = target_genome.get_genome_vector()
        gaps = v_target - v_current

        all_factors = []
        idx = 0
        for dim in GenomeDimension:
            for factor_name in GENOME_FACTORS[dim]:
                all_factors.append({
                    "factor": factor_name,
                    "dimension": dim.value,
                    "current": float(v_current[idx]),
                    "target": float(v_target[idx]),
                    "gap": float(gaps[idx]),
                    "gap_pct": float(gaps[idx] / (v_target[idx] + 1e-9) * 100),
                })
                idx += 1

        significant_gaps = [f for f in all_factors if f["gap"] > 0.1]
        significant_gaps.sort(key=lambda x: x["gap"], reverse=True)

        surplus = [f for f in all_factors if f["gap"] < -0.1]
        surplus.sort(key=lambda x: x["gap"])

        overall_gap = float(np.mean(np.clip(gaps, 0, None)))

        return {
            "overall_gap_score": overall_gap,
            "critical_gaps": significant_gaps[:10],
            "surplus_capabilities": surplus[:5],
            "dimension_gaps": {
                dim.value: float(np.mean([
                    g["gap"] for g in all_factors
                    if g["dimension"] == dim.value and g["gap"] > 0
                ] or [0.0]))
                for dim in GenomeDimension
            },
        }

    # ---- serialization ----

    def to_dict(self) -> Dict:
        return {
            "company_name": self.company_name,
            "industry": self.industry,
            "overall_score": self.get_overall_score(),
            "dimensions": {
                dim.value: {
                    "score": self.get_dimension_score(dim).score,
                    "strengths": self.get_dimension_score(dim).strengths,
                    "weaknesses": self.get_dimension_score(dim).weaknesses,
                }
                for dim in GenomeDimension
            },
            "factors": {
                name: {
                    "value": f.value,
                    "raw_value": f.raw_value,
                    "dimension": f.dimension.value,
                    "confidence": f.confidence,
                    "percentile_rank": f.percentile_rank,
                }
                for name, f in self.factors.items()
            },
            "genome_vector": self.get_genome_vector().tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CompanyGenome":
        genome = cls(data["company_name"], data["industry"])
        for name, fdata in data.get("factors", {}).items():
            genome.set_factor(
                name=name,
                value=fdata["value"],
                raw_value=fdata.get("raw_value", 0.0),
                confidence=fdata.get("confidence", 0.8),
            )
        return genome
