"""Company / digital twin domain model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CompanyMetrics:
    """Baseline financial and operational metrics of a company."""
    revenue: float
    operating_costs: float
    employees: int = 0
    profit_margin: float = 0.0
    market_share: float = 0.0
    industry: str = ""
    geography: str = ""
    industry_tags: List[str] = field(default_factory=list)

    def revenue_per_employee(self) -> float:
        if self.employees == 0:
            return 0.0
        return self.revenue / self.employees

    @property
    def computed_profit_margin(self) -> float:
        if self.revenue == 0:
            return 0.0
        return (self.revenue - self.operating_costs) / self.revenue


@dataclass
class Company:
    """Digital twin of a target company."""
    company_name: str
    metrics: CompanyMetrics
    notes: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> "Company":
        m = data.get("metrics", {})
        return cls(
            company_name=data.get("company_name", ""),
            metrics=CompanyMetrics(
                revenue=float(m.get("revenue", 0)),
                operating_costs=float(m.get("operating_costs", m.get("revenue", 0) * 0.9)),
                employees=int(m.get("employees", 0)),
                profit_margin=float(m.get("profit_margin", 0)),
                market_share=float(m.get("market_share", 0)),
                industry=m.get("industry", ""),
                geography=m.get("geography", data.get("geography", "")),
                industry_tags=m.get("industry_tags", []),
            ),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
        )

    def to_dict(self) -> Dict:
        return {
            "company_name": self.company_name,
            "metrics": {
                "revenue": self.metrics.revenue,
                "operating_costs": self.metrics.operating_costs,
                "employees": self.metrics.employees,
                "profit_margin": self.metrics.profit_margin,
                "market_share": self.metrics.market_share,
                "industry": self.metrics.industry,
                "geography": self.metrics.geography,
                "industry_tags": self.metrics.industry_tags,
            },
            "notes": self.notes,
            "created_at": self.created_at,
        }
