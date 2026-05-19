"""Scenario domain model - a generated transformation hypothesis."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TransformationType(Enum):
    """Types of business transformations."""
    DIGITAL = "digital_transformation"
    PRICING = "pricing_optimization"
    OPERATIONS = "operational_efficiency"
    AUTOMATION = "process_automation"
    MARKET_EXPANSION = "market_expansion"
    PRODUCT_INNOVATION = "product_innovation"
    SUPPLY_CHAIN = "supply_chain_optimization"
    CUSTOMER_EXPERIENCE = "customer_experience"


@dataclass
class Scenario:
    """A single transformation hypothesis to be evaluated."""
    id: int
    name: str
    transformation_type: TransformationType
    description: str
    expected_impact: Dict[str, float]
    investment_required: float
    implementation_time_months: int
    risk_level: str  # "low", "medium", "high"
    based_on_cases: List[str] = field(default_factory=list)
    empirical_prior: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.transformation_type.value,
            "description": self.description,
            "expected_impact": self.expected_impact,
            "investment_required": self.investment_required,
            "implementation_time_months": self.implementation_time_months,
            "risk_level": self.risk_level,
            "based_on_cases": self.based_on_cases,
            "empirical_prior": self.empirical_prior,
        }
