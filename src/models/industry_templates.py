"""
Industry-specific templates for Digital Twins and Scenarios
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class Industry(Enum):
    """Supported industries"""
    TRANSPORTATION = "transportation"
    ENERGY = "energy"
    FINTECH = "fintech"
    SAAS = "saas"
    MANUFACTURING = "manufacturing"


@dataclass
class IndustryTemplate:
    """Template for industry-specific configurations"""
    industry: Industry
    typical_revenue_range: tuple
    typical_margin_range: tuple
    key_metrics: List[str]
    transformation_focus_areas: List[str]


# Industry Templates Database
INDUSTRY_TEMPLATES = {
    Industry.TRANSPORTATION: IndustryTemplate(
        industry=Industry.TRANSPORTATION,
        typical_revenue_range=(100_000_000, 10_000_000_000),
        typical_margin_range=(0.05, 0.15),
        key_metrics=["on_time_performance", "capacity_utilization", "fuel_efficiency"],
        transformation_focus_areas=["Digital Transformation", "Route Optimization", "Predictive Maintenance"]
    ),
    Industry.ENERGY: IndustryTemplate(
        industry=Industry.ENERGY,
        typical_revenue_range=(500_000_000, 50_000_000_000),
        typical_margin_range=(0.10, 0.25),
        key_metrics=["grid_reliability", "energy_loss_rate", "renewable_percentage"],
        transformation_focus_areas=["Smart Grid", "Renewable Integration", "Demand Response"]
    ),
    Industry.FINTECH: IndustryTemplate(
        industry=Industry.FINTECH,
        typical_revenue_range=(10_000_000, 5_000_000_000),
        typical_margin_range=(0.20, 0.40),
        key_metrics=["transaction_volume", "active_users", "fraud_rate"],
        transformation_focus_areas=["AI Risk Assessment", "Super App Strategy", "Open Banking"]
    ),
    Industry.SAAS: IndustryTemplate(
        industry=Industry.SAAS,
        typical_revenue_range=(1_000_000, 1_000_000_000),
        typical_margin_range=(0.15, 0.35),
        key_metrics=["mrr_growth", "churn_rate", "customer_lifetime_value"],
        transformation_focus_areas=["Product-Led Growth", "AI Features", "Enterprise Expansion"]
    ),
    Industry.MANUFACTURING: IndustryTemplate(
        industry=Industry.MANUFACTURING,
        typical_revenue_range=(50_000_000, 20_000_000_000),
        typical_margin_range=(0.08, 0.18),
        key_metrics=["oee", "defect_rate", "inventory_turnover"],
        transformation_focus_areas=["Industry 4.0", "Robotics", "Predictive Quality"]
    )
}


if __name__ == "__main__":
    print("Industry Templates Overview")
    print("=" * 70)
    for industry in Industry:
        template = INDUSTRY_TEMPLATES.get(industry)
        if template:
            print(f"\n{industry.value.upper()}")
            print(f"  Revenue Range: ${template.typical_revenue_range[0]:,} - ${template.typical_revenue_range[1]:,}")
            print(f"  Key Metrics: {', '.join(template.key_metrics)}")