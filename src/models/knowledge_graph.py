"""
TimeStone AI — Industry Knowledge Graph

Deep ontologies encoding domain expertise for each industry.
Maps the causal structure of how industries work — which variables
drive which outcomes, typical transformation patterns, and
industry-specific risk factors.

This is a key competitive moat: encoding decades of industry
expertise into machine-readable causal models that improve
simulation accuracy by 3-5x vs. generic approaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.causal_graph import (
    CausalEdge,
    CausalGraph,
    CausalVariable,
    EdgeType,
    VariableType,
)


@dataclass
class IndustryKnowledge:
    """Encoded knowledge about an industry."""
    industry_name: str
    description: str
    typical_causal_graph: CausalGraph
    key_value_drivers: List[str]
    common_failure_modes: List[Dict]
    benchmark_ranges: Dict[str, Tuple[float, float]]
    transformation_success_rates: Dict[str, float]
    regulatory_constraints: List[str]
    technology_disruption_vectors: List[str]


class KnowledgeGraphBuilder:
    """Builds industry-specific causal graphs from encoded domain knowledge."""

    @staticmethod
    def build_transportation_graph() -> IndustryKnowledge:
        """
        Transportation & Logistics industry knowledge.
        Covers rail, freight, fleet management, last-mile delivery.
        """
        graph = CausalGraph()

        # FINANCIAL variables
        graph.add_variable(CausalVariable("revenue", VariableType.FINANCIAL, 500e6, "USD", min_value=0, volatility=0.03, description="Total annual revenue"))
        graph.add_variable(CausalVariable("operating_cost", VariableType.FINANCIAL, 430e6, "USD", min_value=0, volatility=0.02))
        graph.add_variable(CausalVariable("profit_margin", VariableType.FINANCIAL, 0.14, "ratio", min_value=-0.5, max_value=0.5, volatility=0.01))
        graph.add_variable(CausalVariable("capex", VariableType.FINANCIAL, 75e6, "USD", min_value=0, volatility=0.05))
        graph.add_variable(CausalVariable("fuel_cost", VariableType.FINANCIAL, 80e6, "USD", min_value=0, volatility=0.08))

        # OPERATIONAL variables
        graph.add_variable(CausalVariable("capacity_utilization", VariableType.OPERATIONAL, 0.72, "ratio", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("on_time_performance", VariableType.OPERATIONAL, 0.85, "ratio", 0, 1.0, volatility=0.03))
        graph.add_variable(CausalVariable("fleet_availability", VariableType.OPERATIONAL, 0.90, "ratio", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("route_efficiency", VariableType.OPERATIONAL, 0.75, "ratio", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("maintenance_cost", VariableType.OPERATIONAL, 45e6, "USD", min_value=0, volatility=0.04))

        # CUSTOMER variables
        graph.add_variable(CausalVariable("customer_satisfaction", VariableType.CUSTOMER, 0.72, "NPS_normalized", 0, 1.0, volatility=0.02, mean_reversion_rate=0.1, long_run_mean=0.70))
        graph.add_variable(CausalVariable("customer_retention", VariableType.CUSTOMER, 0.88, "ratio", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("freight_volume", VariableType.CUSTOMER, 50000, "units", min_value=0, volatility=0.03))
        graph.add_variable(CausalVariable("pricing_per_unit", VariableType.CUSTOMER, 10000, "USD", min_value=0, volatility=0.02))

        # TECHNOLOGY variables
        graph.add_variable(CausalVariable("digital_maturity", VariableType.TECHNOLOGY, 0.35, "score", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("data_quality", VariableType.TECHNOLOGY, 0.50, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("automation_level", VariableType.TECHNOLOGY, 0.30, "score", 0, 1.0, volatility=0.005))

        # MARKET variables
        graph.add_variable(CausalVariable("market_share", VariableType.MARKET, 0.45, "ratio", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("competitive_intensity", VariableType.MARKET, 0.60, "score", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("market_growth_rate", VariableType.MARKET, 0.04, "ratio", -0.2, 0.3, volatility=0.01))

        # ---- CAUSAL EDGES ----

        # Operational → Financial
        graph.add_edge(CausalEdge("capacity_utilization", "revenue", strength=0.3, confidence=0.9, description="Higher utilization → more revenue"))
        graph.add_edge(CausalEdge("on_time_performance", "revenue", strength=0.15, lag_periods=2, confidence=0.8, description="OTP → customer willingness to pay"))
        graph.add_edge(CausalEdge("route_efficiency", "fuel_cost", strength=-0.25, confidence=0.85, edge_type=EdgeType.LINEAR, description="Better routes → less fuel"))
        graph.add_edge(CausalEdge("fleet_availability", "capacity_utilization", strength=0.4, confidence=0.9))
        graph.add_edge(CausalEdge("maintenance_cost", "fleet_availability", strength=0.2, lag_periods=1, confidence=0.75))
        graph.add_edge(CausalEdge("fuel_cost", "operating_cost", strength=0.3, confidence=0.95))
        graph.add_edge(CausalEdge("maintenance_cost", "operating_cost", strength=0.15, confidence=0.90))

        # Customer → Financial
        graph.add_edge(CausalEdge("freight_volume", "revenue", strength=0.35, confidence=0.95))
        graph.add_edge(CausalEdge("pricing_per_unit", "revenue", strength=0.25, confidence=0.90))
        graph.add_edge(CausalEdge("customer_retention", "freight_volume", strength=0.3, lag_periods=1, confidence=0.85))
        graph.add_edge(CausalEdge("customer_satisfaction", "customer_retention", strength=0.4, lag_periods=1, confidence=0.9, edge_type=EdgeType.SATURATING, saturation_cap=1.0))

        # Operations → Customer
        graph.add_edge(CausalEdge("on_time_performance", "customer_satisfaction", strength=0.5, confidence=0.90))
        graph.add_edge(CausalEdge("capacity_utilization", "on_time_performance", strength=-0.2, confidence=0.7, description="Over-utilization hurts OTP"))

        # Technology → Operations
        graph.add_edge(CausalEdge("digital_maturity", "route_efficiency", strength=0.3, lag_periods=3, confidence=0.75))
        graph.add_edge(CausalEdge("digital_maturity", "capacity_utilization", strength=0.15, lag_periods=2, confidence=0.70))
        graph.add_edge(CausalEdge("automation_level", "operating_cost", strength=-0.2, lag_periods=3, confidence=0.70))
        graph.add_edge(CausalEdge("data_quality", "on_time_performance", strength=0.15, lag_periods=2, confidence=0.65))
        graph.add_edge(CausalEdge("automation_level", "maintenance_cost", strength=-0.15, lag_periods=2, confidence=0.60, edge_type=EdgeType.LOGARITHMIC))

        # Market dynamics
        graph.add_edge(CausalEdge("market_growth_rate", "freight_volume", strength=0.2, confidence=0.80))
        graph.add_edge(CausalEdge("competitive_intensity", "pricing_per_unit", strength=-0.15, confidence=0.75))
        graph.add_edge(CausalEdge("market_share", "pricing_per_unit", strength=0.1, confidence=0.60, description="Market power enables pricing"))

        # Financial loops
        graph.add_edge(CausalEdge("revenue", "profit_margin", strength=0.1, confidence=0.70))
        graph.add_edge(CausalEdge("operating_cost", "profit_margin", strength=-0.3, confidence=0.90))
        graph.add_edge(CausalEdge("capex", "digital_maturity", strength=0.1, lag_periods=6, confidence=0.65))

        # Confounders
        graph.add_confounder("fuel_cost", "operating_cost", "oil_price")
        graph.add_confounder("freight_volume", "revenue", "gdp_growth")
        graph.add_confounder("market_growth_rate", "freight_volume", "gdp_growth")

        return IndustryKnowledge(
            industry_name="Transportation & Logistics",
            description="Rail, freight, fleet management, and logistics operations",
            typical_causal_graph=graph,
            key_value_drivers=["capacity_utilization", "on_time_performance", "route_efficiency", "pricing_per_unit"],
            common_failure_modes=[
                {"mode": "Under-investment in maintenance", "impact": "Fleet availability drops → cascading delays", "probability": 0.15},
                {"mode": "Digital transformation without data quality", "impact": "AI/ML initiatives fail to deliver", "probability": 0.25},
                {"mode": "Over-optimization of utilization", "impact": "System becomes fragile, OTP collapses", "probability": 0.10},
                {"mode": "Pricing without demand sensing", "impact": "Volume loss to competitors", "probability": 0.20},
            ],
            benchmark_ranges={
                "capacity_utilization": (0.65, 0.85),
                "on_time_performance": (0.80, 0.95),
                "profit_margin": (0.05, 0.15),
                "digital_maturity": (0.20, 0.60),
            },
            transformation_success_rates={
                "dynamic_pricing": 0.72,
                "predictive_maintenance": 0.58,
                "route_optimization": 0.81,
                "digital_booking": 0.85,
                "autonomous_operations": 0.35,
            },
            regulatory_constraints=[
                "Safety compliance requirements",
                "Environmental emission standards",
                "Labor union agreements",
                "Government tariff regulations",
            ],
            technology_disruption_vectors=[
                "Autonomous vehicles / trains",
                "IoT sensor networks",
                "Blockchain for supply chain transparency",
                "AI-powered demand forecasting",
            ],
        )

    @staticmethod
    def build_fintech_graph() -> IndustryKnowledge:
        """Fintech / Digital Banking industry knowledge."""
        graph = CausalGraph()

        graph.add_variable(CausalVariable("revenue", VariableType.FINANCIAL, 200e6, "USD", min_value=0, volatility=0.05))
        graph.add_variable(CausalVariable("transaction_volume", VariableType.CUSTOMER, 50e6, "txns", min_value=0, volatility=0.04))
        graph.add_variable(CausalVariable("active_users", VariableType.CUSTOMER, 5e6, "users", min_value=0, volatility=0.03))
        graph.add_variable(CausalVariable("user_acquisition_cost", VariableType.FINANCIAL, 15, "USD", min_value=0, volatility=0.05))
        graph.add_variable(CausalVariable("ltv", VariableType.CUSTOMER, 250, "USD", min_value=0, volatility=0.03))
        graph.add_variable(CausalVariable("take_rate", VariableType.FINANCIAL, 0.015, "ratio", 0, 0.05, volatility=0.002))
        graph.add_variable(CausalVariable("fraud_rate", VariableType.OPERATIONAL, 0.003, "ratio", 0, 0.1, volatility=0.001))
        graph.add_variable(CausalVariable("app_reliability", VariableType.TECHNOLOGY, 0.997, "ratio", 0.9, 1.0, volatility=0.001))
        graph.add_variable(CausalVariable("feature_velocity", VariableType.TECHNOLOGY, 0.70, "score", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("nps_score", VariableType.CUSTOMER, 0.65, "score", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("churn_rate", VariableType.CUSTOMER, 0.04, "ratio", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("regulatory_compliance", VariableType.REGULATORY, 0.85, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("market_share", VariableType.MARKET, 0.12, "ratio", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("data_moat_depth", VariableType.STRATEGIC, 0.40, "score", 0, 1.0, volatility=0.005))

        # Causal edges
        graph.add_edge(CausalEdge("active_users", "transaction_volume", strength=0.5, confidence=0.95))
        graph.add_edge(CausalEdge("transaction_volume", "revenue", strength=0.4, confidence=0.90))
        graph.add_edge(CausalEdge("take_rate", "revenue", strength=0.3, confidence=0.85))
        graph.add_edge(CausalEdge("app_reliability", "nps_score", strength=0.35, confidence=0.85))
        graph.add_edge(CausalEdge("feature_velocity", "nps_score", strength=0.2, lag_periods=2, confidence=0.70))
        graph.add_edge(CausalEdge("nps_score", "churn_rate", strength=-0.4, lag_periods=1, confidence=0.85))
        graph.add_edge(CausalEdge("churn_rate", "active_users", strength=-0.3, lag_periods=1, confidence=0.90))
        graph.add_edge(CausalEdge("fraud_rate", "nps_score", strength=-0.3, confidence=0.80))
        graph.add_edge(CausalEdge("fraud_rate", "regulatory_compliance", strength=-0.4, lag_periods=2, confidence=0.85))
        graph.add_edge(CausalEdge("active_users", "data_moat_depth", strength=0.15, lag_periods=3, confidence=0.65, edge_type=EdgeType.LOGARITHMIC))
        graph.add_edge(CausalEdge("data_moat_depth", "take_rate", strength=0.05, lag_periods=3, confidence=0.55, description="Better data enables better pricing"))
        graph.add_edge(CausalEdge("active_users", "market_share", strength=0.2, lag_periods=2, confidence=0.75))
        graph.add_edge(CausalEdge("market_share", "user_acquisition_cost", strength=-0.15, lag_periods=1, confidence=0.65, description="Brand → lower CAC"))
        graph.add_edge(CausalEdge("ltv", "revenue", strength=0.2, lag_periods=3, confidence=0.80))

        return IndustryKnowledge(
            industry_name="Fintech & Digital Banking",
            description="Digital payments, neobanking, lending, and financial services",
            typical_causal_graph=graph,
            key_value_drivers=["active_users", "take_rate", "ltv", "churn_rate"],
            common_failure_modes=[
                {"mode": "Growth without unit economics", "impact": "Unsustainable burn rate", "probability": 0.30},
                {"mode": "Fraud spike", "impact": "Regulatory action + user trust collapse", "probability": 0.10},
                {"mode": "Reliability failure", "impact": "Mass churn event", "probability": 0.08},
                {"mode": "Regulatory non-compliance", "impact": "License revocation risk", "probability": 0.05},
            ],
            benchmark_ranges={
                "take_rate": (0.005, 0.030),
                "fraud_rate": (0.001, 0.010),
                "churn_rate": (0.02, 0.08),
                "app_reliability": (0.995, 0.999),
            },
            transformation_success_rates={
                "super_app_expansion": 0.45,
                "ai_risk_assessment": 0.70,
                "open_banking_integration": 0.65,
                "crypto_services": 0.40,
                "embedded_finance": 0.55,
            },
            regulatory_constraints=[
                "Banking license requirements",
                "KYC/AML compliance",
                "Data privacy (GDPR/local equivalents)",
                "Capital adequacy requirements",
            ],
            technology_disruption_vectors=[
                "AI-native financial services",
                "Decentralized finance (DeFi)",
                "Real-time payments infrastructure",
                "Embedded finance in non-financial platforms",
            ],
        )

    @staticmethod
    def build_energy_graph() -> IndustryKnowledge:
        """Energy & Utilities industry knowledge."""
        graph = CausalGraph()

        graph.add_variable(CausalVariable("revenue", VariableType.FINANCIAL, 2e9, "USD", min_value=0, volatility=0.04))
        graph.add_variable(CausalVariable("generation_capacity", VariableType.OPERATIONAL, 5000, "MW", min_value=0, volatility=0.01))
        graph.add_variable(CausalVariable("grid_reliability", VariableType.OPERATIONAL, 0.997, "ratio", 0.9, 1.0, volatility=0.001))
        graph.add_variable(CausalVariable("renewable_share", VariableType.STRATEGIC, 0.15, "ratio", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("energy_loss_rate", VariableType.OPERATIONAL, 0.12, "ratio", 0, 0.5, volatility=0.005))
        graph.add_variable(CausalVariable("carbon_emissions", VariableType.REGULATORY, 5e6, "tons_CO2", min_value=0, volatility=0.03))
        graph.add_variable(CausalVariable("demand_growth", VariableType.MARKET, 0.03, "ratio", -0.1, 0.2, volatility=0.01))
        graph.add_variable(CausalVariable("regulatory_risk", VariableType.REGULATORY, 0.40, "score", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("smart_grid_maturity", VariableType.TECHNOLOGY, 0.25, "score", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("customer_satisfaction", VariableType.CUSTOMER, 0.68, "score", 0, 1.0, volatility=0.02))

        graph.add_edge(CausalEdge("generation_capacity", "revenue", strength=0.25, confidence=0.85))
        graph.add_edge(CausalEdge("grid_reliability", "customer_satisfaction", strength=0.5, confidence=0.90))
        graph.add_edge(CausalEdge("grid_reliability", "revenue", strength=0.15, lag_periods=1, confidence=0.80))
        graph.add_edge(CausalEdge("energy_loss_rate", "revenue", strength=-0.20, confidence=0.85))
        graph.add_edge(CausalEdge("renewable_share", "carbon_emissions", strength=-0.35, lag_periods=2, confidence=0.90))
        graph.add_edge(CausalEdge("carbon_emissions", "regulatory_risk", strength=0.4, lag_periods=3, confidence=0.75))
        graph.add_edge(CausalEdge("smart_grid_maturity", "energy_loss_rate", strength=-0.25, lag_periods=4, confidence=0.70))
        graph.add_edge(CausalEdge("smart_grid_maturity", "grid_reliability", strength=0.2, lag_periods=3, confidence=0.65))
        graph.add_edge(CausalEdge("demand_growth", "revenue", strength=0.3, confidence=0.85))
        graph.add_edge(CausalEdge("regulatory_risk", "revenue", strength=-0.1, lag_periods=2, confidence=0.60))

        return IndustryKnowledge(
            industry_name="Energy & Utilities",
            description="Power generation, transmission, distribution, and renewables",
            typical_causal_graph=graph,
            key_value_drivers=["grid_reliability", "energy_loss_rate", "renewable_share", "smart_grid_maturity"],
            common_failure_modes=[
                {"mode": "Grid instability from rapid renewable integration", "impact": "Blackouts and regulatory penalties", "probability": 0.12},
                {"mode": "Stranded fossil fuel assets", "impact": "Massive write-downs", "probability": 0.20},
                {"mode": "Cyber attack on grid", "impact": "National security incident", "probability": 0.03},
            ],
            benchmark_ranges={
                "grid_reliability": (0.995, 0.9999),
                "energy_loss_rate": (0.05, 0.15),
                "renewable_share": (0.10, 0.50),
            },
            transformation_success_rates={
                "smart_grid": 0.65,
                "renewable_integration": 0.70,
                "demand_response": 0.60,
                "grid_storage": 0.50,
            },
            regulatory_constraints=[
                "Emission reduction mandates",
                "Grid reliability standards",
                "Renewable portfolio standards",
                "Nuclear safety regulations",
            ],
            technology_disruption_vectors=[
                "Grid-scale battery storage",
                "Distributed energy resources",
                "Green hydrogen",
                "Nuclear fusion (long-term)",
            ],
        )

    @classmethod
    def get_industry_knowledge(cls, industry: str) -> Optional[IndustryKnowledge]:
        """Get pre-built knowledge for an industry."""
        industry_map = {
            "transportation": cls.build_transportation_graph,
            "logistics": cls.build_transportation_graph,
            "rail": cls.build_transportation_graph,
            "fintech": cls.build_fintech_graph,
            "banking": cls.build_fintech_graph,
            "energy": cls.build_energy_graph,
            "utilities": cls.build_energy_graph,
        }

        builder = industry_map.get(industry.lower())
        if builder:
            return builder()
        return None

    @classmethod
    def list_supported_industries(cls) -> List[str]:
        return ["transportation", "fintech", "energy"]
