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

    @staticmethod
    def build_healthcare_graph() -> IndustryKnowledge:
        """Healthcare & Life Sciences industry knowledge."""
        graph = CausalGraph()

        graph.add_variable(CausalVariable("revenue", VariableType.FINANCIAL, 1.5e9, "USD", min_value=0, volatility=0.03))
        graph.add_variable(CausalVariable("patient_volume", VariableType.CUSTOMER, 200_000, "patients", min_value=0, volatility=0.02))
        graph.add_variable(CausalVariable("avg_revenue_per_patient", VariableType.FINANCIAL, 7_500, "USD", min_value=0, volatility=0.02))
        graph.add_variable(CausalVariable("clinical_outcomes_score", VariableType.OPERATIONAL, 0.78, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("readmission_rate", VariableType.OPERATIONAL, 0.14, "ratio", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("staff_utilization", VariableType.OPERATIONAL, 0.82, "ratio", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("ehr_adoption", VariableType.TECHNOLOGY, 0.65, "score", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("ai_diagnostic_accuracy", VariableType.TECHNOLOGY, 0.70, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("patient_satisfaction", VariableType.CUSTOMER, 0.72, "score", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("regulatory_compliance", VariableType.REGULATORY, 0.90, "score", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("operating_margin", VariableType.FINANCIAL, 0.08, "ratio", -0.2, 0.3, volatility=0.01))
        graph.add_variable(CausalVariable("talent_retention", VariableType.TALENT, 0.85, "ratio", 0, 1.0, volatility=0.02))

        graph.add_edge(CausalEdge("patient_volume", "revenue", strength=0.35, confidence=0.90))
        graph.add_edge(CausalEdge("avg_revenue_per_patient", "revenue", strength=0.30, confidence=0.90))
        graph.add_edge(CausalEdge("clinical_outcomes_score", "patient_satisfaction", strength=0.4, lag_periods=1, confidence=0.85))
        graph.add_edge(CausalEdge("patient_satisfaction", "patient_volume", strength=0.2, lag_periods=2, confidence=0.75))
        graph.add_edge(CausalEdge("readmission_rate", "revenue", strength=-0.15, lag_periods=1, confidence=0.80))
        graph.add_edge(CausalEdge("readmission_rate", "regulatory_compliance", strength=-0.25, lag_periods=2, confidence=0.75))
        graph.add_edge(CausalEdge("ehr_adoption", "clinical_outcomes_score", strength=0.2, lag_periods=3, confidence=0.70))
        graph.add_edge(CausalEdge("ehr_adoption", "staff_utilization", strength=0.15, lag_periods=2, confidence=0.65))
        graph.add_edge(CausalEdge("ai_diagnostic_accuracy", "clinical_outcomes_score", strength=0.25, lag_periods=2, confidence=0.65))
        graph.add_edge(CausalEdge("ai_diagnostic_accuracy", "readmission_rate", strength=-0.2, lag_periods=3, confidence=0.60))
        graph.add_edge(CausalEdge("staff_utilization", "operating_margin", strength=0.25, confidence=0.80))
        graph.add_edge(CausalEdge("talent_retention", "clinical_outcomes_score", strength=0.15, lag_periods=1, confidence=0.75))
        graph.add_edge(CausalEdge("talent_retention", "staff_utilization", strength=0.2, confidence=0.80))

        return IndustryKnowledge(
            industry_name="Healthcare & Life Sciences",
            description="Hospitals, health systems, pharma, medtech, and digital health",
            typical_causal_graph=graph,
            key_value_drivers=["clinical_outcomes_score", "patient_volume", "readmission_rate", "ai_diagnostic_accuracy"],
            common_failure_modes=[
                {"mode": "EHR implementation without workflow redesign", "impact": "Clinician burnout, data quality collapse", "probability": 0.30},
                {"mode": "AI tools without clinical validation", "impact": "Liability exposure, regulatory action", "probability": 0.15},
                {"mode": "Cost-cutting that harms outcomes", "impact": "Readmission penalties, reputation damage", "probability": 0.20},
                {"mode": "Staffing crisis from burnout", "impact": "Quality collapse, patient safety events", "probability": 0.25},
            ],
            benchmark_ranges={
                "readmission_rate": (0.08, 0.18),
                "patient_satisfaction": (0.65, 0.90),
                "operating_margin": (0.02, 0.15),
                "ehr_adoption": (0.40, 0.90),
            },
            transformation_success_rates={
                "ai_diagnostics": 0.55,
                "telehealth_expansion": 0.75,
                "value_based_care": 0.50,
                "ehr_modernization": 0.60,
                "precision_medicine": 0.40,
            },
            regulatory_constraints=[
                "HIPAA / patient data privacy",
                "FDA approval for AI/ML medical devices",
                "CMS quality reporting requirements",
                "State licensing and credentialing",
            ],
            technology_disruption_vectors=[
                "Foundation models for clinical decision support",
                "Digital therapeutics (DTx)",
                "Genomics-driven precision medicine",
                "Remote patient monitoring at scale",
            ],
        )

    @staticmethod
    def build_saas_graph() -> IndustryKnowledge:
        """SaaS / B2B Software industry knowledge."""
        graph = CausalGraph()

        graph.add_variable(CausalVariable("arr", VariableType.FINANCIAL, 50e6, "USD", min_value=0, volatility=0.03))
        graph.add_variable(CausalVariable("mrr_growth", VariableType.FINANCIAL, 0.05, "ratio", -0.1, 0.3, volatility=0.01))
        graph.add_variable(CausalVariable("gross_margin", VariableType.FINANCIAL, 0.78, "ratio", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("net_revenue_retention", VariableType.CUSTOMER, 1.15, "ratio", 0.5, 2.0, volatility=0.01))
        graph.add_variable(CausalVariable("logo_churn", VariableType.CUSTOMER, 0.05, "ratio", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("cac", VariableType.FINANCIAL, 25_000, "USD", min_value=0, volatility=0.05))
        graph.add_variable(CausalVariable("ltv", VariableType.CUSTOMER, 150_000, "USD", min_value=0, volatility=0.03))
        graph.add_variable(CausalVariable("product_market_fit", VariableType.STRATEGIC, 0.70, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("feature_velocity", VariableType.TECHNOLOGY, 0.65, "score", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("nps", VariableType.CUSTOMER, 42, "points", -100, 100, volatility=2.0))
        graph.add_variable(CausalVariable("sales_efficiency", VariableType.OPERATIONAL, 0.8, "ratio", 0, 3.0, volatility=0.05))
        graph.add_variable(CausalVariable("platform_stickiness", VariableType.STRATEGIC, 0.60, "score", 0, 1.0, volatility=0.005))

        graph.add_edge(CausalEdge("mrr_growth", "arr", strength=0.5, confidence=0.95))
        graph.add_edge(CausalEdge("net_revenue_retention", "arr", strength=0.35, lag_periods=1, confidence=0.90))
        graph.add_edge(CausalEdge("logo_churn", "arr", strength=-0.25, lag_periods=1, confidence=0.90))
        graph.add_edge(CausalEdge("product_market_fit", "nps", strength=15.0, lag_periods=2, confidence=0.80))
        graph.add_edge(CausalEdge("nps", "logo_churn", strength=-0.002, lag_periods=1, confidence=0.75))
        graph.add_edge(CausalEdge("nps", "net_revenue_retention", strength=0.003, lag_periods=2, confidence=0.70))
        graph.add_edge(CausalEdge("feature_velocity", "product_market_fit", strength=0.15, lag_periods=3, confidence=0.65))
        graph.add_edge(CausalEdge("feature_velocity", "platform_stickiness", strength=0.1, lag_periods=4, confidence=0.60))
        graph.add_edge(CausalEdge("platform_stickiness", "logo_churn", strength=-0.15, lag_periods=2, confidence=0.75))
        graph.add_edge(CausalEdge("platform_stickiness", "net_revenue_retention", strength=0.1, lag_periods=1, confidence=0.70))
        graph.add_edge(CausalEdge("sales_efficiency", "cac", strength=-5000, confidence=0.80))
        graph.add_edge(CausalEdge("ltv", "arr", strength=0.1, lag_periods=3, confidence=0.65))

        return IndustryKnowledge(
            industry_name="SaaS & B2B Software",
            description="Subscription software, PLG, enterprise SaaS, and developer tools",
            typical_causal_graph=graph,
            key_value_drivers=["net_revenue_retention", "product_market_fit", "logo_churn", "sales_efficiency"],
            common_failure_modes=[
                {"mode": "Growth without retention", "impact": "Leaky bucket — CAC never pays back", "probability": 0.25},
                {"mode": "Feature bloat without PMF", "impact": "Complexity kills adoption, churn spikes", "probability": 0.20},
                {"mode": "Premature enterprise push", "impact": "Sales cycle lengthens, burn rate explodes", "probability": 0.15},
                {"mode": "Pricing too low for value delivered", "impact": "NRR < 100%, can't fund R&D", "probability": 0.30},
            ],
            benchmark_ranges={
                "net_revenue_retention": (1.0, 1.4),
                "logo_churn": (0.02, 0.10),
                "gross_margin": (0.70, 0.90),
                "ltv_to_cac": (3.0, 8.0),
            },
            transformation_success_rates={
                "product_led_growth": 0.55,
                "ai_features": 0.65,
                "enterprise_expansion": 0.50,
                "platform_play": 0.40,
                "usage_based_pricing": 0.60,
            },
            regulatory_constraints=[
                "SOC 2 / ISO 27001 compliance",
                "GDPR / CCPA data handling",
                "Industry-specific compliance (HIPAA, FedRAMP)",
                "Open source license management",
            ],
            technology_disruption_vectors=[
                "AI copilots replacing point tools",
                "Vertical AI agents",
                "Composable architecture (API-first)",
                "Edge computing for latency-sensitive workloads",
            ],
        )

    @staticmethod
    def build_manufacturing_graph() -> IndustryKnowledge:
        """Manufacturing & Industrial industry knowledge."""
        graph = CausalGraph()

        graph.add_variable(CausalVariable("revenue", VariableType.FINANCIAL, 800e6, "USD", min_value=0, volatility=0.04))
        graph.add_variable(CausalVariable("oee", VariableType.OPERATIONAL, 0.72, "ratio", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("defect_rate", VariableType.OPERATIONAL, 0.03, "ratio", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("inventory_turnover", VariableType.OPERATIONAL, 6.0, "turns", 0, 30, volatility=0.3))
        graph.add_variable(CausalVariable("supply_chain_resilience", VariableType.OPERATIONAL, 0.65, "score", 0, 1.0, volatility=0.02))
        graph.add_variable(CausalVariable("energy_cost", VariableType.FINANCIAL, 30e6, "USD", min_value=0, volatility=0.08))
        graph.add_variable(CausalVariable("labor_productivity", VariableType.OPERATIONAL, 0.70, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("automation_level", VariableType.TECHNOLOGY, 0.45, "score", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("predictive_maintenance_maturity", VariableType.TECHNOLOGY, 0.30, "score", 0, 1.0, volatility=0.005))
        graph.add_variable(CausalVariable("time_to_market", VariableType.OPERATIONAL, 0.55, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("sustainability_score", VariableType.REGULATORY, 0.40, "score", 0, 1.0, volatility=0.01))
        graph.add_variable(CausalVariable("order_backlog", VariableType.CUSTOMER, 200e6, "USD", min_value=0, volatility=0.05))

        graph.add_edge(CausalEdge("oee", "revenue", strength=0.3, confidence=0.85))
        graph.add_edge(CausalEdge("order_backlog", "revenue", strength=0.25, lag_periods=2, confidence=0.85))
        graph.add_edge(CausalEdge("defect_rate", "revenue", strength=-0.2, lag_periods=1, confidence=0.80))
        graph.add_edge(CausalEdge("automation_level", "oee", strength=0.25, lag_periods=3, confidence=0.75))
        graph.add_edge(CausalEdge("automation_level", "labor_productivity", strength=0.3, lag_periods=2, confidence=0.80))
        graph.add_edge(CausalEdge("automation_level", "defect_rate", strength=-0.15, lag_periods=2, confidence=0.70))
        graph.add_edge(CausalEdge("predictive_maintenance_maturity", "oee", strength=0.2, lag_periods=4, confidence=0.70))
        graph.add_edge(CausalEdge("predictive_maintenance_maturity", "energy_cost", strength=-0.1, lag_periods=3, confidence=0.60))
        graph.add_edge(CausalEdge("supply_chain_resilience", "inventory_turnover", strength=1.5, lag_periods=2, confidence=0.70))
        graph.add_edge(CausalEdge("supply_chain_resilience", "time_to_market", strength=0.15, lag_periods=1, confidence=0.75))
        graph.add_edge(CausalEdge("inventory_turnover", "revenue", strength=0.05, lag_periods=1, confidence=0.60))
        graph.add_edge(CausalEdge("energy_cost", "revenue", strength=-0.05, confidence=0.85))
        graph.add_edge(CausalEdge("sustainability_score", "order_backlog", strength=0.1, lag_periods=4, confidence=0.55))

        return IndustryKnowledge(
            industry_name="Manufacturing & Industrial",
            description="Discrete and process manufacturing, industrial automation, and supply chain",
            typical_causal_graph=graph,
            key_value_drivers=["oee", "defect_rate", "automation_level", "supply_chain_resilience"],
            common_failure_modes=[
                {"mode": "Automation without process optimization", "impact": "Automate broken processes, waste capex", "probability": 0.25},
                {"mode": "Single-source supply chain", "impact": "Disruption halts production", "probability": 0.20},
                {"mode": "Deferred maintenance", "impact": "Catastrophic failure, safety incident", "probability": 0.10},
                {"mode": "Over-inventory to mask inefficiency", "impact": "Working capital trap, obsolescence", "probability": 0.30},
            ],
            benchmark_ranges={
                "oee": (0.60, 0.85),
                "defect_rate": (0.005, 0.05),
                "inventory_turnover": (4, 12),
                "automation_level": (0.30, 0.75),
            },
            transformation_success_rates={
                "industry_4_0": 0.50,
                "predictive_quality": 0.60,
                "digital_twin_ops": 0.55,
                "smart_supply_chain": 0.58,
                "lights_out_manufacturing": 0.30,
            },
            regulatory_constraints=[
                "Worker safety (OSHA / equivalent)",
                "Environmental emissions and waste",
                "Product safety and recalls",
                "Trade compliance and tariffs",
            ],
            technology_disruption_vectors=[
                "Generative design and additive manufacturing",
                "Autonomous mobile robots (AMR)",
                "Industrial IoT + edge AI",
                "Carbon-neutral manufacturing",
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
            "healthcare": cls.build_healthcare_graph,
            "pharma": cls.build_healthcare_graph,
            "saas": cls.build_saas_graph,
            "software": cls.build_saas_graph,
            "manufacturing": cls.build_manufacturing_graph,
            "industrial": cls.build_manufacturing_graph,
        }

        builder = industry_map.get(industry.lower())
        if builder:
            return builder()
        return None

    @classmethod
    def list_supported_industries(cls) -> List[str]:
        return ["transportation", "fintech", "energy", "healthcare", "saas", "manufacturing"]
