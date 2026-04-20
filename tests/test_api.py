"""
TimeStone AI — API Tests

Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer ts-demo-key-2026"}


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_info_endpoint(self, client):
        response = client.get("/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert len(data["capabilities"]) > 5
        assert "supported_industries" in data


class TestAuth:
    def test_no_auth(self, client):
        response = client.post("/v1/companies", json={
            "name": "Test", "industry": "fintech",
            "revenue": 1e6, "operating_costs": 8e5,
            "employee_count": 50, "market_share": 0.01,
        })
        assert response.status_code == 401

    def test_invalid_auth(self, client):
        response = client.post(
            "/v1/companies",
            json={
                "name": "Test", "industry": "fintech",
                "revenue": 1e6, "operating_costs": 8e5,
                "employee_count": 50, "market_share": 0.01,
            },
            headers={"Authorization": "Bearer invalid-key"},
        )
        assert response.status_code == 403

    def test_valid_auth(self, client, auth_headers):
        response = client.post(
            "/v1/companies",
            json={
                "name": "TestAuth Corp", "industry": "fintech",
                "revenue": 1e6, "operating_costs": 8e5,
                "employee_count": 50, "market_share": 0.01,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200


class TestCompanies:
    def test_create_company(self, client, auth_headers):
        response = client.post(
            "/v1/companies",
            json={
                "name": "KTZ Railway",
                "industry": "transportation",
                "revenue": 500_000_000,
                "operating_costs": 430_000_000,
                "employee_count": 10_000,
                "market_share": 0.85,
                "metadata": {"country": "Kazakhstan"},
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "company_id" in data
        assert data["name"] == "KTZ Railway"

    def test_list_companies(self, client, auth_headers):
        # Create a company first
        client.post("/v1/companies", json={
            "name": "Test List", "industry": "energy",
            "revenue": 2e9, "operating_costs": 1.5e9,
            "employee_count": 5000, "market_share": 0.30,
        }, headers=auth_headers)

        response = client.get("/v1/companies", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["total"] >= 1


class TestSimulation:
    def test_full_simulation(self, client, auth_headers):
        # Create company
        create_resp = client.post("/v1/companies", json={
            "name": "Sim Test Corp",
            "industry": "transportation",
            "revenue": 500_000_000,
            "operating_costs": 430_000_000,
            "employee_count": 10_000,
            "market_share": 0.85,
        }, headers=auth_headers)
        company_id = create_resp.json()["company_id"]

        # Run simulation
        response = client.post("/v1/simulate", json={
            "company_id": company_id,
            "scenarios": [
                {
                    "name": "Dynamic Pricing",
                    "type": "pricing_optimization",
                    "expected_revenue_increase": 0.15,
                    "expected_cost_reduction": 0.05,
                    "investment_required": 3_000_000,
                    "implementation_time_months": 9,
                    "risk_level": "medium",
                },
                {
                    "name": "Route Optimization",
                    "type": "operational_efficiency",
                    "expected_revenue_increase": 0.08,
                    "expected_cost_reduction": 0.12,
                    "investment_required": 5_000_000,
                    "implementation_time_months": 12,
                    "risk_level": "low",
                },
            ],
            "iterations": 1000,
            "confidence_level": 0.90,
        }, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert "simulation_id" in data
        assert len(data["results"]) == 2
        assert "insights" in data
        assert "recommendations" in data
        assert "executive_summary" in data

        # Validate result structure
        for r in data["results"]:
            assert "mean_roi" in r
            assert "success_probability" in r
            assert "value_at_risk_95" in r
            assert "sharpe_ratio" in r
            assert "percentiles" in r


class TestGenome:
    def test_create_genome(self, client, auth_headers):
        response = client.post("/v1/genome", json={
            "company_name": "TestCorp",
            "industry": "technology",
            "factors": [
                {"name": "revenue_growth_rate", "value": 0.75},
                {"name": "profit_margin", "value": 0.60},
                {"name": "digital_infrastructure_score", "value": 0.85},
                {"name": "data_readiness", "value": 0.70},
            ],
        }, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "genome_id" in data
        assert "overall_score" in data
        assert len(data["genome_vector"]) == 48

    def test_readiness_assessment(self, client, auth_headers):
        # Create genome
        create_resp = client.post("/v1/genome", json={
            "company_name": "ReadyTest",
            "industry": "fintech",
            "factors": [
                {"name": "digital_infrastructure_score", "value": 0.8},
                {"name": "data_readiness", "value": 0.7},
                {"name": "change_management_capability", "value": 0.6},
                {"name": "talent_density", "value": 0.65},
            ],
        }, headers=auth_headers)
        genome_id = create_resp.json()["genome_id"]

        response = client.get(
            f"/v1/genome/{genome_id}/readiness?transformation_type=digital_transformation",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "readiness_score" in data
        assert "readiness_grade" in data


class TestIndustries:
    def test_list_industries(self, client, auth_headers):
        response = client.get("/v1/industries", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["industries"]) >= 3
        for ind in data["industries"]:
            assert "name" in ind
            assert "transformation_success_rates" in ind
