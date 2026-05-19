"""Unit tests for Company / CompanyMetrics domain objects."""
import pytest

from timestone.domain.company import Company, CompanyMetrics


def test_profit_margin_basic():
    m = CompanyMetrics(revenue=1000.0, operating_costs=700.0)
    assert m.computed_profit_margin == pytest.approx(0.30)


def test_profit_margin_zero_revenue():
    m = CompanyMetrics(revenue=0.0, operating_costs=0.0)
    assert m.computed_profit_margin == 0.0


def test_revenue_per_employee():
    m = CompanyMetrics(revenue=1_000_000.0, operating_costs=800_000.0, employees=10)
    assert m.revenue_per_employee() == 100_000.0


def test_revenue_per_employee_zero_employees():
    m = CompanyMetrics(revenue=1000.0, operating_costs=800.0, employees=0)
    assert m.revenue_per_employee() == 0.0


def test_company_roundtrip():
    c = Company(
        company_name="Test",
        metrics=CompanyMetrics(
            revenue=1e9, operating_costs=9e8, employees=1000,
            industry="manufacturing", geography="USA",
            industry_tags=["manufacturing", "B2B"]))
    payload = c.to_dict()
    c2 = Company.from_dict(payload)
    assert c2.company_name == c.company_name
    assert c2.metrics.revenue == c.metrics.revenue
    assert c2.metrics.industry_tags == c.metrics.industry_tags


def test_company_from_legacy_format():
    """Old twin JSONs use 'metrics' dict — must still load."""
    legacy = {
        "company_name": "Legacy Co",
        "metrics": {"revenue": 100.0, "industry": "saas"}}
    c = Company.from_dict(legacy)
    assert c.company_name == "Legacy Co"
    assert c.metrics.revenue == 100.0
    assert c.metrics.industry == "saas"
