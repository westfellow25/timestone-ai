"""Unit tests for ScenarioGenerator."""
from pathlib import Path
import json
import tempfile
import pytest

from timestone.repositories.case_library import CaseLibraryRepository
from timestone.services.knowledge_retrieval import CaseLibrary
from timestone.services.scenario_generation import ScenarioGenerator


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_LIB_PATH = REPO_ROOT / "data" / "case_library.json"


def test_generates_requested_count():
    gen = ScenarioGenerator("Test Co", "Transportation")
    scenarios = gen.generate(count=42)
    assert len(scenarios) == 42


def test_scenarios_have_required_fields():
    gen = ScenarioGenerator("Test Co", "Transportation")
    for s in gen.generate(count=10):
        assert s.id and s.name
        assert "revenue_increase" in s.expected_impact
        assert "cost_reduction" in s.expected_impact
        assert s.investment_required > 0
        assert s.implementation_time_months > 0
        assert s.risk_level in {"low", "medium", "high"}


def test_realistic_revenue_impact_caps():
    """Rule-based templates must produce <= 12% revenue uplift."""
    gen = ScenarioGenerator("Test Co", "Transportation")
    for s in gen.generate(count=500):
        assert s.expected_impact["revenue_increase"] <= 0.12


def test_scenario_generator_uses_case_library():
    lib = CaseLibrary(CaseLibraryRepository(CASE_LIB_PATH).load_all())
    gen = ScenarioGenerator(
        company_name="Test Co", industry="Transportation & Logistics",
        case_library=lib,
        company_profile={"industry": "transportation_rental",
                         "industry_tags": ["transportation", "logistics"],
                         "revenue_usd": 500_000_000, "geography": "USA"})
    scenarios = gen.generate(count=20)
    with_cases = [s for s in scenarios if s.based_on_cases]
    assert len(with_cases) > 0
    for s in with_cases:
        assert s.empirical_prior is not None
        assert "revenue_uplift" in s.empirical_prior


def test_blended_prior_caps_extreme_samples():
    lib = CaseLibrary(CaseLibraryRepository(CASE_LIB_PATH).load_all())
    gen = ScenarioGenerator(
        company_name="Test Co", industry="Transportation & Logistics",
        case_library=lib,
        company_profile={"industry": "transportation_rental",
                         "industry_tags": ["transportation"],
                         "revenue_usd": 500_000_000, "geography": "USA"})
    for s in gen.generate(count=500):
        assert -0.05 <= s.expected_impact["revenue_increase"] <= 0.15


def test_save_and_load_scenarios(tmp_path):
    gen = ScenarioGenerator("Test Co", "Transportation")
    gen.generate(count=10)
    # Round-trip via dict (no repository here - that's an integration test)
    payload = {"total_scenarios": len(gen.scenarios),
               "scenarios": [s.to_dict() for s in gen.scenarios]}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(payload))
    loaded = json.loads(p.read_text())
    assert loaded["total_scenarios"] == 10
    assert len(loaded["scenarios"]) == 10
