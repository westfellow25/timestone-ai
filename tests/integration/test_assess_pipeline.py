"""Integration test: full assess_company pipeline.

This is the highest-value test in the suite. It loads real data, runs
the full pipeline end-to-end, and verifies the resulting report has
the expected shape and reasonable values.
"""
from pathlib import Path
import pytest

from timestone.application import assess_company, AssessOptions
from timestone.domain.company import Company, CompanyMetrics
from timestone.repositories.case_library import CaseLibraryRepository
from timestone.repositories.results import ResultsRepository


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_assess_company_smoke(tmp_path):
    twin = Company(
        company_name="Test KTZ",
        metrics=CompanyMetrics(
            revenue=500_000_000, operating_costs=450_000_000, employees=120_000,
            industry="transportation_rail", geography="Kazakhstan",
            industry_tags=["transportation", "logistics"]))

    case_repo = CaseLibraryRepository(REPO_ROOT / "data" / "case_library.json")
    results_repo = ResultsRepository(runs_dir=tmp_path / "runs")

    opts = AssessOptions(scenario_count=20, iterations=100, random_seed=42)
    report = assess_company(twin, options=opts,
                            case_library_repo=case_repo,
                            results_repo=results_repo)

    assert report.run_id
    assert report.company_name == "Test KTZ"
    assert report.total_scenarios == 20
    assert report.case_library_size > 0
    assert len(report.top_recommendations) == 3
    # Run dir created with all files
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "scenarios.json").exists()
    assert (run_dir / "simulation.json").exists()
    assert (run_dir / "report.json").exists()


def test_results_repository_versions_runs(tmp_path):
    repo = ResultsRepository(runs_dir=tmp_path / "runs")
    a = repo.new_run("KTZ")
    b = repo.new_run("KTZ")
    assert a != b
    assert a.exists() and b.exists()


def test_outcome_records_are_append_only(tmp_path):
    from timestone.repositories.outcomes import OutcomesRepository
    from timestone.domain.outcome import OutcomeRecord

    repo = OutcomesRepository(outcomes_dir=tmp_path / "outcomes")
    rec = OutcomeRecord(
        id="ktz-dyn-pricing-2024",
        run_id="2024-01-01_ktz_test",
        company_name="KTZ", scenario_id=1, scenario_name="Dynamic Pricing",
        prediction_date="2024-01-01", measurement_date="2025-01-01",
        months_elapsed=12,
        predicted_mean_npv=10_000_000, predicted_success_probability=0.7,
        predicted_revenue_uplift_pct=0.03, predicted_cost_reduction_pct=0.01,
        predicted_investment_usd=2_000_000, predicted_payback_years=2.5,
    )
    repo.append(rec)
    with pytest.raises(FileExistsError):
        repo.append(rec)   # cannot delete or overwrite
    assert len(repo.list_all()) == 1
