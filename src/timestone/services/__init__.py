"""Services - pure business logic operating on domain objects."""
from .knowledge_retrieval import CaseLibrary, revenue_to_bucket
from .scenario_generation import ScenarioGenerator
from .monte_carlo import MonteCarloSimulator
from .sensitivity import sensitivity_analysis, SensitivityRow
from .recommendation import build_report

__all__ = [
    "CaseLibrary", "revenue_to_bucket",
    "ScenarioGenerator",
    "MonteCarloSimulator",
    "sensitivity_analysis", "SensitivityRow",
    "build_report",
]
