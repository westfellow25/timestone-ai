"""
Tests for enterprise modules: persistence, auth, connectors, discovery,
validation, RL optimizer, observability, reporting.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---- Persistence ----

from src.persistence.database import initialize_database, get_session, reset_database
from src.persistence.models import Base
from src.persistence.repositories import (
    TenantRepository,
    UserRepository,
    CompanyRepository,
    SimulationRepository,
    AuditLogRepository,
    BenchmarkRepository,
)


@pytest.fixture(scope="module")
def db_engine():
    engine = initialize_database("sqlite:///:memory:", echo=False)
    yield engine
    reset_database()


@pytest.fixture
def session(db_engine):
    with get_session() as s:
        yield s


class TestPersistence:
    def test_create_tenant(self, session):
        repo = TenantRepository(session)
        tenant = repo.create("Test Corp", "test-corp", plan="enterprise")
        assert tenant.id is not None
        assert tenant.slug == "test-corp"

    def test_create_user(self, session):
        tenant_repo = TenantRepository(session)
        tenant = tenant_repo.create("User Test", "user-test")
        user_repo = UserRepository(session, tenant.id)
        user = user_repo.create("alice@test.com", "hash123", "Alice", "admin")
        assert user.email == "alice@test.com"
        assert user.role == "admin"

    def test_tenant_isolation(self, session):
        repo1 = TenantRepository(session)
        t1 = repo1.create("Tenant A", "tenant-a")
        t2 = repo1.create("Tenant B", "tenant-b")

        comp_a = CompanyRepository(session, t1.id)
        comp_a.create("Company A", "fintech", 100e6, 80e6, 500, 0.1)

        comp_b = CompanyRepository(session, t2.id)
        companies_b = comp_b.list()
        assert len(companies_b) == 0  # tenant B can't see tenant A's data

    def test_company_crud(self, session):
        tenant = TenantRepository(session).create("CRUD Test", "crud-test")
        repo = CompanyRepository(session, tenant.id)

        company = repo.create("My Co", "transportation", 500e6, 400e6, 10000, 0.5)
        assert company.name == "My Co"

        fetched = repo.get(company.id)
        assert fetched is not None

        companies = repo.list()
        assert len(companies) >= 1

    def test_audit_log(self, session):
        tenant = TenantRepository(session).create("Audit Test", "audit-test")
        repo = AuditLogRepository(session, tenant.id)
        entry = repo.log("simulate", resource_type="simulation", resource_id="sim-1")
        assert entry.action == "simulate"
        recent = repo.recent(10)
        assert len(recent) >= 1

    def test_benchmark_cases(self, session):
        repo = BenchmarkRepository(session)
        case = repo.add(
            "Test Case", "fintech", "digital_transformation", 2020,
            {"revenue": 1e9}, {"revenue_3y": 1.3e9}, "test",
        )
        assert case.industry == "fintech"
        cases = repo.list_by_industry("fintech")
        assert len(cases) >= 1


# ---- Auth ----

from src.auth.security import (
    hash_password, verify_password,
    generate_api_key, hash_api_key,
    create_access_token, create_refresh_token, decode_token,
    has_permission, require_role,
)


class TestAuth:
    def test_password_hashing(self):
        hashed = hash_password("secret123")
        assert verify_password("secret123", hashed)
        assert not verify_password("wrong", hashed)

    def test_api_key_generation(self):
        plaintext, key_hash = generate_api_key("ts")
        assert plaintext.startswith("ts_")
        assert hash_api_key(plaintext) == key_hash

    def test_jwt_tokens(self):
        token = create_access_token("user-1", "tenant-1", "admin")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "user-1"
        assert payload["tenant_id"] == "tenant-1"
        assert payload["role"] == "admin"
        assert payload["type"] == "access"

    def test_refresh_token(self):
        token = create_refresh_token("user-1", "tenant-1")
        payload = decode_token(token)
        assert payload["type"] == "refresh"

    def test_invalid_token(self):
        assert decode_token("invalid.token.here") is None

    def test_rbac(self):
        assert has_permission("owner", "viewer")
        assert has_permission("admin", "analyst")
        assert not has_permission("viewer", "admin")

    def test_require_role(self):
        require_role("admin", "analyst")  # should not raise
        with pytest.raises(Exception):
            require_role("viewer", "admin")


# ---- Connectors ----

from src.connectors.file_connectors import CSVConnector, DataFrameConnector
from src.connectors.schema_mapper import SchemaMapper


class TestConnectors:
    def test_csv_connector(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("date,revenue,costs\n2024-01,100,80\n2024-02,110,85\n")

        conn = CSVConnector("test-csv", {"path": str(csv_file)})
        result = conn.run()
        assert result.success
        assert result.rows_ingested == 2

    def test_dataframe_connector(self):
        df = pd.DataFrame({"revenue": [100, 200], "costs": [80, 150]})
        conn = DataFrameConnector("test-df", df)
        result = conn.run()
        assert result.success
        assert result.rows_ingested == 2

    def test_data_quality(self):
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": [10, 20, 30, 40, 500],  # outlier
        })
        conn = DataFrameConnector("quality-test", df)
        quality = conn.assess_quality(df)
        assert quality.missing_values_pct > 0
        assert quality.quality_score > 0

    def test_schema_mapper(self):
        df = pd.DataFrame({
            "Total Revenue": [100, 200],
            "OpEx": [80, 150],
            "Headcount": [50, 55],
            "xyzzy_foobar_qqq": [1, 2],
        })
        mapper = SchemaMapper(min_confidence=0.7)
        report = mapper.map_dataframe(df)

        mapped_names = {m.canonical_name for m in report.mappings}
        assert "revenue" in mapped_names
        assert "operating_cost" in mapped_names
        assert "xyzzy_foobar_qqq" in report.unmapped_columns

    def test_apply_mapping(self):
        df = pd.DataFrame({"Total Sales": [100], "staff_count": [50]})
        mapper = SchemaMapper()
        report = mapper.map_dataframe(df)
        renamed = mapper.apply_mapping(df, report)
        assert "revenue" in renamed.columns or "employee_count" in renamed.columns


# ---- Causal Discovery ----

from src.discovery.causal_discovery import (
    PCAlgorithm,
    NOTEARS,
    GrangerDiscovery,
    EnsembleCausalDiscovery,
    partial_correlation,
)


class TestCausalDiscovery:
    @pytest.fixture
    def synthetic_data(self):
        """Generate data with known causal structure: X → Y → Z."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = 0.7 * x + rng.normal(0, 0.3, n)
        z = 0.5 * y + rng.normal(0, 0.3, n)
        return pd.DataFrame({"X": x, "Y": y, "Z": z})

    def test_partial_correlation(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = 0.8 * x + rng.normal(0, 0.2, n)
        corr, p = partial_correlation(x, y)
        assert abs(corr) > 0.5
        assert p < 0.01

    def test_pc_algorithm(self, synthetic_data):
        pc = PCAlgorithm(alpha=0.05)
        result = pc.discover(synthetic_data)
        assert result.method == "pc"
        # PC finds skeleton (undirected edges) — may or may not orient without v-structures
        all_connections = result.skeleton + [(e.source, e.target) for e in result.edges]
        pair_set = set()
        for a, b in all_connections:
            pair_set.add((min(a, b), max(a, b)))
        assert ("X", "Y") in pair_set
        assert ("Y", "Z") in pair_set

    def test_notears(self, synthetic_data):
        notears = NOTEARS(max_iter=50, w_threshold=0.2)
        result = notears.discover(synthetic_data)
        assert result.method == "notears"
        # NOTEARS should find at least one edge
        assert len(result.edges) >= 1

    def test_granger(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.6 * x[t-1] + 0.3 * y[t-1] + rng.normal(0, 0.2)
        df = pd.DataFrame({"X": x, "Y": y})

        granger = GrangerDiscovery(max_lag=3, alpha=0.05)
        result = granger.discover(df)
        assert result.method == "granger"

    def test_ensemble(self, synthetic_data):
        ensemble = EnsembleCausalDiscovery(min_votes=1)
        result = ensemble.discover(synthetic_data)
        assert result.method == "ensemble"


# ---- Validation ----

from src.validation.benchmark_suite import (
    BenchmarkValidator,
    CANONICAL_BENCHMARKS,
    default_predictor,
)


class TestValidation:
    def test_canonical_benchmarks_loaded(self):
        assert len(CANONICAL_BENCHMARKS) >= 10

    def test_benchmark_validation(self):
        validator = BenchmarkValidator()
        report = validator.validate(default_predictor)
        assert report.n_cases > 0
        assert 0 <= report.overall_mape <= 5
        assert 0 <= report.coverage <= 1

    def test_by_industry_breakdown(self):
        validator = BenchmarkValidator()
        report = validator.validate(default_predictor)
        assert len(report.by_industry) > 0

    def test_accuracy_claims(self):
        validator = BenchmarkValidator()
        report = validator.validate(default_predictor)
        assert isinstance(report.accuracy_claims_supported, list)


# ---- RL Strategy Optimizer ----

from src.rl.strategy_optimizer import (
    StrategyOptimizer,
    StrategySpace,
    TwinEnvironment,
    REINFORCEOptimizer,
)
from src.core.causal_graph import CausalGraph, CausalVariable, CausalEdge, VariableType


class TestRLOptimizer:
    @pytest.fixture
    def simple_graph(self):
        g = CausalGraph()
        g.add_variable(CausalVariable("revenue", VariableType.FINANCIAL, 100.0, min_value=0, volatility=0.0))
        g.add_variable(CausalVariable("marketing", VariableType.FINANCIAL, 10.0, min_value=0, volatility=0.0))
        g.add_variable(CausalVariable("satisfaction", VariableType.CUSTOMER, 0.7, min_value=0, max_value=1.0, volatility=0.0))
        g.add_edge(CausalEdge("marketing", "satisfaction", strength=0.05, confidence=0.9))
        g.add_edge(CausalEdge("satisfaction", "revenue", strength=50.0, confidence=0.9))
        return g

    def test_strategy_space_generation(self, simple_graph):
        space = StrategySpace.from_causal_graph(simple_graph)
        assert len(space.actions) > 3  # at least a few actions + hold

    def test_environment(self, simple_graph):
        space = StrategySpace.from_causal_graph(simple_graph)
        env = TwinEnvironment(simple_graph, space, horizon=5, budget=1_000_000)
        state = env.reset()
        assert state.period == 0

        action = space.actions[0]
        next_state, reward, done = env.step(action)
        assert next_state.period == 1

    def test_rl_optimization(self, simple_graph):
        result = StrategyOptimizer.optimize_for_company(
            simple_graph, budget=1_000_000, horizon=5,
            n_episodes=50, seed=42,
        )
        assert result.explored_strategies == 50
        assert len(result.best_strategy) > 0
        assert len(result.convergence_history) == 50
        assert result.computation_time_ms > 0


# ---- Observability ----

from src.observability.monitoring import (
    MetricsCollector,
    SpanTracer,
    get_logger,
    setup_logging,
)


class TestObservability:
    def test_metrics_counter(self):
        m = MetricsCollector()
        m.increment("test_counter", 1.0)
        m.increment("test_counter", 2.0)
        summary = m.get_summary()
        assert summary["counters"]["test_counter"] == 3.0

    def test_metrics_histogram(self):
        m = MetricsCollector()
        for v in [0.1, 0.2, 0.3, 0.5, 1.0]:
            m.observe("latency", v)
        summary = m.get_summary()
        assert "latency" in summary["histograms"]
        assert summary["histograms"]["latency"]["count"] == 5

    def test_metrics_gauge(self):
        m = MetricsCollector()
        m.set_gauge("active", 5.0)
        m.set_gauge("active", 3.0)
        assert m.get_summary()["gauges"]["active"] == 3.0

    def test_metrics_timer(self):
        m = MetricsCollector()
        with m.timer("op_duration"):
            _ = sum(range(1000))
        summary = m.get_summary()
        assert "op_duration" in summary["histograms"]

    def test_tracer(self):
        tracer = SpanTracer()
        with tracer.span("test_span", {"key": "value"}):
            pass
        # Should not raise

    def test_logger(self):
        setup_logging("INFO", json_format=False)
        logger = get_logger("test")
        # structlog may return different wrapper types depending on configuration
        assert logger is not None


# ---- Reporting ----

from src.reporting.case_study import ReportGenerator


class TestReporting:
    def test_executive_brief(self):
        gen = ReportGenerator()
        results = [
            {"scenario_name": "Dynamic Pricing", "mean_roi": 0.25, "success_probability": 0.85,
             "value_at_risk_95": -0.05, "ruin_probability": 0.02, "sharpe_ratio": 1.5},
            {"scenario_name": "Route Opt", "mean_roi": 0.15, "success_probability": 0.90,
             "value_at_risk_95": -0.02, "ruin_probability": 0.01, "sharpe_ratio": 2.0},
        ]
        recommendations = [
            {"title": "Dynamic Pricing", "expected_roi": 0.25, "success_probability": 0.85,
             "risk_level": "medium", "required_investment": 3_000_000},
        ]
        report = gen.generate_executive_brief("KTZ", "transportation", results, recommendations, 0.65, "B")
        assert report.company_name == "KTZ"
        assert len(report.sections) >= 3

    def test_investor_one_pager(self):
        gen = ReportGenerator()
        results = [
            {"scenario_name": "Test", "mean_roi": 0.20, "success_probability": 0.80,
             "sharpe_ratio": 1.2, "value_at_risk_95": -0.03, "ruin_probability": 0.01},
        ]
        report = gen.generate_investor_one_pager(
            "Kaspi", "fintech", {"revenue": 1e9, "users": 10e6}, results, 0.85,
        )
        assert "Kaspi" in report.bottom_line

    def test_markdown_rendering(self):
        gen = ReportGenerator()
        results = [
            {"scenario_name": "Test", "mean_roi": 0.20, "success_probability": 0.80,
             "value_at_risk_95": -0.03, "ruin_probability": 0.01, "sharpe_ratio": 1.0},
        ]
        report = gen.generate_executive_brief("Test Co", "saas", results, [], 0.5, "C")
        md = gen.to_markdown(report)
        assert "# Test Co" in md
        assert "Bottom Line" in md

    def test_validation_report(self):
        gen = ReportGenerator()
        report = gen.generate_validation_report({
            "n_cases": 10, "overall_mape": 0.12, "coverage": 0.90,
            "calibration_score": 0.88, "bias": 0.02,
            "accuracy_claims_supported": ["MAPE below 15%"],
        })
        assert "validation" in report.industry.lower()
