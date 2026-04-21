"""
TimeStone AI — Repository pattern for tenant-safe data access.

All data access MUST go through repositories. Every query is scoped to
a tenant_id to prevent cross-tenant data leaks. Repositories raise
TenantIsolationError if a query is attempted without a tenant context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.persistence.models import (
    ApiKey,
    AuditAction,
    AuditLog,
    BenchmarkCase,
    Company,
    DataSource,
    Prediction,
    Simulation,
    Tenant,
    User,
)


class TenantIsolationError(Exception):
    """Raised when a tenant context is missing or violated."""


class BaseRepository:
    def __init__(self, session: Session, tenant_id: Optional[str] = None):
        self.session = session
        self.tenant_id = tenant_id

    def _require_tenant(self) -> str:
        if not self.tenant_id:
            raise TenantIsolationError(
                "Tenant context is required for this operation"
            )
        return self.tenant_id


class TenantRepository:
    """Tenant management — operates without tenant isolation (super-admin)."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str, slug: str, plan: str = "pilot", metadata: Optional[Dict] = None) -> Tenant:
        tenant = Tenant(name=name, slug=slug, plan=plan, metadata_json=metadata or {})
        self.session.add(tenant)
        self.session.flush()
        return tenant

    def get(self, tenant_id: str) -> Optional[Tenant]:
        return self.session.get(Tenant, tenant_id)

    def get_by_slug(self, slug: str) -> Optional[Tenant]:
        stmt = select(Tenant).where(Tenant.slug == slug)
        return self.session.scalar(stmt)

    def list_all(self) -> List[Tenant]:
        return list(self.session.scalars(select(Tenant)))

    def deactivate(self, tenant_id: str) -> None:
        tenant = self.get(tenant_id)
        if tenant:
            tenant.is_active = False


class UserRepository(BaseRepository):
    def create(
        self,
        email: str,
        password_hash: str,
        full_name: str = "",
        role: str = "viewer",
    ) -> User:
        tenant_id = self._require_tenant()
        user = User(
            tenant_id=tenant_id,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            role=role,
        )
        self.session.add(user)
        self.session.flush()
        return user

    def get(self, user_id: str) -> Optional[User]:
        tenant_id = self._require_tenant()
        stmt = select(User).where(User.id == user_id, User.tenant_id == tenant_id)
        return self.session.scalar(stmt)

    def get_by_email(self, email: str) -> Optional[User]:
        tenant_id = self._require_tenant()
        stmt = select(User).where(User.email == email, User.tenant_id == tenant_id)
        return self.session.scalar(stmt)

    def list_users(self) -> List[User]:
        tenant_id = self._require_tenant()
        return list(self.session.scalars(
            select(User).where(User.tenant_id == tenant_id).order_by(User.created_at.desc())
        ))

    def update_last_login(self, user_id: str) -> None:
        user = self.get(user_id)
        if user:
            user.last_login_at = datetime.now(timezone.utc)


class ApiKeyRepository(BaseRepository):
    def create(
        self,
        user_id: str,
        key_hash: str,
        name: str = "",
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> ApiKey:
        tenant_id = self._require_tenant()
        key = ApiKey(
            tenant_id=tenant_id,
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes or [],
            expires_at=expires_at,
        )
        self.session.add(key)
        self.session.flush()
        return key

    def find_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        stmt = select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
        return self.session.scalar(stmt)

    def record_use(self, key_id: str) -> None:
        key = self.session.get(ApiKey, key_id)
        if key:
            key.last_used_at = datetime.now(timezone.utc)

    def revoke(self, key_id: str) -> None:
        tenant_id = self._require_tenant()
        key = self.session.get(ApiKey, key_id)
        if key and key.tenant_id == tenant_id:
            key.is_active = False


class CompanyRepository(BaseRepository):
    def create(
        self,
        name: str,
        industry: str,
        revenue: float,
        operating_costs: float,
        employee_count: int,
        market_share: float,
        metadata: Optional[Dict] = None,
        genome: Optional[Dict] = None,
        causal_graph: Optional[Dict] = None,
    ) -> Company:
        tenant_id = self._require_tenant()
        company = Company(
            tenant_id=tenant_id,
            name=name,
            industry=industry,
            revenue=revenue,
            operating_costs=operating_costs,
            employee_count=employee_count,
            market_share=market_share,
            metadata_json=metadata or {},
            genome_json=genome,
            causal_graph_json=causal_graph,
        )
        self.session.add(company)
        self.session.flush()
        return company

    def get(self, company_id: str) -> Optional[Company]:
        tenant_id = self._require_tenant()
        stmt = select(Company).where(Company.id == company_id, Company.tenant_id == tenant_id)
        return self.session.scalar(stmt)

    def list(self, limit: int = 100) -> List[Company]:
        tenant_id = self._require_tenant()
        stmt = (
            select(Company)
            .where(Company.tenant_id == tenant_id)
            .order_by(Company.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def update_genome(self, company_id: str, genome: Dict) -> None:
        company = self.get(company_id)
        if company:
            company.genome_json = genome

    def update_causal_graph(self, company_id: str, graph: Dict) -> None:
        company = self.get(company_id)
        if company:
            company.causal_graph_json = graph

    def delete(self, company_id: str) -> bool:
        company = self.get(company_id)
        if company:
            self.session.delete(company)
            return True
        return False


class SimulationRepository(BaseRepository):
    def create(
        self,
        company_id: str,
        iterations: int,
        method: str,
        scenarios: List[Dict],
        created_by: Optional[str] = None,
    ) -> Simulation:
        tenant_id = self._require_tenant()
        sim = Simulation(
            tenant_id=tenant_id,
            company_id=company_id,
            created_by=created_by,
            iterations=iterations,
            method=method,
            scenarios_json=scenarios,
            status="pending",
        )
        self.session.add(sim)
        self.session.flush()
        return sim

    def save_results(
        self,
        sim_id: str,
        results: List[Dict],
        insights: List[Dict],
        recommendations: List[Dict],
        summary: Dict,
        duration_ms: int,
    ) -> None:
        sim = self.get(sim_id)
        if sim:
            sim.results_json = results
            sim.insights_json = insights
            sim.recommendations_json = recommendations
            sim.summary_json = summary
            sim.duration_ms = duration_ms
            sim.status = "completed"
            sim.completed_at = datetime.now(timezone.utc)

    def get(self, sim_id: str) -> Optional[Simulation]:
        tenant_id = self._require_tenant()
        stmt = select(Simulation).where(Simulation.id == sim_id, Simulation.tenant_id == tenant_id)
        return self.session.scalar(stmt)

    def list_for_company(self, company_id: str, limit: int = 50) -> List[Simulation]:
        tenant_id = self._require_tenant()
        stmt = (
            select(Simulation)
            .where(Simulation.tenant_id == tenant_id, Simulation.company_id == company_id)
            .order_by(Simulation.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class PredictionRepository(BaseRepository):
    def create_many(self, simulation_id: str, predictions: List[Dict]) -> List[Prediction]:
        tenant_id = self._require_tenant()
        objs = [
            Prediction(
                tenant_id=tenant_id,
                simulation_id=simulation_id,
                variable_name=p["variable_name"],
                predicted_value=p["predicted_value"],
                ci_lower=p["ci_lower"],
                ci_upper=p["ci_upper"],
                confidence_level=p.get("confidence_level", 0.90),
                horizon_months=p.get("horizon_months", 12),
            )
            for p in predictions
        ]
        self.session.add_all(objs)
        self.session.flush()
        return objs

    def record_outcome(self, prediction_id: str, actual_value: float) -> None:
        tenant_id = self._require_tenant()
        stmt = select(Prediction).where(
            Prediction.id == prediction_id, Prediction.tenant_id == tenant_id
        )
        pred = self.session.scalar(stmt)
        if pred:
            pred.actual_value = actual_value
            pred.recorded_at = datetime.now(timezone.utc)

    def list_with_outcomes(self) -> List[Prediction]:
        tenant_id = self._require_tenant()
        stmt = select(Prediction).where(
            Prediction.tenant_id == tenant_id,
            Prediction.actual_value.isnot(None),
        )
        return list(self.session.scalars(stmt))


class AuditLogRepository(BaseRepository):
    def log(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource_type: str = "",
        resource_id: str = "",
        details: Optional[Dict] = None,
        ip_address: str = "",
        user_agent: str = "",
    ) -> AuditLog:
        tenant_id = self._require_tenant()
        entry = AuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details_json=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def recent(self, limit: int = 100) -> List[AuditLog]:
        tenant_id = self._require_tenant()
        stmt = (
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class BenchmarkRepository:
    """Global benchmark cases — not tenant-scoped."""

    def __init__(self, session: Session):
        self.session = session

    def add(
        self,
        name: str,
        industry: str,
        transformation_type: str,
        year: int,
        initial_state: Dict,
        actual_outcome: Dict,
        source: str = "",
    ) -> BenchmarkCase:
        case = BenchmarkCase(
            name=name,
            industry=industry,
            transformation_type=transformation_type,
            year=year,
            initial_state_json=initial_state,
            actual_outcome_json=actual_outcome,
            source=source,
        )
        self.session.add(case)
        self.session.flush()
        return case

    def list_by_industry(self, industry: str) -> List[BenchmarkCase]:
        stmt = select(BenchmarkCase).where(BenchmarkCase.industry == industry)
        return list(self.session.scalars(stmt))

    def list_all(self) -> List[BenchmarkCase]:
        return list(self.session.scalars(select(BenchmarkCase)))

    def record_prediction(
        self,
        case_id: str,
        predicted: Dict,
        accuracy: float,
        within_ci: bool,
    ) -> None:
        case = self.session.get(BenchmarkCase, case_id)
        if case:
            case.predicted_outcome_json = predicted
            case.accuracy_score = accuracy
            case.within_confidence_interval = within_ci
