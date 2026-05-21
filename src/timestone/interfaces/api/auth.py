"""Multi-tenant auth scaffolding (v0.6 preview).

Tenants live in ``data/tenants.json``. Each tenant has:
  - tenant_id (str)
  - name (str)
  - api_keys (list[str])   — bearer keys hashed in production
  - plan (str)             — "pilot" | "growth" | "enterprise"
  - quota_assess_per_day (int)

Auth is intentionally minimal so a real customer pilot can ship without
us blocking on a full IAM stack. Production would swap this for Auth0 /
Okta / Stytch and store hashed keys in Postgres.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ----- Tenant store -----

@dataclass
class Tenant:
    tenant_id: str
    name: str
    api_keys: List[str] = field(default_factory=list)
    plan: str = "pilot"
    quota_assess_per_day: int = 20

    @classmethod
    def from_dict(cls, d: Dict) -> "Tenant":
        return cls(
            tenant_id=d["tenant_id"],
            name=d.get("name", d["tenant_id"]),
            api_keys=list(d.get("api_keys", [])),
            plan=d.get("plan", "pilot"),
            quota_assess_per_day=int(d.get("quota_assess_per_day", 20)),
        )


# ----- Quota tracker (in-memory) -----

class QuotaTracker:
    """In-memory daily quota counter.

    Keys are (tenant_id, YYYY-MM-DD) → int. Cleared on process restart;
    production would back this with Redis or Postgres.
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = defaultdict(int)

    def _today_key(self, tenant_id: str) -> str:
        return tenant_id + "|" + time.strftime("%Y-%m-%d", time.gmtime())

    def used(self, tenant_id: str) -> int:
        return self._counts[self._today_key(tenant_id)]

    def increment(self, tenant_id: str) -> None:
        self._counts[self._today_key(tenant_id)] += 1


# ----- Tenant registry -----

class TenantRegistry:
    def __init__(self, store_path: Optional[Path] = None) -> None:
        self.store_path = store_path
        self._by_key: Dict[str, Tenant] = {}
        self._by_id: Dict[str, Tenant] = {}
        self._load()

    def _load(self) -> None:
        # Default tenant for local development & demos.
        demo = Tenant(
            tenant_id="demo",
            name="Demo / Local",
            api_keys=["demo-key", "ts_demo_local"],
            plan="pilot",
            quota_assess_per_day=50,
        )
        self._register(demo)

        if self.store_path and self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                for d in data.get("tenants", []):
                    self._register(Tenant.from_dict(d))
            except (json.JSONDecodeError, KeyError):
                pass

    def _register(self, tenant: Tenant) -> None:
        self._by_id[tenant.tenant_id] = tenant
        for k in tenant.api_keys:
            self._by_key[k] = tenant

    def resolve(self, api_key: Optional[str]) -> Optional[Tenant]:
        if not api_key:
            return None
        return self._by_key.get(api_key)

    def list_all(self) -> List[Tenant]:
        return list(self._by_id.values())
