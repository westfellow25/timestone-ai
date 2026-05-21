"""TimeStone REST API (FastAPI).

Exposes the assessment pipeline over HTTP so customer-facing dashboards,
CRM webhooks, and the Telegram bot can hit the same backend.

Endpoints
---------
GET  /health                 — liveness probe
GET  /library                — list cases from the curated corpus
GET  /library/{case_id}      — single case detail
GET  /companies              — list known digital twins
POST /assess                 — run a full assessment for a company twin
GET  /predict                — read pre-registered predictions
GET  /predict/{prediction_id}
GET  /auth/whoami            — resolve current API key to a tenant
GET  /me/runs                — list this tenant's recent assessments
GET  /me/dashboard           — minimal HTML dashboard for this tenant

Auth (multi-tenant scaffolding, v0.6 preview)
---------------------------------------------
Header ``X-API-Key: <key>`` resolves to a tenant in ``data/tenants.json``.
Per-tenant quotas (assess/day) are enforced in-memory; production would
back this with Redis.

Run locally
-----------
    pip install fastapi uvicorn
    uvicorn timestone.interfaces.api.app:app --reload --port 8088
"""
from .app import app, create_app  # noqa: F401
