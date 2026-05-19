# TimeStone AI — Architecture

This document is the source of truth for how TimeStone is structured.
If code does not match this document, the document is wrong or the code is — open an ADR.

## Layered architecture

Code lives in `src/timestone/` and is organised in **strict layers**.
Dependencies flow in one direction only: downward.

```
+--------------------------------------------------------+
| interfaces/   CLI, Streamlit web, REST API, PDF reports |
+--------------------------------------------------------+
                         |
                         v
+--------------------------------------------------------+
| application/  use cases (assess_company, track_outcome) |
+--------------------------------------------------------+
                         |
                         v
+--------------------------------------------------------+
| services/     business logic (simulation, retrieval)    |
+--------------------------------------------------------+
                         |
                         v
+--------------------------------------------------------+
| repositories/  persistence abstraction (JSON, DB)       |
+--------------------------------------------------------+
                         |
                         v
+--------------------------------------------------------+
| domain/       pure dataclasses, no I/O, no deps         |
+--------------------------------------------------------+

infrastructure/  config, logging, LLM client — used anywhere
```

### Forbidden imports

- `domain/` must not import from any other timestone layer.
- `services/` may import only from `domain/` and `infrastructure/`.
- `repositories/` may import only from `domain/` and `infrastructure/`.
- `application/` may import from `domain/`, `services/`, `repositories/`,
  `infrastructure/`.
- `interfaces/` may import from anywhere except other interfaces.

If you ever feel the urge to break these rules, that's a smell — open an ADR.

## Layers in detail

### domain/
Pure data classes. No `open()`, no `requests`, no `numpy.random` global state,
no Streamlit. Just the shape of business concepts.

- `company.py` — `Company`, `CompanyMetrics`
- `case.py` — `TransformationCase` (real historical transformations)
- `scenario.py` — `Scenario` (a generated hypothesis)
- `simulation.py` — `SimulationConfig`, `SimulationResult`
- `outcome.py` — `OutcomeRecord` (actual realised result of a recommended scenario)
- `report.py` — `Recommendation`, `AssessmentReport`

### services/
Functions and classes that operate on domain objects to produce other
domain objects. No file I/O — accept inputs as parameters, return outputs.

- `knowledge_retrieval.py` — `CaseLibrary` (in-memory), `find_similar`,
  `empirical_prior`
- `scenario_generation.py` — `ScenarioGenerator`,
  `CaseLibraryBackedGenerator`
- `monte_carlo.py` — `MonteCarloSimulator`
- `sensitivity.py` — tornado analysis
- `claude_scenarios.py` — Claude-powered scenario generator
- `recommendation.py` — packages top scenarios into a `Recommendation`

### repositories/
Abstracts persistence. Today: JSON files on disk. Tomorrow: PostgreSQL,
S3, whatever. Each repo exposes a small interface (`load`, `save`,
`list_runs`, etc.) so the rest of the code does not know or care.

- `case_library.py` — load/save the case corpus
- `company.py` — load/save digital twins
- `results.py` — store simulation outputs with run versioning
  (`runs/{timestamp}_{company}_{nonce}/`)
- `outcomes.py` — store realised outcomes (the future moat)

### application/
Use cases — orchestrate domain + services + repositories to do one
complete user-meaningful thing.

- `assess_company.py` — `assess(twin) -> AssessmentReport`. Full
  pipeline: load twin -> retrieve cases -> generate scenarios ->
  simulate -> rank -> save run -> return report.
- `add_case.py` — admin: append a case to the library.
- `track_outcome.py` — record actual outcome of a past recommendation.
- `compare.py` — diff two runs side by side.

### interfaces/
Delivery layer. Owns NO business logic. Calls into `application/`.

- `cli.py` — `python -m timestone assess KTZ` etc.
- `web/dashboard.py` — Streamlit dashboard
- `web/pages/*` — page components
- `reports/pdf.py` — executive PDF generator (future)
- `api/routes.py` — REST endpoints (future)

### infrastructure/
Cross-cutting concerns. Independent of business meaning.

- `config.py` — load `.env`, settings
- `logging.py` — structured logging setup
- `paths.py` — REPO_ROOT, data dir, results dir resolution
- `llm/claude.py` — Anthropic client wrapper

## Data on disk

```
data/                      # committed to git; static seed content
  case_library.json
  twins/
    ktz.json
    kaspi.json

results/                   # gitignored; generated artifacts
  runs/
    2026-05-20_ktz_001/
      scenarios.json
      simulation.json
      report.json
      report.pdf
  outcomes/                # the moat — gold-standard realised outcomes
    ktz_dynamic_pricing_2024.json
```

## Why outcomes/ is the moat

Every time TimeStone is used in a real consulting engagement, the
**predicted** distribution (mean ROI, P10, P90, failure probability) and
the **actual** result (measured 12-24 months later) are stored as a
pair in `outcomes/`. Over time this becomes a proprietary
"predicted vs actual" dataset that no LLM and no competitor can
reproduce. It directly feeds Bayesian updating of the priors that
power future simulations.

This file is therefore **append-only** and **never deleted**.

## Tests

```
tests/
  unit/         # domain + service logic, no I/O, no LLM
  integration/  # services + repositories together, hits disk
  e2e/          # full pipeline via application/assess_company
```

Every PR must pass `pytest tests/`.

## Versioning

The project follows [SemVer](https://semver.org/).
- Public stable APIs live in `timestone.application.*`.
- `timestone.domain.*` types are stable; renames go through a deprecation cycle.
- `timestone.services.*` and `repositories.*` are internal — breaking
  changes there are allowed without a major bump.

## Architecture Decision Records

Non-trivial structural choices are recorded in `docs/adr/` using the
[Michael Nygard format](https://github.com/joelparkerhenderson/architecture-decision-record).

- ADR-001: Layered architecture (this document)
- ADR-002: Case library as primary moat (rationale for retrieval-based priors)
