# ADR-001: Layered architecture with strict downward dependencies

Date: 2026-05-20
Status: Accepted

## Context

The project grew from a single Monte Carlo simulator script into a multi-module
system with a case library, scenario generator, Claude integration, Streamlit
dashboard, and tests. Modules cross-import freely; the dashboard reaches
into models and does its own I/O; CLI entry points are scattered across
`__main__` blocks. Adding a PDF report, REST API, or outcome tracking would
require touching every layer.

## Decision

Adopt a 5-layer architecture under `src/timestone/`:
`domain -> repositories -> services -> application -> interfaces`,
plus a cross-cutting `infrastructure` layer.

Dependencies flow downward only.

## Consequences

Positive:
- Single point of entry per use case (`application/`) — easy to wire into
  any interface (CLI, web, API, bot).
- `domain/` is dependency-free, so it can be reused in tests, scripts,
  and external code without dragging in I/O or LLM weight.
- `repositories/` lets us swap JSON files for a database without
  touching services.
- New features (PDF reports, outcome tracking) have a clear home.

Negative:
- More files; more `__init__.py` boilerplate.
- Pulls junior developers into a heavier mental model.
- For a 2-person team this is overkill — accepted tradeoff because the
  product is intended to scale.

## Alternatives considered

1. Flat module structure (status quo). Rejected: long-term cost of
   tangled imports is higher than upfront cost of layering.
2. Hexagonal / ports-and-adapters. Rejected for now: heavier, similar
   benefit. Can adopt later if multi-backend needs grow.
3. One-file-per-feature. Rejected: encourages copy-paste, makes shared
   logic hard to find.
