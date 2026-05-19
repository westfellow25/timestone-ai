# ADR-002: Case library as primary product moat

Date: 2026-05-20
Status: Accepted

## Context

A Monte Carlo simulator with hardcoded parameter ranges has no defensible
moat — any competent engineer can reproduce it in days, and an LLM can
generate one on demand. The same is true for an industry-templates
scenario generator. To build a durable product, TimeStone must rely on
something that compounds over time and cannot be replicated by a
zero-history competitor.

## Decision

The primary moat is the **case library plus outcomes database**:
- `data/case_library.json` — curated real transformations with structured
  `promised_*` vs `actual_*` outcome fields. Each entry is sourced from a
  citable document.
- `results/outcomes/` — append-only proprietary log of "predicted vs
  realised" pairs from every TimeStone engagement.

The simulator and scenario generator read from these. As the library
grows, simulations improve; as outcomes are recorded, priors update
via Bayesian methods.

## Consequences

Positive:
- A competitor starting today is 1-3 years behind on data.
- The moat compounds with usage: every consulting project adds outcome
  data that strengthens future predictions.
- Domain expertise (Arman's consulting experience) translates into
  proprietary data assets, not just code.

Negative:
- Requires disciplined data hygiene: every new case needs sources,
  failure/success classification, and `actual_*` outcome fields.
- Bayesian update loop is non-trivial and must be implemented carefully
  to avoid recency bias.
- Public-source extraction has selection bias toward successes;
  failures must be actively hunted.

## Implementation notes

- `domain.case.TransformationCase` is the canonical schema.
- `services.knowledge_retrieval.CaseLibrary` is the retrieval interface.
- Outcomes have their own dataclass (`domain.outcome.OutcomeRecord`)
  and are persisted in `repositories.outcomes`.
- Bayesian update is a future work item (`services.calibration`).
