"""TimeStone CLI.

Usage:
    python -m timestone assess <company_name>
    python -m timestone list-companies
    python -m timestone list-runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..application import assess_company, AssessOptions
from ..repositories.company import CompanyRepository
from ..repositories.results import ResultsRepository


def cmd_assess(args: argparse.Namespace) -> int:
    company_repo = CompanyRepository()
    twin = company_repo.load_by_name(args.company)
    if twin is None:
        print(f"Company not found: {args.company}", file=sys.stderr)
        print("Available companies:", file=sys.stderr)
        for c in company_repo.list_all():
            print(f"  - {c.company_name}", file=sys.stderr)
        return 1
    opts = AssessOptions(
        scenario_count=args.scenarios,
        iterations=args.iterations,
        discount_rate=args.discount_rate,
        random_seed=args.seed,
    )
    report = assess_company(twin, options=opts)
    print(f"Run ID: {report.run_id}")
    print(f"Company: {report.company_name}")
    print(f"Scenarios: {report.total_scenarios}")
    print(f"Failure rate (% scenarios with P<50%): {report.failure_rate_among_scenarios:.1%}")
    print(f"Case library size: {report.case_library_size}")
    print()
    print("Top recommendations:")
    for rec in report.top_recommendations:
        print(f"  #{rec.rank} {rec.scenario_name}")
        print(f"     P(NPV>0): {rec.success_probability:.1%}")
        print(f"     Mean NPV: ${rec.mean_npv:,.0f}")
        print(f"     Mean ROI: {rec.mean_roi:.1f}x")
        print(f"     Payback: {rec.payback_years:.1f}y")
        print(f"     Headline: {rec.headline}")
        if rec.based_on_cases:
            print(f"     Based on cases: {', '.join(rec.based_on_cases[:3])}")
        print()
    return 0


def cmd_list_companies(args: argparse.Namespace) -> int:
    repo = CompanyRepository()
    for c in repo.list_all():
        print(f"  {c.company_name}  ({c.metrics.industry}, ${c.metrics.revenue/1e9:.1f}B)")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    repo = ResultsRepository()
    runs = repo.list_runs()
    if not runs:
        print("No runs yet. Use `python -m timestone assess <company>`.")
        return 0
    for p in runs[:20]:
        print(f"  {p.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="timestone", description="TimeStone AI CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assess", help="Run the full assessment pipeline for a company.")
    a.add_argument("company", help="Company name (case-insensitive match)")
    a.add_argument("--scenarios", type=int, default=1000)
    a.add_argument("--iterations", type=int, default=1000)
    a.add_argument("--discount-rate", type=float, default=0.12)
    a.add_argument("--seed", type=int, default=42)
    a.set_defaults(func=cmd_assess)

    lc = sub.add_parser("list-companies", help="List available digital twins.")
    lc.set_defaults(func=cmd_list_companies)

    lr = sub.add_parser("list-runs", help="List recent assessment runs.")
    lr.set_defaults(func=cmd_list_runs)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
