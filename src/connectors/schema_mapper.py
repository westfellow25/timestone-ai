"""
TimeStone AI — Schema Mapper

Maps client data schemas to TimeStone's canonical variable schema.
This is critical for onboarding — every company has different column
names for the same business concepts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Canonical variable names grouped by semantic category
CANONICAL_SCHEMA: Dict[str, List[str]] = {
    "revenue": ["revenue", "sales", "total_revenue", "turnover", "gross_revenue", "net_revenue", "total_sales"],
    "operating_cost": ["opex", "operating_cost", "operating_expense", "costs", "expenses", "total_cost"],
    "profit_margin": ["margin", "profit_margin", "operating_margin", "ebit_margin"],
    "employee_count": ["employees", "headcount", "staff_count", "fte", "workforce"],
    "customer_count": ["customers", "users", "clients", "accounts", "subscribers"],
    "churn_rate": ["churn", "attrition", "churn_rate", "customer_churn", "logo_churn"],
    "nps": ["nps", "net_promoter_score", "nps_score", "promoter_score"],
    "market_share": ["market_share", "share", "market_position"],
    "transaction_volume": ["transactions", "txn_count", "transaction_volume", "tx_volume"],
    "avg_transaction_value": ["atv", "avg_ticket", "average_transaction", "mean_order_value"],
    "date": ["date", "timestamp", "period", "month", "quarter", "year", "time"],
}


@dataclass
class ColumnMapping:
    """A single column → canonical variable mapping."""
    source_column: str
    canonical_name: str
    confidence: float
    data_type: str
    sample_values: List = field(default_factory=list)


@dataclass
class MappingReport:
    """Full schema mapping report."""
    mappings: List[ColumnMapping]
    unmapped_columns: List[str]
    missing_required: List[str]
    data_types: Dict[str, str]
    row_count: int
    date_range: Optional[Tuple[str, str]] = None


def normalize_column_name(name: str) -> str:
    """Normalize a column name for comparison."""
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def fuzzy_score(a: str, b: str) -> float:
    """Fuzzy similarity score between two strings."""
    a_norm = normalize_column_name(a)
    b_norm = normalize_column_name(b)
    if a_norm == b_norm:
        return 1.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


class SchemaMapper:
    """
    Map client data to TimeStone's canonical schema using:
    1. Exact match
    2. Fuzzy name matching
    3. Data type / value pattern inference
    """

    def __init__(
        self,
        canonical_schema: Optional[Dict[str, List[str]]] = None,
        min_confidence: float = 0.6,
    ):
        self.schema = canonical_schema or CANONICAL_SCHEMA
        self.min_confidence = min_confidence

    def map_dataframe(
        self,
        df: pd.DataFrame,
        required_fields: Optional[List[str]] = None,
    ) -> MappingReport:
        """Map DataFrame columns to canonical variables."""
        mappings = []
        unmapped = []

        for col in df.columns:
            best_match = None
            best_score = 0.0

            # Try matching against all canonical names and their aliases
            for canonical, aliases in self.schema.items():
                for alias in [canonical] + aliases:
                    score = fuzzy_score(col, alias)
                    if score > best_score:
                        best_score = score
                        best_match = canonical

            if best_score >= self.min_confidence and best_match is not None:
                mappings.append(ColumnMapping(
                    source_column=col,
                    canonical_name=best_match,
                    confidence=best_score,
                    data_type=str(df[col].dtype),
                    sample_values=df[col].head(3).tolist(),
                ))
            else:
                unmapped.append(col)

        # Check required fields
        mapped_canonicals = {m.canonical_name for m in mappings}
        missing = [r for r in (required_fields or []) if r not in mapped_canonicals]

        # Data types
        data_types = {col: str(df[col].dtype) for col in df.columns}

        # Date range if possible
        date_range = None
        date_mapping = next((m for m in mappings if m.canonical_name == "date"), None)
        if date_mapping:
            col = date_mapping.source_column
            try:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(dates) > 0:
                    date_range = (str(dates.min().date()), str(dates.max().date()))
            except Exception:
                pass

        return MappingReport(
            mappings=mappings,
            unmapped_columns=unmapped,
            missing_required=missing,
            data_types=data_types,
            row_count=len(df),
            date_range=date_range,
        )

    def apply_mapping(
        self,
        df: pd.DataFrame,
        report: MappingReport,
    ) -> pd.DataFrame:
        """Rename DataFrame columns according to the mapping report."""
        rename_map = {m.source_column: m.canonical_name for m in report.mappings}
        return df.rename(columns=rename_map)

    def suggest_unmapped_mappings(
        self,
        unmapped: List[str],
        top_k: int = 3,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """For unmapped columns, suggest the top-k closest canonical variables."""
        suggestions = {}
        for col in unmapped:
            scores = []
            for canonical, aliases in self.schema.items():
                max_score = max(fuzzy_score(col, alias) for alias in [canonical] + aliases)
                scores.append((canonical, max_score))
            scores.sort(key=lambda x: x[1], reverse=True)
            suggestions[col] = scores[:top_k]
        return suggestions
