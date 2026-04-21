"""
TimeStone AI — Data Connector Framework

Abstract base for data ingestion connectors. Each connector can:
1. Validate its configuration
2. Test connection
3. Ingest data into a standardized DataFrame format
4. Schedule incremental syncs
5. Report row counts and errors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class IngestionResult:
    """Result of a data ingestion operation."""
    connector_name: str
    rows_ingested: int
    columns: List[str]
    success: bool
    duration_ms: int
    error: Optional[str] = None
    data_quality_score: float = 0.0
    sample_rows: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Data quality assessment."""
    total_rows: int
    total_columns: int
    missing_values_pct: float
    duplicate_rows_pct: float
    type_mismatches: int
    outlier_count: int
    time_gaps: int
    quality_score: float
    issues: List[str]


class BaseConnector(ABC):
    """Abstract base for all data connectors."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def validate_config(self) -> List[str]:
        """Return list of config validation errors (empty = valid)."""

    @abstractmethod
    def test_connection(self) -> bool:
        """Return True if we can successfully reach the data source."""

    @abstractmethod
    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Pull data from the source and return as DataFrame."""

    def assess_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """Assess data quality of ingested DataFrame."""
        total_rows = len(df)
        total_cols = len(df.columns)

        if total_rows == 0:
            return DataQualityReport(
                total_rows=0,
                total_columns=total_cols,
                missing_values_pct=0.0,
                duplicate_rows_pct=0.0,
                type_mismatches=0,
                outlier_count=0,
                time_gaps=0,
                quality_score=0.0,
                issues=["No rows ingested"],
            )

        missing_pct = df.isna().sum().sum() / (total_rows * total_cols) * 100 if total_cols > 0 else 0.0
        dup_pct = df.duplicated().sum() / total_rows * 100

        # Outlier detection using IQR for numeric columns
        outlier_count = 0
        for col in df.select_dtypes(include="number").columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                low = q1 - 3 * iqr
                high = q3 + 3 * iqr
                outlier_count += int(((df[col] < low) | (df[col] > high)).sum())

        issues = []
        if missing_pct > 5:
            issues.append(f"High missing value rate: {missing_pct:.1f}%")
        if dup_pct > 1:
            issues.append(f"Duplicates detected: {dup_pct:.1f}%")
        if outlier_count / total_rows > 0.05:
            issues.append(f"{outlier_count} outliers detected")

        score = max(0.0, 1.0 - missing_pct / 100 - dup_pct / 100 - outlier_count / (total_rows * total_cols + 1))

        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_cols,
            missing_values_pct=missing_pct,
            duplicate_rows_pct=dup_pct,
            type_mismatches=0,
            outlier_count=outlier_count,
            time_gaps=0,
            quality_score=score,
            issues=issues,
        )

    def run(self, limit: Optional[int] = None) -> IngestionResult:
        """Run the connector with full lifecycle: validate, connect, ingest, assess."""
        start = datetime.now()
        errors = self.validate_config()
        if errors:
            return IngestionResult(
                connector_name=self.name,
                rows_ingested=0,
                columns=[],
                success=False,
                duration_ms=0,
                error=f"Config validation failed: {'; '.join(errors)}",
            )

        if not self.test_connection():
            return IngestionResult(
                connector_name=self.name,
                rows_ingested=0,
                columns=[],
                success=False,
                duration_ms=0,
                error="Connection test failed",
            )

        try:
            df = self.ingest(limit=limit)
            quality = self.assess_quality(df)
            duration = int((datetime.now() - start).total_seconds() * 1000)

            return IngestionResult(
                connector_name=self.name,
                rows_ingested=len(df),
                columns=list(df.columns),
                success=True,
                duration_ms=duration,
                data_quality_score=quality.quality_score,
                sample_rows=df.head(5).to_dict(orient="records"),
                warnings=quality.issues,
            )
        except Exception as exc:
            return IngestionResult(
                connector_name=self.name,
                rows_ingested=0,
                columns=[],
                success=False,
                duration_ms=int((datetime.now() - start).total_seconds() * 1000),
                error=str(exc),
            )
