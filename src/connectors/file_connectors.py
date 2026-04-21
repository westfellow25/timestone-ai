"""
TimeStone AI — File-based data connectors.

Supports CSV, Excel (xlsx), JSON, and Parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.connectors.base import BaseConnector


class CSVConnector(BaseConnector):
    """Ingest data from a local CSV file or URL."""

    def validate_config(self) -> List[str]:
        errors = []
        if "path" not in self.config:
            errors.append("Missing 'path' in config")
        return errors

    def test_connection(self) -> bool:
        path = self.config.get("path", "")
        if path.startswith(("http://", "https://")):
            return True
        return Path(path).exists()

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        path = self.config["path"]
        delimiter = self.config.get("delimiter", ",")
        encoding = self.config.get("encoding", "utf-8")

        df = pd.read_csv(
            path,
            delimiter=delimiter,
            encoding=encoding,
            nrows=limit,
        )

        # Parse date columns if specified
        date_columns = self.config.get("date_columns", [])
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df


class ExcelConnector(BaseConnector):
    """Ingest data from an Excel file."""

    def validate_config(self) -> List[str]:
        errors = []
        if "path" not in self.config:
            errors.append("Missing 'path' in config")
        return errors

    def test_connection(self) -> bool:
        return Path(self.config.get("path", "")).exists()

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        path = self.config["path"]
        sheet = self.config.get("sheet_name", 0)
        skiprows = self.config.get("skiprows", 0)

        df = pd.read_excel(path, sheet_name=sheet, skiprows=skiprows, nrows=limit)
        return df


class JSONConnector(BaseConnector):
    """Ingest data from JSON file or JSONL."""

    def validate_config(self) -> List[str]:
        errors = []
        if "path" not in self.config:
            errors.append("Missing 'path' in config")
        return errors

    def test_connection(self) -> bool:
        return Path(self.config.get("path", "")).exists()

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        path = self.config["path"]
        json_format = self.config.get("format", "records")  # records | lines | nested
        nested_key = self.config.get("nested_key", None)

        if json_format == "lines":
            df = pd.read_json(path, lines=True, nrows=limit)
        else:
            with open(path, "r") as f:
                data = json.load(f)
            if nested_key:
                for key in nested_key.split("."):
                    data = data[key]
            df = pd.DataFrame(data)
            if limit:
                df = df.head(limit)

        return df


class ParquetConnector(BaseConnector):
    """Ingest data from a Parquet file (local or S3)."""

    def validate_config(self) -> List[str]:
        errors = []
        if "path" not in self.config:
            errors.append("Missing 'path' in config")
        return errors

    def test_connection(self) -> bool:
        path = self.config.get("path", "")
        if path.startswith("s3://"):
            return True  # assume credentials work; will fail loudly on ingest
        return Path(path).exists()

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        path = self.config["path"]
        columns = self.config.get("columns")

        df = pd.read_parquet(path, columns=columns)
        if limit:
            df = df.head(limit)
        return df


class DataFrameConnector(BaseConnector):
    """Wrap an in-memory DataFrame (useful for tests and direct API uploads)."""

    def __init__(self, name: str, df: pd.DataFrame):
        super().__init__(name, {"source": "dataframe"})
        self._df = df

    def validate_config(self) -> List[str]:
        return [] if self._df is not None else ["No DataFrame provided"]

    def test_connection(self) -> bool:
        return self._df is not None

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        return self._df.head(limit) if limit else self._df.copy()
