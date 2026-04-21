"""
TimeStone AI — Data warehouse connectors.

Snowflake, BigQuery, PostgreSQL, S3.

These are thin wrappers that gracefully degrade when the optional
libraries are not installed — so the core package doesn't have
heavy dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.connectors.base import BaseConnector


class SnowflakeConnector(BaseConnector):
    """
    Snowflake data warehouse connector.

    Config: { account, user, password, warehouse, database, schema, query }
    """

    def validate_config(self) -> List[str]:
        required = ["account", "user", "password", "warehouse", "database", "query"]
        return [f"Missing '{k}'" for k in required if k not in self.config]

    def test_connection(self) -> bool:
        try:
            import snowflake.connector  # type: ignore
        except ImportError:
            return False
        try:
            conn = snowflake.connector.connect(
                account=self.config["account"],
                user=self.config["user"],
                password=self.config["password"],
                warehouse=self.config["warehouse"],
                database=self.config["database"],
            )
            conn.close()
            return True
        except Exception:
            return False

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        import snowflake.connector  # type: ignore

        query = self.config["query"]
        if limit and "limit" not in query.lower():
            query = f"{query.rstrip(';')} LIMIT {limit}"

        conn = snowflake.connector.connect(
            account=self.config["account"],
            user=self.config["user"],
            password=self.config["password"],
            warehouse=self.config["warehouse"],
            database=self.config["database"],
            schema=self.config.get("schema"),
        )
        try:
            return pd.read_sql(query, conn)
        finally:
            conn.close()


class BigQueryConnector(BaseConnector):
    """
    BigQuery connector.

    Config: { project_id, query, credentials_json (optional) }
    """

    def validate_config(self) -> List[str]:
        return [f"Missing '{k}'" for k in ["project_id", "query"] if k not in self.config]

    def test_connection(self) -> bool:
        try:
            from google.cloud import bigquery  # type: ignore
            client = bigquery.Client(project=self.config["project_id"])
            list(client.list_datasets(max_results=1))
            return True
        except Exception:
            return False

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        from google.cloud import bigquery  # type: ignore

        client = bigquery.Client(project=self.config["project_id"])
        query = self.config["query"]
        if limit and "limit" not in query.lower():
            query = f"{query.rstrip(';')} LIMIT {limit}"
        return client.query(query).to_dataframe()


class PostgresConnector(BaseConnector):
    """
    PostgreSQL connector via SQLAlchemy.

    Config: { url, query } — url is a SQLAlchemy-style connection string.
    """

    def validate_config(self) -> List[str]:
        return [f"Missing '{k}'" for k in ["url", "query"] if k not in self.config]

    def test_connection(self) -> bool:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.config["url"])
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        from sqlalchemy import create_engine
        engine = create_engine(self.config["url"])
        query = self.config["query"]
        if limit and "limit" not in query.lower():
            query = f"{query.rstrip(';')} LIMIT {limit}"
        return pd.read_sql(query, engine)


class S3Connector(BaseConnector):
    """
    S3 connector (via pandas/s3fs).

    Config: { s3_uri, format (csv|parquet|json), aws_access_key_id, aws_secret_access_key }
    """

    def validate_config(self) -> List[str]:
        errors = []
        if "s3_uri" not in self.config:
            errors.append("Missing 's3_uri'")
        if "format" not in self.config:
            errors.append("Missing 'format' (csv|parquet|json)")
        return errors

    def test_connection(self) -> bool:
        try:
            import s3fs  # type: ignore
            fs = s3fs.S3FileSystem(
                key=self.config.get("aws_access_key_id"),
                secret=self.config.get("aws_secret_access_key"),
            )
            return fs.exists(self.config["s3_uri"])
        except Exception:
            return False

    def ingest(self, limit: Optional[int] = None) -> pd.DataFrame:
        s3_uri = self.config["s3_uri"]
        fmt = self.config["format"].lower()

        storage_options = {}
        if "aws_access_key_id" in self.config:
            storage_options["key"] = self.config["aws_access_key_id"]
            storage_options["secret"] = self.config["aws_secret_access_key"]

        if fmt == "csv":
            df = pd.read_csv(s3_uri, storage_options=storage_options)
        elif fmt == "parquet":
            df = pd.read_parquet(s3_uri, storage_options=storage_options)
        elif fmt == "json":
            df = pd.read_json(s3_uri, storage_options=storage_options)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return df.head(limit) if limit else df


# ---- Registry ----

CONNECTOR_REGISTRY: Dict[str, Any] = {
    "snowflake": SnowflakeConnector,
    "bigquery": BigQueryConnector,
    "postgres": PostgresConnector,
    "postgresql": PostgresConnector,
    "s3": S3Connector,
}


def get_connector_class(source_type: str) -> Optional[Any]:
    """Get the connector class for a given source type."""
    from src.connectors.file_connectors import (
        CSVConnector,
        ExcelConnector,
        JSONConnector,
        ParquetConnector,
    )
    registry = {
        **CONNECTOR_REGISTRY,
        "csv": CSVConnector,
        "excel": ExcelConnector,
        "xlsx": ExcelConnector,
        "json": JSONConnector,
        "parquet": ParquetConnector,
    }
    return registry.get(source_type.lower())
