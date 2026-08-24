"""Offline mirror of the BigQuery dataset, backed by DuckDB.

The agent always writes BigQuery SQL. This adapter transpiles it to DuckDB with
sqlglot and strips the `bigquery-public-data.thelook_ecommerce` qualification,
so exactly the same generated SQL runs against both backends. That property is
what makes the offline mode a real test surface rather than a separate code
path: the eval suite exercises the production SQL generator end to end without
a cloud account.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, List

import sqlglot
from sqlglot import exp

from .base import (
    ColumnSchema,
    QueryEstimate,
    QueryResult,
    TableSchema,
    WarehouseError,
    classify_sql_error,
)
from .seed_thelook import seed_duckdb

BQ_DATASET_TOKENS = {"bigquery-public-data", "thelook_ecommerce"}


def _strip_qualification(node: exp.Expression) -> exp.Expression:
    """`bigquery-public-data.thelook_ecommerce.orders` -> `orders`."""
    if isinstance(node, exp.Table):
        if node.args.get("db") and node.args["db"].name in BQ_DATASET_TOKENS:
            node.set("db", None)
        if node.args.get("catalog") and node.args["catalog"].name in BQ_DATASET_TOKENS:
            node.set("catalog", None)
    return node


def transpile_to_duckdb(sql: str) -> str:
    try:
        trees = sqlglot.parse(sql, read="bigquery")
    except Exception as exc:
        raise WarehouseError(f"Syntax error: {exc}", sql=sql, kind="syntax") from exc
    out = []
    for tree in trees:
        if tree is None:
            continue
        out.append(tree.transform(_strip_qualification).sql(dialect="duckdb"))
    if not out:
        raise WarehouseError("Empty statement", sql=sql, kind="syntax")
    return ";\n".join(out)


class DuckDBWarehouse:
    dialect = "bigquery"  # the dialect the AGENT writes; translation is internal
    qualified_prefix = "`bigquery-public-data.thelook_ecommerce`."

    def __init__(self, db_path: Path, today: dt.date | None = None) -> None:
        import duckdb

        self._duckdb = duckdb
        self.db_path = db_path
        self.seed_counts = seed_duckdb(db_path, today=today)
        self._con = duckdb.connect(str(db_path), read_only=True)
        self._con.execute("SET TimeZone='UTC'")
        self._schema_cache: Dict[str, TableSchema] = {}

    def estimate(self, sql: str) -> QueryEstimate:
        """DuckDB has no dry-run billing, so we validate by planning the query.

        This still catches the failure modes that matter for the self-correction
        loop - syntax errors, unknown columns, bad joins - before execution.
        """
        translated = transpile_to_duckdb(sql)
        try:
            self._con.execute(f"EXPLAIN {translated}")
        except Exception as exc:
            msg = str(exc)
            raise WarehouseError(msg, sql=sql, kind=classify_sql_error(msg)) from exc
        return QueryEstimate(bytes_processed=0, note="offline mirror: no billing")

    def execute(self, sql: str, *, max_bytes_billed: int | None = None) -> QueryResult:
        translated = transpile_to_duckdb(sql)
        started = time.perf_counter()
        try:
            df = self._con.execute(translated).fetch_df()
        except Exception as exc:
            msg = str(exc)
            raise WarehouseError(msg, sql=sql, kind=classify_sql_error(msg)) from exc
        return QueryResult(
            dataframe=df,
            bytes_billed=0,
            rows=len(df),
            latency_ms=(time.perf_counter() - started) * 1000,
            job_id="duckdb-local",
        )

    def schema(self, table: str) -> TableSchema:
        if table in self._schema_cache:
            return self._schema_cache[table]
        try:
            rows = self._con.execute(f"DESCRIBE {table}").fetchall()
            count = self._con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception as exc:
            raise WarehouseError(str(exc), kind="not_found") from exc
        ts = TableSchema(
            name=table,
            columns=[ColumnSchema(r[0], str(r[1]), "NULLABLE" if r[2] == "YES" else "REQUIRED") for r in rows],
            row_estimate=int(count),
        )
        self._schema_cache[table] = ts
        return ts

    def tables(self) -> List[str]:
        return sorted(r[0] for r in self._con.execute("SHOW TABLES").fetchall())

    def health(self) -> Dict[str, Any]:
        try:
            self._con.execute("SELECT 1").fetchone()
            return {"status": "ok", "backend": "duckdb-offline", "path": str(self.db_path),
                    "tables": self.seed_counts}
        except Exception as exc:
            return {"status": "down", "backend": "duckdb-offline", "error": str(exc)[:200]}
