"""BigQuery adapter.

Built on the provided BigQueryRunner and hardened for production use:

  * every query is dry-run first, so cost is known BEFORE it is incurred;
  * maximum_bytes_billed is set on the real job, so a mis-estimated query is
    killed by BigQuery rather than by our invoice;
  * errors are classified into repairable (syntax/type/grouping) vs terminal
    (permission/cost), which is what drives the self-correction loop.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional


from ..obs import metrics
from ..resilience.policies import RetryPolicy, call_with_resilience
from .base import (
    ColumnSchema,
    QueryEstimate,
    QueryResult,
    TableSchema,
    WarehouseError,
    classify_sql_error,
)

log = logging.getLogger(__name__)

THELOOK = "bigquery-public-data.thelook_ecommerce"


class BigQueryWarehouse:
    dialect = "bigquery"

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = THELOOK,
        location: str = "US",
    ) -> None:
        from google.cloud import bigquery

        self._bq = bigquery
        self.dataset_id = dataset_id
        self.location = location
        self.qualified_prefix = f"`{dataset_id}`."
        try:
            self.client = bigquery.Client(project=project_id)
        except Exception as exc:  # credential resolution failure
            raise WarehouseError(
                f"Could not initialise BigQuery client: {exc}. Run "
                "`gcloud auth application-default login` or set "
                "GOOGLE_APPLICATION_CREDENTIALS.",
                kind="permission",
            ) from exc
        self._schema_cache: Dict[str, TableSchema] = {}

    def estimate(self, sql: str) -> QueryEstimate:
        cfg = self._bq.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = call_with_resilience(
                lambda: self.client.query(sql, job_config=cfg, location=self.location),
                dependency="bigquery",
                policy=RetryPolicy(attempts=2, base_delay_s=0.3),
            )
        except Exception as exc:
            msg = str(exc)
            raise WarehouseError(msg, sql=sql, kind=classify_sql_error(msg)) from exc
        return QueryEstimate(bytes_processed=int(job.total_bytes_processed or 0))

    def execute(self, sql: str, *, max_bytes_billed: int | None = None) -> QueryResult:
        started = time.perf_counter()
        cfg = self._bq.QueryJobConfig(
            use_query_cache=True,
            maximum_bytes_billed=max_bytes_billed,
            labels={"app": "retail-insight-agent"},
        )
        try:
            job = call_with_resilience(
                lambda: self.client.query(sql, job_config=cfg, location=self.location),
                dependency="bigquery",
                policy=RetryPolicy(attempts=3, base_delay_s=0.5),
            )
            df = job.result(timeout=120).to_dataframe(create_bqstorage_client=False)
        except Exception as exc:
            msg = str(exc)
            metrics.incr("sql.failed", kind=classify_sql_error(msg))
            raise WarehouseError(msg, sql=sql, kind=classify_sql_error(msg)) from exc

        latency = (time.perf_counter() - started) * 1000
        billed = int(getattr(job, "total_bytes_billed", 0) or 0)
        metrics.observe("sql.bytes_billed", billed)
        return QueryResult(
            dataframe=df,
            bytes_billed=billed,
            rows=len(df),
            latency_ms=latency,
            job_id=getattr(job, "job_id", ""),
        )

    def schema(self, table: str) -> TableSchema:
        if table in self._schema_cache:
            return self._schema_cache[table]
        try:
            tbl = self.client.get_table(f"{self.dataset_id}.{table}")
        except Exception as exc:
            raise WarehouseError(str(exc), kind="not_found") from exc
        ts = TableSchema(
            name=table,
            columns=[
                ColumnSchema(f.name, f.field_type, f.mode or "NULLABLE", f.description or "")
                for f in tbl.schema
            ],
            row_estimate=int(tbl.num_rows or 0),
            description=tbl.description or "",
        )
        self._schema_cache[table] = ts
        return ts

    def tables(self) -> List[str]:
        return ["orders", "order_items", "products", "users", "distribution_centers", "inventory_items"]

    def health(self) -> Dict[str, Any]:
        try:
            self.estimate("SELECT 1")
            return {"status": "ok", "backend": "bigquery", "dataset": self.dataset_id}
        except Exception as exc:
            return {"status": "degraded", "backend": "bigquery", "error": str(exc)[:200]}
