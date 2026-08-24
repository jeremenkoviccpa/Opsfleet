"""Warehouse selection.

Instances are pooled per mode. Both backends hold a connection that is safe to
share and expensive to re-establish - and DuckDB actively refuses a second
connection to the same file from the same process.
"""
from __future__ import annotations

import os
import threading
from typing import Dict

from ..config import REPO_ROOT, setting
from .base import Warehouse

_POOL: Dict[str, Warehouse] = {}
_POOL_LOCK = threading.Lock()


def build_warehouse(mode: str | None = None) -> Warehouse:
    resolved = (mode or os.getenv("RIA_WAREHOUSE") or setting("runtime.warehouse", "offline")).lower()
    with _POOL_LOCK:
        if resolved not in _POOL:
            _POOL[resolved] = _construct(resolved)
        return _POOL[resolved]


def reset_pool() -> None:
    with _POOL_LOCK:
        _POOL.clear()


def _construct(mode: str) -> Warehouse:
    if mode in ("bigquery", "bq"):
        from .bigquery_adapter import BigQueryWarehouse

        return BigQueryWarehouse(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("BIGQUERY_PROJECT"),
            dataset_id=os.getenv("BQ_DATASET", "bigquery-public-data.thelook_ecommerce"),
        )
    from .duckdb_adapter import DuckDBWarehouse

    return DuckDBWarehouse(REPO_ROOT / ".runtime" / "thelook_offline.duckdb")
