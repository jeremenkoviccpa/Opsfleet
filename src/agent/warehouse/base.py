"""Warehouse adapter protocol.

Adding a new data source means implementing this protocol and registering it -
no node, prompt or graph edge changes. See docs/HLD.md #extensibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

import pandas as pd


@dataclass
class QueryEstimate:
    bytes_processed: int
    cache_hit: bool = False
    note: str = ""


@dataclass
class QueryResult:
    dataframe: pd.DataFrame
    bytes_billed: int = 0
    rows: int = 0
    latency_ms: float = 0.0
    truncated: bool = False
    job_id: str = ""

    @property
    def is_empty(self) -> bool:
        return self.dataframe is None or len(self.dataframe) == 0


@dataclass
class ColumnSchema:
    name: str
    type: str
    mode: str = "NULLABLE"
    description: str = ""


@dataclass
class TableSchema:
    name: str
    columns: List[ColumnSchema] = field(default_factory=list)
    row_estimate: int = 0
    description: str = ""

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]


class WarehouseError(RuntimeError):
    """Raised for a query that failed in a way the agent may be able to repair."""

    def __init__(self, message: str, *, sql: str = "", kind: str = "unknown") -> None:
        super().__init__(message)
        self.sql = sql
        self.kind = kind  # syntax | not_found | permission | timeout | cost | unknown


@runtime_checkable
class Warehouse(Protocol):
    dialect: str
    qualified_prefix: str

    def estimate(self, sql: str) -> QueryEstimate: ...
    def execute(self, sql: str, *, max_bytes_billed: int | None = None) -> QueryResult: ...
    def schema(self, table: str) -> TableSchema: ...
    def tables(self) -> List[str]: ...
    def health(self) -> Dict[str, Any]: ...


def classify_sql_error(message: str) -> str:
    m = message.lower()
    if "syntax error" in m or "unexpected" in m or "expected" in m and "at [" in m:
        return "syntax"
    if "not found" in m or "unrecognized name" in m or "does not exist" in m or "no such" in m:
        return "not_found"
    if "permission" in m or "access denied" in m or "forbidden" in m:
        return "permission"
    if "exceeded" in m and "bytes" in m:
        return "cost"
    if "timeout" in m or "deadline" in m:
        return "timeout"
    if "type" in m and ("mismatch" in m or "cannot be" in m or "no matching signature" in m):
        return "type"
    if "ambiguous" in m:
        return "ambiguous"
    if "aggregate" in m or "group by" in m:
        return "grouping"
    return "unknown"
