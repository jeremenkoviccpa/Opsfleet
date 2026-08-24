"""Dependency container.

Every node receives the same Services object, so swapping the warehouse, the
LLM chain or the knowledge store is a construction-time decision rather than an
edit to the graph.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from .config import RUNTIME_DIR, resolve_path, setting
from .golden.store import GoldenBucket
from .llm import LLMRouter
from .memory import store as memory_store
from .obs import metrics
from .obs.tracing import Tracer
from .safety.guardrail import Guardrail
from .safety.pii import PIIGuard, guard as pii_guard
from .tools.report_store import ReportStore
from .tools.schema_catalog import SchemaCatalog
from .warehouse.base import Warehouse
from .warehouse.factory import build_warehouse


@dataclass
class Services:
    router: LLMRouter
    warehouse: Warehouse
    catalog: SchemaCatalog
    golden: GoldenBucket
    reports: ReportStore
    tracer: Tracer
    guardrail: Guardrail
    pii: PIIGuard

    def today(self) -> str:
        return dt.date.today().isoformat()

    def health(self) -> dict:
        return {
            "warehouse": self.warehouse.health(),
            "llm_providers": self.router.available_providers(),
            "golden_bucket": self.golden.stats(),
            "catalog": self.catalog.health(),
        }


def build_services(warehouse_mode: Optional[str] = None) -> Services:
    memory_store.configure(resolve_path(setting("runtime.session_db", ".runtime/agent_state.db")))
    metrics.configure(RUNTIME_DIR / "metrics.jsonl")

    tracer = Tracer(resolve_path(setting("runtime.trace_dir", ".runtime/traces")))
    router = LLMRouter(tracer=tracer)
    warehouse = build_warehouse(warehouse_mode)
    return Services(
        router=router,
        warehouse=warehouse,
        catalog=SchemaCatalog(warehouse),
        golden=GoldenBucket(),
        reports=ReportStore(),
        tracer=tracer,
        guardrail=Guardrail(router=router, tracer=tracer),
        pii=pii_guard(),
    )
