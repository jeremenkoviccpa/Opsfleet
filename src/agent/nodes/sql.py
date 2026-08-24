"""The SQL loop: generate -> validate -> execute, with bounded self-correction.

The cycle is a real cycle in the graph, not a while-loop hidden inside a node,
so every attempt is a separate traced span and the repair budget is part of the
state rather than a local variable.

What makes the loop terminate:
  * `repair_budget_left` is decremented on every repair and is never refilled;
  * a repair only happens for error classes a rewrite can plausibly fix -
    a permission error or a blown cost ceiling goes straight to "give up";
  * the turn-level TurnBudget caps total LLM calls independently, so even a
    pathological planner cannot run up a bill.

When the budget runs out the step is marked failed and the answer is composed
from whatever succeeded, with the failure stated explicitly. The turn never
crashes and the user never sees a stack trace.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..config import setting
from ..llm import BudgetExceeded, TurnBudget, strip_fences
from ..obs import metrics
from ..prompts import (
    REPAIR_STRATEGIES,
    SQL_REPAIR_SYSTEM,
    SQL_SYSTEM,
    sql_repair_user_block,
    sql_user_block,
)
from ..safety import sql_guard
from ..services import Services
from ..state import AgentState
from ..tools.formatting import df_to_markdown
from ..warehouse.base import WarehouseError

MAX_RECORDS_IN_CONTEXT = 40


def _current(state: AgentState) -> Optional[Dict[str, Any]]:
    steps = state.get("steps") or []
    idx = state.get("current_step", 0)
    return steps[idx] if 0 <= idx < len(steps) else None


def _patched(state: AgentState, **updates: Any) -> List[Dict[str, Any]]:
    steps = [dict(s) for s in (state.get("steps") or [])]
    idx = state.get("current_step", 0)
    if 0 <= idx < len(steps):
        steps[idx].update(updates)
    return steps


def _sql_system(svc: Services, tables: Optional[List[str]] = None) -> str:
    return SQL_SYSTEM.format(
        allowed_tables=", ".join(setting("safety.allowed_tables", [])),
        prefix=svc.catalog.prefix(),
        pii_rules=svc.pii.sensitive_summary(),
        schema=svc.catalog.render_for_sql_prompt(tables),
        semantics=svc.catalog.render_semantics(),
        precedents="(none)",
    )


def make_generate_sql_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        step = _current(state)
        if step is None:
            return {}
        with svc.tracer.span("generate_sql", kind="node", step_id=step["step_id"]) as span:
            system = SQL_SYSTEM.format(
                allowed_tables=", ".join(setting("safety.allowed_tables", [])),
                prefix=svc.catalog.prefix(),
                pii_rules=svc.pii.sensitive_summary(),
                schema=svc.catalog.render_for_sql_prompt(),
                semantics=svc.catalog.render_semantics(),
                precedents=state.get("precedent_block") or "(none)",
            )
            user = sql_user_block(
                goal=step["goal"], question=state["user_query"],
                time_window=state.get("time_window", ""), notes=state.get("plan_notes", ""),
                today=svc.today(),
            )
            try:
                result = svc.router.complete(
                    purpose="sql_generation", system=system,
                    messages=[{"role": "user", "content": user}], tier="fast", budget=budget,
                )
            except BudgetExceeded as exc:
                span.set(error=str(exc))
                return {"steps": _patched(state, status="failed", error=str(exc),
                                          error_kind="budget"),
                        "degraded": True, "degraded_reason": str(exc)}
            except Exception as exc:
                span.set(error=f"{type(exc).__name__}: {exc}")
                return {"steps": _patched(state, status="failed",
                                          error=f"SQL generation unavailable: {exc}",
                                          error_kind="llm_unavailable")}

            sql = strip_fences(result.text).strip().rstrip(";")
            metrics.incr("sql.generated")
            span.set(sql=sql, provider=result.provider)
            return {"steps": _patched(state, sql=sql, attempts=step.get("attempts", 0) + 1,
                                      status="generated")}

    return node


def make_validate_sql_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        step = _current(state)
        if step is None:
            return {}
        with svc.tracer.span("validate_sql", kind="safety", step_id=step["step_id"]) as span:
            verdict = sql_guard.validate(step.get("sql", ""))
            span.set(
                ok=verdict.ok, violations=verdict.violations, warnings=verdict.warnings,
                tables=verdict.tables, pii_actions=list(verdict.column_actions),
            )
            if not verdict.ok:
                metrics.incr("sql.validation_rejected", reason=(verdict.violations or ["?"])[0][:60])
                if any("classified PII" in v for v in verdict.violations):
                    metrics.incr("pii.columns_blocked")
                return {
                    "steps": _patched(state, status="rejected", error=verdict.reason,
                                      error_kind="rejected"),
                    "last_error": verdict.reason, "last_error_kind": "rejected",
                }
            return {
                "steps": _patched(
                    state, status="validated", validated_sql=verdict.sql,
                    column_actions=verdict.column_actions,
                    mask_note="; ".join(verdict.warnings),
                ),
                "warnings": (state.get("warnings") or []) + verdict.warnings,
            }

    return node


def make_execute_sql_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    max_per_query = int(setting("budget.max_bytes_billed_per_query", 2_000_000_000))
    max_rows = int(setting("budget.max_rows_returned", 5000))

    def node(state: AgentState) -> Dict[str, Any]:
        step = _current(state)
        if step is None:
            return {}
        sql = step.get("validated_sql") or step.get("sql", "")
        with svc.tracer.span("execute_sql", kind="sql", step_id=step["step_id"],
                             is_repair=step.get("repairs", 0) > 0) as span:
            # --- cost gate: know the price before paying it -----------------
            try:
                estimate = svc.warehouse.estimate(sql)
                span.set(estimated_bytes=estimate.bytes_processed)
                if estimate.bytes_processed > max_per_query:
                    msg = (f"query would scan {estimate.bytes_processed / 1e9:.2f} GB, over the "
                           f"{max_per_query / 1e9:.2f} GB per-query ceiling")
                    span.set(error=msg)
                    return {"steps": _patched(state, status="failed", error=msg, error_kind="cost"),
                            "last_error": msg, "last_error_kind": "cost"}
                budget.charge_bytes(estimate.bytes_processed)
            except WarehouseError as exc:
                # A dry-run failure is a repairable SQL problem caught for free.
                span.set(error=str(exc), kind=exc.kind)
                metrics.incr("sql.failed", kind=exc.kind, phase="estimate")
                return {"steps": _patched(state, status="failed", error=str(exc),
                                          error_kind=exc.kind),
                        "last_error": str(exc), "last_error_kind": exc.kind}
            except BudgetExceeded as exc:
                span.set(error=str(exc))
                return {"steps": _patched(state, status="failed", error=str(exc), error_kind="budget"),
                        "last_error": str(exc), "last_error_kind": "budget",
                        "degraded": True, "degraded_reason": str(exc)}

            # --- execute ----------------------------------------------------
            try:
                result = svc.warehouse.execute(sql, max_bytes_billed=max_per_query)
            except WarehouseError as exc:
                span.set(error=str(exc), kind=exc.kind)
                metrics.incr("sql.failed", kind=exc.kind, phase="execute")
                return {"steps": _patched(state, status="failed", error=str(exc),
                                          error_kind=exc.kind),
                        "last_error": str(exc), "last_error_kind": exc.kind}
            except Exception as exc:
                span.set(error=f"{type(exc).__name__}: {exc}")
                metrics.incr("sql.failed", kind="unknown", phase="execute")
                return {"steps": _patched(state, status="failed", error=str(exc),
                                          error_kind="unknown"),
                        "last_error": str(exc), "last_error_kind": "unknown"}

            metrics.incr("sql.executed")
            metrics.observe("sql.latency_ms", result.latency_ms)
            if step.get("repairs", 0) > 0:
                metrics.incr("sql.repaired")

            # --- PII layer 2: mask BEFORE the rows reach the model ----------
            masked, report = svc.pii.mask_dataframe(
                result.dataframe.head(max_rows), step.get("column_actions") or {}
            )
            if report.values_redacted:
                metrics.incr("pii.values_redacted", n=report.values_redacted)
            span.set(
                rows=result.rows, latency_ms=round(result.latency_ms, 1),
                bytes_billed=result.bytes_billed, job_id=result.job_id,
                masking=report.describe() or "none",
            )

            if result.rows == 0:
                metrics.incr("sql.empty_result")
                msg = "the query executed successfully but matched zero rows"
                return {"steps": _patched(state, status="empty", rows=0, error=msg,
                                          error_kind="empty", preview_md="_(0 rows)_",
                                          columns=[str(c) for c in result.dataframe.columns],
                                          latency_ms=result.latency_ms),
                        "last_error": msg, "last_error_kind": "empty"}

            records = masked.head(MAX_RECORDS_IN_CONTEXT).to_dict("records")
            return {
                "steps": _patched(
                    state, status="ok", rows=result.rows,
                    columns=[str(c) for c in masked.columns],
                    preview_md=df_to_markdown(masked),
                    records=records, bytes_billed=result.bytes_billed,
                    latency_ms=result.latency_ms,
                    mask_note=report.describe(), error="", error_kind="",
                ),
                "last_error": "", "last_error_kind": "",
            }

    return node


def make_repair_sql_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        step = _current(state)
        if step is None:
            return {}
        kind = step.get("error_kind") or "unknown"
        strategy = REPAIR_STRATEGIES.get(kind, REPAIR_STRATEGIES["unknown"])
        with svc.tracer.span("repair_sql", kind="node", step_id=step["step_id"],
                             error_kind=kind, attempt=step.get("repairs", 0) + 1) as span:
            user = sql_repair_user_block(
                goal=step["goal"], sql=step.get("sql", ""), error=step.get("error", ""),
                error_kind=kind, attempt=step.get("repairs", 0) + 1,
                schema_hint=svc.catalog.render_for_sql_prompt(),
            )
            try:
                result = svc.router.complete(
                    purpose="sql_repair",
                    system=SQL_REPAIR_SYSTEM.format(strategy=strategy),
                    messages=[{"role": "user", "content": user}], tier="fast", budget=budget,
                )
            except (BudgetExceeded, Exception) as exc:
                span.set(error=f"{type(exc).__name__}: {exc}")
                return {"steps": _patched(state, status="failed",
                                          error=f"{step.get('error', '')} (repair unavailable: {exc})"),
                        "repair_budget_left": 0}

            sql = strip_fences(result.text).strip().rstrip(";")
            span.set(repaired_sql=sql)
            return {
                "steps": _patched(state, sql=sql, status="generated",
                                  repairs=step.get("repairs", 0) + 1,
                                  attempts=step.get("attempts", 0) + 1),
                "repair_budget_left": max(0, state.get("repair_budget_left", 0) - 1),
            }

    return node


def make_advance_step_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        step = _current(state)
        if step is not None and step.get("status") not in ("ok", "empty"):
            metrics.incr("sql.gave_up", kind=step.get("error_kind", "unknown"))
        return {
            "current_step": state.get("current_step", 0) + 1,
            # Each step gets its own repair budget; a cheap first step must not
            # starve a harder later one.
            "repair_budget_left": int(setting("budget.max_sql_repair_attempts", 2)),
            "last_error": "", "last_error_kind": "",
        }

    return node


# --------------------------------------------------------------------------
# Conditional edges
# --------------------------------------------------------------------------
REPAIRABLE = {"syntax", "not_found", "type", "grouping", "ambiguous", "rejected", "empty", "unknown"}


def after_generate(state: AgentState) -> str:
    """Skip validation when generation produced nothing.

    Without this the empty string reaches the validator, fails as "empty
    statement", and burns the repair budget on a model that is already down.
    """
    step = _current(state)
    if step is None:
        return "advance"
    return "validate" if step.get("sql") else "advance"


def after_validate(state: AgentState) -> str:
    step = _current(state)
    if step is None:
        return "advance"
    if step.get("status") == "validated":
        return "execute"
    return "repair" if state.get("repair_budget_left", 0) > 0 else "advance"


def after_execute(state: AgentState) -> str:
    step = _current(state)
    if step is None:
        return "advance"
    status = step.get("status")
    if status == "ok":
        return "advance"
    kind = step.get("error_kind", "unknown")
    # An empty result is repaired at most once - widening forever turns a
    # legitimately empty answer into an expensive wrong one.
    if status == "empty" and step.get("repairs", 0) >= 1:
        return "advance"
    if kind in REPAIRABLE and state.get("repair_budget_left", 0) > 0:
        return "repair"
    return "advance"


def after_advance(state: AgentState) -> str:
    return "next_step" if state.get("current_step", 0) < len(state.get("steps") or []) else "synthesize"
