"""Guardrail, precedent retrieval and planning."""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..config import setting
from ..llm import BudgetExceeded, TurnBudget
from ..obs import metrics
from ..prompts import PLANNER_SYSTEM, planner_user_block, render_history
from ..safety.guardrail import REFUSAL_COPY
from ..services import Services
from ..state import AgentState

MAX_STEPS = 4


def make_guardrail_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        with svc.tracer.span("guardrail", kind="safety") as span:
            verdict = svc.guardrail.check(state["user_query"], budget=budget)
            span.set(
                decision=verdict.decision, stage=verdict.stage,
                reasons=verdict.reasons, matched=verdict.matched_rules,
                confidence=verdict.confidence,
            )
            if verdict.allowed:
                return {"guard_decision": verdict.decision, "guard_stage": verdict.stage,
                        "guard_reasons": verdict.reasons}
            return {
                "guard_decision": verdict.decision,
                "guard_stage": verdict.stage,
                "guard_reasons": verdict.reasons,
                "route": "refused",
                "answer": verdict.user_message or REFUSAL_COPY.get(verdict.decision, ""),
            }

    return node


def make_retrieve_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        with svc.tracer.span("retrieve_precedents", kind="retrieval") as span:
            # Retrieve against the question plus the last exchange, so a terse
            # follow-up ("and Texas?") still lands on the right precedent.
            history = state.get("history") or []
            tail = " ".join(m.get("content", "") for m in history[-2:])
            query = f"{state['user_query']} {tail}".strip()
            hits = svc.golden.search(
                query,
                k=int(setting("retrieval.golden_bucket_top_k", 4)),
                bm25_weight=float(setting("retrieval.bm25_weight", 0.45)),
                dense_weight=float(setting("retrieval.semantic_weight", 0.55)),
                min_score=float(setting("retrieval.min_hybrid_score", 0.12)),
            )
            block = "\n\n".join(h.as_prompt_block() for h in hits)
            span.set(
                hits=len(hits),
                top=[{"trio_id": h.trio.trio_id, "score": round(h.score, 3)} for h in hits],
                embedder=svc.golden.embedder.name,
            )
            return {
                "precedents": [
                    {"trio_id": h.trio.trio_id, "question": h.trio.question,
                     "score": round(h.score, 3), "tags": h.trio.intent_tags}
                    for h in hits
                ],
                "precedent_block": block,
            }

    return node


def _fallback_plan(state: AgentState, reason: str) -> Dict[str, Any]:
    """Heuristic route used when the planner LLM is unavailable.

    Degrading to a single-step analysis is better than failing the turn: the SQL
    generator and validator still run, and the user gets an answer with a note.
    """
    q = (state.get("user_query") or "").lower()
    if any(w in q for w in ("what data", "what can you", "which tables", "schema", "available data")):
        route = "schema"
    elif any(w in q for w in ("delete", "remove", "purge")) and "report" in q:
        route = "delete_reports"
    elif len(q.split()) <= 3 and any(w in q for w in ("hi", "hello", "thanks", "thank you")):
        route = "converse"
    else:
        route = "analysis"
    return {
        "route": route,
        "intent_reason": f"heuristic fallback ({reason})",
        "time_window": "",
        "plan_notes": "",
        "steps": [{"step_id": "s1", "goal": state.get("user_query", ""), "attempts": 0,
                   "repairs": 0, "status": "pending"}],
        "current_step": 0,
        "repair_budget_left": int(setting("budget.max_sql_repair_attempts", 2)),
        "degraded": True,
        "degraded_reason": f"planner unavailable, used heuristic routing ({reason})",
    }


def make_plan_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        with svc.tracer.span("plan", kind="node") as span:
            system = PLANNER_SYSTEM.format(max_steps=MAX_STEPS)
            user = planner_user_block(
                question=state["user_query"],
                history_block=render_history(state.get("history") or []),
                precedent_block=state.get("precedent_block", ""),
                capability=svc.catalog.capability_summary(),
                today=svc.today(),
            )
            try:
                parsed, _ = svc.router.complete_json(
                    purpose="plan", system=system,
                    messages=[{"role": "user", "content": user}],
                    default=None, tier="fast", budget=budget,
                )
            except BudgetExceeded as exc:
                span.set(fallback=str(exc))
                return _fallback_plan(state, "budget exhausted")
            except Exception as exc:
                span.set(fallback=f"{type(exc).__name__}: {exc}")
                return _fallback_plan(state, type(exc).__name__)

            if not isinstance(parsed, dict):
                span.set(fallback="planner returned no JSON")
                return _fallback_plan(state, "unparseable planner output")

            route = str(parsed.get("route", "analysis"))
            if route not in ("analysis", "schema", "delete_reports", "save_report", "converse"):
                route = "analysis"

            raw_steps = parsed.get("steps") or []
            steps: List[Dict[str, Any]] = []
            for i, item in enumerate(raw_steps[:MAX_STEPS]):
                goal = (item or {}).get("goal") if isinstance(item, dict) else str(item)
                if not goal:
                    continue
                steps.append({
                    "step_id": (item.get("step_id") if isinstance(item, dict) else None) or f"s{i + 1}",
                    "goal": goal, "attempts": 0, "repairs": 0, "status": "pending",
                })
            if route == "analysis" and not steps:
                steps = [{"step_id": "s1", "goal": state["user_query"], "attempts": 0,
                          "repairs": 0, "status": "pending"}]

            criteria = parsed.get("deletion_criteria") or {}
            span.set(
                route=route, steps=len(steps), reason=parsed.get("reason", ""),
                time_window=parsed.get("time_window", ""),
            )
            metrics.incr(f"route.{route}")
            return {
                "route": route,
                "intent_reason": str(parsed.get("reason", "")),
                "time_window": str(parsed.get("time_window", "") or ""),
                "plan_notes": str(parsed.get("notes", "") or ""),
                "steps": steps,
                "current_step": 0,
                "repair_budget_left": int(setting("budget.max_sql_repair_attempts", 2)),
                "deletion_preview": {"criteria": criteria,
                                     "report_title": parsed.get("report_title", "")},
            }

    return node
