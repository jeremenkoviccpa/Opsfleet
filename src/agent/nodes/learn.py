"""The learning loop.

Two levels, both write-behind so neither can slow down or fail a turn:

  user level   - extract durable formatting/depth preferences from the message
                 and reinforce them in the profile store. Only preferences that
                 clear a confidence threshold are injected into later prompts.
  system level - every successful analysis is written to the Golden Bucket as a
                 *candidate* trio with its question, validated SQL and report.
                 Candidates are not retrievable until a human promotes them, so
                 the agent cannot teach itself its own mistakes.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List

from ..config import setting
from ..golden.store import Trio
from ..llm import BudgetExceeded, TurnBudget
from ..memory import user_profile
from ..prompts import PREFERENCE_SYSTEM, TRIO_CURATION_SYSTEM
from ..services import Services
from ..state import AgentState


def make_learn_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        learned: List[str] = []
        with svc.tracer.span("learn", kind="node") as span:
            learned += _learn_preferences(svc, state, budget, span)
            if state.get("route") in ("analysis", "save_report"):
                learned += _propose_trio(svc, state, budget, span)
            span.set(learned=learned)
        return {"learned": learned}

    return node


def _learn_preferences(svc: Services, state: AgentState, budget: TurnBudget, span) -> List[str]:
    if budget.remaining_llm_calls() < 2:
        # Preference extraction is the first thing to drop under budget
        # pressure - the answer matters more than the memory update.
        span.set(prefs_skipped="budget")
        return []
    try:
        parsed, _ = svc.router.complete_json(
            purpose="preference_extraction", system=PREFERENCE_SYSTEM,
            messages=[{"role": "user", "content": state["user_query"]}],
            default={"signals": []}, tier="fast", budget=budget,
        )
    except (BudgetExceeded, Exception) as exc:
        span.set(prefs_error=f"{type(exc).__name__}: {exc}")
        return []

    out: List[str] = []
    for signal in (parsed or {}).get("signals", [])[:4]:
        if not isinstance(signal, dict):
            continue
        pref = user_profile.record(
            state["user_id"], str(signal.get("key", "")), str(signal.get("value", "")),
            source="explicit" if signal.get("explicit") else "inferred",
            evidence=str(signal.get("evidence", ""))[:200],
        )
        if pref:
            marker = "stated" if pref.source == "explicit" else f"conf {pref.confidence:.2f}"
            out.append(f"{pref.key}={pref.value} ({marker})")
    return out


def _propose_trio(svc: Services, state: AgentState, budget: TurnBudget, span) -> List[str]:
    successful = [s for s in (state.get("steps") or []) if s.get("status") == "ok"]
    if not successful or not state.get("answer"):
        return []
    if budget.remaining_llm_calls() < 1:
        span.set(trio_skipped="budget")
        return []

    sql = successful[0].get("validated_sql", "")
    try:
        parsed, _ = svc.router.complete_json(
            purpose="trio_curation", system=TRIO_CURATION_SYSTEM,
            messages=[{"role": "user", "content":
                       f"QUESTION\n{state['user_query']}\n\nSQL\n{sql}\n\n"
                       f"ANALYST WRITE-UP\n{state['answer'][:2500]}"}],
            default=None, tier="fast", budget=budget,
        )
    except (BudgetExceeded, Exception) as exc:
        span.set(trio_error=f"{type(exc).__name__}: {exc}")
        return []
    if not isinstance(parsed, dict):
        return []

    quality = float(parsed.get("quality", 0.0) or 0.0)
    if quality < 0.5:
        span.set(trio_skipped=f"low reusability ({quality:.2f})")
        return []

    trio = Trio(
        trio_id=f"cand_{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
        question=str(parsed.get("generalised_question") or state["user_query"]),
        sql=sql,
        analyst_report=state["answer"][:4000],
        intent_tags=[str(t) for t in (parsed.get("intent_tags") or [])][:5],
        tables=sorted({t for s in successful for t in _tables_of(s)}),
        analyst_method_notes=[str(n) for n in (parsed.get("method_notes") or [])][:4],
        author=f"agent:{state['user_id']}",
        quality_score=quality,
        status="candidate",
        created_at=time.strftime("%Y-%m-%d"),
    )
    path = svc.golden.add_candidate(trio)
    threshold = float(setting("learning.trio_promotion_min_score", 0.8))
    span.set(trio_candidate=trio.trio_id, quality=quality, path=str(path))
    return [f"golden-bucket candidate {trio.trio_id} (quality {quality:.2f}, "
            f"{'awaiting review' if quality < threshold else 'ready to promote'})"]


def _tables_of(step: Dict[str, Any]) -> List[str]:
    from ..safety.sql_guard import validate

    try:
        return validate(step.get("validated_sql", "")).tables
    except Exception:
        return []
