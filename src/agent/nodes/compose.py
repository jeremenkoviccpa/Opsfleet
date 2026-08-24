"""Answer composition.

The model that writes the answer never sees raw warehouse rows - it sees the
masked frame produced by the execute node. The final text then passes through
one more regex sweep before it reaches the user, which catches anything the
model reconstructed or a free-text column carried through.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..llm import BudgetExceeded, TurnBudget
from ..memory import user_profile
from ..obs import metrics
from ..prompts import (
    ANALYST_SYSTEM,
    CONVERSE_SYSTEM,
    REPORT_DIRECTIVE,
    SCHEMA_SYSTEM,
    analyst_user_block,
    render_history,
)
from ..services import Services
from ..state import AgentState
from ..tools.formatting import truncate
from ..config import load_persona

FALLBACK_ANSWER = (
    "I ran into a problem composing the answer and I don't want to guess at numbers.\n\n"
    "{detail}\n\nThe figures I did retrieve are below — ask me again and I'll retry."
)


def _persona(state: AgentState) -> Dict[str, Any]:
    try:
        return load_persona(state.get("persona_id") or "exec_default")
    except FileNotFoundError:
        return load_persona("exec_default")


def _audience(state: AgentState) -> str:
    name = state.get("user_display_name") or "a store manager"
    return f"{name}, a non-technical retail manager who owns a P&L"


def _results_block(state: AgentState) -> str:
    blocks: List[str] = []
    for step in state.get("steps") or []:
        header = f"--- STEP {step.get('step_id')}: {step.get('goal')}"
        status = step.get("status")
        if status == "ok":
            note = f" (masking applied: {step['mask_note']})" if step.get("mask_note") else ""
            blocks.append(
                f"{header}\nrows returned: {step.get('rows')}{note}\n{step.get('preview_md', '')}"
            )
        elif status == "empty":
            blocks.append(
                f"{header}\nRESULT: zero rows. The filters matched nothing. "
                f"Tell the manager this plainly - do not substitute a guess."
            )
        else:
            blocks.append(
                f"{header}\nFAILED after {step.get('attempts', 0)} attempt(s): "
                f"{truncate(step.get('error', 'unknown error'), 300)}\n"
                f"You do NOT have these figures. Say which part of the question you could not answer."
            )
    return "\n\n".join(blocks) if blocks else "(no queries were run)"


def _degraded_note(state: AgentState) -> str:
    if not state.get("degraded"):
        return ""
    return (
        f"SERVICE NOTE: the agent ran in a degraded mode this turn "
        f"({state.get('degraded_reason', 'unknown')}). Mention this in one short line at the end "
        f"so the manager knows the answer may be less complete than usual."
    )


def _scrub(svc: Services, text: str, state: AgentState) -> str:
    cleaned, hits = svc.pii.scrub_text(text or "")
    if hits:
        metrics.incr("pii.values_redacted", n=len(hits), stage="output")
        with svc.tracer.span("output_scrub", kind="safety") as span:
            span.set(patterns=sorted(set(hits)), count=len(hits))
    return cleaned


def make_synthesize_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        persona = _persona(state)
        wants_report = bool((state.get("deletion_preview") or {}).get("report_title")) or \
            state.get("route") == "save_report"

        with svc.tracer.span("synthesize", kind="node", is_report=wants_report) as span:
            system = ANALYST_SYSTEM.format(
                audience=_audience(state),
                persona_tone=persona.get("tone", ""),
                persona_format=persona.get("report_format" if wants_report else "default_format", ""),
                persona_rules="\n".join(f"- {r}" for r in persona.get("rules", [])),
                user_preferences=user_profile.render_for_prompt(state["user_id"]),
                extra_directive=REPORT_DIRECTIVE if wants_report else
                f"Keep the answer under {persona.get('max_answer_words', 400)} words.",
            )
            user = analyst_user_block(
                question=state["user_query"],
                history_block=render_history(state.get("history") or []),
                precedent_block=state.get("precedent_block", ""),
                results_block=_results_block(state),
                time_window=state.get("time_window", ""),
                notes=state.get("plan_notes", ""),
                degraded_note=_degraded_note(state),
            )
            try:
                result = svc.router.complete(
                    purpose="analysis", system=system,
                    messages=[{"role": "user", "content": user}],
                    tier="reasoning", budget=budget,
                )
                answer = result.text.strip()
                span.set(provider=result.provider, words=len(answer.split()))
            except (BudgetExceeded, Exception) as exc:
                span.status = "error"
                span.error = f"{type(exc).__name__}: {exc}"[:300]
                span.set(error=f"{type(exc).__name__}: {exc}")
                # Never crash the turn: hand back the retrieved figures with an
                # honest explanation instead of an exception.
                answer = FALLBACK_ANSWER.format(detail=str(exc)[:300]) + "\n\n" + _results_block(state)
                return {"answer": _scrub(svc, answer, state), "degraded": True,
                        "degraded_reason": f"answer composition failed: {type(exc).__name__}",
                        "errors": (state.get("errors") or []) + [str(exc)[:300]]}

            title = ""
            if wants_report and answer.upper().startswith("TITLE:"):
                first, _, rest = answer.partition("\n")
                title = first.split(":", 1)[1].strip()
                answer = rest.strip()

            return {
                "answer": _scrub(svc, answer, state),
                "answer_is_report": wants_report,
                "deletion_preview": {**(state.get("deletion_preview") or {}),
                                     "report_title": title or
                                     (state.get("deletion_preview") or {}).get("report_title", "")},
            }

    return node


def make_schema_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        persona = _persona(state)
        with svc.tracer.span("schema_answer", kind="node") as span:
            system = SCHEMA_SYSTEM.format(
                audience=_audience(state),
                persona_tone=persona.get("tone", ""),
                user_preferences=user_profile.render_for_prompt(state["user_id"]),
                catalog=svc.catalog.describe_for_user(),
                precedents=state.get("precedent_block", "") or "(none)",
            )
            try:
                result = svc.router.complete(
                    purpose="schema_answer", system=system,
                    messages=[{"role": "user", "content": state["user_query"]}],
                    tier="fast", budget=budget,
                )
                span.set(provider=result.provider)
                return {"answer": _scrub(svc, result.text.strip(), state)}
            except Exception as exc:
                span.status = "error"
                span.error = str(exc)[:300]
                span.set(error=str(exc))
                # The catalog is local, so a useful answer survives an LLM outage.
                return {"answer": svc.catalog.describe_for_user(), "degraded": True,
                        "degraded_reason": f"LLM unavailable, served the raw catalog ({type(exc).__name__})"}

    return node


def make_converse_node(svc: Services, budget: TurnBudget) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        persona = _persona(state)
        with svc.tracer.span("converse", kind="node") as span:
            system = CONVERSE_SYSTEM.format(
                audience=_audience(state),
                persona_tone=persona.get("tone", ""),
                user_preferences=user_profile.render_for_prompt(state["user_id"]),
            )
            messages = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in (state.get("history") or [])[-6:]
            ] + [{"role": "user", "content": state["user_query"]}]
            try:
                result = svc.router.complete(
                    purpose="converse", system=system, messages=messages,
                    tier="fast", budget=budget,
                )
                span.set(provider=result.provider)
                return {"answer": _scrub(svc, result.text.strip(), state)}
            except Exception as exc:
                span.status = "error"
                span.error = str(exc)[:300]
                span.set(error=str(exc))
                return {"answer": "I'm having trouble reaching my language model right now. "
                                  "Please try again in a moment.",
                        "degraded": True, "degraded_reason": str(exc)[:200]}

    return node


def make_refusal_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        metrics.incr("turns.refused", decision=state.get("guard_decision", "?"))
        return {"answer": state.get("answer", "I can't help with that one.")}

    return node
