"""Saved-reports operations, including the destructive-op confirmation gate.

The gate is a real graph interrupt, not a prompt asking the model to be careful.
`resolve_deletion` is pure read; the graph then suspends at `confirm_deletion`
and the checkpointer persists the pending state. Nothing is mutated until the
process resumes with an explicit approval carrying the token bound to that exact
resolved set.

Consequences of building it this way:
  * the model cannot delete anything by emitting a confident-sounding sentence;
  * a crash between resolve and confirm loses the plan, never the data;
  * approving a set is impossible if the set changed, because the token is
    minted from the resolution.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from langgraph.types import interrupt

from ..services import Services
from ..state import AgentState
from ..tools.formatting import truncate

CANCELLED_COPY = "Nothing was deleted. Your reports are untouched."


def _criteria_from_plan(state: AgentState) -> Dict[str, Any]:
    raw = ((state.get("deletion_preview") or {}).get("criteria") or {})
    criteria: Dict[str, Any] = {
        "mentions": [m for m in (raw.get("mentions") or []) if isinstance(m, str) and m.strip()],
        "all": bool(raw.get("all")),
    }
    if raw.get("session_scope"):
        criteria["session_id"] = state["session_id"]
    return criteria


def make_resolve_deletion_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        with svc.tracer.span("resolve_deletion", kind="destructive_op") as span:
            criteria = _criteria_from_plan(state)
            span.set(criteria=criteria)

            if not (criteria["mentions"] or criteria.get("session_id") or criteria["all"]):
                # Refuse to guess at scope. An under-specified delete is the one
                # case where asking again costs far less than being wrong.
                owned = svc.reports.list(state["user_id"], limit=10)
                listing = "\n".join(f"- `{r.report_id}` — {r.title}" for r in owned) or "_(none yet)_"
                return {
                    "answer": "I need to know which reports you mean before I delete anything.\n\n"
                              "You can say *\"delete the reports mentioning <name>\"*, "
                              "*\"delete the reports from this conversation\"*, or give me the "
                              f"ids directly.\n\nYour most recent reports:\n{listing}",
                    "deletion_preview": {"criteria": criteria, "matched": 0},
                }

            plan = svc.reports.resolve_deletion(user_id=state["user_id"], criteria=criteria)
            span.set(token=plan.token, matched=plan.count, ids=plan.ids())

            if plan.count == 0:
                svc.reports.cancel_plan(plan.token, state["user_id"])
                described = ", ".join(criteria["mentions"]) or (
                    "this conversation" if criteria.get("session_id") else "that description")
                return {
                    "answer": f"I couldn't find any of your saved reports matching **{described}**, "
                              f"so there is nothing to delete.",
                    "deletion_preview": {"criteria": criteria, "matched": 0},
                }

            preview_rows = [
                {"report_id": r.report_id, "title": r.title,
                 "created_at": r.created_at[:16].replace("T", " "),
                 "reason": plan.match_reasons.get(r.report_id, ""),
                 "excerpt": truncate(r.body_md.replace("\n", " "), 90)}
                for r in plan.reports
            ]
            return {
                "deletion_token": plan.token,
                "deletion_preview": {
                    "criteria": criteria, "matched": plan.count, "rows": preview_rows,
                    "requires_phrase": plan.requires_phrase,
                    "confirm_phrase": plan.confirm_phrase, "token": plan.token,
                },
            }

    return node


def make_confirm_deletion_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        preview = state.get("deletion_preview") or {}
        # Suspend the graph. The checkpointer persists everything above; the CLI
        # renders the preview and resumes with the user's decision.
        decision = interrupt({
            "kind": "confirm_deletion",
            "token": state.get("deletion_token", ""),
            "matched": preview.get("matched", 0),
            "rows": preview.get("rows", []),
            "requires_phrase": preview.get("requires_phrase", False),
            "confirm_phrase": preview.get("confirm_phrase", ""),
        })
        approved = bool(isinstance(decision, dict) and decision.get("approved"))
        phrase = str((decision or {}).get("phrase", "")) if isinstance(decision, dict) else ""
        return {"deletion_result": {"approved": approved, "phrase": phrase}}

    return node


def make_apply_deletion_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        result = state.get("deletion_result") or {}
        token = state.get("deletion_token", "")
        preview = state.get("deletion_preview") or {}

        with svc.tracer.span("apply_deletion", kind="destructive_op", token=token) as span:
            if not result.get("approved"):
                svc.reports.cancel_plan(token, state["user_id"])
                span.set(outcome="cancelled")
                return {"answer": CANCELLED_COPY, "deletion_result": {"deleted": 0, "cancelled": True}}

            if preview.get("requires_phrase"):
                expected = (preview.get("confirm_phrase") or "").strip().lower()
                given = (result.get("phrase") or "").strip().lower()
                if given != expected:
                    svc.reports.cancel_plan(token, state["user_id"])
                    span.set(outcome="phrase_mismatch", expected=expected, given=given)
                    return {
                        "answer": f"The confirmation phrase didn't match (I needed exactly "
                                  f"`{preview.get('confirm_phrase')}`), so I stopped. "
                                  f"{CANCELLED_COPY}",
                        "deletion_result": {"deleted": 0, "cancelled": True},
                    }

            try:
                count, ids = svc.reports.confirm_deletion(token, state["user_id"], state.get("trace_id", ""))
            except (KeyError, PermissionError) as exc:
                span.set(outcome="rejected", error=str(exc))
                return {"answer": f"I couldn't complete that deletion: {exc} {CANCELLED_COPY}",
                        "deletion_result": {"deleted": 0, "cancelled": True}}

            span.set(outcome="deleted", count=count, ids=ids)
            days = 30
            titles = "\n".join(f"- {r['title']}" for r in preview.get("rows", [])[:8])
            more = "" if len(preview.get("rows", [])) <= 8 else \
                f"\n- …and {len(preview['rows']) - 8} more"
            return {
                "answer": f"Deleted **{count} report{'s' if count != 1 else ''}**:\n{titles}{more}\n\n"
                          f"They're recoverable for {days} days — say *\"undo that delete\"* "
                          f"and I'll restore them.",
                "deletion_result": {"deleted": count, "ids": ids, "cancelled": False},
            }

    return node


def make_persist_report_node(svc: Services) -> Callable[[AgentState], Dict[str, Any]]:
    def node(state: AgentState) -> Dict[str, Any]:
        if not state.get("answer_is_report") or not state.get("answer"):
            return {}
        with svc.tracer.span("persist_report", kind="node") as span:
            # Prefer the planner's title, then a markdown H1 the writer produced,
            # then the question itself.
            title = (state.get("deletion_preview") or {}).get("report_title") or ""
            if not title:
                for line in (state.get("answer") or "").splitlines():
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
            title = title or truncate(state["user_query"], 80)
            entities = _entities(state)
            report = svc.reports.create(
                user_id=state["user_id"], session_id=state["session_id"],
                title=title, body_md=state["answer"], entities=entities,
                tags=[p.get("tags", []) and p["tags"][0] for p in state.get("precedents", [])[:1]],
                sql_refs=[s.get("validated_sql", "") for s in (state.get("steps") or [])
                          if s.get("status") == "ok"],
                trace_id=state.get("trace_id", ""),
            )
            span.set(report_id=report.report_id, title=title, entities=entities)
            return {
                "saved_report": {"report_id": report.report_id, "title": report.title},
                "answer": state["answer"] +
                f"\n\n---\n_Saved to your report library as_ `{report.report_id}` — "
                f"**{report.title}**",
            }

    return node


def _entities(state: AgentState) -> List[str]:
    """Entity tags used later to answer 'delete reports mentioning X'."""
    import re

    text = f"{state.get('user_query', '')}\n{state.get('answer', '')}"
    found = set()
    # Proper nouns and quoted names carry the entities managers actually name.
    for match in re.findall(r"\b([A-Z][a-zA-Z&'’-]{2,}(?:\s+[A-Z][a-zA-Z&'’-]{2,}){0,2})\b", text):
        if match.lower() not in {"the", "this", "that", "report", "executive summary", "key metrics"}:
            found.add(match.strip())
    for match in re.findall(r"[\"“']([^\"”']{3,40})[\"”']", text):
        found.add(match.strip())
    return sorted(found)[:25]
