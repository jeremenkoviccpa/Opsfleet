"""Graph assembly.

    guardrail ─┬─(blocked)─► refusal ────────────────────────────────► END
               └─► retrieve ─► plan ─┬─► schema  ──────► learn ─► END
                                     ├─► converse ─────► learn ─► END
                                     ├─► resolve_del ─► confirm ─► apply ─► END
                                     ├─► restore ──────────────────────────► END
                                     └─► generate_sql ─► validate ─┬─► execute ─┐
                                                    ▲              └─► repair ──┘
                                                    │                    │
                                          advance ◄─┴────────────────────┘
                                             │
                                             ├─(more steps)─► generate_sql
                                             └─► synthesize ─► persist ─► learn ─► END

The SQL cycle is expressed as edges rather than a loop inside a node so that
each attempt is independently traced, the repair budget is inspectable state,
and a run can be resumed from a checkpoint mid-repair.
"""
from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .llm import TurnBudget
from .nodes import compose, learn, reports, sql, understand
from .services import Services
from .state import AgentState


def route_after_guard(state: AgentState) -> str:
    return "refused" if state.get("route") == "refused" else "continue"


def route_after_plan(state: AgentState) -> str:
    route = state.get("route", "analysis")
    if route == "schema":
        return "schema"
    if route == "converse":
        return "converse"
    if route == "delete_reports":
        return "delete"
    if route == "restore_reports":
        return "restore"
    if route == "analysis" and not (state.get("steps") or []):
        return "converse"
    return "analysis"


def route_after_resolve(state: AgentState) -> str:
    # Nothing matched, or the request was too vague to resolve - the resolver
    # already wrote the answer, so skip the confirmation gate entirely.
    preview = state.get("deletion_preview") or {}
    return "answer" if not preview.get("matched") else "confirm"


def build_graph(svc: Services, budget: Optional[TurnBudget] = None, checkpointer=None):
    budget = budget or TurnBudget()
    g = StateGraph(AgentState)

    g.add_node("guardrail", understand.make_guardrail_node(svc, budget))
    g.add_node("refusal", compose.make_refusal_node(svc))
    g.add_node("retrieve", understand.make_retrieve_node(svc))
    g.add_node("plan", understand.make_plan_node(svc, budget))

    g.add_node("generate_sql", sql.make_generate_sql_node(svc, budget))
    g.add_node("validate_sql", sql.make_validate_sql_node(svc))
    g.add_node("execute_sql", sql.make_execute_sql_node(svc, budget))
    g.add_node("repair_sql", sql.make_repair_sql_node(svc, budget))
    g.add_node("advance_step", sql.make_advance_step_node(svc))

    g.add_node("synthesize", compose.make_synthesize_node(svc, budget))
    g.add_node("persist_report", reports.make_persist_report_node(svc))
    g.add_node("schema_answer", compose.make_schema_node(svc, budget))
    g.add_node("converse", compose.make_converse_node(svc, budget))

    g.add_node("restore_reports", reports.make_restore_node(svc))
    g.add_node("resolve_deletion", reports.make_resolve_deletion_node(svc))
    g.add_node("confirm_deletion", reports.make_confirm_deletion_node(svc))
    g.add_node("apply_deletion", reports.make_apply_deletion_node(svc))

    g.add_node("learn", learn.make_learn_node(svc, budget))

    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", route_after_guard,
                            {"refused": "refusal", "continue": "retrieve"})
    g.add_edge("refusal", END)
    g.add_edge("retrieve", "plan")
    g.add_conditional_edges("plan", route_after_plan, {
        "analysis": "generate_sql",
        "schema": "schema_answer",
        "converse": "converse",
        "delete": "resolve_deletion",
        "restore": "restore_reports",
    })

    g.add_conditional_edges("generate_sql", sql.after_generate, {
        "validate": "validate_sql", "advance": "advance_step",
    })
    g.add_conditional_edges("validate_sql", sql.after_validate, {
        "execute": "execute_sql", "repair": "repair_sql", "advance": "advance_step",
    })
    g.add_conditional_edges("execute_sql", sql.after_execute, {
        "repair": "repair_sql", "advance": "advance_step",
    })
    g.add_edge("repair_sql", "validate_sql")
    g.add_conditional_edges("advance_step", sql.after_advance, {
        "next_step": "generate_sql", "synthesize": "synthesize",
    })

    g.add_edge("synthesize", "persist_report")
    g.add_edge("persist_report", "learn")
    g.add_edge("schema_answer", "learn")
    g.add_edge("converse", "learn")

    g.add_conditional_edges("resolve_deletion", route_after_resolve, {
        "confirm": "confirm_deletion", "answer": END,
    })
    g.add_edge("restore_reports", END)
    g.add_edge("confirm_deletion", "apply_deletion")
    g.add_edge("apply_deletion", END)
    g.add_edge("learn", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def mermaid() -> str:
    """Static rendering of the graph for the docs."""
    return """flowchart TD
    START([user message]) --> G[guardrail]
    G -->|blocked| RF[refusal] --> E([answer])
    G -->|allowed| R[retrieve precedents]
    R --> P[plan]
    P -->|schema| SC[schema answer] --> L[learn]
    P -->|converse| CV[converse] --> L
    P -->|delete_reports| RD[resolve deletion]
    P -->|restore_reports| RS[restore reports] --> E
    P -->|analysis| GS[generate SQL]
    GS --> V{validate SQL}
    V -->|ok| X[execute]
    V -->|rejected| RP[repair]
    X -->|error / empty| RP
    X -->|ok| AD[advance step]
    RP --> V
    AD -->|more steps| GS
    AD -->|done| SY[synthesize]
    SY --> PR[persist report] --> L --> E
    RD -->|no match| E
    RD -->|matched| CF[[confirm - graph interrupt]]
    CF --> AP[apply deletion] --> E"""
