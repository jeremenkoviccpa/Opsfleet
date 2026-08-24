"""Graph state.

One dict flows through every node. Everything a node needs to make a decision
is on it, and everything a node learns is written back to it - which is what
makes any turn fully reconstructible from its trace.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class StepState(TypedDict, total=False):
    step_id: str
    goal: str
    sql: str
    validated_sql: str
    attempts: int
    repairs: int
    status: str              # pending | ok | empty | failed | rejected
    error: str
    error_kind: str
    rows: int
    columns: List[str]
    preview_md: str
    records: List[Dict[str, Any]]
    mask_note: str
    bytes_billed: int
    latency_ms: float
    column_actions: Dict[str, Any]


class AgentState(TypedDict, total=False):
    # --- identity -------------------------------------------------------
    trace_id: str
    session_id: str
    user_id: str
    user_display_name: str
    persona_id: str

    # --- input ----------------------------------------------------------
    user_query: str
    history: List[Dict[str, str]]

    # --- guardrail ------------------------------------------------------
    guard_decision: str
    guard_reasons: List[str]
    guard_stage: str

    # --- retrieval ------------------------------------------------------
    precedents: List[Dict[str, Any]]
    precedent_block: str

    # --- planning -------------------------------------------------------
    route: str               # analysis | schema | delete_reports | save_report | converse | refused
    intent_reason: str
    time_window: str
    plan_notes: str
    steps: List[StepState]
    current_step: int

    # --- sql loop -------------------------------------------------------
    repair_budget_left: int
    last_error: str
    last_error_kind: str

    # --- destructive ops ------------------------------------------------
    deletion_token: str
    deletion_preview: Dict[str, Any]
    deletion_result: Dict[str, Any]

    # --- output ---------------------------------------------------------
    answer: str
    answer_is_report: bool
    saved_report: Dict[str, Any]
    degraded: bool
    degraded_reason: str
    warnings: List[str]
    errors: List[str]
    learned: List[str]


def new_state(
    *, user_query: str, user_id: str, session_id: str, trace_id: str,
    persona_id: str, history: Optional[List[Dict[str, str]]] = None,
    user_display_name: str = "",
) -> AgentState:
    return AgentState(
        trace_id=trace_id, session_id=session_id, user_id=user_id,
        user_display_name=user_display_name or user_id, persona_id=persona_id,
        user_query=user_query, history=history or [],
        guard_decision="allow", guard_reasons=[], guard_stage="",
        precedents=[], precedent_block="", route="", intent_reason="",
        time_window="", plan_notes="", steps=[], current_step=0,
        repair_budget_left=0, last_error="", last_error_kind="",
        answer="", answer_is_report=False, degraded=False, degraded_reason="",
        warnings=[], errors=[], learned=[],
    )
