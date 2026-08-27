"""Turn orchestration.

Owns the things that are per-turn rather than per-graph: the cost budget, the
trace, the conversation history and the human-in-the-loop resume protocol for
destructive operations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .config import setting
from .graph import build_graph
from .llm import TurnBudget
from .obs import metrics
from .obs.tracing import Trace
from .services import Services
from .state import new_state

ConfirmCallback = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class TurnResult:
    answer: str
    state: Dict[str, Any]
    trace: Optional[Trace]
    elapsed_ms: float
    ok: bool = True          # nothing escaped the session; the UI has something to show
    answered: bool = True    # an answer was actually composed - the SLI-relevant one
    error: str = ""

    @property
    def route(self) -> str:
        return self.state.get("route", "")

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return self.state.get("steps") or []

    def sql_used(self) -> List[str]:
        return [s.get("validated_sql") or s.get("sql", "") for s in self.steps if s.get("sql")]


@dataclass
class ChatSession:
    services: Services
    user_id: str
    persona_id: str = "exec_default"
    user_display_name: str = ""
    session_id: str = field(default_factory=lambda: f"sess_{uuid4().hex[:8]}")
    history: List[Dict[str, str]] = field(default_factory=list)
    turns: int = 0
    last_result: Optional[TurnResult] = None

    def __post_init__(self) -> None:
        self._budget = TurnBudget(
            max_llm_calls=int(setting("budget.max_llm_calls_per_turn", 14)),
            max_bytes_billed=int(setting("budget.max_bytes_billed_per_turn", 6_000_000_000)),
            wall_clock_s=float(setting("budget.turn_wall_clock_budget_s", 180)),
            synthesis_reserve_s=float(setting("budget.synthesis_reserve_s", 45)),
            synthesis_reserve_calls=int(setting("budget.synthesis_reserve_calls", 1)),
        )
        self._checkpointer = MemorySaver()
        self._graph = build_graph(self.services, self._budget, self._checkpointer)

    def set_persona(self, persona_id: str) -> None:
        self.persona_id = persona_id

    def ask(self, message: str, on_confirm: Optional[ConfirmCallback] = None) -> TurnResult:
        self.turns += 1
        started = time.perf_counter()
        self._budget.reset()
        metrics.incr("turns.total")

        trace = self.services.tracer.start_trace(self.session_id, self.user_id, message)
        state = new_state(
            user_query=message, user_id=self.user_id, session_id=self.session_id,
            trace_id=trace.trace_id, persona_id=self.persona_id,
            history=list(self.history), user_display_name=self.user_display_name,
        )
        config = {"configurable": {"thread_id": f"{self.session_id}:{self.turns}"},
                  "recursion_limit": 60}

        try:
            result: Dict[str, Any] = self._graph.invoke(state, config)

            # The graph suspends at the destructive-op gate. Nothing has been
            # mutated at this point; we resume only with an explicit decision.
            guard = 0
            while True:
                payload = self._pending_interrupt(result, config)
                if payload is None:
                    break
                guard += 1
                if guard > 5:
                    raise RuntimeError("confirmation loop did not converge")
                decision = (on_confirm or _deny)(payload)
                result = self._graph.invoke(Command(resume=decision), config)

            answer = result.get("answer") or "I wasn't able to produce an answer for that."
            route = result.get("route", "?")
            # A node can fail to compose an answer without raising: it catches the
            # error and hands back the retrieved figures. That is still a failed
            # turn. Counting it as answered makes answer_rate blind to exactly the
            # failure the manager experiences, and hides the turn from any query
            # for failures - so the flag is set at the failure site and read here.
            failed = bool(result.get("answer_failed"))
            if route == "refused":
                outcome = "refused"
            elif failed:
                outcome = "failed"
            else:
                outcome = "ok"
            self.services.tracer.end_trace(outcome)
            # A refusal is a correct outcome, but it is not an answered question -
            # counting it as both makes answer_rate unable to detect over-blocking.
            if outcome == "ok":
                metrics.incr("turns.answered", route=route)
            elif failed:
                metrics.incr("turns.failed", error="answer_composition")
            # `ok` stays "nothing escaped" - resilience depends on that contract.
            # Whether an answer was composed is a separate question.
            ok, answered, error = True, not failed, ""
        except Exception as exc:
            # Last line of defence: the CLI must never see a traceback.
            self.services.tracer.end_trace("error")
            metrics.incr("turns.failed", error=type(exc).__name__)
            result = dict(state)
            answer = (
                "Something went wrong on my side and I stopped rather than guess.\n\n"
                f"`{type(exc).__name__}: {str(exc)[:200]}`\n\n"
                f"Trace `{trace.trace_id}` has the details — run `/trace` to see where it broke."
            )
            ok, answered, error = False, False, f"{type(exc).__name__}: {exc}"

        elapsed = (time.perf_counter() - started) * 1000
        metrics.observe("turn.latency_ms", elapsed)
        metrics.observe("turn.llm_calls", self._budget.llm_calls)

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-24:]

        turn = TurnResult(answer=answer, state=result, trace=trace, elapsed_ms=elapsed,
                          ok=ok, answered=answered, error=error)
        self.last_result = turn
        return turn

    def _pending_interrupt(self, result: Dict[str, Any], config: Dict[str, Any]):
        """Return the pending interrupt payload, or None if the graph finished.

        LangGraph has surfaced interrupts two different ways across versions:
        as an "__interrupt__" key on the invoke result (0.4+), and only on the
        state snapshot (0.3.x). Both are handled so a dependency bump does not
        silently turn the confirmation gate into a no-op.
        """
        if isinstance(result, dict) and result.get("__interrupt__"):
            return _interrupt_payload(result)
        try:
            snapshot = self._graph.get_state(config)
        except Exception:
            return None
        for task in getattr(snapshot, "tasks", ()) or ():
            for pending in getattr(task, "interrupts", ()) or ():
                value = getattr(pending, "value", pending)
                return value if isinstance(value, dict) else {"kind": "unknown", "raw": str(value)}
        return None

    @property
    def budget(self) -> TurnBudget:
        return self._budget


def _interrupt_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    raw = result["__interrupt__"]
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    value = getattr(raw, "value", raw)
    return value if isinstance(value, dict) else {"kind": "unknown", "raw": str(value)}


def _deny(_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Default when no confirmation handler is wired: refuse."""
    return {"approved": False, "phrase": ""}
