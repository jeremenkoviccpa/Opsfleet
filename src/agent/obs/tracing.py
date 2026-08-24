"""Structured tracing.

Every turn opens one trace. Every node, LLM call, SQL execution and guardrail
decision opens a span inside it. Spans are written to
`.runtime/traces/<trace_id>.jsonl` as they close, and kept in an in-process ring
buffer so the CLI can render `/trace` without touching disk.

The span schema is deliberately OpenTelemetry-shaped (trace_id / span_id /
parent_span_id / attributes / status) so the production build can swap the JSONL
sink for an OTLP exporter to Cloud Trace without changing any call site.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional

_local = threading.local()
_RING: Deque["Trace"] = deque(maxlen=25)
_RING_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    kind: str
    parent_span_id: Optional[str]
    start_ts: float
    end_ts: Optional[float] = None
    status: str = "ok"
    error: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return ((self.end_ts or _now()) - self.start_ts) * 1000.0

    def set(self, **attrs: Any) -> "Span":
        self.attributes.update({k: v for k, v in attrs.items() if v is not None})
        return self

    def event(self, name: str, **attrs: Any) -> "Span":
        self.events.append({"name": name, "ts": _now(), **attrs})
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class Trace:
    trace_id: str
    session_id: str
    user_id: str
    user_query: str
    start_ts: float
    end_ts: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    sink_path: Optional[Path] = None
    outcome: str = "in_progress"

    @property
    def duration_ms(self) -> float:
        return ((self.end_ts or _now()) - self.start_ts) * 1000.0

    def summary(self) -> Dict[str, Any]:
        llm = [s for s in self.spans if s.kind == "llm"]
        sql = [s for s in self.spans if s.kind == "sql"]
        return {
            "trace_id": self.trace_id,
            "outcome": self.outcome,
            "duration_ms": round(self.duration_ms, 1),
            "spans": len(self.spans),
            "llm_calls": len(llm),
            "llm_tokens_in": sum(s.attributes.get("tokens_in", 0) for s in llm),
            "llm_tokens_out": sum(s.attributes.get("tokens_out", 0) for s in llm),
            "sql_executions": len(sql),
            "sql_repairs": sum(1 for s in sql if s.attributes.get("is_repair")),
            "bytes_billed": sum(s.attributes.get("bytes_billed", 0) for s in sql),
            "errors": [s.name for s in self.spans if s.status == "error"],
        }


class Tracer:
    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def start_trace(self, session_id: str, user_id: str, user_query: str) -> Trace:
        trace_id = uuid.uuid4().hex[:16]
        trace = Trace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            user_query=user_query,
            start_ts=_now(),
            sink_path=self.trace_dir / f"{trace_id}.jsonl",
        )
        _local.trace = trace
        _local.stack = []
        with _RING_LOCK:
            _RING.append(trace)
        self._write(trace, {"type": "trace_start", "trace_id": trace_id,
                            "session_id": session_id, "user_id": user_id,
                            "user_query": user_query, "ts": trace.start_ts})
        return trace

    def end_trace(self, outcome: str = "ok") -> Optional[Trace]:
        trace = current_trace()
        if trace is None:
            return None
        trace.end_ts = _now()
        trace.outcome = outcome
        self._write(trace, {"type": "trace_end", **trace.summary()})
        return trace

    @staticmethod
    def _write(trace: Trace, payload: Dict[str, Any]) -> None:
        if trace.sink_path is None:
            return
        try:
            with trace.sink_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except OSError:
            # Telemetry must never take the agent down.
            pass

    @contextmanager
    def span(self, name: str, kind: str = "node", **attrs: Any) -> Iterator[Span]:
        trace = current_trace()
        if trace is None:
            yield Span("noop", "noop", name, kind, None, _now())
            return
        stack: List[Span] = getattr(_local, "stack", [])
        span = Span(
            span_id=uuid.uuid4().hex[:12],
            trace_id=trace.trace_id,
            name=name,
            kind=kind,
            parent_span_id=stack[-1].span_id if stack else None,
            start_ts=_now(),
            attributes=dict(attrs),
        )
        trace.spans.append(span)
        stack.append(span)
        _local.stack = stack
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_ts = _now()
            stack.pop()
            self._write(trace, {"type": "span", **span.to_dict()})


def current_trace() -> Optional[Trace]:
    return getattr(_local, "trace", None)


def recent_traces(n: int = 10) -> List[Trace]:
    with _RING_LOCK:
        return list(_RING)[-n:]


def find_trace(trace_id: str) -> Optional[Trace]:
    with _RING_LOCK:
        for t in reversed(_RING):
            if t.trace_id.startswith(trace_id):
                return t
    return None
