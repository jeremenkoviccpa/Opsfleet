"""Agent-level metrics.

Counters and histograms are aggregated in-process and appended to
`.runtime/metrics.jsonl`. In production the same call sites emit to Cloud
Monitoring; the names below are the alertable SLIs described in docs/HLD.md.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_COUNTERS: Dict[str, float] = defaultdict(float)
_HISTOGRAMS: Dict[str, List[float]] = defaultdict(list)
_SINK: Optional[Path] = None

# SLIs the on-call dashboard is built from.
COUNTERS = [
    "turns.total",
    "turns.answered",
    "turns.refused",
    "turns.failed",
    "guardrail.blocked",
    "guardrail.injection_detected",
    "sql.generated",
    "sql.validation_rejected",
    "sql.executed",
    "sql.failed",
    "sql.repaired",
    "sql.empty_result",
    "sql.gave_up",
    "pii.columns_blocked",
    "pii.values_redacted",
    "llm.calls",
    "llm.provider_failover",
    "llm.circuit_open",
    "golden.hits",
    "golden.miss",
    "reports.created",
    "reports.delete_requested",
    "reports.delete_confirmed",
    "reports.delete_cancelled",
    "prefs.learned",
]

HISTOGRAMS = [
    "turn.latency_ms",
    "llm.latency_ms",
    "sql.latency_ms",
    "sql.bytes_billed",
    "turn.llm_calls",
    "golden.top_score",
]


def configure(sink: Path) -> None:
    global _SINK
    sink.parent.mkdir(parents=True, exist_ok=True)
    _SINK = sink


def incr(name: str, value: float = 1.0, **labels: Any) -> None:
    with _LOCK:
        _COUNTERS[name] += value
    _emit({"type": "counter", "name": name, "value": value, "ts": time.time(), **labels})


def observe(name: str, value: float, **labels: Any) -> None:
    with _LOCK:
        _HISTOGRAMS[name].append(value)
    _emit({"type": "histogram", "name": name, "value": value, "ts": time.time(), **labels})


def _emit(payload: Dict[str, Any]) -> None:
    if _SINK is None:
        return
    try:
        with _SINK.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


def snapshot() -> Dict[str, Any]:
    with _LOCK:
        counters = dict(_COUNTERS)
        hists = {}
        for name, vals in _HISTOGRAMS.items():
            if not vals:
                continue
            ordered = sorted(vals)
            hists[name] = {
                "count": len(vals),
                "mean": round(statistics.fmean(vals), 2),
                "p50": round(ordered[len(ordered) // 2], 2),
                "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
                "max": round(ordered[-1], 2),
            }
    return {"counters": counters, "histograms": hists}


def derived() -> Dict[str, float]:
    """Ratios the dashboard actually alerts on."""
    c = _COUNTERS
    def ratio(num: str, den: str) -> float:
        d = c.get(den, 0.0)
        return round(c.get(num, 0.0) / d, 4) if d else 0.0
    return {
        "answer_rate": ratio("turns.answered", "turns.total"),
        "failure_rate": ratio("turns.failed", "turns.total"),
        "sql_first_pass_rate": round(
            (c.get("sql.executed", 0.0) - c.get("sql.repaired", 0.0)) / c["sql.executed"], 4
        ) if c.get("sql.executed") else 0.0,
        "sql_giveup_rate": ratio("sql.gave_up", "sql.generated"),
        "guardrail_block_rate": ratio("guardrail.blocked", "turns.total"),
        "golden_hit_rate": ratio("golden.hits", "turns.total"),
        "provider_failover_rate": ratio("llm.provider_failover", "llm.calls"),
    }


def reset() -> None:
    with _LOCK:
        _COUNTERS.clear()
        _HISTOGRAMS.clear()
