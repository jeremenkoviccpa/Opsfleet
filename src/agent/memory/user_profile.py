"""Per-user preference memory (the user-level learning loop).

Preferences carry a source and a confidence, and only reach the prompt once
they clear a threshold:

  explicit  ("always give me tables")  -> confidence 1.0, applied immediately
  inferred  ("that was too long")      -> confidence grows with repetition,
                                          0.45 -> 0.60 -> 0.75 -> 0.85 -> 0.92

That ramp is the point. Acting on a single ambiguous signal produces an agent
that thrashes between formats; requiring corroboration produces one that
converges. Every stored preference keeps the utterance that caused it, so a
user can ask "why are you doing that?" and get a real answer - and so a wrong
inference can be traced and dropped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import setting
from ..obs import metrics
from .store import connect

# The vocabulary the agent is allowed to learn. A closed set keeps an extraction
# model from inventing preference keys that no prompt ever reads.
PREF_KEYS = {
    "output_format": ["table", "bullets", "prose", "mixed"],
    "analysis_depth": ["headline", "standard", "deep"],
    "wants_charts": ["yes", "no"],
    "wants_action_items": ["always", "on_request", "never"],
    "number_style": ["rounded", "precise"],
    "default_time_window": None,     # free text, e.g. "last 90 days"
    "focus_metrics": None,           # free text, e.g. "margin over revenue"
    "preferred_comparison": None,    # free text, e.g. "always vs last year"
}

CONFIDENCE_RAMP = [0.45, 0.60, 0.75, 0.85, 0.92, 0.95]


@dataclass
class Preference:
    key: str
    value: str
    source: str
    confidence: float
    evidence_count: int
    last_evidence: str
    updated_at: str

    @property
    def active(self) -> bool:
        threshold = float(setting("learning.inferred_pref_min_confidence", 0.6))
        return self.source == "explicit" or self.confidence >= threshold


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def get_preferences(user_id: str, active_only: bool = False) -> List[Preference]:
    rows = connect().execute(
        "SELECT * FROM user_prefs WHERE user_id = ? ORDER BY key", (user_id,)
    ).fetchall()
    prefs = [
        Preference(r["key"], r["value"], r["source"], r["confidence"],
                   r["evidence_count"], r["last_evidence"] or "", r["updated_at"])
        for r in rows
    ]
    return [p for p in prefs if p.active] if active_only else prefs


def record(
    user_id: str,
    key: str,
    value: str,
    *,
    source: str = "inferred",
    evidence: str = "",
) -> Optional[Preference]:
    """Insert or reinforce a preference. Returns the resulting preference."""
    if key not in PREF_KEYS:
        return None
    allowed = PREF_KEYS[key]
    if allowed and value not in allowed:
        return None

    conn = connect()
    row = conn.execute(
        "SELECT * FROM user_prefs WHERE user_id = ? AND key = ?", (user_id, key)
    ).fetchone()

    if row is None:
        confidence = 1.0 if source == "explicit" else CONFIDENCE_RAMP[0]
        conn.execute(
            "INSERT INTO user_prefs (user_id, key, value, source, confidence, "
            "evidence_count, last_evidence, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, key, value, source, confidence, 1, evidence[:400], _now()),
        )
        conn.commit()
        metrics.incr("prefs.learned", key=key, source=source)
        return Preference(key, value, source, confidence, 1, evidence, _now())

    if source == "explicit":
        conn.execute(
            "UPDATE user_prefs SET value=?, source='explicit', confidence=1.0, "
            "evidence_count=evidence_count+1, last_evidence=?, updated_at=? "
            "WHERE user_id=? AND key=?",
            (value, evidence[:400], _now(), user_id, key),
        )
        conn.commit()
        metrics.incr("prefs.learned", key=key, source="explicit")
        return Preference(key, value, "explicit", 1.0, row["evidence_count"] + 1, evidence, _now())

    if row["source"] == "explicit" and row["value"] != value:
        # An explicit instruction outranks a contradicting inference. Record the
        # observation but do not override what the user actually said.
        return Preference(key, row["value"], "explicit", 1.0, row["evidence_count"],
                          row["last_evidence"] or "", row["updated_at"])

    if row["value"] == value:
        count = row["evidence_count"] + 1
        confidence = CONFIDENCE_RAMP[min(count - 1, len(CONFIDENCE_RAMP) - 1)]
    else:
        # Contradicting inference: decay towards the new value instead of flipping.
        count = 1
        confidence = CONFIDENCE_RAMP[0]
    conn.execute(
        "UPDATE user_prefs SET value=?, source='inferred', confidence=?, evidence_count=?, "
        "last_evidence=?, updated_at=? WHERE user_id=? AND key=?",
        (value, confidence, count, evidence[:400], _now(), user_id, key),
    )
    conn.commit()
    metrics.incr("prefs.learned", key=key, source="inferred")
    return Preference(key, value, "inferred", confidence, count, evidence, _now())


def forget(user_id: str, key: Optional[str] = None) -> int:
    conn = connect()
    if key:
        cur = conn.execute("DELETE FROM user_prefs WHERE user_id=? AND key=?", (user_id, key))
    else:
        cur = conn.execute("DELETE FROM user_prefs WHERE user_id=?", (user_id,))
    conn.commit()
    return cur.rowcount


def render_for_prompt(user_id: str) -> str:
    """The block injected into the answer-composition prompt."""
    active = get_preferences(user_id, active_only=True)
    if not active:
        return "No learned preferences yet for this manager - use the persona defaults."
    lines = []
    for p in active:
        marker = "stated" if p.source == "explicit" else f"observed x{p.evidence_count}"
        lines.append(f"- {p.key}: {p.value} ({marker})")
    return "This manager's learned preferences - honour them over the persona defaults:\n" + "\n".join(lines)


def record_feedback(
    user_id: str, session_id: str, trace_id: str, rating: str,
    note: str = "", question: str = "", answer: str = "",
) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO turn_feedback (ts,user_id,session_id,trace_id,rating,note,question,answer) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (_now(), user_id, session_id, trace_id, rating, note, question, answer[:4000]),
    )
    conn.commit()


def feedback_stats() -> Dict[str, Any]:
    rows = connect().execute(
        "SELECT rating, COUNT(*) n FROM turn_feedback GROUP BY rating"
    ).fetchall()
    return {r["rating"]: r["n"] for r in rows}
