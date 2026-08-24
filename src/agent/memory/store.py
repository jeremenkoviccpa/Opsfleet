"""SQLite connection management for agent state.

One file holds user preferences, the saved-reports library and the audit log.
In production these are three Firestore collections; the access patterns here
(point read by user, filtered list, append-only audit) map onto it directly.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

_LOCAL = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_prefs (
    user_id        TEXT NOT NULL,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'inferred',   -- explicit | inferred
    confidence     REAL NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_evidence  TEXT,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS reports (
    report_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    title       TEXT NOT NULL,
    body_md     TEXT NOT NULL,
    entities    TEXT NOT NULL DEFAULT '[]',
    tags        TEXT NOT NULL DEFAULT '[]',
    sql_refs    TEXT NOT NULL DEFAULT '[]',
    trace_id    TEXT,
    created_at  TEXT NOT NULL,
    deleted_at  TEXT,
    deleted_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_user    ON reports(user_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_reports_session ON reports(session_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    user_id   TEXT NOT NULL,
    action    TEXT NOT NULL,
    targets   TEXT NOT NULL DEFAULT '[]',
    detail    TEXT NOT NULL DEFAULT '{}',
    trace_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id, ts);

CREATE TABLE IF NOT EXISTS turn_feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trace_id   TEXT,
    rating     TEXT NOT NULL,          -- up | down
    note       TEXT,
    question   TEXT,
    answer     TEXT
);
"""

_DB_PATH: Optional[Path] = None


def configure(path: Path) -> None:
    global _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    _DB_PATH = path
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()


def connect() -> sqlite3.Connection:
    """Thread-local connection, keyed by the configured path.

    The path is part of the cache key: reconfiguring the store at runtime (the
    CLI switching profiles, a test pointing at a temp directory) must not keep
    handing back a connection to the previous database.
    """
    if _DB_PATH is None:
        raise RuntimeError("memory.store.configure() must be called before connect()")
    conn = getattr(_LOCAL, "conn", None)
    if conn is not None and getattr(_LOCAL, "path", None) == str(_DB_PATH):
        return conn
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _LOCAL.conn = conn
    _LOCAL.path = str(_DB_PATH)
    return conn
