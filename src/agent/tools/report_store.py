"""The Saved Reports library - the only mutable store the agent can touch.

Deletion is the one destructive capability in the system, so it is built as an
explicit two-phase protocol rather than a tool the model can simply call:

    phase 1  resolve   - turn natural language into a concrete, listed set of
                         report ids. Nothing is mutated. The user sees titles,
                         dates and match reasons.
    phase 2  confirm   - the user approves that exact set, addressed by a token
                         bound to the resolved ids. A token for a different set
                         is rejected.

Deletes are soft: rows are tombstoned with deleted_at/deleted_by and remain
restorable for a retention window, and every phase is appended to an audit log.
Ownership is enforced in the query, not in the prompt - a manager can only ever
resolve and delete their own reports.
"""
from __future__ import annotations

import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..config import setting
from ..memory.store import connect
from ..obs import metrics


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Report:
    report_id: str
    user_id: str
    session_id: str
    title: str
    body_md: str
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    sql_refs: List[str] = field(default_factory=list)
    trace_id: str = ""
    created_at: str = ""
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None

    @property
    def deleted(self) -> bool:
        return self.deleted_at is not None

    def short(self) -> str:
        return f"{self.report_id}  {self.created_at[:16].replace('T', ' ')}  {self.title}"


@dataclass
class DeletionPlan:
    """A resolved, immutable deletion target set awaiting confirmation."""

    token: str
    user_id: str
    reports: List[Report]
    criteria: Dict[str, Any]
    match_reasons: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    ttl_s: float = 600.0

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_s

    @property
    def count(self) -> int:
        return len(self.reports)

    @property
    def requires_phrase(self) -> bool:
        return self.count > int(setting("reports.bulk_delete_phrase_threshold", 3))

    @property
    def confirm_phrase(self) -> str:
        return f"delete {self.count}"

    def ids(self) -> List[str]:
        return [r.report_id for r in self.reports]


def _row_to_report(row) -> Report:
    return Report(
        report_id=row["report_id"], user_id=row["user_id"], session_id=row["session_id"],
        title=row["title"], body_md=row["body_md"],
        entities=json.loads(row["entities"] or "[]"), tags=json.loads(row["tags"] or "[]"),
        sql_refs=json.loads(row["sql_refs"] or "[]"), trace_id=row["trace_id"] or "",
        created_at=row["created_at"], deleted_at=row["deleted_at"], deleted_by=row["deleted_by"],
    )


class ReportStore:
    def __init__(self) -> None:
        self._plans: Dict[str, DeletionPlan] = {}

    # ---- create / read ---------------------------------------------------
    def create(
        self, *, user_id: str, session_id: str, title: str, body_md: str,
        entities: Optional[List[str]] = None, tags: Optional[List[str]] = None,
        sql_refs: Optional[List[str]] = None, trace_id: str = "",
    ) -> Report:
        report = Report(
            report_id=f"rpt_{uuid.uuid4().hex[:8]}", user_id=user_id, session_id=session_id,
            title=title.strip()[:200], body_md=body_md, entities=entities or [],
            tags=tags or [], sql_refs=sql_refs or [], trace_id=trace_id, created_at=_now(),
        )
        conn = connect()
        conn.execute(
            "INSERT INTO reports (report_id,user_id,session_id,title,body_md,entities,tags,"
            "sql_refs,trace_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (report.report_id, user_id, session_id, report.title, body_md,
             json.dumps(report.entities), json.dumps(report.tags),
             json.dumps(report.sql_refs), trace_id, report.created_at),
        )
        conn.commit()
        self._audit(user_id, "report.create", [report.report_id],
                    {"title": report.title, "session_id": session_id}, trace_id)
        metrics.incr("reports.created")
        return report

    def get(self, report_id: str, user_id: Optional[str] = None) -> Optional[Report]:
        sql = "SELECT * FROM reports WHERE report_id = ?"
        params: List[Any] = [report_id]
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        row = connect().execute(sql, params).fetchone()
        return _row_to_report(row) if row else None

    def list(self, user_id: str, include_deleted: bool = False, limit: int = 100) -> List[Report]:
        sql = "SELECT * FROM reports WHERE user_id = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        sql += " ORDER BY created_at DESC LIMIT ?"
        rows = connect().execute(sql, (user_id, limit)).fetchall()
        return [_row_to_report(r) for r in rows]

    # ---- phase 1: resolve ------------------------------------------------
    def resolve_deletion(
        self, *, user_id: str, criteria: Dict[str, Any],
    ) -> DeletionPlan:
        """Turn structured criteria into a concrete target set. Read-only.

        Supported criteria:
          mentions      list[str] - substring match on title/body/entities
          session_id    str       - "the reports from this conversation"
          report_ids    list[str] - explicit ids
          created_after / created_before  ISO date
          all           bool      - every report owned by the user
        """
        reports = self.list(user_id, include_deleted=False, limit=1000)
        selected, reasons = self._select(reports, criteria)
        plan = DeletionPlan(
            token=secrets.token_hex(3), user_id=user_id, reports=selected,
            criteria=criteria, match_reasons=reasons,
        )
        self._plans[plan.token] = plan
        metrics.incr("reports.delete_requested", matched=len(selected))
        self._audit(user_id, "report.delete_requested", plan.ids(),
                    {"criteria": criteria, "token": plan.token, "matched": len(selected)})
        return plan

    @staticmethod
    def _select(
        reports: List[Report], criteria: Dict[str, Any],
    ) -> Tuple[List[Report], Dict[str, str]]:
        """Match reports against criteria. Pure, and shared by delete and restore."""
        reasons: Dict[str, str] = {}
        selected: List[Report] = []

        mentions = [m.strip().lower() for m in (criteria.get("mentions") or []) if m and m.strip()]
        wanted_ids = set(criteria.get("report_ids") or [])
        session_id = criteria.get("session_id")
        after = criteria.get("created_after")
        before = criteria.get("created_before")
        # "all" only means all when nothing narrower was asked for. If the
        # request also named entities, a session or explicit ids, the narrower
        # criterion wins - the blast radius of misreading this is the whole
        # library, so it resolves towards the smaller set.
        take_all = bool(criteria.get("all")) and not (mentions or wanted_ids or session_id)

        for report in reports:
            why: List[str] = []
            if take_all:
                why.append("all reports requested")
            if wanted_ids and report.report_id in wanted_ids:
                why.append("id match")
            if session_id and report.session_id == session_id:
                why.append("created in this conversation")
            if mentions:
                haystack = " ".join(
                    [report.title, report.body_md, " ".join(report.entities), " ".join(report.tags)]
                ).lower()
                hit = [m for m in mentions if m in haystack]
                if hit:
                    why.append(f"mentions {', '.join(sorted(set(hit)))}")
                elif not (take_all or wanted_ids or session_id):
                    continue
            if after and report.created_at < str(after):
                continue
            if before and report.created_at > str(before):
                continue
            # A criteria set with no positive matcher selects nothing - this is
            # what stops an under-specified request from becoming "delete all".
            if not why:
                continue
            selected.append(report)
            reasons[report.report_id] = "; ".join(why)

        return selected, reasons

    def resolve_restore(
        self, *, user_id: str, criteria: Dict[str, Any],
    ) -> Tuple[List[Report], Dict[str, str]]:
        """Find soft-deleted reports to bring back. Read-only.

        Restore is the inverse of deletion and reuses its matcher, with one
        addition: an unqualified "undo that delete" resolves to the most recent
        delete batch rather than to nothing. Deletion resolves an unqualified
        request towards the empty set because the blast radius is the library;
        restore can resolve towards the last batch because the worst case is a
        report the manager already asked to keep.
        """
        deleted = [r for r in self.list(user_id, include_deleted=True, limit=1000) if r.deleted]
        if not deleted:
            return [], {}

        selected, reasons = self._select(deleted, criteria)
        if selected:
            return selected, reasons

        has_matcher = bool(
            criteria.get("mentions") or criteria.get("report_ids")
            or criteria.get("session_id") or criteria.get("all")
        )
        if has_matcher:
            return [], {}

        # "The last delete" is an event, not a timestamp. Grouping by deleted_at
        # merges two deletions that happen in the same second, so the last
        # confirmed deletion is read from the audit log - which is the actual
        # record of what happened, and is written on the same transaction.
        still_deleted = {r.report_id: r for r in deleted}
        for entry in self.audit_trail(user_id, limit=200):
            if entry["action"] != "report.delete_confirmed":
                continue
            batch = [still_deleted[i] for i in entry["targets"] if i in still_deleted]
            if batch:
                return batch, {r.report_id: "from the most recent deletion" for r in batch}
            # That batch was already restored; keep walking back.
        return [], {}

    def restore_reports(
        self, reports: List[Report], user_id: str, trace_id: str = "",
    ) -> Tuple[int, List[str]]:
        ids = [r.report_id for r in reports]
        count = self.restore(ids, user_id)
        metrics.incr("reports.restore_confirmed", count=count)
        self._audit(user_id, "report.restore_confirmed", ids,
                    {"restored": count}, trace_id)
        return count, ids

    def get_plan(self, token: str) -> Optional[DeletionPlan]:
        plan = self._plans.get(token)
        if plan and plan.expired:
            self._plans.pop(token, None)
            return None
        return plan

    def cancel_plan(self, token: str, user_id: str = "") -> None:
        plan = self._plans.pop(token, None)
        if plan:
            metrics.incr("reports.delete_cancelled")
            self._audit(user_id or plan.user_id, "report.delete_cancelled", plan.ids(),
                        {"token": token})

    # ---- phase 2: execute ------------------------------------------------
    def confirm_deletion(self, token: str, user_id: str, trace_id: str = "") -> Tuple[int, List[str]]:
        plan = self.get_plan(token)
        if plan is None:
            raise KeyError("That confirmation has expired or was already used.")
        if plan.user_id != user_id:
            raise PermissionError("You can only delete your own reports.")

        ids = plan.ids()
        if not ids:
            self._plans.pop(token, None)
            return 0, []

        conn = connect()
        placeholders = ",".join("?" * len(ids))
        # Ownership is re-checked here, not only at resolve time.
        cur = conn.execute(
            f"UPDATE reports SET deleted_at = ?, deleted_by = ? "
            f"WHERE report_id IN ({placeholders}) AND user_id = ? AND deleted_at IS NULL",
            [_now(), user_id, *ids, user_id],
        )
        conn.commit()
        self._plans.pop(token, None)
        metrics.incr("reports.delete_confirmed", count=cur.rowcount)
        self._audit(user_id, "report.delete_confirmed", ids,
                    {"token": token, "deleted": cur.rowcount, "soft": True}, trace_id)
        return cur.rowcount, ids

    def restore(self, report_ids: List[str], user_id: str) -> int:
        if not report_ids:
            return 0
        conn = connect()
        placeholders = ",".join("?" * len(report_ids))
        cur = conn.execute(
            f"UPDATE reports SET deleted_at = NULL, deleted_by = NULL "
            f"WHERE report_id IN ({placeholders}) AND user_id = ?",
            [*report_ids, user_id],
        )
        conn.commit()
        self._audit(user_id, "report.restore", report_ids, {"restored": cur.rowcount})
        return cur.rowcount

    def purge_expired(self) -> int:
        """Hard-delete tombstones past the retention window."""
        days = int(setting("reports.soft_delete_retention_days", 30))
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        conn = connect()
        cur = conn.execute("DELETE FROM reports WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,))
        conn.commit()
        return cur.rowcount

    # ---- audit -----------------------------------------------------------
    def _audit(self, user_id: str, action: str, targets: List[str],
               detail: Dict[str, Any], trace_id: str = "") -> None:
        conn = connect()
        conn.execute(
            "INSERT INTO audit_log (ts,user_id,action,targets,detail,trace_id) VALUES (?,?,?,?,?,?)",
            (_now(), user_id, action, json.dumps(targets), json.dumps(detail, default=str), trace_id),
        )
        conn.commit()

    def audit_trail(self, user_id: Optional[str] = None, limit: int = 40) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM audit_log"
        params: List[Any] = []
        if user_id:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [
            {"ts": r["ts"], "user_id": r["user_id"], "action": r["action"],
             "targets": json.loads(r["targets"]), "detail": json.loads(r["detail"]),
             "trace_id": r["trace_id"]}
            for r in connect().execute(sql, params).fetchall()
        ]
