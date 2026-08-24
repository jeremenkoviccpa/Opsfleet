"""SQL validation and rewriting.

The database connection is read-only, but that is an infrastructure control -
it protects the data, not the business. This layer protects everything else:
it is a deterministic, non-LLM gate that every generated statement passes
through before it reaches a warehouse.

Checks, in order:
  1. parses as exactly one statement in the BigQuery dialect;
  2. the statement is a SELECT (or a WITH ending in a SELECT);
  3. no DML/DDL/scripting node anywhere in the tree, at any nesting depth;
  4. every referenced table is on the allowlist;
  5. no `deny` PII column appears in any projection, at any nesting depth;
  6. no bare `SELECT *` against a table holding personal data;
  7. a LIMIT is present on the outermost SELECT (injected if missing).

It also returns the mapping from each output column to the PII action that must
be applied post-execution, which is what drives L2 masking in safety/pii.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from ..config import setting
from .pii import guard as pii_guard

FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Merge, exp.TruncateTable, exp.Grant, exp.Command,
)
FORBIDDEN_FUNCTIONS = {
    "external_query", "session_user", "execute_immediate", "net.host",
    "ml.predict", "ml.generate_text", "bqml", "js",
}
PII_BEARING_TABLES = {"users"}

# Aggregates that collapse a column into a STATISTIC. COUNT(users.id) is a
# customer count, not a customer - pseudonymising it would replace the number
# with `cust_a41f9c` and destroy the answer.
STATISTIC_AGGREGATES = (
    exp.Count, exp.Sum, exp.Avg, exp.Stddev, exp.StddevPop, exp.StddevSamp,
    exp.Variance, exp.VariancePop, exp.ApproxDistinct,
)
# Aggregates that return an ELEMENT of the column. MIN(users.email) is still an
# email address, so these stay masked.
ELEMENT_AGGREGATES = (
    exp.Min, exp.Max, exp.AnyValue, exp.ArrayAgg, exp.GroupConcat,
)


def _inside_statistic_aggregate(col: exp.Column, projection: exp.Expression) -> bool:
    """True when the column only reaches the output through COUNT/SUM/AVG/…"""
    node = col.parent
    while node is not None:
        if isinstance(node, ELEMENT_AGGREGATES):
            return False
        if isinstance(node, STATISTIC_AGGREGATES):
            return True
        if node is projection:
            break
        node = node.parent
    return False


@dataclass
class SQLVerdict:
    ok: bool
    sql: str
    original_sql: str = ""
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    column_actions: Dict[str, Tuple[str, Dict[str, Any]]] = field(default_factory=dict)
    rewritten: bool = False

    @property
    def reason(self) -> str:
        return "; ".join(self.violations)


def _base_table(node: exp.Table) -> str:
    return (node.name or "").lower()


def _alias_map(tree: exp.Expression) -> Dict[str, str]:
    """alias -> physical table name, plus CTE names mapped to themselves."""
    mapping: Dict[str, str] = {}
    for cte in tree.find_all(exp.CTE):
        mapping[cte.alias_or_name.lower()] = f"__cte__{cte.alias_or_name.lower()}"
    for tbl in tree.find_all(exp.Table):
        name = _base_table(tbl)
        if not name:
            continue
        mapping.setdefault(name, name)
        alias = tbl.alias_or_name.lower()
        if alias:
            mapping[alias] = mapping.get(name, name) if not alias.startswith("__cte__") else alias
            if alias != name:
                mapping[alias] = name
    return mapping


def _resolve_table(col: exp.Column, aliases: Dict[str, str], candidates: List[str]) -> Optional[str]:
    qualifier = (col.table or "").lower()
    if qualifier:
        resolved = aliases.get(qualifier, qualifier)
        return None if resolved.startswith("__cte__") else resolved
    # Unqualified: if exactly one physical table is in play, it belongs to it.
    physical = [t for t in candidates if not t.startswith("__cte__")]
    return physical[0] if len(physical) == 1 else None


def validate(
    sql: str,
    *,
    allowed_tables: Optional[List[str]] = None,
    force_limit: Optional[int] = None,
) -> SQLVerdict:
    allowed = {t.lower() for t in (allowed_tables or setting("safety.allowed_tables", []))}
    force_limit = force_limit or int(setting("safety.force_limit", 5000))
    guard = pii_guard()
    verdict = SQLVerdict(ok=False, sql=sql, original_sql=sql)

    sql = (sql or "").strip().rstrip(";")
    if not sql:
        verdict.violations.append("empty statement")
        return verdict

    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except Exception as exc:
        verdict.violations.append(f"does not parse as BigQuery SQL: {exc}")
        return verdict

    if len(statements) != 1:
        verdict.violations.append(
            f"expected exactly 1 statement, found {len(statements)} - statement batching is not allowed"
        )
        return verdict

    tree = statements[0]

    # (2) must be a read
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)) and not (
        isinstance(tree, exp.Query)
    ):
        verdict.violations.append(f"only SELECT statements are permitted, got {type(tree).__name__.upper()}")
        return verdict

    # (3) no write nodes anywhere
    for node_type in FORBIDDEN_NODES:
        found = list(tree.find_all(node_type))
        if found:
            verdict.violations.append(
                f"statement contains a {node_type.__name__.upper()} node - writes are not permitted"
            )
    for fn in tree.find_all(exp.Anonymous):
        if (fn.name or "").lower() in FORBIDDEN_FUNCTIONS:
            verdict.violations.append(f"function '{fn.name}' is not permitted")
    lowered = sql.lower()
    for banned in ("execute immediate", "create temp function", "create temporary function"):
        if banned in lowered:
            verdict.violations.append(f"'{banned}' is not permitted")

    # (4) table allowlist
    aliases = _alias_map(tree)
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    referenced: List[str] = []
    for tbl in tree.find_all(exp.Table):
        name = _base_table(tbl)
        if not name or name in cte_names:
            continue
        referenced.append(name)
        if allowed and name not in allowed:
            verdict.violations.append(
                f"table '{name}' is not on the allowlist ({', '.join(sorted(allowed))})"
            )
    verdict.tables = sorted(set(referenced))

    # (5) + (6) PII in projections, at every nesting level
    physical_in_scope = sorted({t for t in referenced})
    for select in tree.find_all(exp.Select):
        local_aliases = _alias_map(select) or aliases
        local_tables = sorted({_base_table(t) for t in select.find_all(exp.Table) if _base_table(t)})
        scope = local_tables or physical_in_scope
        for projection in select.expressions:
            if isinstance(projection, exp.Star):
                if set(scope) & PII_BEARING_TABLES:
                    verdict.violations.append(
                        "SELECT * is not permitted against `users`; list the columns you need"
                    )
                continue
            out_name = projection.alias_or_name
            for col in projection.find_all(exp.Column):
                if isinstance(col.this, exp.Star):
                    if set(scope) & PII_BEARING_TABLES:
                        verdict.violations.append("SELECT <table>.* is not permitted against `users`")
                    continue
                table = _resolve_table(col, local_aliases, scope)
                if table is None:
                    continue
                action, meta = guard.action_for(table, col.name)
                if action == "deny":
                    # An aggregate over a denied column is a statistic, not the
                    # personal data itself: COUNT(users.last_name) leaks nothing.
                    if _inside_statistic_aggregate(col, projection):
                        continue
                    verdict.violations.append(
                        f"column `{table}.{col.name}` is classified PII and may not be selected"
                    )
                elif action in ("hash", "generalize") and out_name:
                    if _inside_statistic_aggregate(col, projection):
                        continue
                    verdict.column_actions[out_name] = (action, meta)

    # deny columns used as filters are legal but audited
    where_cols = []
    for select in tree.find_all(exp.Select):
        for clause in (select.args.get("where"), select.args.get("having")):
            if clause is None:
                continue
            for col in clause.find_all(exp.Column):
                table = _resolve_table(col, aliases, physical_in_scope)
                if table and guard.action_for(table, col.name)[0] == "deny":
                    where_cols.append(f"{table}.{col.name}")
    if where_cols:
        verdict.warnings.append(f"PII columns used as filters (audited): {', '.join(sorted(set(where_cols)))}")

    if verdict.violations:
        return verdict

    # (7) enforce a LIMIT on the outermost SELECT
    outer = tree
    if isinstance(outer, exp.Select) and outer.args.get("limit") is None:
        outer.set("limit", exp.Limit(expression=exp.Literal.number(force_limit)))
        verdict.rewritten = True
        verdict.warnings.append(f"LIMIT {force_limit} was injected")

    verdict.sql = tree.sql(dialect="bigquery", pretty=True)
    verdict.ok = True
    return verdict


def summarise_for_user(verdict: SQLVerdict) -> str:
    if verdict.ok:
        return "passed"
    return verdict.reason
