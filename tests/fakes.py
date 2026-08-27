"""Deterministic LLM test double.

Lets the full graph - guardrail, planner, SQL generation, validation, execution,
repair, synthesis, learning - run in CI with no API key and no network. Rules
are matched on the system prompt, so the double exercises the real prompt
plumbing rather than bypassing it.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from agent.llm import LLMRouter, ScriptedProvider

TL = "`bigquery-public-data.thelook_ecommerce`"

SQL_REVENUE_TREND = f"""SELECT DATE_TRUNC(DATE(o.created_at), MONTH) AS month,
       ROUND(SUM(oi.sale_price), 2) AS revenue,
       COUNT(DISTINCT o.order_id) AS orders
FROM {TL}.orders o
JOIN {TL}.order_items oi ON oi.order_id = o.order_id
WHERE o.status NOT IN ('Cancelled','Returned')
  AND o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
GROUP BY month
ORDER BY month"""

SQL_STATE_GAP = f"""WITH per_customer AS (
  SELECT u.id AS user_id, u.state,
         COUNT(DISTINCT o.order_id) AS orders,
         SUM(oi.sale_price) AS revenue
  FROM {TL}.users u
  JOIN {TL}.orders o ON o.user_id = u.id
  JOIN {TL}.order_items oi ON oi.order_id = o.order_id
  WHERE o.status NOT IN ('Cancelled','Returned')
    AND u.state IN ('Texas','California')
  GROUP BY 1, 2
)
SELECT state, COUNT(*) AS customers,
       ROUND(AVG(revenue), 2) AS avg_revenue_per_customer,
       ROUND(AVG(orders), 2) AS avg_orders_per_customer
FROM per_customer
GROUP BY state
ORDER BY avg_revenue_per_customer DESC"""

SQL_CATEGORY = f"""SELECT p.category,
       COUNT(*) AS units,
       ROUND(SUM(oi.sale_price), 2) AS revenue,
       ROUND(SUM(oi.sale_price - p.cost), 2) AS gross_margin
FROM {TL}.order_items oi
JOIN {TL}.products p ON p.id = oi.product_id
GROUP BY p.category
ORDER BY revenue DESC
LIMIT 10"""

# Deliberately broken and unsafe variants, used to exercise the repair loop and
# the PII validator.
SQL_BROKEN_SYNTAX = f"SELECT category, SUM(sale_price) FROMM {TL}.order_items GROUP BY category"
SQL_PII_LEAK = f"SELECT u.first_name, u.last_name, u.email FROM {TL}.users u LIMIT 10"
SQL_EMPTY = (f"SELECT o.order_id, o.status FROM {TL}.orders o "
             f"WHERE o.status = 'NoSuchStatusValue' LIMIT 100")


def _is(system: str, marker: str) -> bool:
    return marker in (system or "")


def _question(user_block: str) -> str:
    """Pull the manager's actual message out of a rendered prompt block."""
    for marker in ("MANAGER'S MESSAGE\n", "MANAGER'S QUESTION\n"):
        if marker in user_block:
            return user_block.split(marker, 1)[1].strip()
    m = re.search(r"The manager asked: (.+)", user_block)
    return m.group(1).strip() if m else user_block


class FakeLLM:
    """Configurable scripted provider with per-purpose overrides."""

    def __init__(self) -> None:
        self.provider = ScriptedProvider()
        self.sql_queue: List[str] = []
        self.plan_override: Optional[Dict] = None
        self.guard_override: Optional[Dict] = None
        self.fail_purposes: set[str] = set()
        self.transient_purposes: set[str] = set()
        self._install()

    # -- helpers ---------------------------------------------------------
    def queue_sql(self, *statements: str) -> "FakeLLM":
        self.sql_queue.extend(statements)
        return self

    def set_plan(self, plan: Dict) -> "FakeLLM":
        self.plan_override = plan
        return self

    def fail(self, *purposes: str) -> "FakeLLM":
        """Fail one stage without tripping the provider circuit breaker.

        Uses a permanently-classified error, which is not retried, so the
        breaker records a single failure and stays closed. This isolates
        "one stage broke" from "the provider is down".
        """
        self.fail_purposes.update(purposes)
        return self

    def fail_transient(self, *purposes: str) -> "FakeLLM":
        """Simulate a real provider outage: retried, and it opens the breaker."""
        self.fail_purposes.update(purposes)
        self.transient_purposes.update(purposes)
        return self

    def _boom(self, purpose: str):
        if purpose in self.transient_purposes:
            raise TimeoutError(f"{purpose}: 503 service unavailable")
        raise RuntimeError(f"{purpose}: invalid argument - malformed request")

    @property
    def calls(self) -> List[Dict[str, str]]:
        return self.provider.calls

    def call_count(self, marker: str) -> int:
        return sum(1 for c in self.provider.calls if marker in c["system"])

    def router(self, tracer=None) -> LLMRouter:
        r = LLMRouter(chain=[], tracer=tracer)
        r.register(self.provider)
        return r

    # -- rules -----------------------------------------------------------
    def _install(self) -> None:
        p = self.provider

        # Stand-in for the stage-2 scope classifier. Kept crude on purpose: a
        # real model generalises far better, so anything this catches is a
        # lower bound on the deployed behaviour.
        OFF_TOPIC = ("poem", "haiku", "joke", "recipe", "weather", "football",
                     "translate", "write me a story", "python script", "stock price",
                     "medical", "legal advice", "who won")

        def guard(_s, user):
            if "guardrail" in self.fail_purposes:
                self._boom("guardrail")
            if self.guard_override is not None:
                return json.dumps(self.guard_override)
            u = (user or "").lower()
            if any(marker in u for marker in OFF_TOPIC):
                return json.dumps({"decision": "refuse_out_of_scope",
                                   "reason": "unrelated to retail data", "confidence": 0.9})
            return json.dumps({"decision": "allow", "reason": "data question", "confidence": 0.95})

        def plan(_s, user):
            if "plan" in self.fail_purposes:
                self._boom("plan")
            if self.plan_override is not None:
                return json.dumps(self.plan_override)
            user = _question(user)
            q = user.lower()
            base = {"route": "analysis", "reason": "needs data", "time_window": "last 12 months",
                    "notes": "", "steps": [{"step_id": "s1", "goal": "pull the figures"}],
                    "deletion_criteria": {}, "report_title": ""}
            if "what data" in q or "available" in q or "what can you" in q:
                base.update(route="schema", steps=[])
            elif re.search(r"\bundo\b|restore|bring back|put back|recover", q):
                # Must precede the delete rule: "undo that delete" contains both.
                mentions = re.findall(r"mentioning ([A-Za-z ]+)", user)
                base.update(route="restore_reports", steps=[], deletion_criteria={
                    "mentions": [m.strip() for m in mentions],
                    "session_scope": False, "all": False,
                })
            elif re.search(r"delete|remove|purge", q) and "report" in q:
                mentions = re.findall(r"mentioning ([A-Za-z ]+)", user)
                scoped = "this conversation" in q or "we made" in q
                base.update(route="delete_reports", steps=[], deletion_criteria={
                    "mentions": [m.strip() for m in mentions],
                    "session_scope": scoped,
                    "all": (not mentions and not scoped
                            and bool(re.search(r"\ball (my |the )?reports\b", q))),
                })
            elif "report" in q and ("create" in q or "write" in q or "produce" in q):
                base.update(report_title="Q1 Performance Review",
                            steps=[{"step_id": "s1", "goal": "revenue by month and category"}])
            elif "hello" in q or "thanks" in q:
                base.update(route="converse", steps=[])
            elif any(w in q for w in ("jeans", "shorts", "category", "categories", "product")):
                base.update(steps=[{"step_id": "s1",
                                    "goal": "revenue, margin and discount depth by product category"}],
                            notes="decompose into units, price and discount depth")
            elif "texas" in q or "california" in q:
                base.update(steps=[{"step_id": "s1", "goal": "spend per customer by state"}],
                            notes="decompose into frequency and AOV")
            return json.dumps(base)

        def gen_sql(_s, user):
            if "sql_generation" in self.fail_purposes:
                self._boom("sql_generation")
            if self.sql_queue:
                return self.sql_queue.pop(0)
            g = user.split("Write the SQL for exactly this step:")[-1].lower()
            if "state" in g or "texas" in g:
                return SQL_STATE_GAP
            if "categor" in g or "product" in g or "jeans" in g:
                return SQL_CATEGORY
            return SQL_REVENUE_TREND

        def repair(_s, _u):
            if "sql_repair" in self.fail_purposes:
                self._boom("sql_repair")
            return self.sql_queue.pop(0) if self.sql_queue else SQL_REVENUE_TREND

        def analyse(_s, user):
            if "analysis" in self.fail_purposes:
                self._boom("analysis")
            table = re.search(r"\|(.+)\n", user)
            body = ("**Revenue is up.**\n\n- Figures are taken from the query results above.\n"
                    "- Time window: last 12 months.\n")
            if "TITLE:" in _s or "SAVED REPORT" in _s:
                return "TITLE: Q1 Performance Review\n\n# Q1 Performance Review\n\n" + body + \
                       "\n## Recommended actions\n- Rebalance Jeans discounting — owner: Merch — " \
                       "target: margin % — measure over Q2."
            return body

        def prefs(_s, user):
            u = _question(user).lower()
            signals = []
            if "bullet" in u:
                signals.append({"key": "output_format", "value": "bullets",
                                "explicit": "always" in u or "prefer" in u, "evidence": "bullets"})
            if "table" in u and "bullet" not in u:
                signals.append({"key": "output_format", "value": "table",
                                "explicit": "always" in u or "prefer" in u, "evidence": "tables"})
            if "short" in u or "brief" in u:
                signals.append({"key": "analysis_depth", "value": "headline",
                                "explicit": "always" in u or "from now on" in u,
                                "evidence": "keep it short"})
            return json.dumps({"signals": signals})

        def curate(_s, _u):
            return json.dumps({"intent_tags": ["time_series", "revenue"],
                               "method_notes": ["Exclude cancelled and returned orders."],
                               "generalised_question": "What was revenue over <period>?",
                               "quality": 0.72})

        def schema_answer(_s, _u):
            return ("We hold orders, line items, products and customers. "
                    "We do not hold marketing spend, so CAC is not computable.")

        def converse(_s, _u):
            return "Happy to help — ask me anything about sales, products or customers."

        p.rule(lambda s, u: _is(s, "You classify messages sent to"), guard)
        p.rule(lambda s, u: _is(s, "You are the planning stage"), plan)
        p.rule(lambda s, u: _is(s, "You are repairing a BigQuery SQL"), repair)
        p.rule(lambda s, u: _is(s, "You write a single BigQuery Standard SQL"), gen_sql)
        p.rule(lambda s, u: _is(s, "senior retail data analyst"), analyse)
        p.rule(lambda s, u: _is(s, "You detect durable formatting"), prefs)
        p.rule(lambda s, u: _is(s, "curating a candidate entry"), curate)
        p.rule(lambda s, u: _is(s, "explaining what data is available"), schema_answer)
        p.rule(lambda s, u: _is(s, "assistant talking to"), converse)
