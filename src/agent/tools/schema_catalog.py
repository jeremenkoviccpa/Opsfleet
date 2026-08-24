"""Schema catalog: the grounding context handed to the SQL generator.

Raw column lists are a weak prompt. What actually removes SQL errors is the
*semantic layer* - the join graph, the fact that revenue lives on
order_items.sale_price and not on orders, and which statuses to exclude. That
knowledge is curated here and versioned with the code, while the physical
columns are read live from whichever warehouse is attached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..safety.pii import guard as pii_guard
from ..warehouse.base import TableSchema, Warehouse

TABLE_NOTES = {
    "orders": "One row per order. Has NO monetary amount - revenue must come from order_items. "
              "`status` is one of Complete, Shipped, Processing, Cancelled, Returned.",
    "order_items": "One row per line item. `sale_price` is the price actually paid for that line - "
                   "this is the revenue grain. Join to products on product_id for cost/category.",
    "products": "Product catalogue. `retail_price` is list price, `cost` is unit cost. "
                "Margin = order_items.sale_price - products.cost.",
    "users": "Customer master. Contains personal data; most identifying columns are blocked.",
    "distribution_centers": "Fulfilment centres, joined from products.distribution_center_id.",
    "inventory_items": "Per-unit inventory with a denormalised product snapshot.",
}

JOIN_GRAPH = [
    "orders.user_id            -> users.id",
    "order_items.order_id      -> orders.order_id",
    "order_items.user_id       -> users.id",
    "order_items.product_id    -> products.id",
    "products.distribution_center_id -> distribution_centers.id",
    "inventory_items.product_id -> products.id",
]

METRIC_DEFINITIONS = [
    "revenue                = SUM(order_items.sale_price)",
    "gross_margin           = SUM(order_items.sale_price - products.cost)",
    "margin_pct             = gross_margin / revenue * 100",
    "orders                 = COUNT(DISTINCT orders.order_id)",
    "AOV                    = revenue / COUNT(DISTINCT orders.order_id)",
    "units                  = COUNT(order_items.id)",
    "customers              = COUNT(DISTINCT orders.user_id)",
    "revenue_per_customer   = revenue / COUNT(DISTINCT orders.user_id)",
    "discount_rate          = 1 - AVG(order_items.sale_price) / AVG(products.retail_price)",
    "return_rate            = COUNTIF(order_items.status = 'Returned') / COUNT(*)",
    "repeat_rate            = customers with >1 order / customers",
]

HOUSE_RULES = [
    "ALWAYS exclude cancelled and returned orders from revenue: "
    "`WHERE o.status NOT IN ('Cancelled','Returned')`. State this in the answer.",
    "Never compute revenue from `orders` alone - it has no amount column.",
    "The current calendar month/quarter is incomplete. Either exclude it or label it as partial.",
    "When comparing groups of different size, normalise per customer or per order.",
    "Prefer DATE_TRUNC(DATE(created_at), MONTH) for monthly grain.",
    "Always alias aggregate columns with readable snake_case names - they become report headers.",
    "There is no marketing-spend table, so CAC/ROAS cannot be computed. Say so rather than guessing.",
    "There is no subscription table, so churn is behavioural and its definition must be stated.",
]

CAPABILITY_SUMMARY = """The warehouse covers four things:
- WHO bought: customer demographics (age band, gender, state, city), acquisition channel, signup date.
- WHAT they bought: full product catalogue with category, brand, department, list price and unit cost -
  so margin and discount depth are computable.
- WHEN and HOW MUCH: orders and line items with timestamps, fulfilment status and the price actually paid.
- DELIVERY: distribution centre and shipped/delivered timestamps, so fulfilment speed is computable.

Not available: marketing spend (no CAC or ROAS), web/app sessions (no funnel or conversion rate),
subscriptions (churn must be defined behaviourally), competitor or price-index data, store-level
staffing or footfall."""


@dataclass
class CatalogEntry:
    table: str
    columns: List[Dict[str, str]]
    row_estimate: int
    note: str


class SchemaCatalog:
    def __init__(self, warehouse: Warehouse, tables: Optional[List[str]] = None) -> None:
        self.warehouse = warehouse
        self.tables = tables or ["orders", "order_items", "products", "users",
                                 "distribution_centers", "inventory_items"]
        self._cache: Dict[str, TableSchema] = {}
        self._errors: Dict[str, str] = {}

    def load(self) -> Dict[str, TableSchema]:
        for table in self.tables:
            if table in self._cache or table in self._errors:
                continue
            try:
                self._cache[table] = self.warehouse.schema(table)
            except Exception as exc:
                self._errors[table] = str(exc)[:200]
        return self._cache

    def prefix(self) -> str:
        return getattr(self.warehouse, "qualified_prefix", "")

    def render_for_sql_prompt(self, tables: Optional[List[str]] = None) -> str:
        """Compact, PII-annotated schema block for the SQL generator."""
        self.load()
        guard = pii_guard()
        wanted = tables or self.tables
        blocks: List[str] = []
        for table in wanted:
            schema = self._cache.get(table)
            if schema is None:
                continue
            cols: List[str] = []
            for col in schema.columns:
                action, _ = guard.action_for(table, col.name)
                if action == "deny":
                    cols.append(f"{col.name} {col.type} [BLOCKED - never select]")
                elif action == "hash":
                    cols.append(f"{col.name} {col.type} [pseudonymised after execution]")
                elif action == "generalize":
                    cols.append(f"{col.name} {col.type} [bucketed after execution]")
                else:
                    cols.append(f"{col.name} {col.type}")
            note = TABLE_NOTES.get(table, "")
            blocks.append(
                f"TABLE {self.prefix()}{table}  (~{schema.row_estimate:,} rows)\n"
                f"  {note}\n  columns: " + ", ".join(cols)
            )
        return "\n\n".join(blocks)

    def render_semantics(self) -> str:
        return (
            "JOIN GRAPH\n  " + "\n  ".join(JOIN_GRAPH)
            + "\n\nMETRIC DEFINITIONS\n  " + "\n  ".join(METRIC_DEFINITIONS)
            + "\n\nHOUSE RULES\n  " + "\n  ".join(f"- {r}" for r in HOUSE_RULES)
        )

    def capability_summary(self) -> str:
        return CAPABILITY_SUMMARY

    def describe_for_user(self) -> str:
        self.load()
        lines = [CAPABILITY_SUMMARY, "", "**Tables and scale**", ""]
        for table in self.tables:
            schema = self._cache.get(table)
            if schema is None:
                continue
            lines.append(f"- `{table}` — {schema.row_estimate:,} rows. {TABLE_NOTES.get(table, '')}")
        if self._errors:
            lines.append("")
            lines.append("Unavailable right now: " + ", ".join(self._errors))
        return "\n".join(lines)

    def health(self) -> Dict[str, Any]:
        self.load()
        return {"loaded": sorted(self._cache), "errors": self._errors}
