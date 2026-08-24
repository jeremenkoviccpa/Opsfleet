#!/usr/bin/env python
"""End-to-end check of the BigQuery adapter against the real public dataset.

Run after `gcloud auth application-default login`:

    GOOGLE_CLOUD_PROJECT=<your-project> PYTHONPATH=src python scripts/verify_bigquery.py

It exercises the same adapter the agent uses - credentials, schema reads, the
dry-run cost gate, real execution, the SQL validator and PII masking - and
reports the bytes each query would bill so you can see the cost controls working.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.safety import sql_guard  # noqa: E402
from agent.safety.pii import guard as pii_guard  # noqa: E402
from agent.tools.formatting import df_to_markdown  # noqa: E402
from agent.warehouse.base import WarehouseError  # noqa: E402

TL = "`bigquery-public-data.thelook_ecommerce`"

CHECKS = [
    ("revenue by month",
     f"""SELECT DATE_TRUNC(DATE(o.created_at), MONTH) AS month,
       ROUND(SUM(oi.sale_price), 2) AS revenue,
       COUNT(DISTINCT o.order_id) AS orders
FROM {TL}.orders o
JOIN {TL}.order_items oi ON oi.order_id = o.order_id
WHERE o.status NOT IN ('Cancelled','Returned')
  AND o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
GROUP BY month ORDER BY month DESC LIMIT 6"""),
    ("state spend gap (PII columns projected -> must be masked)",
     f"""SELECT u.id AS customer_id, u.state, u.age,
       ROUND(SUM(oi.sale_price), 2) AS revenue
FROM {TL}.users u
JOIN {TL}.orders o ON o.user_id = u.id
JOIN {TL}.order_items oi ON oi.order_id = o.order_id
WHERE u.state IN ('Texas','California') AND o.status NOT IN ('Cancelled','Returned')
GROUP BY 1,2,3 ORDER BY revenue DESC LIMIT 5"""),
    ("category margin",
     f"""SELECT p.category,
       COUNT(*) AS units,
       ROUND(SUM(oi.sale_price), 2) AS revenue,
       ROUND(SUM(oi.sale_price - p.cost), 2) AS gross_margin
FROM {TL}.order_items oi
JOIN {TL}.products p ON p.id = oi.product_id
WHERE p.category IN ('Jeans','Shorts')
GROUP BY p.category"""),
]

REJECT = [
    ("denied PII column", f"SELECT u.first_name, u.email FROM {TL}.users u LIMIT 5"),
    ("write attempt", f"DELETE FROM {TL}.orders WHERE 1=1"),
]


def main() -> int:
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("BIGQUERY_PROJECT")
    if not project:
        print("Set GOOGLE_CLOUD_PROJECT to the project that should be billed for compute.")
        return 2

    print(f"project: {project}\n")
    from agent.warehouse.bigquery_adapter import BigQueryWarehouse

    try:
        wh = BigQueryWarehouse(project_id=project)
    except WarehouseError as exc:
        print(f"FAIL  could not initialise the client: {exc}")
        print("      Run: gcloud auth application-default login")
        return 1

    health = wh.health()
    print(f"health: {health}\n")
    if health.get("status") != "ok":
        return 1

    print("--- schema reads ---")
    for table in ("orders", "order_items", "products", "users"):
        try:
            schema = wh.schema(table)
            print(f"  ok  {table:<12} {schema.row_estimate:>10,} rows, {len(schema.columns)} columns")
        except WarehouseError as exc:
            print(f"  FAIL {table}: {exc}")
            return 1

    total_bytes = 0
    print("\n--- queries (validated, dry-run costed, executed, masked) ---")
    for label, sql in CHECKS:
        print(f"\n[{label}]")
        verdict = sql_guard.validate(sql)
        if not verdict.ok:
            print(f"  FAIL validator rejected a query it should accept: {verdict.reason}")
            return 1
        print(f"  validator: ok  actions={verdict.column_actions or '{}'}")
        try:
            estimate = wh.estimate(verdict.sql)
            total_bytes += estimate.bytes_processed
            print(f"  dry run  : {estimate.bytes_processed/1e6:.1f} MB would be billed")
            result = wh.execute(verdict.sql, max_bytes_billed=2_000_000_000)
        except WarehouseError as exc:
            print(f"  FAIL {exc.kind}: {exc}")
            return 1
        masked, report = pii_guard().mask_dataframe(result.dataframe, verdict.column_actions)
        print(f"  executed : {result.rows} rows in {result.latency_ms:.0f} ms, "
              f"{result.bytes_billed/1e6:.1f} MB billed")
        if report.touched:
            print(f"  masking  : {report.describe()}")
        print("\n" + "\n".join("  " + l for l in df_to_markdown(masked, max_rows=6).splitlines()))

    print("\n--- statements the validator must reject ---")
    for label, sql in REJECT:
        verdict = sql_guard.validate(sql)
        if verdict.ok:
            print(f"  FAIL {label}: validator ACCEPTED a statement it must reject")
            return 1
        print(f"  ok  {label}: {verdict.reason[:80]}")

    print(f"\nPASS — BigQuery adapter verified. Total billed this run: "
          f"{total_bytes/1e6:.1f} MB of the 1 TB monthly free tier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
