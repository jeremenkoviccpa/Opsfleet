"""Generates the offline mirror of bigquery-public-data.thelook_ecommerce.

Column names, types and value domains match the public dataset so that SQL
written for BigQuery runs unmodified (after dialect transpilation) against the
mirror. The generator is seeded, so every machine produces byte-identical data
and the eval suite has stable expected values.

The distributions are shaped so the reference business questions have real,
discoverable answers:
  * Texas customers order less often than California customers, at a nearly
    identical average order value  -> a frequency gap, not a basket gap;
  * Jeans carries a higher list price but a deeper discount than Shorts
    -> higher revenue, thinner margin;
  * an acquisition burst 4 months before "today" inflates the following
    quarter's apparent churn.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260824
N_USERS = 18_000
N_PRODUCTS = 2_400

STATES = {
    "California": dict(weight=0.19, freq=1.00, aov=1.00, organic=0.28),
    "Texas": dict(weight=0.15, freq=0.71, aov=0.97, organic=0.46),
    "New York": dict(weight=0.12, freq=0.98, aov=1.06, organic=0.26),
    "Florida": dict(weight=0.10, freq=0.86, aov=0.95, organic=0.35),
    "Illinois": dict(weight=0.08, freq=0.92, aov=0.99, organic=0.30),
    "Pennsylvania": dict(weight=0.07, freq=0.88, aov=0.96, organic=0.33),
    "Ohio": dict(weight=0.07, freq=0.84, aov=0.93, organic=0.36),
    "Georgia": dict(weight=0.06, freq=0.83, aov=0.94, organic=0.38),
    "Washington": dict(weight=0.05, freq=1.02, aov=1.08, organic=0.24),
    "Arizona": dict(weight=0.05, freq=0.87, aov=0.95, organic=0.34),
    "Michigan": dict(weight=0.06, freq=0.85, aov=0.94, organic=0.35),
}

CATEGORIES = {
    # list price range, cost ratio, discount depth, return rate, popularity
    "Jeans":                         dict(lo=45, hi=140, cost=0.42, disc=0.24, ret=0.11, pop=0.13),
    "Shorts":                        dict(lo=18, hi=60,  cost=0.38, disc=0.09, ret=0.06, pop=0.10),
    "Tops & Tees":                   dict(lo=12, hi=45,  cost=0.34, disc=0.12, ret=0.07, pop=0.18),
    "Sweaters":                      dict(lo=35, hi=120, cost=0.44, disc=0.19, ret=0.09, pop=0.08),
    "Outerwear & Coats":             dict(lo=70, hi=320, cost=0.48, disc=0.22, ret=0.13, pop=0.07),
    "Accessories":                   dict(lo=8,  hi=70,  cost=0.30, disc=0.07, ret=0.05, pop=0.11),
    "Swim":                          dict(lo=20, hi=85,  cost=0.36, disc=0.26, ret=0.14, pop=0.06),
    "Sleep & Lounge":                dict(lo=15, hi=65,  cost=0.35, disc=0.14, ret=0.06, pop=0.08),
    "Fashion Hoodies & Sweatshirts": dict(lo=28, hi=95,  cost=0.40, disc=0.15, ret=0.08, pop=0.10),
    "Intimates":                     dict(lo=10, hi=55,  cost=0.32, disc=0.11, ret=0.09, pop=0.09),
}

BRANDS = ["Allegra K", "Calvin Klein", "Carhartt", "Hanes", "Levi's", "Nautica", "Quiksilver",
          "Volcom", "Wrangler", "Dockers", "Columbia", "Under Armour", "Fruit of the Loom"]
TRAFFIC = ["Organic", "Email", "Facebook", "Search", "Display"]
DC = [("Memphis TN", 35.1, -90.0), ("Chicago IL", 41.8, -87.6), ("Houston TX", 29.7, -95.4),
      ("Los Angeles CA", 34.0, -118.2), ("New Orleans LA", 30.0, -90.1), ("Port Authority NY/NJ", 40.6, -74.0),
      ("Philadelphia PA", 39.9, -75.1), ("Mobile AL", 30.6, -88.0), ("Charleston SC", 32.7, -79.9),
      ("Savannah GA", 32.0, -81.1)]
FIRST = ["Maria", "James", "Aisha", "Chen", "Sofia", "Liam", "Priya", "Diego", "Emma", "Kwame",
         "Yuki", "Noah", "Ines", "Omar", "Hannah", "Luca", "Zara", "Ethan", "Nadia", "Mateo"]
LAST = ["Alvarez", "Novak", "Okafor", "Kim", "Rossi", "Haddad", "Delgado", "Nguyen", "Fischer",
        "Silva", "Kowalski", "Adeyemi", "Petrov", "Larsen", "Marino", "Osei", "Tanaka", "Weber"]

TABLE_DDL_ORDER = ["distribution_centers", "products", "users", "orders", "order_items", "inventory_items"]


def _months_back(today: dt.date, n: int) -> dt.date:
    y, m = today.year, today.month - n
    while m <= 0:
        m += 12
        y -= 1
    return dt.date(y, m, 1)


def build_frames(today: dt.date | None = None) -> dict[str, pd.DataFrame]:
    today = today or dt.date.today()
    rng = np.random.default_rng(SEED)
    horizon_start = _months_back(today, 36)
    total_days = (today - horizon_start).days

    # ---- distribution_centers ---------------------------------------------
    dcs = pd.DataFrame(
        {"id": np.arange(1, len(DC) + 1),
         "name": [d[0] for d in DC],
         "latitude": [d[1] for d in DC],
         "longitude": [d[2] for d in DC]}
    )

    # ---- products ----------------------------------------------------------
    cat_names = list(CATEGORIES)
    cat_p = np.array([CATEGORIES[c]["pop"] for c in cat_names], dtype=float)
    cat_p /= cat_p.sum()
    p_cat = rng.choice(cat_names, size=N_PRODUCTS, p=cat_p)
    lo = np.array([CATEGORIES[c]["lo"] for c in p_cat], dtype=float)
    hi = np.array([CATEGORIES[c]["hi"] for c in p_cat], dtype=float)
    retail = np.round(lo + rng.beta(2.2, 3.0, N_PRODUCTS) * (hi - lo), 2)
    cost_ratio = np.array([CATEGORIES[c]["cost"] for c in p_cat]) * rng.normal(1.0, 0.05, N_PRODUCTS)
    brands = rng.choice(BRANDS, size=N_PRODUCTS)
    products = pd.DataFrame({
        "id": np.arange(1, N_PRODUCTS + 1),
        "cost": np.round(retail * np.clip(cost_ratio, 0.2, 0.75), 2),
        "category": p_cat,
        "name": [f"{b} {c} {i}" for b, c, i in zip(brands, p_cat, range(1, N_PRODUCTS + 1))],
        "brand": brands,
        "retail_price": retail,
        "department": np.where(rng.random(N_PRODUCTS) < 0.52, "Women", "Men"),
        "sku": [f"SKU{i:07d}" for i in range(1, N_PRODUCTS + 1)],
        "distribution_center_id": rng.integers(1, len(DC) + 1, N_PRODUCTS),
    })

    # ---- users -------------------------------------------------------------
    st_names = list(STATES)
    st_p = np.array([STATES[s]["weight"] for s in st_names], dtype=float)
    st_p /= st_p.sum()
    u_state = rng.choice(st_names, size=N_USERS, p=st_p)

    # Signup dates, with a deliberate acquisition burst 4 months ago.
    signup_offset = rng.integers(0, total_days, N_USERS)
    burst_start = (_months_back(today, 4) - horizon_start).days
    burst_mask = rng.random(N_USERS) < 0.14
    signup_offset = np.where(
        burst_mask, rng.integers(burst_start, min(burst_start + 60, total_days), N_USERS), signup_offset
    )
    signup = np.array([horizon_start + dt.timedelta(days=int(d)) for d in signup_offset])

    organic_p = np.array([STATES[s]["organic"] for s in u_state])
    is_organic = rng.random(N_USERS) < organic_p
    other = rng.choice(["Email", "Facebook", "Search", "Display"], size=N_USERS, p=[0.32, 0.24, 0.30, 0.14])
    traffic = np.where(is_organic, "Organic", other)

    users = pd.DataFrame({
        "id": np.arange(1, N_USERS + 1),
        "first_name": rng.choice(FIRST, N_USERS),
        "last_name": rng.choice(LAST, N_USERS),
        "age": rng.integers(16, 71, N_USERS),
        "gender": rng.choice(["F", "M"], N_USERS, p=[0.53, 0.47]),
        "state": u_state,
        "street_address": [f"{n} {s} Street" for n, s in
                           zip(rng.integers(10, 9999, N_USERS), rng.choice(LAST, N_USERS))],
        "postal_code": [f"{c:05d}" for c in rng.integers(10000, 99999, N_USERS)],
        "city": [f"{s} City" for s in u_state],
        "country": "United States",
        "latitude": np.round(rng.uniform(25, 49, N_USERS), 4),
        "longitude": np.round(rng.uniform(-124, -68, N_USERS), 4),
        "traffic_source": traffic,
        "created_at": pd.to_datetime(signup).astype("datetime64[us]"),
    })
    users["email"] = [
        f"{f}.{l}{i}@example.com".lower()
        for f, l, i in zip(users.first_name, users.last_name, users.id)
    ]

    # ---- orders ------------------------------------------------------------
    # Expected order count per user = tenure * state frequency * personal lambda
    tenure_days = np.maximum(1, (today - pd.to_datetime(signup).date if False else
                                 np.array([(today - d).days for d in signup])))
    freq = np.array([STATES[s]["freq"] for s in u_state])
    personal = rng.gamma(1.6, 1.0, N_USERS)
    exp_orders = (tenure_days / 365.0) * 2.4 * freq * personal * 0.55
    n_orders = rng.poisson(np.clip(exp_orders, 0, 40))

    user_idx = np.repeat(np.arange(N_USERS), n_orders)
    total_orders = int(user_idx.size)
    order_ids = np.arange(1, total_orders + 1)

    # Order date uniformly between signup and today, then seasonally re-weighted.
    span = np.array([(today - signup[i]).days for i in user_idx])
    raw_offset = (rng.random(total_orders) ** 0.85 * span).astype(int)
    order_dates = np.array([signup[u] + dt.timedelta(days=int(o)) for u, o in zip(user_idx, raw_offset)])
    months = np.array([d.month for d in order_dates])
    season = np.array([1.0, 0.92, 1.0, 1.02, 1.05, 1.0, 0.96, 1.0, 1.04, 1.10, 1.34, 1.28])[months - 1]
    keep = rng.random(total_orders) < (season / season.max() * 0.92 + 0.08)

    order_ids, user_idx, order_dates = order_ids[keep], user_idx[keep], order_dates[keep]
    total_orders = int(order_ids.size)

    created_ts = pd.to_datetime(order_dates) + pd.to_timedelta(rng.integers(0, 86400, total_orders), unit="s")
    status = rng.choice(
        ["Complete", "Shipped", "Processing", "Cancelled", "Returned"],
        size=total_orders, p=[0.52, 0.20, 0.11, 0.07, 0.10],
    )
    num_items = rng.choice([1, 2, 3, 4, 5], size=total_orders, p=[0.46, 0.28, 0.14, 0.08, 0.04])

    shipped = created_ts + pd.to_timedelta(rng.integers(4, 96, total_orders), unit="h")
    delivered = shipped + pd.to_timedelta(rng.integers(12, 240, total_orders), unit="h")
    orders = pd.DataFrame({
        "order_id": order_ids,
        "user_id": users.id.values[user_idx],
        "status": status,
        "gender": users.gender.values[user_idx],
        "created_at": created_ts,
        "returned_at": np.where(status == "Returned", delivered, pd.NaT),
        "shipped_at": np.where(np.isin(status, ["Shipped", "Complete", "Returned"]), shipped, pd.NaT),
        "delivered_at": np.where(np.isin(status, ["Complete", "Returned"]), delivered, pd.NaT),
        "num_of_item": num_items,
    })
    for col in ("created_at", "returned_at", "shipped_at", "delivered_at"):
        orders[col] = pd.to_datetime(orders[col], errors="coerce").astype("datetime64[us]")

    # ---- order_items -------------------------------------------------------
    oi_order_idx = np.repeat(np.arange(total_orders), num_items)
    n_items = int(oi_order_idx.size)
    prod_pop = np.array([CATEGORIES[c]["pop"] for c in products.category])
    prod_pop = prod_pop / prod_pop.sum()
    prod_pick = rng.choice(np.arange(N_PRODUCTS), size=n_items, p=prod_pop)

    list_price = products.retail_price.values[prod_pick]
    item_cat = products.category.values[prod_pick]
    disc_base = np.array([CATEGORIES[c]["disc"] for c in item_cat])
    aov_mult = np.array([STATES[s]["aov"] for s in users.state.values[user_idx][oi_order_idx]])
    discount = np.clip(rng.normal(disc_base, 0.06), 0.0, 0.65)
    sale_price = np.round(list_price * (1 - discount) * aov_mult, 2)

    ret_p = np.array([CATEGORIES[c]["ret"] for c in item_cat])
    item_status = orders.status.values[oi_order_idx].copy()
    flip = (item_status == "Complete") & (rng.random(n_items) < ret_p)
    item_status = np.where(flip, "Returned", item_status)

    oi_created = orders.created_at.values[oi_order_idx]
    order_items = pd.DataFrame({
        "id": np.arange(1, n_items + 1),
        "order_id": orders.order_id.values[oi_order_idx],
        "user_id": orders.user_id.values[oi_order_idx],
        "product_id": products.id.values[prod_pick],
        "inventory_item_id": np.arange(1, n_items + 1),
        "status": item_status,
        "created_at": oi_created,
        "shipped_at": orders.shipped_at.values[oi_order_idx],
        "delivered_at": orders.delivered_at.values[oi_order_idx],
        "returned_at": np.where(item_status == "Returned",
                                orders.delivered_at.values[oi_order_idx], pd.NaT),
        "sale_price": sale_price,
    })
    for col in ("created_at", "shipped_at", "delivered_at", "returned_at"):
        order_items[col] = pd.to_datetime(order_items[col], errors="coerce").astype("datetime64[us]")

    inventory_items = pd.DataFrame({
        "id": order_items.inventory_item_id.values,
        "product_id": order_items.product_id.values,
        "created_at": order_items.created_at.values,
        "sold_at": order_items.created_at.values,
        "cost": products.cost.values[prod_pick],
        "product_category": item_cat,
        "product_name": products.name.values[prod_pick],
        "product_brand": products.brand.values[prod_pick],
        "product_retail_price": list_price,
        "product_department": products.department.values[prod_pick],
        "product_sku": products.sku.values[prod_pick],
        "product_distribution_center_id": products.distribution_center_id.values[prod_pick],
    })

    return {
        "distribution_centers": dcs,
        "products": products,
        "users": users[[
            "id", "first_name", "last_name", "email", "age", "gender", "state",
            "street_address", "postal_code", "city", "country", "latitude",
            "longitude", "traffic_source", "created_at",
        ]],
        "orders": orders,
        "order_items": order_items,
        "inventory_items": inventory_items,
    }


def seed_duckdb(db_path: Path, today: dt.date | None = None, force: bool = False) -> dict[str, int]:
    import duckdb

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not force:
        # Probe read-only so we do not conflict with a reader already attached.
        probe = duckdb.connect(str(db_path), read_only=True)
        try:
            existing = {r[0] for r in probe.execute("SHOW TABLES").fetchall()}
            if {"orders", "order_items", "products", "users"} <= existing:
                return {t: probe.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                        for t in sorted(existing)}
        finally:
            probe.close()

    con = duckdb.connect(str(db_path))

    frames = build_frames(today)
    counts: dict[str, int] = {}
    for name in TABLE_DDL_ORDER:
        df = frames[name]
        projection = ", ".join(
            f'CAST("{c}" AS TIMESTAMP) AS "{c}"'
            if pd.api.types.is_datetime64_any_dtype(df[c]) or "datetime" in str(df[c].dtype)
            else f'"{c}"'
            for c in df.columns
        )
        con.register("_stage", df)
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT {projection} FROM _stage")
        con.unregister("_stage")
        counts[name] = len(df)
    con.close()
    return counts
