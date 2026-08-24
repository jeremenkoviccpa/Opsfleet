"""Rendering helpers shared by the prompt builders and the CLI."""
from __future__ import annotations

import math
from typing import Any, List

import pandas as pd


def fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()[:19].replace("T", " ")
        except Exception:
            pass
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def df_to_markdown(df: pd.DataFrame, max_rows: int = 40, max_cols: int = 14) -> str:
    """Compact markdown table. Deliberately hand-rolled to avoid a hard tabulate dep."""
    if df is None or len(df.columns) == 0:
        return "_(no columns)_"
    if len(df) == 0:
        return "_(0 rows returned)_"

    frame = df.iloc[:max_rows, :max_cols]
    cols = [str(c) for c in frame.columns]
    rows = [[fmt_value(v) for v in rec] for rec in frame.itertuples(index=False, name=None)]

    widths = [
        min(28, max(len(cols[i]), *(len(r[i]) for r in rows))) if rows else len(cols[i])
        for i in range(len(cols))
    ]
    def line(cells: List[str]) -> str:
        return "| " + " | ".join(c[:widths[i]].ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend(line(r) for r in rows)

    notes = []
    if len(df) > max_rows:
        notes.append(f"{len(df) - max_rows} more rows not shown")
    if len(df.columns) > max_cols:
        notes.append(f"{len(df.columns) - max_cols} more columns not shown")
    if notes:
        out.append(f"_({'; '.join(notes)})_")
    return "\n".join(out)


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 3] + "..."
