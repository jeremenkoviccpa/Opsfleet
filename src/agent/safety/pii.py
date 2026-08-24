"""PII enforcement.

Three independent layers, each of which alone would be insufficient:

  L1  SQL validation   - a column marked `deny` cannot appear in any SELECT
                         projection. The data never leaves the warehouse.
  L2  Result masking   - columns marked `hash` / `generalize` are transformed
                         in the DataFrame BEFORE the rows are serialised into
                         the model's context. The LLM never sees raw PII, so it
                         cannot leak it even under a successful jailbreak.
  L3  Output scrubbing - a regex sweep over the final natural-language answer,
                         catching anything that arrived through a free-text
                         column or was reconstructed by the model.

L2 is the load-bearing one: it moves the trust boundary so that prompt
injection cannot exfiltrate what was never placed in the prompt.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import pii_policy

_PROCESS_SALT = secrets.token_hex(16)

# Output column names that must be masked even when we cannot statically link
# them back to a source column (e.g. the query used SELECT * through a CTE).
NAME_FALLBACK = {
    "email": "hash", "user_email": "hash", "customer_email": "hash",
    "first_name": "deny", "last_name": "deny", "full_name": "deny", "name": "deny",
    "street_address": "deny", "address": "deny",
    "latitude": "deny", "longitude": "deny", "lat": "deny", "lon": "deny", "lng": "deny",
    "postal_code": "generalize", "zip": "generalize", "zipcode": "generalize",
}
# `products.name` and `distribution_centers.name` are not personal data.
NAME_FALLBACK_EXEMPT_PREFIXES = ("product", "brand", "category", "department", "dc_", "center", "store")


def _is_texty(series: pd.Series) -> bool:
    dtype = series.dtype
    if dtype == object:
        return True
    try:
        return bool(pd.api.types.is_string_dtype(dtype))
    except Exception:
        return False


@dataclass
class MaskReport:
    columns_masked: Dict[str, str] = field(default_factory=dict)
    columns_dropped: List[str] = field(default_factory=list)
    values_redacted: int = 0
    patterns_hit: List[str] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        return bool(self.columns_masked or self.columns_dropped or self.values_redacted)

    def describe(self) -> str:
        bits = []
        if self.columns_masked:
            bits.append(", ".join(f"{c}->{a}" for c, a in self.columns_masked.items()))
        if self.columns_dropped:
            bits.append("dropped: " + ", ".join(self.columns_dropped))
        if self.values_redacted:
            bits.append(f"{self.values_redacted} value(s) redacted ({', '.join(sorted(set(self.patterns_hit)))})")
        return "; ".join(bits)


class PIIGuard:
    def __init__(self) -> None:
        self._compiled: Dict[str, re.Pattern] = {}
        self._compiled_for: Optional[int] = None

    # ---- policy access ---------------------------------------------------
    @property
    def policy(self) -> Dict[str, Any]:
        return pii_policy()

    @property
    def columns(self) -> Dict[str, Dict[str, Any]]:
        return self.policy.get("columns", {}) or {}

    def salt(self) -> str:
        env_var = self.policy.get("salt_env_var", "PII_HASH_SALT")
        return os.getenv(env_var) or _PROCESS_SALT

    def action_for(self, table: str, column: str) -> Tuple[str, Dict[str, Any]]:
        entry = self.columns.get(f"{table}.{column}")
        if entry:
            return entry.get("action", "allow"), entry
        return "allow", {}

    def denied_columns(self) -> List[str]:
        return [k for k, v in self.columns.items() if v.get("action") == "deny"]

    def filter_only_columns(self) -> List[str]:
        return [k for k, v in self.columns.items() if v.get("filter_only")]

    def sensitive_summary(self) -> str:
        """A compact statement of the policy, injected into the SQL prompt."""
        deny = sorted(self.denied_columns())
        hashed = sorted(k for k, v in self.columns.items() if v.get("action") == "hash")
        gen = sorted(k for k, v in self.columns.items() if v.get("action") == "generalize")
        return (
            f"NEVER select these columns: {', '.join(deny)}.\n"
            f"These are pseudonymised after execution, select them freely as identifiers: {', '.join(hashed)}.\n"
            f"These are coarsened into buckets after execution: {', '.join(gen)}."
        )

    # ---- value transforms -------------------------------------------------
    def hash_value(self, value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        digest = hashlib.sha256(f"{self.salt()}|{value}".encode("utf-8")).hexdigest()
        return f"cust_{digest[:8]}"

    @staticmethod
    def generalize_value(value: Any, bucket: str) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        if bucket == "age_band":
            try:
                age = int(value)
            except (TypeError, ValueError):
                return value
            lower = max(0, (age // 10) * 10)
            return f"{lower}-{lower + 9}"
        if bucket == "postal_prefix_3":
            return f"{str(value)[:3]}**"
        return value

    # ---- L2: dataframe masking -------------------------------------------
    def mask_dataframe(
        self, df: pd.DataFrame, column_actions: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None
    ) -> Tuple[pd.DataFrame, MaskReport]:
        report = MaskReport()
        if df is None or df.empty and not len(df.columns):
            return df, report
        out = df.copy()
        actions: Dict[str, Tuple[str, Dict[str, Any]]] = dict(column_actions or {})

        # Name-based fallback for anything the SQL analyser could not resolve.
        for col in out.columns:
            key = str(col).lower()
            if key in actions:
                continue
            if any(key.startswith(p) for p in NAME_FALLBACK_EXEMPT_PREFIXES):
                continue
            fallback = NAME_FALLBACK.get(key)
            if fallback:
                bucket = "age_band" if key == "age" else "postal_prefix_3"
                actions[col] = (fallback, {"bucket": bucket})

        for col, (action, meta) in actions.items():
            if col not in out.columns:
                continue
            if action == "deny":
                out = out.drop(columns=[col])
                report.columns_dropped.append(str(col))
            elif action == "hash":
                out[col] = out[col].map(self.hash_value)
                report.columns_masked[str(col)] = "hash"
            elif action == "generalize":
                bucket = meta.get("bucket", "age_band")
                out[col] = out[col].map(lambda v, b=bucket: self.generalize_value(v, b))
                report.columns_masked[str(col)] = f"generalize:{bucket}"

        # L3 applied at cell level: free-text columns can carry PII regardless
        # of which source column they came from. Both the object dtype (pandas 2)
        # and the inferred str dtype (pandas 3) have to be swept.
        for col in out.columns:
            if _is_texty(out[col]):
                out[col] = out[col].map(lambda v: self._scrub_cell(v, report))
        return out, report

    def _scrub_cell(self, value: Any, report: MaskReport) -> Any:
        if not isinstance(value, str):
            return value
        cleaned, hits = self.scrub_text(value)
        if hits:
            report.values_redacted += len(hits)
            report.patterns_hit.extend(hits)
        return cleaned

    # ---- L3: text scrubbing ----------------------------------------------
    def _patterns(self) -> Dict[str, re.Pattern]:
        raw = self.policy.get("patterns", {}) or {}
        fingerprint = hash(tuple(sorted(raw.items())))
        if self._compiled_for != fingerprint:
            self._compiled = {}
            for kind, pattern in raw.items():
                try:
                    self._compiled[kind] = re.compile(pattern)
                except re.error:
                    continue
            self._compiled_for = fingerprint
        return self._compiled

    def scrub_text(self, text: str) -> Tuple[str, List[str]]:
        if not text:
            return text, []
        token_tpl = self.policy.get("redaction_token", "[REDACTED:{kind}]")
        hits: List[str] = []
        out = text
        for kind, pattern in self._patterns().items():
            def _sub(_m, k=kind):
                hits.append(k)
                return token_tpl.replace("{kind}", k)
            out = pattern.sub(_sub, out)
        return out, hits


_GUARD: Optional[PIIGuard] = None


def guard() -> PIIGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = PIIGuard()
    return _GUARD
