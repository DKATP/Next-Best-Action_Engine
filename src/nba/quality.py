"""Data-quality gates. A run stops loudly instead of producing a quietly wrong output.

Lesson behind this module: silent failures are worse than errors.
"""
from __future__ import annotations

import pandas as pd


class DataQualityError(RuntimeError):
    """Raised when an input fails a gate. The pipeline must not continue."""


def validate_schema(df: pd.DataFrame, required: list[str], name: str = "input") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataQualityError(f"{name}: missing columns {missing}")


def coverage(df: pd.DataFrame, col: str) -> float:
    """Share of non-null values in a column."""
    return float(df[col].notna().mean()) if len(df) else 0.0


def validate_coverage(df: pd.DataFrame, minimums: dict[str, float], name: str = "input") -> pd.DataFrame:
    """Stop if any key column is emptier than its minimum. Returns a coverage report."""
    rows = [{"column": c, "coverage": coverage(df, c), "minimum": m} for c, m in minimums.items()]
    report = pd.DataFrame(rows)
    bad = report[report["coverage"] < report["minimum"]]
    if len(bad):
        detail = ", ".join(f"{r.column}={r.coverage:.1%} (min {r.minimum:.0%})" for r in bad.itertuples())
        raise DataQualityError(f"{name}: coverage below minimum -> {detail}")
    return report


def assert_unique(df: pd.DataFrame, key: str | list[str], name: str = "table") -> None:
    dup = int(df.duplicated(key).sum())
    if dup:
        raise DataQualityError(f"{name}: {dup:,} duplicated rows on key {key}")


def assert_same_rows(before: int, after: int, step: str) -> None:
    """Guard after joins: a join that changes the row count is almost always a bug."""
    if before != after:
        raise DataQualityError(f"{step}: row count changed {before:,} -> {after:,} (join duplication?)")


def check_freshness(loaded_at: dict[str, str], snapshot_date: str, max_age_days: int) -> pd.DataFrame:
    """Stop if a source is older than allowed relative to the pinned snapshot date."""
    snap = pd.Timestamp(snapshot_date)
    rows = [{"source": s, "loaded_at": d, "age_days": (snap - pd.Timestamp(d)).days} for s, d in loaded_at.items()]
    report = pd.DataFrame(rows)
    stale = report[report["age_days"] > max_age_days]
    if len(stale):
        raise DataQualityError(f"stale sources: {stale['source'].tolist()} (max {max_age_days} days)")
    return report


def validate_domain(df: pd.DataFrame, col: str, allowed: set, name: str = "input") -> None:
    """Values outside the allowed set (ignoring nulls) are a gate failure."""
    bad = set(df[col].dropna().unique()) - set(allowed)
    if bad:
        raise DataQualityError(f"{name}.{col}: unexpected values {sorted(map(str, bad))[:5]}")
