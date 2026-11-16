"""Traceability: funnel, source coverage and an execution log, per run."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .scoring import BUS


def funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Where do lines drop out? Total -> eligible -> care allows a sale -> at least one BU applies."""
    steps = [
        ("total_lines", len(df)),
        ("eligible", int(df["is_eligible"].sum())),
        ("eligible_and_care_allows_sale", int((df["is_eligible"] & (df["sell_multiplier"] == 1)).sum())),
        ("at_least_one_bu_applies", int((df["n_bu_apply"] > 0).sum())),
    ] + [(f"applies_{b}", int(df[f"applies_{b}"].sum())) for b in BUS]
    out = pd.DataFrame(steps, columns=["step", "lines"])
    out["share_of_total"] = (out["lines"] / len(df)).round(4)
    return out


def ineligibility_reasons(df: pd.DataFrame) -> pd.DataFrame:
    return (df["ineligible_reason"].fillna("ELIGIBLE").value_counts().rename_axis("reason").reset_index(name="lines"))


def source_coverage(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"column": columns, "coverage": [round(float(df[c].notna().mean()), 4) for c in columns]})


class RunLog:
    """One row per step and run. If a step has no OK row for a run, that step did not finish."""

    def __init__(self, run_id: str):
        self.run_id, self.rows = run_id, []

    def add(self, step: str, rows: int | None = None, status: str = "OK", detail: str = "") -> None:
        self.rows.append({"run_id": self.run_id, "step": step, "status": status, "rows": rows, "detail": detail[:500],
                          "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, key: str = "step", value: str = "lines") -> pd.DataFrame:
    """Compare two funnels by step order (not by step name), flagging big moves between runs."""
    m = a[[key, value]].merge(b[[key, value]], on=key, suffixes=("_prev", "_curr"))
    m["change_pct"] = ((m[f"{value}_curr"] - m[f"{value}_prev"]) / m[f"{value}_prev"].replace(0, pd.NA) * 100).astype(float).round(2)
    return m
