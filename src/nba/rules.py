"""Editable business rules.

Rules live in a table (config/rules.csv here; a Delta table with change history in production),
each with its source lift. Points are *derived*: points = (lift - 1) * factor, so a rule can be
traced to the analysis it came from and edited without touching code.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .quality import DataQualityError


def load_rules(path: str | Path) -> pd.DataFrame:
    rules = pd.read_csv(path, dtype={"criterion": str})
    rules["active"] = rules["active"].astype(str).str.lower().eq("true")
    rules["points"] = ((rules["lift"] - 1.0) * rules["factor"]).round(2)
    validate_rules(rules)
    return rules


def validate_rules(rules: pd.DataFrame) -> None:
    if rules.duplicated(["bu", "variable", "criterion"]).any():
        raise DataQualityError("rules: duplicated (bu, variable, criterion)")
    if (rules["lift"] <= 0).any():
        raise DataQualityError("rules: lift must be > 0")


def rule_points(df: pd.DataFrame, rules: pd.DataFrame, bu: str) -> pd.Series:
    """Sum of points from the active rules of a BU. A missing value or unlisted criterion adds 0."""
    sub = rules[(rules["bu"] == bu) & rules["active"]]
    total = pd.Series(0.0, index=df.index)
    for variable, grp in sub.groupby("variable"):
        if variable not in df.columns:
            raise DataQualityError(f"rules: variable '{variable}' is not in the data")
        mapping = dict(zip(grp["criterion"], grp["points"]))
        total += df[variable].astype("string").map(mapping).fillna(0.0).astype(float)
    return total
