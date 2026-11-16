"""Delivery table with a fixed column contract, plus its validation.

Contract: identifiers first, then scores/deciles, then care and product columns, and finally
`applies_<bu>` and `flag_<bu>`, which always sit last so consumers can rely on the order.
"""
from __future__ import annotations

import pandas as pd

from .quality import DataQualityError, assert_unique
from .scoring import BUS

HEAD = ["msisdn", "person_id", "run_id", "is_eligible", "ineligible_reason", "selected"]
MIDDLE = (
    [f"score_{b}" for b in BUS] + [f"decile_{b}" for b in BUS]
    + ["care_action", "care_stage", "care_collections", "care_retention", "collections_source", "retention_source",
       "line_type", "segment", "home_product", "upsell_product", "upsell_category", "n_bu_apply"]
)
TAIL = [f"applies_{b}" for b in BUS] + [f"flag_{b}" for b in BUS]
CONTRACT = HEAD + MIDDLE + TAIL


def build_delivery(scored: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = scored.copy()
    out["run_id"] = run_id
    out["selected"] = pd.NA          # filled later by the distribution step
    return out[CONTRACT]


def validate_delivery(delivery: pd.DataFrame, expected_rows: int) -> None:
    assert_unique(delivery, "msisdn", "delivery")
    if len(delivery) != expected_rows:
        raise DataQualityError(f"delivery has {len(delivery):,} rows, expected {expected_rows:,}")
    if list(delivery.columns) != CONTRACT:
        raise DataQualityError("delivery: column contract violated")
    for b in BUS:
        if delivery[f"score_{b}"].isna().any() or (delivery[f"score_{b}"] <= 0).any():
            raise DataQualityError(f"delivery: lines without score for {b}")
        if ((delivery[f"applies_{b}"] == 1) & (delivery[f"flag_{b}"] == 0)).any():
            raise DataQualityError(f"delivery: applies_{b}=1 with input flag 0")
        if ((delivery[f"applies_{b}"] == 1) & ~delivery["is_eligible"]).any():
            raise DataQualityError(f"delivery: applies_{b}=1 on an ineligible line")
