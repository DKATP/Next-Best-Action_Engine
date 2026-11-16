"""Feature enrichment: sanity filters, bands and segments.

Lesson: corrupted source fields produce impossible values (ages of 250). They are set to NULL
and flagged, never silently used and never silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_SCORE = {"VERY_LOYAL": 100, "LOYAL": 75, "MEDIUM": 50, "NEW": 25}


def _band(s: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    out = pd.cut(s, bins=[-np.inf, *edges, np.inf], labels=labels, right=False)
    return out.astype("object").where(s.notna(), None)


def add_features(df: pd.DataFrame, n_lines_person: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    age_ok = out["age_years"].between(18, 100)
    ten_ok = out["tenure_months"].between(0, 600)
    out["flag_age_invalid"] = ~age_ok
    out["flag_tenure_invalid"] = ~ten_ok
    out["age"] = out["age_years"].where(age_ok)
    out["tenure"] = out["tenure_months"].where(ten_ok)

    out["tenure_band"] = _band(out["tenure"], [12, 24, 36], ["0-12", "12-24", "24-36", "36+"])
    out["generation"] = _band(out["age"], [27, 43, 59], ["GEN_Z", "MILLENNIAL", "GEN_X", "BOOMER"])
    out["credit_limit_band"] = _band(out["available_credit"], [1, 4_000_001, 5_000_001, 6_000_001],
                                     ["0", "0-4M", "4-5M", "5-6M", "6M+"])
    out["tenure_segment"] = np.select(
        [out["tenure"] >= 36, out["tenure"] >= 24, out["tenure"] >= 12, out["tenure"] >= 0],
        ["VERY_LOYAL", "LOYAL", "MEDIUM", "NEW"], default="NO_INFO")
    out["tenure_score"] = out["tenure_segment"].map(SEGMENT_SCORE)      # NaN for NO_INFO
    if n_lines_person is not None:
        out["lines_band"] = _band(n_lines_person.astype(float), [2, 5], ["1", "2-4", "5+"])
    out["cancel_intent_max"] = out[["cancel_intent_mobile", "cancel_intent_fixed"]].max(axis=1)
    return out
