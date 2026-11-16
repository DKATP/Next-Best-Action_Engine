"""Customer care: should we sell to this customer at all?

Principle: *experts decide, the engine's own rules are kept only for comparison.*
- Collections alert comes from the collections team's source.
- Retention risk comes from the retention team's priority (home/postpaid) or the mobile churn model.
- No data from the expert source means no alert (1), never a guess.
Codes: 1 = no alert, 0 = alert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CARE_ACTIONS = {(1, 1): "NORMAL_OFFER", (0, 1): "NO_CREDIT", (1, 0): "RETENTION_FIRST", (0, 0): "NO_SALE"}


def apply_care(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["care"]
    out = df.copy()

    # --- decision: expert sources ---
    coll_alert = out["collections_alert"].eq(1)
    out["care_collections"] = np.where(coll_alert, 0, 1)
    out["collections_source"] = np.where(out["collections_alert"].isna(), "NO_COLLECTIONS_DATA", "COLLECTIONS_TEAM")

    ret_alert = out["retention_priority"].isin(c["retention_risk_priorities"])
    churn_alert = out["churn_model_flag"].eq(1)
    out["care_retention"] = np.where(ret_alert | churn_alert, 0, 1)
    out["retention_source"] = np.select(
        [out["retention_priority"].notna(), out["churn_model_flag"].notna()],
        ["RETENTION_TEAM_PRIORITY", "CHURN_MODEL"], default="NO_RETENTION_DATA")

    out["care_action"] = [CARE_ACTIONS[(a, b)] for a, b in zip(out["care_collections"], out["care_retention"])]
    out["care_stage"] = np.select(
        [out["care_retention"].eq(0), out["care_collections"].eq(0), out["tenure"] < c["onboarding_months"]],
        ["RISK", "COLLECTIONS", "ONBOARDING"], default="DEVELOPMENT")
    out["sell_multiplier"] = (out["care_action"] != "NO_SALE").astype(int)

    # --- comparison only: the engine's own rules (never used for the decision) ---
    out["collections_alert_own_rule"] = out["payment_behavior"].eq("HIGH_RISK")
    out["retention_alert_own_rule"] = out["cancel_intent_max"] >= c["own_rule_cancel_intent_threshold"]
    return out


def care_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """How often do the engine's own rules agree with the expert sources? A validation, not a decision."""
    return pd.DataFrame({
        "collections": pd.crosstab(df["collections_alert_own_rule"], df["care_collections"].eq(0), margins=True).stack(),
    }).reset_index().rename(columns={"level_0": "own_rule_alert", "level_1": "expert_alert", "collections": "lines"})
