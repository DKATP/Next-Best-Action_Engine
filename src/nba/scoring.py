"""Scoring: one 0-1 score per business unit (BU) for every line, plus a 1/0 'applies' flag.

Key decisions (see docs/decisions.md):
- The engine does NOT pick a winning BU. Each BU gets its own score; offers compete later, in the
  distribution step, against business quotas.
- A score is computed for ALL lines. Filters (eligibility x input flag x customer care) live in a
  separate `applies_<bu>` column, so a filter can never silently turn a score into 0.
- score = (footprint*0.5 + consumption*0.3 + tenure*0.2) / effective weight / 100, where the
  effective weight drops the components that are missing for that line.
- Deciles are computed only among lines where the BU applies.
"""
from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd

from .rules import rule_points

BUS = ["device_financing", "upselling", "home", "postpaid", "prepaid"]


def _cap(s):
    return s.clip(0, 100)


def arpu_band(s: pd.Series) -> pd.Series:
    """Consumption band 0/25/50/75/100 by value; NaN stays NaN (missing component)."""
    out = pd.Series(np.select([s >= 160_000, s >= 80_000, s >= 40_000, s >= 20_000], [100, 75, 50, 25], default=0),
                    index=s.index, dtype=float)
    return out.where(s.notna())


def topup_band_3m(s: pd.Series) -> pd.Series:
    out = pd.Series(np.select([s >= 40_000, s >= 30_000, s >= 20_000, s >= 10_000], [100, 75, 50, 25], default=0),
                    index=s.index, dtype=float)
    return out.where(s.notna())


def topup_band_month(s: pd.Series) -> pd.Series:
    return s.astype("string").map({"5": 100, "4": 80, "3": 60, "2": 40, "1": 20}).astype(float)


def build_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flags per BU = input flag x eligibility. Also product suggestions for upselling and home."""
    up, sc = cfg["upselling"], cfg["scoring"]
    out = df.copy()
    up_cols = [c for c in out.columns if c.startswith(tuple(up["prefixes"]))]
    ott_cols = [c for c in up_cols if c.startswith("ott_")]
    ott_fp = [c for c in ott_cols if c not in up["no_footprint"]]
    other = [c for c in up_cols if c not in ott_cols]

    out["n_ott_input"] = out[ott_fp].astype(int).sum(axis=1)
    out["n_other_upsell_input"] = out[other].astype(int).sum(axis=1)
    order = [c for c in up["ott_priority"] if c in up_cols] + other
    prod = pd.Series(None, index=out.index, dtype="object")
    for c in reversed(order):                       # first in priority wins
        prod = prod.where(~out[c].astype(bool), c.upper())
    out["upsell_product_input"] = prod
    has_upsell = out[up_cols].astype(bool).any(axis=1)

    is_pre = out["line_type"].eq("PREPAID")
    topup = out["topup_max_3m"].fillna(0)
    to_postpaid = is_pre & (topup >= sc["postpaid_topup_threshold"])
    hs = out["home_status"]
    offer_home = hs.eq("MOBILE_WITHOUT_HOME")
    offer_mobile = hs.isin(["HOME_WITHOUT_MOBILE", "HOME_WITH_PREPAID"])
    convergent = hs.eq("ALREADY_CONVERGENT")
    elig = out["is_eligible"].astype(int)

    out["flag_device_financing"] = out["potential_device_financing"].astype(int) * elig
    out["flag_upselling"] = has_upsell.astype(int) * elig
    out["flag_home"] = (offer_home | offer_mobile | convergent).astype(int) * elig
    out["flag_postpaid"] = to_postpaid.astype(int) * elig
    out["flag_prepaid"] = (is_pre & ~to_postpaid).astype(int) * elig

    out["home_product"] = np.select(
        [out["flag_home"].eq(0), convergent, offer_home, offer_mobile],
        [None, "LOYALTY", "HOME_SALE", "MOBILE_SALE_IN_HOME_BASE"], default=None)
    out["segment"] = np.select(
        [is_pre, out["line_type"].eq("HOME") | convergent | hs.isin(["HOME_WITH_PREPAID", "HOME_WITHOUT_MOBILE"])],
        ["PREPAID", "HOME"], default="POSTPAID")
    out["upsell_product"] = out["upsell_product_input"].where(out["flag_upselling"].eq(1))
    out["upsell_category"] = np.select(
        [out["upsell_product"].str.startswith("OTT_", na=False), out["upsell_product"].str.startswith("TV_", na=False),
         out["upsell_product"].str.startswith("ADDON_", na=False), out["upsell_product"].notna()],
        ["OTT", "PREMIUM_CHANNELS", "ADDONS", "OTHER"], default=None)
    return out


def compute_scores(df: pd.DataFrame, cfg: dict, rules: pd.DataFrame) -> pd.DataFrame:
    sc = cfg["scoring"]
    w = sc["weights"]
    out = df.copy()
    pts = {bu: rule_points(out, rules, bu) for bu in ["device_financing", "home", "postpaid"]}

    hist = np.select([out["device_financing_count"] >= 2, out["device_financing_count"] == 1], [25, 18], default=0)
    prod_share = np.minimum((out["n_ott_input"] + out["n_other_upsell_input"]) / sc["upsell_product_cap"], 1.0) * 100
    model_sig = out["upsell_model_n_ott"].fillna(0) / 5.0 * 100
    list_sig = out["upsell_priority_score"].fillna(0) * 100
    is_home_line = out["line_type"].eq("HOME")
    mig_rules = _cap(50.0 + pts["postpaid"])
    mw = sc["migration_model_weight"]
    mig_fp = np.where(out["migration_prob"].notna(), mig_rules * (1 - mw) + out["migration_prob"] * 100 * mw, mig_rules)

    components = {  # footprint, consumption
        "device_financing": (_cap(30 * out["flag_device_financing"].clip(upper=1) + 30 + hist + pts["device_financing"]),
                             arpu_band(out["arpu_mobile"])),
        "upselling": (_cap(pd.Series(np.maximum.reduce([prod_share, model_sig, list_sig]), index=out.index)),
                      arpu_band(out["home_consumption"].where(is_home_line, out["arpu_total"]))),
        "home": (_cap(50.0 + pts["home"]),
                 arpu_band(out["home_consumption"].where(is_home_line, out["arpu_mobile"]))),
        "postpaid": (pd.Series(mig_fp, index=out.index), topup_band_3m(out["topup_max_3m"])),
        "prepaid": (pd.Series(float(sc["neutral_footprint"]), index=out.index), topup_band_month(out["topup_month_band"])),
    }
    tenure = out["tenure_score"]
    has_t = tenure.notna().astype(int)
    for bu, (footprint, consumption) in components.items():
        has_c = consumption.notna().astype(int)
        eff = w["footprint"] + w["consumption"] * has_c + w["tenure"] * has_t
        raw = (footprint * w["footprint"] + consumption.fillna(0) * w["consumption"] * has_c
               + tenure.fillna(0) * w["tenure"] * has_t) / eff / 100
        out[f"score_{bu}"] = raw.round(sc["decimals"]).clip(lower=sc["min_score"])
        applies = out["is_eligible"].astype(int) * out[f"flag_{bu}"] * out["sell_multiplier"]
        if bu in sc["credit_dependent_bus"]:
            applies = applies * out["care_collections"]
        out[f"applies_{bu}"] = applies.astype(int)

    boost = sc["upsell_priority_list_boost"]
    in_list = out["upsell_priority_score"].notna()
    out["score_upselling"] = np.where(in_list, (out["score_upselling"] + boost).clip(upper=1.0),
                                      out["score_upselling"]).round(sc["decimals"])
    out["n_bu_apply"] = out[[f"applies_{b}" for b in BUS]].sum(axis=1)
    for bu in BUS:
        out[f"decile_{bu}"] = _decile(out[f"score_{bu}"], out[f"applies_{bu}"].eq(1))
    return out


def _decile(score: pd.Series, applies: pd.Series) -> pd.Series:
    """Tie-aware decile (0.1 = lowest 10% ... 1.0 = highest 10%) among applicable lines; 0 = not applicable."""
    out = pd.Series(0.0, index=score.index)
    s = score[applies]
    if s.empty:
        return out
    cuts = s.quantile([i / 10 for i in range(1, 10)]).to_numpy()
    n_over = reduce(lambda a, q: a + (s > q).astype(int), cuts, pd.Series(0, index=s.index))
    dec = (np.minimum(n_over + 1, 10) / 10).round(1)
    dec[s >= s.max()] = 1.0
    dec[s <= s.min()] = 0.1
    out[applies] = dec
    return out
