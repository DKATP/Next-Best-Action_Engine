"""Synthetic data generator.

No real customer, operator or business data is used anywhere in this repository. Identifiers use a
deliberately invalid prefix (999...) so they cannot match real phone numbers. The generator
injects the kinds of defects the pipeline is built to catch (corrupt ages, missing sources,
mass-registration accounts) and can simulate a randomized campaign with a known hidden effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OTT = ["ott_prime", "ott_disney", "ott_netflix", "ott_sports", "ott_max", "ott_adult"]
SOURCE_AGE_DAYS = {"potential": 3, "billing": 5, "arpu": 12, "collections": 8, "retention": 9, "models": 6}


def source_loaded_at(snapshot: str) -> dict[str, str]:
    snap = pd.Timestamp(snapshot)
    return {k: (snap - pd.Timedelta(days=v)).strftime("%Y-%m-%d") for k, v in SOURCE_AGE_DAYS.items()}


def _lognormal(rng, median, sigma, n):
    return rng.lognormal(np.log(median), sigma, n)


def make_lines(n_lines: int = 20_000, seed: int = 7) -> pd.DataFrame:
    """One row per line. Columns mimic a weekly 'potential' input enriched by several sources."""
    rng = np.random.default_rng(seed)
    n_persons = int(n_lines * 0.62)
    weights = rng.pareto(2.5, n_persons) + 1
    weights /= weights.sum()
    person_idx = rng.choice(n_persons, n_lines, p=weights)
    n_mass = min(6, n_lines // 200)
    for k in range(n_mass):                       # mass-registration accounts: 25 lines each
        person_idx[k * 25:(k + 1) * 25] = k

    df = pd.DataFrame({
        "msisdn": [f"999{x:07d}" for x in rng.permutation(10**7)[:n_lines]],
        "person_id": [f"P{i:07d}" for i in person_idx],
        "line_type": rng.choice(["POSTPAID", "PREPAID", "HOME"], n_lines, p=[0.58, 0.32, 0.10]),
    })
    lt = df["line_type"].to_numpy()
    post, pre, home = lt == "POSTPAID", lt == "PREPAID", lt == "HOME"

    df["active"] = rng.random(n_lines) > 0.03
    df["suppressed_by_line"] = rng.random(n_lines) < 0.04
    person_supp = rng.random(n_persons) < 0.03
    df["suppressed_by_person"] = person_supp[person_idx]

    age = np.clip(rng.normal(40, 13, n_lines), 18, 85).round()
    bad = rng.random(n_lines) < 0.01              # corrupt dates -> impossible ages
    age[bad] = rng.choice([5, 250], bad.sum())
    df["age_years"] = age
    tenure = rng.gamma(2.0, 18, n_lines).round()
    bad = rng.random(n_lines) < 0.008
    tenure[bad] = rng.choice([-3, 900], bad.sum())
    df["tenure_months"] = tenure
    df["handset_brand"] = rng.choice(["BRAND_A", "BRAND_B", "BRAND_C", "BRAND_D", "BRAND_E"], n_lines,
                                     p=[0.30, 0.25, 0.20, 0.15, 0.10])

    person_has_home = pd.Series(home).groupby(person_idx).transform("max").to_numpy()
    arpu = np.where(post, _lognormal(rng, 52_000, 0.55, n_lines), np.nan)
    arpu[post & (rng.random(n_lines) < 0.08)] = np.nan              # no match in the ARPU source
    fixed = np.where(person_has_home, _lognormal(rng, 70_000, 0.4, n_lines), 0.0)
    df["arpu_mobile"] = arpu
    df["arpu_total"] = np.where(post, arpu + fixed, np.nan)
    df["home_consumption"] = np.where(home, _lognormal(rng, 90_000, 0.4, n_lines), np.nan)

    df["payment_behavior"] = np.where(
        pre, None, rng.choice(["VERY_GOOD", "GOOD", "REGULAR", "HIGH_RISK", "NEW"], n_lines,
                              p=[0.25, 0.35, 0.20, 0.10, 0.10]))
    credit = np.select(
        [rng.random(n_lines) < 0.10, rng.random(n_lines) < 0.5, rng.random(n_lines) < 0.5],
        [0.0, rng.uniform(1e6, 4e6, n_lines), rng.uniform(4e6, 6e6, n_lines)],
        default=rng.uniform(6e6, 8e6, n_lines))
    df["available_credit"] = np.where(post, credit, np.nan)
    df["cancel_intent_mobile"] = np.where(post, rng.poisson(0.15, n_lines), 0)
    df["cancel_intent_fixed"] = np.where(home, rng.poisson(0.10, n_lines), 0)
    df["device_financing_count"] = np.where(post, rng.poisson(0.4, n_lines), 0)
    df["topup_max_3m"] = np.where(pre, _lognormal(rng, 18_000, 0.8, n_lines), np.nan)
    band = np.clip(np.ceil(df["topup_max_3m"].fillna(0) / 12_000), 1, 5).astype(int).astype(str)
    df["topup_month_band"] = np.where(pre, band, None)

    # Input flags from the weekly potential file
    credit_ok = np.nan_to_num(df["available_credit"].to_numpy(), nan=0) > 0
    df["potential_device_financing"] = rng.random(n_lines) < np.where(post, 0.10 + 0.25 * credit_ok, 0.03)
    p_ott = {"ott_prime": 0.30, "ott_disney": 0.20, "ott_netflix": 0.25, "ott_sports": 0.15, "ott_max": 0.12, "ott_adult": 0.05}
    for col, p in p_ott.items():
        df[col] = rng.random(n_lines) < np.where(pre, 0.05, np.where(home, 0.25, p))
    df["tv_premium"] = rng.random(n_lines) < np.where(home, 0.10, 0.02)
    df["addon_extra_line"] = rng.random(n_lines) < 0.08
    df["migration_candidate"] = pre & (rng.random(n_lines) < 0.5)

    # Model / expert-source outputs, each with realistic partial coverage
    df["upsell_model_n_ott"] = np.where(rng.random(n_lines) < 0.70, rng.integers(0, 6, n_lines), np.nan)
    df["upsell_priority_score"] = np.where(rng.random(n_lines) < 0.02, rng.uniform(0.4, 0.95, n_lines), np.nan)
    df["migration_prob"] = np.where(pre & (rng.random(n_lines) < 0.66), rng.beta(1.2, 6, n_lines), np.nan)
    risky = (df["payment_behavior"] == "HIGH_RISK").to_numpy()
    covered = (post | home) & (rng.random(n_lines) < 0.85)
    df["collections_alert"] = np.where(covered, (rng.random(n_lines) < np.where(risky, 0.55, 0.08)).astype(float), np.nan)
    ret_cov = np.where(home, 0.9, 0.4)
    ret = rng.choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"], n_lines, p=[0.05, 0.10, 0.25, 0.60]).astype(object)
    ret[~((post | home) & (rng.random(n_lines) < ret_cov))] = None
    df["retention_priority"] = ret
    df["churn_model_flag"] = np.where(post & (rng.random(n_lines) < 0.45), (rng.random(n_lines) < 0.06).astype(float), np.nan)
    return df


def simulate_purchase(df: pd.DataFrame, seed: int = 11, group_col: str | None = None) -> pd.DataFrame:
    """Simulate a randomized campaign outcome with a KNOWN hidden effect (device-financing offer).

    Baseline propensity depends on tenure, credit, brand, ARPU, payment behaviour and the input flag.
    The treatment effect is heterogeneous: positive for younger generations, negative for customers
    with repeated cancellation intent ("sleeping dogs"). The true effect is stored in
    `_oracle_uplift` so estimators can be checked against it. This is only possible with synthetic data.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    if group_col is None:
        out["group"] = np.where(rng.random(n) < 0.5, "TREATMENT", "CONTROL")
        group_col = "group"
    treated = (out[group_col] == "TREATMENT").to_numpy()

    pay = out["payment_behavior"].fillna("")
    z = (-4.5
         + 0.8 * (out["tenure_band"] == "24-36") + 0.4 * (out["tenure_band"] == "12-24")
         + 0.6 * (out["credit_limit_band"] == "0-4M") + 0.5 * (out["handset_brand"] == "BRAND_A")
         + 0.5 * np.log(out["arpu_mobile"].fillna(52_000) / 52_000).clip(-1, 1)
         + 0.7 * out["potential_device_financing"].astype(float)
         + 0.5 * pay.isin(["VERY_GOOD", "GOOD"]) - 0.9 * (pay == "HIGH_RISK")
         - 0.5 * (out["cancel_intent_mobile"] >= 2)).to_numpy(dtype=float)
    tau = (0.25 + 0.6 * out["generation"].isin(["GEN_Z", "MILLENNIAL"])
           - 0.8 * (out["cancel_intent_mobile"] >= 2) - 0.3 * (pay == "HIGH_RISK")).to_numpy(dtype=float)
    sig = lambda x: 1 / (1 + np.exp(-x))
    p0, p1 = sig(z), sig(z + tau)
    out["purchase"] = (rng.random(n) < np.where(treated, p1, p0)).astype(int)
    out["_oracle_uplift"] = p1 - p0
    return out
