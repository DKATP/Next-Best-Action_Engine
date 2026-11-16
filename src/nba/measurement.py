"""Treatment-vs-control measurement: lift by decile and a two-proportion test."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def incremental_effect(df: pd.DataFrame, group_col: str = "group", outcome_col: str = "purchase") -> dict:
    """Absolute and relative lift with a 95% CI and a two-sided z-test (difference of proportions)."""
    t = df[df[group_col] == "TREATMENT"][outcome_col]
    c = df[df[group_col] == "CONTROL"][outcome_col]
    nt, nc = len(t), len(c)
    pt, pc = t.mean(), c.mean()
    diff = pt - pc
    se = math.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)
    pooled = (t.sum() + c.sum()) / (nt + nc)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / nt + 1 / nc))
    z = diff / se0 if se0 > 0 else 0.0
    return {"n_treatment": nt, "n_control": nc, "rate_treatment": pt, "rate_control": pc,
            "abs_lift": diff, "rel_lift": diff / pc if pc > 0 else np.nan,
            "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se, "p_value": 2 * _norm_sf(abs(z))}


def lift_by_decile(df: pd.DataFrame, decile_col: str, group_col: str = "group", outcome_col: str = "purchase") -> pd.DataFrame:
    """Purchase rate of treatment and control inside each score decile.

    A useful score shows higher rates in higher deciles; the incremental column shows where the contact adds sales.
    """
    rows = []
    for dec, g in df.groupby(decile_col):
        t, c = g[g[group_col] == "TREATMENT"][outcome_col], g[g[group_col] == "CONTROL"][outcome_col]
        rows.append({"decile": dec, "n_treatment": len(t), "n_control": len(c),
                     "rate_treatment": t.mean() if len(t) else np.nan, "rate_control": c.mean() if len(c) else np.nan})
    out = pd.DataFrame(rows)
    out["incremental"] = out["rate_treatment"] - out["rate_control"]
    return out


def balance_check(df: pd.DataFrame, cols: list[str], group_col: str = "group") -> pd.DataFrame:
    """Standardized mean difference between groups for each numeric column. |SMD| < 0.1 is the usual 'balanced' bar."""
    t, c = df[df[group_col] == "TREATMENT"], df[df[group_col] == "CONTROL"]
    rows = []
    for col in cols:
        s = math.sqrt((t[col].var() + c[col].var()) / 2)
        rows.append({"column": col, "smd": (t[col].mean() - c[col].mean()) / s if s > 0 else 0.0})
    return pd.DataFrame(rows)


# ---------- experiment design: how big must the test be? ----------
# Done BEFORE launch. Running a test without this is how teams end up unable to tell
# "no effect" from "not enough data", which is the most common failure in A/B testing.
Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}          # one-sided z for power
Z_ALPHA_TWO_SIDED = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}


def required_sample_size(baseline_rate: float, minimum_detectable_effect: float,
                         power: float = 0.80, alpha: float = 0.05, relative: bool = True) -> dict:
    """Customers needed PER GROUP to detect a given lift on a two-proportion test.

    `minimum_detectable_effect` is relative by default (0.20 = a 20% lift over baseline).
    Note how brutally the cost scales when the baseline rate is low: halving the effect
    you want to detect roughly quadruples the sample.
    """
    p0 = baseline_rate
    p1 = p0 * (1 + minimum_detectable_effect) if relative else p0 + minimum_detectable_effect
    if not 0 < p0 < 1 or not 0 < p1 < 1:
        raise ValueError("rates must be strictly between 0 and 1")
    z_a, z_b = Z_ALPHA_TWO_SIDED[alpha], Z[power]
    p_bar = (p0 + p1) / 2
    n = ((z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
         / (p1 - p0) ** 2)
    return {"baseline_rate": p0, "target_rate": p1, "absolute_effect": p1 - p0,
            "per_group": int(math.ceil(n)), "total": int(math.ceil(n)) * 2, "power": power, "alpha": alpha}


def minimum_detectable_effect(baseline_rate: float, n_per_group: int,
                              power: float = 0.80, alpha: float = 0.05, tol: float = 1e-6) -> dict:
    """The smallest effect a test of this size can detect.

    The question to ask when the sample is fixed by a business quota: 'given 300 customers
    per group, what could we even see?' Solved by bisection on `required_sample_size`, so the
    two functions are exact inverses rather than two different approximations.
    """
    if n_per_group < 2:
        raise ValueError("n_per_group must be at least 2")
    low, high = tol, (1 - baseline_rate) / baseline_rate - tol      # target rate must stay below 1
    if required_sample_size(baseline_rate, high, power, alpha)["per_group"] > n_per_group:
        return {"baseline_rate": baseline_rate, "n_per_group": n_per_group,
                "absolute_mde": float("nan"), "relative_mde": float("nan"),
                "note": "no detectable effect at this sample size"}
    for _ in range(200):
        mid = (low + high) / 2
        if required_sample_size(baseline_rate, mid, power, alpha)["per_group"] > n_per_group:
            low = mid
        else:
            high = mid
    return {"baseline_rate": baseline_rate, "n_per_group": n_per_group,
            "absolute_mde": baseline_rate * high, "relative_mde": high}


def sample_size_curve(baseline_rate: float, effects=(0.10, 0.20, 0.30, 0.50, 1.00), power: float = 0.80) -> pd.DataFrame:
    """Sample size per group across a range of relative lifts, for planning conversations."""
    rows = [{"relative_lift": e, **{k: v for k, v in required_sample_size(baseline_rate, e, power).items()
                                     if k in ("target_rate", "per_group", "total")}} for e in effects]
    return pd.DataFrame(rows)
