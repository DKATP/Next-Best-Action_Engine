"""Post-deployment monitoring: population stability and score drift.

This is the control the original engine did NOT have, and the one most often missing in
production decision systems: nothing watched whether the population or the scores moved
between runs. A pipeline can pass every input gate and still quietly decay, because the
customers it scores this month are not the customers it scored last month.

PSI (Population Stability Index) compares the distribution of a variable between a
reference run and a current run. The conventional reading, which is a rule of thumb and
not a statistical test:

    PSI < 0.10   stable
    0.10 - 0.25  moderate shift, investigate
    PSI > 0.25   significant shift, the model or the rules may no longer apply

PSI says a distribution moved. It does not say performance dropped, and it cannot
replace measuring outcomes. Treat it as an alarm, not a verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = [(0.10, "STABLE"), (0.25, "MODERATE_SHIFT"), (np.inf, "SIGNIFICANT_SHIFT")]


def _band(value: float) -> str:
    return next(label for threshold, label in BANDS if value < threshold)


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10, epsilon: float = 1e-6) -> float:
    """PSI for a continuous variable, using quantile bins fixed on the REFERENCE run.

    The bin edges must come from the reference period. Re-binning on the current data
    hides the very shift the metric exists to detect.
    """
    ref, cur = reference.dropna(), current.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_share, cur_share = np.clip(ref_share, epsilon, None), np.clip(cur_share, epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_categorical(reference: pd.Series, current: pd.Series, epsilon: float = 1e-6) -> float:
    """PSI for a categorical variable. A category that appears only in one run still counts."""
    ref_share = reference.fillna("MISSING").value_counts(normalize=True)
    cur_share = current.fillna("MISSING").value_counts(normalize=True)
    categories = ref_share.index.union(cur_share.index)
    r = np.clip(ref_share.reindex(categories, fill_value=0).to_numpy(), epsilon, None)
    c = np.clip(cur_share.reindex(categories, fill_value=0).to_numpy(), epsilon, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], bins: int = 10) -> pd.DataFrame:
    """PSI per column between two runs, with its band. Sorted worst first."""
    rows = []
    for col in columns:
        if col not in reference.columns or col not in current.columns:
            continue
        numeric = pd.api.types.is_numeric_dtype(reference[col]) and not pd.api.types.is_bool_dtype(reference[col])
        value = psi_numeric(reference[col], current[col], bins) if numeric else psi_categorical(reference[col], current[col])
        rows.append({"column": col, "kind": "numeric" if numeric else "categorical",
                     "psi": round(value, 4), "band": _band(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def score_drift(reference: pd.Series, current: pd.Series, bins: int = 10) -> dict:
    """Drift of a score itself: PSI plus the shift in its mean and its top decile cut-off.

    The decile cut-off matters operationally: an audience defined as 'the top decile' can
    change meaning between runs even when the ranking is unchanged.
    """
    return {"psi": round(psi_numeric(reference, current, bins), 4),
            "band": _band(psi_numeric(reference, current, bins)),
            "mean_reference": round(float(reference.mean()), 4),
            "mean_current": round(float(current.mean()), 4),
            "p90_reference": round(float(reference.quantile(0.9)), 4),
            "p90_current": round(float(current.quantile(0.9)), 4)}
