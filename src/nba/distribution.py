"""Weekly audience: business quotas, one offer per line, random treatment/control split.

How it works:
1. Each offer takes its best lines by score (ties broken by a deterministic hash, so nobody is favoured
   and a rerun gives the same answer).
2. Each offer preselects `factor x quota x (1 + control_ratio)` lines so it has replacements.
3. A line that is a candidate for several offers stays with the offer where it ranks best *relative to
   that offer's quota* (position / quota). The BU with the highest raw score does not win: the offer that
   needs the line most to fill its quota does.
4. The selected lines are split at random between treatment and control, so both groups have the same
   score quality and a difference in sales can be attributed to the contact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .quality import DataQualityError


def load_quotas(path: str | Path) -> pd.DataFrame:
    q = pd.read_csv(path, keep_default_na=False)
    q["active"] = q["active"].astype(str).str.lower().eq("true")
    return q[q["active"]].reset_index(drop=True)


def _hash_rank(keys: pd.Series, salt: str) -> pd.Series:
    """Deterministic pseudo-random number per key and salt (stable across runs and machines)."""
    return pd.Series(pd.util.hash_pandas_object(keys.astype(str) + "|" + salt, index=False).to_numpy(), index=keys.index)


def distribute(delivery: pd.DataFrame, quotas: pd.DataFrame, week: str, cfg: dict) -> pd.DataFrame:
    dcfg = cfg["distribution"]
    ratio, factor = dcfg["control_ratio"], dcfg["preselection_factor"]
    cands = []
    for q in quotas.itertuples():
        d = delivery[(delivery["segment"] == q.segment) & (delivery[q.applies_col] == 1)].copy()
        if q.home_product:
            d = d[d["home_product"] == q.home_product]
        d["score"] = d[q.score_col]
        d["_rand"] = _hash_rank(d["msisdn"], week)
        if dcfg["one_line_per_person"]:
            d = d.sort_values(["score", "_rand"], ascending=[False, True]).drop_duplicates("person_id")
        d = d.sort_values(["score", "_rand"], ascending=[False, True])
        d["position"] = range(1, len(d) + 1)
        d = d[d["position"] <= q.weekly_quota * (1 + ratio) * factor].copy()
        d["offer"], d["quota"] = q.offer, q.weekly_quota
        d["pos_relative"] = d["position"] / (q.weekly_quota * (1 + ratio))
        if len(d) < q.weekly_quota:
            print(f"WARNING: {q.offer} has {len(d):,} candidates for a quota of {q.weekly_quota:,}; all are delivered")
        cands.append(d[["msisdn", "person_id", "segment", "offer", "quota", "score", "position", "pos_relative", "_rand"]])
    pre = pd.concat(cands, ignore_index=True)
    pre["n_candidate_offers"] = pre.groupby("msisdn")["msisdn"].transform("size")

    uniq = pre.sort_values(["pos_relative", "_rand"]).drop_duplicates("msisdn").copy()
    uniq = uniq.sort_values(["offer", "score", "_rand"], ascending=[True, False, True])
    uniq["final_position"] = uniq.groupby("offer").cumcount() + 1
    uniq = uniq[uniq["final_position"] <= uniq["quota"] * (1 + ratio)].copy()
    uniq["_grp"] = _hash_rank(uniq["msisdn"], week + "|group")
    uniq["_grp_rank"] = uniq.groupby("offer")["_grp"].rank(method="first")
    uniq["group"] = (uniq["_grp_rank"] <= uniq["quota"]).map({True: "TREATMENT", False: "CONTROL"})
    uniq["week"] = week
    out = uniq.drop(columns=["_rand", "_grp", "_grp_rank", "pos_relative"]).reset_index(drop=True)
    if out["msisdn"].duplicated().any():
        raise DataQualityError("distribution: a line landed in more than one offer")
    return out
