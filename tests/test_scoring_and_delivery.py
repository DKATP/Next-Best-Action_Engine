import numpy as np
import pandas as pd
import pytest

from nba import delivery, scoring
from nba.quality import DataQualityError


def test_every_line_has_a_score_and_filters_live_in_applies(scored):
    for bu in scoring.BUS:
        assert scored[f"score_{bu}"].between(0.01, 1.0).all()
        assert not ((scored[f"applies_{bu}"] == 1) & (scored[f"flag_{bu}"] == 0)).any()
        assert not ((scored[f"applies_{bu}"] == 1) & ~scored["is_eligible"]).any()
    ineligible = scored[~scored["is_eligible"]]
    assert (ineligible[[f"applies_{b}" for b in scoring.BUS]].sum().sum()) == 0
    assert ineligible["score_prepaid"].notna().all()           # ineligible lines still carry a score


def test_effective_weight_does_not_penalise_missing_components(flagged, cfg, rule_table):
    row = flagged[flagged["line_type"] == "PREPAID"].head(1).copy()
    row["tenure_score"] = 100.0
    row["topup_month_band"] = "5"
    w = cfg["scoring"]["weights"]
    full = scoring.compute_scores(row, cfg, rule_table)["score_prepaid"].iloc[0]
    assert full == pytest.approx(round((50 * w["footprint"] + 100 * w["consumption"] + 100 * w["tenure"]) / 1.0 / 100, 2))
    row["topup_month_band"] = None
    missing = scoring.compute_scores(row, cfg, rule_table)["score_prepaid"].iloc[0]
    eff = w["footprint"] + w["tenure"]                                                   # consumption weight drops out
    assert missing == pytest.approx(round((50 * w["footprint"] + 100 * w["tenure"]) / eff / 100, 2))


def test_priority_list_boost_is_capped_at_one(flagged, cfg, rule_table):
    row = flagged.head(1).copy()
    row["upsell_priority_score"] = 0.99
    row["n_ott_input"], row["n_other_upsell_input"] = 3, 3
    assert scoring.compute_scores(row, cfg, rule_table)["score_upselling"].iloc[0] <= 1.0


def test_deciles_are_tie_aware_and_only_among_applicable_lines():
    s = pd.Series([0.2] * 50 + [0.5] * 30 + [0.9] * 20)
    applies = pd.Series([True] * 100)
    d = scoring._decile(s, applies)
    assert d[s == 0.9].eq(1.0).all() and d[s == 0.2].eq(0.1).all()
    assert d[s == 0.5].nunique() == 1                          # tied scores get the same decile
    none = scoring._decile(s, pd.Series([False] * 100))
    assert (none == 0).all()


def test_arpu_band_keeps_missing_as_missing():
    out = scoring.arpu_band(pd.Series([np.nan, 10_000, 25_000, 50_000, 100_000, 250_000]))
    assert out.isna().tolist() == [True, False, False, False, False, False]
    assert out.dropna().tolist() == [0, 25, 50, 75, 100]


def test_delivery_contract_and_validation(scored):
    d = delivery.build_delivery(scored, "RUN")
    delivery.validate_delivery(d, len(scored))
    assert list(d.columns[-5:]) == [f"flag_{b}" for b in scoring.BUS]      # flags always last
    with pytest.raises(DataQualityError):
        delivery.validate_delivery(d[list(reversed(d.columns))], len(scored))
    broken = d.copy()
    broken.loc[broken.index[0], "score_home"] = np.nan
    with pytest.raises(DataQualityError):
        delivery.validate_delivery(broken, len(scored))
