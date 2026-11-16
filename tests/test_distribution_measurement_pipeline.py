import numpy as np
import pandas as pd
import pytest

from nba import delivery, distribution, measurement, pipeline, quality


@pytest.fixture(scope="module")
def dist_inputs(scored, cfg):
    deliv = delivery.build_delivery(scored, "RUN")
    quotas = distribution.load_quotas(pipeline.ROOT / cfg["paths"]["quotas"])
    quotas["weekly_quota"] = 40
    return deliv, quotas


def test_no_line_in_two_offers_and_quotas_respected(dist_inputs, cfg):
    deliv, quotas = dist_inputs
    d = distribution.distribute(deliv, quotas, "2026-S03", cfg)
    assert not d["msisdn"].duplicated().any()
    per_offer = d.groupby(["offer", "group"]).size().unstack(fill_value=0)
    assert (per_offer["TREATMENT"] <= 40).all()
    assert (per_offer["CONTROL"] <= 40).all()


def test_distribution_is_deterministic_and_week_dependent(dist_inputs, cfg):
    deliv, quotas = dist_inputs
    a = distribution.distribute(deliv, quotas, "2026-S03", cfg)
    b = distribution.distribute(deliv, quotas, "2026-S03", cfg)
    c = distribution.distribute(deliv, quotas, "2026-S04", cfg)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["msisdn"]) != set(c["msisdn"]) or not a["group"].equals(c["group"])


def test_control_and_treatment_have_similar_score_quality(dist_inputs, cfg):
    deliv, quotas = dist_inputs
    d = distribution.distribute(deliv, quotas, "2026-S03", cfg)
    gap = d.groupby(["offer", "group"])["score"].mean().unstack().pipe(lambda t: (t["TREATMENT"] - t["CONTROL"]).abs())
    assert (gap < 0.05).all()


def test_incremental_effect_on_known_data():
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame({"group": np.repeat(["TREATMENT", "CONTROL"], n)})
    df["purchase"] = (rng.random(2 * n) < np.where(df["group"] == "TREATMENT", 0.10, 0.06)).astype(int)
    r = measurement.incremental_effect(df)
    assert r["abs_lift"] == pytest.approx(0.04, abs=0.01)
    assert r["ci_low"] < r["abs_lift"] < r["ci_high"] and r["p_value"] < 0.001
    null = df.assign(group=rng.permutation(df["group"].to_numpy()))
    assert measurement.incremental_effect(null)["p_value"] > 0.01


def test_balance_check_on_random_split():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=5000), "group": rng.choice(["TREATMENT", "CONTROL"], 5000)})
    assert measurement.balance_check(df, ["x"])["smd"].abs().iloc[0] < 0.1


def test_pipeline_end_to_end_and_funnel_is_monotonic():
    res = pipeline.run(6_000, seed=5)
    f = res["funnel"].set_index("step")["lines"]
    assert f["total_lines"] >= f["eligible"] >= f["eligible_and_care_allows_sale"] >= f["at_least_one_bu_applies"]
    assert set(res["log"]["status"]) == {"OK"} and len(res["log"]) >= 7
    assert res["delivery"]["msisdn"].is_unique


def test_pipeline_stops_when_a_source_arrives_empty(lines, cfg):
    broken = lines.assign(tenure_months=np.nan)
    with pytest.raises(quality.DataQualityError):
        quality.validate_coverage(broken, cfg["quality"]["min_coverage"], "lines")
