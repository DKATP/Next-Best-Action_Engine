import warnings

import numpy as np
import pandas as pd
import pytest

from nba import cltv, synthetic

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- contractual
@pytest.fixture(scope="module")
def contractual():
    lines = synthetic.make_lines(20_000, seed=7)
    lines["n_lines_person"] = lines.groupby("person_id")["person_id"].transform("size")
    lines["generation"] = pd.cut(lines["age_years"], [0, 27, 43, 59, 200],
                                 labels=["GEN_Z", "MILLENNIAL", "GEN_X", "BOOMER"]).astype(object)
    history = synthetic.make_contractual_history(lines)
    design = cltv.prepare_contractual(history)
    return history, design, cltv.fit_cox(design)


def test_most_customers_are_censored(contractual):
    """The reason survival analysis is needed: a large share have not churned yet."""
    _, design, _ = contractual
    assert 0.2 < design["churned"].mean() < 0.8
    assert (design["duration_months"] > 0).all()


def test_cox_recovers_the_true_hazard_ratios(contractual):
    _, _, model = contractual
    hr = cltv.hazard_ratios(model).set_index("covariate")["hazard_ratio"]
    true = synthetic.TRUE_SURVIVAL_COEF
    for covariate, key in [("payment_behavior_HIGH_RISK", "payment_HIGH_RISK"),
                           ("payment_behavior_VERY_GOOD", "payment_VERY_GOOD"),
                           ("log_arpu", "log_arpu"), ("has_cancel_intent", "cancel_intent"),
                           ("has_extra_lines", "extra_lines")]:
        assert hr[covariate] == pytest.approx(np.exp(true[key]), rel=0.15), covariate


def test_c_index_beats_a_coin_flip(contractual):
    _, _, model = contractual
    assert 0.55 < cltv.concordance(model) < 0.95


def test_proportional_hazards_holds_on_data_generated_that_way(contractual):
    """The data is generated from a proportional-hazards process, so the test should pass.
    This checks the test itself is wired correctly, not just that it returns something."""
    _, design, model = contractual
    ph = cltv.check_proportional_hazards(model, design)
    assert set(ph["assumption"]) <= {"holds", "VIOLATED"}
    assert (ph["assumption"] == "VIOLATED").sum() <= 2
    assert ph["p_value"].between(0, 1).all()


def test_expected_tenure_is_capped_by_the_horizon(contractual):
    _, design, model = contractual
    tenure = cltv.expected_tenure(model, design, horizon_months=24)
    assert tenure.between(0, 24).all()
    assert cltv.expected_tenure(model, design, horizon_months=36).mean() > tenure.mean()


def test_cltv_rises_with_margin_and_falls_with_risk(contractual):
    history, design, model = contractual
    value = cltv.contractual_cltv(model, design, history["monthly_margin"], horizon_months=36)
    joined = history.assign(cltv=value["cltv"].to_numpy(), tenure=value["expected_tenure_months"].to_numpy())
    by_pay = joined.groupby(joined["payment_behavior"].fillna("UNKNOWN"))["tenure"].mean()
    assert by_pay["HIGH_RISK"] < by_pay["VERY_GOOD"]
    assert value["cltv"].corr(history["monthly_margin"]) > 0.5


def test_discounting_reduces_value():
    assert cltv.monthly_discount(0.10) == pytest.approx((1.10) ** (1 / 12) - 1)
    assert cltv.monthly_discount(0.0) == 0


def test_kaplan_meier_separates_payment_segments(contractual):
    history, _, _ = contractual
    curves = cltv.survival_by_segment(history.assign(pay=history["payment_behavior"].fillna("UNKNOWN")), "pay")
    at_24 = curves[curves["month"].between(20, 28)].groupby("segment")["survival"].mean()
    assert at_24["HIGH_RISK"] < at_24["VERY_GOOD"]
    assert curves["survival"].between(0, 1).all()


# ---------------------------------------------------------------- non-contractual
@pytest.fixture(scope="module")
def noncontractual():
    data = synthetic.make_noncontractual_history(6_000, seed=37)
    return data, cltv.fit_bgnbd(data), cltv.fit_gamma_gamma(data)


def test_bgnbd_recovers_its_true_parameters(noncontractual):
    _, params, _ = noncontractual
    for key, true in synthetic.TRUE_BGNBD.items():
        assert params[key] == pytest.approx(true, rel=0.20), key
    assert params["converged"]


def test_gamma_gamma_recovers_the_population_mean_spend(noncontractual):
    data, _, gg = noncontractual
    fitted_mean = gg["v"] * gg["p"] / (gg["q"] - 1)
    observed = data.loc[data["frequency"] > 0, "monetary_value"].mean()
    assert fitted_mean == pytest.approx(observed, rel=0.10)


def test_gamma_gamma_assumption_is_checked(noncontractual):
    data, _, _ = noncontractual
    assert abs(cltv.monetary_frequency_correlation(data)) < 0.10


def test_probability_alive_falls_as_a_customer_goes_quiet(noncontractual):
    data, params, _ = noncontractual
    joined = data.assign(p_alive=cltv.probability_alive(params, data))
    repeat = joined[joined["frequency"] > 0].copy()
    repeat["gap"] = repeat["T"] - repeat["recency"]
    recent = repeat[repeat["gap"] < 10]["p_alive"].mean()
    silent = repeat[repeat["gap"] > 60]["p_alive"].mean()
    assert recent > 0.7 > silent
    assert joined["p_alive"].between(0, 1).all()


def test_one_time_buyers_are_always_treated_as_alive(noncontractual):
    """With no repeat purchase there is no evidence of dropout, so the model cannot infer it."""
    data, params, _ = noncontractual
    p = cltv.probability_alive(params, data)
    assert (p[data["frequency"] == 0] == 1.0).all()


def test_expected_transactions_are_positive_and_ordered(noncontractual):
    data, params, _ = noncontractual
    expected = cltv.expected_transactions(params, data, periods=52.0)
    assert (expected >= 0).all()
    repeat = data.assign(e=expected)
    repeat = repeat[repeat["frequency"] > 0]
    recent = repeat[(repeat["T"] - repeat["recency"]) < 10]["e"].mean()
    silent = repeat[(repeat["T"] - repeat["recency"]) > 60]["e"].mean()
    assert recent > silent * 3


def test_expected_spend_shrinks_a_single_observation_toward_the_population(noncontractual):
    data, _, gg = noncontractual
    population_mean = gg["v"] * gg["p"] / (gg["q"] - 1)
    one_off = data[data["frequency"] == 1].copy()
    one_off["shrunk"] = cltv.expected_spend(gg, one_off)
    extreme = one_off.nlargest(50, "monetary_value")
    assert (extreme["shrunk"] < extreme["monetary_value"]).all()
    assert (extreme["shrunk"] > population_mean).all()


def test_noncontractual_cltv_falls_with_silence(noncontractual):
    data, params, gg = noncontractual
    value = cltv.noncontractual_cltv(params, gg, data, periods=52.0)
    joined = data.join(value)
    repeat = joined[joined["frequency"] > 0]
    recent = repeat[(repeat["T"] - repeat["recency"]) < 10]["cltv"].mean()
    silent = repeat[(repeat["T"] - repeat["recency"]) > 60]["cltv"].mean()
    assert recent > silent * 5
    assert (value["cltv"] >= 0).all()


# ---------------------------------------------------------------- integration
def test_value_segments_cover_every_customer():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"cltv": rng.gamma(2, 50_000, 1000), "uplift": rng.normal(0.02, 0.03, 1000)})
    out = cltv.value_segments(frame, "cltv", "uplift")
    assert out["value_segment"].notna().all()
    assert out["value_segment"].nunique() == 4


def test_expected_value_of_contact_can_be_negative():
    """A customer whose uplift is negative destroys value when contacted. The ranking must say so."""
    frame = pd.DataFrame({"uplift": [0.05, -0.02], "cltv": [500_000, 900_000]})
    value = cltv.expected_value_of_contact(frame, "uplift", "cltv", offer_margin=200_000, contact_cost=500)
    assert value.iloc[0] > 0 > value.iloc[1]
