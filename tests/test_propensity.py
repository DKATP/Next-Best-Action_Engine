import numpy as np
import pytest

from nba import propensity as p


@pytest.fixture(scope="module")
def history():
    return p.make_history(n_lines=50_000, seed=21)


def test_leakage_guard():
    with pytest.raises(ValueError, match="leakage"):
        p.assert_no_leakage(["tenure", "purchase"])
    p.assert_no_leakage(p.FEATURES)


def test_split_is_by_customer(history):
    train, test = p.split_by_customer(history)
    assert not set(train["person_id"]) & set(test["person_id"])
    assert 0.2 < len(test) / len(history) < 0.4


def test_response_model_ranks_better_than_chance_and_than_the_rule_score(history):
    train, test = p.split_by_customer(history)
    model = p.fit(train[train.group == "TREATMENT"])
    tt = test[test.group == "TREATMENT"]
    auc = p.ranking_metrics(tt.purchase, p.predict_proba(model, tt))["auc"]
    assert auc > 0.60
    assert auc > p.ranking_metrics(tt.purchase, tt.score_device_financing.to_numpy())["auc"]


def test_probabilities_are_calibrated_on_average(history):
    train, test = p.split_by_customer(history)
    model = p.fit(train[train.group == "TREATMENT"])
    tt = test[test.group == "TREATMENT"]
    m = p.probability_metrics(tt.purchase, p.predict_proba(model, tt))
    assert abs(m["mean_predicted"] - m["observed_rate"]) < 0.01


def test_rare_outcome_trap_is_visible(history):
    train, test = p.split_by_customer(history)
    model = p.fit(train[train.group == "TREATMENT"])
    tt = test[test.group == "TREATMENT"]
    demo = p.threshold_demo(tt.purchase, p.predict_proba(model, tt))
    assert demo["share_predicted_positive"] < 0.05          # predicts "no purchase" for nearly everyone
    assert demo["accuracy"] > 0.85 and demo["auc"] > 0.6     # looks accurate, yet ranks well: judge by deciles, not by class


def test_uplift_model_recovers_the_hidden_effect(history):
    train, test = p.split_by_customer(history)
    models = p.fit_uplift(train)
    test = test.assign(uplift=p.predict_uplift(models, test))
    assert np.corrcoef(test["uplift"], test["_oracle_uplift"])[0, 1] > 0.5
    top = p.uplift_by_decile(test, "uplift")
    assert top.loc[:2, "incremental"].mean() > top.loc[7:, "incremental"].mean()   # top deciles gain more than bottom ones
    assert p.qini_auuc(test, "uplift") > 0
