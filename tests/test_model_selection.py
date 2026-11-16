import pytest

from nba import propensity as p


@pytest.fixture(scope="module")
def split():
    hist = p.make_history(n_lines=50_000, seed=21)
    train, test = p.split_by_customer(hist)
    return train[train.group == "TREATMENT"], test[test.group == "TREATMENT"]


def test_cross_validation_is_grouped_and_stable(split):
    train, _ = split
    cv = p.cross_validate_auc(train, n_splits=4)
    assert len(cv) == 4
    assert cv.attrs["mean_auc"] > 0.60
    assert cv.attrs["std_auc"] < 0.05                 # folds agree: the estimate is not noise


def test_model_comparison_includes_the_production_baseline(split):
    train, test = split
    table = p.compare_models(train, test)
    assert "rule_score (baseline)" in set(table["model"])
    learned = table[table["model"] != "rule_score (baseline)"]["auc"].max()
    baseline = table[table["model"] == "rule_score (baseline)"]["auc"].iloc[0]
    assert learned > baseline
    assert table.loc[table["model"] == "rule_score (baseline)", "brier"].isna().all()   # not a probability


def test_permutation_importance_ranks_a_real_driver_first(split):
    train, test = split
    importance = p.feature_importance(p.fit(train), test, n_repeats=5)
    top = set(importance.head(4)["feature"])
    assert {"payment_behavior", "credit_limit_band"} & top
    assert importance["auc_drop"].iloc[0] > 0


def test_calibrated_model_keeps_mean_prediction_near_the_observed_rate(split):
    train, test = split
    model = p.fit_calibrated(train)
    predicted = model.predict_proba(test[p.FEATURES])[:, 1]
    assert abs(predicted.mean() - test["purchase"].mean()) < 0.015
