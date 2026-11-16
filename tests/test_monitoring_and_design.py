import numpy as np
import pandas as pd
import pytest

from nba import measurement, monitoring


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(size=20_000)), pd.Series(rng.normal(size=20_000))
    assert monitoring.psi_numeric(a, b) < 0.01
    assert monitoring.score_drift(a, b)["band"] == "STABLE"


def test_psi_detects_a_shifted_population():
    rng = np.random.default_rng(1)
    a, c = pd.Series(rng.normal(0, 1, 20_000)), pd.Series(rng.normal(0.8, 1.3, 20_000))
    psi = monitoring.psi_numeric(a, c)
    assert psi > 0.25 and monitoring.score_drift(a, c)["band"] == "SIGNIFICANT_SHIFT"


def test_psi_bins_come_from_the_reference_not_the_current_run():
    """Re-binning on the current data would hide the shift; the value must stay large."""
    rng = np.random.default_rng(2)
    ref, cur = pd.Series(rng.normal(0, 1, 20_000)), pd.Series(rng.normal(3, 1, 20_000))
    assert monitoring.psi_numeric(ref, cur) > 1.0


def test_psi_categorical_handles_a_category_missing_from_one_run():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["A"] * 1000)
    assert monitoring.psi_categorical(ref, cur) > 0.25


def test_psi_report_is_sorted_and_banded(scored):
    half = len(scored) // 2
    a, b = scored.iloc[:half], scored.iloc[half:]
    report = monitoring.psi_report(a, b, ["score_home", "line_type", "tenure_band"])
    assert list(report["psi"]) == sorted(report["psi"], reverse=True)
    assert set(report["band"]) <= {"STABLE", "MODERATE_SHIFT", "SIGNIFICANT_SHIFT"}


def test_sample_size_grows_as_the_detectable_effect_shrinks():
    big = measurement.required_sample_size(0.06, 0.50)["per_group"]
    small = measurement.required_sample_size(0.06, 0.25)["per_group"]
    assert small > 3 * big                      # halving the effect roughly quadruples the sample


def test_sample_size_grows_as_the_baseline_rate_falls():
    common = measurement.required_sample_size(0.10, 0.20)["per_group"]
    rare = measurement.required_sample_size(0.001, 0.20)["per_group"]
    assert rare > 50 * common


def test_mde_and_sample_size_are_consistent():
    """The MDE of the sample size computed for an effect should return that effect."""
    target = 0.30
    n = measurement.required_sample_size(0.06, target)["per_group"]
    assert measurement.minimum_detectable_effect(0.06, n)["relative_mde"] == pytest.approx(target, rel=0.05)


def test_a_small_quota_can_only_detect_a_huge_effect():
    """300 per group against a 6% baseline cannot see anything smaller than roughly a 90% lift."""
    assert measurement.minimum_detectable_effect(0.06, 300)["relative_mde"] > 0.5


def test_sample_size_curve_is_monotonic():
    curve = measurement.sample_size_curve(0.05)
    assert list(curve["per_group"]) == sorted(curve["per_group"], reverse=True)
