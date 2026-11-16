import numpy as np
import pandas as pd
import pytest

from nba import quality, rules


def test_coverage_gate_stops_the_run():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan]})
    with pytest.raises(quality.DataQualityError, match="coverage below minimum"):
        quality.validate_coverage(df, {"a": 0.5})


def test_coverage_gate_passes_and_reports():
    df = pd.DataFrame({"a": [1, 2, 3, np.nan]})
    rep = quality.validate_coverage(df, {"a": 0.7})
    assert rep.loc[0, "coverage"] == 0.75


def test_join_duplication_guard():
    with pytest.raises(quality.DataQualityError):
        quality.assert_same_rows(100, 140, "join")


def test_stale_source_stops_the_run():
    with pytest.raises(quality.DataQualityError, match="stale"):
        quality.check_freshness({"arpu": "2025-10-01"}, "2026-01-15", 45)


def test_domain_check_flags_unexpected_values():
    df = pd.DataFrame({"t": ["POSTPAID", "BROKEN"]})
    with pytest.raises(quality.DataQualityError):
        quality.validate_domain(df, "t", {"POSTPAID", "PREPAID"})


def test_rule_points_are_derived_from_lift(rule_table):
    row = rule_table[(rule_table.bu == "device_financing") & (rule_table.variable == "tenure_band") & (rule_table.criterion == "24-36")]
    assert row.points.iloc[0] == pytest.approx((1.30 - 1) * 20)


def test_rule_points_apply_missing_and_inactive_as_zero():
    table = pd.DataFrame({"bu": ["x", "x"], "variable": ["v", "v"], "criterion": ["a", "b"], "lift": [1.5, 2.0],
                          "factor": [10, 10], "active": [True, False]})
    table["points"] = (table.lift - 1) * table.factor
    df = pd.DataFrame({"v": ["a", "b", None, "zzz"]})
    assert rules.rule_points(df, table, "x").tolist() == [5.0, 0.0, 0.0, 0.0]


def test_duplicate_rules_are_rejected():
    dup = pd.DataFrame({"bu": ["x", "x"], "variable": ["v", "v"], "criterion": ["a", "a"], "lift": [1.1, 1.2], "factor": [10, 10]})
    with pytest.raises(quality.DataQualityError):
        rules.validate_rules(dup)
