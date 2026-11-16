import numpy as np
import pandas as pd

from nba import care, universe


def _mini(rows):
    base = {"active": True, "suppressed_by_line": False, "suppressed_by_person": False}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_home_status_is_decided_per_person():
    lines = _mini([
        {"msisdn": "1", "person_id": "A", "line_type": "POSTPAID"},
        {"msisdn": "2", "person_id": "A", "line_type": "HOME"},
        {"msisdn": "3", "person_id": "B", "line_type": "POSTPAID"},
        {"msisdn": "4", "person_id": "C", "line_type": "HOME"},
        {"msisdn": "5", "person_id": "C", "line_type": "PREPAID"},
    ])
    status = universe.person_profile(lines).set_index("person_id")["home_status"]
    assert status["A"] == "ALREADY_CONVERGENT"
    assert status["B"] == "MOBILE_WITHOUT_HOME"
    assert status["C"] == "HOME_WITH_PREPAID"


def test_each_list_uses_its_own_key_and_reasons_are_recorded():
    lines = _mini([
        {"msisdn": "1", "person_id": "A", "line_type": "POSTPAID", "suppressed_by_line": True},
        {"msisdn": "2", "person_id": "A", "line_type": "POSTPAID"},
        {"msisdn": "3", "person_id": "B", "line_type": "POSTPAID", "suppressed_by_person": True},
        {"msisdn": "4", "person_id": "B", "line_type": "POSTPAID"},
    ])
    out = universe.build_universe(lines).set_index("msisdn")
    assert out.loc["1", "ineligible_reason"] == "LINE_SUPPRESSION_LIST"
    assert bool(out.loc["2", "is_eligible"])                 # a phone-level suppression does not exclude the person's other line
    assert out.loc["3", "ineligible_reason"] == "PERSON_SUPPRESSION_LIST"
    assert not out.loc["4", "is_eligible"]                    # a person-level suppression excludes all their lines


def test_mass_registration_is_excluded_and_rowcount_preserved(lines):
    out = universe.build_universe(lines, 8)
    assert len(out) == len(lines)
    mass = out[out["n_lines_person"] >= 8]
    assert len(mass) > 0 and (~mass["is_eligible"]).all()


def test_no_expert_data_means_no_alert_and_experts_override_own_rules(cfg):
    df = pd.DataFrame({
        "collections_alert": [np.nan, 1.0, 0.0], "retention_priority": [None, None, "CRITICAL"],
        "churn_model_flag": [np.nan, np.nan, np.nan], "payment_behavior": ["HIGH_RISK", "GOOD", "GOOD"],
        "cancel_intent_max": [3, 0, 0], "tenure": [10, 10, 2]})
    out = care.apply_care(df, cfg)
    assert out["care_collections"].tolist() == [1, 0, 1]      # line 0: own rule says risk, but no expert data -> no alert
    assert out["care_retention"].tolist() == [1, 1, 0]
    assert out["care_action"].tolist() == ["NORMAL_OFFER", "NO_CREDIT", "RETENTION_FIRST"]
    assert out["collections_alert_own_rule"].tolist() == [True, False, False]   # kept for comparison only
    assert out["care_stage"].tolist() == ["DEVELOPMENT", "COLLECTIONS", "RISK"]
