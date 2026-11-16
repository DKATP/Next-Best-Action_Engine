"""Regenerate every chart in docs/img from a real pipeline run:  python examples/make_charts.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nba import pipeline, propensity as pr, viz  # noqa: E402

OUT = ROOT / "docs" / "img"


def main() -> None:
    run = pipeline.run(20_000, seed=7)
    hist = pr.make_history()
    train, test = pr.split_by_customer(hist)
    treated_train, treated_test = train[train.group == "TREATMENT"], test[test.group == "TREATMENT"]

    model = pr.fit(treated_train)
    p_test = pr.predict_proba(model, treated_test)
    deciles = pr.decile_table(treated_test["purchase"], treated_test["score_device_financing"].to_numpy())
    calib = pr.calibration_table(treated_test["purchase"], p_test)

    test = test.assign(p_response=pr.predict_proba(model, test))
    test["uplift"] = pr.predict_uplift(pr.fit_uplift(train), test)
    targeting = pr.compare_targeting(test, {"rule_score": "score_device_financing",
                                            "response_model": "p_response", "uplift_model": "uplift"})

    for mode in ("light", "dark"):
        viz.chart_funnel(run["funnel"], OUT, mode)
        viz.chart_decile_lift(deciles, OUT, mode, overall_rate=treated_test["purchase"].mean())
        viz.chart_targeting(targeting, OUT, mode)
        viz.chart_calibration(calib, OUT, mode)
    print(f"Charts written to {OUT}")
    print(f"top-decile lift of the rule score: {deciles.iloc[0]['lift']:.2f}x")
    print(targeting.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
