"""Regenerate every chart in docs/img from a real pipeline run:  python examples/make_charts.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from nba import cltv, pipeline, propensity as pr, synthetic, viz  # noqa: E402

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

    # --- phase 3: lifetime value on the same customers, so value and uplift can be compared
    history = synthetic.make_contractual_history(hist)
    design = cltv.prepare_contractual(history)
    cox = cltv.fit_cox(design)
    value = cltv.contractual_cltv(cox, design, history["monthly_margin"], horizon_months=36)
    value.index = history["msisdn"].to_numpy()
    curves = cltv.survival_by_segment(history.assign(pay=history["payment_behavior"].fillna("UNKNOWN")), "pay")
    quadrant = test.set_index("msisdn")[["uplift"]].join(value["cltv"], how="inner").dropna()

    for mode in ("light", "dark"):
        viz.chart_funnel(run["funnel"], OUT, mode)
        viz.chart_decile_lift(deciles, OUT, mode, overall_rate=treated_test["purchase"].mean())
        viz.chart_targeting(targeting, OUT, mode)
        viz.chart_calibration(calib, OUT, mode)
        viz.chart_survival(curves, OUT, mode)
        viz.chart_value_quadrant(quadrant, OUT, mode)
    print(f"Charts written to {OUT}")
    print(f"Cox C-index: {cltv.concordance(cox):.3f} | CLTV median: {value.cltv.median():,.0f} "
          f"| value-uplift correlation: {quadrant.cltv.corr(quadrant.uplift):+.3f}")
    print(f"top-decile lift of the rule score: {deciles.iloc[0]['lift']:.2f}x")
    print(targeting.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
