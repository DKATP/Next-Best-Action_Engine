"""Phase 2 walk-through on synthetic data:  python examples/run_phase2.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from nba import propensity as p  # noqa: E402

pd.set_option("display.width", 140)
hist = p.make_history()
train, test = p.split_by_customer(hist)
print(f"history {len(hist):,} | train {len(train):,} | test {len(test):,} | observed purchase rate {hist.purchase.mean():.1%}")
print(f"overall incremental effect of contact (test): {p.incremental_rate(test):+.2%}\n")

# 1) response model: trained on treated customers only
treated_train = train[train.group == "TREATMENT"]
resp = p.fit(treated_train)
test = test.assign(p_response=p.predict_proba(resp, test))
tt = test[test.group == "TREATMENT"]
print("RESPONSE MODEL (treated customers in the test set)")
print(" ranking:", {k: round(v, 3) for k, v in p.ranking_metrics(tt.purchase, tt.p_response.to_numpy()).items()})
print(" probabilities:", {k: round(v, 4) for k, v in p.probability_metrics(tt.purchase, tt.p_response.to_numpy()).items()})
print(" rule-score baseline AUC:", round(p.ranking_metrics(tt.purchase, tt.score_device_financing.to_numpy())["auc"], 3))
print("\ncalibration\n", p.calibration_table(tt.purchase, tt.p_response.to_numpy()).round(3).to_string(index=False))
print("\ntop coefficients\n", p.coefficients(resp).head(8).round(3).to_string(index=False))
print("\nrare-outcome trap:", {k: round(v, 3) for k, v in p.threshold_demo(tt.purchase, tt.p_response.to_numpy()).items()})

# 2) uplift model
up = p.fit_uplift(train)
test["uplift"] = p.predict_uplift(up, test)
print("\nUPLIFT BY PREDICTED DECILE (10 = highest predicted uplift)\n", p.uplift_by_decile(test, "uplift").round(4).to_string(index=False))
print("\ncorrelation of predicted vs TRUE uplift (only possible on synthetic data):",
      round(test["uplift"].corr(test["_oracle_uplift"]), 3))

# 3) which score should decide whom to contact?
print("\nINCREMENTAL PURCHASES PER 1,000 CONTACTED, if we contact only the top share")
print(p.compare_targeting(test, {"rule_score": "score_device_financing", "response_model": "p_response", "uplift_model": "uplift"})
      .round(1).to_string(index=False))
for name, col in {"rule_score": "score_device_financing", "response_model": "p_response", "uplift_model": "uplift"}.items():
    print(f"AUUC {name}: {p.qini_auuc(test, col):+.5f}")
