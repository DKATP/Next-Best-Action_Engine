# Phase 2 learning guide: from rules to learned models

Goal: be able to *build, evaluate and explain* a propensity and an uplift model for customer-knowledge problems — and to say honestly what the evidence does and does not show. Each block has the concept, where it lives in the code, an exercise and a sentence you can say in an interview.

Run first: `python examples/run_phase2.py`, then work through the blocks in order.

---

## 1. Logistic regression, from log-odds to odds ratios
**Concept.** The model learns `log(p / (1 − p)) = b0 + b1·x1 + …`. A coefficient is the change in log-odds per unit of a feature; `exp(coef)` is the odds ratio. With scaled numeric features, coefficients are comparable in size.
**Code.** `propensity.make_model`, `propensity.coefficients`.
**Exercise.** Which three variables raise the odds of purchase most? Compare with the lifts in `config/rules.csv`. Where do the learned odds ratios and the hand-written rule lifts disagree, and which would you trust?
**Say it.** *"I use logistic regression first because the coefficients are auditable by the business — I can show the odds ratio of each driver."*

## 2. Preprocessing is part of the model
**Concept.** Median imputation with a missing-value indicator, scaling for numerics, one-hot with `handle_unknown="ignore"` for categoricals — all inside one `Pipeline`, fitted on training data only.
**Exercise.** Remove the missing indicator and compare AUC. Fit the scaler on the full data and explain why that is leakage.

## 3. Regularization (`C`)
**Concept.** Smaller `C` = stronger L2 penalty = smaller coefficients, less overfitting.
**Exercise.** Loop `C` over `[0.01, 0.1, 1, 10]` using `propensity.fit(..., C=c)`; plot test AUC and Brier score. Then train on 2,000 rows instead of 25,000 and see when regularization starts to matter.

## 4. Rare outcomes, thresholds and the "predicts almost always zero" trap
**Concept.** With a low purchase rate, a model can predict "no purchase" for nearly everyone, show high accuracy, and still be useful for *ranking*. Accuracy at 0.5 is the wrong yardstick; use AUC/Gini/KS, lift by decile and calibration.
**Code.** `propensity.threshold_demo`, `propensity.decile_table`.
**Exercise.** Run `threshold_demo` and compare `accuracy` with `accuracy_of_predicting_all_zero`. Then try `class_weight="balanced"`: what happens to calibration (`calibration_table`) and why?
**Say it.** *"When a propensity model's output is binary and mostly negative, accuracy hides the problem. I look at coverage, ties, ranking and calibration, escalate with evidence, and use the model as one signal among several until it is validated against real outcomes."* (Only say what matches your own work: in the original engine you reviewed upstream models' data and outputs, you did not validate them against real sales.)

## 5. Evaluation: ranking, calibration, lift
**Concept.** Ranking quality (AUC, Gini = 2·AUC − 1, KS) answers *who first*. Calibration answers *is "12%" really 12%*. You need calibration whenever the number itself is used (budgets, expected revenue).
**Code.** `ranking_metrics`, `probability_metrics`, `calibration_table`, `decile_table`.
**Exercise.** Add isotonic or Platt calibration (`CalibratedClassifierCV`) and show before/after calibration tables.

## 6. Avoiding leakage and fooling yourself
**Concept.** Features must be known *before* the contact. Split by customer, never by line. Never tune on the test set.
**Code.** `assert_no_leakage`, `split_by_customer`.
**Exercise.** Deliberately add `purchase` as a feature and watch AUC jump to ~1; add a "days since purchase" feature and explain why it is subtler leakage. Split by line instead of customer and see how AUC changes.

## 7. Response is not impact: uplift
**Concept.** The customers most likely to buy are often the ones who would have bought anyway. What matters is the *incremental* effect of the contact: P(buy | contacted) − P(buy | not contacted). Four groups: persuadables, sure things, lost causes, and *sleeping dogs* (contact reduces purchase).
**Code.** `fit_uplift` (two-model approach), `uplift_by_decile`, `compare_targeting`, `qini_auuc`.
**Exercise.** In `synthetic.simulate_purchase`, the treatment effect is negative for customers with repeated cancellation intent. Check whether the uplift model's lowest deciles contain them. Then change the hidden effect and see whether the model recovers it (`_oracle_uplift` exists only because the data is synthetic).
**Say it.** *"A response model tells me who buys; an uplift model tells me who buys because we contacted them. I validate it on a randomized holdout."*

## 8. Model selection against the incumbent
**Concept.** Always compare a candidate with the thing already in production (`rule_score`) *and* with "contact everyone". Cross-validate grouped by customer. A model that cannot beat the rules it would replace is not worth its operational cost.
**Code.** `cross_validate_auc`, `compare_models`, `make_boosted_model`, `fit_calibrated`, `feature_importance`.
**Exercise.** Run `compare_models`. Gradient boosting does **not** beat logistic regression on this data. Explain to a non-technical manager why you would still ship the logistic model. Then run `feature_importance` and check whether the variables it ranks first are the ones the hidden data-generating function in `synthetic.simulate_purchase` actually uses.
**Say it.** *"I benchmark every candidate against the rules already in production, cross-validated by customer. Here the boosted model matched the logistic one, so I'd ship the interpretable one."*

## 9. Before you launch: sample size and power
**Concept.** A test you cannot power is a test whose "no effect" result means nothing. Decide the sample before launch, not after.
**Code.** `measurement.required_sample_size`, `minimum_detectable_effect`, `sample_size_curve`.
**Exercise.** Run `sample_size_curve(0.05)`. Notice that halving the effect you want to detect roughly **quadruples** the sample. Then run `minimum_detectable_effect(0.06, 300)`: with a quota of 300 per group you can only see a lift of about 110%. Work out what that means for a campaign whose baseline conversion is a fraction of a percent — those tests need hundreds of thousands of customers per arm.
**Say it.** *"Before launching I compute the minimum detectable effect for the quota the business gives me, so we know up front what the test can and cannot prove."*

## 10. After you launch: drift
**Concept.** A pipeline can pass every input gate and still decay, because the population it scores this month is not the population it scored last month. PSI compares a variable's distribution between a reference run and the current one: under 0.10 stable, 0.10–0.25 investigate, above 0.25 a significant shift.
**Code.** `monitoring.psi_numeric`, `psi_categorical`, `psi_report`, `score_drift`.
**Exercise.** Generate two runs with different seeds and run `psi_report`. Then deliberately shift one feature and confirm the report catches it. Note why the bin edges must be fixed on the **reference** run, and why a high PSI is an alarm rather than a verdict: it says a distribution moved, not that performance dropped.
**Say it.** *"PSI tells me the population moved. It never replaces measuring outcomes."*

---

## Next steps (in order of value)

1. **Uncertainty on the targeting table.** Bootstrap the test set 200 times and put an interval on "incremental purchases per 1,000 contacted" for each score. Which differences survive?
2. **Other uplift learners:** S-learner, X-learner and a causal forest, compared with the two-model approach used here.
3. **Experiment tracking:** log runs, parameters and metrics with MLflow; register the chosen model.
4. **Port to Spark:** move the pipeline to PySpark/Delta (rules table, run log) and train with `pyspark.ml`.
5. **A GenAI layer:** use an LLM to *explain* a score to a business user from the odds ratios and the care decision, with the deterministic score staying the source of truth.
6. **Fairness review:** audit score distributions by `generation`; document the decision about protected attributes.

## What you can and cannot claim

- ✅ "I built a propensity and uplift pipeline on synthetic data to practise the method and the evaluation discipline."
- ✅ "On synthetic data with a known effect, the uplift model recovered the hidden heterogeneity — correlation with the true uplift around 0.84."
- ✅ "I benchmarked a boosted model against logistic regression and the production rules, and chose the interpretable model because the gain did not justify the cost."
- ❌ "My model improved campaign results." Phase 2 was never run on real customers.
- ❌ Any statement about real effect sizes from these tables.
