# Phase 3 learning guide: customer lifetime value

Goal: be able to build, validate and defend a CLTV model — and, more importantly, to explain *why the method changes with the business model*. That distinction is the first thing an interviewer probes, and the one most candidates miss.

Run first: `python examples/run_cltv.py`.

---

## 0. The question that decides everything: contractual or not?

| | **Contractual** (postpaid, subscriptions, insurance) | **Non-contractual** (prepaid, retail, e-commerce) |
|---|---|---|
| Churn | **Observed** — the customer cancels | **Never observed** — they just stop |
| Core problem | Most customers have not churned yet (censoring) | Telling "dormant" from "gone" |
| Method | Survival analysis (Kaplan-Meier, Cox) | BG/NBD + Gamma-Gamma, or Pareto/NBD |
| Output | Expected tenure × margin, discounted | P(alive) × expected transactions × expected spend |

**Say it.** *"The first question I ask is whether churn is observable. In postpaid it is, so it's a survival problem with censored data. In prepaid nobody cancels, so 'alive' is latent and the model has to infer it from frequency and recency."*

---

## 1. Censoring, and why averaging tenure is wrong
**Concept.** At any snapshot, most customers are still active. Their final tenure is unknown — only that it is *at least* their current tenure. That is **right-censoring**. Averaging the tenure of customers who already churned uses only the ones who left, so it is biased downward, often badly.
**Code.** `synthetic.make_contractual_history` generates lifetimes and censors them; `cltv.prepare_contractual`.
**Exercise.** Compute the mean tenure of churned customers only, then the expected tenure from the Cox model. Quantify the gap and explain where the bias comes from.

## 2. Kaplan-Meier before any model
**Concept.** A non-parametric survival curve per segment, handling censoring correctly. Always look at it first: it shows whether groups separate at all, and **crossing curves are a warning that proportional hazards will not hold**.
**Code.** `cltv.survival_by_segment`, `viz.chart_survival`.
**Exercise.** Plot by payment behaviour, then by tenure band. Do any curves cross?

## 3. Cox proportional hazards, and why it is the default
**Concept.** Semiparametric: it never assumes a shape for the baseline hazard, only that covariates shift it multiplicatively. `exp(coef)` is a **hazard ratio** — 2.38 means roughly 2.4x the instantaneous churn risk at any moment, all else equal.
**Code.** `cltv.fit_cox`, `cltv.hazard_ratios`.
**Exercise.** Compare the fitted hazard ratios with `synthetic.TRUE_SURVIVAL_COEF`. They should match within noise. Then explain to a non-technical manager what a hazard ratio of 0.67 means in plain words.
**Say it.** *"I prefer Cox over a black box here because every coefficient is a hazard ratio the business can argue with."*

## 4. The assumption test almost nobody runs
**Concept.** Cox assumes each covariate's effect is **constant over time**. Schoenfeld residuals test exactly that. A violation means the effect changes with tenure — a factor that matters in year one and stops mattering later. Fixes: stratify on that covariate, or add a time-varying term.
**Code.** `cltv.check_proportional_hazards`.
**Exercise.** Run it. It should pass here, because the data was *generated* from a proportional-hazards process — that is the point: a correctly wired test says "holds" when the assumption is true. Now break it deliberately: make one coefficient decay with time in the generator and confirm the test catches it.
**Say it.** *"Fitting Cox without testing proportional hazards is the most common error in applied survival analysis, so I check Schoenfeld residuals and stratify when it fails."*

## 5. C-index
**Concept.** The survival analogue of AUC: the share of comparable pairs ranked in the right order. 0.5 is a coin flip. It measures **ranking only** — a model can rank well and still predict tenure badly.
**Code.** `cltv.concordance`.
**Exercise.** Drop the strongest covariate and watch the C-index fall. How much signal does it carry on its own?

## 6. From survival to money
**Concept.** CLTV = Σ over the horizon of `margin × P(still a customer at t) ÷ (1+d)^t`. Two ingredients make it a lifetime value: survival weights each future month by the chance the customer is still there, and discounting prices money later as worth less than money now.
**Code.** `cltv.expected_tenure`, `cltv.contractual_cltv`, `cltv.monthly_discount`.
**Exercise.** Recompute at horizons of 12, 24, 36 and 60 months. Note how the ranking barely changes while the absolute value moves a lot — then argue why the **horizon is a business decision, not a statistical one**, and why an uncapped expectation is mostly extrapolation.

## 7. Non-contractual: two latent processes
**Concept.** BG/NBD models what you cannot see. While active, a customer transacts at a Poisson rate λ, drawn from Gamma(r, α) across the population. After each transaction they drop out with probability θ, drawn from Beta(a, b). Fitting recovers those four population parameters by maximum likelihood; **P(alive)** then follows for each individual.
**Code.** `cltv.fit_bgnbd`, `probability_alive`, `expected_transactions`.
**Exercise.** Compare the fitted parameters with `synthetic.TRUE_BGNBD` — recovering the generating parameters is the real test of an implementation. Then check that a customer with **frequency = 0** always gets P(alive) = 1, and explain why: with no repeat purchase there is no evidence of dropout to observe.
**Say it.** *"In prepaid I never see a cancellation, so I model the dropout process itself and score each customer's probability of still being alive."*

## 8. Gamma-Gamma, and its assumption
**Concept.** Spend per transaction, modelled separately from frequency. Its expected value is each customer's own average **shrunk toward the population mean**, weighted by how many transactions back it up — one large top-up barely moves the estimate, twenty do.
**Code.** `cltv.fit_gamma_gamma`, `expected_spend`, `monetary_frequency_correlation`.
**Exercise.** Verify the shrinkage directly: take one-transaction customers with extreme values and confirm their estimate sits between their own value and the population mean. Then check the independence assumption — if frequency and spend correlate strongly, Gamma-Gamma is the wrong tool, not merely imprecise.

## 9. Putting value into the decision
**Concept.** Three different questions, and they do not agree:
- **propensity** — will they buy?
- **uplift** — will they buy *because* we contacted them?
- **CLTV** — how much are they worth over time?

Ranking by CLTV alone contacts valuable customers who would have bought anyway. Ranking by propensity alone does the same. The economic ranking is `uplift × offer margin − contact cost`, with CLTV as the guardrail: a **high-value, negative-uplift customer is exactly the one an untargeted campaign destroys value on**.
**Code.** `cltv.expected_value_of_contact`, `cltv.value_segments`, `viz.chart_value_quadrant`.
**Exercise.** In this repo, value and uplift correlate only about +0.28. Find the high-value, low-uplift customers and write the one-sentence policy you would recommend for them.

---

## Next steps

1. **Competing risks:** churn to a competitor versus churn for non-payment are different events with different interventions.
2. **Time-varying covariates:** feed the engine's care flags as they change, rather than a snapshot.
3. **Accelerated failure time models** as an alternative to Cox when proportional hazards fails.
4. **Pareto/NBD** compared with BG/NBD, and when the dropout timing assumption matters.
5. **CLTV-aware quotas:** make the distribution step maximise expected incremental value under a contact budget.

## What you can and cannot claim

- ✅ "I built contractual and non-contractual CLTV models on synthetic data, validated by recovering the known generating parameters."
- ✅ "I test proportional hazards with Schoenfeld residuals and report the C-index, and I know what each one does and does not measure."
- ✅ "I can explain why prepaid needs BG/NBD and postpaid needs survival analysis."
- ❌ "I deployed CLTV in production." This phase was never run on real customers.
- ❌ Any real figure for customer value or tenure.
