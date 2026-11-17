# Design decisions

Each entry: the decision, why, and where it lives in the code. These are the choices I would defend in a design review.

### 1. The engine does not pick a winning business unit
Each BU gets its own score; the engine never says "this customer belongs to BU X". Offers compete later, in the distribution step, against **business quotas**. Picking a winner inside the scoring step hides a business trade-off in code and makes every BU's score depend on every other BU's weight. (`scoring.py`, `distribution.py`)

### 2. Score for every line; filters live in `applies_<bu>`
An early version multiplied the score by the input flag and the care decision, so a filter silently produced 0 and nobody could tell "not suitable" from "filtered". Now `score_<bu>` exists for all lines and `applies_<bu> = eligible × input flag × care`. The delivery validation fails if `applies=1` while the flag is 0. (`scoring.py`, `delivery.py`, tested)

### 3. Mark first, exclude later
Every line stays in the universe with `ineligible_reason`. The funnel is then reproducible and an exclusion can be questioned with numbers. (`universe.py`, `trace.funnel`)

### 4. Each list is joined by its own key
Line-level lists (phone) and person-level lists (customer id) are applied at their own level; person-level suppressions propagate to all the person's lines. Crossing everything by person excluded far too many lines; crossing everything by phone missed people who asked not to be contacted. (`universe.py`, tested)

### 5. Candidacy is per person, not per row
Whether to offer a home product depends on everything the *person* holds, so it is computed from a person profile and joined back. (`universe.person_profile`)

### 6. Rules are data, not code
Rules sit in a table with `lift`, `factor` and `source`; `points = (lift − 1) × factor`. A rule can be audited back to the analysis it came from, switched off, or changed without a deployment. In production this is an editable Delta table with change history. (`rules.py`, `config/rules.csv`)

### 7. Experts decide customer care; the engine's own rules are for comparison
Collections and retention teams own those signals. The engine uses their sources, treats missing data as "no alert" (never a guess), and keeps its own heuristic only as a column to compare against. This removed arguments about whose rule was right. (`care.py`)

### 8. Effective weight instead of penalizing missing data
`score = Σ(component × weight) / Σ(weights of available components)`. A line without consumption data is not punished for it, and a component with no rules or model gets a neutral footprint instead of invented signal. (`scoring.compute_scores`, tested)

### 9. Tie-aware deciles and deterministic tie-breaks
Some BUs have few distinct score values, so naive deciles would split identical customers into different deciles. Deciles keep ties together; ranking ties break on a hash of the key and the week, so reruns are identical and no customer is systematically favoured. (`scoring._decile`, `distribution._hash_rank`, tested)

### 10. Stop the run; never produce a quietly wrong output
Quality gates raise errors on coverage, schema, domain, freshness, duplicate keys and join-induced row-count changes. The data version is pinned at the start of a run (`snapshot_date`), not read as "today". (`quality.py`, `pipeline.py`)

### 11. Treatment and control come from the same ranked list
After the quota is filled, the selected lines are split at random, so both groups have the same score quality and a difference in sales can be attributed to the contact. `balance_check` verifies it. (`distribution.py`, `measurement.balance_check`)

### 12. An unvalidated judgement stays labelled as one
The boost for customers on a priority list (`upsell_priority_list_boost`) is a provisional business judgement, not a calibrated value. It lives in config with that comment, and the limitation is documented rather than hidden.

### 13. Gender is not a scoring variable in this repository
A fairness review is needed before using protected attributes such as gender (and, depending on jurisdiction, age) to decide who receives an offer. The public version does not use gender; `generation` is kept for demonstration and flagged for the same review.

### 14. Pandas here, PySpark in production
The original ran on Databricks with Delta tables. This version uses pandas so that anyone can run and test it without a cluster; the logic ports one-to-one (window functions for ranking, joins for lookups, Delta tables for rules and logs).

### 15. Lifetime value is modelled twice, because there are two businesses
Postpaid is contractual (churn is observed, lifetimes are right-censored) and prepaid is not (nobody cancels; "alive" is latent). Using one method for both is the standard mistake: survival analysis cannot be applied where there is no observed event, and a buy-till-you-die model throws away the cancellation signal where it exists. (`cltv.py`)

### 16. Proportional hazards is tested, not assumed
Cox rests on each covariate's effect being constant over time. The Schoenfeld-residual test is run on every covariate and reported alongside the fit, so a violation is visible rather than silently carried into the value estimate. (`cltv.check_proportional_hazards`, tested)

### 17. The CLTV horizon is capped, and the cap is stated
An uncapped expected lifetime is dominated by extrapolation far beyond the observed data. The horizon is a business decision (36 months by default) and appears in the output rather than buried in a constant. (`cltv.contractual_cltv`)

### 18. Value is a guardrail, not the ranking
The engine ranks on the expected incremental value of the contact, not on lifetime value. A high-value customer who would have bought anyway, or whose uplift is negative, is exactly who an untargeted campaign wastes budget on or actively annoys. (`cltv.expected_value_of_contact`)
