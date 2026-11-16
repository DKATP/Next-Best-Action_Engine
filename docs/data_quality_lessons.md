# Data-quality patterns behind the gates

Patterns that recur in customer-scoring pipelines. Each one is tied to a control in this repo. Examples are generic; none come from a specific company's data.

| Pattern | What went wrong | Control here |
|---|---|---|
| **Silent change of meaning** | A new input version kept the same columns, but a flag's values meant the opposite ("N" now meant "does not own the product"). Eligibility for an offer fell for millions of lines with no error. | Benchmark the new version against the previous one (correlation per BU, root cause of every exclusion); confirm semantics with the data owner before changing code; make the mapping a config parameter. (`trace.compare_runs`) |
| **Join duplication** | Joining an exclusion list inflated the universe by a large margin. | `assert_same_rows` after every join; aggregate the lookup to the key first. |
| **Wrong join key** | A source keyed by account number matched 0% until the field was identified as a billing code, then 100%. Elsewhere, an account number collided across mobile lines and created false matches. | Test key hypotheses explicitly and report match rates; restrict joins where a key is unstable. |
| **Corrupt rows** | A header row arrived as data in a key column; type casts failed on malformed numeric strings. | Defensive casting, filters on key validity, and a documented count of what was dropped. |
| **Impossible values** | Corrupt dates produced ages of 250 and negative tenure. | Sanity ranges → NULL + flag, never silent use or silent drop. (`enrichment.add_features`) |
| **Stale sources** | One source's cut-off was months behind the others. | Freshness gate against the pinned snapshot date. (`quality.check_freshness`) |
| **Partial coverage** | A model scored only part of the population and missed a core segment even though keys matched. | Coverage gate per column and a coverage report per run; model outputs are "one signal among several", not the only one. |
| **Schema drift in a model output** | A probability column disappeared in the newest model table. | Schema validation and tolerant reads with an explicit fallback, logged. |
| **Duplicated lookup rows** | Every account appeared several times with different candidate offers. | Explicit, documented dedup rule (best score with a description), not "first row wins". |
| **Compressed or tied scores** | A segment lacked its main variables, so scores bunched up and a fixed threshold was unreachable; one BU had massive ties. | Prefer relative cut-offs (deciles) to fixed thresholds; tie-aware deciles; neutral footprint instead of invented signal. |
| **Mass-registration accounts** | A handful of identifiers held dozens of lines. | Eligibility rule on lines per person. (`universe.build_universe`) |

## What this repo does *not* do

It does not monitor model performance or drift after deployment, does not keep a formal model inventory by risk tier, and does not validate upstream models against real sales outcomes. Those are the next controls I would add (see the learning guide for the monitoring and validation exercises).
