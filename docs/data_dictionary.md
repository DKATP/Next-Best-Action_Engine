# Delivery data dictionary

Column order is a contract (`delivery.CONTRACT`): identifiers, scores, care and product columns, then `applies_*`, then `flag_*` last.

| Column | Type | Meaning |
|---|---|---|
| `msisdn` | text | Synthetic line identifier (`999…`, deliberately invalid). Unique in the delivery. |
| `person_id` | text | Synthetic customer identifier. |
| `run_id` | text | `NBA_<snapshot>_<week>`; ties every row to a run and a data version. |
| `is_eligible` | bool | Passed all exclusions (active, line list, person list, mass registration). |
| `ineligible_reason` | text | Why not eligible; empty if eligible. |
| `selected` | bool | Empty until the distribution step fills it. |
| `score_<bu>` | decimal | 0.01–1.0 for **every** line, for `device_financing`, `upselling`, `home`, `postpaid`, `prepaid`. |
| `decile_<bu>` | decimal | 0.1 (lowest 10%) to 1.0 (highest 10%) among lines where the BU applies; 0 = does not apply. Tie-aware. |
| `care_action` | text | `NORMAL_OFFER`, `NO_CREDIT` (collections alert), `RETENTION_FIRST` (retention alert), `NO_SALE` (both). |
| `care_stage` | text | `RISK`, `COLLECTIONS`, `ONBOARDING` (< 6 months), `DEVELOPMENT`. |
| `care_collections` / `care_retention` | int | 1 = no alert, 0 = alert. No expert data = 1. |
| `collections_source` / `retention_source` | text | Who decided: expert team, model, or `NO_*_DATA`. |
| `line_type` | text | `POSTPAID`, `PREPAID`, `HOME`. |
| `segment` | text | Management segment per person: `HOME` includes the person's postpaid lines. |
| `home_product` | text | `HOME_SALE`, `MOBILE_SALE_IN_HOME_BASE`, `LOYALTY`; empty if the home BU does not apply. |
| `upsell_product` / `upsell_category` | text | Suggested upsell by priority (Prime > Netflix > Disney > Sports > Max > Adult > other) and its category. |
| `n_bu_apply` | int | How many BUs apply to the line. |
| `applies_<bu>` | int | 1 = eligible × input flag × customer care. This is the audience filter. |
| `flag_<bu>` | int | 1 = the BU applies according to the weekly input (× eligibility). |

## Distribution output

`msisdn, person_id, segment, offer, quota, score, position, n_candidate_offers, final_position, group, week` — `group` is `TREATMENT` or `CONTROL`, `n_candidate_offers` shows how many offers wanted the line before it was assigned to one.
