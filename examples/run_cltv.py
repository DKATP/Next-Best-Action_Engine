"""Phase 3 walk-through — customer lifetime value:  python examples/run_cltv.py"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from nba import cltv, synthetic  # noqa: E402

pd.set_option("display.width", 170)


def contractual_history(n_lines: int = 25_000, seed: int = 7) -> pd.DataFrame:
    lines = synthetic.make_lines(n_lines, seed)
    lines["n_lines_person"] = lines.groupby("person_id")["person_id"].transform("size")
    lines["generation"] = pd.cut(lines["age_years"], [0, 27, 43, 59, 200],
                                 labels=["GEN_Z", "MILLENNIAL", "GEN_X", "BOOMER"]).astype(object)
    return synthetic.make_contractual_history(lines, seed=seed + 24)


def main() -> None:
    print("=" * 78)
    print("CONTRACTUAL (postpaid) — churn is observed, lifetimes are censored")
    print("=" * 78)
    history = contractual_history()
    design = cltv.prepare_contractual(history)
    print(f"{len(design):,} customers | {design.churned.mean():.1%} churned, the rest are censored "
          f"(still active, final tenure unknown)")

    model = cltv.fit_cox(design)
    print(f"\nC-index: {cltv.concordance(model):.4f}   (0.5 = coin flip; measures ranking, not calibration)")
    print("\nHAZARD RATIOS\n", cltv.hazard_ratios(model).head(6).round(3).to_string(index=False))
    print("\nTrue log-hazard ratios used to generate the data:", synthetic.TRUE_SURVIVAL_COEF)

    ph = cltv.check_proportional_hazards(model, design)
    print(f"\nPROPORTIONAL HAZARDS (Schoenfeld): {(ph.assumption == 'VIOLATED').sum()} of {len(ph)} covariates violate it")
    print(ph.head(4).round(4).to_string(index=False))

    values = cltv.contractual_cltv(model, design, history["monthly_margin"], horizon_months=36)
    print(f"\nCLTV over 36 months, 10% annual discount:")
    print(values.describe().loc[["mean", "50%", "max"]].round(0).to_string())
    by_pay = history.assign(cltv=values["cltv"].to_numpy(),
                            tenure=values["expected_tenure_months"].to_numpy())
    print("\nBy payment behaviour:")
    print(by_pay.groupby(by_pay.payment_behavior.fillna("UNKNOWN"))
          .agg(customers=("cltv", "size"), expected_tenure=("tenure", "mean"), cltv=("cltv", "mean"))
          .round(1).to_string())

    print("\n" + "=" * 78)
    print("NON-CONTRACTUAL (prepaid) — nobody cancels, 'alive' has to be inferred")
    print("=" * 78)
    data = synthetic.make_noncontractual_history(6_000)
    bgnbd, gg = cltv.fit_bgnbd(data), cltv.fit_gamma_gamma(data)
    print(f"BG/NBD      fitted: { {k: round(float(bgnbd[k]), 3) for k in ('r','alpha','a','b')} }")
    print(f"            true  : {synthetic.TRUE_BGNBD}")
    print(f"Gamma-Gamma fitted: { {k: round(float(gg[k]), 3) for k in ('p','q','v')} }")
    print(f"            true  : {synthetic.TRUE_GAMMA_GAMMA}")
    print(f"frequency vs. spend correlation: {cltv.monetary_frequency_correlation(data):+.4f} "
          f"(Gamma-Gamma assumes this is near zero)")

    value = cltv.noncontractual_cltv(bgnbd, gg, data, periods=52.0)
    joined = data.join(value)
    joined["silent_weeks"] = pd.cut(joined["T"] - joined["recency"], [0, 10, 30, 60, 200])
    print("\nRepeat customers, by how long they have been silent:")
    print(joined[joined.frequency > 0].groupby("silent_weeks", observed=True)
          .agg(customers=("p_alive", "size"), p_alive=("p_alive", "mean"),
               expected_transactions=("expected_transactions", "mean"), cltv=("cltv", "mean"))
          .round(2).to_string())
    print("\nThe same history means different things over time: a customer who bought often and has")
    print("gone quiet is probably gone, and the model says so without anyone ever cancelling.")


if __name__ == "__main__":
    main()
