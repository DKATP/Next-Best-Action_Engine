"""End-to-end run on synthetic data:  python -m nba.pipeline"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import care, delivery, distribution, enrichment, measurement, quality, rules, scoring, synthetic, trace, universe
from .config import ROOT, load_config


def build_scored(lines: pd.DataFrame, cfg: dict, rule_table: pd.DataFrame, log: trace.RunLog | None = None) -> pd.DataFrame:
    """Lines -> universe -> features -> care -> flags -> scores. Reused by the phase-2 notebook/module."""
    log = log or trace.RunLog("adhoc")
    n = len(lines)
    uni = universe.build_universe(lines, cfg["universe"]["mass_registration_min_lines"])
    log.add("01_universe", len(uni))
    enr = enrichment.add_features(uni, uni["n_lines_person"])
    quality.assert_same_rows(n, len(enr), "enrichment")
    log.add("02_enrichment", len(enr))
    cared = care.apply_care(enr, cfg)
    log.add("03_care", len(cared))
    flagged = scoring.build_flags(cared, cfg)
    scored = scoring.compute_scores(flagged, cfg, rule_table)
    quality.assert_same_rows(n, len(scored), "scoring")
    log.add("04_scoring", len(scored))
    return scored


def run(n_lines: int = 20_000, seed: int = 7, out_dir: str | Path | None = None, config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    snap, week = cfg["run"]["snapshot_date"], cfg["run"]["week"]
    run_id = f"NBA_{snap.replace('-', '')}_{week}"
    log = trace.RunLog(run_id)

    lines = synthetic.make_lines(n_lines, seed)
    quality.validate_schema(lines, ["msisdn", "person_id", "line_type", "tenure_months", "age_years"], "lines")
    quality.validate_domain(lines, "line_type", {"POSTPAID", "PREPAID", "HOME"}, "lines")
    quality.assert_unique(lines, "msisdn", "lines")
    coverage_report = quality.validate_coverage(lines, cfg["quality"]["min_coverage"], "lines")
    freshness_report = quality.check_freshness(synthetic.source_loaded_at(snap), snap, cfg["quality"]["max_source_age_days"])
    log.add("00_quality_gates", len(lines))

    rule_table = rules.load_rules(ROOT / cfg["paths"]["rules"])
    scored = build_scored(lines, cfg, rule_table, log)
    deliv = delivery.build_delivery(scored, run_id)
    delivery.validate_delivery(deliv, len(lines))
    log.add("05_delivery", len(deliv))

    quotas = distribution.load_quotas(ROOT / cfg["paths"]["quotas"])
    dist = distribution.distribute(deliv, quotas, week, cfg)
    log.add("07_distribution", len(dist))

    # Simulated campaign outcome (synthetic, known hidden effect) to exercise the measurement code
    feats = scored.set_index("msisdn")
    dist = dist.join(feats[["tenure_band", "credit_limit_band", "handset_brand", "arpu_mobile", "potential_device_financing",
                            "payment_behavior", "cancel_intent_mobile", "generation"]], on="msisdn")
    outcomes = synthetic.simulate_purchase(dist, seed=seed + 1, group_col="group")
    result = {"run_id": run_id, "scored": scored, "delivery": deliv, "distribution": dist, "outcomes": outcomes,
              "funnel": trace.funnel(scored), "reasons": trace.ineligibility_reasons(scored),
              "coverage": coverage_report, "freshness": freshness_report, "log": log.frame(), "rules": rule_table}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ["delivery", "distribution", "funnel", "reasons", "coverage", "freshness", "log"]:
            result[name].to_csv(out / f"{name}.csv", index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the NBA pipeline on synthetic data")
    ap.add_argument("--lines", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()
    res = run(a.lines, a.seed, a.out)
    print(f"Run {res['run_id']} finished. Outputs written to {a.out}\n")
    print("FUNNEL\n", res["funnel"].to_string(index=False))
    print("\nWHY LINES ARE INELIGIBLE\n", res["reasons"].to_string(index=False))
    d = res["distribution"]
    print("\nDISTRIBUTION\n", d.groupby(["offer", "group"]).agg(lines=("msisdn", "size"), mean_score=("score", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
