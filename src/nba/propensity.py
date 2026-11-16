"""PHASE 2 (extension, synthetic data): propensity and uplift models with logistic regression.

Not part of the original delivered engine. It answers the question the engine could not: after scoring
with business rules, can a *learned* model do better at deciding who to contact, and who should
NOT be contacted? Everything here is trained and tested on synthetic data with a known hidden effect.

Three questions, three models:
  1. Response model  P(purchase | contacted, x)         -> "who buys when we contact them?"
  2. Uplift model    P(purchase | treated) - P(purchase | control)  -> "who buys *because* we contacted them?"
  3. Rule score      the phase-1 business-rule score as the baseline to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import synthetic
from .config import ROOT, load_config
from .pipeline import build_scored
from .rules import load_rules

NUMERIC = ["tenure", "age", "arpu_mobile", "device_financing_count", "cancel_intent_mobile", "potential_device_financing"]
CATEGORICAL = ["payment_behavior", "credit_limit_band", "handset_brand", "generation"]
FEATURES = NUMERIC + CATEGORICAL
# Anything known only AFTER the contact must never be a feature (target leakage).
FORBIDDEN = {"purchase", "group", "treated", "_oracle_uplift"}


def assert_no_leakage(features: list[str]) -> None:
    bad = FORBIDDEN & set(features)
    if bad:
        raise ValueError(f"leakage: {sorted(bad)} cannot be used as features")


def make_history(n_lines: int = 80_000, seed: int = 21) -> pd.DataFrame:
    """A past randomized campaign: eligible postpaid lines, random 50/50 treatment/control, simulated purchase."""
    cfg = load_config()
    rules = load_rules(ROOT / cfg["paths"]["rules"])
    scored = build_scored(synthetic.make_lines(n_lines, seed), cfg, rules)
    base = scored[(scored["line_type"] == "POSTPAID") & scored["is_eligible"] & (scored["care_collections"] == 1)].copy()
    base["potential_device_financing"] = base["potential_device_financing"].astype(int)
    return synthetic.simulate_purchase(base, seed=seed + 1).reset_index(drop=True)


def split_by_customer(df: pd.DataFrame, test_share: float = 0.3, salt: str = "split") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by PERSON, not by line: the same customer must never sit in both train and test."""
    h = pd.util.hash_pandas_object(df["person_id"].astype(str) + "|" + salt, index=False).to_numpy() % 1000
    is_test = h < int(test_share * 1000)
    return df[~is_test].copy(), df[is_test].copy()


def make_model(C: float = 1.0, class_weight: str | None = None) -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000))])


def fit(df: pd.DataFrame, C: float = 1.0, class_weight: str | None = None) -> Pipeline:
    assert_no_leakage(FEATURES)
    return make_model(C, class_weight).fit(df[FEATURES], df["purchase"])


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def coefficients(model: Pipeline) -> pd.DataFrame:
    """Interpretable output: sign and odds ratio per (scaled / one-hot) feature."""
    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------- evaluation: ranking, calibration, deciles ----------
def ranking_metrics(y: pd.Series, score: np.ndarray) -> dict:
    auc = roc_auc_score(y, score)
    pos, neg = np.sort(score[y == 1]), np.sort(score[y == 0])
    grid = np.unique(score)
    ks = np.max(np.abs(np.searchsorted(pos, grid, side="right") / len(pos) - np.searchsorted(neg, grid, side="right") / len(neg)))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(ks)}


def probability_metrics(y: pd.Series, p: np.ndarray) -> dict:
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p), "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


def calibration_table(y: pd.Series, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Do predicted probabilities match observed rates? Needed whenever the number itself (not just the rank) is used."""
    d = pd.DataFrame({"y": y.to_numpy(), "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False) + 1
    return d.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index()


def decile_table(y: pd.Series, score: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Decile 10 = highest scores. lift = decile rate / overall rate."""
    d = pd.DataFrame({"y": y.to_numpy(), "s": score})
    d["decile"] = pd.qcut(d["s"].rank(method="first"), n, labels=False) + 1
    t = d.groupby("decile").agg(n=("y", "size"), rate=("y", "mean")).reset_index()
    t["lift"] = t["rate"] / d["y"].mean()
    return t.sort_values("decile", ascending=False).reset_index(drop=True)


def threshold_demo(y: pd.Series, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Why accuracy and a 0.5 cut-off mislead with rare outcomes.

    With a low base rate, a model can predict 'no purchase' for almost everyone, look 'accurate',
    and still rank customers well (high AUC). Judge ranking by deciles/lift and calibration, not by a hard class.
    """
    pred = (p >= threshold).astype(int)
    return {"base_rate": float(y.mean()), "share_predicted_positive": float(pred.mean()),
            "accuracy": float((pred == y.to_numpy()).mean()), "accuracy_of_predicting_all_zero": float(1 - y.mean()),
            "auc": roc_auc_score(y, p)}


# ---------- uplift (two-model / T-learner) ----------
def fit_uplift(train: pd.DataFrame, C: float = 1.0) -> dict:
    """Two separate response models, one on treated and one on control; uplift = difference of predictions."""
    return {"treated": fit(train[train["group"] == "TREATMENT"], C), "control": fit(train[train["group"] == "CONTROL"], C)}


def predict_uplift(models: dict, df: pd.DataFrame) -> np.ndarray:
    return predict_proba(models["treated"], df) - predict_proba(models["control"], df)


def incremental_rate(df: pd.DataFrame) -> float:
    t, c = df[df["group"] == "TREATMENT"]["purchase"], df[df["group"] == "CONTROL"]["purchase"]
    return float(t.mean() - c.mean()) if len(t) and len(c) else float("nan")


def uplift_by_decile(df: pd.DataFrame, score_col: str, n: int = 10) -> pd.DataFrame:
    """Decile 10 = highest predicted score. A good uplift model shows the incremental rate falling with the decile."""
    d = df.copy()
    d["decile"] = pd.qcut(d[score_col].rank(method="first"), n, labels=False) + 1
    rows = []
    for dec, g in d.groupby("decile"):
        rows.append({"decile": dec, "n": len(g),
                     "rate_treatment": g[g.group == "TREATMENT"]["purchase"].mean(),
                     "rate_control": g[g.group == "CONTROL"]["purchase"].mean(), "incremental": incremental_rate(g)})
    return pd.DataFrame(rows).sort_values("decile", ascending=False).reset_index(drop=True)


def compare_targeting(test: pd.DataFrame, score_cols: dict[str, str], fractions=(0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """Incremental purchases per 1,000 customers contacted, if we contact only the top k% by each score.

    Measured on a held-out randomized test, so the treatment/control difference is a causal estimate.
    """
    rows = []
    for k in fractions:
        row = {"top_share": k, "everyone": incremental_rate(test) * 1000}
        for name, col in score_cols.items():
            top = test.nlargest(int(len(test) * k), col)
            row[name] = incremental_rate(top) * 1000
        rows.append(row)
    return pd.DataFrame(rows)


def qini_auuc(test: pd.DataFrame, score_col: str, steps: int = 20) -> float:
    """Area between the cumulative-incremental curve of a score and the random-targeting line (higher is better)."""
    d = test.sort_values(score_col, ascending=False).reset_index(drop=True)
    xs = np.linspace(1 / steps, 1, steps)
    cum = np.array([incremental_rate(d.head(int(len(d) * x))) * x for x in xs])
    rand = incremental_rate(d) * xs
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(cum - rand, xs))


# ---------- model selection done properly ----------
from sklearn.calibration import CalibratedClassifierCV          # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier     # noqa: E402
from sklearn.inspection import permutation_importance           # noqa: E402
from sklearn.model_selection import GroupKFold                  # noqa: E402


def cross_validate_auc(df: pd.DataFrame, model_factory=make_model, n_splits: int = 5, **kwargs) -> pd.DataFrame:
    """Cross-validated AUC, grouped BY CUSTOMER so no person appears in both folds.

    A plain KFold here would leak: one customer's several lines would land on both sides
    and the score would come out flattering and wrong.
    """
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(df, df["purchase"], groups=df["person_id"]), start=1):
        train, test = df.iloc[tr], df.iloc[te]
        model = model_factory(**kwargs).fit(train[FEATURES], train["purchase"])
        p = model.predict_proba(test[FEATURES])[:, 1]
        rows.append({"fold": fold, "n_test": len(test), "auc": roc_auc_score(test["purchase"], p)})
    out = pd.DataFrame(rows)
    out.attrs["mean_auc"], out.attrs["std_auc"] = out["auc"].mean(), out["auc"].std()
    return out


def make_boosted_model(max_iter: int = 200, learning_rate: float = 0.06, max_depth: int | None = 4,
                       random_state: int = 0) -> Pipeline:
    """Gradient boosting as the challenger. It handles missing values and interactions natively,
    so it needs no imputation; the price is that coefficients are gone."""
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", "passthrough", NUMERIC), ("cat", cat, CATEGORICAL)])
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate,
                                         max_depth=max_depth, random_state=random_state)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_calibrated(df: pd.DataFrame, method: str = "isotonic", cv: int = 3, boosted: bool = False) -> Pipeline:
    """Wrap a model so its probabilities are corrected against held-out folds.

    Needed whenever the NUMBER is used, not just the ranking: expected revenue, budgets,
    or a cut-off expressed as 'contact everyone above 10%'.
    """
    base = make_boosted_model() if boosted else make_model()
    return CalibratedClassifierCV(base, method=method, cv=cv).fit(df[FEATURES], df["purchase"])


def compare_models(train: pd.DataFrame, test: pd.DataFrame, baseline_col: str = "score_device_financing") -> pd.DataFrame:
    """Logistic vs. gradient boosting vs. the production rule score, on the same held-out test.

    Always include the thing already in production as a row. A model that cannot beat the
    rules it would replace is not worth its operational cost.
    """
    assert_no_leakage(FEATURES)
    y = test["purchase"]
    # (score, is_a_probability) -- the rule score lives on 0-1 but is NOT a probability,
    # so Brier and log-loss are meaningless for it and are left blank rather than printed.
    candidates = {
        "rule_score (baseline)": (test[baseline_col].to_numpy(), False),
        "logistic": (predict_proba(fit(train), test), True),
        "logistic (isotonic)": (fit_calibrated(train).predict_proba(test[FEATURES])[:, 1], True),
        "gradient_boosting": (make_boosted_model().fit(train[FEATURES], train["purchase"]).predict_proba(test[FEATURES])[:, 1], True),
    }
    rows = []
    for name, (score, is_probability) in candidates.items():
        row = {"model": name, **ranking_metrics(y, score)}
        if is_probability:
            row.update(probability_metrics(y, score))
        top = decile_table(y, score).iloc[0]
        row["lift_top_decile"] = top["lift"]
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def feature_importance(model: Pipeline, test: pd.DataFrame, n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Permutation importance: how much AUC falls when one feature is shuffled.

    Model-agnostic, measured on held-out data, and it answers the question a coefficient
    cannot for a non-linear model: does this variable actually carry the signal?
    """
    r = permutation_importance(model, test[FEATURES], test["purchase"], scoring="roc_auc",
                               n_repeats=n_repeats, random_state=seed)
    out = pd.DataFrame({"feature": FEATURES, "auc_drop": r.importances_mean, "std": r.importances_std})
    return out.sort_values("auc_drop", ascending=False).reset_index(drop=True)
