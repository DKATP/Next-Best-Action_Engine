"""PHASE 3 (extension, synthetic data): customer lifetime value.

The engine decides *what* to offer. This module answers *how much the customer is worth*,
which is what turns a ranking into an economic decision.

**The distinction that governs everything here is contractual vs. non-contractual**, and it
is the question an interviewer will probe first:

- **Postpaid is contractual.** The customer tells you when they leave, so churn is observed
  and lifetimes are *right-censored*: most customers are still active, so their final tenure
  is unknown. Averaging the tenure of customers who already churned is badly biased — it
  throws away everyone still with you. Survival analysis exists for exactly this.
  Cox proportional hazards is the right default: semiparametric (no assumed baseline shape)
  and its coefficients read as hazard ratios a business can argue with.

- **Prepaid is non-contractual.** Nobody announces they have left; they simply stop topping
  up. "Churn" is never observed and has to be *inferred* from frequency and recency. Cox does
  not apply. BG/NBD models the two latent processes — how often a customer transacts while
  active, and the chance they drop out after each transaction — and Gamma-Gamma models spend.

Everything is fitted on synthetic data whose true parameters are known (`synthetic.TRUE_*`),
so the estimators can be checked against the truth rather than merely looking plausible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

CONTRACTUAL_FEATURES = ["payment_behavior", "generation", "log_arpu", "has_cancel_intent", "has_extra_lines"]


def monthly_discount(annual_rate: float = 0.10) -> float:
    """An annual discount rate expressed per month. Compounding, not division by twelve."""
    return (1 + annual_rate) ** (1 / 12) - 1


# ============================================================ contractual (postpaid)
def prepare_contractual(history: pd.DataFrame) -> pd.DataFrame:
    """Model matrix for the Cox fit: duration, event, and the covariates, one-hot encoded."""
    df = history.copy()
    df["log_arpu"] = np.log(df["arpu_mobile"].fillna(df["arpu_mobile"].median()) / 52_000).clip(-1.5, 1.5)
    df["has_cancel_intent"] = (df["cancel_intent_mobile"] >= 1).astype(int)
    df["has_extra_lines"] = (df["n_lines_person"] >= 2).astype(int) if "n_lines_person" in df else 0
    df["payment_behavior"] = df["payment_behavior"].fillna("UNKNOWN")
    df["generation"] = df["generation"].fillna("UNKNOWN") if "generation" in df else "UNKNOWN"
    design = pd.get_dummies(df[CONTRACTUAL_FEATURES], columns=["payment_behavior", "generation"],
                            drop_first=True, dtype=float)
    design["duration_months"] = df["duration_months"].to_numpy()
    design["churned"] = df["churned"].to_numpy()
    return design


def survival_by_segment(history: pd.DataFrame, by: str, min_size: int = 200) -> pd.DataFrame:
    """Kaplan-Meier survival curves per segment — the model-free look that comes first.

    KM is non-parametric and handles censoring correctly. Always plot it before fitting
    anything: it shows whether the groups separate at all, and whether their curves cross
    (a crossing is a warning that proportional hazards will not hold).
    """
    out = []
    for segment, group in history.groupby(by):
        if len(group) < min_size:
            continue
        km = KaplanMeierFitter().fit(group["duration_months"], group["churned"], label=str(segment))
        curve = km.survival_function_.reset_index()
        curve.columns = ["month", "survival"]
        curve["segment"] = str(segment)
        curve["n"] = len(group)
        out.append(curve)
    return pd.concat(out, ignore_index=True)


def fit_cox(design: pd.DataFrame, penalizer: float = 0.01) -> CoxPHFitter:
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(design, duration_col="duration_months", event_col="churned")
    return model


def hazard_ratios(model: CoxPHFitter) -> pd.DataFrame:
    """Coefficients as hazard ratios. Above 1 = churns sooner; below 1 = stays longer.

    This is why Cox is the default over a black box here: every number is arguable by a
    business owner who knows the customers.
    """
    s = model.summary
    out = pd.DataFrame({"covariate": s.index, "hazard_ratio": s["exp(coef)"].to_numpy(),
                        "ci_low": s["exp(coef) lower 95%"].to_numpy(), "ci_high": s["exp(coef) upper 95%"].to_numpy(),
                        "p_value": s["p"].to_numpy()})
    out["direction"] = np.where(out["hazard_ratio"] > 1, "churns sooner", "stays longer")
    return out.reindex(out["hazard_ratio"].sub(1).abs().sort_values(ascending=False).index).reset_index(drop=True)


def check_proportional_hazards(model: CoxPHFitter, design: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Test the assumption the whole model rests on, using Schoenfeld residuals.

    Cox assumes a covariate's effect is CONSTANT over time. If a variable fails, its effect
    changes with tenure — for example, a risk factor that matters in the first year and stops
    mattering later. The usual fixes are stratifying on that variable or adding a
    time-varying term. Fitting Cox without ever running this test is the most common error
    in applied survival analysis.
    """
    summary = proportional_hazard_test(model, design, time_transform="rank").summary
    out = pd.DataFrame({"covariate": summary.index.astype(str),
                        "test_statistic": summary["test_statistic"].to_numpy(),
                        "p_value": summary["p"].to_numpy()})
    out["assumption"] = np.where(out["p_value"] < alpha, "VIOLATED", "holds")
    return out.sort_values("p_value").reset_index(drop=True)


def concordance(model: CoxPHFitter) -> float:
    """C-index: the share of comparable pairs the model ranks in the right order.

    The survival analogue of AUC. 0.5 is a coin flip; 0.65-0.75 is typical for churn with
    behavioural covariates. It measures RANKING, not whether the predicted tenure is right.
    """
    return float(model.concordance_index_)


def expected_tenure(model: CoxPHFitter, design: pd.DataFrame, horizon_months: int = 36) -> pd.Series:
    """Expected remaining months within the horizon: the area under each survival curve.

    The horizon is a business decision, not a statistical one. An unbounded expectation is
    dominated by extrapolation far beyond the data, so it is capped — and the cap is stated.
    """
    covariates = design.drop(columns=["duration_months", "churned"])
    times = np.arange(1, horizon_months + 1)
    curves = model.predict_survival_function(covariates, times=times)
    return pd.Series(curves.to_numpy().sum(axis=0), index=design.index, name="expected_tenure_months")


def contractual_cltv(model: CoxPHFitter, design: pd.DataFrame, monthly_margin: pd.Series,
                     horizon_months: int = 36, annual_discount: float = 0.10) -> pd.DataFrame:
    """CLTV = sum over the horizon of margin x P(still a customer) / (1 + d)^t.

    Two things make this a lifetime value rather than an annualised guess: survival weights
    each future month by the chance the customer is still there, and discounting prices the
    fact that money later is worth less than money now.
    """
    covariates = design.drop(columns=["duration_months", "churned"])
    times = np.arange(1, horizon_months + 1)
    survival = model.predict_survival_function(covariates, times=times).to_numpy()      # months x customers
    discount = 1 / (1 + monthly_discount(annual_discount)) ** times
    margin = np.asarray(monthly_margin, dtype=float)
    cltv = (survival * discount[:, None]).sum(axis=0) * margin
    return pd.DataFrame({"expected_tenure_months": survival.sum(axis=0).round(2),
                         "monthly_margin": margin.round(0),
                         "cltv": cltv.round(0)}, index=design.index)


# ============================================================ non-contractual (prepaid)
def _bgnbd_neg_ll(log_params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> float:
    r, alpha, a, b = np.exp(log_params)
    ll = (gammaln(r + x) - gammaln(r) + r * np.log(alpha)
          + gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x))
    term_still_alive = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        term_dropped = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-9)) - (r + x) * np.log(alpha + t_x),
                                -np.inf)
    ll = ll + np.logaddexp(term_still_alive, term_dropped)
    return -np.sum(ll) if np.isfinite(ll).all() else 1e10


def fit_bgnbd(data: pd.DataFrame, start=(0.5, 6.0, 1.0, 3.0)) -> dict:
    """Fit the BG/NBD transaction-and-dropout process by maximum likelihood.

    Four population parameters: transactions while active are Poisson with a rate drawn from
    Gamma(r, alpha); after each transaction the customer drops out with a probability drawn
    from Beta(a, b). Fitted on log-parameters so they stay positive.
    """
    x = data["frequency"].to_numpy(float)
    t_x = data["recency"].to_numpy(float)
    T = data["T"].to_numpy(float)
    fit = minimize(_bgnbd_neg_ll, np.log(start), args=(x, t_x, T), method="Nelder-Mead",
                   options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-6})
    r, alpha, a, b = np.exp(fit.x)
    return {"r": r, "alpha": alpha, "a": a, "b": b, "log_likelihood": -fit.fun, "converged": bool(fit.success)}


def probability_alive(params: dict, data: pd.DataFrame) -> pd.Series:
    """P(the customer has not silently dropped out), given frequency, recency and observation time.

    This is the quantity that does not exist in a contractual world. A customer who used to
    buy often and has been quiet for a long time gets a low probability; one who bought once
    a long time ago is barely informative and stays near the population average.
    """
    r, alpha, a, b = params["r"], params["alpha"], params["a"], params["b"]
    x, t_x, T = data["frequency"].to_numpy(float), data["recency"].to_numpy(float), data["T"].to_numpy(float)
    ratio = np.where(x > 0, (a / np.maximum(b + x - 1, 1e-9)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return pd.Series(1 / (1 + ratio), index=data.index, name="p_alive")


def expected_transactions(params: dict, data: pd.DataFrame, periods: float) -> pd.Series:
    """Expected transactions over the next `periods`, conditional on the customer's history."""
    r, alpha, a, b = params["r"], params["alpha"], params["a"], params["b"]
    x, t_x, T = data["frequency"].to_numpy(float), data["recency"].to_numpy(float), data["T"].to_numpy(float)
    # (a - 1) is negative whenever a < 1, which is common. So is the bracket that follows, and
    # the two negatives cancel: the formula is correct as written and must NOT be "guarded".
    denominator_a = a - 1 if abs(a - 1) > 1e-9 else 1e-9
    first = (a + b + x - 1) / denominator_a
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, periods / (alpha + T + periods))
    numerator = first * (1 - ((alpha + T) / (alpha + T + periods)) ** (r + x) * hyp)
    ratio = np.where(x > 0, (a / np.maximum(b + x - 1, 1e-9)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return pd.Series(numerator / (1 + ratio), index=data.index, name="expected_transactions")


def _gamma_gamma_neg_ll(log_params: np.ndarray, x: np.ndarray, m: np.ndarray) -> float:
    p, q, v = np.exp(log_params)
    ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
          + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(v + m * x))
    return -np.sum(ll) if np.isfinite(ll).all() else 1e10


def fit_gamma_gamma(data: pd.DataFrame, start=(4.0, 3.0, 10.0)) -> dict:
    """Fit spend per transaction, on repeat customers only.

    Gamma-Gamma assumes spend is independent of transaction frequency. Check that assumption
    with `monetary_frequency_correlation` before trusting the result; a strong correlation
    means the model is the wrong tool, not that the fit is merely imprecise.
    """
    repeat = data[(data["frequency"] > 0) & (data["monetary_value"] > 0)]
    fit = minimize(_gamma_gamma_neg_ll, np.log(start),
                   args=(repeat["frequency"].to_numpy(float), repeat["monetary_value"].to_numpy(float)),
                   method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-6})
    p, q, v = np.exp(fit.x)
    return {"p": p, "q": q, "v": v, "log_likelihood": -fit.fun, "converged": bool(fit.success),
            "n_repeat_customers": len(repeat)}


def monetary_frequency_correlation(data: pd.DataFrame) -> float:
    """The assumption check for Gamma-Gamma. Near zero is what the model needs."""
    repeat = data[(data["frequency"] > 0) & (data["monetary_value"] > 0)]
    return float(repeat["frequency"].corr(repeat["monetary_value"]))


def expected_spend(params: dict, data: pd.DataFrame) -> pd.Series:
    """Expected spend per transaction: the customer's own average shrunk toward the population.

    A customer with one transaction barely moves off the population mean; one with twenty is
    trusted almost entirely. That shrinkage is the whole point — it stops a single large
    top-up from being read as a high-value customer.
    """
    p, q, v = params["p"], params["q"], params["v"]
    x, m = data["frequency"].to_numpy(float), data["monetary_value"].to_numpy(float)
    weight = np.where(x > 0, p * x / (p * x + q - 1), 0.0)
    population_mean = v * p / (q - 1)
    return pd.Series((1 - weight) * population_mean + weight * m, index=data.index, name="expected_spend")


def noncontractual_cltv(bgnbd: dict, gg: dict, data: pd.DataFrame, periods: float = 52.0,
                        annual_discount: float = 0.10, period_weeks: float = 1.0) -> pd.DataFrame:
    """CLTV = expected transactions x expected spend, discounted, with P(alive) alongside."""
    transactions = expected_transactions(bgnbd, data, periods)
    spend = expected_spend(gg, data)
    years = periods * period_weeks / 52.0
    discount = 1 / (1 + annual_discount) ** (years / 2)            # mid-horizon approximation
    return pd.DataFrame({"p_alive": probability_alive(bgnbd, data).round(4),
                         "expected_transactions": transactions.round(3),
                         "expected_spend": spend.round(2),
                         "cltv": (transactions * spend * discount).round(2)}, index=data.index)


# ============================================================ putting value into the decision
def expected_value_of_contact(frame: pd.DataFrame, uplift_col: str, cltv_col: str,
                              offer_margin: float, contact_cost: float = 0.0) -> pd.Series:
    """The number the engine should really rank on.

    propensity answers *will they buy*; uplift answers *does contacting them change anything*;
    CLTV answers *how much are they worth*. Only the three together give an economic ranking:

        expected value = uplift x offer margin - contact cost

    CLTV enters as the guardrail rather than the multiplier: a high-value customer with a
    negative uplift is precisely the customer an untargeted campaign destroys value on.
    """
    return (frame[uplift_col] * offer_margin - contact_cost).rename("expected_value")


def value_segments(frame: pd.DataFrame, cltv_col: str, uplift_col: str) -> pd.DataFrame:
    """Four quadrants on value and incremental response, with the action each one implies."""
    out = frame.copy()
    high_value = out[cltv_col] >= out[cltv_col].median()
    persuadable = out[uplift_col] >= out[uplift_col].median()
    out["value_segment"] = np.select(
        [high_value & persuadable, high_value & ~persuadable, ~high_value & persuadable],
        ["GROW — contact first", "PROTECT — do not disturb, retain", "EFFICIENT — contact if capacity allows"],
        default="DEPRIORITISE")
    return out
