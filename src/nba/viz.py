"""Charts for the README, rendered in a light and a dark variant from the same data.

Design rules applied throughout: one y-axis per chart, a fixed categorical hue order
(never cycled), an ordinal blue ramp for stage magnitude, recessive grid and axes,
selective direct labels rather than a number on every mark, and a legend whenever two
or more series share a plot. Colors are assigned last, by the job they do.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

# Palette: slot order is a colorblind-safety mechanism, not decoration.
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
              "ramp": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
             "ramp": ["#5598e7", "#3987e5", "#256abf", "#184f95"]},
}


def _figure(t: dict, w: float = 8.4, h: float = 4.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    return fig, ax


def _title(ax, t: dict, title: str, subtitle: str = "") -> None:
    ax.set_title(title, color=t["ink"], fontsize=13, fontweight="bold", loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=9.5, va="bottom")


def _bars(ax, positions, values, colors, horizontal: bool, thickness: float = 0.5):
    """Thin marks anchored to the baseline, with a surface gap between adjacent bars."""
    if horizontal:
        ax.barh(positions, values, height=thickness, color=colors, linewidth=0)
    else:
        ax.bar(positions, values, width=thickness, color=colors, linewidth=0)


def _save(fig, out_dir: Path, name: str, mode: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-{mode}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---------------------------------------------------------------- charts
def chart_funnel(funnel, out_dir: Path, mode: str) -> Path:
    """Where lines drop out, stage by stage. Ordinal ramp: the stage order carries the magnitude."""
    t = THEME[mode]
    labels = {"total_lines": "All lines", "eligible": "Eligible",
              "eligible_and_care_allows_sale": "Care allows a sale", "at_least_one_bu_applies": "At least one offer applies"}
    d = funnel[funnel["step"].isin(labels)].copy()
    d["label"] = d["step"].map(labels)
    fig, ax = _figure(t, 8.4, 3.2)
    pos = list(range(len(d)))[::-1]
    _bars(ax, pos, d["lines"].tolist(), t["ramp"], horizontal=True, thickness=0.46)
    for p, v, share in zip(pos, d["lines"], d["share_of_total"]):
        ax.text(v * 1.015, p, f"{v:,}   {share:.0%}", va="center", ha="left", color=t["ink2"], fontsize=9.5)
    ax.set_yticks(pos, d["label"], color=t["ink"], fontsize=10)
    ax.set_xlim(0, d["lines"].max() * 1.26)
    ax.set_xticks([])
    ax.set_ylim(-0.6, len(d) - 0.4)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(False)
    _title(ax, t, "Who survives each stage", "Lines kept with their exclusion reason, so the funnel can be audited")
    return _save(fig, out_dir, "funnel", mode)


def chart_decile_lift(decile_table, out_dir: Path, mode: str, overall_rate: float | None = None) -> Path:
    """Purchase rate by score decile. The question a ranking score exists to answer:
    are the customers it puts first the ones who actually buy?"""
    t = THEME[mode]
    d = decile_table.sort_values("decile")
    fig, ax = _figure(t, 8.4, 4.0)
    _bars(ax, d["decile"].tolist(), (d["rate"] * 100).tolist(), [t["series"][0]] * len(d), horizontal=False, thickness=0.62)
    if overall_rate is not None:
        ax.axhline(overall_rate * 100, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.text(0.45, overall_rate * 100, f" overall {overall_rate:.1%}", color=t["muted"], fontsize=9, va="bottom", ha="left")
    top = d.iloc[-1]
    ax.text(top["decile"], top["rate"] * 100 * 1.04, f"{top['lift']:.1f}x", ha="center", va="bottom",
            color=t["ink"], fontsize=11, fontweight="bold")
    ax.set_xticks(d["decile"].tolist(), [str(int(x)) for x in d["decile"]], color=t["ink2"])
    ax.set_xlabel("Score decile  (10 = highest scores)", color=t["ink2"], fontsize=10, labelpad=8)
    ax.set_ylabel("Purchase rate (%)", color=t["ink2"], fontsize=10)
    ax.set_xlim(0.3, 10.7)
    ax.set_ylim(0, max(d["rate"] * 100) * 1.22)
    ax.grid(axis="y", color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    _title(ax, t, "Does the score rank the right customers?",
           "Purchase rate by decile of the rule-based score, among contacted customers")
    return _save(fig, out_dir, "decile-lift", mode)


def chart_targeting(targeting, out_dir: Path, mode: str) -> Path:
    """Incremental purchases per 1,000 contacted, if only the top share is contacted."""
    t = THEME[mode]
    fig, ax = _figure(t, 8.4, 4.4)
    series = [("uplift_model", "Uplift model", t["series"][0]),
              ("response_model", "Response model", t["series"][1]),
              ("rule_score", "Rule score", t["series"][2])]
    x = (targeting["top_share"] * 100).tolist()
    base = targeting["everyone"].iloc[0]
    ax.axhline(base, color=t["muted"], linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)
    ax.text(x[0], base + 1.5, "contact everyone", color=t["muted"], fontsize=9, va="bottom", ha="left")
    for col, label, color in series:
        ax.plot(x, targeting[col], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=t["surface"], markeredgewidth=2, label=label, zorder=3)
        ax.text(x[-1] + 1.2, targeting[col].iloc[-1], label, color=t["ink2"], fontsize=9.5, ha="left", va="center")
    ax.set_xticks(x, [f"top {int(v)}%" for v in x], color=t["ink2"])
    ax.set_ylabel("Incremental purchases per 1,000 contacted", color=t["ink2"], fontsize=10)
    ax.set_xlim(x[0] - 3, x[-1] + 17)
    ax.set_ylim(0, max(targeting[[c for c, _, _ in series]].max()) * 1.15)
    ax.grid(axis="y", color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=t["ink2"], fontsize=9.5, loc="lower left", ncols=3, handlelength=1.6, columnspacing=1.4)
    _title(ax, t, "Who should we contact?",
           "Measured on a held-out randomized test, so the difference is a causal estimate")
    return _save(fig, out_dir, "targeting", mode)


def chart_calibration(calibration, out_dir: Path, mode: str) -> Path:
    """Predicted probability vs. observed rate. Needed whenever the number itself is used."""
    t = THEME[mode]
    fig, ax = _figure(t, 5.6, 4.4)
    hi = max(calibration["mean_predicted"].max(), calibration["observed_rate"].max()) * 1.1
    ax.plot([0, hi], [0, hi], color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(hi * 0.52, hi * 0.47, "perfect", color=t["muted"], fontsize=9, ha="left", va="top")
    ax.plot(calibration["mean_predicted"], calibration["observed_rate"], color=t["series"][0], linewidth=2,
            marker="o", markersize=8, markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
    ax.set_xlabel("Mean predicted probability", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Observed purchase rate", color=t["ink2"], fontsize=10)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.grid(color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    _title(ax, t, "Is a predicted 10% really 10%?", "Ten equal-sized bins of predicted probability")
    return _save(fig, out_dir, "calibration", mode)


def chart_survival(curves, out_dir: Path, mode: str, max_month: int = 60) -> Path:
    """Kaplan-Meier survival by segment. The model-free look that comes before any Cox fit."""
    t = THEME[mode]
    fig, ax = _figure(t, 8.4, 4.4)
    # Hues are assigned in fixed slot order and never cycled; past eight segments, fold or facet.
    order = curves.groupby("segment")["survival"].min().sort_values().index.tolist()[:len(t["series"])]
    label_ends = len(order) <= 4                     # beyond four, end labels collide: the legend carries identity
    for i, segment in enumerate(order):
        d = curves[(curves["segment"] == segment) & (curves["month"] <= max_month)].sort_values("month")
        ax.plot(d["month"], d["survival"], color=t["series"][i], linewidth=2, label=segment, zorder=3)
        if label_ends:
            last = d.iloc[-1]
            ax.text(last["month"] + 0.8, last["survival"], segment, color=t["ink2"], fontsize=9.5, va="center")
    ax.set_xlabel("Months since acquisition", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Share still a customer", color=t["ink2"], fontsize=10)
    ax.set_xlim(0, max_month + (17 if label_ends else 2))
    ax.set_ylim(0, 1.02)
    ax.grid(color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=t["ink2"], fontsize=9.5, loc="lower left", ncols=3,
              handlelength=1.6, columnspacing=1.4, borderaxespad=0.2)
    _title(ax, t, "How long do customers stay?",
           "Kaplan-Meier by payment behaviour, censoring customers who are still active")
    return _save(fig, out_dir, "survival", mode)


def chart_value_quadrant(frame, out_dir: Path, mode: str, cltv_col: str = "cltv",
                         uplift_col: str = "uplift", sample: int = 2500) -> Path:
    """Value against incremental response. Two different questions, and they do not agree."""
    t = THEME[mode]
    d = frame.sample(min(sample, len(frame)), random_state=0)
    cut_v, cut_u = frame[cltv_col].median(), frame[uplift_col].median()
    fig, ax = _figure(t, 7.6, 5.0)
    for persuadable, color, label in [(True, t["series"][0], "Above-median uplift"),
                                      (False, t["series"][1], "Below-median uplift")]:
        sub = d[(d[uplift_col] >= cut_u) == persuadable]
        ax.scatter(sub[cltv_col] / 1000, sub[uplift_col] * 100, s=9, c=color, alpha=0.5,
                   linewidths=0, label=label, zorder=3)
    ax.axvline(cut_v / 1000, color=t["muted"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(cut_u * 100, color=t["muted"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    px, py = (xhi - xlo) * 0.015, (yhi - ylo) * 0.02
    for x, y, ha, va, label in [(xhi - px, yhi - py, "right", "top", "GROW\ncontact first"),
                                (xhi - px, ylo + py, "right", "bottom", "PROTECT\ndo not disturb"),
                                (xlo + px, yhi - py, "left", "top", "EFFICIENT\nif capacity allows"),
                                (xlo + px, ylo + py, "left", "bottom", "DEPRIORITISE")]:
        ax.text(x, y, label, color=t["ink2"], fontsize=9, fontweight="600", ha=ha, va=va, linespacing=1.4)
    ax.set_xlabel("Customer lifetime value (thousands)", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Predicted uplift (percentage points)", color=t["ink2"], fontsize=10)
    ax.grid(color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=t["ink2"], fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncols=2, markerscale=2.2)
    _title(ax, t, "Worth a lot is not the same as worth contacting",
           "Lifetime value against the incremental effect of a contact")
    return _save(fig, out_dir, "value-quadrant", mode)
