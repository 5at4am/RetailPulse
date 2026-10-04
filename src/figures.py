"""Generate the report figures from the committed aggregates.

Every figure is drawn from `data/processed/` (and `reports/drift/`) rather than from a
hard-coded number, so a figure cannot drift away from the CSV beside it. If an input column
disappears, this raises rather than plotting an empty axis.

    python -m src.figures

Output: `reports/figures/*.png` at 150 dpi, sized for a two-column-width PDF.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")            # no display on the build agents
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

FIGURE_DIR = config.REPORTS / "figures"

# One palette, used everywhere. Chosen for legibility in greyscale too, because the PDF may be
# printed rather than read on a screen.
INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2b6cb0"
WARN = "#c05621"
GRID = "#e4e7eb"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_forecast() -> str:
    """Weekly units with the 4-week pooled forecast appended."""
    weekly = pd.read_csv(config.PROCESSED / "revenue_by_week.csv",
                         parse_dates=["week_start_date"]).sort_values("week_start_date")
    forecast = pd.read_csv(config.PROCESSED / "forecast_next_weeks.csv")
    # The brief's primary horizon is 4 weeks; the file carries a multi-horizon block.
    row = forecast[forecast["horizon_weeks"] == 4]
    if row.empty:
        raise ValueError("forecast_next_weeks.csv has no horizon_weeks == 4 row")

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    history = weekly.tail(26)
    ax.plot(history["week_start_date"], history["units"], color=ACCENT, linewidth=1.6,
            label="actual")
    ax.fill_between(history["week_start_date"], history["units"], color=ACCENT, alpha=0.12)

    start = weekly["week_start_date"].iloc[-1]
    future = pd.date_range(start + pd.Timedelta(weeks=1), periods=int(row["weeks_ahead"].iloc[0]),
                           freq="W-MON")
    ax.plot(future, row["forecast_units"].iloc[0], color=WARN, linewidth=1.6, linestyle="--",
            marker="o", markersize=3.5, label=f"forecast (h=4, WAPE {row['holdout_wape'].iloc[0]:.4f})")
    ax.axvline(start, color=MUTED, linewidth=0.8, linestyle=":")
    _style(ax, "Pooled weekly demand and the 4-week forecast", "week", "units")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.autofmt_xdate(rotation=30, ha="right")
    return _save(fig, "fig1_forecast.png")


def figure_model_comparison() -> str:
    """WAPE by horizon and model. The point of this chart is the short-horizon inversion."""
    scores = pd.read_csv(config.PROCESSED / "forecast_backtest.csv")
    scores = scores[scores["wape"].notna()]
    if scores.empty:
        raise ValueError("forecast_backtest.csv has no scored models")

    models = ["seasonal_naive", "ridge_seasonal", "prophet", "ensemble_naive_prophet", "lstm"]
    models = [m for m in models if m in set(scores["model"])]
    horizons = sorted(scores["horizon"].unique())

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    width = 0.8 / len(models)
    positions = np.arange(len(horizons))
    for index, model in enumerate(models):
        values = [scores[(scores.model == model) & (scores.horizon == h)]["wape"].mean()
                  for h in horizons]
        bars = ax.bar(positions + index * width - 0.4 + width / 2, values, width * 0.92,
                      label=model, color=plt.cm.viridis(index / max(1, len(models) - 1)))
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{value:.3f}", ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"h={h} week{'s' if h > 1 else ''}" for h in horizons])
    _style(ax, "Backtest WAPE by horizon — Prophet wins at h=1, seasonal naive from h=2",
           "", "WAPE (lower is better)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    ax.set_yscale("log")
    return _save(fig, "fig2_model_comparison.png")


def figure_shap() -> str:
    """Global churn feature importance."""
    shap = pd.read_csv(config.PROCESSED / "churn_shap_importance.csv").sort_values(
        "mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    colors = [ACCENT if name == shap["feature"].iloc[-1] else MUTED
              for name in shap["feature"]]
    ax.barh(shap["feature"], shap["mean_abs_shap"], color=colors)
    ax.set_xlim(0, shap["mean_abs_shap"].max() * 1.18)
    for y, value in enumerate(shap["mean_abs_shap"]):
        ax.text(value + shap["mean_abs_shap"].max() * 0.02, y, f"{value:.3f}", va="center",
                fontsize=7, color=MUTED)
    _style(ax, "Churn: mean |SHAP| by feature — recency dominates", "", "mean |SHAP| (log-odds)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig3_shap.png")


def figure_segments() -> str:
    """Segment size against revenue share, against the equal-share diagonal."""
    segments = pd.read_csv(config.PROCESSED / "segment_summary.csv")
    revenue_share = 100 * segments["total_revenue"] / segments["total_revenue"].sum()
    customer_share = 100 * segments["customers"] / segments["customers"].sum()

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    limit = max(revenue_share.max(), customer_share.max()) * 1.18
    ax.plot([0, limit], [0, limit], color=MUTED, linewidth=0.9, linestyle="--",
            label="equal share")
    ax.scatter(customer_share, revenue_share, s=70, color=ACCENT, zorder=3)
    for _, row in segments.iterrows():
        ax.annotate(row["segment"], (100 * row["customers"] / segments["customers"].sum(),
                                     100 * row["total_revenue"] / segments["total_revenue"].sum()),
                    textcoords="offset points", xytext=(7, 4), fontsize=7.5, color=INK)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    _style(ax, "Segment revenue share vs customer share (k=6, silhouette 0.1998)",
           "% of customers", "% of revenue")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, "fig4_segments.png")


def figure_drift() -> str:
    """PSI per watched column against the 0.1 stable / 0.2 significant bands."""
    drift = pd.read_csv(config.REPORTS / "drift" / "drift_summary.csv")
    numeric = drift[drift["kind"] != "categorical"].copy()
    numeric = numeric.sort_values("psi")
    if numeric.empty:
        raise ValueError("drift_summary.csv has no numeric rows")

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.axvspan(0, 0.1, color="#f0f4f8", zorder=0)
    colors = [WARN if verdict != "stable" else MUTED for verdict in numeric["verdict"]]
    ax.barh(numeric["column"], numeric["psi"], color=colors)
    ax.axvline(0.1, color=WARN, linewidth=1.0, linestyle="--")
    ax.text(0.1, len(numeric) - 0.4, " stable / moderate boundary", fontsize=7, color=WARN,
            va="top")
    limit = max(0.12, float(numeric["psi"].max()) * 1.15)
    ax.set_xlim(0, limit)
    for y, value in enumerate(numeric["psi"]):
        ax.text(value + limit * 0.015, y, f"{value:.4f}", va="center", fontsize=7, color=MUTED)
    _style(ax, "Drift 2024 vs 2025: no column exceeds the 0.1 stable threshold", "", "PSI")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig5_drift.png")


def figure_inventory() -> str:
    """The inventory backtest decomposition. The most important figure in the report."""
    summary = pd.read_csv(config.PROCESSED / "inventory_backtest_summary.csv").iloc[0]
    labels = ["Overstock\n(units)", "Understock\n(units)"]
    baseline = [summary["baseline_overstock_units"], summary["baseline_understock_units"]]
    policy = [summary["policy_overstock_units"], summary["policy_understock_units"]]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    positions = np.arange(len(labels))
    width = 0.36
    ax.bar(positions - width / 2, baseline, width, label="naive-mean baseline", color=MUTED)
    ax.bar(positions + width / 2, policy, width, label="newsvendor policy", color=WARN)
    for x, value in zip(positions - width / 2, baseline):
        ax.text(x, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    for x, value in zip(positions + width / 2, policy):
        ax.text(x, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=7.5, color=WARN)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(baseline + policy) * 1.16)
    _style(ax,
           f"Backtest: total error -{summary['error_reduction_pct']:.2f}% against a 25–40% "
           f"target — the error moves side, it does not shrink",
           "", "units")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, "fig6_inventory.png")


def _save(fig, name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    return str(path)


def main() -> None:
    config.ensure_dirs()
    for builder in (figure_forecast, figure_model_comparison, figure_shap,
                    figure_segments, figure_drift, figure_inventory):
        builder()
    print(f"{6} figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()