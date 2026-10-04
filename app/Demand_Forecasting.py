"""Demand forecasting page - F-03.

Shows the five-model backtest, the per-horizon selection, and the forecast itself, with the
comparison that matters stated plainly: pooled WAPE and the brief's panel-level MAPE target
are not the same measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import dashboard_data as dd
from src import config

st.set_page_config(page_title="RetailPulse - Forecasting", page_icon="R", layout="wide")


@st.cache_data(show_spinner=False)
def backtest():
    return dd.load("forecast_backtest.csv")


@st.cache_data(show_spinner=False)
def forecast():
    return dd.load("forecast_next_weeks.csv")


@st.cache_data(show_spinner=False)
def by_store():
    return dd.load("forecast_next_weeks_by_store.csv")


@st.cache_data(show_spinner=False)
def weekly():
    frame = dd.load("revenue_by_week.csv")
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    return frame.sort_values("week_start_date")


def render():
    st.title("Demand forecasting")
    st.caption("Weekly units for all 50 stores, backtested on held-out weeks.")

    scores = backtest()
    forecast_next = forecast()
    history = weekly()

    # ----------------------------------------------------------------- headline forecast
    point = forecast_next[forecast_next["horizon_weeks"] == config.INVENTORY_HORIZON].iloc[0]
    last = history["units"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next week forecast", f"{point['forecast_units']:,.0f} units",
              delta=f"{point['forecast_units'] - last:,.0f} vs last week")
    c2.metric("Model at 1 week", str(point["model"]))
    c3.metric("Holdout WAPE (1 week)", dd.pct(point["holdout_wape"], 2))
    c4.metric("4-week WAPE", dd.pct(
        float(scores[(scores["horizon"] == config.PRIMARY_HORIZON)
                     & (scores["model"] == "seasonal_naive")]["wape"].iloc[0]), 2))

    st.divider()

    # ---------------------------------------------------------------- the full backtest
    st.subheader("Model backtest")
    st.caption("Every candidate scored on the same held-out weeks. Lower WAPE is better.")

    labels = {"horizon": "Horizon", "model": "Model", "wape": "WAPE",
              "mape_nonzero": "MAPE (non-zero)", "mae": "MAE", "bias": "Bias",
              "actual_total": "Actual units", "predicted_total": "Predicted units",
              "error": "Error"}

    horizon = st.slider("Horizon (weeks)", min_value=1, max_value=4, value=1)
    subset = scores[scores["horizon"] == horizon].sort_values("wape")
    # Rename after selecting. Selecting by the new names raises, because the frame still
    # has the old ones at this point.
    st.dataframe(subset.rename(columns=labels), hide_index=True,
                 use_container_width=True)

    best = subset.iloc[0]
    st.success(f"**{best['model']}** wins at {horizon} week(s) with WAPE "
               f"{dd.pct(best['wape'], 2)}.")

    if horizon == 1:
        st.info("The ensemble did not win at any horizon. A 50/50 blend of seasonal naive "
                "and Prophet beat seasonal naive at one week and lost to it at two and four. "
                "Blending inherits the worse model's error wherever the two disagree, so it "
                "is kept in the benchmark for comparison rather than presented as the answer.")

    st.divider()

    # ------------------------------------------------------------------- the forecast
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Forecast vs recent history")
        # One path only. The file holds a 1-week, a 2-week and a 4-week path, so
        # `weeks_ahead` repeats across them and concat on that index raises on duplicate
        # labels. Plotting all three would also draw three different forecasts on one axis,
        # which would imply they are alternatives when they are nested horizons.
        path = forecast_next[forecast_next["horizon_weeks"] == config.PRIMARY_HORIZON]
        combined = pd.concat([
            history.tail(12)["units"].rename("actual"),
            path.set_index("weeks_ahead")["forecast_units"].rename("forecast"),
        ], axis=1).sort_index()
        st.line_chart(combined, height=280)
        st.caption(f"Last 12 actual weeks, then the {config.PRIMARY_HORIZON}-week forecast "
                   f"path from {path['model'].iloc[0]}. Seasonal naive means each forecast "
                   f"week repeats its counterpart from a year earlier.")

    with right:
        st.subheader("Forecast by week ahead")
        st.dataframe(
            forecast_next[["weeks_ahead", "horizon_weeks", "model", "forecast_units"]]
            .rename(columns={"weeks_ahead": "Week", "horizon_weeks": "Model horizon",
                             "model": "Model", "forecast_units": "Units"})
            .sort_values(["Model horizon", "Week"]),
            hide_index=True, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------- reconciliation
    st.subheader("Store-level split")
    st.caption("The pooled total is split by each store's historical share, so the store "
               "forecasts add up to the total exactly rather than approximately.")

    stores = by_store().sort_values("store_units", ascending=False)
    total = stores["store_units"].sum()
    st.caption(f"{len(stores)} stores, summing to {total:,.0f} units - the pooled forecast.")

    st.dataframe(
        stores[["store_id", "share", "store_units"]]
        .rename(columns={"store_id": "Store", "share": "Share",
                         "store_units": "Forecast units"}),
        hide_index=True, use_container_width=True)

    st.divider()

    # ------------------------------------------------------- the comparison that matters
    st.subheader("Why pooled WAPE is not the brief's 12% MAPE target")

    st.markdown(
        f"The brief sets a MAPE target of {dd.pct(config.BRIEF_MAPE_TARGET, 0)}. "
        f"**This dashboard does not claim that target is met, at either level.**\n\n"
        f"The pooled weekly total has no zero-demand weeks - something always sells "
        f"somewhere in 50 stores. So on the pooled series the non-zero MAPE filter excludes "
        f"nothing, and MAPE collapses to a per-point ratio that sits close to WAPE. That "
        f"makes pooled numbers look excellent.\n\n"
        f"The demand panel is {dd.pct(dd.kpi_value(dd.load_kpis(), 'zero-demand panel share'))} "
        f"zero-demand. At that level MAPE is undefined on most rows, and the brief's target "
        f"was written about *that* level. Meeting it at pooled level and reporting the pooled "
        f"number would be true and misleading at once.")

    pooled = scores[(scores["horizon"] == 4) & (scores["model"] == "seasonal_naive")].iloc[0]
    st.markdown(
        f"What is claimed: on the pooled total, seasonal naive reaches WAPE "
        f"**{dd.pct(pooled['wape'], 2)}** at 4 weeks and "
        f"**{dd.pct(float(scores[(scores['horizon'] == 1) & (scores['model'] == 'prophet')]['wape'].iloc[0]), 2)}** "
        f"with Prophet at 1 week.\n\n"
        f"What is not claimed: that MAPE under 12% has been demonstrated at the "
        f"store-product level.")

    st.divider()
    st.subheader("Model notes")
    st.markdown(
        "- **Prophet** is strong at one week (0.68% WAPE) and poor at four (11.2%). Same "
        "model, same data - a single headline number would have hidden that.\n"
        "- **LSTM** was the least accurate candidate at all three horizons, the opposite of "
        "how the brief framed it. Reported as it came out.\n"
        "- **Per-store-product models were not built.** 7,500 Prophet fits cannot meet the "
        "batch budget. The pooled-then-split approach is the substitution, and the "
        "store-level numbers inherit the pooled model's error rather than having their own.")


if __name__ == "__main__":
    render()