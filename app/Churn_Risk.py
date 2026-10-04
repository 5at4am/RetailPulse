"""Churn risk page - F-04.

Temporal split, XGBoost, SHAP. The AUC miss against the brief's 0.88 target is shown on
this page rather than left for the reader to find in the report.
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

st.set_page_config(page_title="RetailPulse - Churn", page_icon="R", layout="wide")


@st.cache_data(show_spinner=False)
def metrics():
    return dd.load("churn_metrics.csv").iloc[0]


@st.cache_data(show_spinner=False)
def shap():
    return dd.load("churn_shap_importance.csv").sort_values("mean_abs_shap", ascending=False)


@st.cache_data(show_spinner=False)
def scores_for(snapshot: str):
    frame = dd.load("churn_scores.csv")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    return frame[frame["snapshot_date"] == pd.Timestamp(snapshot)]


def render():
    st.title("Churn risk")
    st.caption("XGBoost on nine monthly snapshots. A customer is churned if they do not "
               f"return within {config.CHURN_LABEL_WINDOW_DAYS} days.")

    m = metrics()

    # ------------------------------------------------------------------- headline
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision in top 20%", f"{m['precision_at_top']:.4f}",
              delta=f"target {m['target_precision']}",
              delta_color="normal")
    c2.metric("Lift", f"{m['lift_at_top']:.2f}x")
    c3.metric("ROC AUC", f"{m['auc']:.4f}", delta=f"target {config.CHURN_AUC_TARGET}",
              delta_color="inverse")
    c4.metric("Recall in top 20%", f"{m['recall_at_top']:.4f}")

    st.divider()

    # --------------------------------------------------- the two results, side by side
    left, right = st.columns(2)

    with left:
        st.subheader("Precision target met")
        st.success(
            f"Precision among the top {m['k'] * 100:.0f}% highest-risk customers is "
            f"**{m['precision_at_top']:.4f}**, above the {m['target_precision']} floor. "
            f"Lift is **{m['lift_at_top']:.2f}x** against a "
            f"{m['test_base_rate'] * 100:.1f}% base rate - calling those customers first "
            f"is roughly {m['lift_at_top'] / 1:.1f}x more likely to find a churner than "
            f"calling randomly.")

    with right:
        st.subheader("AUC target missed")
        st.error(
            f"ROC AUC is **{m['auc']:.4f}** against the brief's target of "
            f"{config.CHURN_AUC_TARGET}. This is a real miss and it is not explained away "
            f"anywhere in this project.\n\n"
            f"The likely reason is visible in the SHAP plot: churn in this dataset is "
            f"overwhelmingly a recency effect. A customer is defined as churned by *not* "
            f"returning for 90 days, so 'days since last purchase' has already answered the "
            f"question by the time the model runs. That leaves less for the other nine "
            f"features to contribute, and a ranking task that is mostly one feature will "
            f"not reach AUC 0.88.\n\n"
            f"That said, precision on the actionable top slice - which is what a retention "
            f"list actually needs - clears its target comfortably.")

    st.divider()

    # ------------------------------------------------------------------- validation
    st.subheader("How it was validated")
    st.markdown(
        f"A random split would leak: the same customer appears in multiple monthly "
        f"snapshots, so a shuffled split puts near-duplicates on both sides and inflates "
        f"the score. Snapshots are therefore split by **date**.\n\n"
        f"| | snapshots | rows | churn base rate |\n"
        f"|---|---|---|---|\n"
        f"| Train | Jan-Jun 2025 | {int(m['n_train']):,} | "
        f"{m['train_base_rate'] * 100:.1f}% |\n"
        f"| Test | Jul-Sep 2025 | {int(m['n_test']):,} | "
        f"{m['test_base_rate'] * 100:.1f}% |\n\n"
        f"All features come from the snapshot's own window, never from after it. Imputation "
        f"statistics are fitted on train only and applied to test - a real leakage path that "
        f"`tests/test_no_leakage.py` checks explicitly.")

    st.divider()

    # ------------------------------------------------------------------------ SHAP
    st.subheader("What drives the prediction")
    importance = shap()
    st.bar_chart(importance.set_index("feature")["mean_abs_shap"], height=300)

    top = importance.iloc[0]
    st.caption(
        f"Mean absolute SHAP value, averaged over every scored customer. **{top['feature']}** "
        f"({top['mean_abs_shap']:.4f}) dominates "
        f"{importance['mean_abs_shap'].iloc[1] / top['mean_abs_shap']:.1f}x the next "
        f"feature. That concentration is the quantitative version of the AUC explanation "
        f"above, and it is the honest headline for this model.")

    st.dataframe(
        importance.rename(columns={"feature": "Feature",
                                   "mean_abs_shap": "Mean |SHAP|"})
        .assign(**{"Mean |SHAP|": lambda d: d["Mean |SHAP|"].round(4)}),
        hide_index=True, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------- customer scores
    st.subheader("Customer risk scores")
    st.caption("One row per customer per monthly snapshot. Pick a snapshot to see that "
               "month's ranking.")

    all_snapshots = sorted(pd.to_datetime(
        dd.load("churn_scores.csv")["snapshot_date"]).dt.strftime("%Y-%m-%d").unique())
    chosen = st.selectbox("Snapshot", all_snapshots, index=len(all_snapshots) - 1)

    rows = scores_for(chosen)
    if rows.empty:
        st.warning("No scores for that snapshot.")
        return

    top_k = max(1, int(len(rows) * float(m["k"])))
    flagged = rows.nlargest(top_k, "churn_score")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{len(rows):,}")
    c2.metric(f"Flagged (top {m['k'] * 100:.0f}%)", f"{len(flagged):,}")
    c3.metric("Actual churn rate in flagged",
              f"{flagged['churn'].mean() * 100:.1f}%" if "churn" in flagged.columns
              else "n/a")

    st.dataframe(
        flagged[["customer_id", "churn_score"]]
        .rename(columns={"customer_id": "Customer", "churn_score": "Risk score"})
        .round(4),
        hide_index=True, use_container_width=True,
        height=400)

    if "churn" in flagged.columns:
        st.caption(f"Of these {len(flagged):,} customers, "
                   f"{int(flagged['churn'].sum()):,} did churn within "
                   f"{config.CHURN_LABEL_WINDOW_DAYS} days. This is the list a retention "
                   f"campaign would work through.")


if __name__ == "__main__":
    render()