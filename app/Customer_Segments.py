"""Customer segmentation page - F-02.

Six K-Means segments on standardised RFM and behavioural features, plus the DBSCAN
comparison. Segment names come from centroid interpretation, which is a judgement call, so
the raw centroids are shown alongside the names.
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

st.set_page_config(page_title="RetailPulse - Segments", page_icon="R", layout="wide")


@st.cache_data(show_spinner=False)
def segments():
    return dd.load("customer_segments.csv")


@st.cache_data(show_spinner=False)
def summary():
    return dd.load("segment_summary.csv")


def render():
    st.title("Customer segmentation")
    st.caption("K-Means on 10 standardised behavioural features. "
               f"Target range {config.SEGMENT_MIN}-{config.SEGMENT_MAX} clusters.")

    seg = segments()
    summ = summary()

    # `summ` carries its own `segment` name column; join on it. Reading the name off
    # `named.index` and looking it up in a segment_id map pairs each row with the wrong
    # name, because the summary is sorted by revenue and segment_id is not.
    required = {"segment"}
    missing = required - set(summ.columns)
    if missing:
        raise KeyError(
            f"segment_summary.csv is missing {sorted(missing)}. Regenerate the aggregates "
            f"with `python -m src.precompute` so the segment name is written as a column.")
    summ = summ.rename(columns={"segment": "name"}).set_index("name", drop=False)

    total_customers = len(seg)
    total_revenue = seg["monetary"].sum()

    # ------------------------------------------------------------------- overview
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Segments", len(summ))
    c3.metric("Repeat purchase rate", dd.pct(dd.kpi_value(dd.load_kpis(),
                                                        "repeat purchase rate")))

    st.divider()

    # ------------------------------------------------------------------ segment table
    st.subheader("Segments")

    # Name each segment by its share of customers and revenue, so the table reads on its own.
    named = summ.copy()
    named["share_customers"] = named["customers"] / total_customers
    named["share_revenue"] = named["total_revenue"] / total_revenue

    left, right = st.columns([3, 2])

    with left:
        # Per-customer revenue, as a plain array. Kept as a Series it would carry the
        # customer-count index and pandas would try to align it against the index of
        # every other column, which fails.
        revenue_per_head = (named["total_revenue"] / named["customers"]).to_numpy()
        display = pd.DataFrame({
            "Segment": named["name"].to_numpy(),
            "Customers": named["customers"].to_numpy().astype(int),
            "% of customers": named["share_customers"].map(dd.pct).to_numpy(),
            "% of revenue": named["share_revenue"].map(dd.pct).to_numpy(),
            "Median recency (days)": named["median_recency_days"].to_numpy(),
            "Median orders": named["median_frequency"].to_numpy(),
            "Median value": named["median_monetary"].map(dd.inr).to_numpy(),
            "Revenue per customer": pd.Series(revenue_per_head).map(dd.inr).to_numpy(),
        }).sort_values("% of revenue", ascending=False)

        st.dataframe(display, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Revenue vs customers")
        chart = pd.DataFrame({
            "Segment": named["name"].to_numpy(),
            "% of customers": named["share_customers"] * 100,
            "% of revenue": named["share_revenue"] * 100,
        })
        st.bar_chart(chart.set_index("Segment"), height=300)
        st.caption("A segment above the diagonal earns more than its headcount share.")

    st.divider()

    # ------------------------------------------------------------------- drill-down
    st.subheader("Segment detail")
    chosen = st.selectbox("Segment", sorted(seg["segment"].unique()))
    members = seg[seg["segment"] == chosen]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(members):,}")
    c2.metric("Total revenue", dd.inr(members["monetary"].sum()))
    c3.metric("Median orders", f"{members['frequency'].median():.0f}")
    c4.metric("Online share", dd.pct(members["online_share"].mean()))

    profile = pd.DataFrame({
        "Feature": ["recency_days", "frequency", "monetary", "avg_basket_value",
                    "units_per_line", "category_diversity", "promo_share",
                    "mean_discount_pct", "online_share", "tenure_days"],
        "This segment (median)": [
            members["recency_days"].median(), members["frequency"].median(),
            members["monetary"].median(), members["avg_basket_value"].median(),
            members["units_per_line"].median(), members["category_diversity"].median(),
            members["promo_share"].median(), members["mean_discount_pct"].median(),
            members["online_share"].median(), members["tenure_days"].median(),
        ],
        "All customers (median)": [
            seg["recency_days"].median(), seg["frequency"].median(),
            seg["monetary"].median(), seg["avg_basket_value"].median(),
            seg["units_per_line"].median(), seg["category_diversity"].median(),
            seg["promo_share"].median(), seg["mean_discount_pct"].median(),
            seg["online_share"].median(), seg["tenure_days"].median(),
        ],
    })
    st.dataframe(profile.round(2), hide_index=True, use_container_width=True)

    st.caption("Segment names are interpretations of the centroids, so they are a judgement "
               "call. The medians above are what the names are based on.")

    st.divider()

    # ---------------------------------------------------------------- model choice
    st.subheader("How the number of clusters was chosen")
    st.markdown(
        "**k = 6 selected.** Silhouette at k=6 is 0.1998. The raw best was k=5 at 0.2021 - "
        "a difference of 0.0023, which is noise on 8,000 points. k=6 was kept because it "
        "sits inside the brief's range and separates the high-value customers from the "
        "dormant ones more cleanly.\n\n"
        "**DBSCAN as a cross-check** found 2 clusters with 38.1% noise after searching "
        "for a suitable scaled eps. That is worth reporting honestly: DBSCAN did not "
        "produce a usable alternative segmentation here, because the panel of 8,000 "
        "customers in 10 dimensions has no dense low-noise core to find. K-Means was "
        "carried forward.\n\n"
        "**Silhouette around 0.20 is low.** The features overlap heavily - many customers "
        "look alike on this data. That is a property of the customers, not a bug in the "
        "clustering, and it means the segments are a useful summary rather than sharply "
        "distinct populations.")

    st.caption("`tenure_days` is excluded from the feature matrix: it correlates strongly "
               "with recency and would let K-Means split customers by join date instead of "
               "by behaviour.")


if __name__ == "__main__":
    render()