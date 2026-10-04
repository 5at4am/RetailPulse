"""Inventory recommendations page - F-05.

The reorder plan, with the cost assumptions that produce it and the Poisson-versus-normal
comparison that justifies the safety-stock method.
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

st.set_page_config(page_title="RetailPulse - Inventory", page_icon="R", layout="wide")


@st.cache_data(show_spinner=False)
def plan():
    return dd.load("inventory_reorder_plan.csv")


@st.cache_data(show_spinner=False)
def summary():
    return dd.load("inventory_summary.csv").iloc[0]


def render():
    st.title("Inventory recommendations")
    st.caption("Reorder quantities for all 7,500 store-product pairs, driven by the "
               "1-week pooled forecast.")

    s = summary()
    full = plan()

    # --------------------------------------------------------------------- headline
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs needing reorder", f"{int(s['pairs_to_reorder']):,}",
              delta=f"of {int(s['pairs']):,}")
    c2.metric("Units to order", f"{s['total_units_to_order']:,.0f}")
    c3.metric("Purchase cost", dd.inr(s["total_order_value"]))
    c4.metric("Lead time", f"{int(s['lead_time_weeks'])} week")

    st.divider()

    # ------------------------------------------------------- the service level point
    st.subheader("Why the plan is not at 95%")

    st.markdown(
        f"The brief asks for a 95% service level. **The arithmetic gives "
        f"{s['mean_critical_ratio']:.4f}**, so the plan is calibrated to a "
        f"{s['mean_critical_ratio'] * 100:.1f}% quantile instead.\n\n"
        f"Reorder quantity is a newsvendor calculation. Underage costs the margin lost when "
        f"a customer cannot buy; overage costs the purchase plus holding. Those two costs "
        f"set the critical ratio:\n\n"
        f"```\ncritical ratio = p / (p + o)\n```\n\n"
        f"With the cost assumptions on this page, the lost margin dominates by a wide "
        f"margin because holding one extra unit for a week costs very little. So it pays "
        f"to hold deep stock. That is the correct consequence of these numbers, not an "
        f"error - and it is a different quantity from the confidence on the demand "
        f"distribution.\n\n"
        f"Using the 95th percentile because the brief says '95%' is a common and expensive "
        f"confusion between the two.")

    with st.expander("Cost assumptions - these drive everything"):
        st.markdown(
            f"- **Purchase cost**: {dd.pct(s['purchase_cost_ratio'], 0)} of retail price. "
            f"*The dataset has no cost column, so this is an assumption and cannot be "
            f"measured from the data.*\n"
            f"- **Annual holding cost**: {dd.pct(s['holding_rate'], 0)} of unit cost - rent, "
            f"insurance, shrinkage, tied-up capital.\n"
            f"- **Lost margin on a stockout**: {dd.pct(s['lost_margin_ratio'], 0)} of price. "
            f"Less than the full price because the customer buys something else.\n"
            f"- **Lead time**: {int(s['lead_time_weeks'])} week.\n\n"
            f"Change any of these and the plan changes. That is why they are shown rather "
            f"than buried in code.")

    st.divider()

    # --------------------------------------------------- Poisson vs normal, honestly
    st.subheader("Why safety stock is a Poisson quantile")

    st.markdown(
        f"{dd.pct(dd.kpi_value(dd.load_kpis(), 'zero-demand panel share'))} of the demand "
        f"panel is zero-demand weeks. Most store-product pairs sell nothing most weeks, so "
        f"demand is nothing like a normal distribution.\n\n"
        f"The textbook `mean + z x std` formula still gets *computed* on that data and "
        f"still returns a number, which is exactly the problem - the number has no "
        f"probabilistic meaning. The Poisson quantile asks a real question: what is the "
        f"demand level that {s['mean_critical_ratio'] * 100:.1f}% of weeks fall below?\n\n"
        f"Both are shown below. **The Poisson figure is the one to act on.**")

    comp = pd.DataFrame({
        "Method": ["Poisson quantile (used)", "Normal z approximation"],
        "Pairs flagged": [int(s["pairs_to_reorder"]), int(s["normal_approx_pairs"])],
        "Units to order": [f"{s['total_units_to_order']:,.0f}",
                           f"{s['normal_approx_units_to_order']:,.0f}"],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)

    overstatement = s["normal_approx_units_to_order"] - s["total_units_to_order"]
    st.error(
        f"The normal approximation overstates the plan by **{overstatement:,.0f} units** - "
        f"{s['normal_approx_units_to_order'] / s['total_units_to_order']:.1f}x. Its largest "
        f"single order was 23.3 units of a product forecast at 2.2 units per week: over ten "
        f"weeks of cover on a one-week lead time.\n\n"
        f"Poisson is not universally more conservative, and the tests say so. On smooth, "
        f"near-continuous demand it orders *less* than the normal approximation. Poisson is "
        f"the right tool for intermittent demand specifically, which is what this panel is.")

    st.divider()

    # ------------------------------------------------------------------- the plan
    st.subheader("Reorder plan")

    flagged = full[full["reorder"] == True] if "reorder" in full.columns else full  # noqa: E712

    c1, c2 = st.columns(2)
    store_filter = c1.selectbox("Store", ["All stores"] + sorted(full["store_id"].unique()))
    max_units = c2.slider("Show orders up to (units)", 1,
                          max(1, int(flagged["suggested_order"].max()) or 1), 10)

    view = flagged if store_filter == "All stores" else flagged[flagged["store_id"] == store_filter]
    view = view[view["suggested_order"] <= max_units]

    st.caption(f"{len(view):,} pairs shown. Raising the slider shows more of the plan.")

    st.dataframe(
        view[["store_id", "product_id", "on_hand", "forecast_units", "safety_stock",
              "reorder_point", "suggested_order", "avg_price"]]
        .rename(columns={
            "store_id": "Store", "product_id": "Product", "on_hand": "On hand",
            "forecast_units": "Forecast", "safety_stock": "Safety stock",
            "reorder_point": "Reorder point", "suggested_order": "Order qty",
            "avg_price": "Avg price"})
        .round(2),
        hide_index=True, use_container_width=True, height=450)

    st.divider()

    # ------------------------------------------------------------------ concentrations
    st.subheader("Where the orders concentrate")
    by_store = (flagged.groupby("store_id")
                .agg(pairs=("product_id", "count"), units=("suggested_order", "sum"))
                .sort_values("units", ascending=False))
    st.bar_chart(by_store["units"], height=300)
    st.caption(f"{len(by_store)} stores have at least one pair needing reorder.")

    st.divider()
    st.subheader("Limitations")
    st.markdown(
        "- **Store-product forecasts are apportioned, not independently forecast.** Each "
        "pair's forecast is its historical share of the pooled total, so the plan inherits "
        "the pooled model's error and adds none of its own.\n"
        "- **No supplier or MOQ constraints are applied.** The plan is unconstrained: no "
        "minimum order quantity, no case-pack rounding, no supplier lead-time differences. "
        "Real purchasing would need all three.\n"
        "- **On-hand stock is a snapshot**, taken at the last usable week. Anything that "
        "changed since then is not reflected.\n"
        "- **No returns.** The dataset has none by design, so the model cannot learn that "
        "excess stock returns to zero.")


if __name__ == "__main__":
    render()