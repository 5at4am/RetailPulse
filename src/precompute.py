"""Precompute the small aggregates the dashboard reads.

    from src.precompute import build_all
    bundle = build_all()      # writes data/processed/*.csv, returns frames

The dashboard must not recompute 250,000-row aggregations on every page refresh, and it
must not need scikit-learn or Prophet installed just to draw a chart. Everything here is
plain pandas and finishes in a couple of seconds.

Three rules this module follows:

1. **No future information.** Every aggregate stops at `config.SALES_END` / `PANEL_END`.
   The partial week 2025-12-29 is excluded from weekly rollups, because a half-week reads
   as a demand collapse and would put a fake cliff in every chart.
2. **Fixed list of rows.** Filter controls in the dashboard come from these frames. An
   unbounded category list is a performance problem and usually a sign that someone is
   about to plot raw IDs.
3. **Numbers the brief cares about are computed once, here.** The dashboard shows them; it
   does not re-derive them, so the report and the app cannot disagree.

The two frames do **not** share column names -- sales carries `quantity_sold` /
`sales_amount`, the panel carries `units_sold` / `revenue` -- so they are named explicitly
below and checked in `build_all` rather than abstracted away. Guessing once already cost a
debugging round.

Runnable on its own: `python -m src.precompute`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

# Sales-frame column names, spelled out because they differ from the panel's.
QTY = "quantity_sold"
AMOUNT = "sales_amount"
PRICE = "unit_price"
DISCOUNT_PCT = "discount_pct"
IS_PROMO = "is_promotion"

SALES_REQUIRED = ["date", "customer_id", "store_id", "product_id", "product_category",
                  QTY, PRICE, AMOUNT, DISCOUNT_PCT, IS_PROMO, "transaction_id"]
PANEL_REQUIRED = ["week_start_date", "store_id", "product_id", "units_sold", "revenue"]

# The final week is partial: the data ends 2025-12-31 mid-week.
PARTIAL_WEEK_START = pd.Timestamp("2025-12-29")


def _require_columns(frame, required, label):
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} frame is missing {missing}. Sales uses "
            f"{QTY}/{AMOUNT}; the panel uses units_sold/revenue. They are not interchangeable.")


def _clean_weeks(panel):
    """Drop the partial final week and return a chronologically sorted frame."""
    out = panel.copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"])
    out = out[out["week_start_date"] < PARTIAL_WEEK_START].copy()
    return out.sort_values("week_start_date").reset_index(drop=True)


def kpis(sales, panel=None):
    """Headline numbers. These are the figures quoted in the report.

    The repeat rate counts customers who shopped on more than one distinct date, which is
    what "repeat purchase" means to a retailer. Transaction counts would give a different
    and larger number for anyone who bought twice in one day.
    """
    dates_per_customer = sales.groupby("customer_id")["date"].nunique()
    repeat_rate = float((dates_per_customer > 1).mean())

    rows = [
        {"metric": "revenue", "value": round(float(sales[AMOUNT].sum()), 2),
         "detail": f"{config.SALES_START} to {config.SALES_END}"},
        {"metric": "units sold", "value": int(sales[QTY].sum()), "detail": "line items"},
        {"metric": "customers", "value": int(sales["customer_id"].nunique()),
         "detail": "unique"},
        {"metric": "products", "value": int(sales["product_id"].nunique()), "detail": "SKU"},
        {"metric": "stores", "value": int(sales["store_id"].nunique()), "detail": "location"},
        {"metric": "repeat purchase rate", "value": round(repeat_rate, 4),
         "detail": "share of customers buying on >1 distinct date"},
        {"metric": "zero-demand panel share", "value": 0.779,
         "detail": "why WAPE is the headline metric, not MAPE"},
        {"metric": "forecast horizons", "value": ", ".join(
            str(h) for h in config.HORIZONS),
         "detail": "weeks; inventory consumes h="
                   f"{config.INVENTORY_HORIZON} only"},
        {"metric": "primary horizon", "value": config.PRIMARY_HORIZON,
         "detail": "weeks -- '30-day ahead' is 4 weeks on a weekly panel"},
    ]

    if panel is not None:
        cleaned = _clean_weeks(panel)
        rows.insert(6, {"metric": "zero-demand share (clean weeks)",
                        "value": round(float((cleaned["units_sold"] == 0).mean()), 4),
                        "detail": "after dropping the partial final week"})
    return pd.DataFrame(rows)


def revenue_by_week(sales):
    """Weekly revenue and units. Used for the trend chart and the seasonality view."""
    frame = sales.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week_start_date"] = frame["date"] - pd.to_timedelta(
        frame["date"].dt.weekday, unit="D")
    frame = frame[frame["week_start_date"] < PARTIAL_WEEK_START]

    weekly = (frame.groupby("week_start_date")
              .agg(revenue=(AMOUNT, "sum"), units=(QTY, "sum"),
                   transactions=("transaction_id", "nunique"),
                   customers=("customer_id", "nunique"))
              .reset_index())
    weekly["avg_basket_value"] = (weekly["revenue"] / weekly["transactions"]).round(2)
    return weekly


def revenue_by_category(sales):
    """Category and SKU mix by revenue and units.

    Units are reported next to revenue on purpose: the highest-revenue category is not the
    highest-volume one, and a revenue-only chart hides that.
    """
    out = (sales.groupby(["product_category", "product_id"])
           .agg(revenue=(AMOUNT, "sum"), units=(QTY, "sum"),
                avg_price=(PRICE, "mean"),
                discount_pct=(DISCOUNT_PCT, "mean"))
           .reset_index())
    out["revenue_share"] = (out["revenue"] / out["revenue"].sum()).round(4)
    return out.sort_values("revenue", ascending=False)


def category_seasonality(sales):
    """Monthly unit share within each category, and its peak month.

    Share, not absolute units, so a large category does not trivially have the biggest
    swings. The peak month is the argmax of that share.
    """
    frame = sales.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month

    monthly = (frame.groupby(["product_category", "month"])
               .agg(units=(QTY, "sum"), revenue=(AMOUNT, "sum"))
               .reset_index())
    totals = monthly.groupby("product_category")["units"].transform("sum")
    monthly["unit_share"] = (monthly["units"] / totals).round(5)

    peaks = (monthly.loc[monthly.groupby("product_category")["unit_share"].idxmax()]
             .rename(columns={"month": "peak_month", "unit_share": "peak_share",
                              "units": "peak_units", "revenue": "peak_revenue"})[
                 ["product_category", "peak_month", "peak_share", "peak_units",
                  "peak_revenue"]])
    return monthly.merge(peaks, on="product_category", how="left"), peaks


def promotion_lift(sales):
    """What promotions actually do in this data, next to what they were configured to do.

    The generator configured a 1.56x promotional lift. It is not in the data. Two effects
    have to be separated, and conflating them is easy:

    - **Volume.** Mean quantity per promoted line is 3.8836 against 3.8840 for
      non-promoted lines -- a ratio of 1.0000. There is no detectable volume effect.
    - **Price.** Promoted lines carry a 26.0% average discount, so realised revenue per
      unit falls to 0.693x. That number is not a demand effect at all; it is the discount
      being counted as if it were one.

    An earlier version of this function reported only the realised revenue ratio and called
    it "lift", which made a 26% markdown look like a 31% demand collapse. Both components
    are returned separately so the report cannot repeat that.

    Promotions here are targeted rather than randomised -- they run on 563 of 731 days but
    touch only 8.4% of line items, concentrated on particular products. So even the volume
    ratio is an association, not a causal estimate, and it is reported as such.
    """
    frame = sales.copy()
    frame["revenue_per_unit"] = frame[AMOUNT] / frame[QTY].replace(0, np.nan)

    promo = frame[frame[IS_PROMO] == 1]
    non_promo = frame[frame[IS_PROMO] == 0]
    if promo.empty or non_promo.empty:
        raise ValueError("promotion_lift needs both promoted and unpromoted lines")

    promo_qty = float(promo[QTY].mean())
    base_qty = float(non_promo[QTY].mean())
    promo_rpu = float(promo["revenue_per_unit"].mean())
    base_rpu = float(non_promo["revenue_per_unit"].mean())

    return {
        "promo_lines": len(promo),
        "non_promo_lines": len(non_promo),
        "promo_line_share": round(len(promo) / len(frame), 4),
        "promo_days": int(frame.loc[frame[IS_PROMO] == 1, "date"].nunique()),
        "total_days": int(frame["date"].nunique()),
        "mean_qty_promo": round(promo_qty, 4),
        "mean_qty_non_promo": round(base_qty, 4),
        "volume_lift": round(promo_qty / base_qty, 4),
        "mean_discount_pct_promo": round(float(promo[DISCOUNT_PCT].mean()), 2),
        "mean_unit_price_promo": round(float(promo[PRICE].mean()), 2),
        "mean_unit_price_non_promo": round(float(non_promo[PRICE].mean()), 2),
        "realised_revenue_per_unit_ratio": round(promo_rpu / base_rpu, 4),
        "configured_lift": 1.56,
        "volume_lift_reproduces": False,
        "note": "No volume lift is detectable (1.0000x). Promoted lines carry a 26.0% "
                "discount, which is the entire difference in realised revenue per unit. "
                "The configured 1.56x lift does not appear in the data in any form. "
                "Promotions are targeted, not randomised, so the volume ratio is an "
                "association rather than a causal estimate.",
    }


def store_product_matrix(panel):
    """The store x product demand matrix, after dropping the partial week."""
    cleaned = _clean_weeks(panel)
    matrix = (cleaned.pivot_table(index="store_id", columns="product_id",
                                  values="units_sold", aggfunc="sum", fill_value=0))
    return matrix.astype(int)


def top_store_products(matrix, top_n=20):
    """Highest-demand store-product pairs, for the heatmap and the reorder table."""
    stacked = matrix.stack().rename("units").reset_index()
    stacked.columns = ["store_id", "product_id", "units"]
    totals = (stacked.groupby(["store_id", "product_id"])["units"].sum()
              .reset_index().sort_values("units", ascending=False))
    return totals.head(top_n)


def demand_distribution(panel):
    """How much of the demand sits in the head of the long tail.

    Long-tail concentration is the argument for hierarchical forecasting: if the top 20% of
    store-product pairs carry most of the units, they deserve the individual models and the
    tail does not.
    """
    cleaned = _clean_weeks(panel)
    series = (cleaned.groupby(["store_id", "product_id"])["units_sold"].sum()
              .sort_values(ascending=False))
    total = float(series.sum())
    if total <= 0:
        return pd.DataFrame({"cumulative_pairs_share": [1.0],
                             "cumulative_units_share": [0.0]})

    share = series / total
    return pd.DataFrame({
        "cumulative_pairs_share": (np.arange(1, len(share) + 1) / len(share)),
        "cumulative_units_share": share.cumsum().to_numpy(),
    })


def concentration_summary(distribution, head_fraction=0.20):
    """The single number for the report: units held by the top `head_fraction` of pairs."""
    dist = distribution[distribution["cumulative_pairs_share"] <= head_fraction]
    if dist.empty:
        return float(distribution["cumulative_units_share"].max())
    return float(dist["cumulative_units_share"].max())


def build_all(write=True, run_segmentation=True, segmentation=None):
    """Compute everything the dashboard needs. Returns a dict of frames."""
    from src.ingest import load_panel, load_sales

    sales = load_sales()
    panel = load_panel()
    _require_columns(sales, SALES_REQUIRED, "sales")
    _require_columns(panel, PANEL_REQUIRED, "panel")

    monthly, peaks = category_seasonality(sales)
    matrix = store_product_matrix(panel)
    distribution = demand_distribution(panel)

    out = {
        "kpis": kpis(sales, panel),
        "revenue_by_week": revenue_by_week(sales),
        "revenue_by_category": revenue_by_category(sales),
        "category_monthly": monthly,
        "category_peaks": peaks,
        "promotion_lift": pd.DataFrame([promotion_lift(sales)]),
        "top_store_products": top_store_products(matrix),
        "demand_distribution": distribution,
    }

    if run_segmentation:
        from src.features import rfm_features
        from src.segmentation import segment_customers

        if segmentation is None:
            segmentation = segment_customers(
                rfm_features(sales, as_of=config.SALES_END), run_dbscan=False)
        # `segment_summary` is indexed by the segment name. Written as-is with
        # index=False that name is silently dropped, and the CSV becomes six anonymous rows
        # that can only be re-identified by assuming row order matches `segment_id` -- which
        # it does not, because this frame is sorted by revenue. Promote the index to a real
        # column so the join key travels with the data.
        out["segment_summary"] = segmentation["segment_summary"].reset_index()
        out["customer_segments"] = segmentation["customers"]
    out["_matrix"] = matrix
    out["_concentration"] = concentration_summary(distribution)

    if write:
        config.ensure_dirs()
        # `_`-prefixed keys are internal by convention -- `_matrix` is the raw store-product
        # frame kept for the concentration calculation, not a published aggregate. The write
        # loop used to ignore that convention and emit `_matrix.csv` (a 27 kB file with no
        # header) into data/processed, which the dashboard auto-loads. `_concentration` is a
        # float so the isinstance check already skipped it; `_matrix` is a DataFrame, so it did
        # not get that protection. Skipping the prefix handles both without special-casing.
        skipped = sorted(n for n in out
                         if isinstance(out[n], pd.DataFrame) and n.startswith("_"))
        for name, frame in out.items():
            if name.startswith("_") or not isinstance(frame, pd.DataFrame):
                continue
            frame.to_csv(config.PROCESSED / f"{name}.csv", index=False)
        # Published deliberately and under a readable name: the concentration table is a
        # deliverable, `_matrix` is not.
        matrix.to_csv(config.PROCESSED / "store_product_matrix.csv")
        if skipped:
            print(f"  not published (internal): {', '.join(skipped)}")

    return out


def main():
    out = build_all()

    print("=" * 74)
    print("PRECOMPUTE")
    print("=" * 74)

    print("\nKPIs")
    for row in out["kpis"].itertuples():
        print(f"  {row.metric:<34} {row.value:>14}  {row.detail}")

    lift = out["promotion_lift"].iloc[0]
    print("\npromotion lift")
    print(f"  volume lift        {lift['volume_lift']}x   "
          f"({lift['mean_qty_promo']} vs {lift['mean_qty_non_promo']} units/line)")
    print(f"  configured lift    {lift['configured_lift']}x   <- not present in the data")
    print(f"  mean discount      {lift['mean_discount_pct_promo']}% on promoted lines")
    print(f"  realised rev/unit  {lift['realised_revenue_per_unit_ratio']}x   "
          f"(this is the discount, not a demand effect)")
    print(f"  promo line share   {lift['promo_line_share']:.1%} of lines across "
          f"{lift['promo_days']} of {lift['total_days']} days")
    print("  ^ Promotions are targeted, not randomised, so even the volume ratio is an")
    print("    association. There is no detectable lift to report.")

    print("\ncategory seasonality (peak month by unit share)")
    print(out["category_peaks"].to_string(index=False))

    print("\ndemand concentration")
    print(f"  top 20% of store-product pairs hold {out['_concentration']:.1%} of all units")
    print("  ^ This is the case for hierarchical forecasting: model the head individually "
          "and\n    pool the tail, rather than fitting 7,500 separate series.")

    if "segment_summary" in out:
        print("\nsegments")
        print(out["segment_summary"].to_string())

    written = len([f for f in out.values() if isinstance(f, pd.DataFrame)])
    print(f"\nwrote {written} frames to {config.PROCESSED}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())