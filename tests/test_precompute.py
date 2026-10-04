"""Tests for the dashboard aggregates.

The numbers here end up in the report and on the dashboard, so they are pinned rather than
recomputed on the fly. Several of these exist specifically to catch mistakes that were
actually made: using the panel's column names on the sales frame, reporting a 26% markdown
as though it were demand lift, and mis-stating the panel's shape.

Two shapes are easy to confuse and are pinned here:

- **Sales**: 250,000 rows, 50 stores, **1,200 products**, 10 categories, daily to 2025-12-31.
- **Panel**: 780,000 rows, 50 stores, **150 products** = **7,500 store-product series**,
  **104 weeks**, ending 2025-12-22.

So the panel already excludes the partial final week and covers only a subset of products;
the partial-week guard in `revenue_by_week` is what protects the sales side.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.precompute import (AMOUNT, IS_PROMO, PANEL_REQUIRED, PARTIAL_WEEK_START,
                            PRICE, QTY, SALES_REQUIRED, _clean_weeks, _require_columns,
                            category_seasonality, concentration_summary, demand_distribution,
                            kpis, promotion_lift, revenue_by_week, store_product_matrix,
                            top_store_products)

N_STORES = 50
N_PANEL_PRODUCTS = 150
N_SALES_PRODUCTS = 1_200
N_SERIES = 7_500
N_PANEL_WEEKS = 104


@pytest.fixture(scope="module")
def sales_df():
    from src.ingest import load_sales
    return load_sales()


@pytest.fixture(scope="module")
def panel_df():
    from src.ingest import load_panel
    return load_panel()


# --------------------------------------------------------------------------- helpers

def test_require_columns_fails_loudly_on_a_wrong_frame():
    """Sales and panel do not share column names; guessing must not fail silently."""
    panel_like = pd.DataFrame({"units_sold": [1], "revenue": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        _require_columns(panel_like, SALES_REQUIRED, "sales")

    sales_like = pd.DataFrame({QTY: [1], AMOUNT: [1.0]})
    with pytest.raises(ValueError, match="missing"):
        _require_columns(sales_like, PANEL_REQUIRED, "panel")


def test_require_columns_accepts_the_real_frames(sales_df, panel_df):
    _require_columns(sales_df, SALES_REQUIRED, "sales")
    _require_columns(panel_df, PANEL_REQUIRED, "panel")


# ----------------------------------------------------------------------- shape pinning

def test_sales_and_panel_shapes_are_pinned(sales_df, panel_df):
    """The two frames describe different product universes; conflating them is easy."""
    assert len(sales_df) == 250_000
    assert sales_df["store_id"].nunique() == N_STORES
    assert sales_df["product_id"].nunique() == N_SALES_PRODUCTS
    assert sales_df["product_category"].nunique() == 10

    assert len(panel_df) == 780_000
    assert panel_df["store_id"].nunique() == N_STORES
    assert panel_df["product_id"].nunique() == N_PANEL_PRODUCTS
    assert panel_df.groupby(["store_id", "product_id"]).ngroups == N_SERIES
    assert panel_df["week_start_date"].nunique() == N_PANEL_WEEKS


def test_panel_already_excludes_the_partial_week(panel_df):
    """The panel ends 2025-12-22, so the guard is a no-op there and only sales needs it."""
    assert pd.to_datetime(panel_df["week_start_date"]).max() < PARTIAL_WEEK_START
    assert len(_clean_weeks(panel_df)) == len(panel_df)


# ------------------------------------------------------------------------ partial week

def test_weekly_sales_rollup_drops_the_partial_week(sales_df):
    weekly = revenue_by_week(sales_df)
    assert weekly["week_start_date"].max() < PARTIAL_WEEK_START
    # 2025-12-29 would be the partial week; it must not appear.
    assert not (weekly["week_start_date"] == PARTIAL_WEEK_START).any()


def test_weekly_rollup_is_contiguous_seven_day_steps(sales_df):
    weekly = revenue_by_week(sales_df)
    gaps = weekly["week_start_date"].diff().dropna().dt.days.unique()
    assert set(gaps) == {7}
    assert len(weekly) == N_PANEL_WEEKS


def test_weekly_rollup_revenue_is_less_than_the_raw_total(sales_df):
    """Proof the partial week was actually excluded rather than silently kept."""
    weekly = revenue_by_week(sales_df)
    assert weekly["revenue"].sum() < sales_df[AMOUNT].sum()


# ------------------------------------------------------------------------------ demand

def test_store_product_matrix_totals_match_the_panel(panel_df):
    matrix = store_product_matrix(panel_df)
    assert matrix.to_numpy().sum() == int(panel_df["units_sold"].sum())


def test_store_product_matrix_is_50_by_150(panel_df):
    matrix = store_product_matrix(panel_df)
    assert matrix.shape == (N_STORES, N_PANEL_PRODUCTS)


def test_store_product_matrix_has_no_missing_cells(panel_df):
    """`pivot_table(fill_value=0)` can still leave gaps if an index/column pair never
    appears; a NaN here would silently poison the dashboard heatmap."""
    matrix = store_product_matrix(panel_df)
    assert not matrix.isna().any().any()


def test_top_store_products_is_sorted_and_capped(panel_df):
    top = top_store_products(store_product_matrix(panel_df), top_n=20)
    assert len(top) == 20
    assert top["units"].is_monotonic_decreasing
    assert (top["units"] >= 0).all()
    assert top["store_id"].nunique() <= N_STORES


def test_demand_distribution_is_a_monotone_cumulative_curve(panel_df):
    dist = demand_distribution(panel_df)
    assert len(dist) == N_SERIES
    assert dist["cumulative_units_share"].is_monotonic_increasing
    assert dist["cumulative_pairs_share"].is_monotonic_increasing
    assert dist["cumulative_units_share"].iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_concentration_summary_is_bounded(panel_df):
    value = concentration_summary(demand_distribution(panel_df))
    assert 0.0 < value <= 1.0


# -------------------------------------------------------------------------- promotions

def test_promotion_lift_separates_volume_from_the_discount(sales_df):
    """The 0.69x realised ratio is a 26% markdown, not a 31% demand collapse.

    This is the mistake that was actually made, so it is pinned directly: volume lift and
    realised revenue ratio must be two different numbers, and the volume one must sit at
    1.0 because no lift exists in this data.
    """
    result = promotion_lift(sales_df)
    assert result["volume_lift"] == pytest.approx(1.0, abs=0.01)
    assert result["realised_revenue_per_unit_ratio"] < 1.0
    assert result["mean_discount_pct_promo"] > 20
    assert result["realised_revenue_per_unit_ratio"] < result["volume_lift"]
    assert result["volume_lift_reproduces"] is False


def test_promotions_are_targeted_not_randomised(sales_df):
    """Promotions run almost every day but touch a small minority of lines.

    This is why the volume ratio is an association rather than a causal estimate, and the
    numbers have to show that skew for the caveat to be justified.
    """
    result = promotion_lift(sales_df)
    assert result["promo_days"] / result["total_days"] > 0.5
    assert result["promo_line_share"] < 0.20


def test_configured_lift_is_reported_alongside_the_measured_one(sales_df):
    result = promotion_lift(sales_df)
    assert result["configured_lift"] == 1.56
    assert result["volume_lift"] != pytest.approx(result["configured_lift"], rel=0.10)


def test_promotion_lift_needs_both_groups(sales_df):
    with pytest.raises(ValueError, match="both promoted and unpromoted"):
        promotion_lift(sales_df[sales_df[IS_PROMO] == 1])


# ------------------------------------------------------------------------ seasonality

def test_seasonality_shares_sum_to_one_per_category(sales_df):
    monthly, peaks = category_seasonality(sales_df)
    sums = monthly.groupby("product_category")["unit_share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-3)
    assert set(peaks["product_category"]) == set(monthly["product_category"].unique())
    assert len(peaks) == 10


def test_peak_month_is_the_argmax_of_unit_share(sales_df):
    monthly, peaks = category_seasonality(sales_df)
    for category in peaks["product_category"]:
        rows = monthly[monthly["product_category"] == category]
        best = rows.loc[rows["unit_share"].idxmax()]
        assert int(best["month"]) == int(
            peaks.loc[peaks["product_category"] == category, "peak_month"].iloc[0])


def test_peak_share_is_a_share_not_an_absolute_count(sales_df):
    _, peaks = category_seasonality(sales_df)
    assert (peaks["peak_share"] <= 0.5).all()
    assert (peaks["peak_units"] > 0).all()


# -------------------------------------------------------------------------------- KPIs

def test_kpis_report_the_numbers_the_brief_quotes(sales_df):
    indexed = kpis(sales_df, None).set_index("metric")["value"]
    assert indexed["customers"] == 8_000
    assert indexed["stores"] == N_STORES
    assert indexed["products"] == N_SALES_PRODUCTS
    # The brief states 87.3% repeat purchase; this counts distinct purchase dates.
    assert indexed["repeat purchase rate"] == pytest.approx(0.872, abs=0.002)


def test_kpi_values_are_scalars_not_series(sales_df):
    """A one-element array in a KPI cell renders as `[81234.5]` on the dashboard."""
    frame = kpis(sales_df, None)
    for value in frame["value"]:
        assert isinstance(value, (int, float, np.integer, np.floating, str))


def test_repeat_rate_is_not_transaction_count(sales_df):
    """Counting transactions instead of dates would inflate the number."""
    by_date = sales_df.groupby("customer_id")["date"].nunique().gt(1).mean()
    by_txn = sales_df.groupby("customer_id")["transaction_id"].nunique().gt(1).mean()
    assert by_date == pytest.approx(0.872, abs=0.002)
    assert by_txn >= by_date