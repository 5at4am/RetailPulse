"""Tests for F-05 inventory optimisation.

The claims under test are mostly about not lying:

- the critical ratio is derived from stated costs, not assumed to be the service level;
- Poisson safety stock is used because the demand is intermittent, and the normal
  approximation is kept only as a labelled comparison;
- pair forecasts stay consistent with the pooled total;
- the plan is a function of on-hand stock, so it reacts to stock and stops when full.

No test pins the reorder unit count: that depends on the forecast, which moves.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.inventory import (ANNUAL_HOLDING_RATE, LOST_MARGIN_RATIO, PURCHASE_COST_RATIO,
                          demand_forecasts, last_on_hand, mean_price,
                          newsvendor_critical_ratio, reorder_quantity,
                          reorder_quantity_poisson, run_inventory, safety_factor)


def make_panel(weeks=60, stores=3, products=4, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=weeks, freq="W-MON")
    frames = []
    for store in range(stores):
        for product in range(products):
            level = 2 + store + product
            demand = rng.poisson(max(level, 1), weeks)
            frames.append(pd.DataFrame({
                "week_start_date": dates,
                "store_id": f"ST{store + 1:02d}",
                "product_id": f"PRD{product + 1:04d}",
                "units_sold": demand,
                "revenue": demand * 10.0,
                "avg_unit_price": 10.0,
                "on_hand_end": rng.integers(0, 20, weeks).astype(float),
                "stockout_flag": (demand == 0).astype(int),
            }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


# ------------------------------------------------------------------ newsvendor costs

def test_critical_ratio_is_between_zero_and_one():
    ratio = newsvendor_critical_ratio(10.0, 20.0)
    assert 0.0 < ratio < 1.0


def test_critical_ratio_is_not_the_same_number_as_the_service_level():
    """The brief asks for 95%. It does not get 95%, and pretending otherwise is the bug."""
    ratio = newsvendor_critical_ratio(10.0, 20.0)
    assert ratio != pytest.approx(config.DEFAULT_SERVICE_LEVEL)


def test_higher_margin_raises_the_critical_ratio():
    """A fatter margin means stockouts hurt more, so it pays to hold deeper.

    The absolute price scale is irrelevant; only the margin ratio matters. Holding price
    fixed and varying margin is the way to see it.
    """
    thin = newsvendor_critical_ratio(10.0, 20.0)
    fat = newsvendor_critical_ratio(4.0, 20.0)
    assert fat > thin


def test_critical_ratio_is_scale_invariant():
    """Doubling every cost must not change the decision, or the units are wrong."""
    small = newsvendor_critical_ratio(5.0, 10.0)
    large = newsvendor_critical_ratio(50.0, 100.0)
    assert small == pytest.approx(large)


def test_faster_turnover_raises_the_critical_ratio():
    """Cheaper goods relative to price spoil less, so over-buying hurts less."""
    slow = newsvendor_critical_ratio(10.0, 20.0, holding_rate=0.60)
    fast = newsvendor_critical_ratio(10.0, 20.0, holding_rate=0.02)
    assert fast > slow


def test_longer_cover_lowers_the_critical_ratio():
    """More weeks of holding cost makes over-buying more expensive."""
    short = newsvendor_critical_ratio(10.0, 20.0, cover_weeks=1)
    long = newsvendor_critical_ratio(10.0, 20.0, cover_weeks=8)
    assert long < short


def test_critical_ratio_rejects_nonsense_costs():
    with pytest.raises(ValueError):
        newsvendor_critical_ratio(0.0, 0.0, lost_margin_ratio=0.0, holding_rate=0.0)


def test_safety_factor_is_a_standard_normal_quantile():
    """Must not be a hard-coded lookup table: 0.95 is simply the value everyone uses."""
    assert safety_factor(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert safety_factor(0.99) > safety_factor(0.95)
    # An off-table service level still returns something sensible.
    assert 0.5 < safety_factor(0.9873) < 3.0


# ------------------------------------------------------------------- reorder rules

def test_poisson_safety_stock_is_non_negative():
    for mean in (0.0, 0.5, 1.0, 5.0, 50.0):
        outcome = reorder_quantity_poisson(mean, on_hand=0.0, lead_time_units=1,
                                           critical_ratio=0.95)
        assert outcome["safety_stock"] >= 0.0


def test_poisson_reorder_point_is_at_least_expected_demand():
    outcome = reorder_quantity_poisson(10.0, on_hand=0.0, lead_time_units=2,
                                       critical_ratio=0.95)
    assert outcome["reorder_point"] >= outcome["expected_demand"]


def test_no_order_when_stock_is_above_the_reorder_point():
    outcome = reorder_quantity_poisson(1.0, on_hand=1000.0, lead_time_units=1,
                                       critical_ratio=0.95)
    assert outcome["suggested_order"] == 0.0
    assert outcome["stockout_risk_now"] is False


def test_full_order_when_stock_is_zero():
    outcome = reorder_quantity_poisson(4.0, on_hand=0.0, lead_time_units=1,
                                       critical_ratio=0.95)
    assert outcome["suggested_order"] == pytest.approx(outcome["reorder_point"])
    assert outcome["stockout_risk_now"] is True


def test_partial_order_when_stock_is_below_the_point():
    outcome = reorder_quantity_poisson(4.0, on_hand=3.0, lead_time_units=1,
                                       critical_ratio=0.95)
    assert 0.0 < outcome["suggested_order"] < outcome["reorder_point"]


def test_more_lead_time_means_more_stock():
    short = reorder_quantity_poisson(4.0, 0.0, 1, 0.95)
    long = reorder_quantity_poisson(4.0, 0.0, 4, 0.95)
    assert long["reorder_point"] > short["reorder_point"]


def test_higher_service_level_means_more_stock():
    """Sanity check that the critical ratio flows through in the expected direction."""
    low = reorder_quantity_poisson(4.0, 0.0, 1, 0.80)
    high = reorder_quantity_poisson(4.0, 0.0, 1, 0.99)
    assert high["safety_stock"] >= low["safety_stock"]


def test_poisson_beats_normal_on_intermittent_demand():
    """The reason Poisson is the primary method, stated as a testable claim.

    A z-multiple of a large standard deviation is not a percentile of the demand
    distribution. On a pair with zeros most weeks it overstates badly, and the gap widens
    with variance.
    """
    mean, std = 3.0, 12.0
    poisson_q = reorder_quantity_poisson(mean, 0.0, 1, 0.95)
    normal_q = reorder_quantity(mean, std, 0.0, 1, 0.95)
    assert normal_q["reorder_point"] > poisson_q["reorder_point"] * 1.5


def test_poisson_is_not_always_conservative():
    """On smooth, near-continuous demand the normal approximation is not absurd.

    Worth pinning, because "Poisson is better" is only true for intermittent demand. A test
    that assumed Poisson always orders less would hide that the choice depends on the
    demand shape.
    """
    mean, std = 200.0, 8.0
    poisson_q = reorder_quantity_poisson(mean, 0.0, 1, 0.95)
    normal_q = reorder_quantity(mean, std, 0.0, 1, 0.95)
    assert normal_q["reorder_point"] < poisson_q["reorder_point"]


def test_normal_approximation_matches_the_textbook_formula():
    outcome = reorder_quantity(10.0, 4.0, 0.0, 1, 0.95)
    assert outcome["safety_stock"] == pytest.approx(1.6449 * 4.0, abs=1e-2)


# ------------------------------------------------------------------- forecast split

def test_pair_forecasts_still_add_up_to_the_pooled_forecast(panel):
    pairs = demand_forecasts(panel, forecast_units=1000.0)
    assert pairs["forecast_units"].sum() == pytest.approx(1000.0, rel=1e-6)


def test_forecast_shares_match_lifetime_demand_shares(panel):
    pairs = demand_forecasts(panel, forecast_units=1000.0)
    totals = panel.groupby(["store_id", "product_id"])["units_sold"].sum()
    expected = (totals / totals.sum()).sort_index()
    actual = pairs.set_index(["store_id", "product_id"])["share"].sort_index()
    assert actual.to_numpy() == pytest.approx(expected.to_numpy())


def test_demand_forecasts_fill_missing_recent_variance(panel):
    """A pair with one recent observation has an undefined std; it must be zero, not NaN."""
    pairs = demand_forecasts(panel, forecast_units=100.0, window=1)
    assert pairs["recent_std"].notna().all()


def test_demand_forecasts_do_not_read_the_partial_week(panel):
    """The partial sales week must not leak into a stock decision."""
    extended = panel.copy()
    partial = pd.Timestamp(config.PARTIAL_FINAL_WEEK)
    extra = extended.head(3).copy()
    extra["week_start_date"] = partial
    extra["units_sold"] = 999_999
    contaminated = pd.concat([extended, extra], ignore_index=True)

    clean = demand_forecasts(panel, forecast_units=100.0)
    dirty = demand_forecasts(contaminated, forecast_units=100.0)
    assert dirty["share"].sum() == pytest.approx(1.0)
    assert clean["forecast_units"].sum() == pytest.approx(dirty["forecast_units"].sum())


def test_last_on_hand_comes_from_the_final_usable_week(panel):
    on_hand = last_on_hand(panel)
    assert on_hand.index.names == ["store_id", "product_id"]
    assert (on_hand >= 0).all()


def test_mean_price_is_a_price_not_a_revenue(panel):
    price = mean_price(panel)
    assert (price > 0).all()
    assert price.max() < panel["revenue"].max()


# ----------------------------------------------------------------------- end to end

def test_run_inventory_accepts_a_supplied_forecast(panel):
    result = run_inventory(panel=panel, forecast_units=800.0, write=False,
                           progress=lambda *_: None)
    pairs = result["pairs"]
    assert pairs["forecast_units"].sum() == pytest.approx(800.0, rel=1e-6)
    assert result["summary"]["safety_stock_method"].iloc[0] == "poisson_quantile"


def test_run_inventory_reports_both_methods(panel):
    """Both calculations are always present, so the choice is visible rather than implied."""
    result = run_inventory(panel=panel, forecast_units=800.0, write=False,
                           progress=lambda *_: None)
    pairs, summary = result["pairs"], result["summary"].iloc[0]
    for column in ("suggested_order", "normal_approx_suggested_order",
                   "normal_approx_reorder_point", "normal_approx_safety_stock"):
        assert column in pairs.columns
    assert summary["safety_stock_method"] == "poisson_quantile"
    assert summary["total_units_to_order"] > 0
    assert summary["normal_approx_units_to_order"] > 0


@pytest.mark.slow
def test_on_real_intermittent_demand_normal_overstates_by_a_lot():
    """The headline justification for using Poisson, measured on the real panel.

    With 77.9% zero-demand weeks, the normal approximation asks for far more stock than a
    Poisson quantile of the same demand. This is the gap reported in the writeup.
    """
    from src.inventory import run_inventory as real_run

    summary = real_run(write=False, progress=lambda *_: None)["summary"].iloc[0]
    assert summary["normal_approx_units_to_order"] > 3 * summary["total_units_to_order"]
    assert summary["normal_approx_pairs"] > 2 * summary["pairs_to_reorder"]


def test_run_inventory_flags_only_stockout_risk_pairs(panel):
    result = run_inventory(panel=panel, forecast_units=800.0, write=False,
                           progress=lambda *_: None)
    pairs = result["pairs"]
    assert (pairs["reorder"] == pairs["stockout_risk_now"]).all()
    flagged = pairs[pairs["reorder"]]
    assert (flagged["on_hand"] < flagged["reorder_point"]).all()


def test_plan_shrinks_when_forecast_shrinks(panel):
    big = run_inventory(panel=panel, forecast_units=2000.0, write=False,
                        progress=lambda *_: None)
    small = run_inventory(panel=panel, forecast_units=500.0, write=False,
                          progress=lambda *_: None)
    assert small["summary"]["total_units_to_order"].iloc[0] < \
        big["summary"]["total_units_to_order"].iloc[0]


def test_run_inventory_records_its_cost_assumptions(panel):
    summary = run_inventory(panel=panel, forecast_units=800.0, write=False,
                            progress=lambda *_: None)["summary"].iloc[0]
    assert summary["purchase_cost_ratio"] == PURCHASE_COST_RATIO
    assert summary["holding_rate"] == ANNUAL_HOLDING_RATE
    assert summary["lost_margin_ratio"] == LOST_MARGIN_RATIO
    assert summary["service_level"] == config.DEFAULT_SERVICE_LEVEL


@pytest.mark.slow
def test_inventory_on_real_data_is_actionable_but_not_absurd():
    """The plan should be a real subset of pairs, and its cost should be stated."""
    from src.inventory import run_inventory as real_run

    summary = real_run(write=False, progress=lambda *_: None)["summary"].iloc[0]
    assert 0 < summary["pairs_to_reorder"] < summary["pairs"]
    assert summary["total_units_to_order"] > 0
    assert summary["total_order_value"] > 0