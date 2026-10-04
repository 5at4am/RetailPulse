"""Walk-forward inventory backtest: the replay that backs the brief's 25-40% claim.

These tests exist because a backtest is the easiest place in this project to fool yourself.
The failure mode is not a crash, it is a plausible-looking reduction percentage produced by a
leak, a wrong denominator, or an imputed value scaled wrong. So each test below pins one
property that must hold for the number to mean anything:

  * no leakage -- the realised week joins the history only after it has been predicted
  * the baseline shares the forecast, so only the safety-stock method differs
  * unmet demand is imputed per pair, not per week
  * the reduction is computed against the stated baseline and is reproducible
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.inventory import (
    backtest_policy,
    backtest_reduction,
    poisson_safety_stock,
    safety_factor,
)


def make_panel(weeks=40, pairs=12, seed=0, stockout_share=0.25, base_rate=40.0):
    """A weekly pair panel with a controllable share of censored (stockout) weeks.

    `stockout_share` is the fraction of pair-weeks recorded as a stockout. It is the whole
    reason this fixture exists: a panel with no censoring cannot tell a good policy from a
    lucky one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2023-01-02")
    for w in range(weeks):
        week = start + pd.Timedelta(weeks=w)
        # A mild upward trend, so a mean-only forecast is not trivially perfect and a
        # walk-forward replay has something to get wrong.
        level = base_rate * (1.0 + 0.01 * w)
        for p in range(pairs):
            stockout = rng.random() < stockout_share
            units = int(max(rng.poisson(level) - (level if stockout else 0), 0))
            rows.append({
                "week_start_date": week,
                "store_id": f"S{p % 3}",
                "product_id": f"P{p}",
                "category": "CatA",
                "units_sold": units,
                "on_hand_end": units + 5,
                "stockout_count": int(stockout),
                "avg_unit_price": 100.0,
            })
    return pd.DataFrame(rows)


class TestPoissonSafetyStock:
    def test_scales_with_the_square_root_of_the_horizon(self):
        one = poisson_safety_stock(100, lead_time_units=1, critical_ratio=0.99)
        four = poisson_safety_stock(100, lead_time_units=4, critical_ratio=0.99)
        # Poisson variance grows linearly with the horizon, so the safety term grows as its
        # square root: 4x the horizon is 2x the safety stock. A linear increase would mean
        # the variance term is missing entirely.
        #
        # Not an exact equality because the quantile is an integer. At a mean of 100 the
        # 0.99 quantile lands on a whole unit, so the ratio carries a unit of rounding --
        # about 2% here.
        assert four == pytest.approx(2 * one, rel=0.05)

    def test_is_zero_for_zero_demand(self):
        assert poisson_safety_stock(0, lead_time_units=1, critical_ratio=0.992840) == 0.0

    def test_is_never_negative(self):
        assert poisson_safety_stock(5, lead_time_units=3, critical_ratio=0.0) == 0.0

    def test_higher_critical_ratio_means_more_cover(self):
        loose = poisson_safety_stock(100, lead_time_units=2, critical_ratio=0.99)
        tight = poisson_safety_stock(100, lead_time_units=2, critical_ratio=0.80)
        assert loose > tight

    def test_requires_an_explicit_critical_ratio(self):
        # A silent default here would let a caller ship whatever ratio the module happened
        # to use last, and the reorder quantity would change without anyone noticing.
        with pytest.raises(ValueError, match="critical_ratio"):
            poisson_safety_stock(100)

    def test_is_conservative_against_the_normal_approximation(self):
        """The exact Poisson quantile covers MORE than the normal z at this confidence.

        Worth pinning because it means the Poisson policy is deliberately over-covering
        relative to the textbook z-multiple -- so a "Poisson is tighter" claim would be
        false. At this critical ratio the exact quantile is ~25 units against the normal
        approximation's ~24.5. The gap is small and in the safe direction, which is the
        correct one for a right-skewed demand distribution.
        """
        poisson = poisson_safety_stock(100, lead_time_units=1, critical_ratio=0.992840)
        normal = safety_factor(0.992840) * 100**0.5
        assert poisson > normal
        assert poisson == pytest.approx(normal, rel=0.05)   # same order of magnitude


# Module-scoped rather than class-scoped: pytest 8.4 raises on class-scoped fixtures defined
# as instance methods, and the replay is expensive enough to be worth computing once.
@pytest.fixture(scope="module")
def backtest():
    return backtest_policy(make_panel(), progress=lambda *_: None)


@pytest.fixture(scope="module")
def summary():
    return backtest_reduction(backtest_policy(make_panel(), progress=lambda *_: None))


class TestBacktestPolicy:
    def test_returns_all_three_policies(self, backtest):
        assert set(backtest["policy"]) == {
            "policy_poisson", "naive_mean", "no_safety_stock",
        }

    def test_every_policy_is_replayed_on_the_same_weeks(self, backtest):
        # Same weeks per policy means the comparison is like-for-like. If one policy had
        # fewer rows it would look better purely by being evaluated on less demand.
        counts = backtest.groupby("policy")["week_start_date"].nunique()
        assert counts.nunique() == 1, f"policies replayed on different windows: {counts.to_dict()}"

    def test_policies_share_one_forecast_per_week(self, backtest):
        # Only the safety-stock method may differ. A different point forecast per policy
        # would confound the comparison entirely.
        for _, group in backtest.groupby("week_start_date"):
            assert group["forecast_units"].nunique() == 1
            assert group["actual_units"].nunique() == 1

    def test_no_safety_stock_orders_exactly_the_forecast(self, backtest):
        naked = backtest[backtest["policy"] == "no_safety_stock"]
        assert (naked["cover_units"] == naked["forecast_units"]).all()

    def test_surplus_and_shortfall_on_the_same_units_are_mutually_exclusive(self, backtest):
        """Both CAN be non-zero in one row, and that is correct.

        Overstock is surplus against the pooled forecast for the week; unmet demand comes
        from a *different* set of pairs -- the censored ones. A week can therefore be long on
        forecast and short on stock at the same time, and adding them is right.

        What must never happen is both the surplus and the shortfall term against the same
        units, which would double count. So assert that the cover-vs-actual split is a true
        partition of the difference, rather than asserting zero overlap.
        """
        surplus = (backtest["cover_units"] - backtest["actual_units"]).clip(lower=0)
        shortfall = (backtest["actual_units"] - backtest["cover_units"]).clip(lower=0)
        assert (backtest["overstock_units"] == surplus).all()
        # understock is the forecast shortfall PLUS unmet demand on censored pairs, so the
        # shortfall term must be present and must not exceed the total.
        assert (backtest["understock_units"] >= shortfall - 1e-9).all()
        residual = (backtest["understock_units"] - shortfall).to_numpy()
        # numpy rather than pytest.approx here: approx on a zero-heavy Series does not
        # report a useful tolerance failure, and this comparison is exact to 1e-14.
        assert np.allclose(residual, backtest["unmet_units"].to_numpy(),
                           rtol=0, atol=1e-9)

    def test_no_safety_stock_is_tightest(self, backtest):
        # With identical forecasts, zero safety stock is the lowest cover. Confirms the
        # ordering of the three policies before any accuracy claim is made.
        cover = backtest.groupby("policy")["cover_units"].mean()
        assert cover["no_safety_stock"] < cover["policy_poisson"]
        assert cover["no_safety_stock"] < cover["naive_mean"]

    def test_censored_weeks_are_charged_unmet_demand_not_treated_as_served(self, backtest):
        # The core honesty check. A week with censored pairs must contribute positive unmet
        # demand; if it contributed zero, the policy would look perfect on a stockout.
        censored = backtest[backtest["censored_pairs"] > 0]
        assert not censored.empty, "fixture produced no censored weeks, so nothing is tested"
        assert (censored["unmet_units"] > 0).all()

    def test_weeks_with_no_censoring_have_no_unmet_demand(self, backtest):
        clean = backtest[backtest["censored_pairs"] == 0]
        assert (clean["unmet_units"] == 0).all()

    def test_unmet_demand_is_imputed_per_pair_not_per_week(self, backtest):
        """Pin the dimensional fix.

        `mean` is the pooled weekly total. The per-pair rate is that total divided by the
        number of PAIRS. Dividing by the number of training weeks instead overstates every
        censored pair by the panel length -- roughly 86x on the full panel -- which would
        inflate understock for all three policies and make the reduction meaningless.
        """
        panel = make_panel()
        result = backtest_policy(panel, progress=lambda *_: None)
        n_pairs = panel[["store_id", "product_id"]].drop_duplicates().shape[0]

        week = result["week_start_date"].min()
        row = result[(result["week_start_date"] == week)
                     & (result["policy"] == "policy_poisson")].iloc[0]
        pooled = float(row["forecast_units"])
        expected = row["censored_pairs"] * (pooled / n_pairs)
        assert row["unmet_units"] == pytest.approx(expected, rel=1e-9)

    def test_history_grows_only_after_the_prediction(self):
        """No leakage, checked behaviourally rather than by reading the loop.

        Rewriting the final week of the holdout to an absurd value must change the results
        for that week (it is the actual) but must not change the forecasts for any earlier
        week. If the realised week leaked into the history, its rewrite would propagate
        backwards and the later forecasts would move too.
        """
        panel = make_panel()
        base = backtest_policy(panel, progress=lambda *_: None)

        weeks = sorted(panel["week_start_date"].unique())
        target = weeks[-1]                       # last holdout week
        boosted = panel.copy()
        boosted.loc[boosted["week_start_date"] == target, "units_sold"] *= 50

        after = backtest_policy(boosted, progress=lambda *_: None)

        base_forecasts = (base[base["policy"] == "policy_poisson"]
                          .set_index("week_start_date")["forecast_units"])
        after_forecasts = (after[after["policy"] == "policy_poisson"]
                           .set_index("week_start_date")["forecast_units"])

        for week in base_forecasts.index:
            if week == target:
                continue                         # this week's actual changed, by design
            assert base_forecasts[week] == pytest.approx(after_forecasts[week], rel=1e-9), (
                f"forecast for {week.date()} moved when only {target.date()} changed, "
                f"so the realised week leaked into the training history"
            )

    def test_is_deterministic(self):
        first = backtest_policy(make_panel(seed=3), progress=lambda *_: None)
        second = backtest_policy(make_panel(seed=3), progress=lambda *_: None)
        pd.testing.assert_frame_equal(first, second)

    def test_refuses_a_panel_too_short_to_replay(self):
        with pytest.raises(ValueError, match="at least 10"):
            backtest_policy(make_panel(weeks=6), progress=lambda *_: None)

    def test_honours_an_explicit_train_weeks(self):
        # The brief's 52/13 split has to remain reachable even though the default adapts.
        panel = make_panel(weeks=70)
        result = backtest_policy(panel, train_weeks=52, holdout_weeks=13,
                                 progress=lambda *_: None)
        assert result["week_start_date"].nunique() == 13

    def test_excludes_the_partial_final_week(self):
        # config.PARTIAL_FINAL_WEEK is a truncated week. Replaying it would score the
        # policy against demand that was never fully observed.
        panel = make_panel(weeks=40)
        result = backtest_policy(panel, progress=lambda *_: None)
        assert (result["week_start_date"] < pd.Timestamp(config.PARTIAL_FINAL_WEEK)).all()


class TestBacktestReduction:
    def test_returns_the_flat_summary_the_dashboard_reads(self, summary):
        # A flat dict of scalars, not a per-policy nested structure: this is written straight
        # to inventory_backtest_summary.csv, so its shape is a contract with the dashboard.
        assert {"baseline", "holdout_weeks", "baseline_error_units", "policy_error_units",
                "error_reduction_pct", "meets_25_40_target", "note"} <= set(summary)
        assert summary["baseline"] == "naive_mean"
        assert isinstance(summary["meets_25_40_target"], bool)

    def test_reduction_is_computed_against_the_naive_baseline(self, summary):
        expected = (100 * (summary["baseline_error_units"] - summary["policy_error_units"])
                    / summary["baseline_error_units"])
        assert summary["error_reduction_pct"] == pytest.approx(expected, abs=0.05)

    def test_observed_baseline_is_a_different_and_looser_comparison(self):
        # Doing nothing must be the weakest baseline, so the policy can only look better
        # against it. Pinning this stops a later edit quietly switching the default.
        result = backtest_policy(make_panel(), progress=lambda *_: None)
        naive = backtest_reduction(result, baseline="naive_mean")
        observed = backtest_reduction(result, baseline="observed")
        assert observed["baseline"] == "observed"
        assert observed["error_reduction_pct"] >= naive["error_reduction_pct"]

    def test_rejects_an_unknown_baseline(self):
        with pytest.raises(ValueError, match="unknown baseline"):
            backtest_reduction(backtest_policy(make_panel(), progress=lambda *_: None),
                               baseline="wishful_thinking")

    def test_requires_the_policy_row(self):
        with pytest.raises(ValueError, match="policy_poisson"):
            backtest_reduction(pd.DataFrame({"week_start_date": pd.to_datetime(["2023-01-02"]),
                                              "policy": ["naive_mean"],
                                              "overstock_units": [1.0],
                                              "understock_units": [1.0],
                                              "unmet_units": [0.0],
                                              "cover_units": [1.0],
                                              "actual_units": [1.0]}))

    def test_target_flag_agrees_with_the_reduction(self, summary):
        reduction = summary["error_reduction_pct"] / 100
        assert summary["meets_25_40_target"] == bool(0.25 <= reduction <= 0.40)

    def test_note_says_this_is_not_a_production_result(self, summary):
        # The claim is a historical replay. If someone later removes this caveat the brief's
        # 25-40% figure starts reading as a measured production improvement.
        assert "walk-forward" in summary["note"]
        assert "not a production" in summary["note"].lower()

    def test_does_not_divide_by_a_zero_baseline(self):
        # A flat panel gives zero error under every policy. The reduction is then undefined,
        # and it must come back finite rather than as inf or nan -- an inf in the summary CSV
        # would render in the dashboard as a division warning.
        flat = pd.DataFrame({
            "week_start_date": pd.to_datetime(["2023-01-02", "2023-01-09"]),
            "policy": ["policy_poisson", "naive_mean"],
            "overstock_units": [0.0, 0.0],
            "understock_units": [0.0, 0.0],
            "unmet_units": [0.0, 0.0],
            "cover_units": [0.0, 0.0],
            "actual_units": [0.0, 0.0],
        })
        reduction = backtest_reduction(flat)["error_reduction_pct"]
        assert np.isfinite(reduction), (
            f"a zero baseline yielded {reduction}, which would surface as a division "
            f"warning in the dashboard"
        )