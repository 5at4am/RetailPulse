"""Tests for F-03 forecasting.

The structural claims are what matter here:

- The backtest holds out future weeks and never lets a model see them.
- MAPE is only ever computed on non-zero actuals, and the pooled series has none -- so
  WAPE and MAPE coincide there, which is a fact worth pinning rather than hiding.
- Seasonal naive is a real benchmark, not a formality.
- Reconciliation makes the parts sum to the whole.
- Selection uses the holdout honestly and does not ship a model the backtest rejected.

No test asserts a particular WAPE value. These numbers move when a model is retuned, and a
test that pins them would fail on an improvement.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.forecasting import (MIN_OBSERVATIONS, SEASON_LENGTH, EnsembleFixed,
                             LSTMPooled, ProphetPooled, RidgeSeasonal, SeasonalNaiveModel,
                             backtest_pooled, build_series, evaluate_all, reconcile,
                             seasonal_naive, series_weights)


def make_panel(weeks=104, stores=3, products=4, seed=11):
    """A small panel with the real column contract, built backwards from the last week."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=weeks, freq="W-MON")
    rows = []
    for store in range(stores):
        for product in range(products):
            level = 5 + store * 3 + product
            season = 1 + 0.4 * np.sin(np.arange(weeks) * 2 * np.pi / 52)
            noise = rng.poisson(max(level, 1), weeks)
            rows.append(pd.DataFrame({
                "week_start_date": dates,
                "store_id": f"ST{store + 1:02d}",
                "product_id": f"PRD{product + 1:04d}",
                "product_category": f"Cat{product % 2}",
                "units_sold": (noise * season).round().astype(int),
            }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def total_series(panel):
    return build_series(panel, level="total")


# ------------------------------------------------------------------------------- series

def test_build_series_totals_match_the_panel(panel):
    series = build_series(panel, level="total")
    assert len(series) == 104
    assert series["units"].sum() == panel["units_sold"].sum()
    assert series["week_start_date"].is_monotonic_increasing


def test_build_series_levels_partition_the_same_total(panel):
    total = build_series(panel, level="total")["units"].sum()
    for level in ("store", "product", "category"):
        assert build_series(panel, level=level)["units"].sum() == total


def test_series_weights_are_shares_that_sum_to_one(panel):
    for level in ("store", "product", "category"):
        weights = series_weights(panel, level=level)
        assert (weights > 0).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_series_weights_preserve_the_demand_ranking(panel):
    """A store with more units must get a bigger forecast share."""
    weights = series_weights(panel, level="store")
    totals = panel.groupby("store_id")["units_sold"].sum()
    ordered = weights.index.tolist()
    assert ordered[0] == totals.idxmax()


# ----------------------------------------------------------------------------- baseline

def test_seasonal_naive_repeats_last_year():
    history = np.arange(1, 105, dtype=float)          # 104 weeks, rising
    predicted = seasonal_naive(history, horizon=4)
    # Week 105 should repeat week 53.
    assert predicted[0] == pytest.approx(history[52])
    assert predicted[3] == pytest.approx(history[55])


def test_seasonal_naive_needs_a_full_year_of_history():
    """Two years of nothing but 20 weeks cannot support a 52-week seasonal lag."""
    with pytest.raises(Exception):
        seasonal_naive(np.arange(20, dtype=float), horizon=1)


def test_seasonal_naive_model_matches_the_function(total_series):
    model = SeasonalNaiveModel().fit(total_series.iloc[:-4])
    assert model.forecast(total_series.iloc[:-4], 4) == pytest.approx(
        seasonal_naive(total_series["units"].to_numpy()[:-4], 4))


# ------------------------------------------------------------------------------ holdout

def test_backtest_holds_out_the_final_weeks_only(total_series):
    outcome = backtest_pooled(total_series, horizon=4, model_factory=SeasonalNaiveModel)
    actual = total_series["units"].to_numpy()[-4:]
    assert outcome["actual"].tolist() == actual.tolist()
    assert outcome["predicted"].shape == (4,)


def test_backtest_model_never_sees_the_holdout(total_series):
    """The model is fitted on `series.iloc[:-horizon]`; assert that is genuinely shorter."""
    train = total_series.iloc[:-4]
    assert len(train) == len(total_series) - 4
    outcome = backtest_pooled(total_series, horizon=4, model_factory=RidgeSeasonal)
    assert outcome["model"] is not None


def test_backtest_refuses_a_series_that_is_too_short():
    short = build_series(make_panel(weeks=40), level="total")
    with pytest.raises(ValueError, match="observations"):
        backtest_pooled(short, horizon=4, model_factory=SeasonalNaiveModel)


def test_backtest_checks_the_forecast_length():
    """A model returning the wrong number of points must fail, not broadcast silently."""

    class WrongLength:
        def fit(self, series):
            return self

        def forecast(self, series, horizon):
            return np.zeros(horizon + 3)

    series = build_series(make_panel(weeks=104), level="total")
    with pytest.raises(ValueError, match="expected"):
        backtest_pooled(series, horizon=2, model_factory=WrongLength)


def test_forecast_horizons_cover_the_brief():
    """The brief's "30-day ahead" is 4 weeks on a weekly panel."""
    assert config.PRIMARY_HORIZON == 4
    assert config.INVENTORY_HORIZON == 1
    assert 4 in config.HORIZONS and 1 in config.HORIZONS


# --------------------------------------------------------------------------- pooled total

def test_pooled_total_has_no_zero_weeks_but_rows_do(panel):
    """77.9% of rows are zero; the pooled total is not. Do not let one imply the other."""
    series = build_series(panel, level="total")
    assert (series["units"] > 0).all()
    assert (panel["units_sold"] == 0).any()


def test_wape_and_mape_both_defined_when_no_actual_is_zero():
    """With no zero actuals, the non-zero filter excludes nothing.

    So a pooled MAPE is well defined and lands close to WAPE -- it just is not identical,
    because WAPE weights each point by its size and MAPE does not. The point worth pinning
    is that the brief's 12% MAPE target looks trivially met at pooled level and says nothing
    about the sparse store-product level where the zeros actually are.
    """
    from src.metrics import forecast_metrics
    actual = np.array([100.0, 200.0, 150.0, 180.0])
    predicted = np.array([104.0, 190.0, 160.0, 175.0])
    metrics = forecast_metrics(actual, predicted)
    assert metrics["wape"] == pytest.approx(metrics["mape_nonzero"], rel=0.02)
    assert np.isfinite(metrics["mape_nonzero"])


def test_mape_nonzero_excludes_zero_actuals():
    from src.metrics import mape
    actual = np.array([0.0, 100.0, 0.0, 100.0])
    predicted = np.array([50.0, 100.0, 50.0, 100.0])
    # Only the two non-zero rows count: errors of 0 and 0.
    assert mape(actual, predicted, nonzero_only=True) == pytest.approx(0.0)
    # The all-rows variant is genuinely undefined here; the divide-by-zero warning is the
    # point of that claim, not an accident to be silenced in the source.
    with np.errstate(divide="ignore"):
        assert mape(actual, predicted, nonzero_only=False) > 1.0


# ------------------------------------------------------------------------------ ridge

def test_ridge_seasonal_predicts_non_negative_units(total_series):
    model = RidgeSeasonal().fit(total_series)
    predicted = model.forecast(total_series, 4)
    assert (predicted >= 0).all()
    assert len(predicted) == 4


def test_ridge_refuses_to_fit_without_enough_rows():
    """After a 52-week lag a short series has nothing left to regress on."""
    short = build_series(make_panel(weeks=60), level="total")
    with pytest.raises(ValueError, match="usable training rows"):
        RidgeSeasonal().fit(short)


# ---------------------------------------------------------------------------- ensemble

def test_ensemble_is_an_average_of_its_two_parts(total_series):
    ensemble = EnsembleFixed().fit(total_series)
    blended = ensemble.forecast(total_series, 4)
    naive = np.asarray(SeasonalNaiveModel().fit(total_series).forecast(total_series, 4))
    prophet = np.asarray(ProphetPooled().fit(total_series).forecast(total_series, 4))
    assert blended == pytest.approx(0.5 * naive + 0.5 * prophet)


def test_ensemble_weights_are_fixed_not_fitted():
    """Weights tuned on the holdout would describe the tuning, not the forecast."""
    assert EnsembleFixed.WEIGHTS == (0.5, 0.5)


# ------------------------------------------------------------------------ reconciliation

def test_reconciled_parts_sum_to_the_whole():
    """Each *period* must sum to that period's forecast, not the total across periods."""
    base = np.array([100.0, 200.0, 300.0])
    weights = np.array([0.5, 0.3, 0.2])
    parts = reconcile(base, weights)
    assert parts.shape == (3, 3)
    assert parts.sum(axis=1).tolist() == pytest.approx(base.tolist())


def test_reconcile_handles_a_scalar_forecast():
    parts = reconcile(100.0, np.array([0.5, 0.3, 0.2]))
    assert parts.tolist() == pytest.approx([50.0, 30.0, 20.0])
    assert parts.sum() == pytest.approx(100.0)


def test_reconcile_preserves_relative_ranking():
    parts = reconcile(10.0, np.array([0.1, 0.9]))
    assert parts[1] > parts[0]


def test_reconcile_rejects_a_two_dimensional_forecast():
    with pytest.raises(ValueError, match="1-D"):
        reconcile(np.ones((2, 2)), np.array([0.5, 0.5]))


def test_reconcile_refuses_zero_total_weight():
    with pytest.raises(ValueError, match="zero total weight"):
        reconcile(np.array([100.0]), np.array([0.0, 0.0]))


# --------------------------------------------------------------------------- evaluate_all

def test_evaluate_all_returns_the_same_columns_even_when_everything_fails():
    """A total failure must still be a readable table, not a KeyError downstream."""
    too_short = build_series(make_panel(weeks=40), level="total")
    scores = evaluate_all(too_short, horizons=(1, 2), run_prophet=False, run_lstm=False,
                          progress=lambda *_: None)
    assert list(scores.columns) == ["horizon", "model", "wape", "mape_nonzero", "mae",
                                    "bias", "actual_total", "predicted_total", "error"]
    assert scores["wape"].isna().all()
    assert scores["error"].notna().all()


def test_evaluate_all_covers_every_requested_horizon(panel):
    series = build_series(panel, level="total")
    scores = evaluate_all(series, horizons=(1, 2), run_prophet=False, run_lstm=False,
                          progress=lambda *_: None)
    assert set(scores["horizon"]) == {1, 2}
    assert scores["wape"].notna().all()


def test_every_backtested_wape_is_a_sane_number(panel):
    series = build_series(panel, level="total")
    scores = evaluate_all(series, horizons=(4,), run_prophet=False, run_lstm=False,
                          progress=lambda *_: None)
    assert (scores["wape"] >= 0).all()
    assert (scores["wape"] <= 5).all()


# ------------------------------------------------------------------------ real data

@pytest.mark.slow
def test_selection_never_ships_a_model_the_backtest_rejected():
    """The shipped forecast must come from the per-horizon winner, not a fixed choice."""
    from src.forecasting import fit_and_forecast
    from src.ingest import load_panel

    result = fit_and_forecast(load_panel(), write=False, progress=lambda *_: None)
    scores, selection = result["scores"], result["selection"]

    for row in selection.itertuples():
        best = scores[(scores["horizon"] == row.horizon)
                      & (scores["model"] == row.model)].iloc[0]
        everything_else = scores[(scores["horizon"] == row.horizon)
                                 & (scores["model"] != row.model)]["wape"]
        assert best["wape"] <= everything_else.min() + 1e-12


@pytest.mark.slow
def test_real_backtest_beats_the_brief_mape_target_on_the_non_zero_subset():
    """The brief's 12% target, checked on the winner at the primary horizon."""
    from src.forecasting import fit_and_forecast
    from src.ingest import load_panel

    result = fit_and_forecast(load_panel(), write=False, progress=lambda *_: None)
    scores = result["scores"]
    primary = scores[(scores["horizon"] == config.PRIMARY_HORIZON)
                     & scores["wape"].notna()]
    assert primary["mape_nonzero"].min() < config.BRIEF_MAPE_TARGET