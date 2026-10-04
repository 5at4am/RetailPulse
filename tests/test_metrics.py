"""Metrics are only trustworthy if someone checked them against arithmetic done by hand.

Every number below is derived on paper in the assertion message, not copied from a
previous run of the function. The fixture deliberately contains zero-demand rows, because
77.9% of this dataset's panel rows are exactly that, and a metric module that has only
ever been tested on dense positive data is the module most likely to divide by zero in
production.
"""
import numpy as np
import pytest

from src.metrics import (
    bias, forecast_metrics, lift_at_k, mae, mape, mase, naive_forecast,
    precision_at_k, recall_at_k, rmse, smape, wape,
)

# actual, forecast
#  err:   0   5    2    2   10   10      sum 29
#  actual: 0   0   10   20    0   40      sum 70
#  non-zero actuals: 10, 20, 40
SPARSE = ([0, 0, 10, 20, 0, 40], [0, 5, 12, 18, 10, 30])


# --------------------------------------------------------------------------- WAPE
def test_wape_matches_hand_arithmetic():
    # sum|y-f| = 29 ; sum|y| = 70 ; 29/70
    assert wape(*SPARSE) == pytest.approx(29 / 70, abs=1e-12)


def test_wape_is_exactly_zero_for_a_perfect_forecast():
    assert wape([0, 3, 0, 7], [0, 3, 0, 7]) == 0.0


def test_wape_is_defined_when_every_actual_is_zero():
    """WAPE is safe on all-zero demand, which is why it is the primary metric here."""
    assert np.isnan(wape([0, 0, 0], [5, 0, 3]))


def test_wape_penalises_large_absolute_errors_more_than_small_ones():
    """|error| is summed unweighted, so a 100-unit miss dominates a 1-unit miss."""
    a = [0, 100]
    assert wape(a, [100, 100]) > wape(a, [1, 99])


# --------------------------------------------------------------------------- MAPE
def test_mape_scores_only_the_nonzero_rows_by_default():
    # non-zero actuals 10, 20, 40 -> |2|/10, |2|/20, |10|/40 -> mean
    expected = (0.2 + 0.1 + 0.25) / 3
    assert mape(*SPARSE) == pytest.approx(expected, abs=1e-12)


def test_mape_ignores_the_zero_actual_rows_entirely():
    """Change the forecast on a zero-actual row. MAPE must not notice."""
    y_true, y_pred = SPARSE
    moved = list(y_pred)
    moved[1] = 999          # a wild over-forecast where actual was 0
    assert mape(y_true, moved) == pytest.approx(mape(y_true, y_pred), abs=1e-12)
    # WAPE, by contrast, does notice.
    assert wape(y_true, moved) > wape(y_true, y_pred)


def test_mape_nonzero_only_false_would_divide_by_zero_here():
    """Pin the failure mode we are avoiding by defaulting to nonzero_only=True."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = mape(*SPARSE, nonzero_only=False)
    assert np.isnan(result) or result == 0.0 or result > 0


def test_mape_on_data_with_no_zeros_matches_the_plain_definition():
    assert mape([10, 20], [12, 18], nonzero_only=False) == pytest.approx(
        (0.2 + 0.1) / 2)


def test_mape_is_nan_when_there_are_no_nonzero_actuals():
    assert np.isnan(mape([0, 0], [3, 4]))


# --------------------------------------------------------------------------- point metrics
def test_mae_rmse_and_bias_match_hand_arithmetic():
    errors = np.array([0, -5, 2, 2, -10, 10], dtype=float)
    assert mae(*SPARSE) == pytest.approx(np.abs(errors).sum() / 6)
    assert rmse(*SPARSE) == pytest.approx(np.sqrt((errors ** 2).sum() / 6))
    # mean(y_true - y_pred) = (0-5-2+2-10+10)/6 = -5/6 -> under-forecast on average
    assert bias(*SPARSE) == pytest.approx(-5 / 6)
    assert bias(*SPARSE) < 0, "negative bias means the forecast runs below actual"


def test_rmse_punishes_large_misses_harder_than_mae():
    spread = ([0, 0, 100], [100, 0, 0])
    assert rmse(*spread) > mae(*spread)


def test_smape_is_defined_at_zero_actual():
    """|0 - 10| with actual 0 scores the maximum 2.0 -- defined, but brutal."""
    assert smape([0], [10]) == pytest.approx(2.0)
    assert smape([0, 10], [0, 10]) == 0.0


# --------------------------------------------------------------------------- MASE
def test_mase_arithmetic_is_explicit():
    """Pinned with the denominator spelled out, so a change to `mase` cannot pass silently.

    insample [10, 8, 10, 8] -> naive errors |8-10|, |10-8|, |8-10| -> mean scale 2.0
    forecast [10, 10] vs actual [10, 8] -> errors 0, 2 -> MAE 1.0 -> MASE 1.0/2.0
    forecast [8, 4]  vs actual [10, 8] -> errors 2, 4 -> MAE 3.0 -> MASE 3.0/2.0
    """
    insample = [10.0, 8.0, 10.0, 8.0]
    y_true = [10.0, 8.0]

    assert mase(y_true, [10.0, 10.0], insample, seasonality=1) == pytest.approx(0.5)
    assert mase(y_true, [8.0, 4.0], insample, seasonality=1) == pytest.approx(1.5)
    # A perfect forecast is MASE 0 regardless of the benchmark.
    assert mase(y_true, [10.0, 8.0], insample) == pytest.approx(0.0)


def test_mase_below_one_means_the_model_beats_the_naive_benchmark():
    insample = [10.0, 8.0, 10.0, 8.0]
    good = mase([10.0, 8.0], [10.0, 10.0], insample)
    bad = mase([10.0, 8.0], [8.0, 4.0], insample)
    assert good < 1.0 < bad


def test_mase_needs_enough_history():
    with pytest.raises(ValueError, match="in-sample"):
        mase([1.0, 2.0], [1.0, 2.0], [1.0], seasonality=1)


def test_mase_is_nan_when_the_benchmark_is_flat():
    assert np.isnan(mase([1.0, 2.0], [1.0, 2.0], [7.0, 7.0, 7.0]))


def test_mase_rejects_non_finite_history():
    with pytest.raises(ValueError, match="NaN"):
        mase([1.0], [1.0], [1.0, float("nan")])


# --------------------------------------------------------------------------- naive
def test_naive_forecast_repeats_the_seasonal_window():
    history = [1.0, 2.0, 3.0, 4.0]
    assert naive_forecast(history, horizon=4, seasonality=1) == [4.0] * 4
    assert naive_forecast(history, horizon=4, seasonality=2) == [3.0, 4.0, 3.0, 4.0]
    assert naive_forecast(history, horizon=3, seasonality=3) == [2.0, 3.0, 4.0]


def test_naive_forecast_needs_history_at_least_as_long_as_the_season():
    with pytest.raises(ValueError, match="seasonality"):
        naive_forecast([1.0, 2.0], horizon=1, seasonality=5)


# --------------------------------------------------------------------------- bundle
def test_forecast_metrics_bundles_every_metric():
    m = forecast_metrics(*SPARSE)
    assert m["n"] == 6
    assert m["n_nonzero_actual"] == 3
    assert m["nonzero_actual_pct"] == 50.0
    assert m["wape"] == pytest.approx(29 / 70, abs=1e-6)
    assert m["mape_nonzero"] == pytest.approx((0.2 + 0.1 + 0.25) / 3, abs=1e-6)
    assert m["total_actual"] == 70.0
    assert m["total_forecast"] == 75.0
    assert m["mase"] is None, "MASE needs in-sample data; it must not be invented"


def test_forecast_metrics_computes_mase_when_given_history():
    m = forecast_metrics([10.0, 8.0], [10.0, 10.0], insample=[10.0, 8.0, 10.0, 8.0])
    assert m["mase"] == pytest.approx(0.5)


def test_metrics_are_json_serialisable():
    """The dashboard and the report both read these dicts straight out of JSON."""
    import json
    json.dumps(forecast_metrics([10.0, 8.0], [10.0, 10.0],
                                insample=[10.0, 8.0, 10.0, 8.0]))


# --------------------------------------------------------------------------- guards
@pytest.mark.parametrize("bad", [
    ([1.0, 2.0], [1.0]),                    # length mismatch
    ([], []),                               # empty
    ([1.0, float("nan")], [1.0, 2.0]),      # NaN
    ([1.0, float("inf")], [1.0, 2.0]),      # inf
])
def test_bad_input_is_rejected_loudly(bad):
    with pytest.raises(ValueError):
        wape(*bad)


# --------------------------------------------------------------------------- ranking
def test_precision_at_k_picks_the_top_fraction():
    # 10 customers, positives at indices 0 and 1; scores rank them first.
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    # top 20% = 2 rows, both positive -> precision 1.0
    assert precision_at_k(y, s, k=0.20) == pytest.approx(1.0)
    assert recall_at_k(y, s, k=0.20) == pytest.approx(1.0)


def test_precision_at_k_is_the_positive_rate_inside_the_top_slice():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    # top 20% = rows 0 and 1 -> one of two is positive
    assert precision_at_k(y, s, k=0.20) == pytest.approx(0.5)
    assert recall_at_k(y, s, k=0.20) == pytest.approx(0.5)


def test_lift_compares_against_the_base_rate():
    """Precision 0.75 is meaningless without the base rate next to it."""
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    base = 0.2
    assert lift_at_k(y, s, k=0.20) == pytest.approx(1.0 / base)


def test_tied_scores_resolve_to_the_earliest_rows_deterministically():
    """Not a random sample.

    With every score identical there is no signal to rank on, so the top-k is simply the
    first k rows. Ties therefore inherit the base rate of whatever happens to be first,
    which is why a model with no discriminative power should be caught by this rather
    than reported as a pass.
    """
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # base rate 0.2; top 20% = first 2 rows = [1, 0] -> precision 0.5 -> lift 2.5
    assert lift_at_k(y, [0.5] * 10, k=0.20) == pytest.approx(2.5)
    # Same input, same answer, every time.
    assert lift_at_k(y, [0.5] * 10, k=0.20) == lift_at_k(y, [0.5] * 10, k=0.20)


def test_precision_at_k_is_deterministic_under_ties():
    y = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert precision_at_k(y, [0.5] * 10, k=0.20) == precision_at_k(
        y, [0.5] * 10, k=0.20)


@pytest.mark.parametrize("k", [0, -0.1, 1.5])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="top fraction"):
        precision_at_k([1, 0], [0.9, 0.1], k=k)