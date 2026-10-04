"""Forecast accuracy metrics: RMSE and MASE, and the scoring contract around them.

Added in the final audit pass because the brief's metric list includes both and neither was
being published. MASE in particular is the only scale-free measure on the list, and it is
the one most easily published wrong:

  * a random-walk denominator (seasonality=1) and a seasonal-naive denominator
    (seasonality=52) differ by an order of magnitude on annual retail data
  * a MASE computed against the wrong in-sample window is not a MASE at all
  * a NaN denominator silently produces a NaN score that still gets written to CSV

So these tests pin the definition, the seasonality choice, and the failure modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.forecasting import SCORE_COLUMNS, backtest_pooled
from src.metrics import forecast_metrics, mae, mase, rmse, wape


class TestRmse:
    def test_is_the_root_mean_square_error(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        assert rmse(a, b) == pytest.approx(np.sqrt((1 + 0 + 1) / 3))

    def test_is_at_least_mae(self):
        # RMSE weights large errors more heavily, so it is never the smaller of the two.
        # A pipeline that reported RMSE below MAE would have a bug worth finding.
        rng = np.random.default_rng(3)
        a = rng.normal(100, 20, 500)
        b = a + rng.normal(0, 25, 500)
        assert rmse(a, b) >= mae(a, b)

    def test_equals_mae_when_every_error_is_identical(self):
        a = np.array([10.0, 20.0, 30.0])
        b = a + 5.0
        assert rmse(a, b) == pytest.approx(mae(a, b))

    def test_punishes_a_single_large_miss_more_than_a_small_one(self):
        # The property that distinguishes RMSE from MAE, and the reason both are reported.
        one_big = np.array([100.0, 100.0, 100.0, 100.0])
        big_error = np.array([100.0, 100.0, 100.0, 200.0])
        small_error = np.array([105.0, 95.0, 105.0, 95.0])
        assert rmse(one_big, big_error) > rmse(one_big, small_error)

    def test_is_unaffected_by_a_constant_offset_in_a_way_mae_is_not(self):
        # A systematic bias of 10 must move MAE and RMSE by the same absolute amount; this
        # is a sanity check that neither is secretly normalised by the target scale.
        a = np.full(50, 100.0)
        assert rmse(a, a + 10) == pytest.approx(rmse(a, a) + 10)


class TestMase:
    def test_a_perfect_forecast_scores_zero(self):
        a = np.arange(1.0, 61.0)
        assert mase(a, a, insample=a, seasonality=1) == pytest.approx(0.0)

    def test_naive_forecast_scores_one(self):
        # The defining property: predicting the in-sample seasonal-naive value for the
        # in-sample series must give MASE = 1 exactly. If this fails the denominator and the
        # numerator are not using the same definition.
        # A LINEAR series. Every lag-52 step is exactly +104, so the in-sample scale
        # is exactly 104 and the seasonal-naive forecast of the holdout is off by exactly
        # 104 too. The ratio is then 1 to the last bit, with no floating-point noise.
        #
        # A perfectly PERIODIC series cannot be used here, which is the trap: if
        # `a[i] - a[i-52]` is identically zero the true scale is zero, `mase` returns NaN, and
        # a perturbed version lands on a ratio of two ~1e-14 noise terms instead of 1.0. A
        # meaningful-looking 1.05 is exactly what that mistake looks like.
        t = np.arange(156.0)
        a = 100.0 + 2.0 * t
        history, actual = a[:-4], a[-4:]
        predicted = actual - 2.0 * 52            # what seasonal naive returns for this series
        assert mase(actual, predicted, insample=history, seasonality=52) == pytest.approx(1.0)

    def test_better_than_seasonal_naive_is_below_one(self):
        rng = np.random.default_rng(9)
        a = 100 + rng.normal(0, 5, 120)
        naive = np.roll(a, 52)
        smooth = pd.Series(a).rolling(4, min_periods=1).mean().to_numpy()
        assert mase(a, smooth, insample=a, seasonality=52) < 1.0

    def test_seasonality_changes_the_denominator_and_therefore_the_number(self):
        # The reason `config.MASE_SEASONALITY` is named rather than defaulted silently. On
        # annual retail data a random-walk denominator is much smaller than a seasonal one,
        # so the same forecast scores far worse.
        rng = np.random.default_rng(11)
        a = 100 + 40 * np.sin(np.arange(120) / 52 * 2 * np.pi) + rng.normal(0, 5, 120)
        predicted = pd.Series(a).rolling(8, min_periods=1).mean().to_numpy()
        weekly = mase(a, predicted, insample=a, seasonality=52)
        random_walk = mase(a, predicted, insample=a, seasonality=1)
        assert weekly != pytest.approx(random_walk)
        assert weekly > random_walk, (
            "a seasonal-naive denominator should be the LARGER, harsher denominator here"
        )

    def test_project_seasonality_is_annual_weekly(self):
        assert config.MASE_SEASONALITY == 52, (
            "the series is weekly with an annual cycle; anything else silently rescales MASE"
        )

    def test_is_infinite_when_the_insample_history_is_flat(self):
        # A constant in-sample series has zero naive error, so the scale is zero and MASE is
        # undefined. The result must be inf or nan -- never 0, which would read as a perfect
        # forecast.
        a = np.full(60, 50.0)
        value = mase(np.full(4, 55.0), np.full(4, 55.0), insample=a, seasonality=1)
        assert not np.isfinite(value)
        assert value != 0.0

    def test_refuses_an_insample_shorter_than_the_season(self):
        # seasonality=52 against a 10-week in-sample window cannot produce a seasonal-naive
        # error at all. Raising is the right answer: returning inf would put an inf in
        # forecast_backtest.csv, and evaluate_all catches exceptions and records them as a
        # visible model failure rather than a silent infinity.
        rng = np.random.default_rng(13)
        a = rng.normal(100, 5, 60)
        with pytest.raises(ValueError, match="at least"):
            mase(a[-4:], a[-4:], insample=a[:10], seasonality=52)

    def test_refuses_an_empty_in_sample(self):
        a = np.arange(20.0)
        with pytest.raises(ValueError):
            mase(a, a, insample=np.array([]), seasonality=1)


class TestForecastMetricsBundle:
    def test_reports_every_metric_the_brief_lists(self):
        a = np.array([100.0, 110.0, 90.0, 120.0])
        b = np.array([104.0, 106.0, 95.0, 118.0])
        result = forecast_metrics(a, b, insample=np.arange(60.0))
        for key in ("wape", "mape_nonzero", "mae", "rmse", "mase", "smape", "bias"):
            assert key in result, f"{key} missing from forecast_metrics"

    def test_mase_is_none_without_insample_rather_than_guessed(self):
        # Filling MASE with 0 or with MAE would be a fabrication: it would report a perfect
        # forecast for a model that has none.
        result = forecast_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert result["mase"] is None

    def test_metrics_are_internally_consistent(self):
        a = np.array([100.0, 110.0, 90.0, 120.0])
        b = np.array([104.0, 106.0, 95.0, 118.0])
        result = forecast_metrics(a, b, insample=np.arange(60.0))
        # MAE must equal the mean absolute error of the reported pair, not of some other one.
        assert result["mae"] == pytest.approx(np.abs(a - b).mean())
        assert result["rmse"] == pytest.approx(np.sqrt(((a - b) ** 2).mean()))
        assert result["total_actual"] == pytest.approx(a.sum())
        # metrics.bias is mean(actual - pred): positive means OVER-forecast. The sign
        # convention is the opposite of the obvious one and is worth pinning, because
        # flipping it turns over-forecast warnings into under-forecast ones.
        assert result["bias"] == pytest.approx((a - b).mean())

    def test_wape_is_zero_for_a_perfect_forecast(self):
        a = np.array([10.0, 20.0, 30.0])
        assert forecast_metrics(a, a)["wape"] == pytest.approx(0.0)

    def test_wape_does_not_blow_up_when_actuals_are_nearly_zero(self):
        # This is why WAPE and not MAPE is the headline. A single near-zero week makes MAPE
        # explode; WAPE is weighted by volume and stays readable.
        a = np.array([0.01, 1000.0, 1000.0, 1000.0])
        b = np.array([50.0, 1000.0, 1000.0, 1000.0])
        result = forecast_metrics(a, b)
        assert np.isfinite(result["wape"])
        assert result["wape"] < 1.0
        assert result["mape_nonzero"] > result["wape"], (
            "MAPE should be the volatile one; if it is not, the outlier was excluded"
        )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            forecast_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


class TestScoringContract:
    def test_score_columns_include_the_new_metrics(self):
        # The CSV header is a contract with the dashboard and with the CI column check.
        for column in ("rmse", "mase", "wape", "mape_nonzero", "mae", "model", "horizon"):
            assert column in SCORE_COLUMNS

    def test_every_score_column_has_a_producer(self):
        # SCORE_COLUMNS is the CSV header. horizon/model/error come from the caller,
        # ctual_total/predicted_total from evaluate_all, and the rest from the bundle --
        # so check that each column is produced SOMEWHERE rather than by one function.
        result = forecast_metrics(np.array([10.0, 12.0]), np.array([11.0, 11.0]),
                                  insample=np.arange(60.0))
        from_the_caller = {"horizon", "model", "error", "actual_total", "predicted_total"}
        for column in SCORE_COLUMNS:
            if column in from_the_caller:
                continue
            assert column in result, f"{column} is declared but never computed"
        assert set(SCORE_COLUMNS) <= (set(result) | from_the_caller)

    def test_backtest_returns_the_insample_window_mase_needs(self):
        # Without `insample`, MASE cannot be computed at all. Dropping it would silently
        # return None for every model and make the whole column meaningless.
        series = pd.DataFrame({
            "week_start_date": pd.date_range("2023-01-02", periods=120, freq="W-MON"),
            "units": 100 + np.arange(120) * 0.5,
        })

        class Flat:
            def fit(self, frame):
                return self

            def forecast(self, frame, horizon):
                return np.full(horizon, float(frame["units"].mean()))

        outcome = backtest_pooled(series, horizon=4, model_factory=Flat)
        assert len(outcome["insample"]) == len(series) - 4
        assert outcome["insample"].tolist() == series["units"].to_numpy()[:-4].tolist()
        assert len(outcome["actual"]) == 4
        assert len(outcome["predicted"]) == 4

    def test_backtest_holdout_never_leaks_into_the_fit(self):
        series = pd.DataFrame({
            "week_start_date": pd.date_range("2023-01-02", periods=120, freq="W-MON"),
            "units": 100 + np.arange(120) * 0.5,
        })
        seen = {}

        class Spy:
            def fit(self, frame):
                seen["fit_max"] = frame["units"].max()
                return self

            def forecast(self, frame, horizon):
                return np.full(horizon, 1.0)

        backtest_pooled(series, horizon=4, model_factory=Spy)
        # The final 4 points are strictly increasing, so if they leaked the fit would see the
        # largest value in the series.
        assert seen["fit_max"] < series["units"].iloc[-1]

    def test_backtest_refuses_a_series_too_short_to_hold_out(self):
        series = pd.DataFrame({
            "week_start_date": pd.date_range("2023-01-02", periods=12, freq="W-MON"),
            "units": np.arange(12.0),
        })

        class Flat:
            def fit(self, frame):
                return self

            def forecast(self, frame, horizon):
                return np.zeros(horizon)

        with pytest.raises(ValueError, match="need more than"):
            backtest_pooled(series, horizon=4, model_factory=Flat)

    def test_backtest_rejects_a_model_that_returns_the_wrong_length(self):
        series = pd.DataFrame({
            "week_start_date": pd.date_range("2023-01-02", periods=120, freq="W-MON"),
            "units": 100.0 + np.arange(120.0),
        })

        class WrongLength:
            def fit(self, frame):
                return self

            def forecast(self, frame, horizon):
                return np.zeros(horizon + 2)

        # Silently accepting a misaligned prediction would shift every metric by two weeks.
        with pytest.raises(ValueError, match="expected"):
            backtest_pooled(series, horizon=4, model_factory=WrongLength)


class TestPublishedForecastArtefact:
    def test_backtest_csv_carries_rmse_and_mase(self):
        path = config.PROCESSED / "forecast_backtest.csv"
        if not path.exists():
            pytest.skip("run `python -m src.forecasting` first")
        frame = pd.read_csv(path)
        for column in ("rmse", "mase", "wape", "horizon", "model"):
            assert column in frame.columns, f"{column} missing from forecast_backtest.csv"

    def test_published_mase_is_finite_for_every_scored_model(self):
        path = config.PROCESSED / "forecast_backtest.csv"
        if not path.exists():
            pytest.skip("run `python -m src.forecasting` first")
        frame = pd.read_csv(path)
        scored = frame[frame["wape"].notna()]
        assert not scored.empty
        assert scored["mase"].notna().all(), (
            "a model with a WAPE but no MASE means the in-sample window was dropped"
        )
        assert np.isfinite(scored["mase"]).all()

    def test_rmse_and_mae_are_consistent_in_the_published_file(self):
        # RMSE >= MAE holds for every model on every dataset. A published row where it
        # inverts means the two columns came from different holdouts.
        path = config.PROCESSED / "forecast_backtest.csv"
        if not path.exists():
            pytest.skip("run `python -m src.forecasting` first")
        frame = pd.read_csv(path)
        scored = frame[frame["wape"].notna() & frame["mae"].notna()]
        violations = scored[scored["rmse"] + 1e-6 < scored["mae"]]
        assert violations.empty, (
            f"{len(violations)} rows have RMSE below MAE:\n{violations[['horizon', 'model', 'mae', 'rmse']]}"
        )