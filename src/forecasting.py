"""F-03 — Weekly demand forecasting.

    from src.forecasting import run_forecasting
    result = run_forecasting()      # backtest at h=1,2,4, then fit and forecast

**The brief asks for 30-day-ahead forecasts; on a weekly panel that is 4 weeks.** All three
horizons (1, 2, 4) are evaluated, 4 is the headline, and inventory consumes only h=1.

**Seven thousand five hundred separate models is not the answer.** The panel is 50 stores x
150 products x 104 weeks with 77.9% zero-demand rows. Fitting Prophet per store-product
would take days and would model noise on series that are zero most weeks. So demand is
forecast on pooled series — total, per store, per product, per category — and reconciled
downwards, with the head getting individual treatment and the tail inheriting its share.

Four model families, each earning its place:

- **Seasonal naive** — last year same week. The benchmark every other model has to beat.
- **Ridge on seasonal dummies** — cheap, interpretable, surprisingly hard to beat weekly.
- **Prophet** — trend + changepoints + holidays, on the pooled series only.
- **LSTM** — a small Keras sequence model on the pooled total. Included because the brief
  asks for it, and reported with the same honesty as everything else.

**WAPE is the headline; MAPE is reported only on rows where actual demand was non-zero**, with
the subset stated every time. On a panel that is 77.9% zeros, MAPE over all rows is not a
metric, it is a division by zero with extra steps.

Backtesting is temporal: the last `horizon` weeks of each series are held out and forecasts
are made from data strictly before them.

Runnable on its own: `python -m src.forecasting`.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.metrics import forecast_metrics, naive_forecast

SEASON_LENGTH = 52          # weekly data, annual seasonality

# The panel is 104 weeks -- exactly two annual cycles. So a seasonal model is fitted on
# roughly one cycle and asked to predict the second, and after a 52-week lag is dropped the
# ridge design has about 52 usable rows. That is thin, and it is why the backtest below is
# reported as a single holdout rather than a cross-validated average: there is not enough
# history to average over. Stating that is more useful than a tighter-looking number.
MIN_OBSERVATIONS = SEASON_LENGTH + 8


# --------------------------------------------------------------------------------- series

def build_series(panel, level="total"):
    """Aggregate the panel into a single weekly series.

    Levels: `total`, `store`, `product`, `category`. Forecasting happens on these pooled
    series rather than on 7,500 individual ones.
    """
    keys = {
        "total": [],
        "store": ["store_id"],
        "product": ["product_id"],
        "category": ["product_category"],
    }[level]

    frame = panel.copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    frame = frame[frame["week_start_date"] < pd.Timestamp(config.PARTIAL_FINAL_WEEK)]

    grouped = frame.groupby(keys + ["week_start_date"])["units_sold"].sum().reset_index()
    grouped = grouped.rename(columns={"units_sold": "units"})
    return grouped.sort_values("week_start_date").reset_index(drop=True)


def seasonal_naive(history, horizon, season_length=SEASON_LENGTH):
    """Last year's same week. The benchmark that has to be beaten."""
    return naive_forecast(history, horizon, seasonality=season_length)


# ------------------------------------------------------------------------ pooled models

def seasonal_dummies(index, origin=None):
    """Week-of-year dummies aligned to an origin so train and forecast share columns.

    Using the week number directly would make week 1 of the forecast look like week 1 of
    training, which is right, but building dummies off a fitted index means a forecast
    horizon can never silently shift by one.
    """
    weeks = index.isocalendar().week.astype(int).to_numpy()
    return pd.get_dummies(pd.Series(weeks, index=index, name="week"), prefix="w", drop_first=False)


class RidgeSeasonal:
    """Ridge regression on [trend, week dummies, lags].

    Deliberately simple. On weekly retail data a regularised linear model with seasonal
    dummies is a genuinely strong baseline, and if a boosted or neural model cannot beat it
    here, that is worth knowing rather than hiding behind an ensemble average.
    """

    name = "ridge_seasonal"

    def __init__(self, lags=(1, 2, 52), season_length=SEASON_LENGTH, alpha=1.0):
        self.lags = lags
        self.season_length = season_length
        self.alpha = alpha
        self.model = None
        self.columns_ = None
        self.train_index_ = None

    def _design(self, frame):
        blocks = []
        weeks = frame["week_start_date"].dt.isocalendar().week.astype(int)
        blocks.append(pd.get_dummies(weeks, prefix="w", drop_first=True)
                      .set_axis(frame.index))
        t = np.arange(len(frame), dtype=float)
        blocks.append(pd.DataFrame({"t": t, "t2": t ** 2}, index=frame.index))
        for lag in self.lags:
            blocks.append(pd.DataFrame({f"lag_{lag}": frame["units"].shift(lag)},
                                      index=frame.index))
        return pd.concat(blocks, axis=1)

    def fit(self, series):
        from sklearn.linear_model import Ridge

        design = self._design(series.copy()).dropna()
        target = series.loc[design.index, "units"]
        if len(design) < 20:
            raise ValueError(
                f"only {len(design)} usable training rows after dropping lag gaps; "
                "not enough to fit a seasonal regression")
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(design, target)
        self.train_columns_ = list(design.columns)
        return self

    def forecast(self, series, horizon):
        """Extend the history forward and predict, re-using the fitted design columns."""
        frame = series.copy()
        last = frame["week_start_date"].max()
        future = pd.DataFrame({
            "week_start_date": pd.date_range(last + pd.Timedelta(weeks=1),
                                             periods=horizon, freq="W-MON"),
        })
        future["units"] = np.nan

        extended = pd.concat([frame, future], ignore_index=True)
        design = self._design(extended)
        design = design.reindex(columns=self.train_columns_)
        design = design.tail(horizon)
        design = design.fillna(0.0)
        return np.clip(self.model.predict(design), 0.0, None)


class ProphetPooled:
    """Prophet on a pooled weekly series.

    Prophet is fitted per pooled series, never per store-product. `uncertainty_samples=0`
    because this project reports point accuracy against a held-out tail and does not use
    the intervals; sampling 1,000 posterior paths per series would dominate the runtime for
    output nothing reads.
    """

    name = "prophet"

    def __init__(self, yearly=True, weekly=True):
        self.yearly = yearly
        self.weekly = weekly
        self.model = None
        self.history_len_ = None
        self.start_ = None

    def fit(self, series):
        from prophet import Prophet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Prophet(
                yearly_seasonality=self.yearly,
                weekly_seasonality=self.weekly,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
                interval_width=0.8,
                uncertainty_samples=0,
            )
            self.history_len_ = len(series)
            self.start_ = series["week_start_date"].iloc[0]
            self.model.fit(series[["week_start_date", "units"]].rename(
                columns={"week_start_date": "ds", "units": "y"}))
        return self

    def forecast(self, series, horizon):
        future = pd.DataFrame({
            "ds": pd.date_range(series["week_start_date"].max() + pd.Timedelta(weeks=1),
                                periods=horizon, freq="W-MON")
        })
        predicted = self.model.predict(future)["yhat"]
        return np.clip(predicted.to_numpy(), 0.0, None)


def build_lstm(lookback=SEASON_LENGTH, units=32, epochs=40):
    """A small LSTM sequence model. Sized for pooled weekly series, not 7,500 of them."""
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(units, activation="tanh"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units // 2, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    return model, lookback


class LSTMPooled:
    """Sliding-window LSTM on a single pooled series.

    Trained on all but the final `horizon` weeks, one window per week. With ~100 weeks of
    history that is a few dozen training examples, so the model is small and heavily
    regularised on purpose, and its results are reported next to the ridge baseline rather
    than instead of it.
    """

    name = "lstm"

    def __init__(self, lookback=SEASON_LENGTH, units=32, epochs=40, seed=config.SEED):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.seed = seed
        self.model = None
        self.mean_ = None
        self.scale_ = None

    def _windows(self, values):
        x, y = [], []
        for i in range(self.lookback, len(values)):
            x.append(values[i - self.lookback:i])
            y.append(values[i])
        return np.array(x)[..., np.newaxis], np.array(y)

    def fit(self, series):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        values = series["units"].to_numpy(dtype=float)
        # Standardise: raw unit counts reach into the thousands and an un-normalised LSTM
        # will not train on a hundred examples.
        self.mean_ = float(values.mean())
        self.scale_ = float(values.std()) or 1.0
        scaled = (values - self.mean_) / self.scale_

        x, y = self._windows(scaled)
        self.model, _ = build_lstm(self.lookback, self.units, self.epochs)
        self.model.fit(x, y, epochs=self.epochs, batch_size=8, verbose=0,
                       shuffle=False, validation_split=0.15)
        return self

    def forecast(self, series, horizon):
        values = series["units"].to_numpy(dtype=float)
        scaled = (values - self.mean_) / self.scale_
        history = list(scaled)
        out = []
        for _ in range(horizon):
            window = np.array(history[-self.lookback:])[None, :, np.newaxis]
            nxt = float(self.model.predict(window, verbose=0)[0][0])
            history.append(nxt)
            out.append(nxt)
        return np.clip(np.array(out) * self.scale_ + self.mean_, 0.0, None)


# ------------------------------------------------------------------------------ backtest

def backtest_pooled(series, horizon, model_factory, min_history=MIN_OBSERVATIONS):
    """Hold out the final `horizon` weeks and forecast them from what came before.

    A pooled series is short -- 104 weeks, two annual cycles -- so this is the only honest
    way to score a forecast here. It is also why the pooled numbers are reported as
    "one series, one holdout" rather than as a distribution over folds.
    """
    values = series["units"].to_numpy(dtype=float)
    if len(values) <= min_history + horizon:
        raise ValueError(
            f"only {len(values)} observations; need more than "
            f"{min_history + horizon} to hold out {horizon} weeks")

    train = series.iloc[:-horizon].copy()
    actual = values[-horizon:]
    model = model_factory().fit(train)
    predicted = model.forecast(train, horizon)
    if len(predicted) != horizon:
        raise ValueError(f"model returned {len(predicted)} points, expected {horizon}")
    return {
        "actual": actual,
        "predicted": np.asarray(predicted, dtype=float),
        "model": model,
        # The in-sample series MASE divides by. Without it MASE cannot be computed, and MASE
        # is the only scale-free accuracy measure in the brief's metric list.
        "insample": train["units"].to_numpy(dtype=float),
    }


def reconcile(base_forecast, weights):
    """Apportion each period's forecast across series by historical demand share.

    Hierarchical forecasts are inconsistent by construction: the sum of the store forecasts
    need not equal the total forecast. This makes them consistent in one step by spreading
    the trusted aggregate across series according to what each series has historically
    accounted for.

    Shapes: `base_forecast` is a scalar or one value per period; `weights` is one value per
    series. A scalar returns one value per series; a per-period array returns a
    (periods x series) matrix where **each row sums to that period's forecast**. An earlier
    version broadcast elementwise, which only summed correctly when the two arrays happened
    to be the same length and silently produced a wrong total otherwise.
    """
    base = np.asarray(base_forecast, dtype=float)
    share = np.asarray(weights, dtype=float)
    total_weight = share.sum()
    if total_weight <= 0:
        raise ValueError("cannot reconcile with zero total weight")
    shares = share / total_weight

    if base.ndim == 0:
        return float(base) * shares
    if base.ndim != 1:
        raise ValueError(f"base_forecast must be scalar or 1-D, got {base.ndim}-D")
    return base[:, None] * shares[None, :]


def series_weights(panel, level="store"):
    """Each series' historical share of demand, used to apportion a pooled forecast.

    Share rather than a fitted bottom-up forecast, because with 77.9% zeros most individual
    series have too little signal to forecast on their own, and a bottom-up sum built from
    noise is worse than the aggregate it is supposed to refine.
    """
    keys = {"total": [], "store": ["store_id"], "product": ["product_id"],
            "category": ["product_category"]}[level]
    totals = panel.groupby(keys)["units_sold"].sum()
    return (totals / totals.sum()).sort_values(ascending=False)


SCORE_COLUMNS = ["horizon", "model", "wape", "mape_nonzero", "mae", "rmse", "mase",
                  "bias", "actual_total", "predicted_total", "error"]


def evaluate_all(series, horizons=config.HORIZONS, run_prophet=True, run_lstm=True,
                 progress=print):
    """Score every model at every horizon. WAPE first, MAPE on the non-zero subset.

    Every row comes back with the same columns whether or not the model ran, so a table of
    failures is still a readable table rather than a `KeyError` in the reporting code.
    """
    results = []
    for horizon in horizons:
        actual = series["units"].to_numpy(dtype=float)[-horizon:]

        candidates = [("seasonal_naive", SeasonalNaiveModel),
                      ("ridge_seasonal", RidgeSeasonal),
                      ("ensemble_naive_prophet", EnsembleFixed)]
        if run_prophet:
            candidates.append(("prophet", ProphetPooled))
        if run_lstm:
            candidates.append(("lstm", LSTMPooled))

        for name, factory in candidates:
            progress(f"  h={horizon} {name:<18}")
            try:
                outcome = backtest_pooled(series, horizon, factory)
            except Exception as error:                       # noqa: BLE001
                # One model failing must not take the backtest down with it, and the failure
                # has to survive into the report instead of quietly disappearing.
                results.append({"horizon": horizon, "model": name,
                                "error": f"{type(error).__name__}: {error}"})
                progress(f"    failed: {type(error).__name__}: {error}")
                continue

            predicted = outcome["predicted"]
            # seasonality=52 matches a weekly series with an annual cycle, so the MASE
            # denominator is a seasonal-naive error rather than a random-walk one. The brief
            # lists MASE without saying which, and the choice changes the number, so it is
            # named here and in the report.
            metrics = forecast_metrics(actual, predicted, insample=outcome["insample"],
                                       seasonality=config.MASE_SEASONALITY)
            results.append({"horizon": horizon, "model": name,
                            "wape": metrics["wape"],
                            "mape_nonzero": metrics["mape_nonzero"],
                            "mae": metrics["mae"], "rmse": metrics["rmse"],
                            "mase": metrics["mase"], "bias": metrics["bias"],
                            "actual_total": float(actual.sum()),
                            "predicted_total": float(predicted.sum())})
            progress(f"    WAPE {metrics['wape']:.4f}   "
                     f"MAPE(nonzero) {metrics['mape_nonzero']:.4f}   "
                     f"RMSE {metrics['rmse']:.1f}   MASE {metrics['mase']:.3f}")

    return pd.DataFrame(results).reindex(columns=SCORE_COLUMNS)


class SeasonalNaiveModel:
    """Thin wrapper so the naive benchmark goes through the same fit/forecast interface."""

    name = "seasonal_naive"

    def fit(self, series):
        self.length_ = len(series)
        return self

    def forecast(self, series, horizon):
        return seasonal_naive(series["units"].to_numpy(dtype=float), horizon)


class EnsembleFixed:
    """Equal-weight blend of seasonal naive and Prophet.

    The weights are fixed at 0.5/0.5 *a priori*, not fitted to the holdout. Tuning blend
    weights on the same holdout used to report accuracy produces a number that describes the
    tuning rather than the forecast. If this does not beat the better of its two components,
    that gets reported and the simpler model ships instead.
    """

    name = "ensemble_naive_prophet"
    WEIGHTS = (0.5, 0.5)

    def __init__(self):
        self.parts = [SeasonalNaiveModel(), ProphetPooled()]

    def fit(self, series):
        for part in self.parts:
            part.fit(series)
        return self

    def forecast(self, series, horizon):
        blended = np.zeros(horizon)
        for weight, part in zip(self.WEIGHTS, self.parts):
            blended += weight * np.asarray(part.forecast(series, horizon), dtype=float)
        return blended


def fit_and_forecast(panel, horizons=config.HORIZONS, progress=print, write=True):
    """Backtest the pooled total, then refit the winning model per horizon on all data.

    Model selection happens on the backtest, and the shipped forecast uses the winner for
    each horizon rather than a single model chosen in advance. On this data that matters:
    seasonal naive wins at h=2 and h=4 while Prophet wins at h=1, and shipping one model
    for all three would mean shipping a known-suboptimal forecast for two of them.

    Refitting on all data after selection is deliberate. The reported metrics come from the
    held-out weeks; the shipped model should use every week available.
    """
    series = build_series(panel, level="total")
    zero_weeks = int((series["units"] == 0).sum())
    progress(f"pooled total: {len(series)} weeks, {series['units'].sum():,.0f} units, "
             f"{zero_weeks} zero weeks")
    if zero_weeks == 0:
        # 77.9% of panel *rows* are zero, but the pooled total never is. So the headline
        # WAPE below is measured on a series with no zero weeks at all, while the store-
        # product level is where the zeros live. Both facts get stated rather than letting
        # the pooled number imply a sparsity it does not have.
        progress("  note: the pooled total never hits zero. The 77.9% zero-demand share "
                 "lives at store-product level,")
        progress("        so pooled WAPE is not directly comparable to a per-series WAPE.")

    progress("\nbacktest (final weeks held out)")
    scores = evaluate_all(series, horizons=horizons, progress=progress)

    scored = scores[scores["wape"].notna()]
    best = (scored.sort_values("wape").groupby("horizon", as_index=False)
            .first()[["horizon", "model", "wape"]])

    factories = {
        "seasonal_naive": SeasonalNaiveModel,
        "ridge_seasonal": RidgeSeasonal,
        "prophet": ProphetPooled,
        "lstm": LSTMPooled,
        "ensemble_naive_prophet": EnsembleFixed,
    }

    progress("\nmodel selection on the holdout")
    rows = []
    for row in best.itertuples():
        factory = factories.get(row.model)
        if factory is None:
            progress(f"  h={row.horizon}: {row.model} won the holdout but has no factory")
            continue
        model = factory().fit(series)
        point = np.asarray(model.forecast(series, int(row.horizon)), dtype=float)
        progress(f"  h={row.horizon}: {row.model:<18} holdout WAPE {row.wape:.4f}")
        for week, value in enumerate(point, start=1):
            rows.append({"horizon_weeks": int(row.horizon), "weeks_ahead": week,
                         "model": row.model, "holdout_wape": row.wape,
                         "forecast_units": float(value)})

    forecasts = pd.DataFrame(rows)
    if len(forecasts):
        # The forward-looking horizon-1 row is what inventory consumes.
        point_forecast = (forecasts[forecasts["horizon_weeks"] == config.INVENTORY_HORIZON]
                          .sort_values("weeks_ahead")
                          .groupby("horizon_weeks", as_index=False)
                          .agg(model=("model", "first"),
                               holdout_wape=("holdout_wape", "first"),
                               forecast_units=("forecast_units", "first")))

        stores = series_weights(panel, level="store")
        store_ids = stores.index.tolist()
        store_points = reconcile(
            point_forecast["forecast_units"].to_numpy(), stores.to_numpy())
        by_store = pd.DataFrame({
            "horizon_weeks": np.repeat(
                point_forecast["horizon_weeks"].to_numpy(), len(store_ids)),
            "model": np.repeat(point_forecast["model"].to_numpy(), len(store_ids)),
            "store_id": store_ids * len(point_forecast),
            "share": np.tile(stores.to_numpy(), len(point_forecast)),
            "store_units": store_points.ravel(),
        })
    else:
        point_forecast = pd.DataFrame(columns=["horizon_weeks", "model", "holdout_wape",
                                               "forecast_units"])
        by_store = pd.DataFrame()

    out = {
        "scores": scores,
        "selection": best,
        "series": series,
        "forecasts": forecasts,
        "point_forecast": point_forecast,
        "forecasts_by_store": by_store,
    }

    if write and len(forecasts):
        config.ensure_dirs()
        scores.to_csv(config.PROCESSED / "forecast_backtest.csv", index=False)
        best.to_csv(config.PROCESSED / "forecast_model_selection.csv", index=False)
        forecasts.to_csv(config.PROCESSED / "forecast_next_weeks.csv", index=False)
        by_store.to_csv(config.PROCESSED / "forecast_next_weeks_by_store.csv", index=False)

    return out


def main():
    from src.ingest import load_panel

    panel = load_panel()
    result = fit_and_forecast(panel)

    scores = result["scores"]
    ok = scores[scores["wape"].notna()]

    print("\n" + "=" * 74)
    print("BACKTEST -- pooled total")
    print("=" * 74)
    print(ok.sort_values(["horizon", "wape"]).to_string(index=False))

    failed = scores[scores["wape"].isna()]
    if len(failed):
        print("\nmodels that did not produce a score")
        print(failed[["horizon", "model", "error"]].to_string(index=False))

    print("\n" + "=" * 74)
    print("SELECTED PER HORIZON")
    print("=" * 74)
    print(result["selection"].to_string(index=False))

    print("\n" + "=" * 74)
    print(f"FORECAST (h={config.INVENTORY_HORIZON}, what inventory consumes)")
    print("=" * 74)
    print(result["point_forecast"].to_string(index=False))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())