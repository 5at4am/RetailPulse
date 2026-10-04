"""Forecast and classification metrics, with the traps in this dataset handled explicitly.

    from src.metrics import forecast_metrics, precision_at_k

    forecast_metrics(y_true, y_pred)      # WAPE primary, MAPE on non-zero rows only

**Why MAPE is not the headline here.** MAPE divides by the actual value, and 77.9% of
panel rows have `units_sold == 0`, so MAPE is undefined on more than three quarters of the
data. Any MAPE number printed without saying which rows it covers is meaningless. So:

- **WAPE is primary** -- `sum|y - f| / sum|y|`. Safe with zeros, and the natural choice
  when demand is lumpy.
- **MAPE is reported on the non-zero subset only**, and the subset size travels with the
  number (`mape_nonzero_pct` says what fraction of rows it covered).
- MAE, RMSE, MASE and signed bias come alongside, because they answer different questions:
  MAE is in the units you order in, MASE says whether you beat a naive forecast, and
  bias says whether you are systematically short or long -- which is the difference
  between "slightly wrong" and "under-orders every week and stockouts".

Stdlib + numpy only, so this module is cheap to import and has no heavyweight
dependency that could fail to resolve on Streamlit Cloud.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _as_float_pair(y_true, y_pred):
    """Two equal-length 1-D float arrays, with non-finite values rejected."""
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"y_true and y_pred differ in length: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("cannot score an empty forecast")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("y_true and y_pred must not contain NaN or inf")
    return a, b


def wape(y_true, y_pred):
    """Weighted absolute percentage error: `sum|y - f| / sum|y|`.

    The primary metric for this project. Defined when actual demand is zero, unlike MAPE.

    Returns NaN when every actual is zero -- the error is genuinely undefined, and
    silently returning 0.0 would look like a perfect forecast.
    """
    a, b = _as_float_pair(y_true, y_pred)
    denominator = np.abs(a).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(a - b).sum() / denominator)


def mape(y_true, y_pred, nonzero_only=True):
    """Mean absolute percentage error.

    `nonzero_only=True` (the default) scores only rows where the actual is non-zero, and
    is what gets compared against the brief's 12% target. Pass `False` only on data with
    no zeros -- it will divide by zero otherwise.
    """
    a, b = _as_float_pair(y_true, y_pred)
    mask = a > 0 if nonzero_only else np.ones_like(a, dtype=bool)
    if not mask.any():
        return float("nan")
    return float((np.abs(a[mask] - b[mask]) / np.abs(a[mask])).mean())


def smape(y_true, y_pred):
    """Symmetric MAPE: `mean(2|y - f| / (|y| + |f|))`. Defined at zero, unlike MAPE."""
    a, b = _as_float_pair(y_true, y_pred)
    denominator = np.abs(a) + np.abs(b)
    mask = denominator > 0
    if not mask.any():
        return float("nan")
    return float((2 * np.abs(a[mask] - b[mask]) / denominator[mask]).mean())


def mae(y_true, y_pred):
    """Mean absolute error, in units sold."""
    a, b = _as_float_pair(y_true, y_pred)
    return float(np.abs(a - b).mean())


def rmse(y_true, y_pred):
    """Root mean squared error -- punishes large misses harder than MAE."""
    a, b = _as_float_pair(y_true, y_pred)
    return float(np.sqrt(((a - b) ** 2).mean()))


def bias(y_true, y_pred):
    """Mean signed error, `mean(y_true - y_pred)`.

    Positive means the forecast runs **above** actual (over-forecast, excess stock);
    negative means it runs below (under-forecast, stockouts). A model with a good WAPE
    but a large negative bias is still operationally broken.
    """
    a, b = _as_float_pair(y_true, y_pred)
    return float((a - b).mean())


def mase(y_true, y_pred, insample, seasonality=1):
    """Mean absolute scaled error.

    `MAE(model) / MAE(naive)`, where the naive benchmark repeats the value from
    `seasonality` weeks earlier on the **in-sample** series. `< 1` means the model beats
    that benchmark. Scales the error by the data's own volatility, so it is comparable
    across series with different volumes.

    `insample` must be training data that ends before `y_true` begins -- passing the test
    window would leak. That is the caller's responsibility, and `test_no_leakage.py`
    exists to check it.
    """
    a, b = _as_float_pair(y_true, y_pred)
    history = np.asarray(insample, dtype=float).ravel()
    if history.size < seasonality + 1:
        raise ValueError(
            f"mase needs at least {seasonality + 1} in-sample values for "
            f"seasonality={seasonality}, got {history.size}")
    if not np.isfinite(history).all():
        raise ValueError("insample must not contain NaN or inf")

    naive_errors = np.abs(history[seasonality:] - history[:-seasonality])
    scale = naive_errors.mean()
    if scale == 0:
        return float("nan")
    return float(np.abs(a - b).mean() / scale)


def naive_forecast(history, horizon, seasonality=1):
    """Repeat the value from `seasonality` steps back, the MASE benchmark."""
    history = list(history)
    if len(history) < seasonality:
        raise ValueError(f"history shorter than seasonality={seasonality}")
    return [history[-seasonality + (i % seasonality)] for i in range(horizon)]


def forecast_metrics(y_true, y_pred, insample=None, seasonality=1):
    """Every forecast metric at once, as a dict ready to be written to JSON.

    `insample` is optional; pass it and MASE is computed. When it is omitted, MASE comes
    back None rather than being quietly filled with something else.
    """
    a, b = _as_float_pair(y_true, y_pred)
    nonzero_pct = 100.0 * float((a > 0).mean())

    out = {
        "n": int(a.size),
        "n_nonzero_actual": int((a > 0).sum()),
        "nonzero_actual_pct": round(nonzero_pct, 2),
        "wape": round(wape(a, b), 6),
        "mape_nonzero": round(mape(a, b), 6),
        "smape": round(smape(a, b), 6),
        "mae": round(mae(a, b), 6),
        "rmse": round(rmse(a, b), 6),
        "bias": round(bias(a, b), 6),
        "total_actual": float(a.sum()),
        "total_forecast": float(b.sum()),
    }
    out["mase"] = (round(mase(a, b, insample, seasonality), 6)
                   if insample is not None else None)
    return out


# --------------------------------------------------------------------------- ranking

def precision_at_k(y_true, scores, k=0.20):
    """Precision among the top `k` fraction of cases ranked by `scores`.

    The brief's churn criterion is "precision ≥ 0.75 among the top 20% highest-risk
    customers", which is this function with `k=0.20`. Top-k-by-rank rather than a
    probability threshold, so it is stable regardless of calibration.
    """
    a = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if a.shape != s.shape:
        raise ValueError(f"y_true and scores differ in length: {a.shape} vs {s.shape}")
    if a.size == 0:
        raise ValueError("cannot score an empty set")
    if not 0 < k <= 1:
        raise ValueError(f"k must be the top fraction in (0, 1], got {k}")

    n_top = max(1, int(round(k * a.size)))
    # Highest score first. Ties broken by original position, so the result is
    # deterministic rather than dependent on the sort implementation.
    order = np.argsort(-s, kind="stable")[:n_top]
    return float(a[order].mean())


def recall_at_k(y_true, scores, k=0.20):
    """Share of all positives captured by the top `k` fraction by score."""
    a = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if a.shape != s.shape:
        raise ValueError(f"y_true and scores differ in length: {a.shape} vs {s.shape}")
    if not 0 < k <= 1:
        raise ValueError(f"k must be the top fraction in (0, 1], got {k}")

    total_positive = a.sum()
    if total_positive == 0:
        return float("nan")
    n_top = max(1, int(round(k * a.size)))
    order = np.argsort(-s, kind="stable")[:n_top]
    return float(a[order].sum() / total_positive)


def lift_at_k(y_true, scores, k=0.20):
    """Precision in the top `k` divided by the base positive rate.

    1.0 means the ranking is no better than picking at random. This is what makes the
    top-20% precision number interpretable: 0.75 precision is impressive at a 4% base
    rate and useless at a 70% one.
    """
    a = np.asarray(y_true, dtype=float).ravel()
    base = a.mean()
    if base == 0:
        return float("nan")
    return precision_at_k(a, scores, k) / base


def main():
    """Self-check on hand-computable values. Run: python -m src.metrics"""
    y_true = [0, 0, 10, 20, 0, 40]
    y_pred = [0, 5, 12, 18, 10, 30]

    m = forecast_metrics(y_true, y_pred)
    print("=" * 68)
    print("METRICS SELF-CHECK")
    print("=" * 68)
    for key, value in m.items():
        print(f"  {key:<20} {value}")

    print("\nby hand:")
    print(f"  WAPE  = (0+5+2+2+10+10) / 70 = 29/70      = {29 / 70:.6f}")
    print(f"  MAPE  = mean(2/10, 2/20, 10/40)          = {(0.2 + 0.1 + 0.25) / 3:.6f}")
    print("  ^ MAPE scored on 3 of 6 rows -- the three non-zero actuals only.")
    print("\n  sMAPE is 0.9146, much worse than WAPE, because a row with actual 0 and")
    print("  forecast 10 scores the maximum 2.0. sMAPE is defined at zero, but it is")
    print("  brutal about forecasting anything where demand was nil -- another reason")
    print("  WAPE leads here.")

    # forecast_metrics rounds to 6 dp for JSON output, so compare at that precision.
    assert abs(m["wape"] - 29 / 70) < 1e-6
    assert abs(m["mape_nonzero"] - (0.2 + 0.1 + 0.25) / 3) < 1e-6
    assert abs(m["bias"] - (-5 / 6)) < 1e-6
    assert m["nonzero_actual_pct"] == 50.0
    assert m["n_nonzero_actual"] == 3
    print("\nall assertions passed")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())