"""Tests for F-04 churn.

The things worth pinning are structural, not numeric. That the split is by date and raises
rather than falling back to random; that the imputer runs inside the pipeline so train
medians cannot leak into test; and that the headline metric is compared against the base
rate instead of against 0.5.

The exact AUC will move if the model is retuned, so it is not asserted. What is asserted is
that `precision@top-k` beats the trivial always-flag-everyone baseline by a wide margin --
that is the claim the report makes.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.churn import (TRAIN_SNAPSHOTS, build_pipeline, check_no_customer_overlap, evaluate,
                       temporal_split)
from src.features import FEATURE_COLUMNS, model_matrix


def make_snapshots(dates=("2025-01-31", "2025-02-28", "2025-03-31"), customers=60, seed=3):
    """A small labelled snapshot frame with the real column contract."""
    rng = np.random.default_rng(seed)
    blocks = []
    for date in dates:
        n = customers
        recency = rng.integers(1, 400, n).astype(float)
        blocks.append(pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "snapshot_date": pd.Timestamp(date),
            "feature_as_of": pd.Timestamp(date),
            "recency_days": recency,
            "frequency": rng.integers(0, 20, n).astype(float),
            "monetary": rng.gamma(2.0, 3000.0, n),
            "avg_basket_value": rng.gamma(3.0, 800.0, n),
            "units_per_line": rng.gamma(3.0, 1.4, n),
            "category_diversity": rng.integers(1, 6, n).astype(float),
            "promo_share": rng.random(n),
            "mean_discount_pct": rng.gamma(2.0, 6.0, n),
            "online_share": rng.random(n),
            "tenure_days": rng.integers(30, 700, n).astype(float),
            # Derived from the SAME recency that is stored as a feature. An earlier version
            # drew a second, independent recency for the label, which made churn unrelated
            # to every column and the model correctly scored lift 1.0.
            #
            # The base rate is kept near 40%, like the real snapshots. At a 77% base rate
            # lift is arithmetically capped at 1/0.77 = 1.30, so any threshold above that is
            # unreachable no matter how good the model is.
            "churn": ((recency > 250) | (rng.random(n) < 0.05)).astype("int8"),
        }))
    return pd.concat(blocks, ignore_index=True)


# ------------------------------------------------------------------------ temporal split

def test_split_is_by_date_not_by_row():
    snapshots = make_snapshots()
    train, test = temporal_split(snapshots, train_snapshots=2)
    assert train["snapshot_date"].max() < test["snapshot_date"].min()
    assert sorted(train["snapshot_date"].unique()) == [
        pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")]
    assert sorted(test["snapshot_date"].unique()) == [pd.Timestamp("2025-03-31")]


def test_split_rows_sum_to_the_input():
    snapshots = make_snapshots()
    train, test = temporal_split(snapshots, train_snapshots=2)
    assert len(train) + len(test) == len(snapshots)


def test_split_refuses_to_leave_nothing_for_testing():
    """A degenerate split must fail loudly rather than report a meaningless score."""
    snapshots = make_snapshots()
    with pytest.raises(ValueError, match="training snapshots"):
        temporal_split(snapshots, train_snapshots=3)
    with pytest.raises(ValueError, match="at least one block"):
        temporal_split(snapshots, train_snapshots=0)


def test_split_shares_no_snapshot_dates():
    snapshots = make_snapshots()
    train, test = temporal_split(snapshots, train_snapshots=2)
    assert check_no_customer_overlap(train, test) is True


def test_overlap_check_catches_a_shared_date():
    snapshots = make_snapshots()
    train = snapshots[snapshots["snapshot_date"] == pd.Timestamp("2025-02-28")]
    test = snapshots[snapshots["snapshot_date"] == pd.Timestamp("2025-02-28")]
    with pytest.raises(ValueError, match="share snapshot dates"):
        check_no_customer_overlap(train, test)


def test_every_snapshot_block_reaches_the_test_side():
    """The brief asks for a forward-looking test set; all of it must be later than training."""
    snapshots = make_snapshots(dates=("2025-01-31", "2025-02-28", "2025-03-31",
                                      "2025-04-30", "2025-05-31", "2025-06-30",
                                      "2025-07-31", "2025-08-31", "2025-09-30"))
    train, test = temporal_split(snapshots, train_snapshots=TRAIN_SNAPSHOTS)
    assert train["snapshot_date"].nunique() == TRAIN_SNAPSHOTS
    assert test["snapshot_date"].nunique() == 9 - TRAIN_SNAPSHOTS
    assert test["snapshot_date"].min() > train["snapshot_date"].max()


# ------------------------------------------------------------------------------ pipeline

def test_pipeline_imputes_before_fitting():
    """Medians must come from the training rows; a pipeline is what guarantees it."""
    pipeline = build_pipeline()
    assert list(pipeline.named_steps) == ["impute", "model"]
    assert pipeline.named_steps["impute"].strategy == "median"


def test_pipeline_uses_the_supplied_class_weight():
    pipeline = build_pipeline(scale_pos_weight=2.0)
    assert pipeline.named_steps["model"].scale_pos_weight == 2.0


def test_dormant_rows_with_nan_survive_the_imputer():
    """Dormant customers have an undefined window average; they are the ones to find.

    Dropping them would shrink the problem and bias it, so they must be imputed, not
    filtered.
    """
    snapshots = make_snapshots(customers=200)
    model = build_pipeline()
    x = model_matrix(snapshots)
    x = x.copy()
    x.loc[x.index[:20], "avg_basket_value"] = np.nan

    metrics = evaluate(model, x.iloc[:400], snapshots["churn"].iloc[:400],
                       x.iloc[400:], snapshots["churn"].iloc[400:],
                       x.columns, k=0.2)
    assert metrics["n_test"] == len(x) - 400
    assert 0.0 <= metrics["auc"] <= 1.0


# ------------------------------------------------------------------------------ matrices

def test_model_matrix_is_numeric_only():
    snapshots = make_snapshots()
    matrix = model_matrix(snapshots)
    assert list(matrix.columns) == FEATURE_COLUMNS
    assert "customer_id" not in matrix.columns
    assert "snapshot_date" not in matrix.columns
    for column in matrix.columns:
        assert pd.api.types.is_numeric_dtype(matrix[column])


def test_model_matrix_rejects_a_missing_column():
    snapshots = make_snapshots().drop(columns=["monetary"])
    with pytest.raises(ValueError, match="monetary"):
        model_matrix(snapshots)


def test_model_matrix_rejects_a_string_column():
    """A stale column list must fail here, not deep inside the imputer."""
    snapshots = make_snapshots().copy()
    snapshots["customer_id"] = "CUS0001"
    with pytest.raises(ValueError, match="not numeric"):
        model_matrix(snapshots, columns=FEATURE_COLUMNS + ["customer_id"])


# ---------------------------------------------------------------------------- evaluation

def test_evaluate_returns_the_metrics_the_brief_asks_for():
    snapshots = make_snapshots(customers=400)
    train, test = temporal_split(snapshots, train_snapshots=2)
    x_train, x_test = model_matrix(train), model_matrix(test)

    metrics = evaluate(build_pipeline(), x_train, train["churn"], x_test, test["churn"],
                       x_train.columns, k=0.2)
    for key in ("auc", "precision_at_top", "recall_at_top", "lift_at_top",
                "test_base_rate", "target_precision", "meets_target"):
        assert key in metrics

    assert metrics["target_precision"] == config.CHURN_PRECISION_TARGET == 0.75
    assert metrics["k"] == config.CHURN_PRECISION_AT_K == 0.20


def test_lift_is_measured_against_the_base_rate():
    """On a ~40% positive population, lift near 1.0 means the model has learned nothing."""
    snapshots = make_snapshots(customers=1500)
    train, test = temporal_split(snapshots, train_snapshots=2)
    x_train, x_test = model_matrix(train), model_matrix(test)

    metrics = evaluate(build_pipeline(), x_train, train["churn"], x_test, test["churn"],
                       x_train.columns, k=0.2)
    assert metrics["lift_at_top"] == pytest.approx(
        metrics["precision_at_top"] / metrics["test_base_rate"], rel=0.02)
    assert metrics["lift_at_top"] > 1.5


def test_a_useless_model_would_score_lift_of_one():
    """Documents why lift is the headline: constant scores tie, and ties rank everything."""
    from src.metrics import lift_at_k
    y = np.array([1, 0] * 50)
    constant = np.full(100, 0.5)
    assert lift_at_k(y, constant, k=0.2) < 1.2


# ---------------------------------------------------------------------------- real data

@pytest.mark.slow
def test_recency_dominates_shap_on_real_data():
    """The single most important sanity check on the churn model.

    If `recency_days` is not the top driver, something is wrong with the label, the window
    or the split -- and the model is not worth reporting.
    """
    from src.churn import run_churn

    result = run_churn(write=False, shap_sample=2000, progress=lambda *_: None)
    importance = result["shap_importance"]
    assert importance["feature"].iloc[0] == "recency_days"
    assert (importance["mean_abs_shap"] >= 0).all()
    assert len(importance["mean_abs_shap"]) == len(FEATURE_COLUMNS)


@pytest.mark.slow
def test_precision_beats_the_trivial_baseline_on_real_data():
    """Flagsging 20% of customers at random scores about 0.43 precision here."""
    from src.churn import run_churn

    result = run_churn(write=False, with_shap=False, progress=lambda *_: None)
    metrics = result["metrics"]
    assert metrics["precision_at_top"] > metrics["test_base_rate"]
    assert metrics["lift_at_top"] > 1.5
    # Every test snapshot is later than every training snapshot.
    assert (result["train"]["snapshot_date"].max()
            < result["test"]["snapshot_date"].min())