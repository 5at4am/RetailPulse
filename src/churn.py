"""F-04 — Churn prediction.

    from src.churn import run_churn
    result = run_churn()      # temporal split, XGBoost, SHAP, top-20% precision

Predicts whether a customer stops buying in the 90 days after a snapshot date, from
behaviour up to that snapshot.

**The split is temporal, and that is the whole point.** Nine monthly snapshots of the same
8,000 customers means every customer appears in several blocks, so a random split would put
the same person on both sides and the model would score well while learning nothing it could
use in production. Snapshots are assigned to train or test by *date*: the earliest blocks
train, the later blocks test, and the test period is entirely in the future relative to
training. Reported precision is therefore an honest forward-looking number.

**Precision@top-20% and lift are the headline metrics, not AUC.** The population is 42.9%
positive and heavily imbalanced, so AUC will look respectable while a classifier that simply
ranks every customer as churned already scores 0.43. `lift` is measured against that base
rate, which is what makes it comparable to the brief's 0.75 target.

**Imputation is fit on train only.** Window aggregates are NaN for dormant customers --
someone who bought nothing in the last 90 days has an undefined average basket. Those rows
are kept, because they are exactly the customers the model needs to find, and the gap is
filled from the training median. A pipeline is used so the same medians apply to both sides
without ever touching the test set.

Runnable on its own: `python -m src.churn`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.features import (FEATURE_COLUMNS, assert_no_leakage, churn_snapshots, feature_matrix,
                          model_matrix)
from src.metrics import lift_at_k, precision_at_k, recall_at_k

# Splits are by date, so no customer block can straddle the boundary.
# Nine snapshots run 2025-01-31 .. 2025-09-30; train on the first six, test on the last two.
TRAIN_SNAPSHOTS = 6


def temporal_split(snapshots, train_snapshots=TRAIN_SNAPSHOTS):
    """Split snapshots by date.

    Raises rather than falling back to a random split. A random split here would leak the
    same customer across the boundary and quietly inflate every metric in the report.
    """
    if train_snapshots < 1:
        raise ValueError("train_snapshots must leave at least one block for training")

    dates = sorted(pd.to_datetime(snapshots["snapshot_date"]).unique())
    if train_snapshots >= len(dates):
        raise ValueError(
            f"asked for {train_snapshots} training snapshots but only {len(dates)} exist")

    boundary = dates[train_snapshots]
    train = snapshots[pd.to_datetime(snapshots["snapshot_date"]) < boundary].copy()
    test = snapshots[pd.to_datetime(snapshots["snapshot_date"]) >= boundary].copy()
    return train, test


def check_no_customer_overlap(train, test):
    """A customer in both sides means the temporal split did not actually hold.

    Overlap across *different* snapshot dates is expected and fine -- it is a different
    point in time for that customer. Overlap at the *same* snapshot date would not be, and
    cannot happen from a date-based split, so this asserts the weaker but useful property
    that no snapshot date is shared.
    """
    shared_dates = set(pd.to_datetime(train["snapshot_date"]).unique()) & \
        set(pd.to_datetime(test["snapshot_date"]).unique())
    if shared_dates:
        raise ValueError(f"train and test share snapshot dates: {sorted(shared_dates)}")
    return True


def build_pipeline(scale_pos_weight=None):
    """Impute then classify. Imputer medians come from the training rows only."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    # scale_pos_weight corrects for the 42.9% base rate. A model optimised for accuracy on
    # a 57/43 split is not badly wrong, but it is not useful either.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=20,
            scale_pos_weight=scale_pos_weight,
            random_state=config.SEED,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=4,
        )),
    ])


def evaluate(model, train_x, train_y, test_x, test_y, features, k=config.CHURN_PRECISION_AT_K):
    """Fit, then score. Precision@top-k and lift are the headline numbers."""
    from sklearn.metrics import roc_auc_score

    model.fit(train_x, train_y)
    scores = model.predict_proba(test_x)[:, 1]

    return {
        "n_train": len(train_y),
        "n_test": len(test_y),
        "train_base_rate": round(float(train_y.mean()), 4),
        "test_base_rate": round(float(test_y.mean()), 4),
        "auc": round(float(roc_auc_score(test_y, scores)), 4),
        "precision_at_top": round(precision_at_k(test_y, scores, k), 4),
        "recall_at_top": round(recall_at_k(test_y, scores, k), 4),
        "lift_at_top": round(lift_at_k(test_y, scores, k), 4),
        "k": k,
        "target_precision": config.CHURN_PRECISION_TARGET,
        "meets_target": bool(precision_at_k(test_y, scores, k) >= config.CHURN_PRECISION_TARGET),
    }


def shap_importance(model, test_x, features, top_n=15):
    """Mean |SHAP| per feature.

    TreeSHAP, not `feature_importances_`: gain-based importance is inflated by
    high-cardinality splits and cannot separate a feature's direction from its magnitude.
    SHAP gives signed contributions, which is what "why was this customer flagged" needs.

    The pipeline is unwrapped before explaining. TreeExplainer rejects a `Pipeline`
    outright, and explaining the imputed matrix with the booster that was actually fitted is
    both what SHAP requires and the honest description of the model: the booster never saw
    the NaNs, so SHAP must not either. The imputer is `transform`ed, never re-fit, so the
    training medians still apply.
    """
    import shap

    steps = getattr(model, "named_steps", None)
    if steps is not None:
        if "impute" in steps:
            matrix = steps["impute"].transform(test_x)
        else:
            matrix = test_x
        booster = steps["model"]
    else:
        matrix, booster = test_x, model

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):          # some shap versions return one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:                  # (rows, features, classes)
        values = values[:, :, -1]

    importance = pd.DataFrame({
        "feature": list(features),
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return importance.head(top_n).reset_index(drop=True), values


def run_churn(train_snapshots=TRAIN_SNAPSHOTS, top_k=config.CHURN_PRECISION_AT_K, with_shap=True,
              shap_sample=4000, write=True, progress=print):
    """End-to-end: snapshots -> temporal split -> model -> metrics -> SHAP."""
    from src.ingest import load_sales

    sales = load_sales()

    progress("building churn snapshots (this is the slow part)")
    snapshots = churn_snapshots(sales, progress=progress)
    assert_no_leakage(snapshots, allow_labels=True)

    train, test = temporal_split(snapshots, train_snapshots)
    check_no_customer_overlap(train, test)

    progress(f"\ntrain snapshots {train['snapshot_date'].nunique()} "
             f"({train['snapshot_date'].min():%Y-%m} to {train['snapshot_date'].max():%Y-%m})")
    progress(f"test snapshots  {test['snapshot_date'].nunique()} "
             f"({test['snapshot_date'].min():%Y-%m} to {test['snapshot_date'].max():%Y-%m})")

    # `model_matrix` drops customer_id and the date columns; `feature_matrix` keeps them for
    # joining and auditing, so neither one alone is right for both jobs.
    train_x = model_matrix(train)
    test_x = model_matrix(test)
    train_y = train["churn"].to_numpy()
    test_y = test["churn"].to_numpy()

    progress(f"\ntrain rows {len(train):,}  churn {train_y.mean():.1%}")
    progress(f"test rows  {len(test):,}  churn {test_y.mean():.1%}")

    model = build_pipeline(scale_pos_weight=float((1 - train_y.mean()) / train_y.mean()))
    metrics = evaluate(model, train_x, train_y, test_x, test_y, train_x.columns, k=top_k)

    progress("\nresults")
    for key in ("auc", "precision_at_top", "recall_at_top", "lift_at_top"):
        progress(f"  {key:<20} {metrics[key]}")
    progress(f"  {'test base rate':<20} {metrics['test_base_rate']}")
    progress(f"  brief target        precision@top-{int(top_k * 100)}% "
             f">= {config.CHURN_PRECISION_TARGET}")
    progress(f"  meets target        {metrics['meets_target']}")

    result = {
        "metrics": metrics,
        "model": model,
        "snapshots": snapshots,
        "train": train,
        "test": test,
        "features": list(train_x.columns),
    }

    if with_shap:
        progress("\ncomputing SHAP values")
        sample = test_x.head(shap_sample)
        importance, values = shap_importance(model, sample, train_x.columns)
        result["shap_importance"] = importance
        result["shap_values"] = values
        progress("\ntop features by mean |SHAP|")
        for row in importance.itertuples():
            progress(f"  {row.feature:<24} {row.mean_abs_shap:.4f}")

    if write:
        config.ensure_dirs()
        metrics_frame = pd.DataFrame([metrics])
        metrics_frame.to_csv(config.PROCESSED / "churn_metrics.csv", index=False)
        if with_shap:
            result["shap_importance"].to_csv(
                config.PROCESSED / "churn_shap_importance.csv", index=False)
        scored = test[["customer_id", "snapshot_date", "churn"]].copy()
        scored["churn_score"] = model.predict_proba(test_x)[:, 1]
        scored.sort_values("churn_score", ascending=False).to_csv(
            config.PROCESSED / "churn_scores.csv", index=False)

    return result


def main():
    run_churn()
    return 0


if __name__ == "__main__":
    sys.exit(main())