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


def local_explanations(model, test_x, features, scores, n=25, identifiers=None,
                       reasons=config.SHAP_LOCAL_REASONS):
    """Per-customer SHAP contributions for the highest-risk cohort.

    Global importance answers "what does the model uses". It does not answer "why is *this*
    customer flagged", which is the question a retention team asks when they are about to call
    someone. The brief asks for both, and the local half was the missing piece.

    Contributions are signed, so a customer can be flagged for two opposite reasons: a high
    `recency_days` pushes risk up while a large `tenure_days` pushes it down. Averaging those
    into one importance number would hide the case entirely. For that reason the ranking is by
    magnitude, not by signed value.

    UNITS. For a classifier, `TreeExplainer` works in the model's link space, so for this
    gradient-boosted model the contributions are in LOG-ODDS, not probability. A contribution
    of +2.19 therefore does not mean "adds 2.19 to a probability" -- it cannot, probabilities
    are bounded. It means two log-odds. The identity that holds is

        sum(contributions) + expected_value == logit(predict_proba)

    and `expected_value` is negative here (-0.627), i.e. the base rate is already below 50%.
    Reading the column as a probability delta is the single easiest mistake to make with this
    artefact, so it is stated here and asserted in tests/test_churn_explainability.py.
    """
    import shap

    steps = getattr(model, "named_steps", None)
    if steps is not None:
        matrix = steps["impute"].transform(test_x) if "impute" in steps else test_x
        booster = steps["model"]
    else:
        matrix, booster = test_x, model

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    order = np.argsort(scores)[::-1][:n]
    rows = []
    for row in order:
        contribution = values[row]
        # Rank by MAGNITUDE, not by signed value.
        #
        # Ranking by signed value sorts every risk-reducer below every risk-raiser, so a
        # customer whose score is being held down by a very long tenure never sees that
        # reason at all. The docstring's whole point is that opposite-signed reasons both
        # matter, which is only true if the ordering is by |contribution|. The sign is then
        # preserved in the column itself, so the direction is still readable.
        ranked = np.argsort(np.abs(contribution))[::-1]
        # Publish the strongest `reasons` only. In the real output ranks 7-10 sit around
        # 1e-3 log-odds -- noise. Emitting all ten would make the per-customer file a
        # reprint of the global importance table, which defeats the point of having one.
        ranked = ranked[:max(1, min(reasons, len(ranked)))]
        for rank, column in enumerate(ranked, start=1):
            rows.append({
                "customer_id": (identifiers.iloc[row]["customer_id"]
                                if identifiers is not None else None),
                "snapshot_date": (identifiers.iloc[row]["snapshot_date"]
                                  if identifiers is not None else None),
                "row": int(row),
                "churn_score": round(float(scores[row]), 6),
                "feature_rank": rank,
                "feature": features[column],
                "feature_value": float(matrix[row][column]),
                "shap_contribution": round(float(contribution[column]), 6),
            })
    frame = pd.DataFrame(rows)
    frame.attrs["top_n"] = n
    return frame



def evaluate(model, train_x, train_y, test_x, test_y, features, k=config.CHURN_PRECISION_AT_K):
    """Fit, then score. Precision@top-k and lift are the headline numbers.

    Also reports PR-AUC, F1 and the confusion matrix at the operating threshold. With a 42%
    base rate, ROC-AUC flatters a model, and PR-AUC is the measure that does not -- a 0.73
    ROC-AUC can sit against a much weaker PR-AUC, and only the latter says what happens when
    a retention team works the list.
    """
    from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)

    model.fit(train_x, train_y)
    scores = scores_of(model, test_x)

    # The operating threshold is the one the brief actually cares about: the score that
    # selects the top k of the list, rather than 0.5, which on a 42% base rate classifies
    # almost everything as churned.
    cutoff = float(np.sort(scores)[::-1][max(int(len(scores) * k) - 1, 0)])
    predicted = (scores >= cutoff).astype(int)
    counts = confusion_of(test_y, predicted)

    return {
        "n_train": len(train_y),
        "n_test": len(test_y),
        "train_base_rate": round(float(train_y.mean()), 4),
        "test_base_rate": round(float(test_y.mean()), 4),
        "auc": round(float(roc_auc_score(test_y, scores)), 4),
        "pr_auc": round(float(average_precision_score(test_y, scores)), 4),
        "f1_at_operating_threshold": round(float(f1_score(test_y, predicted, zero_division=0)), 4),
        # Published at full precision, NOT rounded to 4dp like the other metrics.
        #
        # The confusion matrix below is computed from the unrounded `cutoff`, so rounding the
        # published threshold makes the two files disagree: recomputing tn/fp/fn/tp from
        # churn_scores.csv at the printed threshold gives a different matrix, because ~40
        # scores sit inside the rounding band. A threshold a reader cannot reproduce the
        # matrix from is not a published number. Keep this one exact.
        "operating_threshold": cutoff,
        "tn": counts["tn"], "fp": counts["fp"],
        "fn": counts["fn"], "tp": counts["tp"],
        "precision_at_top": round(precision_at_k(test_y, scores, k), 4),
        "recall_at_top": round(recall_at_k(test_y, scores, k), 4),
        "lift_at_top": round(lift_at_k(test_y, scores, k), 4),
        "k": k,
        "target_precision": config.CHURN_PRECISION_TARGET,
        "meets_target": bool(precision_at_k(test_y, scores, k) >= config.CHURN_PRECISION_TARGET),
        "meets_auc_target": bool(roc_auc_score(test_y, scores) >= config.CHURN_AUC_TARGET),
    }


def scores_of(model, x):
    """Probabilities as float64, always.

    `XGBClassifier.predict_proba` returns float32. That is normally harmless, but it is
    not harmless here: the published `operating_threshold` is a float64 taken from that
    float32 array, and roughly 40 scores sit exactly ON the cutoff. Comparing a float32 array
    against a float64 scalar in NumPy casts the scalar DOWN to float32, so those 40 ties are
    included in memory. Once the same scores are written to CSV as float64 and read back,
    the comparison happens in float64 and the cutoff's low bits fall the other way, so those
    40 rows drop out and the published matrix stops matching the published scores.

    Widening once, here, means the in-memory result and the CSV agree exactly.
    """
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def confusion_of(actual, predicted):
    """The four confusion cells from labels and a binary prediction.

    Split out of `evaluate` so the published matrix can be recomputed independently from
    `churn_scores.csv`. Without that, a transposed or stale matrix would sit in the CSV
    looking entirely plausible, and nobody would have a way to tell.
    """
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])}


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

    progress("\\nresults")
    for key in ("auc", "pr_auc", "f1_at_operating_threshold",
                "precision_at_top", "recall_at_top", "lift_at_top"):
        progress(f"  {key:<28} {metrics[key]}")
    progress(f"  {'test base rate':<28} {metrics['test_base_rate']}")
    progress(f"  {'operating threshold':<28} {metrics['operating_threshold']}")
    progress(f"  {'confusion tn/fp/fn/tp':<28} "
             f"{metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}")
    progress(f"  {'brief targets':<28} precision@top-{int(top_k * 100)}% "
             f">= {config.CHURN_PRECISION_TARGET}, AUC >= {config.CHURN_AUC_TARGET}")
    progress(f"  {'meets precision':<28} {metrics['meets_target']}")
    progress(f"  {'meets AUC':<28} {metrics['meets_auc_target']}")
    if not metrics["meets_auc_target"]:
        progress(f"  ^ AUC missed by {config.CHURN_AUC_TARGET - metrics['auc']:.4f}. "
                 f"PR-AUC {metrics['pr_auc']} is the number that describes the list a "
                 f"retention team actually works.")

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

        # Per-customer explanations for the cohort most likely to be called. Global
        # importance cannot answer "why this customer", which is the question asked at the
        # moment someone dials the number.
        scored_test = scores_of(model, test_x)
        local = local_explanations(model, test_x, train_x.columns, scored_test,
                                   n=config.SHAP_LOCAL_TOP_N,
                                   identifiers=test[["customer_id", "snapshot_date"]])
        result["shap_local"] = local
        worst = local[local["feature_rank"] == 1].nsmallest(1, "row")
        if len(worst):
            first = worst.iloc[0]
            progress(f"\nlocal explanation, highest-risk customer "
                     f"{first['customer_id']} "
                     f"(score {first['churn_score']:.4f})")
            for row in local[local["row"] == first["row"]].nsmallest(4, "feature_rank").itertuples():
                direction = "raises risk" if row.shap_contribution > 0 else "lowers risk"
                progress(f"  {row.feature:<24} value {row.feature_value:>10.2f}  "
                         f"{row.shap_contribution:+.4f}  {direction}")

    if write:
        config.ensure_dirs()
        metrics_frame = pd.DataFrame([metrics])
        metrics_frame.to_csv(config.PROCESSED / "churn_metrics.csv", index=False)
        if with_shap:
            result["shap_importance"].to_csv(
                config.PROCESSED / "churn_shap_importance.csv", index=False)
            result["shap_local"].to_csv(
                config.PROCESSED / "churn_shap_local.csv", index=False)
        scored = test[["customer_id", "snapshot_date", "churn"]].copy()
        scored["churn_score"] = scores_of(model, test_x)
        scored.sort_values("churn_score", ascending=False).to_csv(
            config.PROCESSED / "churn_scores.csv", index=False)

    return result


def main():
    run_churn()
    return 0


if __name__ == "__main__":
    sys.exit(main())