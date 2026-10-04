"""Churn operating metrics, confusion matrix and per-customer local SHAP.

Covers the three pieces added in the final audit pass:

  * PR-AUC and F1 at a realistic operating threshold, not just ROC-AUC
  * the confusion matrix, published so precisely that it can be recomputed from the scores
  * local SHAP explanations that name a customer rather than a row index

The local-explanation tests matter most. A global-importance file and a local file with the
wrong row-to-customer mapping look identical on inspection, and the second one is worthless
to the retention team that is supposed to act on it.

The fixture deliberately gives the test period a lower churn rate than the training period.
That is the real shape of this problem, and a fixture with a stable prior would let a
threshold pass for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from src import config
from src.churn import (
    FEATURE_COLUMNS,
    build_pipeline,
    confusion_of,
    evaluate,
    lift_at_k,
    local_explanations,
    precision_at_k,
    recall_at_k,
)

RNG = np.random.default_rng(11)
TOP_REASONS = 6          # reasons published per customer
CUSTOMERS = 10           # size of the explained cohort


def make_split(n_train=4000, n_test=1500, train_rate=0.35, test_rate=0.18):
    """A temporal-style split over the REAL feature schema.

    Built from `FEATURE_COLUMNS` rather than a hand-written column list. A fixture with its
    own six invented names would pass even if `local_explanations` zipped the wrong label
    onto each contribution, because there is nothing to misalign against. Using the project's
    actual ten features is what gives the rank-order assertion teeth -- and it makes the
    6-of-10 truncation a real one instead of a no-op.
    """
    rng = np.random.default_rng(11)

    def draw(n):
        return pd.DataFrame({
            "recency_days": rng.gamma(2.0, 40.0, n),
            "frequency": rng.poisson(6, n).astype(float),
            "monetary": rng.gamma(2.0, 500.0, n),
            "avg_basket_value": rng.gamma(3.0, 30.0, n),
            "units_per_line": rng.gamma(2.0, 1.5, n),
            "category_diversity": rng.poisson(4, n).astype(float),
            "promo_share": rng.beta(2, 8, n),
            "mean_discount_pct": rng.uniform(0, 40, n),
            "online_share": rng.beta(5, 5, n),
            "tenure_days": rng.gamma(3.0, 90.0, n),
        })

    def label(x, rate):
        # Drive the label off real relationships rather than noise, so these tests assert
        # metrics on a learnable problem instead of an arbitrary one.
        risk = (x["recency_days"] / 250.0
                + (1 - x["promo_share"])
                + (1 - x["online_share"]) * 0.5
                + (x["frequency"] < 2).astype(float) * 0.4)
        return (risk >= np.quantile(risk, 1 - rate)).astype(int)

    x_train, x_test = draw(n_train), draw(n_test)
    return (x_train[FEATURE_COLUMNS], label(x_train, train_rate),
            x_test[FEATURE_COLUMNS], label(x_test, test_rate))


def separable_frame(n=200):
    """All ten real features varying, with a learnable signal.

    Every column must vary. A constant column makes the fit degenerate, and a fixture with
    only one varying feature is a much weaker test than it looks.
    """
    rng = np.random.default_rng(5)
    half = n // 2
    churn = np.concatenate([np.zeros(half), np.ones(half)])
    x = pd.DataFrame({
        "recency_days": np.where(churn == 1, 900.0, 5.0) + rng.normal(0, 5, n),
        "frequency": np.where(churn == 1, 0.0, 12.0) + rng.normal(0, 0.5, n),
        "monetary": rng.uniform(1, 5000, n),
        "avg_basket_value": rng.uniform(1, 300, n),
        "units_per_line": rng.uniform(0.1, 9, n),
        "category_diversity": rng.poisson(4, n).astype(float),
        "promo_share": np.where(churn == 1, 0.0, 0.4) + rng.normal(0, 0.02, n),
        "mean_discount_pct": rng.uniform(0, 40, n),
        "online_share": np.where(churn == 1, 0.05, 0.95) + rng.normal(0, 0.02, n),
        "tenure_days": rng.uniform(1, 900, n),
    })
    return x[FEATURE_COLUMNS], churn.astype(int)


@pytest.fixture(scope="module")
def split():
    return make_split()


@pytest.fixture(scope="module")
def scores(split):
    x_train, y_train, x_test, _ = split
    model = build_pipeline()
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


@pytest.fixture(scope="module")
def metrics(split):
    x_train, y_train, x_test, y_test = split
    return evaluate(build_pipeline(), x_train, y_train, x_test, y_test, FEATURE_COLUMNS)


class TestOperatingMetrics:
    def test_pr_auc_is_below_auc_for_an_imbalanced_problem(self, metrics):
        # PR-AUC weights the positive class, so on an imbalanced problem it sits below
        # ROC-AUC. If it ever exceeds AUC here, one of the two is reading the wrong labels.
        assert metrics["pr_auc"] < metrics["auc"]
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_confusion_counts_reconstruct_the_labels(self, metrics, split):
        _, _, _, y_test = split
        # The check that catches a transposed confusion matrix, which is otherwise very easy
        # to publish because it still looks like a plausible 2x2 table.
        assert (metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
                == len(y_test))
        # Churners split between caught and missed; everyone else between false alarms and
        # correct non-calls. Both partitions must be exact.
        assert metrics["tp"] + metrics["fn"] == int(y_test.sum())
        assert metrics["tn"] + metrics["fp"] == len(y_test) - int(y_test.sum())

    def test_precision_and_f1_follow_from_the_matrix(self, metrics):
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        # Both must be derivable from the same published cells. Disagreement means one of
        # them was computed at a different cutoff than the matrix.
        assert metrics["precision_at_top"] == pytest.approx(precision, abs=0.01)
        assert metrics["f1_at_operating_threshold"] == pytest.approx(f1, abs=0.01)

    def test_operating_threshold_calls_about_the_target_share(self, metrics, split):
        _, _, _, y_test = split
        expected_called = len(y_test) * metrics["k"]
        called = metrics["tp"] + metrics["fp"]
        # Ties on the score push this over by a few, which is why it is a band and not an
        # equality.
        assert called == pytest.approx(expected_called, abs=len(y_test) * 0.02)

    def test_f1_is_not_one_for_a_realistic_model(self, metrics):
        # A perfect F1 on a real split means labels leaked. Pinning this keeps a future
        # fixture change from turning the suite trivially green.
        assert metrics["f1_at_operating_threshold"] < 0.99

    def test_target_flags_agree_with_their_metrics(self, metrics):
        assert metrics["meets_target"] == bool(
            metrics["precision_at_top"] >= config.CHURN_PRECISION_TARGET)
        assert metrics["meets_auc_target"] == bool(
            metrics["auc"] >= config.CHURN_AUC_TARGET)

    def test_precision_at_k_is_recomputable_by_hand(self, split, scores):
        _, _, _, y_test = split
        k = 0.20
        called = int(len(y_test) * k)
        top = np.argsort(scores)[::-1][:called]
        assert precision_at_k(y_test, scores, k) == pytest.approx(y_test[top].sum() / called)
        assert recall_at_k(y_test, scores, k) == pytest.approx(
            y_test[top].sum() / y_test.sum())

    def test_lift_is_precision_over_base_rate(self, split, scores):
        _, _, _, y_test = split
        k = 0.20
        called = int(len(y_test) * k)
        top = np.argsort(scores)[::-1][:called]
        assert lift_at_k(y_test, scores, k) == pytest.approx(
            (y_test[top].sum() / called) / y_test.mean())

    def test_precision_at_k_beats_the_base_rate(self, metrics, split):
        _, _, _, y_test = split
        assert metrics["precision_at_top"] > y_test.mean(), (
            "the call list is no better than picking at random"
        )
        assert metrics["lift_at_top"] > 1.0

    def test_a_perfectly_separating_model_scores_one(self):
        x, y = separable_frame()
        result = evaluate(build_pipeline(), x, y, x, y, list(x.columns), k=0.20)
        assert result["auc"] == pytest.approx(1.0, abs=0.01)
        assert result["f1_at_operating_threshold"] == pytest.approx(1.0, abs=0.02)
        assert result["precision_at_top"] == pytest.approx(1.0, abs=0.02)

    def test_inverting_the_strongest_feature_cannot_improve_the_score(self, split):
        """A sign flip must make the model worse.

        `evaluate` fits the model itself, so the flip is injected by handing it data whose
        strongest feature has been negated -- not by flipping the returned numbers, which
        would only prove that arithmetic works.
        """
        x_train, y_train, x_test, y_test = split
        model = build_pipeline()
        model.fit(x_train, y_train)
        good = model.predict_proba(x_test)[:, 1]

        flipped_train = x_train.copy()
        flipped_train["recency_days"] = -flipped_train["recency_days"]
        flipped_features = x_test.copy()
        flipped_features["recency_days"] = -flipped_features["recency_days"]
        flipped = build_pipeline().fit(flipped_train, y_train)
        flipped_scores = flipped.predict_proba(flipped_features)[:, 1]

        assert roc_auc_score(y_test, good) > 0.6, "the unflipped model should work"
        assert roc_auc_score(y_test, flipped_scores) < roc_auc_score(y_test, good), (
            "inverting the strongest feature must not improve the score"
        )


@pytest.fixture(scope="class")
def local(split):
    x_train, y_train, x_test, _ = split
    model = build_pipeline()
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    identifiers = pd.DataFrame({
        "customer_id": [f"CUS{i:05d}" for i in range(len(x_test))],
        "snapshot_date": pd.Timestamp("2025-09-30"),
    })
    frame = local_explanations(model, x_test, list(x_test.columns), scores,
                               n=CUSTOMERS, identifiers=identifiers)
    return frame, model, scores, x_test


class TestLocalExplanations:
    def test_names_a_customer_not_a_row_index(self, local):
        frame, _, _, _ = local
        # The whole point. A file of row numbers cannot be acted on.
        assert frame["customer_id"].notna().all()
        assert frame["customer_id"].str.startswith("CUS").all()
        assert frame["snapshot_date"].notna().all()

    def test_covers_n_customers_with_their_top_reasons(self, local):
        frame, _, _, _ = local
        # Six reasons per customer, not every feature. The brief asks for the reasons a
        # caller would hear on the phone, and six reasons per customer is not that.
        assert frame["customer_id"].nunique() == CUSTOMERS
        assert frame["feature_rank"].max() == TOP_REASONS
        for customer, group in frame.groupby("customer_id"):
            assert sorted(group["feature_rank"]) == list(range(1, TOP_REASONS + 1)), (
                f"{customer}: feature_rank has a gap or a duplicate"
            )

    def test_ranks_by_magnitude_not_by_signed_value(self, local):
        frame, _, _, _ = local
        # Ranking by signed value sorts every risk-reducer below every risk-raiser, so a
        # customer held down by a long tenure never sees that reason at all. The sign is
        # still readable in the column itself.
        first = frame.iloc[0]
        same = frame[frame["customer_id"] == first["customer_id"]]
        assert abs(first["shap_contribution"]) == same["shap_contribution"].abs().max()

    def test_contributions_are_signed(self, local):
        frame, _, _, _ = local
        # Signed, not absolute: a customer can be flagged for long absence AND protected by
        # long tenure. Averaging the sign away hides both reasons, which is the one thing
        # the local view exists to show.
        assert (frame["shap_contribution"] < 0).any(), (
            "no negative contribution anywhere, so the signs were probably absolute values"
        )

    def test_selects_the_highest_scoring_customers(self, local):
        frame, _, scores, _ = local
        expected = {f"CUS{i:05d}" for i in np.argsort(scores)[::-1][:CUSTOMERS]}
        assert set(frame["customer_id"]) == expected

    def test_contributions_are_additive_in_log_odds(self, local):
        """SHAP additivity holds in LINK space, not probability space.

        For a classifier, `TreeExplainer` works in the model's link space, so the identity is

            sum(contributions) + expected_value == logit(predict_proba)

        and NOT `== probability`. Asserting the probability form is the easiest wrong
        assertion to write against this artefact, and it fails for a reason that has nothing
        to do with the explainer: a score of 0.9995 minus a -0.627 baseline is 1.63 while the
        contributions sum to 8.26, because 8.26 is a log-odds delta.

        Only the top reasons are published, so the full feature set is recomputed here.
        """
        import shap

        frame, model, scores, x_test = local
        target = frame["customer_id"].iloc[0]
        row = int(target.replace("CUS", ""))

        explainer = shap.TreeExplainer(model.named_steps["model"])
        matrix = model.named_steps["impute"].transform(x_test)
        contributions = np.asarray(explainer.shap_values(matrix))[row]
        baseline = float(np.mean(explainer.expected_value))

        logit = np.log(scores[row] / (1 - scores[row]))
        assert contributions.sum() + baseline == pytest.approx(logit, abs=0.05)

        # Rank the ROUNDED values, because those are what the artefact publishes. Ordering on
        # the unrounded floats would disagree wherever rounding collapses two small
        # contributions onto the same 6dp value and the tie breaks differently -- which is
        # a real difference in the file, not a test artefact.
        published_values = np.round(contributions, 6)
        strongest = [FEATURE_COLUMNS[i] for i in
                     np.argsort(np.abs(published_values))[::-1][:TOP_REASONS]]
        published = frame[frame["customer_id"] == target].sort_values("feature_rank")
        assert published["feature"].tolist() == strongest, (
            "the published top reasons are not the model's strongest contributions"
        )

    def test_a_contribution_larger_than_one_is_not_a_probability_delta(self, local):
        """Pins the unit hazard.

        A naive reader sees "+2.19 raises risk" and concludes the probability rose by 2.19,
        which is impossible. Asserting that such a value exists is what keeps the docstring's
        UNITS note attached to the number.
        """
        frame, _, _, _ = local
        assert (frame["shap_contribution"].abs() > 1).any(), (
            "if every contribution is inside [-1, 1] the units are ambiguous enough that "
            "the log-odds note could be dropped by mistake"
        )

    def test_feature_values_are_the_values_the_model_scored(self, local, split):
        frame, _, _, _ = local
        _, _, x_test, _ = split
        sample = frame.sample(min(20, len(frame)), random_state=0)
        for row in sample.itertuples():
            source = float(x_test.iloc[int(row.row)][row.feature])
            assert row.feature_value == pytest.approx(source, abs=1e-6), (
                "local SHAP shows raw values while the model scored imputed ones, so the "
                "explanation would not match the prediction"
            )

    def test_degrades_to_row_numbers_without_identifiers(self, split):
        x_train, y_train, x_test, _ = split
        model = build_pipeline()
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        frame = local_explanations(model, x_test, list(x_test.columns), scores, n=3)
        # Degrading to row numbers is acceptable; silently pretending to be per-customer is
        # not. The column must exist and be empty rather than absent.
        assert "customer_id" in frame.columns
        assert frame["customer_id"].isna().all()




class TestWrittenArtefacts:
    """The committed CSVs must carry the same guarantees the in-memory frames do."""

    def test_local_csv_names_customers(self):
        path = config.PROCESSED / "churn_shap_local.csv"
        if not path.exists():
            pytest.skip("run `python -m src.churn` first")
        frame = pd.read_csv(path)
        assert "customer_id" in frame.columns
        assert frame["customer_id"].notna().all(), (
            "the committed local SHAP file has no customer identifiers, so it is per-row "
            "rather than per-customer"
        )
        assert frame["customer_id"].nunique() == config.SHAP_LOCAL_TOP_N

    def test_metrics_csv_carries_the_operating_block(self):
        path = config.PROCESSED / "churn_metrics.csv"
        if not path.exists():
            pytest.skip("run `python -m src.churn` first")
        published = pd.read_csv(path).iloc[0].to_dict()
        for column in ("auc", "pr_auc", "precision_at_top", "recall_at_top",
                       "lift_at_top", "operating_threshold", "f1_at_operating_threshold"):
            assert column in published, f"{column} missing from churn_metrics.csv"

        # The confusion cells must be published as numbers. As a string like
        # "tn=11458 fp=1002" nobody can check them against the metrics.
        for cell in ("tn", "fp", "fn", "tp"):
            assert cell in published, f"confusion cell {cell} missing"
            assert float(published[cell]) >= 0

    def test_confusion_of_matches_the_published_matrix(self):
        scores_path = config.PROCESSED / "churn_scores.csv"
        metrics_path = config.PROCESSED / "churn_metrics.csv"
        if not (scores_path.exists() and metrics_path.exists()):
            pytest.skip("run `python -m src.churn` first")
        scored = pd.read_csv(scores_path)
        published = pd.read_csv(metrics_path).iloc[0]

        # Recompute from the published scores: the matrix in the CSV has to be derivable
        # from the file next to it, or the two have silently drifted apart. This is what
        # caught the threshold being rounded to 4dp while the matrix used full precision.
        predicted = (scored["churn_score"] >= published["operating_threshold"]).astype(int)
        counts = confusion_of(scored["churn"].to_numpy(), predicted)
        for cell in ("tn", "fp", "fn", "tp"):
            assert counts[cell] == int(published[cell]), (
                f"{cell}: churn_metrics.csv says {int(published[cell])}, recomputing from "
                f"churn_scores.csv at the published threshold gives {counts[cell]}"
            )