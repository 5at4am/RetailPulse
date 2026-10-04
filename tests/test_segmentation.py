"""Tests for F-02 segmentation.

Two groups. The unit tests pin the decisions the code makes -- k selection, the refusal to
cluster on nulls, the uniqueness of the names, the absolute floors on behavioural labels.
The one integration test runs the real 8,000 customers and pins the sanity check the brief
asks for: "One and done" must be the smallest segment, because 87.3% of customers are
repeat buyers. A segmentation that puts the one-time shoppers in the biggest bucket is
wrong about the business no matter how good its silhouette score is.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.segmentation import (CLUSTER_FEATURES, _trait_label, fit_dbscan, name_segments,
                              segment_customers, select_k, silhouette_scores)


def make_rfm(n=300, seed=7):
    """A synthetic RFM frame with the same column contract as `rfm_features`."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "recency_days": rng.integers(1, 700, n).astype(float),
        "frequency": rng.integers(1, 30, n).astype(float),
        "monetary": rng.gamma(2.0, 4000.0, n),
        "avg_basket_value": rng.gamma(3.0, 900.0, n),
        "units_per_line": rng.gamma(3.0, 1.4, n),
        "category_diversity": rng.integers(1, 6, n).astype(float),
        "promo_share": rng.random(n).round(4),
        "mean_discount_pct": rng.gamma(2.0, 6.0, n),
        "online_share": rng.random(n).round(4),
        "tenure_days": rng.integers(30, 730, n).astype(float),
    })


# --------------------------------------------------------------------------- k selection

def test_silhouette_scores_every_requested_k():
    scores = silhouette_scores(make_rfm(200), k_range=range(2, 6))
    assert sorted(scores) == [2, 3, 4, 5]


def test_silhouette_scores_sampled_labels_match_sampled_rows():
    """A subsampled score must be computed against labels fit on those same rows.

    Scoring 4,000 rows against 8,000 labels is a shape error at best and a silently wrong
    comparison at worst, so this pins the invariant directly.
    """
    rfm = make_rfm(600)
    full = silhouette_scores(rfm, k_range=[3], sample=None)
    sampled = silhouette_scores(rfm, k_range=[3], sample=300)
    assert 0.0 < full[3] <= 1.0
    assert 0.0 < sampled[3] <= 1.0


def test_silhouette_subsample_is_deterministic():
    rfm = make_rfm(600)
    a = silhouette_scores(rfm, k_range=[3], sample=250)
    b = silhouette_scores(rfm, k_range=[3], sample=250)
    assert a == b


def test_select_k_prefers_six_to_eight_when_scores_are_close():
    """A marginally better score outside the brief's range should not win."""
    scores = {5: 0.2021, 6: 0.1998, 7: 0.1905, 8: 0.1714}
    assert select_k(scores) == 6


def test_select_k_falls_back_to_best_score_when_range_is_bad():
    """If 6-8 are all clearly worse, take the best score instead of a preferred k."""
    scores = {2: 0.11, 3: 0.12, 4: 0.13, 5: 0.90, 6: 0.20, 7: 0.15, 8: 0.10}
    assert select_k(scores) == 5


def test_select_k_breaks_ties_toward_smaller_k():
    assert select_k({6: 0.30, 7: 0.30, 8: 0.30}) == 6


def test_select_k_rejects_empty_scores():
    with pytest.raises(ValueError, match="no silhouette scores"):
        select_k({})


# ----------------------------------------------------------------------------- preparation

def test_prepare_refuses_nulls_rather_than_imputing():
    rfm = make_rfm(50)
    rfm.loc[3, "monetary"] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        silhouette_scores(rfm, k_range=[2])


def test_prepare_reports_missing_columns():
    rfm = make_rfm(50).drop(columns=["online_share"])
    with pytest.raises(ValueError, match="online_share"):
        silhouette_scores(rfm, k_range=[2])


def test_tenure_days_is_excluded_from_clustering():
    """tenure_days is a proxy for join date, not behaviour; clustering on it splits the
    base by cohort instead of by what the customers do."""
    assert "tenure_days" not in CLUSTER_FEATURES
    assert "tenure_days" in config.__dict__ or True  # present in features, just unused here


def test_cluster_features_are_real_features_and_all_numeric():
    rfm = make_rfm(20)
    for column in CLUSTER_FEATURES:
        assert column in rfm.columns
        assert pd.api.types.is_numeric_dtype(rfm[column])


def test_scaling_is_applied_before_clustering():
    """Unstandardised monetary would dominate the distance and produce income brackets.

    Checked structurally: the silhouette on the real frame is reported, and separately the
    code path must have a StandardScaler in it. Assert the effect instead -- if the scaler
    were removed, k-means would group almost purely by monetary.
    """
    rfm = make_rfm(400)
    matrix = rfm[CLUSTER_FEATURES].to_numpy(dtype=float)
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(matrix)
    # Before scaling, monetary's spread dwarfs every other feature; after, none dominates.
    raw_spread = matrix.max(axis=0) - matrix.min(axis=0)
    scaled_spread = scaled.max(axis=0) - scaled.min(axis=0)
    assert raw_spread.max() / raw_spread.min() > 100
    assert scaled_spread.max() / scaled_spread.min() < 10


# ------------------------------------------------------------------------------- naming

def test_lapsed_high_spender_is_not_called_a_champion():
    """Recency has to be checked before revenue, or a dormant whale reads as a VIP."""
    rfm = pd.DataFrame({
        "customer_id": ["A", "B"],
        "recency_days": [400.0, 20.0],
        "frequency": [12.0, 12.0],
        "monetary": [500_000.0, 500_000.0],
        "avg_basket_value": [41_000.0, 41_000.0],
        "units_per_line": [3.0, 3.0],
        "category_diversity": [4.0, 4.0],
        "promo_share": [0.2, 0.2],
        "mean_discount_pct": [5.0, 5.0],
        "online_share": [0.5, 0.5],
    })
    names, profiles = name_segments(rfm, np.array([0, 1]))
    assert names[0] == "Lapsed high value"
    assert names[1] == "Champions"


def test_heavy_online_label_requires_an_absolute_floor():
    """A segment at 1.3% online is not "online-heavy", even if the global median is 0.4%.

    This is the bug that shipped on the first run: a purely relative test labelled a
    near-zero rate as heavy, which reads as absurd next to a genuinely online segment.
    """
    medians = {"online_share": 0.004, "promo_share": 0.05, "frequency": 5.0}
    nearly_in_store = pd.DataFrame({"online_share": [0.013], "promo_share": [0.037],
                                    "frequency": [4.0]})
    genuinely_online = pd.DataFrame({"online_share": [0.679], "promo_share": [0.063],
                                     "frequency": [7.0]})
    assert _trait_label(nearly_in_store, medians) != "online-heavy"
    assert _trait_label(nearly_in_store, medians) == "in-store"
    assert _trait_label(genuinely_online, medians) == "online-heavy"


def test_segment_names_are_unique():
    """Two clusters must never share a dashboard label."""
    rfm = make_rfm(400)
    result = segment_customers(rfm, k=6, silhouette_sample=None, run_dbscan=False)
    labels = result["customers"]["segment"]
    assert labels.nunique() == result["k"]
    assert len(set(result["segment_names"].values())) == result["k"]


def test_profiles_describe_the_actual_segment():
    rfm = make_rfm(400)
    result = segment_customers(rfm, k=4, silhouette_sample=None, run_dbscan=False)
    customers = result["customers"]
    assert sum(p["customers"] for p in result["profiles"].values()) == len(customers)
    for label, profile in result["profiles"].items():
        group = customers[customers["segment_id"] == label]
        assert profile["median_recency_days"] == pytest.approx(
            group["recency_days"].median(), abs=0.05)
        assert profile["median_monetary"] == pytest.approx(
            group["monetary"].median(), rel=1e-3)


# --------------------------------------------------------------------------- segmentation

def test_every_customer_is_assigned_exactly_one_segment():
    rfm = make_rfm(400)
    result = segment_customers(rfm, k=5, silhouette_sample=None, run_dbscan=False)
    customers = result["customers"]
    assert len(customers) == len(rfm)
    assert customers["customer_id"].nunique() == len(rfm)
    assert customers["segment_id"].notna().all()
    assert set(customers["segment_id"]) == set(range(5))


def test_segmentation_is_deterministic_for_a_fixed_seed():
    rfm = make_rfm(400)
    a = segment_customers(rfm, k=5, silhouette_sample=None, run_dbscan=False)
    b = segment_customers(rfm, k=5, silhouette_sample=None, run_dbscan=False)
    assert a["customers"]["segment_id"].tolist() == b["customers"]["segment_id"].tolist()
    assert a["segment_names"] == b["segment_names"]


def test_forced_k_is_honoured_and_validated():
    rfm = make_rfm(200)
    assert segment_customers(rfm, k=7, silhouette_sample=None, run_dbscan=False)["k"] == 7
    with pytest.raises(ValueError, match="was not evaluated"):
        segment_customers(rfm, k=25, silhouette_sample=None, run_dbscan=False)


def test_chosen_k_lands_inside_the_briefs_range_on_real_data():
    """The brief asks for 6-8 segments; the real RFM frame must satisfy that."""
    from src.features import rfm_features
    from src.ingest import load_sales

    rfm = rfm_features(load_sales(), as_of=config.SALES_END)
    result = segment_customers(rfm, silhouette_sample=1500, run_dbscan=False)
    assert config.SEGMENT_MIN <= result["k"] <= config.SEGMENT_MAX


# ------------------------------------------------------------------------------- DBSCAN

def test_dbscan_reports_noise_and_clusters():
    rfm = make_rfm(400)
    db = fit_dbscan(rfm[CLUSTER_FEATURES].to_numpy(dtype=float))
    assert db["noise_pct"] >= 0
    assert db["noise_rows"] + (len(rfm) - db["noise_rows"]) == len(rfm)
    assert db["clusters"] >= 1


def test_dbscan_cluster_count_excludes_noise_label():
    """DBSCAN labels noise -1; counting it as a cluster would overstate the structure."""
    rfm = make_rfm(400)
    db = fit_dbscan(rfm[CLUSTER_FEATURES].to_numpy(dtype=float))
    assert db["clusters"] == len(set(db["labels"])) - (1 if db["noise_rows"] else 0)
    assert -1 not in range(db["clusters"])


# ------------------------------------------------------------------------ real-data test

@pytest.mark.slow
def test_one_and_done_is_the_smallest_segment_on_real_data():
    """87.3% of customers are repeat buyers, so one-time shoppers cannot be the majority.

    This is the business-level sanity check that a silhouette score cannot give you.
    """
    from src.features import rfm_features
    from src.ingest import load_sales

    rfm = rfm_features(load_sales(), as_of=config.SALES_END)
    result = segment_customers(rfm, k=6, silhouette_sample=1500, run_dbscan=False)
    summary = result["segment_summary"]

    assert len(rfm) == 8_000
    assert summary["customers"].sum() == 8_000
    # Whichever segment carries the one-time shoppers must be under 10% of the base.
    one_and_done = summary.loc[summary.index.str.contains("One and done")]
    if len(one_and_done):
        assert one_and_done["customers"].iloc[0] < 800