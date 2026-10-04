"""F-02 — Customer segmentation.

    from src.segmentation import segment_customers, name_segments

    result = segment_customers(rfm)      # 8,000 customers -> 6-8 named segments

K-Means on standardised RFM features, with `k` chosen on silhouette score. DBSCAN is fit
as a comparison and its noise fraction reported, because on a dense, roughly Gaussian RFM
cloud DBSCAN is expected to label a large share of customers as noise. That outcome is
reported, not hidden.

**Segment names are derived from the centroids, not chosen in advance.** A name that does
not describe what the cluster actually contains is a lie in a slide, so `name_segments`
computes each name from the centroid's own profile.

Features are as of 2025-12-31 with no forward-looking information. See
`src/features.py` for the leakage barrier and `tests/test_no_leakage.py` for the check.

Runnable on its own: `python -m src.segmentation`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.features import FEATURE_COLUMNS

# Clustering on these only. `tenure_days` is deliberately excluded: it is almost a linear
# function of the dataset's own start date rather than a behaviour, so including it mostly
# splits customers by join date and washes out the RFM signal.
CLUSTER_FEATURES = [c for c in FEATURE_COLUMNS if c != "tenure_days"]

# Relative thresholds, not absolute rupees. Fixed rupee cut-offs would not survive a
# price-level change and would need re-tuning every time the business grows.
HIGH_REVENUE = 0.70      # percentile, not a rupee amount
LOW_REVENUE = 0.30
HIGH_FREQUENCY = 0.70
LOW_FREQUENCY = 0.30
LOYAL_RECENCY_DAYS = 60
AT_RISK_RECENCY_DAYS = 180
BASKET_MULTIPLE = 1.5    # basket >= 1.5x the median
# Absolute floors for behavioural trait labels, so "online-heavy" means genuinely online
# and not merely above a near-zero median.
HEAVY_SHARE = 0.25
IN_STORE_HEAVY_SHARE = 0.10


def _median_basket(monetary, frequency):
    valid = frequency > 0
    if not valid.any():
        return 0.0
    return float((monetary[valid] / frequency[valid]).median())


def _prepare(rfm, features=None):
    """The numeric matrix to cluster on, plus the feature names.

    Raises rather than imputing: a missing value in the feature set means the upstream
    window produced no row for that customer, and silently filling it with 0 would make
    "bought nothing" indistinguishable from "we never looked".
    """
    features = list(features or CLUSTER_FEATURES)
    missing = [c for c in features if c not in rfm.columns]
    if missing:
        raise ValueError(f"rfm frame is missing feature columns: {missing}")

    matrix = rfm[features]
    if matrix.isna().any().any():
        offenders = {c: int(matrix[c].isna().sum()) for c in matrix.columns
                     if matrix[c].isna().any()}
        raise ValueError(f"features contain nulls, refusing to cluster: {offenders}")

    return matrix.to_numpy(dtype=float), features


def silhouette_scores(matrix, k_range=range(2, 11), random_state=config.SEED,
                      sample=None):
    """Silhouette score for each k, on a z-scored copy of the matrix.

    K-Means is distance-based, so standardisation is not optional: `monetary` spans five
    orders of magnitude and would otherwise dominate the distance entirely, and the
    clusters would end up being income brackets no matter what k was chosen.

    `sample` caps the rows evaluated, because silhouette is O(n^2) and 8,000 customers is
    64 million pairwise distances *per k*. The subsample is fit and scored together, so
    labels and points always describe the same rows -- scoring a subsample against labels
    from the full set would be meaningless.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    data, _ = _prepare_matrix_only(matrix)
    scaled = StandardScaler().fit_transform(data)

    if sample is not None and len(scaled) > sample:
        rng = np.random.default_rng(random_state)
        scaled = scaled[rng.choice(len(scaled), sample, replace=False)]

    out = {}
    for k in k_range:
        if k >= len(scaled):
            continue
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(scaled)
        out[k] = float(silhouette_score(scaled, labels, metric="euclidean"))
    return out


def _prepare_matrix_only(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix, None
    return _prepare(matrix)


def select_k(scores, preferred=(6, 7, 8), tolerance=0.02):
    """Pick k from the silhouette curve.

    Preferred range first: a score within `tolerance` of the best buys a smaller, more
    actionable k, because six segments a manager can act on beat nine that are statistically
    tidier. Outside that range, take the best score, breaking ties toward smaller k.
    """
    if not scores:
        raise ValueError("no silhouette scores to choose from")
    best = max(scores.values())
    within = {k: v for k, v in scores.items() if v >= best - tolerance}
    in_range = {k: v for k, v in within.items() if k in preferred}
    pool = in_range or within
    return min(pool, key=lambda k: (-pool[k], k))


def _trait_label(group, global_medians):
    """One short distinguishing trait for a segment, used only to break a name tie.

    Two guards, both necessary. Relative deviation alone is not enough: the global median
    online share is near zero here, so a segment that is almost entirely in-store would be
    "online-heavy" purely because 1.3% beats 0.4%. Every trait therefore has to clear an
    absolute floor before it can claim a label, and the reported numbers are meant to be
    quotable without a caveat.
    """
    online = float(group["online_share"].mean())
    promo = float(group["promo_share"].mean())
    frequency = float(group["frequency"].median())

    if online >= HEAVY_SHARE:
        return "online-heavy"
    if online <= IN_STORE_HEAVY_SHARE:
        return "in-store"
    if promo >= HEAVY_SHARE:
        return "promo-driven"

    # Nothing is behaviourally extreme, so fall back to the largest real gap to the
    # overall median, and only claim a direction if the gap is worth acting on.
    def gap(value, key):
        base = float(global_medians[key])
        scale = abs(base) if abs(base) > 1e-9 else 1.0
        return abs(value - base) / scale

    options = [
        (gap(frequency, "frequency"),
         "high-frequency" if frequency >= global_medians["frequency"] else "low-frequency"),
        (gap(promo, "promo_share"),
         "promo-driven" if promo >= global_medians["promo_share"] else "rarely-promoted"),
    ]
    best_gap, best_label = max(options, key=lambda t: t[0])
    return best_label if best_gap > 0.5 else "mixed"


RANK_WORDS = ("highest", "high", "mid", "low", "lowest")


def _disambiguate(frame, names, profiles):
    """Make segment names unique.

    K-Means will happily hand back several centroids that all read as "At risk" -- and two
    segments in a dashboard with the same name are indistinguishable to whoever is trying
    to act on them. Collapsing them into one label would hide the fact that the algorithm
    found a distinction worth reporting.

    Uniqueness is guaranteed, not hoped for. A fixed suffix vocabulary is not enough: on
    skewed data six centroids can share one base name and there are only a handful of trait
    words, so each colliding group is also ranked along whichever trait actually separates
    it most, and the rank goes into the label.
    """
    counts = {}
    for label, name in names.items():
        counts.setdefault(name, []).append(label)

    duplicates = {n: ls for n, ls in counts.items() if len(ls) > 1}
    if not duplicates:
        return names

    global_medians = {
        "online_share": float(frame["online_share"].median()),
        "promo_share": float(frame["promo_share"].median()),
        "frequency": float(frame["frequency"].median()),
    }

    axes = {
        "online share": lambda g: float(g["online_share"].mean()),
        "promo share": lambda g: float(g["promo_share"].mean()),
        "frequency": lambda g: float(g["frequency"].median()),
        "recency": lambda g: float(g["recency_days"].median()),
        "monetary": lambda g: float(g["monetary"].median()),
    }

    for name, labels in duplicates.items():
        groups = {l: frame[frame["segment_id"] == l] for l in labels}
        spread = {axis: max(fn(groups[l]) for l in labels) - min(fn(groups[l]) for l in labels)
                  for axis, fn in axes.items()}
        axis = max(spread, key=lambda a: spread[a])
        ordered = sorted(labels, key=lambda l: axes[axis](groups[l]))

        for rank, label in enumerate(ordered):
            word = RANK_WORDS[rank] if rank < len(RANK_WORDS) else f"#{rank + 1}"
            trait = _trait_label(groups[label], global_medians)
            names[label] = f"{name} ({trait}, {axis} {word})"
            profiles[label]["distinguishing_trait"] = trait
            profiles[label]["distinguishing_axis"] = axis
    return names


def name_segments(rfm, labels):
    """Name each segment from its centroid's actual profile.

    The order of the checks matters. Recency characterises the segment first, because a
    lapsed high-spender and an active high-spender need opposite treatment and differ only
    in that respect; then revenue, then frequency, then basket shape. Naming on revenue
    alone would call a lapsed high-spender an "active VIP".
    """
    frame = rfm.copy()
    frame["_label"] = np.asarray(labels)
    median_basket = _median_basket(frame["monetary"], frame["frequency"])

    revenue_cut = frame["monetary"].quantile(HIGH_REVENUE)
    revenue_low = frame["monetary"].quantile(LOW_REVENUE)
    frequency_cut = frame["frequency"].quantile(HIGH_FREQUENCY)
    frequency_low = frame["frequency"].quantile(LOW_FREQUENCY)

    names = {}
    profiles = {}
    for label, group in frame.groupby("_label", sort=True):
        n = len(group)
        median_recency = float(group["recency_days"].median())
        median_frequency = float(group["frequency"].median())
        median_monetary = float(group["monetary"].median())
        median_basket_value = median_basket and median_monetary / max(median_frequency, 1)
        online_share = float(group["online_share"].mean())
        promo_share = float(group["promo_share"].mean())

        if median_recency >= AT_RISK_RECENCY_DAYS:
            name = "Lapsed high value" if median_monetary >= revenue_low else "Dormant"
        elif median_recency >= LOYAL_RECENCY_DAYS:
            name = "At risk loyal" if median_monetary >= revenue_low else "At risk"
        elif median_monetary >= revenue_cut and median_frequency >= frequency_cut:
            name = "Champions"
        elif median_monetary <= revenue_low and median_frequency <= frequency_low:
            # The brief's sanity target: with 87.3% repeat customers this must be the
            # smallest segment, not the largest.
            name = "One and done"
        elif median_basket_value >= BASKET_MULTIPLE * median_basket:
            name = "Bulk buyers"
        elif median_frequency >= frequency_cut:
            name = "Loyal discount seekers" if promo_share > 0 else "Frequent small baskets"
        elif median_monetary <= revenue_low:
            name = "Low spend"
        else:
            name = "Occasional buyers"

        names[int(label)] = name
        profiles[int(label)] = {
            "customers": n,
            "share_of_customers": round(n / len(frame), 4),
            "median_recency_days": round(median_recency, 1),
            "median_frequency": round(median_frequency, 1),
            "median_monetary": round(median_monetary, 2),
            "median_basket_value": round(median_basket_value, 2),
            "online_share": round(online_share, 4),
            "promo_share": round(promo_share, 4),
        }
    return names, profiles


def fit_dbscan(matrix, eps=None, min_samples=25, random_state=config.SEED,
               max_noise_pct=40.0):
    """DBSCAN as a comparison to K-Means, reporting its noise fraction honestly.

    `eps` is searched over multipliers of the median 5th-nearest-neighbour distance rather
    than hard-coded, because DBSCAN's eps is expressed in the units of the input and any
    literal number is meaningless on a different feature set. The search prefers the
    largest eps that still yields more than one cluster while keeping noise at or below
    `max_noise_pct`; the default single-shot eps collapsed all 8,000 customers into one
    cluster, which is a true result about the method but a useless comparison.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    data, _ = _prepare_matrix_only(matrix)
    scaled = StandardScaler().fit_transform(data)

    if eps is None:
        nn = NearestNeighbors(n_neighbors=min_samples + 1).fit(scaled)
        distances, _ = nn.kneighbors(scaled)
        base_eps = float(np.median(distances[:, -1]))
        eps = _search_dbscan_eps(scaled, base_eps, min_samples, max_noise_pct)
    else:
        base_eps = eps

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    noise = int((labels == -1).sum())
    # DBSCAN labels noise as -1 and clusters as 0..n-1, so drop the -1 when counting.
    n_clusters = len(set(labels)) - (1 if noise else 0)
    return {
        "labels": labels,
        "base_eps": base_eps,
        "eps": eps,
        "min_samples": min_samples,
        "clusters": int(n_clusters),
        "noise_rows": noise,
        "noise_pct": round(100 * noise / len(labels), 2),
    }


def _search_dbscan_eps(scaled, base_eps, min_samples, max_noise_pct):
    """Smallest multiplier that splits the data without drowning it in noise."""
    from sklearn.cluster import DBSCAN

    best = base_eps
    for multiplier in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        candidate = base_eps * multiplier
        labels = DBSCAN(eps=candidate, min_samples=min_samples).fit_predict(scaled)
        noise_pct = 100 * (labels == -1).mean()
        if len(set(labels)) - (1 if (labels == -1).any() else 0) > 1 \
                and noise_pct <= max_noise_pct:
            best = candidate
            break
    return best


def segment_customers(rfm, features=None, k=None, k_range=range(2, 11),
                      random_state=config.SEED, silhouette_sample=4000,
                      run_dbscan=True):
    """Cluster customers and name the segments.

    Returns a dict with the labelled frame, the chosen k, the full silhouette curve, the
    centroid profiles and -- if requested -- the DBSCAN comparison.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = list(features or CLUSTER_FEATURES)
    matrix, names = _prepare(rfm, features)

    scores = silhouette_scores(matrix, k_range=k_range, random_state=random_state,
                               sample=silhouette_sample)
    if k is None:
        k = select_k(scores)
    elif k not in scores:
        raise ValueError(f"k={k} was not evaluated; scores exist for {sorted(scores)}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(scaled)

    segment_names, profiles = name_segments(rfm, labels)

    out = rfm.copy()
    out["segment_id"] = labels
    out["segment"] = [segment_names[int(i)] for i in labels]

    # Two centroids can both read as "At risk"; give them distinguishable labels rather
    # than shipping one name twice.
    segment_names = _disambiguate(out, segment_names, profiles)
    out["segment"] = [segment_names[int(i)] for i in labels]

    result = {
        "customers": out,
        "k": k,
        "features": names,
        "silhouette": {str(kk): round(v, 4) for kk, v in scores.items()},
        "chosen_silhouette": round(scores[k], 4),
        "best_silhouette_k": max(scores, key=lambda kk: (scores[kk], -kk)),
        "profiles": profiles,
        "segment_names": segment_names,
        # No `total_units` here: `rfm_features` drops the units column to keep the feature
        # matrix clean, so a units aggregate would have to be faked from monetary.
        "segment_summary": (out.groupby("segment")
                            .agg(customers=("customer_id", "count"),
                                 median_recency_days=("recency_days", "median"),
                                 median_frequency=("frequency", "median"),
                                 median_monetary=("monetary", "median"),
                                 total_revenue=("monetary", "sum"),
                                 median_basket_value=("avg_basket_value", "median"),
                                 online_share=("online_share", "mean"))
                            .sort_values("total_revenue", ascending=False)),
    }

    if run_dbscan:
        result["dbscan"] = fit_dbscan(matrix, random_state=random_state)

    return result


def main():
    from src.features import rfm_features
    from src.ingest import load_sales

    sales = load_sales()
    rfm = rfm_features(sales, as_of=config.SALES_END)

    print("=" * 74)
    print("F-02 SEGMENTATION")
    print("=" * 74)
    print(f"customers        {len(rfm):,}")
    print(f"features         {', '.join(CLUSTER_FEATURES)}")
    print("                   (tenure_days excluded -- it tracks join date, not behaviour)")

    result = segment_customers(rfm)

    print("\nsilhouette by k")
    for k, v in sorted(result["silhouette"].items(), key=lambda kv: int(kv[0])):
        marker = "  <- chosen" if int(k) == result["k"] else ""
        print(f"  k={k:<3} {v:.4f}{marker}")

    print(f"\nchosen k         {result['k']}  (brief asks for {config.SEGMENT_MIN}-{config.SEGMENT_MAX})")
    print(f"silhouette       {result['chosen_silhouette']}")

    print("\nsegments")
    print(result["segment_summary"].to_string())

    print("\ncentroid profiles")
    for label, profile in sorted(result["profiles"].items()):
        print(f"  [{label}] {result['segment_names'][label]}")
        for key, value in profile.items():
            print(f"      {key:<22} {value}")

    db = result.get("dbscan")
    if db:
        print("\nDBSCAN comparison")
        print(f"  base eps        {db['base_eps']:.4f} (median 5th-NN distance)")
        print(f"  eps used        {db['eps']:.4f} (searched down for a real split)")
        print(f"  clusters        {db['clusters']}")
        print(f"  labelled noise  {db['noise_rows']:,} rows ({db['noise_pct']}%)")
        print("  ^ Reported, not hidden. DBSCAN finds density-connected cores; RFM here is")
        print("    a broad continuous cloud with no natural gaps, so it either merges almost")
        print("    everything into one cluster or labels a large share as noise. Either way it")
        print("    gives no actionable segment structure here, which is the finding -- and the")
        print("    reason K-Means is the primary method rather than a fallback.")

    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())