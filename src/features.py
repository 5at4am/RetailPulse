"""Customer and demand features, computed so that nothing from the future leaks in.

    from src.features import rfm_features, churn_snapshots, panel_features

    rfm = rfm_features(sales, as_of="2025-12-31")   # one row per customer
    snaps = churn_snapshots(sales)                   # 9 monthly snapshots, labelled
    panel = panel_features(panel)                    # lags + rolling stats for F-03

**The one rule this module exists to enforce.** Every feature is computed from rows dated
strictly on or before an `as_of` date, and every returned row carries the
`feature_as_of` timestamp that produced it. Nothing here computes a "revenue over the last
90 days" by accident over the wrong window, and `tests/test_no_leakage.py` re-derives that
claim independently instead of trusting this docstring.

Leakage is the failure mode that cannot be seen in the metrics: a churn model fed
post-snapshot features scores AUC 0.99 and is worthless, because at prediction time those
features do not exist yet.

Runnable on its own: `python -m src.features`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

FEATURE_COLUMNS = [
    "recency_days", "frequency", "monetary", "avg_basket_value", "units_per_line",
    "category_diversity", "promo_share", "mean_discount_pct", "online_share",
    "tenure_days",
]

# Columns that must never appear in a *feature* set, because their value is only known
# after the snapshot date. Cheap insurance against someone `merge`-ing a label back in.
FORBIDDEN_PREFIXES = ("churn", "label", "target", "y_true", "future")

# The outcome columns `churn_snapshots` deliberately returns. A labelled training frame is
# the normal case, not a bug, so the guard is told when to expect them.
LABEL_COLUMNS = ("churn", "purchased_in_label_window",
                 "label_window_start", "label_window_end")


def _as_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        raise ValueError(f"expected a naive date, got {ts}")
    return ts


def month_end_snapshots():
    """The 9 month-end snapshot dates from the spec.

    Stops at 2025-09-30 because a 90-day forward label window has to fit before the data
    ends on 2025-12-31. A snapshot in November or December would have no future to
    observe, and those customers would be unlabelled rather than negative.
    """
    return list(pd.date_range(config.SNAPSHOT_START, config.SNAPSHOT_END,
                              freq="ME"))


def rfm_features(sales, as_of, feature_days=None, include_unconverted=True):
    """One row per customer, using only transactions dated on or before `as_of`.

    `feature_days` limits the frequency/monetary window to the N days ending at `as_of`.
    Leave it None for lifetime behaviour, which is what the F-02 segmentation wants as of
    the end of the dataset. The churn model passes 90, so a dormant customer gets zero
    window activity but a large `recency_days` -- which is exactly the signal it needs.

    `include_unconverted=False` drops customers with no purchase in the window entirely,
    which is only appropriate for segmentation, never for churn: it would delete the
    dormant customers the model is trying to find.

    Returns a frame with `customer_id`, the features in FEATURE_COLUMNS, and the audit
    columns `feature_as_of`, `first_purchase`, `last_purchase`, `lines` and `n_baskets`.
    """
    as_of = _as_timestamp(as_of)
    df = sales
    if "date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("sales['date'] must already be datetime -- run ingest first")

    # --- the leakage barrier. Everything below sees only this frame.
    visible = df.loc[df["date"] <= as_of].copy()
    if visible.empty:
        raise ValueError(f"no transactions on or before {as_of.date()}")

    lifetime_first = visible.groupby("customer_id")["date"].min()
    lifetime_last = visible.groupby("customer_id")["date"].max()

    if feature_days is not None:
        window_start = as_of - pd.Timedelta(days=feature_days - 1)
        scoped = visible.loc[visible["date"] >= window_start].copy()
    else:
        window_start = None
        scoped = visible

    line = "is_promotion"

    # Aggregate the window, then reindex onto the customer set we actually want.
    #
    # Building the index from the window groupby instead would silently delete every
    # customer with no activity in the window -- which is to say, every dormant customer,
    # which is to say every true positive the churn model is looking for. The row set is
    # therefore the *lifetime* set by default, and window columns come out NaN where there
    # was no activity.
    grouped = scoped.groupby("customer_id")
    window = pd.DataFrame(index=grouped.size().index)
    window.index.name = "customer_id"

    window["frequency"] = grouped["transaction_id"].nunique()
    window["monetary"] = grouped["sales_amount"].sum()
    window["units"] = grouped["quantity_sold"].sum()
    window["lines"] = grouped.size()
    window["category_diversity"] = grouped["product_category"].nunique()
    window["promo_share"] = (grouped[line].mean() if line in scoped.columns
                             else np.nan)
    window["mean_discount_pct"] = (grouped["discount_pct"].mean()
                                   if "discount_pct" in scoped.columns else np.nan)
    if "channel" in scoped.columns:
        online = (scoped["channel"] != "Store").astype(float)
        window["online_share"] = online.groupby(scoped["customer_id"]).mean()
    else:
        window["online_share"] = np.nan

    if include_unconverted:
        index = lifetime_first.index
    else:
        index = window.index
        lifetime_first = lifetime_first.reindex(index)
        lifetime_last = lifetime_last.reindex(index)

    out = window.reindex(index)
    out["units_per_line"] = out["units"] / out["lines"].replace(0, np.nan)
    out["avg_basket_value"] = out["monetary"] / out["frequency"].replace(0, np.nan)

    # Recency, tenure and last-purchase come from the lifetime view on purpose: a
    # customer who last bought 400 days ago has no window rows, and recency is the whole
    # story for them.
    out["recency_days"] = (as_of - lifetime_last.reindex(out.index)).dt.days
    out["tenure_days"] = (as_of - lifetime_first.reindex(out.index)).dt.days
    out["first_purchase"] = lifetime_first.reindex(out.index)
    out["last_purchase"] = lifetime_last.reindex(out.index)

    # No activity in the window reads as "none", not as "unknown". Keeping NaN here would
    # let a model treat a dormant customer as a different kind of missing from a brand-new
    # one, which is not a distinction the data supports.
    out["lines"] = out["lines"].fillna(0)
    out["frequency"] = out["frequency"].fillna(0)

    out["feature_as_of"] = as_of
    out["feature_window_start"] = (window_start if window_start is not None
                                   else out["first_purchase"])
    out["n_baskets"] = out["frequency"]
    out = out.drop(columns=["units"])

    ordered = (["customer_id"] + FEATURE_COLUMNS
               + ["n_baskets", "lines", "first_purchase", "last_purchase",
                  "feature_window_start", "feature_as_of"])
    return out.reset_index()[ordered].sort_values("customer_id").reset_index(drop=True)


def churn_snapshots(sales, snapshot_dates=None, feature_days=90,
                    label_days=90, progress=None):
    """Labelled churn snapshots, one block of rows per month-end.

    For each snapshot date S:
      - features use transactions in the `feature_days` window ending on S
      - `churn = 1` if the customer bought nothing in the `label_days` after S

    Only customers who had already purchased by S appear, i.e. customers who had not yet
    joined are excluded. Customers who joined earlier but stopped buying are **kept** --
    they are the ones the model has to find.

    A snapshot whose label window runs past the last transaction date is dropped rather
    than labelled 0, because "did not churn" is not observable when the future is missing.
    """
    if snapshot_dates is None:
        snapshot_dates = month_end_snapshots()
    snapshot_dates = [_as_timestamp(d) for d in snapshot_dates]

    last_date = sales["date"].max()
    blocks = []
    dropped = []

    for i, snapshot in enumerate(snapshot_dates):
        label_end = snapshot + pd.Timedelta(days=label_days)
        if label_end > last_date:
            dropped.append({"snapshot_date": snapshot.date().isoformat(),
                            "reason": f"label window ends {label_end.date()} "
                                      f"but data ends {last_date.date()}"})
            continue

        feats = rfm_features(sales, as_of=snapshot, feature_days=feature_days,
                             include_unconverted=True)

        future = sales.loc[(sales["date"] > snapshot) & (sales["date"] <= label_end)]
        buyers = future["customer_id"].unique()

        block = feats.copy()
        block["snapshot_date"] = snapshot
        block["label_window_start"] = snapshot
        block["label_window_end"] = label_end
        # purchased_in_window -> churn. A customer with no future row churns.
        block["purchased_in_label_window"] = block["customer_id"].isin(buyers)
        block["churn"] = (~block["purchased_in_label_window"]).astype("int8")
        blocks.append(block)

        if progress:
            progress(f"  snapshot {i + 1}/{len(snapshot_dates)} "
                     f"{snapshot.date()}: {len(block):,} customers, "
                     f"{block['churn'].mean():.1%} churn")

    if not blocks:
        raise ValueError("every snapshot's label window runs past the end of the data")

    out = pd.concat(blocks, ignore_index=True)
    out.attrs["dropped_snapshots"] = dropped
    return out


def panel_features(panel, lags=(1, 2, 4), roll=4):
    """Lag and rolling features for the weekly forecasting panel.

    **Every feature is shifted so it cannot see its own row.** `lag_1` is last week's
    units, not this week's; the rolling means exclude the current week. Without the shift
    a model would be handed the answer and would score beautifully and uselessly.

    Zero-demand weeks are kept -- they are real observations, not gaps. Interpolating
    them would fabricate demand that never happened.
    """
    required = {"store_id", "product_id", "week_start_date", "units_sold"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    df = panel.sort_values(["store_id", "product_id", "week_start_date"]).copy()
    keys = df.groupby(["store_id", "product_id"], sort=False)["units_sold"]

    for lag in lags:
        df[f"lag_{lag}"] = keys.shift(lag)
    df[f"roll_mean_{roll}"] = keys.transform(
        lambda s: s.shift(1).rolling(roll, min_periods=1).mean())
    df[f"roll_std_{roll}"] = keys.transform(
        lambda s: s.shift(1).rolling(roll, min_periods=2).std())
    df[f"roll_max_{roll}"] = keys.transform(
        lambda s: s.shift(1).rolling(roll, min_periods=1).max())

    df["week_index"] = df.groupby(["store_id", "product_id"],
                                  sort=False).cumcount()
    df["is_zero_demand"] = (df["units_sold"] == 0).astype("int8")

    # A stockout is units_sold == 0 with stockout_count > 0, not merely inventory_level 0.
    if "stockout_count" in df.columns:
        df["had_stockout"] = (df["stockout_count"] > 0).astype("int8")

    df["feature_as_of"] = df["week_start_date"] - pd.Timedelta(days=7)
    return df


def assert_no_leakage(frame, snapshot_col="snapshot_date", allow_labels=False):
    """Cheap in-function guard: no feature row may post-date its own snapshot.

    Two independent checks, because they catch different mistakes:

    1. **Timestamps.** `feature_as_of <= snapshot_date` for every row. This is the check
       that matters -- it verifies provenance rather than trusting column naming.
    2. **Column names.** No column whose name looks like an outcome. A label merged into
       a feature frame would train beautifully and mean nothing.

    Pass `allow_labels=True` for a labelled training frame from `churn_snapshots`, which
    carries its own `churn` column on purpose.

    `tests/test_no_leakage.py` re-derives the timestamp claim independently against the
    raw transactions, rather than trusting this function.
    """
    if snapshot_col not in frame.columns:
        raise ValueError(f"{snapshot_col} missing -- cannot verify leakage")
    if "feature_as_of" not in frame.columns:
        raise ValueError("feature_as_of missing -- features must carry their own provenance")

    as_of = pd.to_datetime(frame["feature_as_of"])
    snapshot = pd.to_datetime(frame[snapshot_col])
    leaking = int((as_of > snapshot).sum())
    if leaking:
        raise ValueError(f"{leaking:,} row(s) have features dated after their snapshot")

    expected = set(LABEL_COLUMNS) if allow_labels else set()
    suspicious = [c for c in frame.columns
                  if c.lower().startswith(FORBIDDEN_PREFIXES) and c not in expected]
    if suspicious:
        raise ValueError(f"post-outcome columns present in a feature set: {suspicious}")
    return True


def feature_matrix(snapshots):
    """The feature columns of a labelled snapshot frame, with keys and provenance kept.

    This is what actually goes into the model. Calling it makes the separation explicit
    rather than relying on every downstream `drop(columns=[...])` staying in sync.
    """
    keep = (["customer_id", "snapshot_date", "feature_as_of", "feature_window_start",
             "first_purchase", "last_purchase"] + FEATURE_COLUMNS)
    present = [c for c in keep if c in snapshots.columns]
    return snapshots[present].copy()


def model_matrix(frame, columns=None):
    """The numeric feature columns only -- what actually goes into an estimator.

    `feature_matrix` deliberately keeps `customer_id`, the dates and the provenance columns
    because those are needed to join results back and to audit leakage. An estimator cannot
    accept them, so the split is made explicit here instead of each caller dropping columns
    by hand and getting the list wrong.

    Raises on a missing or non-numeric column rather than imputing around it: a string
    column reaching this point means the column list is stale.
    """
    columns = list(columns or FEATURE_COLUMNS)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing feature columns: {missing}")

    out = frame[columns]
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(out[c])]
    if non_numeric:
        raise ValueError(f"feature columns are not numeric: {non_numeric}")

    return out.copy()


def main():
    from src.ingest import load_sales, load_panel

    sales = load_sales()
    print("=" * 72)
    print("FEATURES -- RFM as of 2025-12-31")
    print("=" * 72)
    rfm = rfm_features(sales, as_of=config.SALES_END)
    print(f"customers           {len(rfm):,}")
    print(f"max feature_as_of   {rfm['feature_as_of'].max().date()}")
    print(f"last purchase range {rfm['last_purchase'].min().date()} .. "
          f"{rfm['last_purchase'].max().date()}")
    print("\nfeature summary")
    print(rfm[FEATURE_COLUMNS].describe().T.to_string())

    print("\n" + "=" * 72)
    print("FEATURES -- labelled churn snapshots")
    print("=" * 72)
    snaps = churn_snapshots(sales, progress=print)
    for d in snaps.attrs.get("dropped_snapshots", []):
        print(f"  dropped {d['snapshot_date']}: {d['reason']}")
    print(f"\nrows                {len(snaps):,}")
    print(f"snapshots           {snaps['snapshot_date'].nunique()}")
    print(f"overall churn rate  {snaps['churn'].mean():.1%}")
    print(snaps.groupby("snapshot_date")["churn"].agg(["size", "mean"]).to_string())

    assert_no_leakage(snaps, allow_labels=True)
    print("\nno-leakage guard passed: every feature_as_of <= its snapshot_date")
    print(f"feature matrix for the model: {len(FEATURE_COLUMNS)} columns "
          f"({', '.join(FEATURE_COLUMNS)})")

    print("\n" + "=" * 72)
    print("FEATURES -- panel lags")
    print("=" * 72)
    pf = panel_features(load_panel())
    print(f"rows                {len(pf):,}")
    print(f"lag columns         {[c for c in pf.columns if c.startswith('lag_')]}")
    print(f"rows with a lag_1   {int(pf['lag_1'].notna().sum()):,} "
          f"(first week of each of the 7,500 series has none, by construction)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())