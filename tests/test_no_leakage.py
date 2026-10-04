"""The most important test in this project.

Churn models fail quietly. Leak a post-snapshot feature in and AUC jumps to 0.99, every
number in the report looks excellent, and the model is worthless -- because at prediction
time the feature does not exist yet. Nothing in the metrics reveals the problem.

So these tests do not trust `src/features.py`'s own bookkeeping. They go back to the raw
transactions and re-derive, independently, whether each feature could have been computed
from information available on the snapshot date.

The fixture is a small hand-built frame with a known ground truth, so the expected values
are arithmetic rather than whatever the code happened to produce.
"""
import datetime as dt

import pandas as pd
import pytest

from src import config
from src.features import (
    FEATURE_COLUMNS, FORBIDDEN_PREFIXES, assert_no_leakage, churn_snapshots,
    feature_matrix, model_matrix, month_end_snapshots, panel_features, rfm_features,
)
from src.ingest import PANEL_COLUMNS, SALES_COLUMNS

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _price_band(price):
    return ("Budget" if price < 299 else "Mid" if price < 1000
            else "Premium" if price < 5000 else "Luxury")


def _row(date, cust, txn, qty=2, price=500.0, disc=0.0, promo=None, channel="Store",
         category="Grocery", inventory=40):
    return {
        "date": pd.Timestamp(date), "customer_id": cust, "store_id": "ST01",
        "product_id": f"P{category[:2].upper()}", "product_category": category,
        "quantity_sold": qty, "unit_price": price, "discount": disc, "promotion": promo,
        "inventory_level": inventory, "sales_amount": round(qty * (price - disc), 2),
        "city": "Mumbai", "discount_pct": round(disc / price * 100, 2) if price else 0.0,
        "is_promotion": 0 if promo is None else 1, "transaction_id": txn,
        "product_name": "P", "region": "West", "weekday": WEEKDAYS[date.weekday()],
        "is_weekend": 1 if date.weekday() >= 5 else 0, "is_holiday": 0,
        "year_month": f"{date.year}-{date.month:02d}",
        "week_of_year": date.isocalendar()[1],
        "quarter": f"Q{(date.month - 1) // 3 + 1}", "price_band": _price_band(price),
        "customer_segment": "Loyal", "loyalty_tier": "Silver", "channel": channel,
    }


@pytest.fixture
def toy_sales():
    """Four customers with deliberately different behaviour around one snapshot.

    ALIVE   buys before the snapshot and again inside the label window.
    DORMANT buys only before the snapshot -- so churn = 1.
    NEW     first buys *after* the snapshot, so they cannot appear in it at all.
    LATER   buys only in late 2025, so tests can use a late snapshot.

    The frame has to run past 2025-05-01 for the 90-day label window of a 2025-01-31
    snapshot to be observable at all -- otherwise every snapshot is correctly dropped as
    unlabelled and there is nothing to assert.
    """
    return pd.DataFrame([
        # ALIVE: two baskets before the snapshot, one inside the label window.
        _row(dt.date(2025, 1, 5), "ALIVE", "A1", qty=2, price=500.0, channel="Web"),
        _row(dt.date(2025, 1, 20), "ALIVE", "A2", qty=4, price=500.0, channel="Web",
             promo="Weekend_Flash", disc=100.0, category="Beverages"),
        _row(dt.date(2025, 4, 10), "ALIVE", "A3", qty=1, price=500.0),
        # DORMANT: last purchase 2025-01-25, then silence.
        _row(dt.date(2025, 1, 6), "DORMANT", "D1", qty=3, price=200.0),
        _row(dt.date(2025, 1, 25), "DORMANT", "D2", qty=1, price=200.0, category="Bakery"),
        # NEW: had not joined by the snapshot.
        _row(dt.date(2025, 2, 10), "NEW", "N1", qty=5, price=100.0),
        # LATER: gives late snapshots some in-window activity.
        _row(dt.date(2025, 11, 5), "LATER", "L1", qty=2, price=300.0),
        _row(dt.date(2025, 12, 20), "LATER", "L2", qty=6, price=300.0, category="Bakery"),
    ], columns=SALES_COLUMNS)


SNAPSHOT = dt.date(2025, 1, 31)


# --------------------------------------------------------------------------- core rule
def test_features_ignore_transactions_after_the_snapshot(toy_sales):
    """ALIVE's post-snapshot purchase must not touch its features.

    Without this barrier the model reads the answer off the label window.
    """
    feats = rfm_features(toy_sales, as_of=SNAPSHOT, feature_days=90)
    alive = feats.loc[feats["customer_id"] == "ALIVE"].iloc[0]

    # Only A1 and A2 count: 2 baskets, 6 units, 1000.00 + 1600.00 = 2600.00 spend.
    assert alive["frequency"] == 2
    assert alive["monetary"] == pytest.approx(2600.0)
    assert alive["units_per_line"] == pytest.approx(6 / 2)
    assert alive["avg_basket_value"] == pytest.approx(1300.0)


def test_a_customer_who_joined_after_the_snapshot_does_not_exist_at_it(toy_sales):
    feats = rfm_features(toy_sales, as_of=SNAPSHOT, feature_days=90)
    assert "NEW" not in set(feats["customer_id"])


def test_recency_is_measured_from_the_snapshot_not_from_the_data_end(toy_sales):
    """ALIVE last bought 2025-01-20, which is 11 days before the snapshot.

    Recomputed on 2025-04-10 it would read 0, which would tell the model the customer is
    active -- the exact opposite of the truth.
    """
    feats = rfm_features(toy_sales, as_of=SNAPSHOT, feature_days=90)
    alive = feats.loc[feats["customer_id"] == "ALIVE"].iloc[0]
    assert alive["recency_days"] == 11
    assert alive["last_purchase"] == pd.Timestamp("2025-01-20")


def test_feature_as_of_is_recorded_on_every_row(toy_sales):
    feats = rfm_features(toy_sales, as_of=SNAPSHOT, feature_days=90)
    assert (feats["feature_as_of"] == pd.Timestamp(SNAPSHOT)).all()
    assert (feats["feature_window_start"] <= pd.Timestamp(SNAPSHOT)).all()


def test_window_length_is_exactly_as_requested(toy_sales):
    """A 90-day window ending on 2025-12-31 opens on 2025-10-03, 89 days earlier.

    90 inclusive days, not 90 elapsed days -- the off-by-one that makes a "90-day" window
    silently 91 days wide and lets one extra day of future data in.
    """
    feats = rfm_features(toy_sales, as_of="2025-12-31", feature_days=90)
    assert (feats["feature_window_start"] == pd.Timestamp("2025-10-03")).all()


# --------------------------------------------------------------------------- re-derived
def test_every_feature_is_reproducible_from_pre_snapshot_rows_only(toy_sales):
    """Recompute the features by hand from the raw rows, then compare.

    This is the independent check. It does not call the feature function at all, so it
    fails if the function's own `feature_as_of` bookkeeping is a lie.
    """
    snapshot = pd.Timestamp(SNAPSHOT)
    feats = rfm_features(toy_sales, as_of=SNAPSHOT, feature_days=90)

    for _, row in feats.iterrows():
        cust = row["customer_id"]
        visible = toy_sales.loc[
            (toy_sales["customer_id"] == cust)
            & (toy_sales["date"] <= snapshot)          # <= snapshot, nothing later
            & (toy_sales["date"] >= snapshot - pd.Timedelta(days=89))
        ]

        assert row["frequency"] == visible["transaction_id"].nunique()
        assert row["monetary"] == pytest.approx(visible["sales_amount"].sum())
        assert row["lines"] == len(visible)
        assert row["category_diversity"] == visible["product_category"].nunique()
        assert row["promo_share"] == pytest.approx(visible["is_promotion"].mean())
        assert row["mean_discount_pct"] == pytest.approx(visible["discount_pct"].mean())
        assert row["online_share"] == pytest.approx(
            (visible["channel"] != "Store").mean())
        assert row["last_purchase"] == visible["date"].max()
        assert row["recency_days"] == (snapshot - visible["date"].max()).days
        assert (visible["date"] <= snapshot).all(), "source rows post-date the snapshot"


def test_no_feature_row_references_a_date_after_its_snapshot(toy_sales):
    feats = rfm_features(toy_sales, as_of=SNAPSHOT, feature_days=90)
    assert (pd.to_datetime(feats["feature_as_of"])
            <= pd.to_datetime(SNAPSHOT)).all()
    assert (pd.to_datetime(feats["last_purchase"])
            <= pd.to_datetime(SNAPSHOT)).all()


# --------------------------------------------------------------------------- labels
def test_labels_match_a_hand_computed_outcome(toy_sales):
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT],
                            feature_days=90, label_days=90)
    labelled = dict(zip(snaps["customer_id"], snaps["churn"]))

    # ALIVE buys on 2025-04-10, inside the 90 days after the snapshot -> not churned.
    # DORMANT buys nothing after 2025-01-31 -> churned.
    assert labelled["ALIVE"] == 0
    assert labelled["DORMANT"] == 1
    assert "NEW" not in labelled, "cannot be labelled at a snapshot it had not reached"


def test_label_window_is_the_ninety_days_after_the_snapshot(toy_sales):
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT],
                            feature_days=90, label_days=90)
    assert (snaps["label_window_start"] == pd.Timestamp(SNAPSHOT)).all()
    assert (snaps["label_window_end"] == pd.Timestamp("2025-05-01")).all()


def test_a_purchase_exactly_on_the_window_edge_is_counted(toy_sales):
    """Off-by-one here silently changes the label, so pin both edges.

    The snapshot day itself belongs to the *feature* window, and the day after belongs to
    the label window.
    """
    snaps = churn_snapshots(toy_sales, snapshot_dates=[pd.Timestamp("2025-01-10")],
                            feature_days=90, label_days=90)
    labelled = dict(zip(snaps["customer_id"], snaps["churn"]))

    # ALIVE's A1 is on 2025-01-05, before this snapshot, so it is already history.
    # A2 on 2025-01-20 falls inside the label window -> not churned.
    assert labelled["ALIVE"] == 0
    # DORMANT's last purchase was 2025-01-25, inside the window -> not churned either.
    assert labelled["DORMANT"] == 0


def test_snapshots_whose_label_window_runs_past_the_data_are_dropped(toy_sales):
    """Not labelled 0 -- unlabelled. The future simply was not observed."""
    snaps = churn_snapshots(toy_sales,
                            snapshot_dates=[pd.Timestamp("2025-12-01"),
                                            pd.Timestamp("2025-01-31")],
                            feature_days=90, label_days=90)
    dropped = snaps.attrs["dropped_snapshots"]
    assert len(dropped) == 1
    assert dropped[0]["snapshot_date"] == "2025-12-01"
    assert set(snaps["snapshot_date"]) == {pd.Timestamp("2025-01-31")}


def test_every_snapshot_in_the_default_set_has_room_for_its_label_window():
    """The spec's 9 snapshots, checked against when the real data actually ends.

    The last snapshot is 2025-09-30 precisely because a 90-day forward window has to fit
    before 2025-12-31. A snapshot in November or December would be unlabelled.
    """
    dates = month_end_snapshots()
    assert len(dates) == 9
    assert str(dates[0].date()) == "2025-01-31"
    assert str(dates[-1].date()) == "2025-09-30"
    sales_end = pd.Timestamp(config.SALES_END)
    for d in dates:
        assert d + pd.Timedelta(days=90) <= sales_end, f"{d.date()} has no future"


def test_dormant_customers_are_kept_not_filtered_out(toy_sales):
    """They are the answer, not noise.

    At 2025-06-15 DORMANT has no purchase inside the 90-day window -- their last was
    2025-01-25. A window that silently dropped customers with no recent activity would
    delete every true positive before the model ever saw it.
    """
    as_of = "2025-06-15"
    feats = rfm_features(toy_sales, as_of=as_of, feature_days=90,
                         include_unconverted=True)
    assert "DORMANT" in set(feats["customer_id"])
    dormant = feats.loc[feats["customer_id"] == "DORMANT"].iloc[0]
    assert dormant["lines"] == 0, "no in-window activity, which is the whole signal"
    assert dormant["frequency"] == 0, "frequency is the 90-day count, by design"
    assert dormant["recency_days"] > 100, "but recency still knows how long it has been"
    assert dormant["tenure_days"] > 100, "and tenure keeps the lifetime context"
    assert dormant["last_purchase"] == pd.Timestamp("2025-01-25")


def test_include_unconverted_false_is_the_explicit_opt_in(toy_sales):
    feats = rfm_features(toy_sales, as_of="2025-06-15", feature_days=90,
                         include_unconverted=False)
    assert "DORMANT" not in set(feats["customer_id"])
    assert (feats["lines"] > 0).all()


# --------------------------------------------------------------------------- guard
def test_guard_rejects_a_row_whose_features_post_date_the_snapshot(toy_sales):
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT],
                            feature_days=90, label_days=90)
    tampered = snaps.copy()
    tampered.loc[tampered.index[0], "feature_as_of"] = pd.Timestamp("2025-06-01")
    with pytest.raises(ValueError, match="after their snapshot"):
        assert_no_leakage(tampered, allow_labels=True)


def test_guard_rejects_a_label_smuggled_into_a_feature_frame(toy_sales):
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT],
                            feature_days=90, label_days=90)
    leaky = feature_matrix(snaps)
    leaky["churn"] = snaps["churn"].values        # merged back in by mistake
    with pytest.raises(ValueError, match="post-outcome"):
        assert_no_leakage(leaky)


def test_guard_accepts_a_labelled_frame_when_told_to(toy_sales):
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT],
                            feature_days=90, label_days=90)
    assert assert_no_leakage(snaps, allow_labels=True) is True


def test_feature_matrix_carries_no_label(toy_sales):
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT],
                            feature_days=90, label_days=90)
    matrix = feature_matrix(snaps)
    for column in ("churn", "purchased_in_label_window",
                   "label_window_start", "label_window_end"):
        assert column not in matrix.columns
    for column in FEATURE_COLUMNS:
        assert column in matrix.columns


# --------------------------------------------------------------------------- panel
def _panel_row(week, units):
    return {
        "week_start_date": pd.Timestamp(week), "year": week.isocalendar()[0],
        "week_of_year": week.isocalendar()[1], "store_id": "ST01", "city": "Mumbai",
        "region": "West", "product_id": "P01", "product_category": "Grocery",
        "units_sold": units, "revenue": units * 100.0, "avg_unit_price": 100.0,
        "avg_discount_pct": 0.0, "is_promo_week": 0, "stockout_count": 0,
        "on_hand_end": 50,
    }


def test_panel_lags_never_contain_the_current_week():
    monday = dt.date(2025, 1, 6)
    rows = [_panel_row(monday + dt.timedelta(days=7 * i), units)
            for i, units in enumerate([10, 20, 30, 40, 50])]
    panel = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    out = panel_features(panel)

    assert out["lag_1"].isna().sum() == 1, "first week has no lag by construction"
    assert out["lag_1"].dropna().tolist() == [10, 20, 30, 40]
    assert out["lag_2"].dropna().tolist() == [10, 20, 30]
    # final rolling mean covers the four *previous* weeks: mean(10,20,30,40) = 25
    assert out["roll_mean_4"].iloc[-1] == pytest.approx(25.0)


def test_panel_rolling_features_exclude_the_current_week():
    """A rolling mean that included the current week would leak the target directly."""
    monday = dt.date(2025, 1, 6)
    rows = [_panel_row(monday + dt.timedelta(days=7 * i), units)
            for i, units in enumerate([10, 1000, 30, 40, 50])]
    out = panel_features(pd.DataFrame(rows, columns=PANEL_COLUMNS))
    # The 1000-unit week must not appear in its own rolling mean.
    assert out["roll_mean_4"].iloc[1] == pytest.approx(10.0)


def test_panel_feature_as_of_is_the_previous_week():
    monday = dt.date(2025, 1, 6)
    rows = [_panel_row(monday + dt.timedelta(days=7 * i), 10) for i in range(5)]
    out = panel_features(pd.DataFrame(rows, columns=PANEL_COLUMNS))
    assert (out["feature_as_of"]
            == out["week_start_date"] - pd.Timedelta(days=7)).all()


def test_panel_rejects_a_frame_missing_the_grain():
    panel = pd.DataFrame(_panel_row(dt.date(2025, 1, 6), 5), index=[0])
    with pytest.raises(ValueError, match="missing required columns"):
        panel_features(panel.drop(columns=["store_id", "units_sold"]))

# ------------------------------------------------------------------- model_matrix contract

def test_feature_matrix_keeps_keys_and_model_matrix_drops_them(toy_sales):
    """The two views have different jobs and confusing them is how a string column ends
    up inside an estimator.

    `feature_matrix` keeps `customer_id` and the dates so results can be joined back and
    leakage audited; `model_matrix` keeps only what a model can consume.
    """
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT])

    wide = feature_matrix(snaps)
    narrow = model_matrix(snaps)

    assert "customer_id" in wide.columns
    assert "snapshot_date" in wide.columns
    assert "customer_id" not in narrow.columns
    assert "snapshot_date" not in narrow.columns
    assert list(narrow.columns) == FEATURE_COLUMNS
    assert set(narrow.columns).issubset(set(wide.columns))


def test_model_matrix_never_contains_a_label_column(toy_sales):
    """`churn` is numeric, so an imputer would happily accept it. Only this check stops it."""
    snaps = churn_snapshots(toy_sales, snapshot_dates=[SNAPSHOT])

    narrow = model_matrix(snaps)
    for column in narrow.columns:
        assert not column.lower().startswith(FORBIDDEN_PREFIXES)
        assert "churn" not in column.lower()


def test_model_matrix_rejects_a_string_feature_rather_than_coercing_it():
    frame = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "recency_days": [1.0, 2.0],
        "label_note": ["a", "b"],
    })
    with pytest.raises(ValueError, match="not numeric"):
        model_matrix(frame, columns=["recency_days", "label_note"])


def test_model_matrix_rejects_a_missing_feature_by_name():
    frame = pd.DataFrame({"recency_days": [1.0]})
    with pytest.raises(ValueError, match="monetary"):
        model_matrix(frame, columns=["recency_days", "monetary"])