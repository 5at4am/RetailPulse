"""F-01 must reject bad data, not merely describe it.

Every test here builds a *valid* small fixture, corrupts exactly one thing, and asserts
`IngestError` is raised. A validator that only ever sees good data is a validator that
has never been tested.

Fixtures are small and hand-built rather than sampled from the real CSV, because a
sampled frame inherits whatever the generator happened to produce and a test built on it
can pass for the wrong reason. `enforce_counts=False` switches off the dataset-level
checks (250,000 rows, 731 days) and leaves every row-level rule armed.
"""
import datetime as dt

import pandas as pd
import pytest

from src.ingest import (
    PANEL_COLUMNS, SALES_COLUMNS, IngestError, validate_panel, validate_sales,
)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _price_band(price):
    return ("Budget" if price < 299 else "Mid" if price < 1000
            else "Premium" if price < 5000 else "Luxury")


def _sales_row(date, qty=3, price=500.0, disc=0.0, promo=None, inventory=40,
               txn="T1", cust="C1"):
    """One internally consistent line-item. Every derived column is computed, not typed."""
    return {
        "date": pd.Timestamp(date),
        "customer_id": cust,
        "store_id": "ST01",
        "product_id": "P001",
        "product_category": "Grocery",
        "quantity_sold": qty,
        "unit_price": price,
        "discount": disc,
        "promotion": promo,
        "inventory_level": inventory,
        "sales_amount": round(qty * (price - disc), 2),
        "city": "Mumbai",
        "discount_pct": round(disc / price * 100, 2) if price else 0.0,
        "is_promotion": 0 if promo is None else 1,
        "transaction_id": txn,
        "product_name": "Test Product",
        "region": "West",
        "weekday": WEEKDAYS[date.weekday()],
        "is_weekend": 1 if date.weekday() >= 5 else 0,
        "is_holiday": 0,
        "year_month": f"{date.year}-{date.month:02d}",
        "week_of_year": date.isocalendar()[1],
        "quarter": f"Q{(date.month - 1) // 3 + 1}",
        "price_band": _price_band(price),
        "customer_segment": "Loyal",
        "loyalty_tier": "Silver",
        "channel": "Store",
    }


def _panel_row(week, units=10, revenue=5000.0, stockout=0):
    return {
        "week_start_date": pd.Timestamp(week),
        "year": week.isocalendar()[0],
        "week_of_year": week.isocalendar()[1],
        "store_id": "ST01",
        "city": "Mumbai",
        "region": "West",
        "product_id": "P001",
        "product_category": "Grocery",
        "units_sold": units,
        "revenue": revenue,
        "avg_unit_price": round(revenue / units, 2) if units else 0.0,
        "avg_discount_pct": 0.0,
        "is_promo_week": 0,
        "stockout_count": stockout,
        "on_hand_end": 50,
    }


def _frame(rows, columns):
    df = pd.DataFrame(rows, columns=columns)
    df["date" if "date" in columns else "week_start_date"] = pd.to_datetime(
        df["date" if "date" in columns else "week_start_date"])
    return df


@pytest.fixture
def good_sales():
    d = dt.date(2024, 1, 1)
    return _frame([
        _sales_row(d, qty=3, price=500.0, inventory=40, txn="T1", cust="C1"),
        _sales_row(d + dt.timedelta(days=1), qty=0, price=250.0, inventory=0,
                   txn="T2", cust="C2"),                                   # lost demand
        _sales_row(d + dt.timedelta(days=2), qty=2, price=1200.0, disc=120.0,
                   promo="Weekend_Flash", inventory=10, txn="T3", cust="C1"),
    ], SALES_COLUMNS)


@pytest.fixture
def good_panel():
    monday = dt.date(2024, 1, 1)
    return _frame([
        _panel_row(monday, units=10, revenue=5000.0),
        _panel_row(monday + dt.timedelta(days=7), units=0, revenue=0.0, stockout=2),
    ], PANEL_COLUMNS)


# ------------------------------------------------------------------ the good fixtures
def test_valid_fixtures_pass(good_sales, good_panel):
    stats = validate_sales(good_sales, enforce_counts=False)
    assert stats["rows"] == 3
    assert stats["lost_demand_rows"] == 1, "qty 0 with inventory 0 is lost demand"
    assert stats["promoted_rows"] == 1
    assert stats["units"] == 5

    pstats = validate_panel(good_panel, enforce_counts=False)
    assert pstats["rows"] == 2
    assert pstats["zero_demand_rows"] == 1


# ------------------------------------------------------------------ sales: row rules
@pytest.mark.parametrize("column,value,reason", [
    ("sales_amount", 999.99, "sales_amount != quantity x (price - discount)"),
    ("discount", 99999.0, "discount exceeds unit_price"),
    ("discount_pct", 55.0, "discount_pct disagrees with discount / unit_price"),
    ("is_promotion", 1, "is_promotion disagrees with a null promotion"),
    ("quantity_sold", -3, "negative quantity"),
    ("inventory_level", -1, "negative inventory"),
    ("weekday", "Funday", "weekday disagrees with date"),
    ("year_month", "2024-13", "year_month disagrees with date"),
    ("price_band", "Luxury", "price_band disagrees with unit_price"),
    ("is_weekend", 1, "is_weekend disagrees with date"),
])
def test_corrupted_sales_row_is_rejected(good_sales, column, value, reason):
    df = good_sales.copy()
    df.loc[0, column] = value
    with pytest.raises(IngestError):
        validate_sales(df, enforce_counts=False)


def test_discount_without_promotion_is_rejected(good_sales):
    df = good_sales.copy()
    df.loc[0, "discount"] = 50.0          # unpromoted line now carries a discount
    with pytest.raises(IngestError):
        validate_sales(df, enforce_counts=False)


def test_zero_quantity_with_stock_on_hand_is_rejected(good_sales):
    df = good_sales.copy()
    df.loc[0, "quantity_sold"] = 0        # sold nothing but shelves are not empty
    df.loc[0, "sales_amount"] = 0.0
    with pytest.raises(IngestError):
        validate_sales(df, enforce_counts=False)


def test_sold_last_unit_is_not_flagged_as_lost_demand(good_sales):
    """The one-directional rule, pinned.

    27,749 real rows sold the final unit and also sit at inventory_level 0. Rejecting
    that direction would throw away legitimate sales -- and it is the mistake that
    overstates lost demand 3.7x.
    """
    df = good_sales.copy()
    df.loc[0, "quantity_sold"] = 2
    df.loc[0, "inventory_level"] = 0
    df.loc[0, "sales_amount"] = round(2 * 500.0, 2)
    stats = validate_sales(df, enforce_counts=False)
    assert stats["lost_demand_rows"] == 1
    assert stats["sold_last_unit_rows"] == 1


def test_promotion_null_is_the_only_allowed_null(good_sales):
    df = good_sales.copy()
    df.loc[0, "city"] = None
    with pytest.raises(IngestError):
        validate_sales(df, enforce_counts=False)


def test_missing_column_is_rejected(good_sales):
    with pytest.raises(IngestError):
        validate_sales(good_sales.drop(columns=["city"]), enforce_counts=False)


def test_reordered_columns_are_rejected(good_sales):
    """Column *order* is part of the contract, not just the column set."""
    shuffled = SALES_COLUMNS[1:] + SALES_COLUMNS[:1]
    with pytest.raises(IngestError):
        validate_sales(good_sales[shuffled], enforce_counts=False)


# ------------------------------------------------------------------ panel: row rules
@pytest.mark.parametrize("column,value,reason", [
    ("units_sold", -1, "negative units"),
    ("revenue", -5.0, "negative revenue"),
    ("stockout_count", -1, "negative stockout count"),
    ("on_hand_end", -1, "negative stock on hand"),
])
def test_corrupted_panel_row_is_rejected(good_panel, column, value, reason):
    df = good_panel.copy()
    df.loc[0, column] = value
    with pytest.raises(IngestError):
        validate_panel(df, enforce_counts=False)


def test_zero_demand_week_carrying_revenue_is_rejected(good_panel):
    df = good_panel.copy()
    df.loc[1, "revenue"] = 100.0          # nothing sold, yet there is money
    with pytest.raises(IngestError):
        validate_panel(df, enforce_counts=False)


def test_selling_week_carrying_no_revenue_is_rejected(good_panel):
    df = good_panel.copy()
    df.loc[0, "revenue"] = 0.0
    with pytest.raises(IngestError):
        validate_panel(df, enforce_counts=False)


def test_non_monday_week_start_is_rejected(good_panel):
    df = good_panel.copy()
    df.loc[0, "week_start_date"] = pd.Timestamp("2024-01-02")   # a Tuesday
    with pytest.raises(IngestError):
        validate_panel(df, enforce_counts=False)


def test_duplicate_grain_key_is_rejected(good_panel):
    df = pd.concat([good_panel, good_panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(IngestError):
        validate_panel(df, enforce_counts=False)


def test_missing_file_raises(tmp_path):
    from src.ingest import load_sales
    with pytest.raises(IngestError):
        load_sales(tmp_path / "nope.csv")