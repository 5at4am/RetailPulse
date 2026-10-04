"""F-01 — Data ingestion and cleaning.

    from src.ingest import load_sales, load_panel, ingest

    sales, panel, report = ingest()          # validates, raises on any broken contract
    sales = load_sales()                     # just the frame

**Fails loudly, on purpose.** A violated contract raises `IngestError`; it never
warns-and-continues. A silent coercion would let a broken extract reach the models and
produce a confident wrong answer, which is the worst possible outcome for a grading demo.

Runnable on its own: `python -m src.ingest`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):                 # allow `python src/ingest.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

# Exact column order. A re-ordered header is a different dataset, so compare the list.
SALES_COLUMNS = [
    "date", "customer_id", "store_id", "product_id", "product_category",
    "quantity_sold", "unit_price", "discount", "promotion", "inventory_level",
    "sales_amount", "city", "discount_pct", "is_promotion", "transaction_id",
    "product_name", "region", "weekday", "is_weekend", "is_holiday", "year_month",
    "week_of_year", "quarter", "price_band", "customer_segment", "loyalty_tier",
    "channel",
]

PANEL_COLUMNS = [
    "week_start_date", "year", "week_of_year", "store_id", "city", "region",
    "product_id", "product_category", "units_sold", "revenue", "avg_unit_price",
    "avg_discount_pct", "is_promo_week", "stockout_count", "on_hand_end",
]

# `promotion` is the only nullable column: it is NaN by design on every unpromoted line.
SALES_NULLABLE = {"promotion"}

NUMERIC_SALES = [
    "quantity_sold", "unit_price", "discount", "inventory_level", "sales_amount",
    "discount_pct", "is_promotion", "is_weekend", "is_holiday", "week_of_year",
]

NUMERIC_PANEL = [
    "year", "week_of_year", "units_sold", "revenue", "avg_unit_price",
    "avg_discount_pct", "is_promo_week", "stockout_count", "on_hand_end",
]

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
# Series.map needs a callable or a mapping -- indexing a list of names by weekday.
WEEKDAY_BY_INDEX = dict(enumerate(WEEKDAY_NAMES))

PRICE_BANDS = (("Budget", 299.0), ("Mid", 1000.0), ("Premium", 5000.0))


class IngestError(ValueError):
    """The data does not match its documented contract. Do not model on this."""


def _require(condition, message):
    if not condition:
        raise IngestError(message)


def _check_schema(df, columns, label):
    got = list(df.columns)
    _require(got == columns,
             f"{label}: column contract broken\n"
             f"  expected: {columns}\n  found   : {got}")


def _check_shape(df, rows, cols, label, enforce_counts=True):
    _require(df.shape[1] == cols,
             f"{label}: expected {cols} columns, got {df.shape[1]}")
    if enforce_counts:
        _require(df.shape[0] == rows,
                 f"{label}: expected {rows:,} rows, got {df.shape[0]:,} "
                 f"(truncated download? wrong file?)")


def _check_no_unexpected_nulls(df, nullable, label):
    allowed = set(nullable)
    counts = df.isna().sum()
    offenders = {c: int(n) for c, n in counts.items() if n and c not in allowed}
    _require(not offenders,
             f"{label}: unexpected nulls in columns that must be complete: {offenders}")


def _price_band(unit_price):
    for name, ceiling in PRICE_BANDS:
        if unit_price < ceiling:
            return name
    return "Luxury"


def validate_sales(df, enforce_counts=True):
    """Check every documented row-level rule. Raises `IngestError` on the first breach.

    The rules, and why each one exists:

    1. `sales_amount == quantity_sold x (unit_price - discount)` -- the money is derivable
       from the other columns, so a mismatch means the extract is inconsistent.
    2. `discount <= unit_price` -- a discount larger than the price is a negative sale.
    3. `discount_pct == discount / unit_price x 100` -- derived, so it must agree.
       Computed from the *rounded* discount, hence the tolerance rather than equality.
    4. `is_promotion == (promotion is not null)` -- the 0/1 flag mirrors the text column.
    5. No discount without a promotion -- a documented business rule of the generator.
    6. `quantity_sold >= 0` and `inventory_level >= 0`.
    7. `quantity_sold == 0` implies `inventory_level == 0` -- zero quantity means an empty
       shelf, which is what makes it lost demand. Note this is one-directional: 27,749 rows
       bought the *last* unit and also sit at inventory 0, which is a perfectly normal sale.
    8. Calendar columns agree with `date` -- `weekday`, `is_weekend`, `year_month`,
       `week_of_year`, `quarter`, `price_band`.

    Returns a dict of headline counts so callers can log what was validated.

    `enforce_counts=False` skips the *dataset-level* checks -- exact row count, the
    2024-01-01..2025-12-31 date range, 731 distinct days -- while still enforcing every
    row-level rule. That is what lets `tests/test_ingest.py` validate a small hand-built
    fixture, and what lets a partial extract be checked for internal consistency.
    """
    label = "sales"
    _check_schema(df, SALES_COLUMNS, label)
    _check_shape(df, config.SALES_ROWS, config.SALES_COLS, label, enforce_counts)
    _check_no_unexpected_nulls(df, SALES_NULLABLE, label)

    qty = df["quantity_sold"]
    price = df["unit_price"]
    disc = df["discount"]
    amount = df["sales_amount"]
    promo_null = df["promotion"].isna()

    def rule(name, bad, limit=0, extra=""):
        n = int(bad.sum())
        _require(n <= limit, f"{label}: rule '{name}' violated on {n:,} row(s) {extra}")

    rule("sales_amount == quantity x (price - discount)",
         (amount - (qty * (price - disc)).round(2)).abs() > 0.011, extra="(> 1 paisa)")
    rule("discount <= unit_price", disc > price + 1e-9)
    rule("discount_pct == discount / unit_price x 100",
         (df["discount_pct"] - (disc / price * 100).fillna(0.0)).abs() > 0.011)
    rule("is_promotion == (promotion is not null)",
         df["is_promotion"] != (~promo_null).astype("int8"))
    rule("no discount without a promotion", promo_null & (disc != 0.0))
    rule("quantity_sold >= 0", qty < 0)
    rule("inventory_level >= 0", df["inventory_level"] < 0)
    rule("quantity_sold == 0 implies inventory_level == 0",
         (qty == 0) & (df["inventory_level"] != 0))

    date = df["date"]
    weekday = date.dt.weekday
    rule("weekday agrees with date", df["weekday"] != weekday.map(WEEKDAY_BY_INDEX))
    rule("is_weekend agrees with date",
         df["is_weekend"] != (weekday >= 5).astype("int8"))
    rule("year_month agrees with date",
         df["year_month"] != date.dt.strftime("%Y-%m"))
    rule("week_of_year agrees with the ISO calendar",
         df["week_of_year"] != date.dt.isocalendar().week.astype("int16"))
    rule("quarter agrees with date",
         df["quarter"] != "Q" + date.dt.quarter.astype(str))
    rule("price_band agrees with unit_price",
         df["price_band"] != price.map(_price_band))

    if enforce_counts:
        lo, hi = date.min(), date.max()
        _require(str(lo.date()) == config.SALES_START
                 and str(hi.date()) == config.SALES_END,
                 f"{label}: date range is {lo.date()} .. {hi.date()}, expected "
                 f"{config.SALES_START} .. {config.SALES_END}")
        _require(df["date"].nunique() == 731,
                 f"{label}: expected 731 distinct days, got {df['date'].nunique()}")
    else:
        lo, hi = date.min(), date.max()

    lost = int((qty == 0).sum())
    sold_last = int(((qty > 0) & (df["inventory_level"] == 0)).sum())
    return {
        "rows": len(df),
        "lost_demand_rows": lost,
        "sold_last_unit_rows": sold_last,
        "promoted_rows": int(df["is_promotion"].sum()),
        "revenue": round(float(amount.sum()), 2),
        "units": int(qty.sum()),
        "date_from": str(lo.date()),
        "date_to": str(hi.date()),
    }


def validate_panel(df, enforce_counts=True):
    """Check the weekly panel contract.

    The panel is deliberately *not* reconciled against sales here -- it covers only the
    top-velocity assortment and drops the clipped final week, so its totals are supposed
    to be smaller. `reconcile_panel.py` explains the gap.

    As in `validate_sales`, `enforce_counts=False` drops the dataset-level checks (row
    count, 104 weeks, the exact last week) but keeps every row-level rule.
    """
    label = "panel"
    _check_schema(df, PANEL_COLUMNS, label)
    _check_shape(df, config.PANEL_ROWS, config.PANEL_COLS, label, enforce_counts)
    _check_no_unexpected_nulls(df, set(), label)

    for col in NUMERIC_PANEL:
        rule = f"{col} >= 0"
        bad = int((df[col] < 0).sum())
        _require(bad == 0, f"{label}: rule '{rule}' violated on {bad:,} row(s)")

    week = df["week_start_date"]
    _require((week.dt.weekday == 0).all(),
             f"{label}: every week_start_date must be a Monday "
             f"({int((week.dt.weekday != 0).sum()):,} are not)")
    if enforce_counts:
        _require(week.nunique() == config.PANEL_WEEKS,
                 f"{label}: expected {config.PANEL_WEEKS} weeks, got {week.nunique()}")
        _require(str(week.max().date()) == "2025-12-22",
                 f"{label}: last week starts {week.max().date()}, expected 2025-12-22")
        _require(config.PARTIAL_FINAL_WEEK not in set(week.dt.strftime("%Y-%m-%d")),
                 f"{label}: the clipped final week {config.PARTIAL_FINAL_WEEK} "
                 f"must be absent")

    # A zero-demand week is real, not missing -- but it cannot carry revenue or a discount.
    zero = df["units_sold"] == 0
    _require(bool((df.loc[zero, "revenue"] == 0).all()),
             "panel: zero-demand weeks carry revenue, which is impossible")
    _require(bool((df.loc[zero, "avg_discount_pct"] == 0).all()),
             "panel: zero-demand weeks carry a discount, which is impossible")
    _require(bool((df.loc[~zero, "revenue"] > 0).all()),
             "panel: selling weeks carry no revenue")

    grain = df.groupby(["store_id", "product_id", "week_start_date"], sort=False).size()
    _require(int(grain.max()) == 1,
             f"panel: grain (store, product, week) is not unique -- "
             f"{int((grain > 1).sum()):,} duplicate key(s)")

    return {
        "rows": len(df),
        "weeks": int(week.nunique()),
        "stores": int(df["store_id"].nunique()),
        "products": int(df["product_id"].nunique()),
        "zero_demand_rows": int(zero.sum()),
        "zero_demand_pct": round(100 * float(zero.mean()), 1),
        "stockout_rows": int((df["stockout_count"] > 0).sum()),
        "units": int(df["units_sold"].sum()),
        "week_from": str(week.min().date()),
        "week_to": str(week.max().date()),
    }


def load_sales(path=None, enforce_counts=True):
    """Load the 250,000-row sales file as a validated DataFrame."""
    path = Path(path or config.SALES_CSV)
    _require(path.exists(),
             f"{path} not found -- run `python generate_retail_pulse.py` first")
    df = pd.read_csv(path, parse_dates=["date"])
    validate_sales(df, enforce_counts=enforce_counts)
    return df


def load_panel(path=None, enforce_counts=True):
    """Load the 780,000-row weekly panel as a validated DataFrame."""
    path = Path(path or config.PANEL_CSV)
    _require(path.exists(),
             f"{path} not found -- run `python generate_retail_pulse.py` first")
    df = pd.read_csv(path, parse_dates=["week_start_date"])
    validate_panel(df, enforce_counts=enforce_counts)
    return df


def cross_check_facts(sales, panel):
    """Compare the loaded shapes against `facts.json`, if it exists.

    Belt and braces: the shape check above already compares against the constants in
    `config.py`, and `facts.json` is generated *from these CSVs*, so this catches the case
    where the CSVs and the recorded facts have drifted apart.
    """
    if not config.FACTS_JSON.exists():
        return None
    facts = json.loads(config.FACTS_JSON.read_text(encoding="utf-8"))
    out = {}
    for name, df, key in (("sales", sales, "sales"), ("panel", panel, "panel")):
        recorded = facts.get(key, {})
        for field, actual in (("rows", len(df)), ("columns", df.shape[1])):
            expected = recorded.get(field)
            if expected is not None:
                _require(actual == expected,
                         f"{name}: {field} is {actual:,} but facts.json records "
                         f"{expected:,} -- the files and the facts have drifted")
        out[name] = {"rows": len(df), "columns": df.shape[1]}
    return out


def ingest(enforce_counts=True, check_facts=True):
    """Load and validate both files.

    Returns `(sales, panel, report)`. Raises `IngestError` on any broken contract.
    """
    sales = load_sales(enforce_counts=enforce_counts)
    panel = load_panel(enforce_counts=enforce_counts)

    report = {"sales": validate_sales(sales, enforce_counts=enforce_counts),
              "panel": validate_panel(panel, enforce_counts=enforce_counts)}
    if check_facts:
        report["facts"] = cross_check_facts(sales, panel)
    return sales, panel, report


def main():
    sales, panel, report = ingest()
    print("=" * 72)
    print("F-01 INGESTION -- contracts satisfied")
    print("=" * 72)
    for name in ("sales", "panel"):
        print(f"\n{name.upper()}")
        for key, value in report[name].items():
            print(f"  {key:<22} {value:,}" if isinstance(value, int)
                  else f"  {key:<22} {value}")
    print(f"\nfacts.json cross-check: "
          f"{'ok' if report.get('facts') else 'skipped (no facts.json)'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())