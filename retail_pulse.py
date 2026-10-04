"""Loader for the Retail Pulse AI dataset.

    from retail_pulse import load_sales, load_panel, validate_sales, to_pandas

    sales  = load_sales()             # list[dict], stdlib only
    panel  = load_panel()
    validate_sales(sales)             # asserts the documented contracts

    df = to_pandas("sales")           # optional, needs pandas

Pure standard library -- no dependencies. Column names, types and meanings are the ones
documented in retail_pulse_data_dictionary.md; this module is the executable version of it.

A note on types: `is_promotion`, `is_weekend`, `is_holiday` and `is_promo_week` stay as
int 0/1 rather than bool, because that is how they are stored on disk and how the data
dictionary specifies them. Cast to bool yourself if you prefer.
"""
import csv
import datetime as dt
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "raw")
SALES_CSV = os.path.join(DATA_DIR, "retail_pulse_sales.csv")
PANEL_CSV = os.path.join(DATA_DIR, "retail_pulse_demand_panel.csv")

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

_FLOAT = {"unit_price", "discount", "sales_amount", "discount_pct"}
_INT = {"quantity_sold", "inventory_level", "is_promotion", "is_weekend",
        "is_holiday", "week_of_year"}
_DATE = {"date"}

# "None" in the CSV means "not promoted". pandas' CSV parser already reads it as a
# missing value, so this loader does too -- otherwise the same column would be the
# string "None" here and NaN there, and `r["promotion"] is None` would silently be
# False for 91.6% of rows depending on which loader you used.
_NULL_TOKENS = {"None"}


def _cast(row, ints, floats, dates, nulls=()):
    for k in row:
        if k in nulls and row[k] in _NULL_TOKENS:
            row[k] = None
        elif k in dates:
            row[k] = dt.date.fromisoformat(row[k])
        elif k in ints:
            row[k] = int(row[k])
        elif k in floats:
            row[k] = float(row[k])
    return row


def load_sales(path=SALES_CSV, as_dicts=True):
    """Load the transaction line-items.

    `promotion` is None (not the string "None") where the line was not promoted,
    matching what pandas.read_csv returns for the same file. Use the `is_promotion`
    0/1 column if you just want a flag.

    Use `as_dicts=False` for a list of tuples in SALES_COLUMNS order, which is
    faster and lighter when you only need a handful of columns. Note that in tuple
    mode nothing is cast and "None" stays a literal string.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python generate_retail_pulse.py` first")
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header != SALES_COLUMNS:
            raise ValueError(
                f"unexpected sales header\n  found:    {header}\n"
                f"  expected: {SALES_COLUMNS}")
        if as_dicts:
            return [_cast(dict(zip(header, r)), _INT, _FLOAT, _DATE,
                          nulls={"promotion"}) for r in rd]
        return [tuple(r) for r in rd]


def load_panel(path=PANEL_CSV, as_dicts=True):
    """Load the weekly store-product forecasting panel."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python generate_retail_pulse.py` first")
    ints = {"year", "week_of_year", "units_sold", "is_promo_week",
            "stockout_count", "on_hand_end"}
    floats = {"revenue", "avg_unit_price", "avg_discount_pct"}
    dates = {"week_start_date"}
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header != PANEL_COLUMNS:
            raise ValueError(
                f"unexpected panel header\n  found:    {header}\n"
                f"  expected: {PANEL_COLUMNS}")
        if as_dicts:
            return [_cast(dict(zip(header, r)), ints, floats, dates) for r in rd]
        return [tuple(r) for r in rd]


def validate_sales(rows, sample=None):
    """Check the contracts the data dictionary promises. Raises AssertionError.

    `sample` checks only the first N rows -- the full 250k takes a couple of seconds.
    Pass sample=None to check everything.
    """
    if sample:
        rows = rows[:sample]
    for i, r in enumerate(rows):
        where = f"row {i + 2}"                       # +2: header is line 1
        # sales_amount == quantity_sold x (unit_price - discount), exactly
        assert abs(r["sales_amount"]
                   - r["quantity_sold"] * (r["unit_price"] - r["discount"])) < 0.005, \
            f"{where}: sales_amount does not match quantity x (price - discount)"
        assert r["discount"] <= r["unit_price"] + 1e-9, \
            f"{where}: discount exceeds unit_price"
        # discount_pct is derived from the ROUNDED discount, so these must agree exactly
        if r["unit_price"]:
            assert abs(r["discount_pct"] - r["discount"] / r["unit_price"] * 100) < 0.01, \
                f"{where}: discount_pct inconsistent with discount / unit_price"
        # is_promotion is exactly `promotion is not None`
        assert r["is_promotion"] == (0 if r["promotion"] is None else 1), \
            f"{where}: is_promotion disagrees with promotion"
        # documented rule: no discount without a promotion
        if r["promotion"] is None:
            assert r["discount"] == 0, \
                f"{where}: discount present on a non-promoted line"
        assert r["quantity_sold"] >= 0 and r["inventory_level"] >= 0, \
            f"{where}: negative quantity or inventory"
        # quantity 0 means the shelf was empty, so stock must be 0 too
        if r["quantity_sold"] == 0:
            assert r["inventory_level"] == 0, \
                f"{where}: zero quantity with stock still on hand"
    return True


def baskets(rows):
    """Group line-items into baskets by transaction_id -> {txn_id: [row, ...]}."""
    out = defaultdict(list)
    for r in rows:
        out[r["transaction_id"]].append(r)
    return out


def _week_start(d):
    """Monday of the week containing date `d`."""
    return d - dt.timedelta(days=d.weekday())


def aggregate(rows, freq="daily", by=(), drop_partial=True):
    """Roll the line-item sales up to daily or weekly totals.

    freq : "daily" or "weekly"
    by   : extra grouping columns, e.g. ("city",) or ("store_id", "product_category")
    drop_partial : weekly only. If the first or last week is cut short by the start
        or end of the data, exclude it by default.

        This matters: the data ends 2025-12-31, so the week starting 2025-12-29 holds
        only 3 of its 7 days and shows ~37% of a typical week's revenue. Left in, every
        weekly chart ends in a fake demand cliff. The shipped demand panel already
        keeps only the 104 complete weeks; this keeps the roll-up consistent with it.

        Set drop_partial=False to keep the short week, e.g. when the partial period is
        genuinely part of what you want to model.

    Returns a list of dicts sorted by period. Stockout lines carry quantity 0 and
    revenue 0, so they add nothing to the totals but are counted separately in
    `stockout_lines` -- lost demand, which is why units sold understates demand.

    The sales file is line-item grain, so this is what you want for company-wide or
    per-city reporting. For store x product forecasting use load_panel() instead,
    which is already weekly and dense (keeps the zero-demand weeks).
    """
    if freq not in ("daily", "weekly"):
        raise ValueError(f"freq must be 'daily' or 'weekly', got {freq!r}")

    keyfn = (lambda r: r["date"]) if freq == "daily" else (lambda r: _week_start(r["date"]))

    # Which weeks are complete? This is a property of the WEEK, not of any
    # (week, city) or (week, store) subgroup -- a city can easily have no sales on
    # one day while the week as a whole is fully covered. Deciding per group would
    # silently discard those groups' revenue.
    complete_periods = None
    if freq == "weekly" and drop_partial:
        days_per_week = defaultdict(set)
        for r in rows:
            days_per_week[keyfn(r)].add(r["date"])
        complete_periods = {p for p, d in days_per_week.items() if len(d) == 7}

    acc = defaultdict(lambda: {
        "lines": 0, "units": 0, "revenue": 0.0, "discount": 0.0,
        "stockout_lines": 0, "transactions": set(), "customers": set(),
        "days": set(),
    })

    for r in rows:
        key = (keyfn(r),) + tuple(r[c] for c in by)
        a = acc[key]
        a["lines"] += 1
        a["units"] += r["quantity_sold"]
        a["revenue"] += r["sales_amount"]
        a["discount"] += r["discount"] * r["quantity_sold"]
        if r["quantity_sold"] == 0:
            a["stockout_lines"] += 1
        a["transactions"].add(r["transaction_id"])
        a["customers"].add(r["customer_id"])
        a["days"].add(r["date"])

    out = []
    for key in sorted(acc):
        a = acc[key]
        period, *rest = key

        if complete_periods is not None and period not in complete_periods:
            continue        # week clipped by the start or end of the data

        row = {freq + "_start": period}
        row.update(dict(zip(by, rest)))
        row.update({
            "lines": a["lines"],
            "transactions": len(a["transactions"]),
            "customers": len(a["customers"]),
            "units": a["units"],
            "revenue": round(a["revenue"], 2),
            "avg_line_value": round(a["revenue"] / a["lines"], 2),
            "discount_value": round(a["discount"], 2),
            "stockout_lines": a["stockout_lines"],
            "days_covered": len(a["days"]),
        })
        out.append(row)
    return out


def aggregate_daily(rows=None, by=()):
    """Daily roll-up. Convenience wrapper around aggregate()."""
    return aggregate(rows if rows is not None else load_sales(), "daily", by)


def aggregate_weekly(rows=None, by=(), drop_partial=True):
    """Weekly roll-up, Monday-start weeks.

    Drops the clipped first/last week by default -- see aggregate(drop_partial=...).
    """
    return aggregate(rows if rows is not None else load_sales(),
                     "weekly", by, drop_partial=drop_partial)


def to_pandas(which="sales", path=None):
    """Optional convenience: same data as a pandas DataFrame. Requires pandas."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("to_pandas needs pandas installed") from None
    if which == "sales":
        df = pd.read_csv(path or SALES_CSV, parse_dates=["date"])
        return df.sort_values(["date", "transaction_id", "product_id"]).reset_index(drop=True)
    if which == "panel":
        return pd.read_csv(path or PANEL_CSV, parse_dates=["week_start_date"])
    raise ValueError(f"which must be 'sales' or 'panel', got {which!r}")


def summary(rows=None):
    """Print the headline facts about the loaded dataset."""
    if rows is None:
        rows = load_sales()
    qty = [r["quantity_sold"] for r in rows]
    rev = sum(r["sales_amount"] for r in rows)
    units = sum(qty)
    dates = sorted({r["date"] for r in rows})
    stockouts = sum(1 for q in qty if q == 0)
    multi = len(baskets(rows)) and len({t for t, v in baskets(rows).items() if len(v) > 1})
    print(f"rows            : {len(rows):,}")
    print(f"date range      : {dates[0]} .. {dates[-1]}  ({len(dates)} days)")
    print(f"customers       : {len({r['customer_id'] for r in rows}):,}")
    print(f"stores          : {len({r['store_id'] for r in rows}):,}")
    print(f"products        : {len({r['product_id'] for r in rows}):,}")
    print(f"categories      : {len({r['product_category'] for r in rows}):,}")
    print(f"cities          : {len({r['city'] for r in rows}):,}")
    print(f"transactions    : {len(baskets(rows)):,}  ({multi:,} multi-line)")
    print(f"units           : {units:,}")
    print(f"revenue         : INR {rev:,.0f}")
    print(f"avg sales_amount: INR {rev / len(rows):,.2f} per line-item")
    print(f"avg unit price  : INR {sum(r['unit_price'] for r in rows) / len(rows):,.2f} per unit")
    print(f"avg units/line  : {units / len(rows):,.2f}")
    print(f"stockout rows   : {stockouts:,} ({stockouts / len(rows):.2%} = lost demand)")


if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        rows = load_sales()
        summary(rows)
        print("\nvalidating every row ...")
        validate_sales(rows)
        print("all contracts hold on all rows")
    else:
        summary()