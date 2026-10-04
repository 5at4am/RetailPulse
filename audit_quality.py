"""Data-quality audit for both Retail Pulse CSVs.

Three questions this answers, per file:
  1. MISSING VALUES - empty cells, whitespace-only cells, and null-like spellings
     ('', 'NA', 'N/A', 'null', 'NaN', 'unknown', '-', 'nil', '?'), not just in the
     required columns but in every column.
  2. DUPLICATES - exact full-row duplicates, plus the natural key that actually
     matters for each grain, because a duplicate there silently breaks groupby.
  3. DATE FORMAT - strict YYYY-MM-DD, real calendar dates, no gaps, no future dates,
     and denormalised year/week columns that agree with the date.

Plus text hygiene and numeric sanity, which are the usual places synthetic data
lies. Exits non-zero if anything is found, so it can gate CI.

    python audit_quality.py
"""
import csv
import datetime as dt
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "raw"
SALES = DATA / "retail_pulse_sales.csv"
PANEL = DATA / "retail_pulse_demand_panel.csv"

MISSING_TOKENS = {"", "na", "n/a", "null", "nan", "unknown", "-", "nil", "?"}
STRICT_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STRICT_MONTH = re.compile(r"^\d{4}-\d{2}$")

TODAY = dt.date.today()
findings = []      # (file, check, detail)


def flag(fname, check, detail):
    findings.append((fname, check, detail))


def report(fname, check, bad_count, detail=""):
    """Print one row of the results table and record it if it failed."""
    ok = bad_count == 0
    mark = "ok  " if ok else "FAIL"
    extra = f"   {detail}" if detail else ""
    print(f"  [{mark}] {check:<44} {bad_count if bad_count else '0':>7}{extra}")
    if not ok:
        flag(fname, check, f"{bad_count} offending rows {detail}")


def read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return header, list(reader)


def check_missing(fname, header, rows, exempt=()):
    print(f"\n1. MISSING VALUES  ({len(header)} columns)")
    empty, nullish, ws_only = Counter(), Counter(), Counter()
    for r in rows:
        for col, val in zip(header, r):
            s = val.strip()
            if s == "":
                empty[col] += 1
            elif s != val:
                ws_only[col] += 1
            if s.lower() in MISSING_TOKENS and (col, s) not in exempt:
                nullish[col] += 1

    report(fname, "no empty cells", sum(empty.values()),
           f"-> {empty.most_common(4)}" if empty else "")
    report(fname, "no whitespace-only cells", sum(ws_only.values()),
           f"-> {ws_only.most_common(4)}" if ws_only else "")
    report(fname, "no null-like tokens", sum(nullish.values()),
           f"-> {nullish.most_common(4)}" if nullish else "")


def check_text(fname, header, rows, cols):
    print("\n2. TEXT HYGIENE (stray leading/trailing whitespace)")
    i = {c: header.index(c) for c in cols}
    for col in cols:
        bad = sum(1 for r in rows if r[i[col]] != r[i[col]].strip())
        report(fname, f"{col} has no stray whitespace", bad)


def check_duplicates_sales(header, rows):
    print("\n3. DUPLICATES")
    exact = Counter(tuple(r) for r in rows)
    report("sales", "no exact duplicate rows",
           sum(c - 1 for c in exact.values() if c > 1))

    i_txn, i_prod = header.index("transaction_id"), header.index("product_id")
    pair = Counter((r[i_txn], r[i_prod]) for r in rows)
    report("sales", "one product per basket max (txn, product)",
           sum(c - 1 for c in pair.values() if c > 1))

    # a transaction must be one customer in one store on one day
    i_date = header.index("date")
    txn = defaultdict(set)
    for r in rows:
        txn[r[i_txn]].add((r[i_date], r[1], r[2]))
    report("sales", "no transaction spans date/customer/store",
           sum(1 for v in txn.values() if len(v) > 1))
    print(f"  [ok  ] distinct transaction_id                     {len(txn):>7}")


def check_duplicates_panel(header, rows):
    print("\n3. DUPLICATES")
    exact = Counter(tuple(r) for r in rows)
    report("panel", "no exact duplicate rows",
           sum(c - 1 for c in exact.values() if c > 1))

    i = {c: header.index(c) for c in
         ("store_id", "product_id", "week_start_date", "city", "product_category")}
    key = Counter((r[i["store_id"]], r[i["product_id"]],
                   r[i["week_start_date"]]) for r in rows)
    report("panel", "natural key (store,product,week) unique",
           sum(c - 1 for c in key.values() if c > 1))

    s2c, p2cat = defaultdict(set), defaultdict(set)
    for r in rows:
        s2c[r[i["store_id"]]].add(r[i["city"]])
        p2cat[r[i["product_id"]]].add(r[i["product_category"]])
    report("panel", "store_id maps to exactly one city",
           sum(1 for v in s2c.values() if len(v) > 1))
    report("panel", "product_id maps to one category",
           sum(1 for v in p2cat.values() if len(v) > 1))


def check_dates_sales(header, rows, sales_end):
    print("\n4. DATE FORMAT")
    i = header.index("date")
    bad, parsed = 0, set()
    for r in rows:
        v = r[i]
        if not STRICT_DATE.match(v):
            bad += 1
            continue
        try:
            parsed.add(dt.date.fromisoformat(v))
        except ValueError:
            bad += 1
    report("sales", "all dates strict YYYY-MM-DD + real", bad)

    lo, hi = min(parsed), max(parsed)
    print(f"  [ok  ] range{'':<35} {lo} .. {hi}  ({len(parsed)} distinct)")

    span = (hi - lo).days + 1
    report("sales", "calendar has no missing days",
           span - len(parsed), f"span {span}d vs {len(parsed)} distinct")

    report("sales", "no dates in the future",
           sum(1 for d in parsed if d > TODAY),
           f"(today is {TODAY})")
    report("sales", "range ends 2025-12-31", 0 if hi == sales_end else 1,
           f"-> {hi}")

    iw = header.index("week_of_year")
    bad_derived = sum(1 for r in rows
                      if int(r[iw]) != dt.date.fromisoformat(r[i]).isocalendar()[1])
    report("sales", "week_of_year matches ISO calendar", bad_derived)

    im = header.index("year_month")
    report("sales", "year_month strict YYYY-MM",
           sum(1 for r in rows if not STRICT_MONTH.match(r[im])))


def check_dates_panel(header, rows, sales_end):
    print("\n4. DATE FORMAT")
    i = header.index("week_start_date")
    bad, parsed = 0, set()
    for r in rows:
        v = r[i]
        if not STRICT_DATE.match(v):
            bad += 1
            continue
        try:
            parsed.add(dt.date.fromisoformat(v))
        except ValueError:
            bad += 1
    report("panel", "all week_start_date strict YYYY-MM-DD + real", bad)

    lo, hi = min(parsed), max(parsed)
    print(f"  [ok  ] range{'':<35} {lo} .. {hi}  ({len(parsed)} distinct weeks)")

    report("panel", "every week starts on a Monday",
           sum(1 for d in parsed if d.weekday() != 0))
    report("panel", "no week runs past the sales period",
           sum(1 for d in parsed if d + dt.timedelta(days=6) > sales_end))
    report("panel", "no future weeks",
           sum(1 for d in parsed if d > TODAY))

    iy, iw = header.index("year"), header.index("week_of_year")
    bad_y = bad_w = 0
    for r in rows:
        d = dt.date.fromisoformat(r[i])
        if int(r[iy]) != d.year:
            bad_y += 1
        if int(r[iw]) != d.isocalendar()[1]:
            bad_w += 1
    report("panel", "year agrees with week_start_date", bad_y)
    report("panel", "week_of_year agrees with ISO calendar", bad_w)


def check_numbers_sales(header, rows):
    print("\n5. NUMERIC SANITY")
    i = {c: header.index(c) for c in
         ("quantity_sold", "inventory_level", "sales_amount", "unit_price", "discount")}
    report("sales", "no negative quantity",
           sum(1 for r in rows if int(r[i["quantity_sold"]]) < 0))
    report("sales", "no negative inventory",
           sum(1 for r in rows if int(r[i["inventory_level"]]) < 0))
    report("sales", "no negative revenue",
           sum(1 for r in rows if float(r[i["sales_amount"]]) < 0))
    report("sales", "discount never exceeds unit_price",
           sum(1 for r in rows if float(r[i["discount"]]) > float(r[i["unit_price"]])))
    report("sales", "stockout rows carry zero revenue",
           sum(1 for r in rows if int(r[i["quantity_sold"]]) == 0
               and float(r[i["sales_amount"]]) != 0))
    report("sales", "selling rows carry revenue",
           sum(1 for r in rows if int(r[i["quantity_sold"]]) > 0
               and float(r[i["sales_amount"]]) <= 0))


def check_numbers_panel(header, rows):
    print("\n5. NUMERIC SANITY")
    i = {c: header.index(c) for c in
         ("units_sold", "revenue", "stockout_count", "on_hand_end",
          "avg_discount_pct", "avg_unit_price")}
    for col in ("units_sold", "revenue", "stockout_count", "on_hand_end"):
        report("panel", f"no negative {col}",
               sum(1 for r in rows if float(r[i[col]]) < 0))
    report("panel", "zero-demand weeks carry no revenue or discount",
           sum(1 for r in rows if int(r[i["units_sold"]]) == 0
               and (float(r[i["revenue"]]) != 0
                    or float(r[i["avg_discount_pct"]]) != 0)))
    report("panel", "selling weeks carry revenue",
           sum(1 for r in rows if int(r[i["units_sold"]]) > 0
               and float(r[i["revenue"]]) <= 0))
    prices = sorted(float(r[i["avg_unit_price"]]) for r in rows
                    if int(r[i["units_sold"]]) > 0)
    print(f"  [ok  ] avg_unit_price p50/p99/max{'':<21} "
          f"{prices[len(prices)//2]:.2f} / {prices[int(len(prices)*.99)]:.2f} / "
          f"{prices[-1]:,.2f}")


def main():
    print("=" * 78)
    print("RETAIL PULSE AI - DATA QUALITY AUDIT")
    print("=" * 78)

    s_head, s_rows = read(SALES)
    print(f"\nSALES  {SALES.name}: {len(s_rows):,} rows x {len(s_head)} cols")

    # promotion='None' is a deliberate "not promoted" sentinel, not a missing value
    exempt = {("promotion", "None")}
    check_missing("sales", s_head, s_rows, exempt)
    sent = sum(1 for r in s_rows if r[s_head.index("promotion")] == "None")
    print(f"  [note] promotion='None' on {sent:,} rows ({sent/len(s_rows):.1%}) - a "
          f"deliberate sentinel,\n         NOT missing. Use "
          f"df.promotion.replace('None', pd.NA) if your tool reads it as null.")

    check_text("sales", s_head, s_rows,
               ["product_name", "customer_id", "store_id", "product_id",
                "product_category", "city", "region"])
    check_duplicates_sales(s_head, s_rows)
    check_dates_sales(s_head, s_rows, dt.date(2025, 12, 31))
    check_numbers_sales(s_head, s_rows)

    p_head, p_rows = read(PANEL)
    print("\n" + "=" * 78)
    print(f"PANEL  {PANEL.name}: {len(p_rows):,} rows x {len(p_head)} cols")
    print("=" * 78)
    check_missing("panel", p_head, p_rows)
    check_duplicates_panel(p_head, p_rows)
    check_dates_panel(p_head, p_rows, dt.date(2025, 12, 31))
    check_numbers_panel(p_head, p_rows)

    print("\n" + "=" * 78)
    if findings:
        print(f"AUDIT FAILED: {len(findings)} finding(s)")
        for fname, check, detail in findings:
            print(f"  - [{fname}] {check}: {detail}")
        return 1
    print("AUDIT PASSED: no missing values, no duplicate rows, all dates well formed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())