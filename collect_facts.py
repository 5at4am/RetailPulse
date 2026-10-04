"""Collect every number that will appear in the manager PDF, straight from the CSVs.

Nothing in the document is typed from memory -- it is all computed here and dumped
to facts.json, so the document can be checked against the data afterwards.
"""
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "raw"
# facts.json describes the raw files but stays at the project root, next to the scripts
# that verify against it (and next to methodology/verify_pdf.py, which reads it).
ROOT = Path(__file__).resolve().parent
out = {}

# ---------------------------------------------------------------- sales
units = revenue = 0
rowcount = 0
txns, custs, prods, stores, cats, cities = set(), set(), set(), set(), set(), set()
dates, stockouts = set(), 0
lines = Counter()
channels = Counter()
promos = Counter()
multi = 0
txn_lines = defaultdict(int)
par = defaultdict(float)
par_by = defaultdict(float)

with (DATA / "retail_pulse_sales.csv").open(newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rowcount += 1
        q = int(r["quantity_sold"])
        units += q
        revenue += float(r["sales_amount"])
        txns.add(r["transaction_id"])
        custs.add(r["customer_id"])
        prods.add(r["product_id"])
        stores.add(r["store_id"])
        cats.add(r["product_category"])
        cities.add(r["city"])
        dates.add(r["date"])
        if q == 0:
            stockouts += 1
        lines[r["transaction_id"]] += 1
        channels[r["channel"]] += 1
        if r["is_promotion"] == "1":
            promos[r["promotion"]] += 1
        v = float(r["sales_amount"])
        par[r["product_id"]] += v
        par_by[r["product_id"]] += v

basket_sizes = list(lines.values())
multi = sum(1 for v in basket_sizes if v > 1)
single = sum(1 for v in basket_sizes if v == 1)

# revenue concentration (Pareto): how few SKUs generate X% of revenue?
# i.e. the share of the CATALOG needed -- NOT the revenue share of the top X%
# of the catalog, which is a different and much less informative number.
skus = sorted(par, key=lambda k: -par[k])
tot = sum(par.values())
def catalog_share_for(rev_frac):
    """% of SKUs whose cumulative revenue reaches rev_frac of total."""
    need, run = 0, 0.0
    for s in skus:
        run += par[s]
        need += 1
        if run >= tot * rev_frac:
            break
    return 100.0 * need / len(skus)

out["sales"] = {
    "rows": rowcount,
    "columns": 27,
    "units": units,
    "revenue": round(revenue, 2),
    "transactions": len(txns),
    "customers": len(custs),
    "products": len(prods),
    "stores": len(stores),
    "categories": len(cats),
    "cities": len(cities),
    "date_from": min(dates),
    "date_to": max(dates),
    "days": len(dates),
    "stockout_rows": stockouts,
    # denominator is the ROW COUNT -- never the unit total
    "stockout_pct": round(100 * stockouts / rowcount, 2),
    "selling_rows": rowcount - stockouts,
    "avg_units_per_line": round(units / rowcount, 2),
    "avg_line_value": round(revenue / rowcount, 2),
    "multi_line_txn_pct": round(100 * multi / len(txns), 1),
    "single_line_txn_pct": round(100 * single / len(txns), 1),
    "basket_median": sorted(basket_sizes)[len(basket_sizes) // 2],
    "basket_max": max(basket_sizes),
    "channels": dict(channels),
    "promoted_lines": sum(promos.values()),
    "promotions": dict(promos),
    "pareto50": round(catalog_share_for(0.50), 1),
    "pareto80": round(catalog_share_for(0.80), 1),
    "pareto90": round(catalog_share_for(0.90), 1),
}

# repeat customers = customers with >1 transaction
per_cust = defaultdict(set)
with (DATA / "retail_pulse_sales.csv").open(newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        per_cust[r["customer_id"]].add(r["transaction_id"])
rep = sum(1 for v in per_cust.values() if len(v) > 1)
out["sales"]["repeat_customer_pct"] = round(100 * rep / len(custs), 1)

# ---------------------------------------------------------------- panel
punits = prev = 0
pweeks = set()
pprods, pstores = set(), set()
zeros = 0
prows = 0
pstock = 0
stockout_weeks = 0
with (DATA / "retail_pulse_demand_panel.csv").open(newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        prows += 1
        punits += int(r["units_sold"])
        prev += float(r["revenue"])
        pweeks.add(r["week_start_date"])
        pprods.add(r["product_id"])
        pstores.add(r["store_id"])
        sc = int(r["stockout_count"])
        pstock += sc
        if sc > 0:
            stockout_weeks += 1
        if int(r["units_sold"]) == 0:
            zeros += 1

out["panel"] = {
    "rows": prows,
    "columns": 15,
    "units": punits,
    "revenue": round(prev, 2),
    "weeks": len(pweeks),
    "week_from": min(pweeks),
    "week_to": max(pweeks),
    "products": len(pprods),
    "stores": len(pstores),
    "zero_demand_pct": round(100 * zeros / prows, 1),
    "stockout_week_rows": stockout_weeks,
    "stockout_count_total": pstock,
    "grain": "store x product x week",
}

# ---------------------------------------------------------------- files
for name in ("retail_pulse_sales.csv", "retail_pulse_demand_panel.csv",
             "retail_pulse_data_dictionary.md"):
    p = DATA / name
    out.setdefault("files", {})[name] = {"mb": round(p.stat().st_size / 1e6, 1)}

(ROOT / "facts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))