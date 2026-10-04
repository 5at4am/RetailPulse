"""Reconcile the weekly panel against the sales file.

The panel is not derived from the sales file -- the generator builds both
independently. So a user who compares them will find they disagree, and needs to
know exactly WHY. This script localises every unit of the difference.
"""
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "raw"
WEEK1 = dt.date(2024, 1, 1)
LAST_WEEK = dt.date(2025, 12, 22)
LAST_WEEK_END = LAST_WEEK + dt.timedelta(days=6)   # 2025-12-28

sales_units = defaultdict(int)
sales_rev = defaultdict(float)
panel_units = defaultdict(int)
panel_rev = defaultdict(float)
all_prods, panel_prods, panel_stores = set(), set(), set()
n_sales_rows = n_panel_rows = 0

with (DATA / "retail_pulse_sales.csv").open(newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        d = dt.date.fromisoformat(r["date"])
        all_prods.add(r["product_id"])
        sales_units[d] += int(r["quantity_sold"])
        sales_rev[d] += float(r["sales_amount"])
        n_sales_rows += 1

with (DATA / "retail_pulse_demand_panel.csv").open(newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        d = dt.date.fromisoformat(r["week_start_date"])
        panel_prods.add(r["product_id"])
        panel_stores.add(r["store_id"])
        panel_units[d] += int(r["units_sold"])
        panel_rev[d] += float(r["revenue"])
        n_panel_rows += 1

su, sr = sum(sales_units.values()), sum(sales_rev.values())
pu, pr = sum(panel_units.values()), sum(panel_rev.values())

print("=" * 74)
print("PANEL vs SALES RECONCILIATION")
print("=" * 74)
print(f"sales rows {n_sales_rows:,}   units {su:>12,}   revenue INR {sr:>15,.0f}")
print(f"panel rows {n_panel_rows:,}   units {pu:>12,}   revenue INR {pr:>15,.0f}")
print(f"{'gap':>28} units {su - pu:>12,}   revenue INR {sr - pr:>15,.0f}")

print("\n--- why they differ: scope ---")
print(f"products   sales {len(all_prods):>5}   panel {len(panel_prods):>5}"
      f"   panel covers {len(panel_prods)/len(all_prods):.1%} of catalogue")
print(f"stores     sales {50:>5}   panel {len(panel_stores):>5}   (all 50)")

# sales restricted to the panel's product set and date window
scope_units = scope_rev = 0
only_scope = only_window = both = 0
for r in csv.DictReader((DATA / "retail_pulse_sales.csv").open(newline="", encoding="utf-8")):
    d = dt.date.fromisoformat(r["date"])
    q = int(r["quantity_sold"])
    v = float(r["sales_amount"])
    in_scope = r["product_id"] in panel_prods
    in_window = d <= LAST_WEEK_END
    if in_scope and in_window:
        scope_units += q
        scope_rev += v
    else:
        # mutually exclusive buckets, so they sum to the total gap
        if in_scope and not in_window:
            only_window += q
        elif in_window and not in_scope:
            only_scope += q
        else:
            both += q

print(f"\nsales restricted to panel product set + 2024-01-01..{LAST_WEEK_END}:")
print(f"    units {scope_units:>12,}   revenue INR {scope_rev:>15,.0f}")
print(f"panel:")
print(f"    units {pu:>12,}   revenue INR {pr:>15,.0f}")
print(f"residual (panel vs restricted sales):")
print(f"    units {pu - scope_units:>12,}   revenue INR {pr - scope_rev:>15,.0f}"
      f"   ({abs(pu - scope_units)/scope_units:.2%} of scoped sales)")

print("\n--- units excluded from the panel, and why (mutually exclusive) ---")
print(f"product outside panel assortment, in window : {only_scope:>12,}")
print(f"in panel assortment, but after {LAST_WEEK_END}    : {only_window:>12,}")
print(f"both reasons                               : {both:>12,}")
print(f"total excluded                             : "
      f"{only_scope + only_window + both:>12,}"
      f"   (sales total {su:,})")

print("\n--- week alignment ---")
wk_sales = sum(sales_units[d] for d in sales_units
               if WEEK1 <= d <= LAST_WEEK_END and (d - WEEK1).days // 7 < 104)
print(f"sales units inside the 104-week window: {wk_sales:,}")
print(f"panel units                             : {pu:,}")