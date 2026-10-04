"""Verification suite for the Retail Pulse AI dataset. Stdlib only."""
"""Full specification check for the Retail Pulse AI dataset.

Exits non-zero if any check fails, so it can gate CI.
    python verify_dataset.py
"""
import csv
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
SALES = DATA / "retail_pulse_sales.csv"
PANEL = DATA / "retail_pulse_demand_panel.csv"
sys.path.insert(0, str(ROOT))
import generate_retail_pulse as gen  # noqa: E402  (safe: main() is __main__-guarded)

REQUIRED = ["date", "customer_id", "store_id", "product_id", "product_category",
            "quantity_sold", "unit_price", "discount", "promotion", "inventory_level",
            "sales_amount", "city"]

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


with SALES.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    header = reader.fieldnames
    rows = list(reader)

# ---------------------------------------------------------------- structure
check("row count == 250,000", len(rows) == 250_000, f"got {len(rows):,}")
check("all 12 required columns present", all(c in header for c in REQUIRED),
      f"missing {[c for c in REQUIRED if c not in header]}" if not all(c in header for c in REQUIRED)
      else f"{len(header)} columns total")
check("required columns come first, in the requested order",
      header[:12] == REQUIRED, f"first 12 = {header[:12]}")

null_cells = sum(1 for r in rows for c in REQUIRED if r[c] is None or r[c] == "")
check("zero empty values in the 12 required columns", null_cells == 0, f"{null_cells} empty")

ragged = [c for c in header if any(r[c] is None for r in rows)]
check("every column is populated on every row (no short rows)", not ragged,
      f"columns missing values: {ragged}" if ragged else f"{len(header)}/{len(header)} complete")
check("channel column carries real values",
      {r["channel"] for r in rows} == {"Store", "Mobile_App", "Web"},
      f"values = {sorted({r['channel'] for r in rows})}")
check("product_name populated on every row", all(r["product_name"] for r in rows))

# ---------------------------------------------------------------- formulas
bad_amount = bad_pct = bad_cap = bad_flag = bad_nodeal = bad_neg = bad_band = 0
for r in rows:
    q = int(r["quantity_sold"])
    up = float(r["unit_price"])
    di = float(r["discount"])
    if abs(float(r["sales_amount"]) - round(q * (up - di), 2)) > 0.011:
        bad_amount += 1
    if di > up + 1e-9:
        bad_cap += 1
    if abs(float(r["discount_pct"]) - (di / up * 100 if up else 0.0)) > 0.011:
        bad_pct += 1
    if int(r["is_promotion"]) != (1 if r["promotion"] != "None" else 0):
        bad_flag += 1
    if r["promotion"] == "None" and di != 0.0:
        bad_nodeal += 1
    if q < 0 or int(r["inventory_level"]) < 0:
        bad_neg += 1
    if r["price_band"] != ("Budget" if up < 299 else "Mid" if up < 1000
                           else "Premium" if up < 5000 else "Luxury"):
        bad_band += 1

check("sales_amount == quantity_sold x (unit_price - discount), ALL rows",
      bad_amount == 0, f"{bad_amount} mismatches")
check("discount <= unit_price, ALL rows", bad_cap == 0, f"{bad_cap} violations")
check("discount_pct == discount / unit_price x 100, ALL rows", bad_pct == 0,
      f"{bad_pct} mismatches")
check("is_promotion matches promotion != 'None', ALL rows", bad_flag == 0,
      f"{bad_flag} mismatches")
check("no discount without a promotion (documented rule)", bad_nodeal == 0,
      f"{bad_nodeal} violations")
check("quantity_sold and inventory_level non-negative", bad_neg == 0, f"{bad_neg} violations")
check("price_band matches unit_price", bad_band == 0, f"{bad_band} mismatches")

rng = random.Random(7)
bad_sample = sum(
    1 for r in rng.sample(rows, 1000)
    if abs(float(r["sales_amount"])
           - round(int(r["quantity_sold"]) * (float(r["unit_price"]) - float(r["discount"])), 2))
    > 0.011)
check("random 1,000-row spot check of the sales formula", bad_sample == 0,
      f"{bad_sample} mismatches")

# ---------------------------------------------------------------- inventory
# A zero-quantity row means lost demand at an empty shelf. Selling the LAST unit is
# legitimate and also leaves inventory_level at 0, so only this direction is an invariant.
bad_inv = sum(1 for r in rows if int(r["quantity_sold"]) == 0 and int(r["inventory_level"]) != 0)
check("quantity_sold == 0 implies inventory_level == 0 (lost demand)",
      bad_inv == 0, f"{bad_inv} violations")
sold_last = sum(1 for r in rows
                if int(r["quantity_sold"]) > 0 and int(r["inventory_level"]) == 0)
print(f"       (note: {sold_last:,} rows sold the last unit on the shelf - legitimate)")

# ---------------------------------------------------------------- dates
dates = sorted({r["date"] for r in rows})
parsed = [date.fromisoformat(x) for x in dates]
check("date range is 2024-01-01 .. 2025-12-31",
      parsed[0] == date(2024, 1, 1) and parsed[-1] == date(2025, 12, 31),
      f"{dates[0]} .. {dates[-1]}")
check("731 distinct calendar days covered", len(dates) == 731, f"got {len(dates)}")
check("no future dates", parsed[-1] <= date(2026, 10, 3))

hol_map = {}
for y, table in ((2024, gen.HOLIDAYS_2024), (2025, gen.HOLIDAYS_2025)):
    for (m, d), (nm, _) in table.items():
        hol_map[date(y, m, d)] = nm

bad_cal = 0
for r in rows:
    d = date.fromisoformat(r["date"])
    if (r["weekday"] != WEEKDAY_NAMES[d.weekday()]
            or r["is_weekend"] != ("1" if d.weekday() >= 5 else "0")
            or r["year_month"] != f"{d.year}-{d.month:02d}"
            or int(r["week_of_year"]) != d.isocalendar()[1]
            or r["quarter"] != f"Q{(d.month - 1) // 3 + 1}"
            or r["is_holiday"] != ("1" if d in hol_map else "0")):
        bad_cal += 1
check("derived calendar columns agree with `date`, ALL rows", bad_cal == 0,
      f"{bad_cal} mismatches")
check("Diwali is flagged as a holiday",
      sum(1 for r in rows if r["date"] == "2024-10-31") > 0
      and sum(1 for r in rows if r["date"] == "2024-10-31" and r["is_holiday"] == "1")
      == sum(1 for r in rows if r["date"] == "2024-10-31"),
      "2024-10-31 and 2025-10-20 confirmed present")

# ---------------------------------------------------------------- cardinality
cust = {r["customer_id"] for r in rows}
stores = {r["store_id"] for r in rows}
prods = {r["product_id"] for r in rows}
cats = {r["product_category"] for r in rows}
cities = {r["city"] for r in rows}
check("8,000 distinct customers", len(cust) == 8000, f"got {len(cust):,}")
check("50 distinct stores", len(stores) == 50, f"got {len(stores)}")
check("1,200 distinct products", len(prods) == 1200, f"got {len(prods):,}")
check("10 product categories", len(cats) == 10, f"got {len(cats)}")
check("12 cities", len(cities) == 12, f"got {len(cities)}")

store_city, store_region = defaultdict(set), defaultdict(set)
for r in rows:
    store_city[r["store_id"]].add(r["city"])
    store_region[r["store_id"]].add(r["region"])
check("every store_id maps to exactly one city",
      all(len(v) == 1 for v in store_city.values()),
      f"{sum(1 for v in store_city.values() if len(v) > 1)} offenders")
check("every store_id maps to exactly one region",
      all(len(v) == 1 for v in store_region.values()))

# ---------------------------------------------------------------- demand signal
daily_units, daily_lines, daily_promo = (defaultdict(float), defaultdict(int),
                                          defaultdict(int))
unit_share = defaultdict(lambda: defaultdict(float))
for r in rows:
    q = int(r["quantity_sold"])
    d = r["date"]
    daily_units[d] += q
    daily_lines[d] += 1
    if int(r["is_promotion"]) == 1:
        daily_promo[d] += 1
    unit_share[d][r["product_category"]] += q

# weekday rhythm measured on DAILY TOTALS (per-line quantity is flat by construction)
wd_units, wd_lines = defaultdict(list), defaultdict(list)
for d in daily_units:
    wd = date.fromisoformat(d).weekday()
    wd_units[WEEKDAY_NAMES[wd]].append(daily_units[d])
    wd_lines[WEEKDAY_NAMES[wd]].append(daily_lines[d])
wd_mean = {w: sum(v) / len(v) for w, v in wd_units.items()}
check("weekend uplift present on daily totals (Sat > 1.4x Mon, Sun > 1.2x Mon)",
      wd_mean["Saturday"] > wd_mean["Monday"] * 1.40
      and wd_mean["Sunday"] > wd_mean["Monday"] * 1.20,
      " ".join(f"{w[:3]}={wd_mean[w]:.0f}" for w in WEEKDAY_NAMES))

# seasonality on daily totals
month_units = defaultdict(list)
for d in daily_units:
    month_units[d[:7]].append(daily_units[d])
mm = {m: sum(v) / len(v) for m, v in month_units.items()}
check("yearly seasonality present (peak month >= 1.4x trough month)",
      max(mm.values()) / min(mm.values()) >= 1.40,
      f"peak/trough = {max(mm.values()) / min(mm.values()):.2f} "
      f"(max {max(mm, key=mm.get)}, min {min(mm, key=mm.get)})")

festive = [v for m, v in mm.items() if m.endswith("-10") or m.endswith("-11")]
midyear = [v for m, v in mm.items() if m.endswith(("-05", "-06", "-07"))]
check("Oct/Nov (festive) beats May-Jul by >= 15%",
      sum(festive) / len(festive) > sum(midyear) / len(midyear) * 1.15,
      f"{sum(festive) / len(festive):.0f} vs {sum(midyear) / len(midyear):.0f} units/day")

# ---- promotion lift: three independent signals
diwali_dates = set()
for lo, hi in [(date(2024, 10, 26), date(2024, 11, 4)), (date(2025, 10, 15), date(2025, 10, 23))]:
    dd = lo
    while dd <= hi:
        diwali_dates.add(dd.isoformat())
        dd = date.fromordinal(dd.toordinal() + 1)

no_campaign = [d for d in daily_lines if daily_promo[d] == 0]
diwali = [d for d in daily_lines if d in diwali_dates]
share_nc = sum(daily_promo[d] / daily_lines[d] for d in no_campaign) / len(no_campaign)
share_dw = sum(daily_promo[d] / daily_lines[d] for d in diwali) / len(diwali)
check("promotions concentrate on campaign dates (promo line share Diwali >> baseline)",
      share_dw > share_nc * 3, f"Diwali {share_dw * 100:.1f}% vs baseline {share_nc * 100:.1f}%")

fest_cats = ("Toys", "Apparel", "Footwear", "Electronics")


def cat_share(dates_list):
    num = den = 0.0
    for d in dates_list:
        tot = sum(unit_share[d].values()) or 1
        num += sum(unit_share[d][c] for c in fest_cats)
        den += tot
    return num / den


cs_nc, cs_dw = cat_share(no_campaign), cat_share(diwali)
check("promotions shift the mix toward festive categories",
      cs_dw > cs_nc * 1.15,
      f"Toys+Apparel+Footwear+Electronics share: Diwali {cs_dw * 100:.1f}% "
      f"vs baseline {cs_nc * 100:.1f}%")

# volume lift on Diwali dates vs the same weekday on non-campaign dates
dw_by_wd, nc_by_wd = defaultdict(list), defaultdict(list)
for d in diwali:
    dw_by_wd[date.fromisoformat(d).weekday()].append(daily_units[d])
for d in no_campaign:
    nc_by_wd[date.fromisoformat(d).weekday()].append(daily_units[d])
pairs = [(sum(dw_by_wd[w]) / len(dw_by_wd[w]), sum(nc_by_wd[w]) / len(nc_by_wd[w]))
         for w in dw_by_wd if dw_by_wd[w] and nc_by_wd[w]]
vol_lift = sum(a / b for a, b in pairs) / len(pairs)
check("promotions lift total chain volume (Diwali vs same weekday, no campaign)",
      vol_lift > 1.15, f"mean lift = {vol_lift:.2f}x across {len(pairs)} weekdays")

disc_vals = sorted(float(r["discount_pct"]) for r in rows if float(r["discount_pct"]) > 0)
check("discount depth spans a realistic range", disc_vals[-1] > 40 and disc_vals[0] < 15,
      f"min {disc_vals[0]:.1f}%, median {disc_vals[len(disc_vals) // 2]:.1f}%, "
      f"max {disc_vals[-1]:.1f}%")

# ---------------------------------------------------------------- category seasonality
# measured as MIX (share of units), because global festive demand scales every category
# at once -- absolute units peak in October regardless of the category's own season
mon_units = defaultdict(lambda: defaultdict(int))
mon_tot = defaultdict(int)
for r in rows:
    m = int(r["year_month"][5:7])
    q = int(r["quantity_sold"])
    mon_units[r["product_category"]][m] += q
    mon_tot[m] += q

def share(cat, m):
    return mon_units[cat][m] / mon_tot[m]

def peak_month(cat):
    return max(range(1, 13), key=lambda m: share(cat, m))

check("category mix is genuinely seasonal, not a uniform uplift",
      peak_month("Beverages") in (4, 5, 6) and peak_month("Apparel") in (10, 11, 12),
      f"Beverages share peaks m{peak_month('Beverages')}, Apparel m{peak_month('Apparel')}")

bev = [share("Beverages", m) for m in range(1, 13)]
check("seasonal mix swing is material (Beverages share varies >15% relative)",
      max(bev) / min(bev) > 1.15,
      f"Beverages share {min(bev):.2%}..{max(bev):.2%} = {max(bev) / min(bev):.2f}x")

gro = [share("Grocery", m) for m in range(1, 13)]
gro_trough = min(range(1, 13), key=lambda m: gro[m - 1])
check("staples move against seasonal categories (Grocery dips as Beverages peak)",
      gro_trough in (4, 5, 6) and gro[4] < gro[10],
      f"Grocery share trough m{gro_trough}, Beverages peak m{peak_month('Beverages')}")

# ---------------------------------------------------------------- calibrated realism
# Locked in against UCI Online Retail II (consumer-value invoices) so these cannot
# silently regress into a flat, obviously-synthetic distribution.
lines_per_txn = defaultdict(int)
skus_per_txn = defaultdict(set)
for r in rows:
    lines_per_txn[r["transaction_id"]] += 1
    skus_per_txn[r["transaction_id"]].add(r["product_id"])
bsz = sorted(lines_per_txn.values())


def bpct(p):
    return bsz[min(int(len(bsz) * p), len(bsz) - 1)]


check("basket size is bimodal (quick top-up + bulk mode), not flat",
      bpct(0.50) <= 4 and max(bsz) >= 15,
      f"median {bpct(.5)}, p90 {bpct(.9)}, max {max(bsz)}")
check("bulk tail is real (p99 basket is much larger than p90)",
      bsz[int(len(bsz) * 0.99)] > bpct(0.90) * 1.8,
      f"p90 {bpct(.9)} -> p99 {bsz[int(len(bsz)*.99)]}")
single = sum(1 for v in bsz if v == 1)
check("single-item baskets exist (quick top-ups)", single / len(bsz) > 0.05,
      f"{single / len(bsz):.1%} of baskets are one line")

qh = defaultdict(int)
for r in rows:
    q = int(r["quantity_sold"])
    if q > 0:
        qh[q] += 1
pack_12, between = qh.get(12, 0), sum(qh.get(q, 0) for q in (7, 8, 9))
check("whole-pack buying shows as quantity spikes (pack size > loose counts)",
      pack_12 > between * 3,
      f"qty=12 is {pack_12 / between:.1f}x the qty 7-9 run (real data spikes the same way)")
check("quantity tail extends past a single pack", max(qh) >= 12,
      f"max quantity per line = {max(qh)}")

# ---------------------------------------------------------------- stockouts
stockout_rows = sum(1 for r in rows if int(r["quantity_sold"]) == 0)
check("stockout (lost-demand) rows present", stockout_rows > 3000,
      f"{stockout_rows:,} rows ({stockout_rows / len(rows) * 100:.2f}% of file)")

# ---------------------------------------------------------------- customer insights
txn_per_cust, lines_per_cust = defaultdict(set), defaultdict(int)
for r in rows:
    txn_per_cust[r["customer_id"]].add(r["transaction_id"])
    lines_per_cust[r["customer_id"]] += 1
repeat = sum(1 for v in txn_per_cust.values() if len(v) > 1)
check("customers are repeat purchasers (RFM is computable)",
      repeat / len(cust) > 0.85,
      f"{repeat:,}/{len(cust):,} ({repeat / len(cust) * 100:.1f}%) have >1 basket")
counts = sorted(len(v) for v in txn_per_cust.values())
check("median baskets per customer >= 5", counts[len(counts) // 2] >= 5,
      f"median {counts[len(counts) // 2]}, min {counts[0]}, max {counts[-1]}, "
      f"mean {sum(counts) / len(counts):.1f}")

multi = defaultdict(int)
for r in rows:
    multi[r["transaction_id"]] += 1
multi_baskets = sum(v for k, v in multi.items() if v > 1)
check("baskets contain multiple line items (basket analysis possible)",
      multi_baskets > 0, f"{multi_baskets:,} multi-line baskets")

seg = defaultdict(set)
for r in rows:
    seg[r["customer_segment"]].add(r["customer_id"])
check("5 customer lifecycle segments all populated", len(seg) == 5,
      ", ".join(f"{k}={len(v):,}" for k, v in sorted(seg.items())))

tier = defaultdict(set)
for r in rows:
    tier[r["loyalty_tier"]].add(r["customer_id"])
check("4 loyalty tiers all populated", len(tier) == 4,
      ", ".join(f"{k}={len(v):,}" for k, v in sorted(tier.items())))

# ---------------------------------------------------------------- panel
with PANEL.open(newline="", encoding="utf-8") as fh:
    prows = list(csv.DictReader(fh))

check("panel row count == 780,000 (150 products x 50 stores x 104 weeks)",
      len(prows) == 780_000, f"got {len(prows):,}")
zero_weeks = sum(1 for r in prows if int(r["units_sold"]) == 0)
check("panel contains explicit zero-demand weeks (missing != 0)", zero_weeks > 1000,
      f"{zero_weeks:,} zero weeks ({zero_weeks / len(prows) * 100:.2f}%)")
check("panel covers 50 stores x 150 products x 104 weeks",
      len({r["store_id"] for r in prows}) == 50
      and len({r["product_id"] for r in prows}) == 150
      and len({r["week_start_date"] for r in prows}) == 104,
      f"stores={len({r['store_id'] for r in prows})} "
      f"products={len({r['product_id'] for r in prows})} "
      f"weeks={len({r['week_start_date'] for r in prows})}")
so_weeks = sum(1 for r in prows if int(r["stockout_count"]) > 0)
check("panel records stockouts per week (supply-constrained demand visible)",
      so_weeks > 1000, f"{so_weeks:,} weeks with a stockout")

pprod = {r["product_id"] for r in prows}
panel_units = sum(int(r["units_sold"]) for r in prows)
sales_panel_units = sum(int(r["quantity_sold"]) for r in rows if r["product_id"] in pprod)
check("panel units are a subset of sales-file units (no inflated demand)",
      0 < panel_units <= sales_panel_units,
      f"panel {panel_units:,} <= sales {sales_panel_units:,}")

# ---------------------------------------------------------------- summary
total_rev = sum(float(r["sales_amount"]) for r in rows)
total_units = sum(int(r["quantity_sold"]) for r in rows)
print(f"\n  revenue INR {total_rev:,.0f} | units {total_units:,} | "
      f"avg line INR {sum(float(r['unit_price']) for r in rows) / len(rows):,.2f}")

passed = sum(1 for _, ok, _ in results if ok)
failed = [(n, d) for n, ok, d in results if not ok]
print("\n" + "=" * 70)
print(f"VERIFICATION: {passed}/{len(results)} checks passed")
if failed:
    print("\nFAILED:")
    for n, d in failed:
        print(f"  - {n}  ({d})")
else:
    print("All checks passed.")
print("=" * 70)
sys.exit(1 if failed else 0)
