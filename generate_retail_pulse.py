"""
Retail Pulse AI - synthetic retail sales dataset generator.

Deterministic: the same SEED always produces byte-identical output.

Outputs (written next to this script):
  1. retail_pulse_sales.csv           250,000 rows x 27 columns  (transaction line-items)
  2. retail_pulse_demand_panel.csv    780,000 rows x 15 columns (dense weekly demand panel)
  3. retail_pulse_data_dictionary.md  column-by-column reference

Run:  python generate_retail_pulse.py
"""

from __future__ import annotations

import csv
import math
import random
from array import array
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from itertools import accumulate
from pathlib import Path

# --------------------------------------------------------------------------- config

SEED = 20240101
OUT_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2024, 1, 1)          # a Monday
N_DAYS = 731                           # 2024 is a leap year: 366 + 365 days
N_WEEKS = N_DAYS // 7                  # 104 complete Monday-start weeks

N_PRODUCTS = 1200
N_STORES = 50
N_CUSTOMERS = 8000
TARGET_ROWS = 250_000
STOCKOUT_ROWS = 7_500                  # ~3% of rows: demand that hit zero stock
TX_ROWS = TARGET_ROWS - STOCKOUT_ROWS  # 242,500 completed transactions
# a customer's guaranteed first basket re-uses a slot in the day's quota rather than
# adding a row, so the quota target stays TX_ROWS
PANEL_PRODUCTS = 150                   # top-velocity products tracked in the dense panel
VELOCITY_SIGMA = 0.62                  # catalog long-tail spread
PACK_SHARE = 0.22                      # share of lines bought as a whole pack


# --------------------------------------------------------------------------- holidays
# Dates confirmed against published Drik Panchang / press calendars.
# name -> demand multiplier applied on that single day.

HOLIDAYS_2024 = {
    (1, 1): ("New_Year_Day", 1.30),
    (1, 26): ("Republic_Day", 1.35),
    (2, 14): ("Valentines_Day", 1.12),
    (3, 8): ("Womens_Day", 1.10),
    (3, 25): ("Holi", 1.18),
    (4, 14): ("Ambedkar_Jayanti", 1.08),
    (5, 1): ("Labour_Day", 1.20),
    (8, 15): ("Independence_Day", 1.35),
    (9, 7): ("Ganesh_Chaturthi", 1.15),
    (10, 2): ("Gandhi_Jayanti", 1.07),
    (10, 20): ("Dussehra", 1.45),
    (10, 29): ("Dhanteras", 1.60),
    (10, 31): ("Diwali", 2.10),
    (11, 2): ("Govardhan_Puja", 1.20),
    (11, 3): ("Bhai_Dooj", 1.10),
    (11, 14): ("Childrens_Day", 1.12),
    (12, 24): ("Christmas_Eve", 1.55),
    (12, 25): ("Christmas", 2.00),
    (12, 31): ("New_Year_Eve", 1.60),
}

HOLIDAYS_2025 = {
    (1, 1): ("New_Year_Day", 1.30),
    (1, 26): ("Republic_Day", 1.35),
    (2, 14): ("Valentines_Day", 1.12),
    (3, 8): ("Womens_Day", 1.10),
    (3, 14): ("Holi", 1.18),
    (4, 14): ("Ambedkar_Jayanti", 1.08),
    (5, 1): ("Labour_Day", 1.20),
    (8, 15): ("Independence_Day", 1.35),
    (8, 27): ("Ganesh_Chaturthi", 1.15),
    (10, 2): ("Dussehra", 1.45),
    (10, 18): ("Dhanteras", 1.60),
    (10, 20): ("Diwali", 2.10),
    (10, 23): ("Bhai_Dooj", 1.10),
    (11, 5): ("Childrens_Day", 1.12),
    (12, 24): ("Christmas_Eve", 1.55),
    (12, 25): ("Christmas", 2.00),
    (12, 31): ("New_Year_Eve", 1.60),
}


# --------------------------------------------------------------------------- products

BRANDS = [
    "Nova", "Aster", "Verve", "Lumina", "Trident", "Maple", "Zenith", "Cobalt",
    "Saffron", "Indigo", "Marigold", "Quartz", "Onyx", "Cedar", "Basalt", "Ivory",
    "Ember", "Coral", "Nimbus", "Halcyon", "Pinnacle", "Rivera", "Solstice", "Tundra",
    "Vertex", "Willow", "Yarrow", "Zephyr", "Amber", "Beryl", "Cypress", "Dune",
]

# season = 12 monthly demand multipliers (Jan..Dec)
# qty    = (min, max) units per line-item
CATEGORIES = [
    dict(
        name="Grocery", share=0.18, price=(30, 350), qty=(1, 6), velocity=1.55, packs=[2, 5, 6, 10, 12, 20, 25],
        season=[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.20, 1.25, 1.15],
        items=["Shahi Atta", "Sona Masoori Rice", "Toor Dal", "Chana Dal", "Masoor Dal",
               "Sugar", "Rock Salt", "Sunflower Oil", "Mustard Oil", "Groundnut Oil",
               "Tea Powder", "Instant Coffee", "Turmeric Powder", "Red Chilli Powder",
               "Coriander Powder", "Besan", "Basmati Rice", "Oats", "Honey", "Jaggery"],
        specs=["500 g", "1 kg", "2 kg", "5 kg", "10 kg", "250 g", "750 ml", "1 L"],
    ),
    dict(
        name="Beverages", share=0.13, price=(25, 400), qty=(1, 6), velocity=1.40, packs=[6, 12, 24],
        season=[0.90, 0.90, 1.00, 1.15, 1.25, 1.20, 1.10, 1.05, 1.00, 1.00, 0.95, 0.95],
        items=["Orange Juice", "Mango Juice", "Green Tea", "Lemon Tea", "Cola",
               "Mineral Water", "Instant Coffee", "Energy Drink", "Lassi",
               "Cold Drink", "Apple Cider", "Soya Milk", "Toned Milk", "Buttermilk"],
        specs=["200 ml", "500 ml", "1 L", "1.5 L", "250 ml", "6 x 250 ml"],
    ),
    dict(
        name="Bakery", share=0.08, price=(20, 150), qty=(1, 4), velocity=1.30, packs=[2, 4, 6, 12],
        season=[0.95, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.10, 1.10, 1.10],
        items=["White Bread", "Brown Bread", "Multigrain Bread", "Butter Croissant",
               "Chocolate Bun", "Rusk", "Digestive Biscuit", "Butter Cookies",
               "Muffins", "Pizza Base", "Puff Pastry", "Garlic Bread"],
        specs=["200 g", "400 g", "6 pcs", "12 pcs"],
    ),
    dict(
        name="Apparel", share=0.14, price=(299, 2999), qty=(1, 2), velocity=0.55, packs=[2, 3, 6],
        season=[0.70, 0.75, 0.85, 1.00, 1.00, 0.95, 0.90, 0.95, 1.10, 1.40, 1.50, 1.20],
        items=["Cotton T-Shirt", "Polo T-Shirt", "Slim Fit Jeans", "Straight Jeans",
               "Cotton Kurta", "Anarkali Kurta", "Formal Shirt", "Casual Shirt",
               "Winter Jacket", "Sweatshirt", "Hoodie", "Midi Dress", "A-Line Skirt"],
        specs=["Small", "Medium", "Large", "XL", "XXL"],
    ),
    dict(
        name="Electronics", share=0.11, price=(199, 24999), qty=(1, 2), velocity=0.42, packs=[2, 3, 5, 10],
        season=[0.85, 0.90, 0.95, 0.95, 0.95, 0.90, 0.90, 0.95, 1.05, 1.35, 1.35, 1.15],
        items=["Wireless Earbuds", "Bluetooth Speaker", "Power Bank", "USB-C Cable",
               "LED Bulb", "Smart Watch", "Wireless Mouse", "Keyboard",
               "Fast Charger", "HDMI Cable", "Webcam", "Desk Microphone"],
        specs=["10 W", "20 W", "30 W", "1 m", "2 m", "v2", "1080p", "Black"],
    ),
    dict(
        name="Personal_Care", share=0.12, price=(99, 1299), qty=(1, 3), velocity=1.25, packs=[2, 3, 4, 6, 12],
        season=[0.95, 0.95, 1.00, 1.05, 1.10, 1.10, 1.05, 1.00, 1.00, 1.05, 1.05, 1.05],
        items=["Anti-Dandruff Shampoo", "Body Wash", "Face Cream", "Face Wash",
               "Toothpaste", "Sandalwood Soap", "Hair Oil", "Deodorant",
               "Sunscreen Gel", "Lip Balm", "Hand Sanitiser", "Conditioner"],
        specs=["100 ml", "200 ml", "400 ml", "50 g", "75 ml", "180 ml"],
    ),
    dict(
        name="Home_Appliances", share=0.08, price=(599, 19999), qty=(1, 1), velocity=0.30, packs=[2, 3],
        season=[0.80, 0.85, 0.95, 1.00, 1.05, 1.05, 1.00, 1.00, 1.05, 1.35, 1.30, 1.10],
        items=["Mixer Grinder", "Electric Kettle", "Induction Cooktop", "Air Fryer",
               "Pop-up Toaster", "Table Fan", "Vacuum Flask", "Steam Iron",
               "Water Purifier", "Hand Blender", "Electric Cooker", "Room Heater"],
        specs=["1.5 L", "500 W", "750 W", "1 L", "1200 W", "Single"],
    ),
    dict(
        name="Stationery", share=0.07, price=(15, 999), qty=(1, 6), velocity=0.95, packs=[6, 12, 24, 50],
        season=[1.10, 1.05, 0.95, 1.15, 1.20, 1.15, 1.00, 1.05, 1.05, 1.00, 1.00, 0.95],
        items=["Notebook", "Ballpoint Pen", "Gel Pen", "Pencil Box", "Stapler",
               "Highlighter", "Geometry Box", "A4 Sheets", "Whiteboard Marker",
               "Eraser", "School Backpack", "File Organiser"],
        specs=["A4", "A5", "100 pages", "200 sheets", "Pack of 10", "Pack of 6"],
    ),
    dict(
        name="Footwear", share=0.05, price=(599, 4999), qty=(1, 2), velocity=0.35, packs=[2, 3, 6],
        season=[0.75, 0.80, 0.90, 1.00, 1.00, 1.00, 0.95, 1.00, 1.10, 1.35, 1.40, 1.20],
        items=["Running Shoes", "Sneakers", "Flip Flops", "Formal Shoes", "Sandals",
               "Sports Sandals", "Slippers", "Loafers", "Block Heels", "Ankle Boots"],
        specs=["UK 5", "UK 6", "UK 7", "UK 8", "UK 9", "UK 10"],
    ),
    dict(
        name="Toys", share=0.04, price=(99, 2499), qty=(1, 2), velocity=0.48, packs=[2, 3, 4, 6, 12],
        season=[0.80, 0.85, 0.95, 1.00, 1.10, 1.05, 1.00, 1.05, 1.05, 1.45, 1.50, 1.20],
        items=["Building Blocks", "Remote Control Car", "Plush Bear", "Puzzle Set",
               "Board Game", "Rubber Ball", "Doll House", "Toy Train Set",
               "Stacking Rings", "Art Kit", "RC Drone", "Slide Game"],
        specs=["50 pcs", "100 pcs", "Ages 3+", "Ages 6+", "Ages 8+"],
    ),
]

CAT_NAMES = [c["name"] for c in CATEGORIES]
N_CAT = len(CATEGORIES)
CAT_IDX = {n: i for i, n in enumerate(CAT_NAMES)}


# --------------------------------------------------------------------------- geo

# (city, tier, region) - 12 cities across 4 regions, 3 tiers
CITIES = [
    ("Mumbai", 1, "West"), ("Delhi NCR", 1, "North"), ("Bengaluru", 1, "South"),
    ("Hyderabad", 1, "South"),
    ("Pune", 2, "West"), ("Ahmedabad", 2, "West"), ("Jaipur", 2, "North"),
    ("Kochi", 2, "South"),
    ("Indore", 3, "Central"), ("Coimbatore", 3, "South"), ("Bhubaneswar", 3, "East"),
    ("Nagpur", 3, "Central"),
]

TIER_PRICE_MULT = {1: 1.18, 2: 1.04, 3: 0.90}
TIER_TRAFFIC_MULT = {1: 1.35, 2: 1.00, 3: 0.68}
STORE_PLAN = [  # city index -> number of stores
    5, 5, 4, 4,          # tier 1
    5, 4, 4, 4,          # tier 2
    4, 4, 4, 3,          # tier 3
]

STORE_TYPES = {
    "Supermarket": dict(size=340, mix={"Grocery": 2.4, "Beverages": 2.0, "Bakery": 1.8,
                                      "Personal_Care": 1.5, "Stationery": 0.8,
                                      "Toys": 0.4, "Home_Appliances": 0.7,
                                      "Apparel": 0.5, "Footwear": 0.4,
                                      "Electronics": 0.5}),
    "Department_Store": dict(size=300, mix={"Apparel": 2.2, "Footwear": 1.8,
                                           "Electronics": 1.5, "Toys": 1.4,
                                           "Home_Appliances": 1.2, "Personal_Care": 1.0,
                                           "Stationery": 0.7, "Bakery": 0.3,
                                           "Grocery": 0.4, "Beverages": 0.5}),
    "Neighbourhood_Store": dict(size=190, mix={"Grocery": 2.0, "Beverages": 1.6,
                                              "Bakery": 1.5, "Personal_Care": 1.4,
                                              "Stationery": 1.1, "Home_Appliances": 0.8,
                                              "Toys": 0.6, "Apparel": 0.6,
                                              "Footwear": 0.5, "Electronics": 0.6}),
}


# --------------------------------------------------------------------------- promotions
# base_lift, discount range, per-category lift multipliers

PROMO_SPECS = {
    "Diwali_Sale": dict(
        base_lift=2.60, disc=(0.30, 0.55),
        cat={"Toys": 1.35, "Apparel": 1.20, "Footwear": 1.18, "Electronics": 1.12,
             "Home_Appliances": 1.08, "Stationery": 0.75, "Grocery": 0.80,
             "Beverages": 0.85, "Bakery": 0.85, "Personal_Care": 0.85},
    ),
    "New_Year_Offer": dict(
        base_lift=2.20, disc=(0.20, 0.40),
        cat={"Electronics": 1.10, "Apparel": 1.10, "Toys": 1.10, "Grocery": 0.85,
             "Beverages": 0.95, "Bakery": 0.95, "Personal_Care": 0.90,
             "Stationery": 0.95, "Footwear": 1.00, "Home_Appliances": 1.00},
    ),
    "Republic_Day_Deal": dict(
        base_lift=1.60, disc=(0.15, 0.30),
        cat={"Apparel": 1.20, "Footwear": 1.15, "Personal_Care": 1.05,
             "Stationery": 0.95, "Grocery": 0.85, "Electronics": 1.00,
             "Toys": 1.00, "Beverages": 0.95, "Bakery": 0.95,
             "Home_Appliances": 1.00},
    ),
    "Weekend_Flash": dict(
        base_lift=1.50, disc=(0.10, 0.25),
        cat={"Bakery": 1.15, "Grocery": 1.10, "Beverages": 1.10, "Stationery": 1.10,
             "Personal_Care": 1.05, "Electronics": 0.90, "Home_Appliances": 0.90,
             "Apparel": 0.95, "Footwear": 0.95, "Toys": 1.00},
    ),
    "Loyalty_Flash": dict(
        base_lift=1.80, disc=(0.15, 0.35),
        cat={"Grocery": 1.10, "Beverages": 1.05, "Bakery": 1.05, "Personal_Care": 1.05,
             "Apparel": 1.00, "Footwear": 1.00, "Electronics": 1.00,
             "Home_Appliances": 1.00, "Stationery": 1.00, "Toys": 1.00},
    ),
    "Clearance_Stock": dict(
        base_lift=2.00, disc=(0.40, 0.70),
        cat={"Apparel": 1.40, "Footwear": 1.40, "Home_Appliances": 1.30,
             "Stationery": 1.20, "Toys": 1.10, "Electronics": 1.00,
             "Personal_Care": 0.95, "Grocery": 0.70, "Beverages": 0.75,
             "Bakery": 0.75},
    ),
}

PROMO_NONE = "None"

# date ranges where a national campaign is forced at every store
CAMPAIGN_RANGES = [
    ("New_Year_Offer", date(2024, 1, 1), date(2024, 1, 5)),
    ("Republic_Day_Deal", date(2024, 1, 22), date(2024, 1, 31)),
    ("New_Year_Offer", date(2024, 12, 20), date(2025, 1, 5)),
    ("Diwali_Sale", date(2024, 10, 26), date(2024, 11, 4)),
    ("Republic_Day_Deal", date(2025, 1, 20), date(2025, 1, 31)),
    ("Diwali_Sale", date(2025, 10, 15), date(2025, 10, 23)),
    ("Clearance_Stock", date(2024, 7, 12), date(2024, 7, 14)),
    ("Clearance_Stock", date(2025, 3, 8), date(2025, 3, 10)),
    ("Clearance_Stock", date(2025, 8, 16), date(2025, 8, 18)),
]

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_MULT = [0.88, 0.85, 0.90, 0.95, 1.10, 1.85, 1.50]

LOYALTY_TIERS = [("Platinum", 0.08), ("Gold", 0.20), ("Silver", 0.35), ("Bronze", 0.37)]


# --------------------------------------------------------------------------- helpers

def weighted_cum(weights):
    """Cumulative weights for O(log n) weighted sampling."""
    return list(accumulate(weights))


def pick(cum, total, r):
    return bisect_right(cum, r * total)


def loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def price_band(p):
    if p < 299:
        return "Budget"
    if p < 1000:
        return "Mid"
    if p < 5000:
        return "Premium"
    return "Luxury"


# --------------------------------------------------------------------------- build

def plan_promotions(rng, cal, store_cat_vel):
    """Decide every store's campaign for every day up front.

    Returns (plan, day_lift) where plan[d][si] == (promo_type_or_None, [categories])
    and day_lift[d] is the chain-average demand multiplier that day. Deciding this
    before the line-item quotas are computed is what lets promotions lift *volume*
    instead of only shifting category mix.
    """
    store_total_vel = [sum(v) for v in store_cat_vel]
    plan, day_lift = [], []

    for d in range(N_DAYS):
        is_we = cal["is_weekend"][d]
        d_obj = cal["days"][d]
        forced = None
        for name, lo, hi in CAMPAIGN_RANGES:
            if lo <= d_obj <= hi:
                forced = name
                break

        day_rows = []
        lift_sum = 0.0
        for si in range(N_STORES):
            if forced:
                pt = forced
            elif is_we and rng.random() < 0.45:
                pt = "Weekend_Flash"
            elif rng.random() < 0.030:
                pt = "Loyalty_Flash"
            elif rng.random() < 0.020:
                pt = "Clearance_Stock"
            else:
                pt = PROMO_NONE

            if pt == PROMO_NONE:
                day_rows.append((PROMO_NONE, ()))
                lift_sum += 1.0
                continue

            spec = PROMO_SPECS[pt]
            ncat = 1 if rng.random() < 0.55 else 2
            ws = [spec["cat"][n] * CATEGORIES[ci]["share"] for ci, n in enumerate(CAT_NAMES)]
            cum = weighted_cum(ws)
            tot = cum[-1]
            chosen = []
            for _ in range(ncat):
                ci = pick(cum, tot, rng.random())
                if ci not in chosen:
                    chosen.append(ci)
            day_rows.append((pt, tuple(chosen)))

            # only the promoted categories gain, so the chain-wide lift is diluted
            mass = sum(store_cat_vel[si][c] for c in chosen) / store_total_vel[si]
            avg_lift = sum(spec["cat"][CAT_NAMES[c]] for c in chosen) / len(chosen)
            lift_sum += 1.0 + mass * (spec["base_lift"] * avg_lift - 1.0)

        plan.append(day_rows)
        day_lift.append(lift_sum / N_STORES)

    return plan, day_lift


def build_calendar():
    """Per-day arrays: dates, weekday flags, seasonality, holiday flags, demand index."""
    hol_lookup = {}
    for y, table in ((2024, HOLIDAYS_2024), (2025, HOLIDAYS_2025)):
        for (m, d), (name, mult) in table.items():
            hol_lookup[date(y, m, d)] = (name, mult)

    days = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]
    date_str = [d.isoformat() for d in days]
    weekday = [d.weekday() for d in days]
    weekday_name = [WEEKDAY_NAMES[w] for w in weekday]
    is_weekend = [1 if w >= 5 else 0 for w in weekday]
    day_of_month = [d.day for d in days]
    year_month = [f"{d.year}-{d.month:02d}" for d in days]
    week_of_year = [d.isocalendar()[1] for d in days]
    quarter = [f"Q{(d.month - 1) // 3 + 1}" for d in days]

    is_holiday, holiday_name = [], []
    for d in days:
        hit = hol_lookup.get(d)
        is_holiday.append(1 if hit else 0)
        holiday_name.append(hit[0] if hit else "None")

    # category seasonality, averaged across categories -> global index
    seas_sum = [0.0] * N_DAYS
    for c in CATEGORIES:
        s = c["season"]
        for i, d in enumerate(days):
            seas_sum[i] += s[d.month - 1]
    global_season = [v / N_CAT for v in seas_sum]

    holiday_mult = []
    for i, d in enumerate(days):
        hit = hol_lookup.get(d)
        holiday_mult.append(hit[1] if hit else 1.0)

    index = []
    for i in range(N_DAYS):
        d = days[i]
        # pay cycle: salary days at the start of the month
        dom = day_of_month[i]
        if dom <= 7:
            pay = 1.25
        elif dom <= 25:
            pay = 1.00
        else:
            pay = 0.85
        # month-end dip, year-on-year gentle growth
        trend = 1.0 + 0.08 * (i / N_DAYS)
        val = global_season[i] * WEEKDAY_MULT[weekday[i]] * pay * holiday_mult[i] * trend
        index.append(val)

    # per-category monthly curve, normalised to mean 1.0 within each category so it
    # reshapes the category MIX (which category a shopper reaches for this month)
    # without inflating or deflating total demand.
    cat_season = []
    for c in CATEGORIES:
        raw = [c["season"][days[i].month - 1] for i in range(N_DAYS)]
        mean = sum(raw) / N_DAYS
        cat_season.append([v / mean for v in raw])

    return dict(
        days=days, date_str=date_str, weekday=weekday, weekday_name=weekday_name,
        is_weekend=is_weekend, day_of_month=day_of_month, year_month=year_month,
        week_of_year=week_of_year, quarter=quarter, is_holiday=is_holiday,
        holiday_name=holiday_name, index=index, global_season=global_season,
        holiday_mult=holiday_mult, cat_season=cat_season,
    )


def daily_line_counts(index, target):
    """Deterministic per-day transaction counts that sum to exactly `target`."""
    w = [v for v in index]
    total_w = sum(w)
    raw = [target * v / total_w for v in w]
    base = [int(v) for v in raw]
    short = target - sum(base)
    order = sorted(range(N_DAYS), key=lambda i: raw[i] - base[i], reverse=True)
    for k in range(short):
        base[order[k % N_DAYS]] += 1
    return base


def build_products(rng):
    """1200 products across 10 categories with names, prices and velocity weights."""
    counts = []
    assigned = 0
    for c in CATEGORIES[:-1]:
        k = int(round(c["share"] * N_PRODUCTS))
        counts.append(k)
        assigned += k
    counts.append(N_PRODUCTS - assigned)

    p_id, p_name, p_cat, p_price, p_vel, p_qlo, p_qhi, p_expqty, p_maxpack = (
    [], [], [], [], [], [], [], [], [])
    idx = 0
    for ci, (c, k) in enumerate(zip(CATEGORIES, counts)):
        for _ in range(k):
            brand = BRANDS[rng.randrange(len(BRANDS))]
            item = c["items"][rng.randrange(len(c["items"]))]
            spec = c["specs"][rng.randrange(len(c["specs"]))]
            p_id.append(f"PRD{idx + 1:04d}")
            p_name.append(f"{brand} {item} {spec}")
            p_cat.append(ci)
            p_price.append(round(loguniform(rng, c["price"][0], c["price"][1]), 2))
            # cheaper items move faster; lognormal spread creates a realistic long tail
            p_vel.append(c["velocity"] * math.exp(-0.35 * math.log(p_price[-1]) / 8.0)
                         * rng.lognormvariate(0.0, VELOCITY_SIGMA))
            p_qlo.append(c["qty"][0])
            p_qhi.append(c["qty"][1])
            # expected units per line, pack-aware -- inventory capacity is sized from
            # this, so it must account for whole-pack buys or the long tail over-stocks out
            pk = c["packs"]
            p_expqty.append((1.0 - PACK_SHARE) * (c["qty"][0] + c["qty"][1]) / 2.0
                            + PACK_SHARE * (sum(pk) / len(pk)))
            # a shelf must hold at least one whole pack, or bulk buyers get clipped
            p_maxpack.append(max(pk))
            idx += 1

    return dict(id=p_id, name=p_name, cat=p_cat, base_price=p_price,
                vel=p_vel, qlo=p_qlo, qhi=p_qhi, expqty=p_expqty, maxpack=p_maxpack,
                packs=[c["packs"] for c in CATEGORIES])


def build_stores(rng):
    stores = []
    for ci, ((city, tier, region), n) in enumerate(zip(CITIES, STORE_PLAN)):
        for _ in range(n):
            stores.append(dict(store_id=f"ST{len(stores) + 1:02d}", city=city,
                               city_idx=ci, tier=tier, region=region))
    assert len(stores) == N_STORES, len(stores)
    return stores


def build_assortments(rng, stores, products):
    """Each store stocks the top PANEL_PRODUCTS plus a type-specific long tail."""
    panel_set = set(sorted(range(N_PRODUCTS),
                           key=lambda i: -products["vel"][i])[:PANEL_PRODUCTS])
    tail = [i for i in range(N_PRODUCTS) if i not in panel_set]

    assortment = []
    for s in stores:
        stype = STORE_TYPES[rng.choice(list(STORE_TYPES))]
        extra = stype["size"] - PANEL_PRODUCTS
        pool = []
        for i in tail:
            pool += [i] * max(1, int(round(stype["mix"][CAT_NAMES[products["cat"][i]]] * 4)))
        picked = rng.sample(pool, extra)
        assortment.append(sorted(panel_set | set(picked)))

    # category-skewed sampling can leave a niche SKU in nobody's store; swap it into a
    # few random stores so the catalog has no product that can never transact
    stocked = set()
    for items in assortment:
        stocked.update(items)
    for p in (i for i in range(N_PRODUCTS) if i not in stocked):
        for si in rng.sample(range(N_STORES), 3):
            evictable = [q for q in assortment[si] if q not in panel_set]
            if evictable:
                assortment[si].remove(rng.choice(evictable))
            assortment[si].append(p)
    for si in range(N_STORES):
        assortment[si].sort()

    store_cat_vel, prod_lists, prod_cums = [], [], []
    for si, items in enumerate(assortment):
        by_cat = defaultdict(list)
        for p in items:
            by_cat[products["cat"][p]].append(p)
        cl, cc = [], []
        for c in range(N_CAT):
            lst = by_cat.get(c, [])
            cl.append(lst)
            cc.append(weighted_cum([products["vel"][p] for p in lst]))
        prod_lists.append(cl)
        prod_cums.append(cc)
        store_cat_vel.append([sum(products["vel"][p] for p in lst) for lst in cl])

    return assortment, store_cat_vel, prod_lists, prod_cums, panel_set


def build_customers(rng, stores):
    city_stores = defaultdict(list)
    for i, s in enumerate(stores):
        city_stores[s["city_idx"]].append(i)
    store_traffic_cum = weighted_cum(
        [TIER_TRAFFIC_MULT[s["tier"]] * rng.uniform(0.85, 1.15) for s in stores])

    cust = []
    for i in range(N_CUSTOMERS):
        home_city = rng.choices(range(len(CITIES)), weights=[5, 5, 4, 4, 5, 4, 4, 4, 4, 4, 4, 3])[0]

        # lifecycle segment
        r = rng.random()
        if r < 0.10:                                  # New
            seg = "New"
            first = rng.randint(N_DAYS - 120, N_DAYS - 5)
            last = N_DAYS - 1
            freq = rng.uniform(0.4, 1.6)
        elif r < 0.35:                                # Growing
            seg = "Growing"
            first = rng.randint(N_DAYS - 600, N_DAYS - 180)
            last = N_DAYS - 1
            freq = rng.uniform(1.0, 2.6)
        elif r < 0.80:                                # Loyal
            seg = "Loyal"
            first = rng.randint(0, N_DAYS - 420)
            last = N_DAYS - 1
            freq = rng.uniform(1.6, 4.2)
        elif r < 0.92:                                # At risk - long gap at the end
            seg = "At_Risk"
            first = rng.randint(0, N_DAYS - 500)
            last = rng.randint(N_DAYS - 100, N_DAYS - 25)
            freq = rng.uniform(0.6, 2.0)
        else:                                         # Churned mid-period
            seg = "Churned"
            first = rng.randint(0, N_DAYS - 560)
            last = rng.randint(first + 60, N_DAYS - 150)
            freq = rng.uniform(0.4, 1.4)

        tier = rng.choices([t for t, _ in LOYALTY_TIERS],
                           weights=[w for _, w in LOYALTY_TIERS])[0]

        # category affinity (Dirichlet-ish: each category gets a base weight)
        aff = []
        for c in CATEGORIES:
            aff.append(rng.uniform(0.55, 1.9) * (c["share"] + 0.02))
        ssum = sum(aff)
        aff = [a / ssum for a in aff]

        # Retail baskets are bimodal, not spread evenly: a quick top-up of 1-4 items,
        # and an occasional monthly stock-up of 10-28. Measured against real retail
        # (UCI Online Retail II, consumer-value invoices: median 2 lines, p90 16,
        # 46% single-line) a flat 2-5 basket reads as obviously synthetic.
        # Bulk shoppers are also lower-frequency, which is what keeps the visit mix sane.
        bulk = rng.random() < 0.18
        if bulk:
            bsize = rng.randint(10, 28)
        else:
            bsize = rng.choices([1, 2, 3, 4, 5, 6],
                                weights=[0.14, 0.26, 0.25, 0.18, 0.11, 0.06])[0]

        freq_eff = freq * (0.32 if bulk else 1.0)

        cust.append(dict(
            id=f"CUS{i + 1:05d}",
            home_city=home_city,
            first=first,
            last=last,
            freq=freq_eff,
            seg=seg,
            tier=tier,
            aff=aff,
            sens=rng.uniform(0.60, 1.80),          # promotion responsiveness
            basket=bsize,
            bulk=bulk,
            online=rng.random() < 0.30,
            weight=freq_eff * (1.25 if tier in ("Platinum", "Gold") else 1.0),
        ))

    return cust, city_stores, store_traffic_cum


# --------------------------------------------------------------------------- main

SALES_HEADER = [
    # the 12 requested columns, exact names
    "date", "customer_id", "store_id", "product_id", "product_category",
    "quantity_sold", "unit_price", "discount", "promotion", "inventory_level",
    "sales_amount", "city",
    # engineered helpers
    "discount_pct", "is_promotion", "transaction_id", "product_name", "region",
    "weekday", "is_weekend", "is_holiday", "year_month", "week_of_year",
    "quarter", "price_band", "customer_segment", "loyalty_tier", "channel",
]

PANEL_HEADER = [
    "week_start_date", "year", "week_of_year", "store_id", "city", "region",
    "product_id", "product_category", "units_sold", "revenue", "avg_unit_price",
    "avg_discount_pct", "is_promo_week", "stockout_count", "on_hand_end",
]


def main():
    rng = random.Random(SEED)

    print("Building calendar ...")
    cal = build_calendar()
    cat_season = cal["cat_season"]

    print("Building products ...")
    products = build_products(rng)

    print("Building stores and assortments ...")
    stores = build_stores(rng)
    assortment, store_cat_vel, prod_lists, prod_cums, panel_set = build_assortments(
        rng, stores, products)

    print("Building customers ...")
    customers, city_stores, store_traffic_cum = build_customers(rng, stores)
    store_traffic_total = store_traffic_cum[-1]

    # ------------------------------------------------------------ inventory setup
    print("Setting up inventory ...")
    stock, reorder_pt, capacity = [], [], []
    for si, s in enumerate(stores):
        # store_traffic_cum already carries TIER_TRAFFIC_MULT, so do not apply it twice
        w_si = store_traffic_cum[si] - (store_traffic_cum[si - 1] if si else 0.0)
        share_store = w_si / store_traffic_total
        lines_per_day = TX_ROWS / N_DAYS * share_store
        wsum = sum(products["vel"][p] for p in assortment[si])
        st, rp, cap = {}, {}, {}
        for p in assortment[si]:
            pshare = products["vel"][p] / wsum
            avg_qty = products["expqty"][p]
            exp_units = max(0.01, lines_per_day * pshare * avg_qty)
            days_cover = rng.randint(10, 30)
            # a shelf holds at least a modest multiple of one pack (scaled down so a
            # 50-pack Stationery SKU does not immunise itself against stocking out),
            # plus a reorder buffer. Fast movers still run dry.
            floor_p = max(6, min(12, products["maxpack"][p]))
            cap_p = max(floor_p, int(round(exp_units * days_cover)))
            st[p] = max(floor_p, int(round(cap_p * rng.uniform(0.80, 1.0))))
            rp[p] = max(floor_p, int(round(cap_p * 0.35)))
            cap[p] = cap_p
        stock.append(st)
        reorder_pt.append(rp)
        capacity.append(cap)

    pending = [set() for _ in range(N_STORES)]
    restock_queue = defaultdict(list)

    # store price multipliers (each store nudges its own price level)
    store_price_mult = [rng.uniform(0.96, 1.05) for _ in range(N_STORES)]

    # ------------------------------------------------------------ daily simulation
    print(f"Planning promotions across {N_STORES} stores x {N_DAYS} days ...")
    promo_plan, day_promo_lift = plan_promotions(rng, cal, store_cat_vel)

    print(f"Simulating {N_DAYS} days ...")
    # promotions lift chain volume, not just category mix
    index = [cal["index"][d] * (1.0 + 0.75 * (day_promo_lift[d] - 1.0))
             for d in range(N_DAYS)]
    lines_per_day = daily_line_counts(index, TX_ROWS)

    panel_pairs = [(s, p) for s in range(N_STORES)
                   for p in assortment[s] if p in panel_set]
    pair_index = {k: i for i, k in enumerate(panel_pairs)}
    pw_units = [array("i", [0] * N_WEEKS) for _ in panel_pairs]
    pw_rev = [array("d", [0.0] * N_WEEKS) for _ in panel_pairs]
    pw_price = [array("d", [0.0] * N_WEEKS) for _ in panel_pairs]
    pw_disc = [array("d", [0.0] * N_WEEKS) for _ in panel_pairs]
    pw_lines = [array("i", [0] * N_WEEKS) for _ in panel_pairs]
    pw_stock = [array("i", [0] * N_WEEKS) for _ in panel_pairs]
    pw_stockout = [array("i", [0] * N_WEEKS) for _ in panel_pairs]
    pw_promo = [array("i", [0] * N_WEEKS) for _ in panel_pairs]

    tx_rows = []          # completed sale line-items
    stockout_rows = []    # demand that hit zero stock
    txn_seq = 0
    promo_line_totals = defaultdict(int)
    promo_line_units = defaultdict(int)
    month_rev = defaultdict(float)

    # customers sorted by first-purchase day, used to guarantee full coverage
    new_queue = sorted(customers, key=lambda c: c["first"])
    new_q = 0
    pending_new = []

    # guarantee every product transacts at least once: a forced "launch" sale per SKU,
    # spread over the first 180 days so no single day shows a visible spike
    store_of_product = defaultdict(list)
    for si in range(N_STORES):
        for p in assortment[si]:
            store_of_product[p].append(si)
    launches = defaultdict(list)
    for p in range(N_PRODUCTS):
        launches[(p * 180) // N_PRODUCTS].append(
            (rng.choice(store_of_product[p]), p, products["cat"][p]))

    def emit_one(c, si, ci, p, pt, pcats, txn_id, channel, trend_price):
        """Emit one sale line-item. Returns 'sale', 'stockout', or 'skip'.

        Reads the current day's locals (d, dstr, pw_week, ...) from the enclosing scope.
        """
        st = stores[si]
        stk = stock[si]
        on_hand = stk[p]
        if on_hand <= 0:
            pi_so = pair_index.get((si, p))
            if pi_so is not None and pw_week >= 0:
                pw_stockout[pi_so][pw_week] += 1
            if len(stockout_rows) < 150_000:
                stockout_rows.append(
                    (d, dstr, c, si, st, p, ci, txn_id, channel, pt,
                     pcats, wkn, is_we, hol, hname, ym, wow, qtr, week))
            return "stockout"

        # Most lines are a single unit or a small count. A minority are whole
        # packs, which is what creates the spikes at 6/10/12/24 that real retail
        # quantity data shows (buyers take the pack, not a loose unit).
        if rng.random() < PACK_SHARE:
            pk = products["packs"][ci]
            want = pk[rng.randrange(len(pk))]
        else:
            want = rng.randint(products["qlo"][p], products["qhi"][p])
        qty = want if want <= on_hand else on_hand

        stk[p] = on_hand - qty
        if stk[p] <= reorder_pt[si][p] and p not in pending[si]:
            pending[si].add(p)
            restock_queue[d + rng.randint(5, 14)].append((si, p))

        unit_price = round(products["base_price"][p] * trend_price, 2)
        on_promo = pt != PROMO_NONE and ci in pcats
        if on_promo:
            lo, hi = PROMO_SPECS[pt]["disc"]
            dpct = min(0.75, max(0.0, rng.uniform(lo, hi) * rng.uniform(0.85, 1.15)))
        else:
            dpct = 0.0
        disc_amt = round(unit_price * dpct, 2)
        sales_amount = round(qty * (unit_price - disc_amt), 2)
        # derived from the ROUNDED discount so the two columns agree exactly
        dpct_out = round(disc_amt / unit_price * 100, 2) if unit_price else 0.0

        tx_rows.append((
            d, dstr, c["id"], st["store_id"], products["id"][p], CAT_NAMES[ci],
            qty, unit_price, disc_amt, pt if on_promo else PROMO_NONE,
            stk[p], sales_amount, st["city"],
            dpct_out, 1 if on_promo else 0, txn_id, products["name"][p], st["region"],
            wkn, is_we, hol, ym, wow, qtr, price_band(unit_price),
            c["seg"], c["tier"], channel,
        ))

        if on_promo:
            promo_line_totals[pt] += 1
            promo_line_units[pt] += qty
        month_rev[ym] += sales_amount

        pi = pair_index.get((si, p))
        if pi is not None and pw_week >= 0:
            pw_units[pi][pw_week] += qty
            pw_rev[pi][pw_week] += sales_amount
            pw_price[pi][pw_week] += unit_price
            pw_disc[pi][pw_week] += dpct * 100
            pw_lines[pi][pw_week] += 1
            pw_promo[pi][pw_week] += 1 if on_promo else 0
        return "sale"

    for d in range(N_DAYS):
        dstr = cal["date_str"][d]
        wk = cal["weekday"][d]
        wkn = cal["weekday_name"][d]
        is_we = cal["is_weekend"][d]
        ym = cal["year_month"][d]
        wow = cal["week_of_year"][d]
        qtr = cal["quarter"][d]
        hol = cal["is_holiday"][d]
        hname = cal["holiday_name"][d]
        week = d // 7
        # the trailing 2 days spill into a partial week that the panel does not cover
        pw_week = week if week < N_WEEKS else -1

        # --- process scheduled restocks
        for (si, p) in restock_queue.pop(d, []):
            if stock[si][p] < capacity[si][p]:
                stock[si][p] = capacity[si][p]
            pending[si].discard(p)

        # --- campaign for each store today (decided in the planning pass)
        promo_type, promo_cats = [None] * N_STORES, [None] * N_STORES
        for si in range(N_STORES):
            pt, chosen = promo_plan[d][si]
            promo_type[si] = pt
            promo_cats[si] = list(chosen)

        # --- customers eligible today, plus a queue of everyone making their first
        #     purchase; the first baskets of the day are handed to them so that all
        #     8,000 customers appear in the file. A basket can legitimately yield no
        #     line-item (everything out of stock), so unserved customers are retried.
        while new_q < len(new_queue) and new_queue[new_q]["first"] <= d:
            pending_new.append(new_queue[new_q])
            new_q += 1

        active = [c for c in customers if c["first"] <= d <= c["last"]]
        act_cum = weighted_cum([c["weight"] for c in active])
        act_total = act_cum[-1] if act_cum else 1.0

        # --- guarantee every SKU transacts at least once in the window
        emitted = 0
        for (lsi, lp, lci) in launches.get(d, ()):
            pt0, pcats0 = promo_plan[d][lsi]
            c0 = (active[pick(act_cum, act_total, rng.random())] if active
                  else customers[0])
            txn_seq += 1
            trend0 = ((1.0 + 0.05 * (d / N_DAYS)) * store_price_mult[lsi]
                      * TIER_PRICE_MULT[stores[lsi]["tier"]])
            if emit_one(c0, lsi, lci, lp, pt0, pcats0,
                        f"TXN{txn_seq:07d}", "Store", trend0) == "sale":
                emitted += 1

        # `lines_per_day` is a LINE-ITEM quota, not a basket count: keep opening
        # baskets until the day's line-item quota is met.
        d_target = lines_per_day[d]
        guard = 0
        new_i = 0
        served = set()
        while emitted < d_target and guard < d_target * 30 + 60:
            guard += 1
            if new_i < len(pending_new):
                c = pending_new[new_i]
                forced_idx = new_i
                new_i += 1
            else:
                c = active[pick(act_cum, act_total, rng.random())]
                forced_idx = None
            emitted_before = emitted

            # --- store: mostly the customer's home city
            if rng.random() < 0.85:
                cand = city_stores[c["home_city"]]
                sub = weighted_cum([TIER_TRAFFIC_MULT[stores[s]["tier"]] for s in cand])
                si = cand[pick(sub, sub[-1], rng.random())]
            else:
                si = pick(store_traffic_cum, store_traffic_total, rng.random())

            st = stores[si]
            txn_seq += 1
            txn_id = f"TXN{txn_seq:07d}"

            if c["online"] and rng.random() < 0.55:
                channel = "Mobile_App" if rng.random() < 0.74 else "Web"
            else:
                channel = "Store"

            # --- promotion lift for this store-day
            pt = promo_type[si]
            pcats = promo_cats[si]
            boost = [1.0] * N_CAT
            if pt != PROMO_NONE:
                spec = PROMO_SPECS[pt]
                for ci in pcats:
                    lift = 1.0 + (spec["base_lift"] * spec["cat"][CAT_NAMES[ci]] - 1.0) * c["sens"]
                    boost[ci] = max(0.6, lift)

            # --- basket. Online adds an impulse item. The cap bounds the bulk mode so a
            # single monthly shop cannot eat a whole day's line quota on its own.
            n_items = c["basket"]
            if channel != "Store":
                n_items += 1 if rng.random() < 0.30 else 0
            n_items = min(n_items, 30)

            # category pick weights: affinity x store mix x promotion boost x season
            base_w = store_cat_vel[si]
            cw = [c["aff"][ci] * (base_w[ci] if base_w[ci] > 0 else 1e-9) * boost[ci]
                  * cat_season[ci][d] for ci in range(N_CAT)]
            ccum = weighted_cum(cw)
            ctot = ccum[-1]

            stk = stock[si]
            rp = reorder_pt[si]
            seen = set()
            tier_mult = TIER_PRICE_MULT[st["tier"]]
            inflation = 1.0 + 0.05 * (d / N_DAYS)
            trend_price = inflation * store_price_mult[si] * tier_mult

            for _ in range(n_items):
                if emitted >= d_target:
                    break
                ci = pick(ccum, ctot, rng.random())
                lst = prod_lists[si][ci]
                if not lst:
                    continue
                pcum = prod_cums[si][ci]
                p = lst[pick(pcum, pcum[-1], rng.random())]
                if p in seen:
                    continue
                seen.add(p)

                if emit_one(c, si, ci, p, pt, pcats, txn_id, channel,
                            trend_price) == "sale":
                    emitted += 1

            if forced_idx is not None and emitted > emitted_before:
                served.add(forced_idx)

        if served:
            pending_new = [cu for i, cu in enumerate(pending_new) if i not in served]

        # --- snapshot stock for the dense panel at the end of each week
        if wk == 6:
            for pi, (si, p) in enumerate(panel_pairs):
                pw_stock[pi][week] = stock[si][p]

    print(f"  transactions: {len(tx_rows)}  stockout candidates: {len(stockout_rows)}")
    print(f"  raw stockout attempt rate: "
          f"{len(stockout_rows) / (len(stockout_rows) + len(tx_rows)) * 100:.2f}%")

    # ------------------------------------------------------------ exact row budget
    rng2 = random.Random(SEED + 1)
    if len(stockout_rows) >= STOCKOUT_ROWS:
        chosen_so = rng2.sample(stockout_rows, STOCKOUT_ROWS)
    else:
        chosen_so = stockout_rows

    so_final = []
    for (d, dstr, c, si, st, p, ci, txn_id, channel, pt, pcats,
         wkn, is_we, hol, hname, ym, wow, qtr, week) in chosen_so:
        unit_price = round(products["base_price"][p]
                           * (1.0 + 0.05 * (d / N_DAYS)) * store_price_mult[si]
                           * TIER_PRICE_MULT[st["tier"]], 2)
        on_promo = pt != PROMO_NONE and ci in pcats
        if on_promo:
            lo, hi = PROMO_SPECS[pt]["disc"]
            dpct = min(0.75, max(0.0, rng2.uniform(lo, hi) * rng2.uniform(0.85, 1.15)))
        else:
            dpct = 0.0
        disc_amt = round(unit_price * dpct, 2)
        dpct_out = round(disc_amt / unit_price * 100, 2) if unit_price else 0.0
        so_final.append((
            d, dstr, c["id"], st["store_id"], products["id"][p], CAT_NAMES[ci],
            0, unit_price, disc_amt, pt if on_promo else PROMO_NONE,
            0, 0.0, st["city"], dpct_out,
            1 if on_promo else 0, txn_id, products["name"][p], st["region"],
            wkn, is_we, hol, ym, wow, qtr, price_band(unit_price),
            c["seg"], c["tier"], channel,
        ))

    all_rows = tx_rows + so_final
    assert len(tx_rows) == TX_ROWS, f"expected {TX_ROWS} transactions, got {len(tx_rows)}"
    assert len(so_final) == STOCKOUT_ROWS, f"expected {STOCKOUT_ROWS} stockouts, got {len(so_final)}"
    assert len(all_rows) == TARGET_ROWS, f"expected {TARGET_ROWS} rows, got {len(all_rows)}"
    all_rows.sort(key=lambda r: (r[0], r[15], r[4]))

    # ------------------------------------------------------------ write sales file
    sales_path = OUT_DIR / "retail_pulse_sales.csv"
    print(f"Writing {sales_path.name} ({len(all_rows)} rows) ...")
    with sales_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SALES_HEADER)
        for r in all_rows:
            assert len(r) == len(SALES_HEADER) + 1, f"row width {len(r)} != {len(SALES_HEADER) + 1}"
            w.writerow((r[1], r[2], r[3], r[4], r[5], r[6], f"{r[7]:.2f}",
                        f"{r[8]:.2f}", r[9], r[10], f"{r[11]:.2f}", r[12],
                        f"{r[13]:.2f}", r[14], r[15], r[16], r[17], r[18], r[19],
                        r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27]))

    # ------------------------------------------------------------ write panel file
    panel_path = OUT_DIR / "retail_pulse_demand_panel.csv"
    print(f"Writing {panel_path.name} ({len(panel_pairs) * N_WEEKS} rows) ...")
    week_starts = [cal["date_str"][w * 7] for w in range(N_WEEKS)]
    week_years = [cal["days"][w * 7].year for w in range(N_WEEKS)]
    week_nums = [cal["days"][w * 7].isocalendar()[1] for w in range(N_WEEKS)]

    with panel_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PANEL_HEADER)
        for pi, (si, p) in enumerate(panel_pairs):
            st = stores[si]
            for wk_i in range(N_WEEKS):
                n = pw_lines[pi][wk_i]
                u = pw_units[pi][wk_i]
                avg_price = round(pw_price[pi][wk_i] / n, 2) if n else 0.0
                avg_disc = round(pw_disc[pi][wk_i] / n, 2) if n else 0.0
                w.writerow((
                    week_starts[wk_i], week_years[wk_i], week_nums[wk_i],
                    st["store_id"], st["city"], st["region"],
                    products["id"][p], CAT_NAMES[products["cat"][p]],
                    u, f"{pw_rev[pi][wk_i]:.2f}", f"{avg_price:.2f}",
                    f"{avg_disc:.2f}", 1 if pw_promo[pi][wk_i] else 0,
                    pw_stockout[pi][wk_i], pw_stock[pi][wk_i],
                ))

    # ------------------------------------------------------------ console report
    print("\n" + "=" * 68)
    print("DATASET SUMMARY")
    print("=" * 68)
    print(f"rows                 : {len(all_rows):,}")
    print(f"transactions         : {len(tx_rows):,}")
    print(f"stockout rows        : {len(so_final):,}")
    print(f"date range in file   : {all_rows[0][1]} .. {all_rows[-1][1]}")
    print(f"customers            : {N_CUSTOMERS:,}")
    print(f"stores               : {N_STORES}")
    print(f"products             : {N_PRODUCTS:,}")
    print(f"store-product pairs  : {len(panel_pairs):,} (in panel)")
    print(f"total assortment     : {sum(len(a) for a in assortment):,}")

    total_rev = sum(r[11] for r in all_rows)
    total_units = sum(r[6] for r in all_rows)
    print(f"total units sold     : {total_units:,}")
    print(f"total revenue        : INR {total_rev:,.0f}")
    print(f"avg unit price       : INR {sum(r[7] for r in all_rows) / len(all_rows):,.2f}")
    print(f"stockout rate        : {len(so_final) / len(all_rows) * 100:.2f}%")

    print("\nMonthly revenue (seasonality check):")
    for ym in sorted(month_rev):
        bar = "#" * int(month_rev[ym] / max(month_rev.values()) * 46)
        print(f"  {ym}  {month_rev[ym] / 1e6:6.2f}M  {bar}")

    print("\nPromotion mix (line count / share of promo lines):")
    tot_promo = sum(promo_line_totals.values()) or 1
    for pt in sorted(promo_line_totals):
        avg_disc = ""
        print(f"  {pt:<18} {promo_line_totals[pt]:>7,}  "
              f"{promo_line_totals[pt] / tot_promo * 100:5.1f}%  "
              f"units {promo_line_units[pt]:>8,}")

    print("\nTop 10 cities by revenue:")
    city_rev = defaultdict(float)
    for r in all_rows:
        city_rev[r[12]] += r[11]
    for city, rev in sorted(city_rev.items(), key=lambda x: -x[1])[:10]:
        print(f"  {city:<14} INR {rev / 1e6:7.2f}M")

    print("\nFiles written:")
    for f in sorted(OUT_DIR.glob("retail_pulse_*")):
        print(f"  {f.name:<38} {f.stat().st_size / 1e6:8.2f} MB")
    print("=" * 68)


if __name__ == "__main__":
    main()
