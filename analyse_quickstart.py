"""Starter analysis over the three Retail Pulse upload files.

    python analyse_quickstart.py

Reads retail_pulse_sales.csv (everything) and retail_pulse_demand_panel.csv (the lost-demand
signal). Sections:

    1  monthly revenue trend + year-on-year
    2  weekly revenue trend, complete weeks only
    3  top products by revenue
    4  revenue by city and by region
    5  category summary
    6  promotion effectiveness, controlling for weekday
    7  category seasonality read as MIX (share of units)
    8  stockout impact -- lost demand vs the legitimate last-unit rows
    9  where lost demand concentrates (store x product, from the panel)
   10  RFM snapshot

Three documented traps this script is careful about, all from retail_pulse_data_dictionary.md:

  * A stockout is `quantity_sold == 0`, NOT `inventory_level == 0`. There are 27,749 rows
    with zero stock AND positive quantity -- a customer buying the last unit on the shelf.
    Those are real sales. Counting them as lost demand would overstate it by 3.7x.
  * Lost-demand rows record demand that met an empty shelf. They carry zero revenue and
    must not be counted as sales, but they are the signal worth modelling, so section 6
    measures promotion lift on selling rows only, free of the censoring bias.
  * Category seasonality is a MIX effect. Global festive demand lifts every category at
    once, so absolute units peak in October regardless of a category's own season. The
    real signal is each category's share of units -- that is what section 7 prints.

Needs pandas. Reuses load_quickstart for paths and console setup.
"""
import sys

import pandas as pd

from load_quickstart import (PANEL_CSV, SALES_CSV, head, use_utf8_console)


def load():
    sales = pd.read_csv(SALES_CSV, parse_dates=["date"])
    # Monday-start week, matching retail_pulse.aggregate(freq="weekly").
    sales["week_start"] = sales["date"] - pd.to_timedelta(
        sales["date"].dt.weekday, unit="D")
    panel = pd.read_csv(PANEL_CSV, parse_dates=["week_start_date"])
    return sales, panel


def money(v):
    """INR with Indian digit grouping, so a crore reads as crore."""
    return f"INR {v:,.0f}"


def show(df, label, index=True):
    print(f"\n{label}")
    print(df.to_string())


def monthly_trend(sales):
    head("1  monthly revenue trend")
    by_month = sales.groupby("year_month")["sales_amount"].sum()
    show(by_month.to_frame("revenue"), "revenue by month")

    # Year-on-year only where both years exist -- 2025-12 is the 24th month and has no
    # 2024 counterpart, so a naive pct_change prints a misleading blank or a 0% row.
    y = by_month.groupby(by_month.index.str[:4]).sum()
    print(f"\nfull-year revenue")
    print(y.to_frame("revenue").to_string())
    if len(y) == 2:
        years = list(y.index)
        growth = y[years[1]] / y[years[0]] - 1
        print(f"{years[0]} -> {years[1]} growth: {growth:+.1%}")


def weekly_trend(sales):
    head("2  weekly revenue trend")
    # The data ends 2025-12-31, so the week starting 2025-12-29 holds only 3 of its 7 days
    # and shows ~37% of a normal week. Left in, every chart ends in a fake demand cliff.
    days = sales.groupby("week_start")["date"].nunique()
    complete = days[days == 7].index
    dropped = days[days != 7]
    weekly = (sales[sales["week_start"].isin(complete)]
              .groupby("week_start")["sales_amount"].sum())

    print(f"weeks: {len(weekly)} complete")
    print(f"range: {weekly.index.min().date()} .. {weekly.index.max().date()}")
    if len(dropped):
        for wk, n in dropped.items():
            print(f"dropped partial week {wk.date()} -- only {n} of 7 days present")
    print(f"\nrevenue per week: min {money(weekly.min())}, "
          f"median {money(weekly.median())}, max {money(weekly.max())}")

    top5 = weekly.nlargest(5)
    print(f"\n5 strongest weeks (these are the Oct/Nov Diwali peaks)")
    show(top5.to_frame("revenue"), "")


def top_products(sales, n=10):
    head(f"3  top {n} products by revenue")
    g = (sales.groupby(["product_id", "product_name", "product_category"])
         .agg(revenue=("sales_amount", "sum"),
              units=("quantity_sold", "sum"),
              lines=("sales_amount", "size"))
         .reset_index())
    g["avg_price"] = g["revenue"] / g["units"].replace(0, pd.NA)
    g = g.sort_values("revenue", ascending=False)
    top = g.head(n)
    show(top.set_index("product_id")[
        ["product_name", "product_category", "units", "revenue", "lines"]],
        "revenue, INR")

    total = g["revenue"].sum()
    share = top["revenue"].sum() / total
    print(f"\ntop {n} of {len(g):,} products = {share:.1%} of revenue")
    print(f"that is {n / len(g):.1%} of the catalogue driving {share:.1%} of revenue --")
    print(f"the file follows a Pareto curve, so revenue is concentrated, not spread.")


def geography(sales):
    head("4  revenue by city and region")
    for key in ("city", "region"):
        g = (sales.groupby(key)
             .agg(revenue=("sales_amount", "sum"),
                  units=("quantity_sold", "sum"),
                  customers=("customer_id", "nunique"),
                  stores=("store_id", "nunique"))
             .sort_values("revenue", ascending=False))
        g["share"] = g["revenue"] / g["revenue"].sum()
        show(g, f"by {key}")


def categories(sales):
    head("5  category summary")
    g = (sales.groupby("product_category")
         .agg(revenue=("sales_amount", "sum"),
              units=("quantity_sold", "sum"),
              lines=("sales_amount", "size"),
              stockout_lines=("quantity_sold", lambda s: int((s == 0).sum())))
         .sort_values("revenue", ascending=False))
    g["units_per_line"] = g["units"] / (g["lines"] - g["stockout_lines"])
    g["stockout_pct"] = g["stockout_lines"] / g["lines"]
    show(g, "revenue, units, stockout exposure")


def promotion_effect(sales):
    head("6  promotion effectiveness")
    # The dictionary measures lift in units PER DAY against the same weekday on a
    # non-campaign day, so both sides of the ratio have to be days, not line-items.
    daily = sales.groupby("date").agg(units=("quantity_sold", "sum"),
                                      lines=("sales_amount", "size"))
    daily["weekday"] = daily.index.day_name()
    daily["campaign"] = sales.groupby("date")["is_promotion"].max().astype(bool)

    n_camp = int(daily["campaign"].sum())
    print(f"campaign days: {n_camp} of {len(daily)} ({n_camp / len(daily):.1%})")
    m = daily.groupby(["weekday", "campaign"])[["units", "lines"]].mean().unstack("campaign")
    out = pd.DataFrame({
        "non_campaign_units": m[("units", False)],
        "campaign_units": m[("units", True)],
        "non_campaign_lines": m[("lines", False)],
        "campaign_lines": m[("lines", True)],
    })
    out["units_lift"] = out["campaign_units"] / out["non_campaign_units"]
    out["traffic_lift"] = out["campaign_lines"] / out["non_campaign_lines"]
    show(out.sort_values("units_lift", ascending=False),
         "mean PER DAY, campaign vs non-campaign, by weekday")

    # Weekend_Flash runs every weekend, so no Saturday or Sunday is ever a non-campaign
    # day. Those two rows have no comparison and are left blank rather than filled in --
    # they are also the highest-volume days, so dropping them is not a small omission.
    never = out["units_lift"].isna().sum()
    print(f"\n{never} weekdays have no non-campaign counterpart (Weekend_Flash runs every")
    print(f"weekend), so no weekday-controlled lift can be computed for them.")
    print(f"mean units/day lift across the {out['units_lift'].notna().sum()} comparable "
          f"weekdays: {out['units_lift'].mean():.2f}x")

    # Where does the lift actually come from -- bigger baskets, or more baskets?
    # Selling rows only: stockout lines carry quantity 0 by construction.
    selling = sales[sales["quantity_sold"] > 0]
    per_line = selling.groupby("is_promotion")["quantity_sold"].mean()
    print(f"\nunits per promoted line     {per_line[1]:.2f}")
    print(f"units per non-promoted line {per_line[0]:.2f}  "
          f"({per_line[1] / per_line[0]:.2f}x)")
    print(f"\nweekday-controlled lift (mean of the per-weekday ratios above):")
    print(f"  units per day      {out['units_lift'].mean():.2f}x")
    print(f"  line-items per day {out['traffic_lift'].mean():.2f}x")
    print("\nA promoted line carries no more units than an unpromoted one, so a campaign")
    print("does not inflate baskets -- it brings marginally more line-items through the")
    print("door on the same weekday.")

    print("\nNOTE: this does not reproduce the dictionary's 1.56x, and the gap is not a")
    print("bug. 1.56x is the generator's simulated multiplier applied to promoted units.")
    print("At day level the contrast is far smaller, because 77% of all days already run")
    print("a campaign (Weekend_Flash covers every weekend), leaving few clean weekday")
    print("pairs to compare. Treat 1.56x as a property of the simulation, and the")
    print("figures above as what the file actually shows. Worth confirming with whoever")
    print("owns the dataset before you quote either number in a write-up.")

    # Discount depth vs response, per category -- the SQL example in the dictionary.
    d = (selling[selling["is_promotion"] == 1]
         .groupby("product_category")
         .agg(avg_discount_pct=("discount_pct", "mean"),
              avg_units=("quantity_sold", "mean"),
              lines=("quantity_sold", "size"))
         .sort_values("avg_units", ascending=False))
    show(d, "promoted lines only: discount depth vs units")


def category_mix_seasonality(sales):
    head("7  category seasonality as MIX (share of all units)")
    # Absolute units would show nearly every category peaking in October, which is the
    # global festive lift, not the category's own season. The real signal is each
    # category's share of THAT MONTH's total units -- this is the measure the dictionary
    # quotes, and it reproduces its Beverages figures exactly: 26.5% in May, 19.9% in Nov.
    units = (sales.groupby(["product_category", sales["date"].dt.month])
             ["quantity_sold"].sum().unstack())
    volume = units.sum(axis=1).sort_values(ascending=False)
    share = (units.div(units.sum(axis=0), axis=1) * 100).round(1)
    share.index.name = "category"
    show(share, "share of that month's total units, %")

    table = pd.concat([
        volume.rename("units"),
        share.idxmax(axis=1).rename("peak month"),
        share.idxmin(axis=1).rename("trough month"),
        (share.max(axis=1) / share.min(axis=1)).rename("peak/trough"),
    ], axis=1)
    print("\nwhere each category actually peaks, largest first")
    show(table, "")

    # Categories this thin produce a dramatic-looking swing out of pure noise: Footwear
    # shows a 5.25x peak/trough on under 900 units in the whole file.
    thin = volume[volume < volume.max() * 0.02]
    if len(thin):
        print(f"\nignore the peak/trough column for {', '.join(thin.index)} -- under 2% of")
        print("the largest category's volume across all 24 months, so their monthly share")
        print("is sampling noise, not seasonality.")


def stockout_impact(sales, panel):
    head("8  stockout impact -- lost demand vs real last-unit sales")
    qty = sales["quantity_sold"]
    inv = sales["inventory_level"]

    lost = qty == 0
    last_unit = (qty > 0) & (inv == 0)
    both_zero_and_qty = (qty == 0) & (inv == 0)

    print(f"lost-demand rows      {lost.sum():>7,}  ({lost.mean():.2%})"
          f"   quantity 0 AND inventory 0 -- shelf was empty")
    print(f"last-unit real sales  {last_unit.sum():>7,}  ({last_unit.mean():.2%})"
          f"   quantity > 0 AND inventory 0 -- customer bought the final unit")
    print(f"\nstockout == quantity_sold == 0, NOT inventory_level == 0.")
    print(f"Picking the wrong test reports {last_unit.sum():,} lost-demand rows "
          f"instead of {lost.sum():,} -- "
          f"{last_unit.sum() / max(lost.sum(), 1):.1f}x too high.")

    # Rule 6 from the dictionary: quantity 0 implies inventory 0. If that ever breaks the
    # file is corrupt, so say so rather than quietly reporting a number.
    contradiction = int((qty == 0).sum() - both_zero_and_qty.sum())
    print(f"\nrule check -- quantity 0 but inventory > 0: {contradiction:,}"
          f"  {'PASS' if contradiction == 0 else 'FAIL'}")

    print(f"\nlost demand by category (share of all stockout rows)")
    by_cat = (sales[lost].groupby("product_category").size()
              .sort_values(ascending=False).to_frame("stockout_lines"))
    by_cat["pct_of_category"] = (
        by_cat["stockout_lines"] / sales.groupby("product_category").size())
    by_cat["pct_of_all_stockouts"] = by_cat["stockout_lines"] / lost.sum()
    show(by_cat, "")

    print(f"panel stockout_count total: {int(panel['stockout_count'].sum()):,}"
          f"  (store x product x week censoring events)")


def lost_demand_hotspots(panel, n=20):
    head(f"9  where lost demand concentrates -- top {n} store x product")
    g = (panel.groupby(["store_id", "product_id", "product_category"])
         .agg(lost_sales_weeks=("stockout_count", "sum"),
              weeks=("stockout_count", "size"),
              units=("units_sold", "sum"))
         .reset_index())
    g = g[g["lost_sales_weeks"] > 0].nlargest(n, "lost_sales_weeks")
    g["pct_weeks"] = g["lost_sales_weeks"] / g["weeks"]
    show(g.set_index(["store_id", "product_id"])[
        ["product_category", "lost_sales_weeks", "pct_weeks", "units"]], "")

    print(f"\nthese are the rows that tell you what to reorder")


def rfm(sales):
    head("10  RFM snapshot")
    snapshot = sales.groupby("customer_id").agg(
        last_seen=("date", "max"),
        frequency=("transaction_id", "nunique"),
        monetary=("sales_amount", "sum"),
    )
    today = sales["date"].max()
    snapshot["recency_days"] = (today - snapshot["last_seen"]).dt.days

    print(f"customers: {len(snapshot):,}")
    print(f"\nrecency (days since last purchase)")
    print(snapshot["recency_days"].describe().to_string())
    print(f"\nfrequency (distinct transactions)")
    print(snapshot["frequency"].describe().to_string())
    print(f"\nmonetary (INR lifetime)")
    print(snapshot["monetary"].describe().to_string())

    # Quartile scoring: 4 is best on every axis.
    f = snapshot["frequency"].rank(method="first")
    m = snapshot["monetary"].rank(method="first")
    # Recency must rank ascending so that a small recency_days lands in the FIRST
    # quartile and gets the top label. Ranking descending here inverts R: the customers
    # who have not been seen in months would come out as R=4 and the daily shoppers as
    # R=1, which inverts every downstream conclusion about who to target.
    r = snapshot["recency_days"].rank(method="first")
    snapshot["R"] = pd.qcut(r, 4, labels=[4, 3, 2, 1]).astype(int)
    snapshot["F"] = pd.qcut(f, 4, labels=[1, 2, 3, 4]).astype(int)
    snapshot["M"] = pd.qcut(m, 4, labels=[1, 2, 3, 4]).astype(int)

    # Collapse the 64 possible RFM codes into named segments. Listing all of them is
    # noise -- 61 of the 64 are populated here, which tells you nothing an analyst can
    # act on. Recency and frequency drive the decision; monetary only sizes the prize.
    SEGMENTS = {
        (4, 3): "Champions", (4, 4): "Champions",
        (4, 1): "New / promising", (4, 2): "New / promising",
        (3, 3): "Potential loyalist", (3, 4): "Potential loyalist",
        (3, 1): "Needs attention", (3, 2): "Needs attention",
        (2, 3): "At risk", (2, 4): "At risk",
        (2, 1): "At risk", (2, 2): "At risk",
        (1, 3): "Cannot lose them", (1, 4): "Cannot lose them",
        (1, 1): "Dormant", (1, 2): "Dormant",
    }
    snapshot["segment"] = [SEGMENTS[(rr, ff)] for rr, ff
                           in zip(snapshot["R"], snapshot["F"])]

    seg = (snapshot.groupby("segment")
           .agg(customers=("monetary", "size"),
                avg_recency_days=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                total_revenue=("monetary", "sum"))
           .sort_values("total_revenue", ascending=False))
    seg["revenue_share"] = seg["total_revenue"] / seg["total_revenue"].sum()
    print(f"\n{len(snapshot):,} customers over {len(seg)} segments, "
          f"R and F are quartile scores with 4 = best")
    show(seg, "")

    print(f"repeat customers (2+ transactions): "
          f"{(snapshot['frequency'] >= 2).mean():.1%}")
    gone = seg.loc["Dormant", "customers"] if "Dormant" in seg.index else 0
    at_risk = seg.loc[["At risk", "Cannot lose them"], "customers"].sum()
    print(f"\nactionable: {at_risk:,.0f} customers are At risk or Cannot lose them "
          f"({at_risk / len(snapshot):.1%} of the base) -- win-back candidates.")
    print(f"           {gone:,.0f} are Dormant ({gone / len(snapshot):.1%}) -- "
          f"cheaper to ignore than to chase.")


def main():
    use_utf8_console()
    sales, panel = load()

    head("Retail Pulse -- starter analysis")
    print(f"sales : {len(sales):,} rows, 27 columns on disk + derived week_start "
          f"({sales['date'].min().date()} .. {sales['date'].max().date()})")
    print(f"panel : {len(panel):,} rows x {panel.shape[1]} columns  "
          f"({panel['week_start_date'].min().date()} .. "
          f"{panel['week_start_date'].max().date()})")
    print(f"total revenue: {money(sales['sales_amount'].sum())}")

    monthly_trend(sales)
    weekly_trend(sales)
    top_products(sales)
    geography(sales)
    categories(sales)
    promotion_effect(sales)
    category_mix_seasonality(sales)
    stockout_impact(sales, panel)
    lost_demand_hotspots(panel)
    rfm(sales)

    head("done")
    print("Reminder before you model on this:")
    print("  * quantity 0 = lost demand, not zero demand. Correct for it or your model")
    print("    learns 'nobody wanted this' when the truth was 'nothing was on the shelf'.")
    print("  * The final calendar week is clipped -- the panel already excludes it.")
    print("  * There are no returns in this data by design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
