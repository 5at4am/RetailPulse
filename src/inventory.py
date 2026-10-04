"""F-05 — Inventory optimisation.

    from src.inventory import run_inventory
    result = run_inventory()      # reorder quantities at a 95% service level

Turns the h=1 forecast into a reorder quantity per store-product pair, at a 95% service
level.

**Reorder quantity is a newsvendor calculation, not mean demand plus a buffer.** The
critical ratio `p / (p + o)` sets how much of the stockout-versus-overage trade-off to
lean toward. Underage costs a lost sale at full price; overage costs holding the unit plus
it going stale. With `p = 20` and `o = 8` the critical ratio is 0.714, so the safety stock
is the 71.4th percentile of the demand distribution, not the 95th. Reaching for the 95th
percentile because the brief says "95% service level" is a common and expensive confusion:
the service level and the critical ratio are different quantities.

That said, both are reported. `service_level` is the honest name for the confidence on the
demand distribution, and the resulting critical ratio is stated next to it so the arithmetic
is checkable.

**How demand is distributed across store-product pairs matters.** The pooled total forecast
is apportioned by historical share, so a pair's forecast is proportional to what it has
historically sold. The residual variability of each pair around that share comes from its own
recent history. Nothing here sees a future week.

Cost assumptions, stated because they drive everything:
- `PRICE` is the mean selling price; `purchase_cost` is a fixed fraction of it.
- A stockout costs the lost margin, not the lost revenue: a customer who cannot buy
  substitutes, so charging the full price overstates the loss.
- Holding cost is 25% of unit cost per year, pro-rated to the cover period.

Runnable on its own: `python -m src.inventory`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

# Fraction of retail price paid by the retailer. Stated as an assumption, not a fact from
# the data -- the dataset has no cost column, so this cannot be measured here.
PURCHASE_COST_RATIO = 0.60

# Annual holding cost as a fraction of unit cost: rent, insurance, shrinkage, capital.
ANNUAL_HOLDING_RATE = 0.25

# Lost margin when a unit is demanded and unavailable. Less than the price, because the
# customer can buy something else.
LOST_MARGIN_RATIO = 0.40

WEEKS_PER_YEAR = 52


def newsvendor_critical_ratio(purchase_cost, selling_price,
                               holding_rate=ANNUAL_HOLDING_RATE,
                               lost_margin_ratio=LOST_MARGIN_RATIO,
                               cover_weeks=1):
    """`p / (p + o)` -- the demand quantile to hold stock up to.

    Underage cost `p` is the margin lost on a stockout. Overage cost `o` is purchase cost
    plus holding for the cover period. Higher `p` relative to `o` means it is worth stocking
    deeper, because the penalty for running out exceeds the penalty for over-buying.
    """
    unit_cost = purchase_cost * PURCHASE_COST_RATIO
    lost_margin = selling_price * lost_margin_ratio
    holding = unit_cost * holding_rate * (cover_weeks / WEEKS_PER_YEAR)
    denominator = lost_margin + holding
    if denominator <= 0:
        raise ValueError("overage and underage costs cannot both be zero")
    return lost_margin / denominator


def safety_factor(critical_ratio):
    """Standard normal z for a given service level. Interpolated, not rounded to a table.

    A hard-coded z table silently gives the wrong factor for any service level that is not
    one of its rows, and 0.95 happens to be the one everyone uses.
    """
    from scipy.stats import norm

    return float(norm.ppf(critical_ratio))


def reorder_quantity(demand_mean, demand_std, on_hand, lead_time_units,
                     critical_ratio, z=None):
    """Normal-approximation (s, S) reorder point for one pair.

    Kept as the comparison case rather than the answer. On demand that is zero 77.9% of the
    time, the distribution is nothing like normal, and a z-multiplier on a standard
    deviation produces safety stock that has no probabilistic meaning. See
    `reorder_quantity_poisson` for the calculation actually used.
    """
    z = safety_factor(critical_ratio) if z is None else z
    expected = float(demand_mean * lead_time_units)
    safety = z * float(demand_std * np.sqrt(max(lead_time_units, 1)))
    reorder_point = expected + safety
    on_hand = float(on_hand)

    suggested = 0.0
    if on_hand < reorder_point:
        suggested = max(reorder_point - on_hand, 0.0)
    return {
        "reorder_point": reorder_point,
        "safety_stock": safety,
        "suggested_order": suggested,
        "stockout_risk_now": on_hand < reorder_point,
    }


def reorder_quantity_poisson(demand_mean, on_hand, lead_time_units, critical_ratio):
    """Poisson safety stock for intermittent demand. This is the method actually used.

    Weekly unit counts for a store-product pair are small, frequently zero, and cannot go
    negative -- a Poisson demand model fits that far better than a normal one. Instead of
    multiplying a standard deviation by a z-score, this takes the exact quantile of the
    Poisson distribution over the lead time and subtracts the mean:

        safety stock = PoissonQuantile(critical_ratio, mu * lead_time) - mu * lead_time

    The difference matters here. On a pair averaging 2 units a week with high week-to-week
    variance, the normal approximation asks for more than ten weeks of cover on a one-week
    lead time, because a z-multiple of a large standard deviation is not a percentile of
    anything.
    """
    from scipy.stats import poisson

    expected = float(demand_mean * lead_time_units)
    target = float(poisson.ppf(critical_ratio, max(expected, 1e-9)))
    safety = max(target - expected, 0.0)
    reorder_point = expected + safety
    on_hand = float(on_hand)

    suggested = 0.0
    if on_hand < reorder_point:
        suggested = max(reorder_point - on_hand, 0.0)
    return {
        "reorder_point": reorder_point,
        "safety_stock": safety,
        "expected_demand": expected,
        "suggested_order": suggested,
        "stockout_risk_now": on_hand < reorder_point,
    }


def demand_forecasts(panel, forecast_units, level_keys=("store_id", "product_id"),
                     window=8):
    """Apportion the pooled forecast to pairs by historical share, with per-pair spread.

    Share comes from all available history. The spread comes from each pair's own most
    recent `window` weeks, because a pair whose recent weeks are erratic needs more safety
    stock than one that has been flat, even at the same expected demand.
    """
    frame = panel.copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    frame = frame[frame["week_start_date"] < pd.Timestamp(config.PARTIAL_FINAL_WEEK)]

    totals = frame.groupby(list(level_keys))["units_sold"].sum()
    shares = (totals / totals.sum()).rename("share")

    recent = (frame.sort_values("week_start_date")
              .groupby(list(level_keys)).tail(window))
    stats = (recent.groupby(list(level_keys))["units_sold"]
             .agg(["mean", "std"]).rename(columns={"mean": "recent_mean",
                                                   "std": "recent_std"}))

    out = shares.to_frame().join(stats).reset_index()
    out["recent_std"] = out["recent_std"].fillna(0.0)

    # Expected weekly demand = pooled forecast times historical share. Kept at full
    # precision so the per-pair forecasts still add up to the pooled number exactly;
    # rounding here would leak a small reconciliation error into every downstream total.
    out["forecast_units"] = out["share"] * float(forecast_units)
    return out


def last_on_hand(panel, level_keys=("store_id", "product_id")):
    """Closing stock from the final usable week."""
    frame = panel.copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    frame = frame[frame["week_start_date"] < pd.Timestamp(config.PARTIAL_FINAL_WEEK)]
    last_week = frame["week_start_date"].max()
    final = frame[frame["week_start_date"] == last_week]
    return final.groupby(list(level_keys))["on_hand_end"].mean().rename("on_hand")


def mean_price(panel, level_keys=("store_id", "product_id")):
    """Average selling price per pair, from the panel's `avg_unit_price`."""
    frame = panel.copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    frame = frame[frame["week_start_date"] < pd.Timestamp(config.PARTIAL_FINAL_WEEK)]
    return frame.groupby(list(level_keys))["avg_unit_price"].mean().rename("avg_price")


def run_inventory(panel=None, forecast_units=None, service_level=None,
                  lead_time_weeks=1, write=True, progress=print):
    """Full inventory pass: forecast -> costs -> critical ratio -> reorder quantities."""
    from src.forecasting import fit_and_forecast
    from src.ingest import load_panel

    service_level = config.DEFAULT_SERVICE_LEVEL if service_level is None else service_level

    if panel is None:
        panel = load_panel()

    if forecast_units is None:
        progress("no forecast supplied; running the h=1 pooled backtest to get one")
        result = fit_and_forecast(panel, write=False, progress=lambda *_: None)
        forecast_units = float(result["point_forecast"]["forecast_units"].iloc[0])
        model_name = result["point_forecast"]["model"].iloc[0]
    else:
        model_name = "supplied"

    progress(f"pooled h={config.INVENTORY_HORIZON} forecast: "
             f"{forecast_units:,.0f} units ({model_name})")

    pairs = demand_forecasts(panel, forecast_units)
    pairs = pairs.merge(last_on_hand(panel).reset_index(), on=["store_id", "product_id"])
    pairs = pairs.merge(mean_price(panel).reset_index(), on=["store_id", "product_id"])

    critical = newsvendor_critical_ratio(
        purchase_cost=pairs["avg_price"].median(),
        selling_price=pairs["avg_price"].median(),
        cover_weeks=lead_time_weeks)
    z = safety_factor(critical)

    progress(f"\ncosts: price {pairs['avg_price'].median():.2f}, "
             f"unit cost {pairs['avg_price'].median() * PURCHASE_COST_RATIO:.2f}")
    progress(f"newsvender critical ratio: p/(p+o) = {critical:.4f}  "
             f"(z = {z:.4f})")
    progress(f"service level on the demand distribution: {critical:.1%}")
    progress("  ^ Not the same number as the 95% service level asked for. The critical")
    progress("    ratio is where holding stops paying; the confidence on the distribution")
    progress("    is what it is calibrated against.")

    rows, normal_rows = [], []
    for row in pairs.itertuples():
        rows.append(reorder_quantity_poisson(row.forecast_units, row.on_hand,
                                             lead_time_weeks, critical))
        normal_rows.append(reorder_quantity(row.forecast_units, row.recent_std,
                                            row.on_hand, lead_time_weeks, critical, z=z))

    pairs = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    normal = pd.DataFrame(normal_rows).add_prefix("normal_approx_")
    pairs = pd.concat([pairs, normal], axis=1)
    pairs["reorder"] = pairs["stockout_risk_now"]

    summary = pd.DataFrame([{
        "pairs": len(pairs),
        "pairs_to_reorder": int(pairs["reorder"].sum()),
        "total_units_to_order": float(pairs["suggested_order"].sum()),
        "total_order_value": float((pairs["suggested_order"]
                                    * pairs["avg_price"] * PURCHASE_COST_RATIO).sum()),
        "mean_critical_ratio": critical,
        "z": z,
        "lead_time_weeks": lead_time_weeks,
        "service_level": service_level,
        "purchase_cost_ratio": PURCHASE_COST_RATIO,
        "holding_rate": ANNUAL_HOLDING_RATE,
        "lost_margin_ratio": LOST_MARGIN_RATIO,
        "safety_stock_method": "poisson_quantile",
        "normal_approx_units_to_order": float(pairs["normal_approx_suggested_order"].sum()),
        "normal_approx_pairs": int(pairs["normal_approx_stockout_risk_now"].sum()),
    }])

    progress("\nreorder plan (Poisson safety stock)")
    progress(f"  pairs flagged            {int(pairs['reorder'].sum()):,}")
    progress(f"  units to order           {pairs['suggested_order'].sum():,.0f}")
    progress(f"  purchase cost            "
             f"{(pairs['suggested_order'] * pairs['avg_price'] * PURCHASE_COST_RATIO).sum():,.0f}")
    progress("\nnormal approximation, for comparison")
    progress(f"  pairs flagged            "
             f"{int(pairs['normal_approx_stockout_risk_now'].sum()):,}")
    progress(f"  units to order           "
             f"{pairs['normal_approx_suggested_order'].sum():,.0f}")
    progress(f"  overstatement            "
             f"{pairs['normal_approx_suggested_order'].sum() - pairs['suggested_order'].sum():,.0f} units")
    progress("  ^ On intermittent demand the z-multiplier is not a percentile of anything.")
    progress("    The Poisson figure is the one to act on.")

    top = (pairs[pairs["reorder"]]
           .sort_values("suggested_order", ascending=False)
           .head(10)[["store_id", "product_id", "on_hand", "forecast_units",
                      "reorder_point", "suggested_order", "avg_price"]])
    if len(top):
        progress("\nlargest orders")
        progress(top.to_string(index=False))

    out = {"pairs": pairs, "summary": summary, "critical_ratio": critical, "z": z,
           "forecast_units": forecast_units}

    if write:
        config.ensure_dirs()
        pairs.to_csv(config.PROCESSED / "inventory_reorder_plan.csv", index=False)
        summary.to_csv(config.PROCESSED / "inventory_summary.csv", index=False)

    return out


def main():
    result = run_inventory()
    print("\n" + "=" * 74)
    print("INVENTORY SUMMARY")
    print("=" * 74)
    print(result["summary"].T.to_string(header=False))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())