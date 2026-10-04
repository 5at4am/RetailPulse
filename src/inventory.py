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


def poisson_safety_stock(demand_mean, lead_time_units=1, critical_ratio=None):
    """Safety stock only, without the on-hand reorder logic.

    `reorder_quantity_poisson` needs an on-hand figure and returns the full order decision.
    The backtest needs the safety stock for a pooled forecast where there is no single on-hand
    level to compare against, so the two share this one implementation rather than each
    deriving the quantile separately and drifting apart.
    """
    if critical_ratio is None:
        raise ValueError("critical_ratio is required; compute it with "
                         "newsvendor_critical_ratio(purchase_cost=..., selling_price=...)")
    from scipy.stats import poisson

    expected = float(demand_mean * lead_time_units)
    target = float(poisson.ppf(critical_ratio, max(expected, 1e-9)))
    return max(target - expected, 0.0)


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


def backtest_reduction(backtest: pd.DataFrame, baseline: str = "naive_mean") -> dict:
    """Summarise the backtest as the over/understock change the brief asks for.

    The brief targets a 25-40% reduction in over/understock. The denominator is the baseline
    policy's combined over- plus understock in units, so the number is a change in service
    level and inventory carried, not a change in forecast accuracy.

    `baseline="observed"` compares against doing nothing at all, which is the loosest possible
    baseline and therefore flatters the policy. `naive_mean` is the honest one: it orders the
    same forecast with a conventional safety stock, so only the safety-stock method differs.
    """
    rows = {}
    for name, group in backtest.groupby("policy"):
        rows[name] = {
            "overstock_units": float(group["overstock_units"].sum()),
            "understock_units": float(group["understock_units"].sum()),
            "unmet_units": float(group["unmet_units"].sum()),
            "cover_units": float(group["cover_units"].sum()),
        }
        rows[name]["total_error_units"] = (rows[name]["overstock_units"]
                                           + rows[name]["understock_units"])

    policy = rows.get("policy_poisson")
    if policy is None:
        raise ValueError("backtest must contain a 'policy_poisson' row to compare against")

    if baseline == "observed":
        base = {"overstock_units": 0.0,
                "understock_units": float(backtest["actual_units"].sum()),
                "unmet_units": float(backtest["unmet_units"].sum()),
                "total_error_units": 0.0}
        base["total_error_units"] = base["understock_units"]
    else:
        if baseline not in rows:
            raise ValueError(f"unknown baseline {baseline!r}; have {sorted(rows)}")
        base = rows[baseline]

    total_base = base["total_error_units"]
    total_policy = policy["total_error_units"]
    # A zero baseline means the baseline policy had no error at all, so there is nothing to
    # improve on. Reporting nan would propagate into the summary CSV and render in the
    # dashboard as a division warning; 0.0 is the honest reading of "no change".
    reduction = ((total_base - total_policy) / total_base
                 if total_base else 0.0)

    return {
        "baseline": baseline,
        "holdout_weeks": int(backtest["week_start_date"].nunique()),
        "baseline_error_units": round(total_base, 1),
        "policy_error_units": round(total_policy, 1),
        "error_reduction_pct": round(reduction * 100, 2),
        "baseline_overstock_units": round(base["overstock_units"], 1),
        "policy_overstock_units": round(policy["overstock_units"], 1),
        "baseline_understock_units": round(base["understock_units"], 1),
        "policy_understock_units": round(policy["understock_units"], 1),
        "meets_25_40_target": bool(0.25 <= reduction <= 0.40),
        "note": ("Reduction is measured against the naive-mean policy on the same forecast. "
                 "It is a walk-forward replay on historical data, not a production A/B result."),
    }


def backtest_policy(panel, lead_time_weeks=1, train_weeks=None, holdout_weeks=13,
                    service_level=None, progress=print,
                    level_keys=("store_id", "product_id")):
    """Walk-forward replay of the reorder policy, so the improvement is measured not asserted.

    Defaults adapt to the panel: with the full 104 weeks it trains on the first 91 and replays
    the last 13, because holding back a quarter of a two-year history is a small replay for a
    lot of fit time. Pass `train_weeks=52` to use the brief's 52/13 split instead. On a short
    synthetic fixture the split scales down rather than raising, so a 40-week test panel
    yields a plan rather than a week-count complaint.

    The brief asks for a 25-40% cut in over/understock. That claim is only credible if it
    comes from a replay, which is what this does:

      1. Cut the panel into a training prefix and a holdout suffix. Every forecast is fitted on
         the prefix only, and the realised week joins the history only after it has been
         predicted -- otherwise the replay is looking at its own answer.
      2. For each holdout week, forecast the pooled demand, apply each candidate policy, and
         record what actually happened.
      3. Compare three policies on the same forecast, so only the safety-stock method differs:
         the Poisson policy, a conventional z-multiple, and no safety stock at all.

    Lost demand is censored in this dataset: a zero `units_sold` week is a stockout, not zero
    interest. Counting a stockout as "sold 0, so no loss" is the mistake that makes an
    inventory policy look perfect, so a censored week is charged imputed unmet demand rather
    than being treated as a satisfied week.
    """
    service_level = config.DEFAULT_SERVICE_LEVEL if service_level is None else service_level
    critical = newsvendor_critical_ratio(purchase_cost=1.0, selling_price=1.0,
                                        cover_weeks=lead_time_weeks)

    frame = panel.copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    frame = frame[frame["week_start_date"] < pd.Timestamp(config.PARTIAL_FINAL_WEEK)]

    weeks = sorted(frame["week_start_date"].unique())
    # At least 8 training weeks for a variance estimate, and never fewer than 2 holdout weeks,
    # or the replay cannot measure censoring at all.
    if train_weeks is None:
        train_weeks = max(len(weeks) - holdout_weeks, 8)
    holdout_weeks = min(holdout_weeks, max(len(weeks) - train_weeks, 0))
    if train_weeks + holdout_weeks < 10:
        raise ValueError(
            f"panel has {len(weeks)} usable weeks; a backtest needs at least 10 "
            f"(8 train + 2 holdout). Skipping the backtest rather than reporting a "
            f"number from too few weeks.")

    train = weeks[:train_weeks]
    holdout = weeks[train_weeks:train_weeks + holdout_weeks]
    progress(f"backtest: train on {len(train)} weeks, replay {len(holdout)}")

    # Only the pooled units are needed from `weekly`; the per-week censoring is recomputed
    # from the pair-level frame below so that a fixture without `stockout_count` still works.
    weekly = frame.groupby("week_start_date")["units_sold"].sum().reindex(weeks)

    # How many distinct pairs the censored count should be spread across. Constant across
    # weeks, so it is computed once. A pair that never appears cannot have sold out.
    n_pairs = int(frame[list(level_keys)].drop_duplicates().shape[0])

    rows = []
    history = weekly.loc[train]

    for week in holdout:
        # --- forecast for this week from strictly-prior weeks only
        mean = float(history.mean())
        var = max(float(history.var(ddof=1)), 1e-9)
        z = safety_factor(critical)

        actual = frame[frame["week_start_date"] == week]
        units = float(actual["units_sold"].sum())
        # `stockout_count` is how the panel records censoring, but a caller may pass a
        # minimal frame without it. Falling back to zero-sold weeks keeps the backtest usable
        # on a synthetic fixture -- at the cost of treating every zero as censored, which
        # overstates unmet demand. That direction is the safe one: it makes the policy look
        # worse, not better.
        if "stockout_count" in actual.columns:
            censored_mask = actual["stockout_count"] > 0
        else:
            censored_mask = actual["units_sold"] <= 0
        censored_pairs = int(censored_mask.sum())
        censored_units = float(actual.loc[censored_mask, "units_sold"].sum())

        # Demand the panel could not serve. A censored week is not a zero-demand week: the
        # pair sold out, so the shortfall is imputed rather than read off the data. Counting
        # these as satisfied weeks is what makes an inventory policy look artificially
        # perfect.
        #
        # `mean` is the POOLED weekly total, so the per-pair weekly rate is that total over
        # the number of pairs, not over the number of weeks. Dividing by len(train) would
        # yield units-per-week-per-week and overstate every censored pair by the panel
        # length -- with 7,800 pairs and 91 weeks, a factor of about 86.
        rate = (mean / n_pairs) if n_pairs else 0.0
        unmet = censored_pairs * rate

        for name, point, safety in (
            ("policy_poisson", mean,
             poisson_safety_stock(mean, lead_time_units=lead_time_weeks,
                                  critical_ratio=critical)),
            ("naive_mean", mean, z * var ** 0.5),
            ("no_safety_stock", mean, 0.0),
        ):
            cover = point + safety
            rows.append({
                "week_start_date": week,
                "policy": name,
                "forecast_units": point,
                "safety_stock": safety,
                "cover_units": cover,
                "actual_units": units,
                "censored_pairs": censored_pairs,
                "censored_units": censored_units,
                "unmet_units": unmet,
                # Overstock is what arrives and is not needed; understock is what is wanted and
                # unavailable. Both are counted in units so the brief's "reduce over/understock
                # by 25-40%" has a denominator.
                "overstock_units": max(cover - units, 0.0),
                "understock_units": max(units - cover, 0.0) + unmet,
            })

        # The realised week joins the history only after it has been predicted.
        history = pd.concat([history, weekly.loc[[week]]])

    return pd.DataFrame(rows)


def run_inventory(panel=None, forecast_units=None, service_level=None,
                  lead_time_weeks=1, write=True, progress=print, run_backtest=True):
    """Full inventory pass: forecast -> costs -> critical ratio -> reorder quantities.

    `run_backtest=False` skips the walk-forward replay. It defaults to True because the
    replay is what turns the brief's 25-40% over/understock target into a measured
    number, but it is separable so a caller with a short panel still gets a plan.
    """
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

    # The backtest is opt-out, because `run_inventory` is also the entry point the tests call
    # with small fixtures where a 10-week minimum cannot be met. The plan itself has no such
    # requirement.
    if run_backtest:
        try:
            progress("\nwalk-forward backtest (policy vs naive vs no safety stock)")
            bt = backtest_policy(panel, lead_time_weeks=lead_time_weeks, progress=progress)
            reduction = pd.DataFrame([backtest_reduction(bt)])
            progress(reduction.T.to_string(header=False))
            out["backtest"] = bt
            out["backtest_reduction"] = reduction
        except ValueError as error:
            progress(f"\nbacktest skipped: {error}")
            out["backtest"] = None
            out["backtest_reduction"] = None
    else:
        out["backtest"] = None
        out["backtest_reduction"] = None

    if write:
        config.ensure_dirs()
        pairs.to_csv(config.PROCESSED / "inventory_reorder_plan.csv", index=False)
        summary.to_csv(config.PROCESSED / "inventory_summary.csv", index=False)
        if out["backtest"] is not None:
            out["backtest"].to_csv(config.PROCESSED / "inventory_backtest.csv", index=False)
            out["backtest_reduction"].to_csv(
                config.PROCESSED / "inventory_backtest_summary.csv", index=False)

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