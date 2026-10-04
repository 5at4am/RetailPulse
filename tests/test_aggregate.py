"""The daily/weekly roll-ups must reconcile back to the line-item totals, and the clipped
trailing week must be handled the way the shipped panel handles it.
"""
import datetime as dt

import pytest

import retail_pulse as rp


def test_daily_reconciles_exactly(sales):
    raw_units = sum(r["quantity_sold"] for r in sales)
    raw_rev = sum(r["sales_amount"] for r in sales)

    daily = rp.aggregate_daily(sales)

    # 731 days, nothing clipped, so every unit and every rupee must survive the roll-up.
    assert len(daily) == 731
    assert sum(a["units"] for a in daily) == raw_units
    assert abs(sum(a["revenue"] for a in daily) - raw_rev) < 0.01
    assert sum(a["lines"] for a in daily) == len(sales)


def test_weekly_drops_the_clipped_final_week(sales):
    weekly = rp.aggregate_weekly(sales)

    assert len(weekly) == 104, "expected the same 104 complete weeks as the shipped panel"
    assert all(a["days_covered"] == 7 for a in weekly)
    assert all(a["weekly_start"].weekday() == 0 for a in weekly), "weeks are Monday-start"
    assert weekly[0]["weekly_start"] == dt.date(2024, 1, 1)
    assert weekly[-1]["weekly_start"] == dt.date(2025, 12, 22)


def test_the_dropped_week_is_the_clipped_one(sales):
    kept = rp.aggregate_weekly(sales, drop_partial=True)
    all_weeks = rp.aggregate_weekly(sales, drop_partial=False)
    typical = sorted(a["revenue"] for a in kept)[51]

    assert len(all_weeks) - len(kept) == 1
    clipped = all_weeks[-1]
    assert clipped["weekly_start"] == dt.date(2025, 12, 29)
    assert clipped["days_covered"] == 3
    # The point of dropping it: it would otherwise look like a ~63% demand collapse.
    assert clipped["revenue"] / typical < 0.5
    # Units only reconcile once the clipped week is put back.
    assert sum(a["units"] for a in all_weeks) == sum(r["quantity_sold"] for r in sales)


def test_grouped_rollup_loses_no_revenue(sales):
    weekly = rp.aggregate_weekly(sales)
    by_city = rp.aggregate_weekly(sales, by=("city",))

    assert len(by_city) == len(weekly) * 12
    assert abs(sum(x["revenue"] for x in by_city)
               - sum(a["revenue"] for a in weekly)) < 0.01


def test_grouping_does_not_clip_subgroups(sales):
    """drop_partial is a property of the WEEK, never of a (week, city) subgroup.

    A city can have no sales on a given day while the week as a whole is complete.
    Deciding per group would silently delete that city's revenue.
    """
    ungrouped = {a["weekly_start"]: a for a in rp.aggregate_weekly(sales)}
    for row in rp.aggregate_weekly(sales, by=("city",)):
        assert row["weekly_start"] in ungrouped


def test_edge_cases(sales):
    first_day = min(r["date"] for r in sales)
    one_day = [r for r in sales if r["date"] == first_day]
    one_week = [r for r in sales
                if first_day <= r["date"] <= first_day + dt.timedelta(days=6)]

    assert len(rp.aggregate_daily(one_day)) == 1
    assert len(rp.aggregate_weekly(one_day)) == 0, "3 days is not a complete week"
    assert len(rp.aggregate_weekly(one_week)) == 1

    with pytest.raises(ValueError):
        rp.aggregate(sales, "monthly")

    assert rp.aggregate([]) == []
    assert rp.aggregate([], "weekly") == []