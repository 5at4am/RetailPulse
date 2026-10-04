"""Tests for the F-06 dashboard.

**Why `AppTest` rather than checking that files import.** Importing a Streamlit page proves
nothing: the page body runs inside a function, so a typo three lines below the title is
invisible until someone opens the page in a demo. `AppTest` executes the page the same way
Streamlit does and raises on any exception, which means these tests catch the failure mode
that actually matters for a graded live demo.

Each page gets:
- a render test that fails on any exception,
- an assertion that the headline number is on the page,
- and for the pages with a claim worth protecting, a test that the honest framing survives.

The last category is the important one. A dashboard that quietly stops saying "the AUC
target was missed" still passes every render test, and that is precisely the regression
worth guarding.
"""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import dashboard_data as dd
from src import config


def run_page(relative_path):
    return AppTest.from_file(str(config.ROOT / relative_path), default_timeout=120).run()


PAGES = {
    "home": "app/Home.py",
    "forecasting": "app/Demand_Forecasting.py",
    "segments": "app/Customer_Segments.py",
    "churn": "app/Churn_Risk.py",
    "inventory": "app/Inventory_Recommendations.py",
}


# ------------------------------------------------------------------ every page renders

@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_renders_without_error(page):
    at = run_page(PAGES[page])
    assert not at.exception, f"{page} raised: {at.exception}"
    assert len(at.title) > 0, f"{page} rendered no title"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_renders_twice_without_error(page):
    """Second render exercises the cached path, which is what a real viewer hits.

    The first render populates `st.cache_data`; the second reads from it. A page that works
    once and fails on reload would look fine in development and break in a live demo.
    """
    AppTest.from_file(str(config.ROOT / PAGES[page]), default_timeout=120).run()
    second = run_page(PAGES[page])
    assert not second.exception, f"{page} failed on second render: {second.exception}"


# ------------------------------------------------------------------------ the data layer

def test_aggregates_all_load_within_the_budget():
    import time

    start = time.perf_counter()
    files = sorted(dd.PROCESSED.glob("*.csv"))
    for path in files:
        pd.read_csv(path)
    elapsed = time.perf_counter() - start

    assert len(files) > 0
    assert elapsed < dd.LOAD_BUDGET_SECONDS, (
        f"loading all {len(files)} aggregates took {elapsed:.2f}s, over the "
        f"{dd.LOAD_BUDGET_SECONDS}s budget")


def test_dashboard_never_reads_the_raw_panel(monkeypatch):
    """The 8-second budget exists because the raw panel is 63 MB.

    A guard, not a description: if someone adds a raw read to a page for convenience, this
    fails instead of quietly costing ten seconds per session. Patching `read_csv` catches
    `load()` too, because that is the only door to disk the app layer has.
    """
    seen = []
    real = pd.read_csv

    def spy(path, *args, **kwargs):
        seen.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)

    for page in PAGES.values():
        at = run_page(page)
        assert not at.exception, f"{page} failed while auditing reads"

    raw_dir = str(dd.PROCESSED.parent / "raw")
    offenders = [p for p in seen if raw_dir in p]
    assert not offenders, f"pages read from data/raw/: {offenders}"


def test_every_read_by_the_dashboard_lands_in_processed():
    """Same intent as above, from the other side: assert the positive property.

    The cache has to be cleared first. `st.cache_data` lives for the whole process, so
    without this the pages read nothing at all - the aggregates are already cached from an
    earlier test - and the spy sees an empty list. An audit that passes because it observed
    nothing is worse than no audit, hence the explicit assertion that reads happened.
    """
    try:
        import streamlit as st
        st.cache_data.clear()
    except Exception:
        pass

    real = pd.read_csv
    seen = []

    def spy(path, *args, **kwargs):
        seen.append(str(path))
        return real(path, *args, **kwargs)

    pd.read_csv = spy
    try:
        for page in PAGES.values():
            assert not run_page(page).exception
    finally:
        pd.read_csv = real
        try:
            import streamlit as st
            st.cache_data.clear()
        except Exception:
            pass

    assert seen, "the spy recorded nothing, so the audit proved nothing"
    # Compare resolved paths: the pages hand `pd.read_csv` a Path, and comparing strings
    # would depend on how the OS spells it. `.resolve()` also collapses any `..`.
    processed = dd.PROCESSED.resolve()
    for path in seen:
        assert Path(path).resolve().is_relative_to(processed), \
            f"read outside data/processed/: {path}"


def test_missing_aggregate_explains_how_to_fix_it():
    with pytest.raises(FileNotFoundError, match="src.precompute"):
        dd.load("definitely_not_a_real_file.csv", cache=False)


def test_unknown_kpi_fails_loudly():
    """A typo'd metric name must raise, not silently return a blank dashboard."""
    kpis = dd.load_kpis()
    with pytest.raises(KeyError, match="known KPI"):
        dd.kpi_value(kpis, "reveneu")


# ------------------------------------------------------------------------ INR formatting

@pytest.mark.parametrize("value,expected", [
    (210242624.35, "21,02,42,624"),
    (123456789, "12,34,56,789"),
    (100000, "1,00,000"),
    (1000, "1,000"),
    (999, "999"),
    (0, "0"),
    (-5000, "-5,000"),
])
def test_inr_uses_lakh_and_crore_grouping(value, expected):
    assert dd.inr(value) == expected


def test_inr_shows_decimals_when_asked():
    assert dd.inr(1234567.891, 2) == "12,34,567.89"


def test_inr_groups_every_magnitude_without_losing_a_digit():
    for value in (999, 1000, 99999, 100000, 9999999, 10000000, 210242624.35):
        digits = dd.inr(value).replace(",", "").replace(".", "").lstrip("-")
        assert digits == str(int(abs(value))), f"lost digits formatting {value}"


# ---------------------------------------------------------------------- home page claims

def test_home_states_the_missed_forecast_target_rather_than_claiming_it():
    """Pooled WAPE is not the brief's MAPE target. The page must say so out loud."""
    at = run_page("app/Home.py")
    assert not at.exception

    text = " ".join(str(m.value) for m in at.markdown).lower()
    assert "mape" in text
    assert "target" in text


def test_home_shows_the_promotion_finding_as_measured():
    promo = dd.load("promotion_lift.csv").iloc[0]
    at = run_page("app/Home.py")

    # The headline goes through st.info, the supporting numbers through st.markdown. Checking
    # only one of them is how a test ends up asserting on an empty string and passing for
    # the wrong reason.
    text = " ".join([str(i.value) for i in at.info]
                    + [str(m.value) for m in at.markdown])

    assert f"{promo['volume_lift']:.4f}" in text, "the measured lift must be stated"
    # The configured multiplier must never be presented as an observed effect.
    assert "configured multiplier" in text


def test_home_covers_all_six_requirements():
    at = run_page("app/Home.py")
    frames = [pd.DataFrame(f.value) for f in at.dataframe]
    joined = " ".join(f.to_csv() for f in frames)
    for requirement in ("F-01", "F-02", "F-03", "F-04", "F-05", "F-06"):
        assert requirement in joined


# ------------------------------------------------------------------ forecasting claims

def test_forecasting_page_carries_the_reported_wape():
    """The headline accuracy on the page must equal the number in the backtest file."""
    scores = dd.load("forecast_backtest.csv")
    point = dd.load("forecast_next_weeks.csv").iloc[0]
    at = run_page("app/Demand_Forecasting.py")

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Next week forecast"] == f"{point['forecast_units']:,.0f} units"

    prophet_h1 = float(scores[(scores["horizon"] == 1)
                              & (scores["model"] == "prophet")]["wape"].iloc[0])
    assert dd.pct(prophet_h1, 2) in " ".join(str(v) for v in metrics.values())


def test_forecasting_page_explains_the_ensemble_did_not_win():
    at = run_page("app/Demand_Forecasting.py")
    text = " ".join(str(i.value) for i in at.info)
    assert "ensemble" in text.lower()
    assert "did not win" in text.lower()


def test_forecasting_page_reconciles_store_totals():
    stores = dd.load("forecast_next_weeks_by_store.csv")
    pooled = dd.load("forecast_next_weeks.csv")
    h1 = pooled[pooled["horizon_weeks"] == config.INVENTORY_HORIZON]["forecast_units"].iloc[0]

    assert stores["store_units"].sum() == pytest.approx(h1, rel=1e-9), \
        "store forecasts must sum to the pooled forecast exactly"


def test_forecasting_page_warns_pooled_wape_is_not_the_mape_target():
    at = run_page("app/Demand_Forecasting.py")
    text = " ".join(str(m.value) for m in at.markdown)
    assert "not claim" in text.lower() or "not the brief" in text.lower()


# ------------------------------------------------------------------------- churn claims

def test_churn_page_shows_the_missed_auc_target():
    """The most important test in this file.

    AUC missed 0.88 and came out at 0.7315. If a future edit removes that framing the page
    becomes actively misleading while still rendering perfectly. This fails instead.
    """
    metrics = dd.load("churn_metrics.csv").iloc[0]
    at = run_page("app/Churn_Risk.py")

    text = " ".join(str(e.value) for e in at.error)
    assert f"{metrics['auc']:.4f}" in text
    assert str(config.CHURN_AUC_TARGET) in text
    assert "miss" in text.lower()


def test_churn_page_shows_the_met_precision_target():
    metrics = dd.load("churn_metrics.csv").iloc[0]
    at = run_page("app/Churn_Risk.py")
    text = " ".join(str(s.value) for s in at.success)
    assert f"{metrics['precision_at_top']:.4f}" in text


def test_churn_page_explains_why_auc_missed_rather_than_just_reporting_it():
    at = run_page("app/Churn_Risk.py")
    text = " ".join(str(e.value) for e in at.error)
    assert "recency" in text.lower(), "the miss needs an explanation, not just a number"


# ----------------------------------------------------------------------- inventory claims

def test_inventory_page_explains_the_critical_ratio_is_not_the_service_level():
    summary = dd.load("inventory_summary.csv").iloc[0]
    at = run_page("app/Inventory_Recommendations.py")

    text = " ".join(str(m.value) for m in at.markdown)
    assert f"{summary['mean_critical_ratio']:.4f}" in text
    assert "critical ratio" in text.lower()
    assert "95%" in text, "the brief's ask must be named so the difference is legible"


def test_inventory_page_shows_the_poisson_versus_normal_gap():
    summary = dd.load("inventory_summary.csv").iloc[0]
    at = run_page("app/Inventory_Recommendations.py")

    frames = [pd.DataFrame(f.value) for f in at.dataframe]
    joined = " ".join(f.to_csv() for f in frames)
    assert "Poisson quantile (used)" in joined
    assert f"{summary['total_units_to_order']:,.0f}" in joined
    assert f"{summary['normal_approx_units_to_order']:,.0f}" in joined

    errors = " ".join(str(e.value) for e in at.error)
    assert "overstates" in errors.lower()


def test_inventory_page_reports_its_cost_assumptions():
    """These drive the whole plan, so they cannot live only in code."""
    at = run_page("app/Inventory_Recommendations.py")
    summary = dd.load("inventory_summary.csv").iloc[0]

    body = " ".join(str(m.value) for m in at.markdown)
    assert dd.pct(summary["purchase_cost_ratio"], 0) in body
    assert dd.pct(summary["holding_rate"], 0) in body
    assert dd.pct(summary["lost_margin_ratio"], 0) in body


def test_inventory_plan_only_flags_pairs_below_their_reorder_point():
    plan = dd.load("inventory_reorder_plan.csv")
    flagged = plan[plan["reorder"] == True]  # noqa: E712
    assert len(flagged) > 0
    assert (flagged["on_hand"] < flagged["reorder_point"]).all(), \
        "a flagged pair must actually be below its reorder point"
    assert (flagged["suggested_order"] > 0).all()


def test_inventory_safety_stock_is_the_poisson_value():
    """The reported safety stock must be the Poisson one, not the normal comparison."""
    plan = dd.load("inventory_reorder_plan.csv")
    assert plan["safety_stock"].max() <= plan["normal_approx_safety_stock"].max() * 1.01
    summary = dd.load("inventory_summary.csv").iloc[0]
    assert summary["safety_stock_method"] == "poisson_quantile"


# -------------------------------------------------------------------------- segmentation

def test_segments_page_has_a_row_per_segment():
    segments = dd.load("customer_segments.csv")
    summary = dd.load("segment_summary.csv")
    at = run_page("app/Customer_Segments.py")

    assert summary.shape[0] == segments["segment_id"].nunique()
    assert not at.exception


def test_segments_page_reports_the_low_silhouette_honestly():
    """Silhouette near 0.20 means overlapping segments. Saying so is required."""
    at = run_page("app/Customer_Segments.py")
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "0.20" in text
    assert "dbscan" in text.lower()


def test_segment_sizes_sum_to_the_customer_count():
    segments = dd.load("customer_segments.csv")
    summary = dd.load("segment_summary.csv")
    assert summary["customers"].sum() == len(segments)


def test_segment_summary_carries_the_name_as_a_column():
    """Regression: the name was the index, so `index=False` dropped it.

    With six anonymous rows the only way to label them was to assume row order matched
    `segment_id`. It does not -- the summary is sorted by revenue -- so every name on the
    page was paired with the wrong segment. Champions, the largest earner, was displayed as
    an at-risk group. The names must ship as data.
    """
    summary = dd.load("segment_summary.csv")
    assert "segment" in summary.columns, (
        "segment_summary.csv lost its segment names; rerun `python -m src.precompute`")
    assert summary["segment"].notna().all()
    # Every name must correspond to a real group in the per-customer file.
    segments = dd.load("customer_segments.csv")
    assert set(summary["segment"]) == set(segments["segment"].unique())


def test_segment_summary_names_match_their_own_customer_counts():
    """The name and the numbers in the same row must describe the same group."""
    segments = dd.load("customer_segments.csv")
    summary = dd.load("segment_summary.csv")

    truth = segments.groupby("segment").agg(
        customers=("customer_id", "count"), total_revenue=("monetary", "sum"))

    merged = summary.set_index("segment").join(truth, rsuffix="_truth")
    assert (merged["customers"] == merged["customers_truth"]).all(), (
        "a row's name does not match its own customer count")
    assert (merged["total_revenue"].round(2)
            == merged["total_revenue_truth"].round(2)).all()


def test_segments_page_names_each_row_from_the_summary_not_row_position():
    """The displayed name must travel with its own numbers, whatever the sort order."""
    segments = dd.load("customer_segments.csv")
    summary = dd.load("segment_summary.csv")
    at = run_page("app/Customer_Segments.py")
    assert not at.exception

    # Ground truth: name -> customer count, straight from the per-customer file.
    truth = (segments.groupby("segment")["customer_id"].count().to_dict())
    frame = at.dataframe[0].value if isinstance(at.dataframe[0].value, pd.DataFrame) \
        else pd.DataFrame(at.dataframe[0].value)

    displayed = dict(zip(frame["Segment"], frame["Customers"]))
    assert displayed == truth, (
        "the segment table pairs names with the wrong counts")


# -------------------------------------------------------------------------------- misc

def test_pages_do_not_leak_absolute_paths_into_user_text():
    """A judge should not see the participant's home directory on a projected screen."""
    for page in PAGES.values():
        at = run_page(page)
        text = " ".join([str(m.value) for m in at.markdown]
                        + [str(c.value) for c in at.caption])
        assert str(config.ROOT) not in text, f"{page} leaks the absolute project path"


def test_every_page_has_a_title_and_some_content():
    for page in PAGES.values():
        at = run_page(page)
        assert at.title[0].value, f"{page} has an empty title"
        assert len(at.markdown) + len(at.metric) + len(at.dataframe) > 0, \
            f"{page} rendered no content"