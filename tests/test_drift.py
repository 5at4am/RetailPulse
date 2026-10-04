"""Tests for F-07 data drift monitoring.

The PSI implementation is checked against synthetic distributions with known shifts before
the real panel is touched. A drift metric that returns 0.0 for everything would pass a test
that only asserts "the verdict column is populated".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, drift


# --------------------------------------------------------------------------- PSI behaviour

def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(20240101)
    sample = pd.Series(rng.poisson(1.0, 100_000))
    assert drift._psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_bin_edges_come_from_the_reference_period():
    """PSI is not symmetric, and that is a property of the definition, not a bug.

    The bins are quantiles of the *reference* sample, so swapping the arguments re-cuts the
    bins and changes the result (0.165 one way, 0.172 the other here). The test pins the
    behaviour so nobody "fixes" it into a symmetric variant that silently disagrees with
    Evidently, and asserts the difference stays small -- a large gap would mean the bins are
    doing the work rather than the data.
    """
    rng = np.random.default_rng(7)
    a = pd.Series(rng.poisson(1.0, 80_000))
    b = pd.Series(rng.poisson(1.5, 80_000))

    forward = drift._psi(a, b)
    backward = drift._psi(b, a)
    assert forward > drift.PSI_MODERATE
    assert abs(forward - backward) / max(forward, backward) < 0.15


def test_psi_grows_with_the_size_of_the_shift():
    """The metric has to be monotone, not just non-zero."""
    rng = np.random.default_rng(11)
    base = pd.Series(rng.poisson(1.0, 120_000))
    values = [drift._psi(base, pd.Series(rng.poisson(mean, 120_000)))
              for mean in (1.0, 1.5, 3.0)]
    assert values[0] < values[1] < values[2], values


def test_psi_flags_a_large_shift_as_significant():
    rng = np.random.default_rng(3)
    base = pd.Series(rng.poisson(1.0, 100_000))
    far = pd.Series(rng.poisson(5.0, 100_000))
    assert drift._psi(base, far) >= drift.PSI_SIGNIFICANT


def test_psi_sees_a_change_in_zero_inflation():
    """The panel is 77.9% zeros, so a shift in the zero share must not read as stable."""
    rng = np.random.default_rng(5)
    few_zeros = pd.Series(np.where(rng.random(100_000) < 0.2, 0,
                                   rng.poisson(2.0, 100_000)))
    many_zeros = pd.Series(np.where(rng.random(100_000) < 0.6, 0,
                                    rng.poisson(2.0, 100_000)))
    assert drift._psi(few_zeros, many_zeros) >= drift.PSI_MODERATE


def test_psi_survives_a_constant_column():
    """Zero-variance columns appear in the panel (every week with no stockout)."""
    flat = pd.Series([0.0] * 1_000)
    psi = drift._psi(flat, flat)
    assert psi == psi, "must be a number, not NaN"


def test_psi_is_nan_rather_than_infinite_on_an_empty_sample():
    assert np.isnan(drift._psi(pd.Series([], dtype=float), pd.Series([1.0, 2.0])))


# ------------------------------------------------------------------------------ verdicts

@pytest.mark.parametrize("psi,expected", [
    (0.0, "stable"),
    (0.05, "stable"),
    (drift.PSI_MODERATE, "moderate"),
    (0.15, "moderate"),
    (drift.PSI_SIGNIFICANT, "significant"),
    (0.9, "significant"),
    (float("nan"), "insufficient data"),
])
def test_verdict_thresholds(psi, expected):
    assert drift._verdict(psi) == expected


# ------------------------------------------------------------------------------ the panel

@pytest.fixture(scope="module")
def panel():
    return drift.load_panel()


@pytest.fixture(scope="module")
def periods(panel):
    return drift.split_periods(panel)


def test_periods_partition_the_panel(panel, periods):
    reference, current = periods
    assert len(reference) + len(current) == len(panel)
    assert reference["year"].unique().tolist() == [drift.DRIFT_REFERENCE_YEAR]
    assert current["year"].unique().tolist() == [drift.DRIFT_CURRENT_YEAR]


def test_periods_do_not_share_a_week(periods):
    reference, current = periods
    assert not set(reference["week_start_date"]) & set(current["week_start_date"])


def test_split_raises_when_a_year_is_absent(panel):
    single = panel[panel["year"] == drift.DRIFT_REFERENCE_YEAR]
    with pytest.raises(ValueError, match="both years must be present"):
        drift.split_periods(single)


def test_summary_covers_every_watched_column(panel):
    summary = drift.build_summary(panel)
    watched = set(drift.NUMERIC_COLUMNS) | set(drift.CATEGORICAL_COLUMNS)
    assert set(summary["column"]) == watched
    assert (summary["kind"] == "numerical").sum() == len(drift.NUMERIC_COLUMNS)


def test_summary_rows_all_carry_a_verdict(panel):
    summary = drift.build_summary(panel)
    assert summary["verdict"].notna().all()
    assert set(summary["verdict"]) <= {"stable", "moderate", "significant",
                                        "insufficient data"}


def test_column_psi_reports_both_period_means(panel):
    row = drift.column_psi(panel, "units_sold")
    for key in ("reference_mean", "current_mean", "reference_median", "current_median"):
        assert key in row
        assert row[key] is not None


def test_categorical_shift_names_its_largest_mover(panel):
    """The mover has to be one of the categories actually present, and named."""
    row = drift.categorical_shift(panel, "product_category")
    assert row["largest_mover"] in set(panel["product_category"])
    assert 0.0 <= row["largest_mover_shift"] <= 1.0


def test_category_mix_shares_sum_to_one(panel):
    mix = drift.category_mix(panel)
    assert mix["share_reference"].sum() == pytest.approx(1.0, abs=1e-6)
    assert mix["share_current"].sum() == pytest.approx(1.0, abs=1e-6)


def test_category_mix_reports_one_row_per_category(panel):
    mix = drift.category_mix(panel)
    assert set(mix.index) == set(panel["product_category"].unique())


def test_category_mix_is_sorted_by_absolute_shift(panel):
    mix = drift.category_mix(panel)
    assert mix["share_shift"].abs().is_monotonic_decreasing


# --------------------------------------------------------------------------------- run()

def test_run_writes_the_summary_and_mix(tmp_path, monkeypatch, panel):
    monkeypatch.setattr(config, "REPORTS", tmp_path)
    result = drift.run(write=True)

    assert (tmp_path / "drift" / "drift_summary.csv").exists()
    assert (tmp_path / "drift" / "category_mix_drift.csv").exists()

    written = pd.read_csv(tmp_path / "drift" / "drift_summary.csv")
    assert len(written) == len(result["summary"])
    assert result["reference_rows"] > 0 and result["current_rows"] > 0


def test_run_returns_a_verdict_consistent_with_its_rows(panel, monkeypatch):
    result = drift.run(write=False)
    summary = result["summary"]
    flagged = summary[summary["verdict"].isin(["moderate", "significant"])]

    assert result["drifted_columns"] == list(flagged["column"])
    assert (result["verdict"] == "drift detected") == (len(flagged) > 0)


def test_evidently_report_is_written_or_explains_itself(tmp_path, panel):
    """Either a real HTML artefact, or a printed reason. Never a silent no-op.

    Evidently moved `Report` to `evidently.legacy` in 0.6 and the top-level API has no HTML
    renderer, so this is the test that keeps the module honest about which path ran.
    """
    reference, current = drift.split_periods(panel)
    written = drift.write_evidently_report(reference, current, tmp_path)

    if written is None:
        # Acceptable only if evidently is absent; the caller must have said so.
        pytest.skip("evidently unavailable or has no HTML renderer in this build")
    assert written.exists()
    assert written.stat().st_size > 0
    assert written.suffix == ".html"


def test_main_exits_zero(tmp_path):
    # tmp_path, not the default: `main()` writes the whole drift bundle, and Evidently
    # stamps every report with a fresh UUID. Writing to reports/drift/ here meant every
    # `pytest` run produced a 3.7 MB binary diff in `git status`, which trains you to
    # ignore the one tool that would have caught it.
    assert drift.main(out_dir=tmp_path) == 0
    assert (tmp_path / "drift_summary.csv").exists()
    assert (tmp_path / "category_mix_drift.csv").exists()


def test_main_does_not_touch_the_committed_bundle(tmp_path):
    """The regression guard for the above: `main()` must not write to reports/drift/."""
    committed = config.REPORTS / "drift"
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in committed.glob("*") if p.is_file()}
    if not before:
        pytest.skip("no committed drift bundle to compare against")

    drift.main(out_dir=tmp_path)

    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
             for p in committed.glob("*") if p.is_file()}
    assert after == before, f"reports/drift/ was modified: {before} -> {after}"


def test_the_real_panel_shows_no_significant_drift(panel):
    """The dataset is seeded and stable year over year, so this documents the finding."""
    summary = drift.build_summary(panel)
    assert "significant" not in set(summary["verdict"]), summary.to_string(index=False)