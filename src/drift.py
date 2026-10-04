"""Data drift monitoring - compares two halves of the sales period.

The panel is weekly, so the two periods compared here are the 2024 and 2025 calendar years
on `data/raw/retail_pulse_demand_panel.csv`. That is the most decision-relevant comparison
available: it asks whether last year's behaviour still describes this year's, which is the
question a forecaster has to answer every time it re-runs.

What is worth watching, and why:

  * `units_sold` and `revenue` drift means the demand model is being applied to a market that
    moved. A store-level forecast that assumed last year's volume is now wrong everywhere.
  * `avg_discount_pct` and `is_promo_week` drift means the promotion baseline moved, which
    invalidates the observed 0.9999x lift -- that number describes 2024-25 as a whole.
  * `stockout_count` drift is a data-quality signal, not a demand signal. Rising stockouts
    suppress observed sales, so demand-based forecasts drift upward from nothing real.
  * The per-category mix (`units_sold` by `product_category`) catches a supplier or
    assortment change that a pooled total would hide.

Nothing here gates a deployment. A drift report tells you whether to distrust last month's
model, and the honest response to most of these rows is "yes, refit" rather than "stop".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Columns Evidently should check. Each maps to a decision above.
NUMERIC_COLUMNS = [
    "units_sold",
    "revenue",
    "avg_unit_price",
    "avg_discount_pct",
    "stockout_count",
    "on_hand_end",
]
CATEGORICAL_COLUMNS = ["product_category", "is_promo_week", "region"]

# Thresholds. The PSI default of 0.2 is the "significant" line; 0.1 is the "worth a look"
# line, which is the one that matters at 780,000 rows where tiny shifts cross 0.2 easily.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.20

DRIFT_REFERENCE_YEAR = 2024
DRIFT_CURRENT_YEAR = 2025


def load_panel() -> pd.DataFrame:
    """Read the demand panel, the one artefact that carries weekly business behaviour."""
    frame = pd.read_csv(config.PANEL_CSV, parse_dates=["week_start_date"])
    return frame


def split_periods(panel: pd.DataFrame):
    """Return (reference, current) split by calendar year.

    Weeks that straddle the year boundary land wherever their `week_start_date` falls, which
    keeps every week in exactly one period. No week is compared against itself.
    """
    reference = panel[panel["year"] == DRIFT_REFERENCE_YEAR]
    current = panel[panel["year"] == DRIFT_CURRENT_YEAR]

    if reference.empty or current.empty:
        raise ValueError(
            f"both years must be present: {DRIFT_REFERENCE_YEAR} has {len(reference)} rows, "
            f"{DRIFT_CURRENT_YEAR} has {len(current)} rows")
    return reference, current


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two samples of the same column.

    PSI is the sum of `(current% - reference%) * ln(current% / reference%)` across quantile
    bins. Zero means identical; above 0.2 conventionally means the distribution moved enough
    that a model fitted on the reference should be refitted.

    Implemented directly rather than pulled from evidently because the same number is needed
    for the printed summary, the CSV and the test, and it has to behave identically when
    evidently is not installed.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan")

    edges = pd.qcut(ref, q=bins, duplicates="drop", retbins=True)[1]
    edges = [edges[0], *edges[1:-1], edges[-1]]

    ref_counts = pd.cut(ref, bins=edges, include_lowest=True).value_counts(normalize=True)
    cur_counts = pd.cut(cur, bins=edges, include_lowest=True).value_counts(normalize=True)

    # A bin that is empty on one side makes the ratio undefined; replace with a small floor
    # rather than dropping the bin, because an empty bin is itself the signal.
    floor = 1e-6
    ref_pct = ref_counts.clip(lower=floor)
    cur_pct = cur_counts.clip(lower=floor)

    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def column_psi(panel: pd.DataFrame, column: str) -> dict:
    reference, current = split_periods(panel)
    psi = _psi(reference[column], current[column])
    return {
        "column": column,
        "kind": "numerical",
        "psi": round(psi, 4) if psi == psi else None,
        "reference_mean": round(float(pd.to_numeric(reference[column], errors="coerce").mean()), 4),
        "current_mean": round(float(pd.to_numeric(current[column], errors="coerce").mean()), 4),
        "reference_median": round(float(pd.to_numeric(reference[column], errors="coerce").median()), 4),
        "current_median": round(float(pd.to_numeric(current[column], errors="coerce").median()), 4),
        "verdict": _verdict(psi),
    }


def categorical_shift(panel: pd.DataFrame, column: str) -> dict:
    """Total variation distance between two categorical distributions, plus the biggest mover."""
    reference, current = split_periods(panel)
    ref_share = reference[column].value_counts(normalize=True)
    cur_share = current[column].value_counts(normalize=True)

    index = ref_share.index.union(cur_share.index)
    ref_share = ref_share.reindex(index).fillna(0.0)
    cur_share = cur_share.reindex(index).fillna(0.0)

    tvd = float((cur_share - ref_share).abs().sum() / 2.0)
    shift = (cur_share - ref_share).abs().sort_values(ascending=False)

    return {
        "column": column,
        "kind": "categorical",
        # Same 0-1 scale as PSI's practical range, so one threshold applies to both.
        "psi": round(tvd, 4),
        "largest_mover": str(shift.index[0]) if len(shift) else None,
        "largest_mover_shift": round(float(shift.iloc[0]), 4) if len(shift) else 0.0,
        "verdict": _verdict(tvd),
    }


def _verdict(psi: float) -> str:
    if psi != psi:
        return "insufficient data"
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def category_mix(panel: pd.DataFrame) -> pd.DataFrame:
    """Units by category, per year -- the assortment question a pooled total hides."""
    reference, current = split_periods(panel)
    ref = reference.groupby("product_category")["units_sold"].sum()
    cur = current.groupby("product_category")["units_sold"].sum()

    out = pd.DataFrame({
        "units_reference": ref,
        "units_current": cur,
    }).fillna(0.0)
    out["share_reference"] = out["units_reference"] / out["units_reference"].sum()
    out["share_current"] = out["units_current"] / out["units_current"].sum()
    out["share_shift"] = out["share_current"] - out["share_reference"]
    # PSI on the shares, not on the raw unit counts. Counts carry the same drift twice --
    # once from the category's own volume change and once from its share of the total -- so a
    # category that merely grew with the business would read as significant.
    out["psi"] = [_psi(out.loc[[idx], "share_reference"],
                       out.loc[[idx], "share_current"]) for idx in out.index]
    out["verdict"] = [_verdict(v) for v in out["psi"]]
    return out.sort_values("share_shift", key=lambda s: s.abs(), ascending=False)


def build_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per checked column. This is the artefact the test asserts on."""
    rows = [column_psi(panel, c) for c in NUMERIC_COLUMNS]
    rows += [categorical_shift(panel, c) for c in CATEGORICAL_COLUMNS]
    return pd.DataFrame(rows)


def write_evidently_report(reference: pd.DataFrame, current: pd.DataFrame,
                           out_dir: Path | None = None) -> Path | None:
    """Write an Evidently HTML report, or return None if evidently is unavailable.

    The summary CSV and the printed table above are the primary outputs and do not depend on
    evidently. This is the visual companion.

    Evidently 0.6 moved `Report` to `evidently.legacy.report`, where it still exposes
    `save_html`. The new top-level `Report` is the metrics API and has no HTML renderer at
    all. Both import paths are tried so this keeps working on 0.5 and 0.7, and a third path
    is used when only the new API exists.
    """
    out_dir = out_dir or (config.REPORTS / "drift")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "evidently_drift.html"

    cols = [c for c in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if c in reference.columns]
    numerical = [c for c in NUMERIC_COLUMNS if c in cols]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in cols]

    # Preferred: the legacy Report renders a full self-contained HTML dashboard.
    try:
        from evidently.legacy.metric_preset import DataDriftPreset as LegacyDrift
        from evidently.legacy.report import Report as LegacyReport
    except ImportError:
        pass
    else:
        try:
            report = LegacyReport([LegacyDrift()])
            report.run(current_data=current[cols], reference_data=reference[cols])
            report.save_html(str(target))
            return target
        except Exception as exc:  # noqa: BLE001 -- fall through to the newer API
            print(f"legacy evidently report failed ({exc}); trying the 0.6+ metrics API")

    # Fallback: the metrics API computes drift but writes JSON, not HTML.
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("evidently is not installed -- wrote the summary table only.")
        print("install it with:  pip install evidently>=0.5")
        return None

    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    report = Report([DataDriftPreset()])
    report.run(current_data=Dataset.from_pandas(current[cols], data_definition=definition),
               reference_data=Dataset.from_pandas(reference[cols], data_definition=definition))

    for method in ("save_json", "save"):
        writer = getattr(report, method, None)
        if callable(writer):
            writer(str(target.with_suffix(".json")))
            print(f"evidently 0.6+ has no HTML renderer; wrote "
                  f"{target.with_suffix('.json').name} instead.")
            return None

    print("this evidently build exposes no report writer; wrote the summary table only.")
    return None


def run(write: bool = True, out_dir: Path | None = None) -> dict:
    panel = load_panel()
    reference, current = split_periods(panel)
    summary = build_summary(panel)
    mix = category_mix(panel)

    if write:
        # out_dir exists so tests can write to tmp_path. Without it every `pytest` run
        # overwrote the committed reports/drift/ bundle, because Evidently stamps each
        # report with a fresh UUID. That left `git status` permanently dirty with a 3.7 MB
        # binary diff, which is the surest way to teach someone to stop reading it.
        out_dir = out_dir or (config.REPORTS / "drift")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "drift_summary.csv", index=False)
        mix.to_csv(out_dir / "category_mix_drift.csv")
        write_evidently_report(reference, current, out_dir)

    drifted = summary[summary["verdict"].isin(["moderate", "significant"])]
    return {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "summary": summary,
        "category_mix": mix,
        "drifted_columns": list(drifted["column"]),
        "verdict": "drift detected" if len(drifted) else "stable",
    }


def main(out_dir: Path | None = None) -> int:
    result = run(out_dir=out_dir)
    summary = result["summary"]
    mix = result["category_mix"]

    print("=" * 78)
    print("DATA DRIFT: {0} vs {1}".format(DRIFT_REFERENCE_YEAR, DRIFT_CURRENT_YEAR))
    print("=" * 78)
    print("reference rows: {0:,}".format(result["reference_rows"]))
    print("current rows:   {0:,}".format(result["current_rows"]))
    print()
    print(summary.to_string(index=False))
    print()
    print("category mix")
    print(mix.to_string())
    print()
    print("verdict:", result["verdict"])
    if result["drifted_columns"]:
        print("columns to re-examine before trusting a refit:",
              ", ".join(result["drifted_columns"]))
    print()
    print("PSI < {0} stable, {0}-{1} moderate, >= {1} significant".format(
        PSI_MODERATE, PSI_SIGNIFICANT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())