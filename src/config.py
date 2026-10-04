"""Paths, constants and service levels for the whole project.

    from src.config import RAW, SALES_CSV, load_sales

This is the only module allowed to know where files live. Everything else imports paths
from here, so re-rooting the project (it is renamed `retail-pulse-data/` -> `RetailPulse/`
at the end of the build) is a one-file change.

Paths are derived from this file's own location, never from the current working
directory, so `python -m src.ingest`, `python src/ingest.py`, a pytest run and a Streamlit
Cloud deploy all resolve to the same folders regardless of where they are invoked from.

Stdlib only -- no pandas, no numpy. Importing config must stay cheap.
"""
from __future__ import annotations

import sys
from pathlib import Path

# src/config.py -> src/ -> project root
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
APP = ROOT / "app"
TESTS = ROOT / "tests"
DOCS = ROOT / "docs"

# Raw uploads. Regenerate with `python generate_retail_pulse.py` rather than committing
# these -- the generator is seeded, so reruns are byte-identical.
SALES_CSV = RAW / "retail_pulse_sales.csv"
PANEL_CSV = RAW / "retail_pulse_demand_panel.csv"
DATA_DICTIONARY = RAW / "retail_pulse_data_dictionary.md"

# Machine-checked dataset facts, produced by collect_facts.py.
FACTS_JSON = ROOT / "facts.json"

# Artefacts written by the pipeline.
PROCESSED_DIRS = (RAW, PROCESSED, MODELS, FIGURES)


def ensure_dirs():
    """Create every output folder. Safe to call repeatedly.

    Raw data is not created here -- it is either downloaded or generated.
    """
    for d in PROCESSED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def runnable(script: str) -> str:
    """Path to a project-root script, for `subprocess` calls from tests and CLI glue."""
    return str(ROOT / script)


def add_root_to_path():
    """Put the project root on `sys.path` so `import retail_pulse` works.

    Needed by the root-level stdlib scripts and by tests, which import the root-level
    `retail_pulse` module rather than the `src` package.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return ROOT


# --------------------------------------------------------------------------- data facts
# Asserted by src/ingest.py against the loaded frames. Duplicated here on purpose: a
# pipeline that reads its own expectations out of the file it is validating cannot fail.
SALES_ROWS = 250_000
SALES_COLS = 27
PANEL_ROWS = 780_000
PANEL_COLS = 15

SALES_START = "2024-01-01"
SALES_END = "2025-12-31"
PANEL_WEEKS = 104

# The sales file's final week starts 2025-12-29 and holds only 3 of its 7 days. Any
# weekly roll-up drops it, or every weekly chart ends in a fake demand cliff.
PARTIAL_FINAL_WEEK = "2025-12-29"

CURRENCY = "INR"

# --------------------------------------------------------------------------- forecasting
# "30-day ahead" is 4 weeks (28 days) on a weekly panel. Horizons are evaluated at
# 1, 2 and 4 weeks; inventory consumes the 1-week horizon only.
HORIZONS = (1, 2, 4)
PRIMARY_HORIZON = 4
INVENTORY_HORIZON = 1

# MASE needs a denominator: the in-sample error of a naive forecast. On a weekly series with
# an annual cycle, 52 is the meaningful choice -- a random-walk denominator (seasonality=1)
# would flatter every model, because the seasonal-naive error is large on this data.
MASE_SEASONALITY = 52

# 77.9% of panel rows have units_sold == 0, so plain MAPE is undefined there. WAPE is the
# primary metric; MAPE is reported on the non-zero subset only, with the subset stated.
MAPE_ZERO_SUBSET = "y_true > 0"
BRIEF_MAPE_TARGET = 0.12

# --------------------------------------------------------------------------- churn
CHURN_AUC_TARGET = 0.88
CHURN_PRECISION_AT_K = 0.20          # precision among the top 20% highest-risk
CHURN_PRECISION_TARGET = 0.75        # the brief's floor for that precision
CHURN_LABEL_WINDOW_DAYS = 90          # no purchase in the next 90 days = churned
SHAP_LOCAL_TOP_N = 25                 # customers given per-customer explanations
# Reasons published per customer. Six of the ten features: ranks 7-10 in the real output
# sit around 1e-3 log-odds, which is noise, and printing them turns "why is this customer
# flagged" into a reprint of the global importance table.
SHAP_LOCAL_REASONS = 6
SNAPSHOT_START = "2025-01-31"
SNAPSHOT_END = "2025-09-30"
SNAPSHOT_STRIDE_DAYS = 30             # one label set per month

# --------------------------------------------------------------------------- segmentation
SEGMENT_MIN = 6
SEGMENT_MAX = 8

# --------------------------------------------------------------------------- inventory
# safety_stock = z * sigma(forecast_error); z from the target service level.
DEFAULT_SERVICE_LEVEL = 0.95
Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537}

# --------------------------------------------------------------------------- reproducibility
SEED = 20240101