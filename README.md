# RetailPulse

Demand forecasting, customer segmentation, churn prediction and inventory
recommendations for a 970,998-line retail panel, plus a Streamlit dashboard.

## What is here

| Stage | Module | Output |
|---|---|---|
| F-01 Ingest | `src/ingest.py` | validated sales frame, weekly rollups, store-product panel |
| F-02 Segmentation | `src/segmentation.py` | 6 K-Means segments, DBSCAN cross-check |
| F-03 Forecasting | `src/forecasting.py` | per-horizon model selection, 4-week forecast |
| F-04 Churn | `src/churn.py` | XGBoost scores, precision@k, SHAP importance |
| F-05 Inventory | `src/inventory.py` | reorder plan for 7,500 store-product pairs |
| F-06 Dashboard | `app/` | five Streamlit pages |
| F-07 Drift | `src/drift.py` | year-over-year PSI report |

## Setup

```bash
pip install -r requirements-ml.txt   # app + modelling stack + tests
python generate_retail_pulse.py      # raw data, seeded, not committed
python -m src.precompute             # aggregates the dashboard reads
python -m pytest tests/ -q           # 510 tests
```

To run only the dashboard, install the lean app set instead:

```bash
pip install -r requirements.txt       # streamlit, pandas, numpy
```

`requirements.txt` is deliberately just those three packages, because it is the file
**Streamlit Community Cloud installs** and Cloud offers no way to point it elsewhere. Keeping
Prophet, TensorFlow, XGBoost, SHAP and Evidently in it would make a ~2.5 GB install the price
of opening the public demo. The modelling stack lives in `requirements-ml.txt`, which extends
the app set rather than restating it. `tests/test_tier3.py` fails the build if that stops
being true.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub and publish the current branch.
2. In [share.streamlit.io](https://share.streamlit.io) choose **Deploy**, then that repo and
   branch.
3. Main file path: `app/Home.py`. Leave everything else at its default.

`.python-version` pins `3.11`, which is what Cloud reads to choose a runtime. The Python
version is the one thing worth setting deliberately: Cloud does not run 3.14, and the app is
written to stay inside the 3.11-compatible syntax.

## Running the dashboard

```bash
streamlit run app/Home.py
```

`app/Home.py` is the entry point for all five pages, declared with `st.navigation` in the same
file. Streamlit only auto-discovers a `pages/` *directory*, so the flat `app/` layout in the
design spec needs the explicit declaration — without it every URL served Home.

Pages read only `data/processed/`. The raw inputs are 103.7 MB in total
(`retail_pulse_demand_panel.csv` 60.4 MB, `retail_pulse_sales.csv` 43.3 MB) and cost
10-20 seconds per session against an 8-second budget, so a test fails the build if
any page touches them. Aggregates load in about 0.1s.

Run an individual stage, or the whole nightly chain:

```bash
python -m src.forecasting
python -m src.inventory
python -m src.drift

python -m src.pipeline                  # all stages, in order, stop at first failure
python -m src.pipeline --only churn     # a subset
```

`src.pipeline` is the single definition of the stage order and of which artefacts
each stage must leave behind. The Airflow DAG and the Kubernetes CronJob both import
that list rather than restating it, and `tests/test_pipeline.py` checks the manifests
against it.

## Results

**Forecasting** — Prophet at 1 week (WAPE 0.0068), seasonal naive at 2 and 4 weeks
(0.0213 / 0.0383). Fitted on the pooled weekly total and apportioned by historical
share; 7,500 independent fits cannot meet the batch budget.

**Segmentation** — k=6, silhouette 0.1998. DBSCAN found 2 clusters with 38.1% noise,
so it did not produce a usable alternative here.

**Churn** — precision@top-20% 0.7703 (target 0.75, met), lift 1.81. **AUC 0.7315
against a 0.88 target, not met.** See the report.

**Inventory** — 496 pairs to reorder, 2,081 units, INR 48,771. Uses a Poisson
quantile; a normal approximation would order 5,384 more units.

**Drift** — no significant drift between 2024 and 2025.

## Honest limitations

Three targets were missed or redefined, and the report states each one rather than
averaging them into a favourable summary:

1. Churn AUC is 0.7315 against a 0.88 target.
2. The forecast MAPE target is not demonstrated at the panel level. The 0.68%
   figure is pooled, and the pooled total has no zero-demand weeks — 77.9% of
   panel rows are zero, which is why WAPE is the primary metric.
3. Silhouette 0.1998 means the segments overlap substantially.

Full detail in [`reports/RetailPulse_Report.md`](reports/RetailPulse_Report.md).

## Layout

```
app/            Streamlit pages + cached data loader
data/processed/ committed aggregates (the dashboard's only input)
data/raw/       generated, gitignored
src/            one module per stage
tests/          510 tests
notebooks/      01_results_and_targets, 02_data_and_drift
reports/        report, drift artefacts, figures, screenshots
DAILY_LOG.md    dated log of decisions and verification runs
```

## Notebooks

Both read committed aggregates only, so they run on a fresh clone in seconds without
regenerating the raw inputs.

- `notebooks/01_results_and_targets.ipynb` — checks each headline target against the published
  numbers, including the two that were missed.
- `notebooks/02_data_and_drift.ipynb` — the panel's zero-demand share and why that decides the
  forecasting metric, plus what the drift artefacts do and do not establish.

## Screenshots

`reports/screenshots/` holds one PNG per page plus a manifest. Regenerate with a server
running:

```bash
streamlit run app/Home.py          # in one terminal
python -m src.screenshots          # in another
```

`tests/test_screenshots.py` fails the build if two captures are byte-identical or if a page
shows content from another page. Both checks exist because the first version of the script
produced five identical images of Streamlit's loading skeleton and every size and format check
passed on them.