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
pip install -r requirements.txt
python generate_retail_pulse.py     # raw data, seeded, not committed
python -m src.precompute            # aggregates the dashboard reads
python -m pytest tests/ -q          # 474 tests
```

## Running the dashboard

```bash
streamlit run app/Home.py
```

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
tests/          474 tests
reports/        report, drift artefacts, figures
DAILY_LOG.md    dated log of decisions and verification runs
```