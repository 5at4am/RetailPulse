# RetailPulse — Design Spec

**Date:** 2026-10-04
**Author:** prepared with OpenWork
**Status:** awaiting participant review
**Target submission:** Zidio Development — Data Science & Analytics, March 2026 edition

---

## 1. Context and constraints

A 28-day industry project brief with **5 days remaining**. The brief asks for six functional
requirements, an MLOps layer, production deployment, and five graded deliverables. That is
roughly 3–4× the available time.

Two constraints shape everything below:

1. **The grading is on deliverables, not on model count.** Of 100 points, 95 come from the
   PDF report (25), live public demo URL (30), GitHub repo (20), README (15), and demo video
   (10). Model quality is only visible *inside* the PDF and the demo. Effort therefore goes
   to making those two excellent, not to maximising the number of models.
2. **The brief disqualifies AI-generated submissions.** "Plagiarism / AI-generated content
   will result in disqualification for stipend." So: this spec and the codebase are a
   scaffold and a source of explanation. The participant must read, understand, and be able
   to defend every number, and must write the report and README in their own words.

### 1.1 What already exists

Built and verified before this spec, in the project root:

| Asset | Status |
|---|---|
| `generate_retail_pulse.py` | Deterministic generator, `SEED = 20240101`, byte-identical reruns |
| `retail_pulse_sales.csv` | 250,000 rows × 27 cols, 2024-01-01 → 2025-12-31, INR 210,242,624.35 |
| `retail_pulse_demand_panel.csv` | 780,000 rows × 15 cols, 104 complete weeks, 50 stores × 150 products |
| `retail_pulse_data_dictionary.md` | 471 lines, 22 sections, column-level documentation |
| `retail_pulse.py` | Standard-library loader + validator + roll-up helpers |
| `verify_dataset.py`, `audit_quality.py`, `reconcile_panel.py` | Verification suite |
| `test_retail_pulse.py`, `test_aggregate.py`, `test_null_handling.py` | 3 test modules, passing |
| `facts.json` | Machine-checked dataset facts |
| `methodology/` | Methodology document (HTML + PDF) |
| `load_quickstart.py`, `analyse_quickstart.py` | Load/EDA report and 10-section exploratory analysis |

Because the generator is seeded, **the two 104 MB CSVs never need to be committed** — they
are reproducible. This is what makes the repository small enough to be pleasant to use.

### 1.2 Environment

Local: Python 3.14.7, pandas 3.0.6. The brief specifies Python 3.11. Streamlit Community
Cloud will not run 3.14. **Constraint: no 3.14-only syntax, and `requirements.txt` pins
`pandas>=2.2` rather than an exact version**, so the app resolves to a runtime the host
actually has. See R1 in §11.

---

## 2. Scope — three tiers, nothing omitted

The instruction was to omit nothing. So nothing is omitted. What varies is depth, and the
depth is stated honestly in the report.

| Tier | Contents | Depth | Evidence offered |
|---|---|---|---|
| **1** | F-01 … F-06, all metrics, Streamlit dashboard, live deployment, README, PDF report, demo video | Fully built, run, and tested | Judges run it themselves |
| **2** | Prophet + LSTM + ensemble forecasting | Models fully built and trained at a defensible scale (§6) | Real WAPE / MAPE numbers |
| **3** | Docker, Kubernetes manifests, GitHub Actions, Airflow DAG, Prometheus + Grafana, Evidently | Authored completely and statically validated | `kubectl --dry-run`, config lint. **Not deployed or operated** — reported as such |

The exact wording used in the report for Tier 3, so there is no ambiguity later:

> Infrastructure artefacts (Dockerfile, Kubernetes manifests, CI workflow, Airflow DAG,
> Prometheus/Grafana configuration) are complete and statically validated, but were not
> deployed to a live cluster within the submission window. Streamlit Community Cloud was used
> for the public demo.

Priority order if time runs short: Tier 1 before Tier 2 before Tier 3. The dashboard and the
report carry 55% combined, so they are never traded away for infrastructure.

### 2.1 Deliberately not built

- **Returns / net-of-returns metrics.** The dataset contains no returns, by design. Adding
  them would mean inventing data.
- **Real-time / streaming ingestion.** The brief's daily-batch SLA is met by a batch
  `precompute` step, not a stream.
- **A per-store-product Prophet model.** 7,500 series is computationally out of reach for a
  5-minute batch. §6 explains the substitute.

---

## 3. Repository layout

Everything lives under the single project root (`retail-pulse-data/`, renamed to
`RetailPulse/` at the end of the build — deferred so no path breaks mid-build).

```
retail-pulse-data/            <- project root == GitHub repo root
├── README.md                 deliverable, 15%
├── requirements.txt
├── .gitignore
├── DAILY_LOG.md              daily progress log
├── app/                      Streamlit dashboard, F-06
│   ├── Home.py
│   ├── Demand_Forecasting.py
│   ├── Customer_Segments.py
│   ├── Churn_Risk.py
│   └── Inventory_Recommendations.py
├── src/
│   ├── config.py             paths, constants, service levels
│   ├── ingest.py             F-01 load + validate
│   ├── features.py           RFM + engineered features, leakage-safe
│   ├── segmentation.py       F-02 K-Means + DBSCAN
│   ├── forecasting.py        F-03 Prophet + LSTM + ensemble
│   ├── churn.py              F-04 XGBoost + SHAP
│   ├── inventory.py          F-05 reorder quantities
│   ├── precompute.py         small aggregates for the dashboard
│   └── drift.py              Evidently drift report
├── data/
│   ├── raw/                  2 CSVs + dictionary   (gitignored)
│   └── processed/            small aggregates ~3 MB (committed)
├── models/                   saved artefacts
├── reports/
│   ├── figures/
│   └── RetailPulse_Report.md source for the PDF
├── tests/
├── docs/specs/
└── methodology/
```

**Module boundary rule:** every module in `src/` is runnable on its own, takes already-clean
DataFrames as input, and returns DataFrames plus a metrics dict. No module reads a global
cache, and no module imports another module's internals. This is what makes them testable
and re-runnable in isolation, and it is what makes the pipeline order in §4 explicit rather
than emergent.

---

## 4. Data flow

```
generate_retail_pulse.py          (seeded, reproducible)
        │
        ▼
data/raw/retail_pulse_sales.csv  ──┐
data/raw/retail_pulse_demand_panel.csv ──┤
data/raw/retail_pulse_data_dictionary.md │
        ▼                              │
   ingest.py  (F-01)                   │  validates contracts before anything
   · schema + row/col count vs facts.json
   · date parsing, dtype contract
   · 8 documented row-level rules
        │  fails loudly, never silently coerces
        ├──────────────────────────────┘
        ▼
   features.py
   · RFM (recency, frequency, monetary)
   · basket shape, category diversity, promo sensitivity, tenure
   · weekly panel → lag features, rolling stats
        │
        ├──────────────► segmentation.py (F-02) ──┐
        ├──────────────► churn.py        (F-04) ──┤
        ├──────────────► forecasting.py  (F-03) ─┤
        │                        │               │
        │                        ▼               │
        │                   inventory.py  (F-05) ┤
        │                                        │
        ▼                                        ▼
   precompute.py ────────────► data/processed/*.csv  ──► app/ (F-06)
        │                              small, ~3 MB            │
        ▼                                                       ▼
   drift.py  (Evidently)                              Streamlit Community Cloud
        │                                                       │
        └──────────────► models/ + reports/figures/ ◄────────────┘
                                 │
                                 ▼
                     reports/RetailPulse_Report.md → PDF
```

**Why `precompute.py` exists.** The dashboard must load in under 8 seconds (brief §5). The
raw panel is 63 MB; parsing it on every Streamlit session start costs 10–20 seconds and
would fail the requirement. So `precompute.py` reduces the dataset to small aggregates once,
and the dashboard reads only those. This is also a legitimate F-01 ETL deliverable, so the
constraint costs nothing.

Outputs written by `precompute.py`, all committed:

| File | Rows approx. |
|---|---|
| `agg_monthly_revenue.csv` | 24 |
| `agg_weekly_demand.csv` | 104 |
| `agg_city.csv` / `agg_region.csv` | 12 / 5 |
| `agg_category.csv` | 10 |
| `agg_category_month_mix.csv` | 10 × 12 |
| `top_products.csv` | 50 |
| `rfm_segments.csv` | 8,000 |
| `churn_scores.csv` | 8,000 |
| `forecast_results.csv` | series × horizon |
| `inventory_recommendations.csv` | store × product |
| `model_metrics.json` | 1 |

---

## 5. F-01 — Data ingestion and cleaning

Load both CSVs, enforce the contract, and refuse to continue if it is broken.

Checks performed, all already implemented in `retail_pulse.py` and
`load_quickstart.py`:

- header matches the expected column list exactly
- row and column counts match `facts.json` (250,000 × 27 and 780,000 × 15)
- `date` and `week_start_date` parse as datetimes
- the 8 documented row-level rules, e.g. `sales_amount = quantity_sold × (unit_price −
  discount)` to the paisa, and `quantity_sold == 0 ⟹ inventory_level == 0`
- no nulls except `promotion`, which is `NaN` by design on the 228,996 unpromoted lines

**Design decision — fail loudly.** A violated contract raises. It does not warn-and-continue.
A silent coercion would let a broken extract reach the models and produce a confident wrong
answer, which is the worst outcome for a grading demo.

**Known data traps carried forward into every module:**

| Trap | Correct handling |
|---|---|
| A stockout is `quantity_sold == 0`, **not** `inventory_level == 0` | 7,500 lost-demand rows vs 27,749 rows where a customer bought the final unit. Using the wrong test overstates lost demand 3.7× |
| The final calendar week (starting 2025-12-29) holds only 3 of 7 days | Dropped. The shipped panel already excludes it; sales roll-ups must too, or every weekly chart ends in a fake cliff |
| 77.9% of panel rows have `units_sold = 0` | Not missing data. Genuine zero-demand weeks, retained on purpose |
| 563 of 731 days (77.0%) run some promotion | Any promotion-lift metric is diluted, and Sat/Sun have no non-promotion counterpart at all |

---

## 6. F-03 — Demand forecasting

### 6.1 The scale problem, stated plainly

The panel is `store × product × week`: 50 × 150 × 104 = 7,500 distinct series. A Prophet
model per series does not fit a 5-minute batch SLA and does not fit a 5-day build. The brief
does not say the forecast must be per-series, so the design is:

| Component | Granularity | Purpose |
|---|---|---|
| Prophet | 10 category-level weekly series | Trend, weekly and yearly seasonality, holiday effects. Interpretable, and the holiday regressors are where Diwali shows up |
| LSTM (PyTorch) | same 10 series + lag/rolling features | Nonlinear residual structure |
| Ensemble | weighted blend of the two | Weights fitted on the validation window, not guessed |
| Gradient-boosted regressor | full 780,000-row panel, all series | The scalable per-series baseline, so results are not limited to 10 aggregates |

**Reported limitation, verbatim for the report:**

> The brief specifies a Prophet + LSTM ensemble. At the panel's native grain that implies
> 7,500 Prophet fits, which cannot complete inside the 5-minute batch target. The ensemble is
> therefore fitted at category level, with a single global gradient-boosted model covering all
> store-product series. Category-level captures seasonality and holidays; the global model
> carries per-series signal. Fitting Prophet per store-product remains future work.

### 6.2 Metrics — the honest treatment

**MAPE as literally specified cannot be computed on this data.** MAPE divides by the actual
value, and 77.9% of panel rows have `units_sold = 0`. MAPE is undefined there. Any MAPE number
quoted over all rows is either silently wrong or silently computed on a 22.1% subset.

So the report carries, side by side:

| Metric | Definition | Over |
|---|---|---|
| **WAPE** (primary) | `Σ\|y − ŷ\| / Σ\|y\|` | all rows, safe with zeros |
| **MAPE (non-zero rows)** | `mean(\|y − ŷ\| / \|y\|)` for `y > 0` only — the 22.1% subset | stated subset, so it is comparable to the brief's 12% |
| MAE | mean absolute error, in units | all rows |
| RMSE | penalises large misses | all rows |
| MASE | MAE ÷ MAE of a naive forecast | all rows, scale-free |
| Bias | mean signed error | all rows |

The brief's **MAPE ≤ 12%** is compared against MAPE on non-zero rows, with the exclusion
stated in the same sentence, every time the number appears. WAPE and MASE are reported as the
honest primary measures. Bias is reported because lost-demand censoring biases forecasts
downward, and a forecaster who does not say so is not being honest.

Horizons are evaluated at 1, 2, and 4 weeks. "30-day ahead" is 4 weeks (28 days) on a weekly
grain; this rounding is stated in the report rather than left implicit.

---

## 7. F-02 — Customer segmentation

**Target: 6–8 segments with business interpretation** (brief F-02).

Feature set, all computed as of a single snapshot date (2025-12-31) with no forward-looking
information:

`recency_days`, `frequency_90d`, `monetary_90d`, `avg_basket_value`, `units_per_line`,
`category_diversity`, `promo_share`, `mean_discount_pct`, `online_share`, `tenure_days`

Procedure:

1. Standardise (z-score). K-Means is distance-based, so this is not optional.
2. Fit `k = 2…10`, select on **silhouette score**, prefer 6–8 when scores are close.
   Ties broken toward the smaller k, since fewer segments are easier to act on.
3. Refit DBSCAN as a comparison, and report the noise fraction. DBSCAN is expected to label
   a large share of customers as noise on this data; that result is reported, not hidden.
4. Name each segment from its centroid profile — the names must describe what the centroid
   actually is, not what would sound good in a slide.

Segments are described using **share of units and revenue**, never absolute volume alone.
The dataset has a strong global festive lift that makes almost every category peak in
October; the category-specific signal lives in the mix.

Sanity target from the existing analysis: repeat customers are 87.3%, so a "one-and-done"
segment should be the smallest, not the largest.

---

## 8. F-04 — Churn prediction

**Targets: AUC-ROC ≥ 0.88, precision@top-20% ≥ 0.75.**

### 8.1 Labelling — time-based, leakage-free

A single end-of-period label would give 8,000 rows and no way to validate temporally. Instead,
**monthly snapshots**:

- Snapshots at each month-end from **2025-01-31 to 2025-09-30** (9 snapshots).
- Feature window: 90 days ending on the snapshot date.
- Label window: the 90 days *after* the snapshot. `churn = 1` if the customer made no purchase
  in that window.
- The last snapshot is 2025-09-30 because a 90-day forward window must fit before the data
  ends on 2025-12-31. A snapshot in November or December would have an unlabelled future.

Every feature is computed from rows dated on or before the snapshot date. This is enforced by
a test, not by convention — see §10.

Customers not yet active at a snapshot are excluded from it, so the row count per snapshot
grows over time. Roughly 8,000 × 9 ≈ 60,000 rows before that exclusion.

### 8.2 Split and model

- **Split is temporal, not random.** Train on snapshots ≤ 2025-06-30, validate on 2025-07-31,
  test on 2025-08-31 and 2025-09-30. A random split would leak future behaviour into training
  and inflate AUC.
- Model: XGBoost, with class weighting for the expected minority class.
- Explainability: SHAP global summary plus per-customer local explanations for the top-risk
  cohort.
- Reported: AUC-ROC, precision@top-20%, PR-AUC, F1 at the chosen threshold, and the confusion
  matrix. precision@top-20% is the operational number — it answers "if I call 1,600 customers
  today, how many are right?", which is the question a retention team actually asks.

If AUC falls short of 0.88, the number is reported as it lands. A reported 0.84 with an honest
error analysis scores better than an unexplained 0.90.

---

## 9. F-05 — Inventory optimisation

**Target: reduce over/understock by 25–40%.** That claim is only credible if it comes from a
backtest, so that is how it is produced.

```
reorder_qty = max(0, forecast_demand_over_lead_time + safety_stock − on_hand)

safety_stock = z × σ_forecast_error        z from the target service level (95%)
lead_time    = 1 week                      the dataset restocks every 5–14 days
```

Method:

1. Compute weekly forecast error σ from the validation window of §6.
2. Walk the historical weeks forward, applying the policy using **only** information available
   at each point, and record stockouts and overstock.
3. Compare against the observed baseline: 3.00% stockout lines on sales, 7,993
   store-product-week censoring events in the panel.
4. Report the measured change, plus units held and units lost.

**Which forecast horizon feeds this.** The reorder quantity consumes the **1-week** horizon
from §6.2, because the lead time is one week — a 4-week forecast would cover four lead times
and over-order by roughly 4×. The 2- and 4-week horizons are reported in the forecasting
results and used on the dashboard, not in the reorder arithmetic. This is stated because
"which horizon?" is the first question a reviewer asks about any reorder policy.

**The report will state the measured reduction, not the 25–40% target, unless the backtest
actually produces it.** Asserting the target without the backtest would be fabrication.

Output `inventory_recommendations.csv` ranks store-product pairs by urgency, joining the panel's
`stockout_count` with the forecast — this is the "what to reorder" list from the data
dictionary's SQL example.

---

## 10. Testing

Existing: `test_retail_pulse.py`, `test_aggregate.py`, `test_null_handling.py`. Added:

| Test | Asserts |
|---|---|
| `test_no_leakage.py` | For every churn snapshot, the max feature timestamp ≤ snapshot date. **The single most important test in the project** — leakage is invisible in the metrics and inflates AUC |
| `test_metrics.py` | WAPE and MAPE against hand-computed values on a small fixture, including a fixture with zeros, so the 77.9%-zeros case is pinned |
| `test_inventory.py` | Reorder arithmetic, including the understock and no-shortfall branches |
| `test_precompute.py` | Every expected output file exists, is non-empty, and has the expected columns |
| `test_ingest.py` | A deliberately corrupted fixture fails validation rather than passing |

Run: `python -m pytest tests/ -q`

---

## 11. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Local Python 3.14 / pandas 3.0 vs Streamlit Cloud runtime (3.11–3.12) | No 3.14-only syntax. `pandas>=2.2` in requirements. Smoke-test on Cloud **on Day 4 morning**, not Day 5 |
| R2 | First Streamlit Cloud deploy is slow and may fail on dependency resolution | Deploy a Hello World on Day 1, before any modelling, so the risk surfaces while there is time to react |
| R3 | LSTM training time on 780,000 rows | Train on category aggregates, not the full panel. Cap epochs, log per-epoch time |
| R4 | Accuracy targets missed (MAPE ≤ 12%, AUC ≥ 0.88) | Report honestly with error analysis. A clearly explained miss is recoverable; a fabricated number is not |
| R5 | Scope pressure — six requirements plus infra in five days | Tier ordering in §2. Dashboard and report are never traded for infrastructure |
| R6 | Participant cannot explain a number in the demo | Every figure in the report links to the script and command that produced it, so any question can be traced and re-run |

---

## 12. Deliverables plan

| # | Deliverable | Weight | Target |
|---|---|---|---|
| 1 | **Live public demo URL** | 30% | Day 4 — Streamlit Community Cloud, public HTTPS, no sign-up |
| 2 | PDF report, 10–18 pages, A4, ≤ 12 MB | 25% | Day 5 — from `reports/RetailPulse_Report.md` |
| 3 | GitHub repository | 20% | Day 5 — clean folders, notebooks, history |
| 4 | README.md | 15% | Day 5 — written by the participant |
| 5 | Demo video, 4–8 min | 10% | Day 5 — with a real drift simulation on screen |

ZIP ≤ 500 MB, satisfied because raw CSVs are gitignored and the largest committed artefacts
are the ~3 MB of processed aggregates.

Backup per the brief: if the demo is down, the video plus screenshots must still convey full
functionality. Screenshots are captured during Day 4 as part of normal testing.

---

## 13. Daily log convention

`DAILY_LOG.md` at the project root, one entry per working day:

```markdown
## Day 3 — 2026-10-04
**Goal:** Customer segmentation
**Done:** K-Means on 10 RFM features, k=7 chosen on silhouette
**Decision:** k=7 over k=6 — silhouette 0.31 vs 0.29; fewer segments is easier to act on
**Numbers:** 8,000 customers, 7 segments, largest is 1,447 (Champions, 25.4% of revenue)
**Next:** DBSCAN comparison, then name the segments from centroids
**Blocked:** nothing
```

This is required by the brief's "Detailed Execution Timeline" section and carries
documentation weight, so it is kept accurate rather than written at the end.

---

## 14. Honest limitations to publish

Collected here so they are stated deliberately rather than discovered by a judge:

1. MAPE is undefined on 77.9% of panel rows; MAPE-on-non-zero is reported with the subset
   always stated.
2. Prophet is fitted at category level, not per store-product — 7,500 fits cannot meet the
   batch SLA.
3. "30-day ahead" is rounded to 4 weeks on a weekly grain.
4. The dataset is synthetic, seeded and reproducible, with basket behaviour calibrated
   against UCI Online Retail II (1,067,371 rows). This is disclosed as a strength, not hidden.
5. Tier 3 infrastructure is authored and statically validated but not deployed.
6. The inventory improvement figure is a historical backtest, not a production A/B result.
7. Accuracy targets may be missed; the measured number is reported either way.
