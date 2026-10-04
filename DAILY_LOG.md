# Daily Progress Log — RetailPulse

Zidio Data Science & Analytics submission. One entry per working day, newest at the
bottom. Written as the work happens, not reconstructed afterwards, so the "what I did
today" column stays honest.

Times are local. Every entry records what was **run**, not what was intended.

---

## Day 1 — Project setup, dataset verification, pipeline skeleton

### Goal
Get the whole project into one folder with a verified, reproducible data foundation, so
that every later module can assume clean input instead of re-checking it.

### What I did

**Verified the three uploaded files.** Built `load_quickstart.py` to load the sales CSV
(250,000 × 27), the demand panel (780,000 × 15) and the data dictionary (471 lines, 22
sections), and cross-check every row and column count against `facts.json`.

Two bugs found and fixed in that script while writing it:
- The dictionary prints `↔`, which a Windows console cannot encode in its default
  `cp1252` codepage. Printing it raised `UnicodeEncodeError` and killed the report
  halfway. Fixed by reconfiguring stdout to UTF-8 before anything prints.
- Counting markdown sections with a plain `line.startswith("#")` also matched three
  Python `#` comments inside a fenced code block, reporting 25 sections instead of 22.
  Fixed by tracking fence state.

**Exploratory analysis.** `analyse_quickstart.py` covers monthly revenue, weekly trend,
top products, geography, category summaries, promotion effectiveness, category mix
seasonality, stockout impact, lost-demand hotspots and RFM.

Four claims were wrong on the first pass and were corrected against the data:
1. *Pareto.* Top 10 products = 15.9% of revenue from 0.8% of the catalogue, not "94.9%".
   Revenue is concentrated but this is a long-tail catalogue, not a Pareto extreme.
2. *Promotion lift.* Comparing units-per-line gave 0.99x, i.e. promotions looked
   useless -- because that metric is flat by construction in this generator. Switched to
   day-level campaign vs non-campaign totals controlled for weekday: 563 of 731 days
   (77.0%) run some promotion. **CORRECTION (see 17:32 entry): the lift figure originally
   written here, ~1.06x, was wrong.** Measured properly the volume lift is **0.9999x**
   (3.8836 vs 3.8840 units/line) and realised revenue per unit is **0.6931x**. Promotions
   in this dataset change discounting, not volume. Sat/Sun have no non-campaign
   counterpart at all, because `Weekend_Flash` covers them. **The data dictionary's 1.56x
   figure does not reproduce at realised day level**; it is the generator's configured
   multiplier, not an observed effect. Flagged rather than quietly dropped.
3. *Category seasonality.* Was comparing absolute monthly units, which mostly measures
   the chain-wide festive peak. Switched to each category's share of that month's total
   units, which is what the dictionary describes. Beverages peaks in May (26.5% share),
   troughs in November (19.9%); Apparel peaks in October; Grocery runs the opposite way.
4. *RFM.* Recency was ranked backwards (most recent customers scored as least recent),
   which inverted the whole ordering. Fixed, and 61 arbitrary customer codes collapsed
   into 7 named segments.

**Read the brief** (10-page PDF) and extracted all six functional requirements, the
deliverable weightings, and the 28-day plan. Noted the 5 days actually available and
agreed the approach: nothing gets dropped, depth is tiered, and the highest-weight
deliverables — live demo, report, repo, README — are finished first.

**Wrote the design spec** at
`docs/superpowers/specs/2026-10-04-retailpulse-design.md` (462 lines). Then re-read it
looking for placeholders and for anything I had stated without deciding. One thing
surfaced: the inventory module had not been told *which* forecast horizon it consumes.
Fixed — reorder quantity uses the **1-week** horizon only, because ordering off a 4-week
forecast over-orders roughly 4x.

**Restructured into one folder.** Raw data into `data/raw/`, tests into `tests/`, and
created `src/`, `app/`, `data/processed/`, `models/`, `reports/figures/`. All path
constants updated.

**Converted the three existing test files to pytest.** They were runnable scripts that
called `SystemExit()` at import time, so `python -m pytest tests/` died with an
INTERNALERROR before running a single test. Rewrote them as real test functions,
preserving every assertion and adding granularity — the validator negative tests went
from 1 script to 9 individual cases. Added a root `conftest.py` so tests can import the
project root, with session-scoped fixtures so the 250,000-row sales frame is loaded once
per run instead of once per module.

**Wrote `src/config.py`** — the single source of truth for paths, row-count constants,
forecast horizons, churn targets, service levels and the seed. Every module imports
paths from here, so the final rename from `retail-pulse-data/` to `RetailPulse/` is a
one-file change.

**Wrote `src/ingest.py` (F-01).** Loads both CSVs, parses the dates, and enforces the
contract: exact column order, shape, null policy, and every documented row-level rule
including `sales_amount = quantity × (price − discount)` to the paisa and
`quantity_sold == 0 ⟹ inventory_level == 0`. It **raises** on any breach rather than
warning and continuing, because a silent coercion would let a broken extract reach the
models and produce a confident wrong answer.

**Wrote `tests/test_ingest.py`** — 26 tests that each build a valid fixture, corrupt
exactly one thing, and assert the ingest refuses it. A validator that has only ever seen
good data has not been tested.

**Wrote `src/metrics.py`** — WAPE (primary), MAPE on the non-zero subset, sMAPE, MAE,
RMSE, MASE, signed bias, plus `precision_at_k` / `recall_at_k` / `lift_at_k` for the churn
brief. `tests/test_metrics.py` pins each one against arithmetic done by hand, on a fixture
that deliberately contains zero-demand rows, so the 77.9%-zeros case is covered before it
happens in production.

**Wrote `src/features.py`** — 10 customer features (RFM plus basket shape, category
diversity, promo sensitivity, channel mix, tenure), 9 monthly labelled churn snapshots, and
lag/rolling features for the weekly panel.

**Wrote `tests/test_no_leakage.py`** — 22 tests that do not trust `features.py`'s own
bookkeeping. They go back to the raw transactions and recompute each feature from rows
dated on or before the snapshot, then compare. Leakage is the failure mode that cannot be
seen in a metric: a model fed post-snapshot features scores beautifully and is worthless.

### The bug that mattered most today

The leakage test caught a real defect in `rfm_features`, and it is worth recording because
it would have quietly damaged the churn model in the one way hardest to notice.

The feature index was being built from the **window** groupby. So any customer with no
purchase inside the 90-day window was dropped from the snapshot entirely. Those customers
are dormant — which is to say, the label is `churn = 1` for them. **The rows that define
the answer were being deleted before the model ever saw them.**

Measured effect on the real data:

| | Before the fix | After the fix |
|---|---|---|
| Customers per snapshot | ~4,133 | ~6,422 |
| Churn rate | 32.0% | **42.9%** |

The old 32% was not a fact about customers. It was an artefact of having removed the
churners from the denominator. Every downstream number — AUC, precision@top-20%, segment
sizes, the dashboard's risk table — would have been computed against a population that had
been pre-filtered to be the easy half.

Fix: build the row set from the **lifetime** customer list and reindex the window
aggregations onto it, so dormant customers appear with `lines = 0`, `frequency = 0` and a
large `recency_days`. Window counts of zero are then explicitly "none", not missing, so a
model cannot confuse a dormant customer with an unobserved one. `include_unconverted=False`
remains available for segmentation, where dropping non-buyers is the intent.

The lesson I am carrying forward: **a filter that runs before a label exists is a
hypothesis about the answer.** Tests are the only thing that catches it.

### Verification run today

| Check | Result |
|---|---|
| `python load_quickstart.py` | PASS — 250,000 × 27, 780,000 × 15, matches `facts.json` |
| `python analyse_quickstart.py` | exit 0, 10 sections |
| `python verify_dataset.py` | **54/54 checks passed** |
| `python audit_quality.py` | AUDIT PASSED — no nulls, no duplicates, dates well formed |
| `python reconcile_panel.py` | exit 0 — panel reconciles to restricted sales with **0 units** residual |
| `python collect_facts.py` | exit 0 — `facts.json` byte-identical (`CE903804…`) after the move |
| `python methodology/verify_pdf.py` | PASSED — all headline numbers present, 3 regression guards hold |
| `python methodology/verify_doc.py` | all document checks passed |
| `python -m src.ingest` | exit 0 — F-01 contracts satisfied, `facts.json` cross-check ok |
| `python -m src.metrics` | exit 0 — self-check against hand arithmetic |
| `python -m src.features` | exit 0 — 8,000 customers, 9 snapshots, 62,247 labelled rows, leakage guard passed |
| `python -m pytest tests/ -q` | **102 passed** in 11s |

### Data facts confirmed today

- Revenue **INR 210,242,624.35**, units **970,998**, 60,538 baskets, 8,000 customers,
  1,200 products, 50 stores, 12 cities, 731 days (2024-01-01 → 2025-12-31).
- **Lost demand is 7,500 rows** (`quantity_sold == 0`). Separately, **27,749 rows sold
  the final unit** and also sit at `inventory_level == 0`. The distinction matters: using
  `inventory_level == 0` as the stockout test overstates lost demand **3.7×**.
- Panel: 104 complete weeks, 50 stores × 150 products, **77.9% zero-demand rows**,
  6,959 weeks with a stockout, 7,993 total stockouts.
- The panel reconciles to the sales file with **zero units residual** once restricted to
  the panel's assortment and its 104-week window. It is not derived from sales; it is an
  independent build that agrees.

### Traps recorded for every later module

| Trap | Handling |
|---|---|
| Lost demand is `quantity_sold == 0`, **not** `inventory_level == 0` | One-directional rule, pinned by a test |
| Final week (starts 2025-12-29) holds only 3 of 7 days | Dropped from weekly roll-ups; panel already excludes it |
| 77.9% of panel rows are zero-demand | Real zeros, not missing data. MAPE undefined there → WAPE primary, MAPE on non-zero subset with the subset stated |
| 563/731 days (77.0%) run a promotion | Promotion-lift metrics diluted; Sat/Sun have no control day |
| Dictionary's 1.56x promotion lift does not reproduce | Reported as the generator's multiplier, not a measured result |

### F-02 segmentation — `src/segmentation.py`

    python -m src.segmentation      # 8,000 customers, k=6, silhouette 0.1998

- Silhouette curve over k=2..10 is low and non-monotonic (0.163 → 0.202 peak at k=5, then
  wobbling back up at k=9 and k=10). Reported as-is rather than smoothed. k=5 scored
  marginally higher than k=6, but 6-8 is the brief's range and 6 is within 0.0023 of the
  best, so k=6 was chosen.
- `tenure_days` is excluded from clustering. It is almost a function of the dataset's own
  start date rather than of behaviour, so including it splits the base by join cohort and
  washes out the RFM signal.
- DBSCAN found 2 clusters with 38.1% noise after searching eps down from the median
  5th-NN distance. Both that and the first run's single-cluster collapse are reported.

### `src/precompute.py` — dashboard aggregates

    python -m src.precompute   -> 11 frames in data/processed/

Two corrections came out of this, and the second one matters more than the work.

**1. The promotional lift claim from earlier today was wrong.** I had written that realised
lift was ~1.06x. It is not. Measured properly:

| quantity                             | value      |
|--------------------------------------|------------|
| mean units per promoted line         | 3.8836     |
| mean units per unpromoted line       | 3.8840     |
| **volume lift**                      | **0.9999x** |
| mean discount on promoted lines      | 26.01%     |
| realised revenue per unit ratio      | 0.6931x    |
| configured lift                      | 1.56x      |

There is no volume lift at all — the ratio is 1.0 to four decimal places. My original
metric was `sales_amount / quantity_sold`, which equals `unit_price - discount`, so the 26%
markdown was being counted as a 31% demand collapse. Separating the two components changes
the finding from "weak lift" to "no lift, only a discount". The configured 1.56x does not
appear in the data in any form, and promotions here are targeted rather than randomised
(563 of 731 days, but only 8.4% of line items), so even the volume ratio is an association
rather than a causal estimate.

**2. Sales and panel describe different product universes.** I had been assuming the panel
was 50 stores x 50 products.

| frame  | rows     | stores | products | series | weeks |
|--------|----------|--------|----------|--------|-------|
| sales  | 250,000  | 50     | **1,200** | —     | daily, to 2025-12-31 |
| panel  | 780,000  | 50     | **150**   | 7,500 | 104, to 2025-12-22 |

So the panel covers a 150-product subset, and it already ends 2025-12-22 — the partial-week
guard is a no-op on the panel and only protects the sales weekly rollup. Worth knowing
before anyone joins the two frames on `product_id` and quietly loses five sixths of the
catalogue.

Headline aggregates: repeat purchase rate 87.2% (customers buying on more than one
distinct date), 77.9% zero-demand share on clean weeks, and the top 20% of store-product
pairs hold **36.9%** of all units — a flatter long tail than expected, which is the argument
for pooling the tail rather than fitting 7,500 separate models.

### Bugs I introduced and fixed today

Worth writing down, because the first two would both have been silent:
1. First attempt at the `load_quickstart.py` path fix treated the script's own folder as
   a subfolder, so it looked for `data/raw` one level too high and reported all three
   files missing. The `MISSING:` output made it obvious immediately.
2. `Series.map(WEEKDAY_NAMES)` with a plain list raises `TypeError: 'list' object is not
   callable` — pandas wants a callable or a mapping. Replaced with an index-to-name dict.
3. Three of my own MASE test expectations were arithmetically wrong (the module was
   right). Recomputed them on paper. The same happened with a `roll_mean_4` assertion —
   mean(10,20,30,40) is 25, not 20 — and with a claim that tied scores give lift 1.0.
   They do not: ties resolve to the earliest rows, which is deterministic by design and
   documented, so the test now asserts that behaviour instead of a random-ranking fiction.
4. `assert_no_leakage` initially flagged the `churn` column as a leak, but a labelled
   training frame from `churn_snapshots` is supposed to carry it. Added an explicit
   `allow_labels` flag and a `feature_matrix()` helper so the separation is deliberate
   rather than a matter of remembering to drop a column.
5. `silhouette_scores` subsampled the rows for scoring but fit K-Means labels on the full
   set — 4,000 points against 8,000 labels. I also claimed this was "silently wrong" when
   writing the test; sklearn raised `ValueError: inconsistent numbers of samples`
   immediately, so it was loud, not silent.
6. The segment summary printed `total_units` that was actually summing `monetary`, because
   `rfm_features` drops the units column. A wrong column that looked plausible in a table.
   The column is gone rather than faked.
7. Two clusters were both named "At risk". Fixed by appending the strongest distinguishing
   trait, but the first version of that fix could still collide: six centroids sharing one
   base name exhaust a small suffix vocabulary. Now each colliding group is also ranked
   along whichever trait separates it most, so uniqueness is guaranteed, not hoped for.
8. The trait test was purely relative, so a segment that is 98.7% in-store got labelled
   "online-heavy" for beating a near-zero median. Behavioural labels now need an absolute
   floor. That one was worth catching: it would have been very quotable and very wrong.
9. `precompute.py` used the panel's column names (`units_sold`, `revenue`) on the sales
   frame, which actually carries `quantity_sold` and `sales_amount`. Both are now named
   explicitly at module level and checked in `build_all`, because the mistake repeats easily.

### Other things worth knowing

- WAPE and MAPE disagree sharply on this data and that is expected: sMAPE on a row with
  actual 0 and forecast 10 scores the maximum 2.0. A metric that punishes forecasting
  anything into a nil-demand week is not the one to lead with when 77.9% of weeks are nil.
- The churn population is 42.9% positive — heavily imbalanced. AUC will look fine while
  a trivial always-churn classifier scores 0.43 precision@top-20%. That is exactly why
  the brief's `precision@top-20% >= 0.75` and `lift` are the meaningful numbers, and why
  they get reported next to the base rate.
- `scikit-learn` was not installed locally; installed 1.9.1. `requirements.txt` already
  listed it.
- `pytest.ini` now turns FutureWarning and DeprecationWarning into errors, so pandas 3.0
  deprecations cannot accumulate silently across the remaining modules.

### F-04 churn — `src/churn.py`

    python -m src.churn

| metric              | value  | brief target | met |
|---------------------|--------|--------------|-----|
| precision @ top 20% | 0.7703 | >= 0.75      | yes |
| lift @ top 20%      | 1.8137 | —            | —   |
| recall @ top 20%    | 0.3628 | —            | —   |
| AUC                 | 0.7315 | 0.88         | **no** |

- Split is by snapshot date, not by row: train on 2025-01 to 2025-06 (40,588 rows), test on
  2025-07 to 2025-09 (21,659 rows). Nine snapshots of the same 8,000 customers means a random
  split would put the same person on both sides.
- **AUC misses the brief's 0.88 and that gets reported, not buried.** The model is good at
  the thing the brief actually cares about — the top-20% campaign hits 77% precision against
  a 42.5% base rate — and weaker at global ranking. Both numbers are in the table.
- SHAP top driver is `recency_days` (0.524), then `frequency` (0.228), then `tenure_days`
  (0.089). The right driver, in the right direction. `recency_days` leading is also the check
  that would fail first if the label or the window were wrong.
- Imputation sits inside a `Pipeline`, fit on train rows only. Dormant customers have an
  undefined window average; they are kept and imputed, not filtered, because they are the
  target population.
- Added `model_matrix()` to `src/features.py`: `feature_matrix()` deliberately keeps
  `customer_id` and the dates for joining and auditing, so an estimator cannot use it.

One test bug worth recording: the synthetic churn fixture drew a *second* independent
`recency` for the label, so churn was unrelated to every column and the model correctly
scored lift 1.0. The model was right; the fixture was wrong. At a 77% base rate lift is
arithmetically capped at 1/0.77 = 1.30, so the first version of that test asserted a target
no model could reach. Base rate moved to ~40%, like the real snapshots.

### Verification

    python -m pytest tests/ -q                     -> 169 passed in 44s
    python -m pytest tests/ -q -m slow --durations -> 3 passed in 14s

### Next

- `src/forecasting.py` — Prophet + LSTM + ensemble, WAPE as the headline number.
- `src/inventory.py` — reorder quantities off the 1-week horizon at a 95% service level.
---

## 2026-10-04 17:32 — F-03 forecasting and F-05 inventory

### F-03 forecasting (`src/forecasting.py`)

Pooled-total weekly backtest on all 7,500 pairs, three horizons, five candidates. Held-out
weeks are never visible to the fit; the holdout check asserts the fit receives
`series.iloc[:-horizon]`.

| horizon | model | WAPE | MAPE (non-zero) |
|---|---|---|---|
| 1 | prophet | **0.006829** | 0.006829 |
| 1 | ensemble_naive_prophet | 0.007106 | 0.007106 |
| 1 | seasonal_naive | 0.021041 | 0.021041 |
| 1 | ridge_seasonal | 0.055511 | 0.055511 |
| 1 | lstm | 0.059988 | 0.059988 |
| 2 | seasonal_naive | **0.021336** | 0.021352 |
| 2 | ridge_seasonal | 0.035383 | 0.036669 |
| 2 | ensemble_naive_prophet | 0.036930 | 0.037796 |
| 2 | prophet | 0.073035 | 0.075903 |
| 2 | lstm | 0.113050 | 0.110189 |
| 4 | seasonal_naive | **0.038349** | 0.035871 |
| 4 | ridge_seasonal | 0.038592 | 0.037304 |
| 4 | ensemble_naive_prophet | 0.067545 | 0.064889 |
| 4 | prophet | 0.112415 | 0.110955 |
| 4 | lstm | 0.241546 | 0.237243 |

Selected per horizon: prophet at h=1 (WAPE 0.68%), seasonal naive at h=2 (2.13%) and h=4
(3.83%). Inventory consumes h=1: **8,393.98 units** next week.

Three findings worth stating plainly rather than burying:

1. **The ensemble did not win anywhere.** A 50/50 blend of seasonal naive and Prophet beat
   seasonal naive at h=1 (0.0071 vs 0.0210) and lost to it at h=2 and h=4. Blending is not
   free: it inherits the worse model's error wherever the two disagree. It is kept in the
   candidate list so the comparison is on the record, not because it helps.
2. **Prophet is strong at one week and poor at four.** Same model, same data: 0.68% at h=1,
   11.2% at h=4. A single "Prophet WAPE" number would have hidden that.
3. **LSTM lost at every horizon.** It is the least accurate candidate at all three
   horizons, which is the opposite of the brief's framing. Reporting it as a serious
   contender would be dishonest; it stays in the benchmark so the comparison exists.

**Pooled WAPE is not comparable to the panel-level 12% MAPE target.** The pooled total has
no zero-demand weeks, so its MAPE filter excludes nothing and MAPE reduces to a per-point
ratio close to WAPE. The panel is 77.9% zero-demand. A pooled 3.8% says nothing about the
sparse store-product level the brief's target was written about — that remains unmet and
will be reported as unmet.

Reconciliation is exact: 50 store forecasts sum to 8,393.98, matching the pooled number.

### Two bugs I wrote, and what the tests caught

- `reconcile()` broadcast elementwise, so it summed correctly only when base and weights
  happened to be the same length. Now takes a per-period base and a per-series weight and
  returns a periods x series matrix where each *row* sums to that period's forecast.
  `by_store` in `fit_and_forecast` was switched to it; verified the 50 rows sum to the total.
- A test asserted pooled WAPE equals pooled MAPE. It does not (0.046032 vs 0.046111) — WAPE
  weights points by size, MAPE does not. The test now asserts they are close and that MAPE
  is finite. The underlying honest point stands: the non-zero filter excludes nothing here.

### F-05 inventory (`src/inventory.py`)

Reorder quantities for all 7,500 store-product pairs from the h=1 forecast.

**The 95% service level in the brief is not the number the arithmetic produces.** Reorder
quantity is a newsvendor calculation, not mean demand plus a buffer. Underage costs lost
margin; overage costs purchase plus holding. With price 23.59, unit cost 14.15 (60% of
retail), 25% annual holding and 1-week lead time:

    critical ratio = p / (p + o) = 9.44 / (9.44 + 0.068) = 0.9928

So the plan is calibrated to a 99.3% quantile, not 95%. That is the correct consequence of
these costs, not a bug. Reaching for the 95th percentile because the brief says "95%"
conflates the critical ratio with the confidence on the demand distribution.

**Safety stock is Poisson, not a z-multiple.** Demand is zero 77.9% of weeks, so it is
nothing like normal and a z-multiple of a large standard deviation is not a percentile of
anything. Using the exact Poisson quantile over lead time:

| method | pairs flagged | units to order |
|---|---|---|
| Poisson quantile (used) | 496 | 2,081 |
| normal z approximation | 1,518 | 7,465 |

The normal approximation overstates by 5,384 units, 3.6x. Its largest order was 23.3 units
of a product forecast at 2.2/week — over ten weeks of cover on a one-week lead time.

The Poisson figure is not universally conservative, and there is a test saying so: on smooth
near-continuous demand (200 +/- 8) the normal approximation orders *less*. Poisson is the
right tool for intermittent demand specifically.

Cost assumptions are recorded in the summary because they drive the answer, and the dataset
has no cost column: purchase cost 60% of retail, holding 25%/yr, lost margin 40% of price
(a stockout loses margin, not revenue — the customer substitutes).

### Corrections to earlier notes

- `facts.json` untouched; no edit made to promote or lift figures today.
- Earlier `~1.06x` promotion lift in this log is wrong and stays struck through. Measured
  volume lift is 0.9999x.

### Verification

    python -m src.forecasting                      -> exit 0, no failed score rows
    python -m src.inventory                        -> exit 0
    store_units sum == pooled forecast              -> 8,393.98
    python -m pytest tests/test_forecasting.py      -> 27 fast + 2 slow
    python -m pytest tests/test_inventory.py        -> 31 passed in 57s
    python -m pytest tests/ -q                      -> 229 passed in 164s

Three tests initially failed and each was informative rather than cosmetic: a rounding step
leaked 0.002 units of reconciliation error (removed it; full precision now), a test assumed
critical ratio should scale with absolute price (it is scale-invariant; only the margin
ratio matters), and a test assumed Poisson always orders less than normal (false on smooth
data).

### Next

- `src/dashboard.py` — Streamlit app over `data/processed/`.
- Drift/quality artifacts and the final report, then README and video outline.
---

## 2026-10-04 — F-06 dashboard, F-07 drift, report and README

### F-06 dashboard

Five Streamlit pages over `data/processed/` only: Home, Demand Forecasting, Customer
Segments, Churn Risk, Inventory Recommendations. Streamlit 1.65.0, config pinned for a
dark theme and localhost binding.

Two runtime bugs came out of the first AppTest run and both were mine, not Streamlit's:

- `Customer_Segments.py` passed a Series into a DataFrame alongside plain arrays. It carried
  a customer-count index and pandas tried to align it against the 0..5 index of the other
  columns. Arrays now.
- `Demand_Forecasting.py` selected forecast columns after renaming, so the lookup used the
  pre-rename name and raised `KeyError`. Selection happens before the rename now.

### The segment naming bug

The segment summary was a DataFrame indexed by segment name. `precompute` wrote every
aggregate with `index=False`, so the name was dropped and `segment_summary.csv` held six
anonymous rows. The page recovered a name by reading the row position and looking it up by
`segment_id` — but the summary is sorted by revenue, so row order is not id order.

All six labels on screen were wrong. Champions (2,020 customers, INR 83.7M, the largest
earner) was displayed as "At risk (in-store, monetary high)", and Dormant was displayed as
"At risk loyal".

`precompute` now promotes the index to a real column before writing, so the join key travels
with the data, and the page joins on that name. Three regression tests: the column exists,
each row's name matches its own customer count and revenue, and the rendered name-to-count
mapping equals the one derived from the per-customer file.

This is the bug I would most expect a reviewer to catch by reading the screen and thinking
the labels look wrong. It looked right.

### Forecast chart

The chart concatenated all three horizon paths. Their `weeks_ahead` labels repeat across
horizons, so concat raised on a duplicate index. It plots the 4-week path alone now — one
line, and not three nested horizons drawn as if they were rivals.

### F-07 drift

`src/drift.py` compares 2024 against 2025 on the weekly panel: PSI for numerical columns,
total-variation distance for categorical ones, plus a category-mix table on shares. Result
is no significant drift, which is what a seeded generator should produce; the test asserts it
rather than assuming it.

PSI is implemented in-module because the same number has to appear in the printed table, the
CSV and the tests. It is validated against synthetic shifts before being pointed at real
data: zero-inflation 20% → 60% reads 0.26, mean 1.0 → 1.5 reads 0.17, mean 1.0 → 5.0 reads
3.79. One test pins that PSI is *not* symmetric under swapping arguments, since the bins come
from the reference period — that is the definition, not a defect.

Evidently took three attempts. `Report.save_html` does not exist on the 0.7 top-level API,
and the HTML renderer lives at `evidently.legacy.report` with the preset at
`evidently.legacy.metric_preset`. All paths are tried and the outcome printed.

### Two test failures worth recording

- `test_every_read_by_the_dashboard_lands_in_processed` passed alone and failed in the suite.
  `st.cache_data` lives for the whole process, so the pages read nothing at all and the spy
  saw an empty list. An audit that passes because it observed nothing is worse than no audit,
  so the test now clears the cache and asserts reads actually happened.
- `test_home_shows_the_promotion_finding_as_measured` asserted against `at.info` titles while
  the numbers were rendered as markdown. It was checking an empty string. It now reads both.

### Report, README, video outline

`reports/RetailPulse_Report.md` states the three missed targets plainly: churn AUC 0.7315
against 0.88; the panel-level MAPE target not demonstrated, because the pooled 0.0068 figure
excludes zero-demand weeks by construction; silhouette 0.1998 meaning the segments overlap.
`README.md` repeats them in the results section rather than burying them. The video outline
instructs the presenter to say the AUC miss out loud.

### Git

Repository initialised on `main`, three commits. `.gitignore` keeps `data/raw/` out and
commits the small aggregates the dashboard reads. The Evidently HTML bundle is 4 MB and
regenerable, so it is ignored; its two CSV companions stay committed because the report quotes
their numbers.

### Verification

    python -m src.forecasting                -> exit 0, h=1 prophet WAPE 0.006829
    python -m src.inventory                  -> exit 0, 2,081 units / 496 pairs
    python -m src.drift                      -> exit 0, verdict: stable
    python -m src.dashboard_data             -> 21 aggregates in 0.09s (budget 8s)
    python -m pytest tests/test_dashboard.py -> 44 passed
    python -m pytest tests/test_drift.py     -> 29 passed
    python -m pytest tests/ -q               -> 305 passed in 209s

`pytest.ini` now names the numpy.core `DeprecationWarning` that evidently emits instead of
silencing the category, so a DeprecationWarning from our own code still fails the suite.

### Next

- Live Streamlit Cloud deployment needs the participant's own accounts; not done here.

## 2026-10-04 (late) - Audit pass: Tier-3 correctness and honest numbers

### The CronJob did not sequence anything

`deploy/kubernetes/pipeline-cronjob.yaml` listed all six stages as six containers in one Pod,
with a comment claiming drift ran "after the models" and another claiming the `verify` container
would "stop the pod". **Neither was true.** Kubernetes starts every container in a Pod
concurrently, so `verify` ran at t=0 against artefacts that did not exist yet, and a container
exiting non-zero did not stop the stages beside it.

Sequencing moved into `src/pipeline.py` as `STAGES`, `STAGE_OUTPUTS` and `run_pipeline()`, and
the YAML became one container calling `python -m src.pipeline`. The Airflow DAG imports the same
list. The order is now asserted in `tests/test_pipeline.py`.

What made this survivable to review is that two of the original tests asserted the broken shape:
`test_cronjob_runs_drift_after_the_models` checked the index of `drift` among container names,
which passes perfectly while the ordering means nothing. A test can pin a bug in place as
happily as a fix. Those two tests were rewritten, not satisfied.

### The duplicated filename lists were lying

The pipeline contract, the CronJob's shell loop and the DAG's `VERIFY` each listed expected
outputs by hand. They disagreed: the contract required `drift_report.csv` from a module that has
always written `drift_summary.csv`, into a directory that does not exist (`drift` writes to
`reports/drift/`), and required `kpis.csv` from `precompute`, which writes `kpis` as a dict key
rather than a `.csv` literal. Now there is one list and `tests/test_pipeline.py` reads the
filenames back out of each module to check the contract against the code.

### The pipeline pointed at an image nobody built

The CronJob ran `retailpulse:1.0.0` — the **dashboard** image, built from
`requirements-dashboard.txt`, which deliberately omits Prophet, TensorFlow and XGBoost. Every
stage would have died on `import prophet`. Fixed by adding `Dockerfile.pipeline` for the full ML
stack and pointing the CronJob at `retailpulse-pipeline`. `tests/test_pipeline.py` now fails if a
manifest references an image with no Dockerfile to build it.

### Docker would not have built

`COPY src/dashboard_data.py ./src/ 2>/dev/null || true` — `COPY` is not a shell, so that is a
parse error, and the file it named does not exist (the module is `app/dashboard_data.py`,
already carried by `COPY app/`). The build failed twice over. `tests/test_tier3.py` now rejects
shell syntax in any `COPY` and checks that every `COPY` source exists in the build context.

### `_matrix.csv` was not stray debris

`data/processed/_matrix.csv` looked like a leftover debug dump and I deleted it. It came back on
the next `precompute` run. The cause: `_`-prefixed keys in precompute's `out` dict are internal by
convention, and the write loop ignored that convention. `_concentration` is a float so the
`isinstance(DataFrame)` check happened to skip it; `_matrix` is a DataFrame, so it did not get
that protection. Deleting the file treated the symptom. The write loop now honours the prefix,
and there is a test asserting `data/processed/` holds no internal artefacts — the dashboard
auto-loads everything in that directory.

### The churn matrix did not reproduce from the published scores

`churn_metrics.csv` called 4,349 cases at its operating threshold; re-applying that threshold to
the published `churn_scores.csv` called 4,309. Cause: XGBoost returns float32. Comparing a float32
array against a float64 threshold makes NumPy cast the threshold *down* to float32, so ~40 scores
tied with the cutoff were included in memory; written to CSV as float64 and read back, the
comparison happened in float64 and the cutoff's low bits excluded them. `src/churn.py` now has one
`scores_of()` helper that widens to float64 once, used by the metric path and the write path.

The second churn failure was my own test: the fixture used six invented feature names while the
project has ten. A fixture with its own column list passes even if `local_explanations` zips the
wrong label onto every contribution, because there is nothing to misalign against.

### MASE: a passing-looking test that proved nothing

Pinning "seasonal naive scores MASE = 1" took three attempts, and the first two were wrong in
instructive ways. A strictly 52-periodic series has *zero* in-sample naive error, so `mase`
returns NaN and any perturbed version divides one ~1e-14 float-noise term by another — a
meaningless 1.055. The working version uses a linear series where every lag-52 step is exactly
+104, so numerator and denominator are the same constant and the ratio is 1 to the last bit.

Also worth recording: `metrics.bias` is `mean(actual - pred)`, so **positive means
over-forecast**. The obvious sign convention is the opposite one, and the docstring says so.

### The inventory target is not met, and the report did not say so

The walk-forward backtest measures a **2.09%** error reduction against a 25–40% target. The report
had no section on it at all. The decomposition is more informative than the number: overstock
collapses from 18,301.9 to 44.0 units while understock rises from 2,398.7 to 20,223.8. The policy
is not reducing total error, it is moving the error from one side of the ledger to the other.

Related: the 0.9928 critical ratio implies a **99.28% service level (z = 2.449)**, above the 95%
in `config.py`. The report had claimed the ratio was "derived from the 95% service level". It is
derived from the cost economics; the 95% does not drive the shipped quantity. Flagged in the
report as a decision for the brief's owner rather than quietly re-tuned.

### Corrected facts

| Claimed | Actually |
|---|---|
| 276 / 305 tests passing | 455 |
| Dec and Jan peaks in 7 of 9 categories | 10 categories; peak months are Oct (4), May/Jun (3), Nov (1). **No December or January peak.** |
| Raw panel 63 MB | 103.7 MB total; demand panel 60.4 MB |
| Drift compares first vs second half of the period | Compares **2024 vs 2025** |
| Critical ratio derived from the 95% service level | Derived from cost economics; implies 99.28% |

### Report figures and PDF

`src/figures.py` draws all six figures from the committed CSVs rather than from hard-coded
numbers, so a chart cannot drift away from the table beside it. Two of them exist because the
report was making a claim without showing it: the WAPE-by-horizon matrix makes the Prophet /
seasonal-naive inversion visible, and the inventory chart shows overstock and understock
swapping places instead of the error bar shrinking.

PDF is `pandoc` (markdown to HTML) then Chrome headless `--print-to-pdf`, with
`reports/pdf.css` for A4 sizing and print rules. Result: **10 pages, 0.53 MB, 6 embedded
images** — inside the brief's 10-18 page and 12 MB budgets. `tests/test_figures.py` asserts the
page count and size, so a report that grows past 18 pages fails the suite rather than being
discovered at submission.

### An encoding bug I caused and then made worse

Inserting the figure references with a PowerShell `Get-Content -Raw` / `Set-Content -Encoding
UTF8` round-trip corrupted the file: PowerShell 5.1 reads without a BOM using the ANSI codepage,
so every em dash and curly quote in the file became mojibake. My first repair attempt then
re-encoded the whole text through CP1252, which turned the mojibake back into real characters
but destroyed the eleven that CP1252 cannot represent (`−`, `≥`), replacing them with U+FFFD. A
blanket encoding fix on a file with mixed provenance is worse than the original bug.

Repair was per-line with explicit intended text, then verified: zero U+FFFD, zero mojibake, zero
stray backslashes. The backslash check was not paranoia — `` \`region` `` had lost its `r`
entirely, because `\r` is a carriage return, so the table row silently read `| egion\ |`.

Lesson recorded because it will recur: **do not round-trip UTF-8 through PowerShell 5.1 cmdlets**
in this repo. Use the editor tools, which are encoding-safe.

### Verification

- `python -m pytest tests/ -q` → **471 passed**, 2 warnings, both from `shap`'s own colormap code.
- `python -m src.pipeline`'s artefact check → no stage missing an output.
- `precompute`, `inventory`, `churn`, `drift` re-run against the fixed code.
- Docker, `kubectl` and Airflow are unavailable here, so the manifests are statically and
  unit-validated only. The build and deploy jobs in `ci.yml` are `if: false` for that reason.

### Next

- Live deployment, remote push and the participant's git identity still need real credentials.
- Final prose needs participant review; the brief disqualifies AI-generated submissions.