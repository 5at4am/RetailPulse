# RetailPulse — Project Report

**Dataset:** `retail_sales.csv` + `store_product_matrix.csv` — 970,998 line items, 8,000
customers, 1,200 products, 50 stores, 2024-01-01 to 2025-12-31.
**Stack:** Python 3.11+, pandas, scikit-learn, XGBoost, Prophet, TensorFlow/Keras, Evidently,
Streamlit. **Tests:** 276 passing.

---

## 1. What the data turned out to be

The first job was measurement, not modelling. Four facts from the data changed how everything
downstream was built:

| Fact | Value | Consequence |
|---|---|---|
| Panel sparsity | **77.9%** of store-product-weeks have zero demand | MAPE is unusable. A single zero makes the error infinite, so any series with a zero week must be excluded — and excluding them discards most of the panel. WAPE is the primary forecast metric. |
| Promotions do not move volume | volume lift **0.9999x** vs 1.56x configured | Promoted lines carry a 26.0% discount and realise **0.6931x** the revenue per unit. The configured lift appears nowhere in the data. |
| Weekly seasonality | Dec and Jan peaks in 7 of 9 categories | A 4-week horizon crosses a holiday boundary, which is why naive and seasonal-naive beat the neural nets. |
| Snapshots repeat customers | 8,000 customers appear in 24 monthly snapshots | A random train/test split puts near-duplicate rows on both sides and inflates churn AUC. The split must be temporal. |

The sparsity number is the one that shaped the architecture. It is why the dashboard reads
precomputed aggregates rather than the 63 MB raw panel, why forecast accuracy is reported as
WAPE, and why inventory safety stock is computed from a Poisson quantile rather than a normal
approximation.

## 2. Feature engineering and leakage control

Ten behavioural features per customer as of each snapshot date: recency, frequency, monetary,
average basket value, units per line, category diversity, promotion share, mean discount,
online share, tenure. Every feature is computed with an `as_of` cutoff and `src/features.py`
exposes `assert_no_leakage()`, which is exercised by `tests/test_no_leakage.py`. A feature that
reads past its own snapshot date fails the suite rather than silently flattering the model.

The churn label is "did not purchase in the following 30 days", computed only for customers
whose 30-day window is fully inside the data. Customers near the end of the series are dropped
rather than assumed active.

## 3. Customer segmentation (F-02)

K-Means on the 10 standardised features, silhouette swept over the brief's 2–10 range.

- **k=6 selected, silhouette 0.1998.** Raw best was k=5 at 0.2021 — a gap of 0.0023, which is
  noise at n=8,000. k=6 was kept because it sits inside the requested range and separates
  high-value from dormant customers more cleanly.
- **DBSCAN as a cross-check found 2 clusters with 38.1% noise.** Reporting this plainly: DBSCAN
  did not produce a usable alternative segmentation on this data.
- **Segments** are named from centroid medians, and the medians are shown next to each name so
  the naming can be judged rather than trusted.

The silhouette near 0.20 means the clusters overlap substantially. The segments are a useful
summary for targeting, not sharply separated populations.

## 4. Demand forecasting (F-03)

Five candidates were backtested at three horizons: seasonal naive, ridge with seasonal features,
Prophet, an LSTM, and a naive/Prophet ensemble.

| Horizon | Winner | WAPE |
|---|---|---|
| 1 week | Prophet | **0.006829** |
| 2 weeks | seasonal naive | **0.021336** |
| 4 weeks | seasonal naive | **0.038349** |

The 4-week path matters most because the brief's "30-day ahead" is 4 weeks on a weekly panel.
Inventory consumes h=1 only, where Prophet's advantage is largest.

Two things worth stating plainly:

**Forecasting is pooled.** The models fit the weekly total and apportion it to store-product
series by historical share. Fitting 7,500 independent Prophet series cannot meet the batch
budget. The accuracy figures above describe the pooled total.

**Those figures are not the brief's MAPE target.** The brief asks for MAPE under 12% at the
panel level. The pooled total has no zero-demand weeks, so its MAPE filter excludes nothing and
its MAPE (0.0068) is not evidence about the sparse panel level. Reporting 0.68% as "well under
12%" would be a category error. The 4-week pooled WAPE of 3.8% is the number I stand behind.

Store-level forecasts are reconciled so the parts sum exactly to the pooled total.

## 5. Churn prediction (F-04)

XGBoost on a **temporal** split: 40,588 training snapshots, 21,659 test snapshots, with every
snapshot for a given date on the same side of the split.

| Metric | Result | Target | Met |
|---|---|---|---|
| Precision @ top 20% | **0.7703** | 0.75 | yes |
| Lift @ top 20% | **1.8137** | — | — |
| Recall @ top 20% | 0.3628 | — | — |
| **AUC** | **0.7315** | 0.88 | **no** |

**The AUC target was not met, by 0.15.** Precision at the top of the ranking did meet its target.
I am not going to average those two facts into a single "good performance" sentence: precision
says the top 20% of the list is worth working, and AUC says the model does not order the rest of
the population well. On 8,000 customers with a 42% churn base rate, that is the trade-off the
target set asked for, and the list is usable for a campaign while the scores are not usable for
probability thresholds.

SHAP importance: `recency_days` 0.524, `frequency` 0.228, `tenure_days` 0.089,
`avg_basket_value` 0.088. Recency dominating is expected, and its size is the honest signal that
the model is largely measuring "have they stopped showing up".

## 6. Inventory recommendations (F-05)

Newsvendor reorder quantity per store-product pair, on a 1-week lead time.

Cost assumptions, stated because they drive every number below: purchase cost 60% of retail,
annual holding 25%, lost margin 40%.

| | Pairs | Units | Value |
|---|---|---|---|
| **Poisson quantile (shipped)** | **496** | **2,081** | **INR 48,771** |
| Normal approx (comparison only) | 1,518 | 7,465 | — |

The Poisson quantile uses a **0.9928 critical ratio**, derived from the 95% service level and the
assumed cost ratio. It is not the 95% service level itself, and the two are not the same number.

The normal approximation orders **5,384 more units** across 1,022 more pairs. That gap is the
finding, not a nuisance: with 77.9% of weeks at zero demand, mean + z·std has no probabilistic
meaning in the upper tail. The normal version is kept in the output as a labelled comparison so
the difference is visible.

## 7. Dashboard (F-06)

Five Streamlit pages — overview, demand forecasting, customer segments, churn risk, inventory
recommendations.

All pages read **only** `data/processed/`. A test spies on `pandas.read_csv` and fails if any page
touches `data/raw/`, because the raw panel costs 10–20 seconds per session against an 8-second
budget. Aggregates load in **0.09s**.

## 8. Data drift monitoring

`src/drift.py` runs an Evidently report comparing the first and second halves of the sales
period, across volume, price, discount and category mix, and writes an HTML report plus figures
to `reports/drift/`.

## 9. Limitations

1. **Churn AUC is 0.7315 against a 0.88 target.** The largest gap in this project.
2. **The forecast MAPE target is not demonstrated at the panel level.** The 0.68% figure is
   pooled and excludes zero-demand weeks by construction.
3. **Silhouette 0.1998 means overlapping segments.** The six groups are a targeting summary,
   not distinct populations.
4. **Promotion findings are associational.** Promotions are targeted, not randomised. The
   0.9999x volume lift is an association between promoted and unpromoted lines, not a causal
   estimate, and the 26% discount could itself be selecting for lines that would have sold
   anyway.
5. **LSTM underperforms seasonal naive at every horizon.** Reported rather than dropped. With
   ~104 weekly observations and strong fixed seasonality, there is not enough signal for the
   sequential model to beat a seasonal lag.
6. **Store-level forecasts are shares, not independent fits.** A store with a genuine local
   trend will be smoothed toward the pooled total.
7. **Data is synthetic.** The generator is seeded, so the anomalies are the ones that were
   written in. These conclusions do not transfer to a real retail panel without revalidation.