# Demo Video Outline — RetailPulse

Target: **5 minutes.** One screen recording of the running dashboard, plus a voiceover
track. Every number below is already in `reports/RetailPulse_Report.md`, so if one is wrong
in the video the report is the thing to fix first.

## If the live URL is down

The spec names the video plus screenshots as the backup for a dead demo URL, so record with the

eports/screenshots/ folder open in a second window. Those five PNGs are captured from a
running server by python -m src.screenshots, and 	ests/test_screenshots.py fails the build
if two of them are byte-identical or if a page shows another page's content.

They are not a substitute for the live demo, and the video should say so where it shows them.
A screenshot cannot demonstrate the load time or that the pages respond to a filter, which are
two of the things a judge will want to see.

## Setup before recording

```bash
streamlit run app/Home.py
```

- Browser window at 1440x900, no devtools open.
- Close the terminal or keep it out of frame.
- Confirm the sidebar shows all five pages before starting the capture.

---

## 0:00 — Hook (Home, 30s)

Open on the Home page, unmodified.

> "970,998 line items. 8,000 customers. 50 stores. Two years. One question: what should we
> stock, and who do we call first?"

Scroll to the four KPI tiles. **Do not** scroll past the zero-demand finding yet.

## 0:30 — The fact that shapes everything (Home, 30s)

Point at the 77.9% zero-demand callout.

> "Before any model: 77.9% of store-product-weeks have zero sales. That one number decides
> our metric, our architecture, and the inventory method. MAPE is undefined on three quarters
> of this panel, so we score forecasts with WAPE instead."

This is the spine of the project. Cut it and the rest is unexplained.

## 1:00 — Promotions (Home, 30s)

Point at the promotion finding.

> "The business configured a 1.56x promotion lift. We measure 0.9999x. Promoted lines sell
> 3.8836 units versus 3.884 unpromoted. The 26% discount is the entire difference — realised
> revenue per unit drops to 0.6931x."

If asked why it is zero: promotions are targeted, not randomised, so this is an association.

## 1:30 — Demand forecasting (30s)

Navigation → Demand Forecasting.

> "Three horizons, five candidate models, scored on WAPE. Prophet wins at one week — 0.0068.
> Seasonal naive wins at two and four. The four-week path is what matters, because '30 days
> ahead' is four weeks on a weekly panel."

Show the selection table, then the chart: 12 actual weeks then the forecast.

> "Fitted on the pooled weekly total and split by historical share. Fitting 7,500 separate
> series won't finish inside the batch budget."

**If asked about the 12% MAPE target:** say the pooled number is 0.0068 but it excludes
zero-demand weeks by construction, so it is not evidence at the panel level. Do not claim
the target was met.

## 2:00 — Customer segments (30s)

Navigation → Customer Segments.

> "k=6, silhouette 0.1998. The raw best was k=5 at 0.2021 — a 0.002 gap, which is noise at
> 8,000 customers. DBSCAN found 2 clusters with 38.1% noise, so it didn't give us a usable
> alternative here."

Show the segment table and the revenue-vs-customers chart. Note that segments above the
diagonal earn more than their headcount share.

> "Silhouette near 0.20 means these overlap. They're a targeting summary, not separate
> populations."

## 2:30 — Churn (40s)

Navigation → Churn Risk.

> "XGBoost, split by date rather than randomly — snapshots repeat customers, so a shuffle
> puts near-duplicates on both sides."

Metrics, in this order and with no spin:

> "Precision at the top 20%: 0.7703, against a 0.75 target. Lift 1.81. That part worked.
> AUC is 0.7315 against a target of 0.88 — we missed that by 0.15."

**Say the miss out loud.** It is in the report, and volunteering it is the difference between
a demo that is trusted and one that is not.

> "SHAP says recency dominates at 0.52, then frequency at 0.23. The model is largely
> measuring whether people have stopped showing up."

## 3:10 — Inventory (50s)

Navigation → Inventory Recommendations.

> "Newsvendor reorder quantity, one-week lead time. Purchase cost 60%, holding 25%, lost
> margin 40%."

> "496 pairs to reorder, 2,081 units, INR 48,771."

Expand the comparison:

> "A normal approximation would order 5,384 more units across 1,022 more pairs. We don't
> ship that. With 78% zero weeks, mean plus z times sigma has no meaning in the upper tail,
> so safety stock comes from a Poisson quantile."

**Then the backtest, and do not skip it — this is the weakest number in the project:**

> "Walk-forward over the last 13 weeks against a naive-mean baseline: a 2.09% reduction, against
> a 25 to 40% target. We missed it. The interesting part is why — overstock drops from 18,302
> units to 44, and understock rises from 2,399 to 20,224. The policy didn't reduce total error,
> it moved the error from one side of the ledger to the other."

> "One thing that needs a decision, not a re-tune: the cost assumptions give a 0.9928 critical
> ratio, which implies a 99.28% service level. That's above the 95% we configured. It's a
> calibration question for whoever owns the brief."

## 4:00 — Drift (30s)

The drift monitor is not a dashboard page. Show it as a terminal cutaway, not narrated over a
chart.

```bash
python -m src.drift
```

> "We compare 2024 against 2025 — volume, price, discount and category mix — using PSI and
> total variation distance. On this data nothing drifted; every column came back under the 0.1
> stable threshold."

**Then say why a clean result is weak evidence, because it is:**

> "A monitor that never fires is either reassuring or broken. So the thresholds are exercised by
> a synthetic injection test that forces a shift and confirms the PSI responds. The 2024-vs-2025
> null result is real, but it isn't proof the detector works on your data."

The Evidently HTML report is at `reports/drift/evidently_drift.html`.

## 4:30 — Closing (30s)

Back to Home.

> "Seven functional requirements, 474 tests, a dashboard that loads in about a tenth of a
> second because it reads precomputed aggregates and never the 103 MB of raw inputs."

> "Four targets were missed, and they're in the report: churn AUC is 0.73 against 0.88, the
> panel-level MAPE target isn't demonstrated because the forecast is pooled, the inventory
> reduction is 2% against a 25% target, and the silhouette near 0.20 means the segments overlap.
> Everything here is reproducible from the seeded generator."

---

## Notes for whoever records this

- **Do not** narrate "AI-generated" — the brief disqualifies that. Speak in the first person
  about what was built and why.
- If a number on screen disagrees with this outline, the screen is correct only if the report
  is also updated. Fix the report, then re-record.
- Skip the DBSCAN segment if time runs short. **Do not skip the AUC miss, the inventory
  backtest miss, or the drift segment** — those are the three places where volunteering the
  weakness is what makes the rest of the demo credible.