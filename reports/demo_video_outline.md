# Demo Video Outline — RetailPulse

Target: **4 to 5 minutes.** One screen recording of the running dashboard, plus a voiceover
track. Every number below is already in `reports/RetailPulse_Report.md`, so if one is wrong
in the video the report is the thing to fix first.

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

## 3:10 — Inventory (40s)

Navigation → Inventory Recommendations.

> "Newsvendor reorder quantity, one-week lead time. Purchase cost 60%, holding 25%, lost
> margin 40%."

> "496 pairs to reorder, 2,081 units, INR 48,771."

Expand the comparison:

> "A normal approximation would order 5,384 more units across 1,022 more pairs. We don't
> ship that. With 78% zero weeks, mean plus z times sigma has no meaning in the upper tail,
> so safety stock comes from a Poisson quantile."

## 3:50 — Closing (30s)

Back to Home.

> "Six functional requirements, 305 tests, a dashboard that loads in about a tenth of a
> second because it reads precomputed aggregates and never the 63 MB raw panel."

> "Three targets were missed and they're in the report: churn AUC, the panel-level MAPE
> definition, and the silhouette. Everything here is reproducible from the seeded generator."

---

## Notes for whoever records this

- **Do not** narrate "AI-generated" — the brief disqualifies that. Speak in the first person
  about what was built and why.
- If a number on screen disagrees with this outline, the screen is correct only if the report
  is also updated. Fix the report, then re-record.
- Skip the DBSCAN and Evidently sections if time runs short. Do not skip the AUC miss.
- The drift module is not a dashboard page; if asked, show `python -m src.drift` in a
  terminal separately rather than trying to narrate it in the walkthrough.