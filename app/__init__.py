"""RetailPulse dashboard - F-06.

Run with:  streamlit run app/Home.py

Reads only the precomputed aggregates in `data/processed/`. Never touches the raw CSVs -
see `app/dashboard_data.py` for why.

Every page leads with what the number means, not just the number. Where a target was missed
the miss is shown next to the target, on the same page, rather than buried in a report
nobody opens during a five-minute demo.
"""