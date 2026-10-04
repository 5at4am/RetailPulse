"""The project root README is the deliverable. This is the page-navigation wrapper.

`streamlit run app/Home.py` runs the app. Streamlit's own multipage detection picks up the
sibling `*.py` files, but that depends on the run directory, so the pages are listed
explicitly in `.streamlit/config.toml` as well. This file exists so `app/` is an importable
package and the shared data helpers in `dashboard_data.py` resolve predictably.
"""

PAGES = [
    ("Home", "app/Home.py", "Overview, KPIs and the honest headline findings"),
    ("Demand Forecasting", "app/Demand_Forecasting.py", "F-03 backtest, selection, forecast"),
    ("Customer Segments", "app/Customer_Segments.py", "F-02 K-Means segmentation"),
    ("Churn Risk", "app/Churn_Risk.py", "F-04 XGBoost churn scoring"),
    ("Inventory Recommendations", "app/Inventory_Recommendations.py", "F-05 reorder plan"),
]

if __name__ == "__main__":
    for name, path, description in PAGES:
        print(f"{name:<28} {path:<42} {description}")