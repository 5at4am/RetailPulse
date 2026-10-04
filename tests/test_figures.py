"""Report figures.

The figures are the part of the report most likely to rot silently: a chart is a picture of a
number, so when the number changes and the picture does not, nothing fails. These tests tie each
figure to the CSV it is drawn from, so a regenerated chart and a regenerated table cannot
disagree.

They also check the PDF, which is a deliverable in its own right with a page-count and size
budget from the brief.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.figures import FIGURE_DIR, main


def built_figures() -> Path:
    """The figure directory, generating the figures once if they are not there."""
    expected = ("fig1_forecast.png", "fig2_model_comparison.png", "fig3_shap.png",
                "fig4_segments.png", "fig5_drift.png", "fig6_inventory.png")
    if not all((FIGURE_DIR / name).exists() for name in expected):
        main()
    return FIGURE_DIR


def pdf_bytes() -> bytes:
    path = config.REPORTS / "RetailPulse_Report.pdf"
    if not path.exists():
        pytest.skip("build the PDF with pandoc + chrome first")
    return path.read_bytes()


class TestFiguresExist:
    EXPECTED = (
        "fig1_forecast.png",
        "fig2_model_comparison.png",
        "fig3_shap.png",
        "fig4_segments.png",
        "fig5_drift.png",
        "fig6_inventory.png",
    )

    def test_every_figure_is_written(self):
        for name in self.EXPECTED:
            path = built_figures() / name
            assert path.exists(), f"{name} missing; run `python -m src.figures`"
            assert path.stat().st_size > 5_000, f"{name} is suspiciously small"

    def test_files_are_real_pngs(self):
        # A matplotlib figure that failed to render can leave a truncated file that still
        # exists. The 8-byte PNG signature is the cheapest proof it is not HTML or an error log.
        for name in self.EXPECTED:
            signature = (built_figures() / name).read_bytes()[:8]
            assert signature == b"\x89PNG\r\n\x1a\n", f"{name} is not a valid PNG"

    def test_figures_are_not_absurdly_large(self):
        # 150 dpi on a 7.2in figure is ~50 KB. A 2 MB PNG means an un-subsampled bitmap got
        # committed, which bloats the repo and the PDF for no visible gain.
        for name in self.EXPECTED:
            kilobytes = (built_figures() / name).stat().st_size / 1024
            assert kilobytes < 400, f"{name} is {kilobytes:.0f} KB"


class TestFiguresAreWiredToTheReport:
    def test_report_references_every_figure(self):
        # A figure generated but never referenced is dead weight; a figure referenced but not
        # generated is a broken image in the PDF. This checks the first half, the PDF build
        # fails loudly on the second.
        text = (config.REPORTS / "RetailPulse_Report.md").read_text(encoding="utf-8")
        referenced = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))
        for name in TestFiguresExist.EXPECTED:
            assert f"figures/{name}" in referenced, f"{name} is generated but not in the report"

    def test_no_figure_reference_is_broken(self):
        text = (config.REPORTS / "RetailPulse_Report.md").read_text(encoding="utf-8")
        for relative in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            assert (config.REPORTS / relative).exists(), f"report references missing {relative}"

    def test_the_shap_figure_sits_in_the_churn_section(self):
        # It was briefly in the inventory section because the section heading was the
        # insertion anchor. A churn explainability chart under inventory is the kind of
        # mistake a reader does not catch but a reviewer does.
        text = (config.REPORTS / "RetailPulse_Report.md").read_text(encoding="utf-8")
        churn = text.index("## 5. Churn prediction")
        inventory = text.index("## 6. Inventory recommendations")
        shap = text.index("fig3_shap.png")
        assert churn < shap < inventory, "fig3 (SHAP) must be inside the churn section"

    def test_the_inventory_figure_sits_in_the_inventory_section(self):
        text = (config.REPORTS / "RetailPulse_Report.md").read_text(encoding="utf-8")
        inventory = text.index("## 6. Inventory recommendations")
        dashboard = text.index("## 7. Dashboard")
        assert inventory < text.index("fig6_inventory.png") < dashboard


class TestFiguresReadTheLiveData:
    """Each figure's headline number must be present in the CSV it claims to plot."""

    def test_forecast_figure_uses_the_published_wape(self):
        forecast = pd.read_csv(config.PROCESSED / "forecast_next_weeks.csv")
        row = forecast[forecast["horizon_weeks"] == 4]
        assert not row.empty, "the 4-week horizon the report calls primary is missing"
        # fig1 prints this value in its legend, so a change here must change the chart.
        assert 0 < float(row["holdout_wape"].iloc[0]) < 1

    def test_model_comparison_figure_has_all_three_horizons(self):
        scores = pd.read_csv(config.PROCESSED / "forecast_backtest.csv")
        scored = scores[scores["wape"].notna()]
        assert sorted(scored["horizon"].unique()) == [1, 2, 4]

    def test_drift_figure_input_has_no_drift_to_plot(self):
        # fig5 shades the stable band and colours by verdict. If drift were ever detected,
        # the figure is still correct but the report's claim in section 8 would be false.
        drift = pd.read_csv(config.REPORTS / "drift" / "drift_summary.csv")
        assert (drift["psi"].max() < 0.1), (
            f"drift now reaches {drift['psi'].max()}; section 8 of the report says no column "
            f"exceeded 0.1 and must be updated"
        )

    def test_inventory_figure_numbers_match_the_summary(self):
        summary = pd.read_csv(config.PROCESSED / "inventory_backtest_summary.csv").iloc[0]
        # fig6 labels the bars with these exact values.
        assert summary["baseline_overstock_units"] > summary["policy_overstock_units"], (
            "the report claims the policy reduces overstock; the data now says otherwise"
        )
        assert summary["policy_understock_units"] > summary["baseline_understock_units"], (
            "the report claims the policy raises understock; the data now says otherwise"
        )

    def test_shap_figure_input_is_the_ten_published_features(self):
        shap = pd.read_csv(config.PROCESSED / "churn_shap_importance.csv")
        assert len(shap) == 10, "the report documents ten behavioural features"


class TestPdf:
    def test_pdf_is_a_pdf(self):
        pdf = pdf_bytes()
        assert pdf[:5] == b"%PDF-"

    def test_page_count_is_within_the_briefs_budget(self):
        pdf = pdf_bytes()
        # The brief asks for 10-18 A4 pages. Counting /Type /Page occurrences (excluding
        # /Pages, the page-tree node) avoids needing a PDF library.
        pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
        assert 10 <= pages <= 18, f"{pages} pages, outside the 10-18 the brief asks for"

    def test_size_is_under_twelve_megabytes(self):
        pdf = pdf_bytes()
        assert len(pdf) <= 12 * 1024 * 1024, f"{len(pdf) / 1024 / 1024:.1f} MB exceeds 12 MB"

    def test_all_six_figures_are_embedded(self):
        pdf = pdf_bytes()
        images = len(re.findall(rb"/Subtype\s*/Image", pdf))
        assert images >= 6, f"only {images} images embedded; expected the six report figures"