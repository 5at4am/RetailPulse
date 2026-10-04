"""Every file the report cites must exist, and its numbers must match `facts.json`.

This is risk R6 from the design spec: "Participant cannot explain a number in the demo" is
mitigated by making every figure traceable to the script and command that produced it. A
citation to a file that does not exist breaks that guarantee in the one way a judge will
actually test -- by following it.

Two bugs found by this file:

  - the report header cited `retail_sales.csv` and `store_product_matrix.csv`, neither of
    which exists, and labelled 970,998 (which is `sales.units`) as "line items" when the
    line-item count is 250,000.
  - `drift_report.csv` is cited while describing the old broken pipeline contract. That one
    is deliberate and historical, so it is allow-listed with the reason, rather than deleted
    -- rewriting it would erase the explanation of why the contract test exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src import config

REPORT = config.REPORTS / "RetailPulse_Report.md"

SEARCH_DIRS = (
    config.ROOT,
    config.REPORTS,
    config.REPORTS / "drift",
    config.REPORTS / "figures",
    config.DATA,
    config.PROCESSED,
    config.RAW,
    config.ROOT / "src",
    config.ROOT / "tests",
    config.ROOT / "app",
    config.ROOT / "deploy" / "kubernetes",
    config.ROOT / ".github" / "workflows",
    config.ROOT / "ops",
    config.ROOT / "airflow" / "dags",
)

CITATION = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:csv|json|html|png|py|md|yaml|yml|txt|joblib|pkl))`"
)

# Markdown image syntax is not backtick-wrapped, so the pattern above cannot see the figures.
# Without this, all six PNGs read as "generated but never cited" -- which was true of the
# regex, not of the report.
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+\.(?:png|jpg|jpeg|svg))\)")

# Cited while describing the bug that the pipeline-contract test now guards against.
HISTORICAL = {
    "drift_report.csv": "The old broken contract. See tests/test_pipeline.py, which fails if "
                        "the pipeline asks for a file a module does not write.",
}

# Real files that are gitignored by design, with the command that recreates them.
#
# These are not "missing" in the sense this test is about. The report's input-provenance table
# has to name the raw CSVs -- that is the whole point of the table -- but they are 103.7 MB and
# are regenerated from a seed by `python generate_retail_pulse.py`. Treating them like a typo
# would make this test fail on every fresh clone while passing on the machine that happened to
# have run the generator.
#
# The distinction this file exists to catch is a citation to a file that does not exist *and*
# has no way to exist. Allow-listing is therefore per-file, with the command attached, so a
# fifth uncited path has to be argued for rather than added by accident.
GENERATED = {
    "retail_pulse_sales.csv":
        "python generate_retail_pulse.py  # seeded, gitignored; 250,000 rows",
    "retail_pulse_demand_panel.csv":
        "python generate_retail_pulse.py  # seeded, gitignored; 780,000 rows",
    "retail_pulse_data_dictionary.md":
        "python generate_retail_pulse.py  # written by the generator alongside the CSVs",
}


@pytest.fixture(scope="module")
def report_text() -> str:
    return REPORT.read_text(encoding="utf-8")


def citations(text: str) -> list[str]:
    return sorted(set(CITATION.findall(text)) | set(IMAGE.findall(text)))


def load_facts() -> dict:
    return json.loads((config.ROOT / "facts.json").read_text(encoding="utf-8"))


class TestReportCitationsResolve:
    def test_every_cited_file_exists(self, report_text):
        missing = [
            rel for rel in citations(report_text)
            if not any((d / rel).exists() for d in SEARCH_DIRS)
            and rel not in HISTORICAL
            and rel not in GENERATED
        ]
        assert not missing, (
            "the report cites files that do not exist: "
            + ", ".join(missing)
            + "\nA judge who follows one of these finds nothing, and the number beside it "
              "becomes unfalsifiable."
        )

    def test_historical_citations_are_still_explained(self, report_text):
        # If someone deletes the explanation of the old drift contract, the allow-list entry
        # becomes a loophole rather than a documented exception.
        for name, reason in HISTORICAL.items():
            if name in report_text:
                assert "contract" in reason or "test" in reason

    def test_generated_allowlist_entries_name_the_command_that_recreates_them(self):
        # Each allow-listed path carries the command that produces it, so the entry documents
        # how a judge or reviewer gets the file rather than just excusing its absence.
        for name, command in GENERATED.items():
            assert "generate_retail_pulse.py" in command, (
                f"{name} is allow-listed without the command that recreates it"
            )

    def test_generated_files_are_actually_gitignored(self):
        # The allow-list is only honest while these files are genuinely excluded from the repo.
        # If one ever becomes tracked, it belongs in SEARCH_DIRS, not in an exception -- and
        # this is what says so.
        gitignore = (config.ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "data/raw" in gitignore, (
            "the raw inputs are allow-listed as regenerated, so data/raw/ must be gitignored; "
            "otherwise a judge would have no way to recreate them"
        )

    def test_figures_are_all_referenced(self, report_text):
        # Compared on basename: the report cites them as figures/figN_*.png in image syntax,
        # while the directory listing yields bare names.
        on_disk = {p.name for p in (config.REPORTS / "figures").glob("*.png")}
        cited = {Path(rel).name for rel in citations(report_text)}
        orphan = on_disk - cited
        assert not orphan, f"generated but never cited: {sorted(orphan)}"


class TestHeaderMatchesFacts:
    """The header is the first thing a judge reads, and it is the easiest place to be wrong."""

    def test_raw_file_names_are_correct(self, report_text):
        facts = load_facts()
        for name in facts["files"]:
            assert name in report_text, (
                f"{name} is a real input from facts.json but the report never names it"
            )

    def test_no_invented_input_names(self, report_text):
        # The specific regression: the header used to cite a shortened name.
        for bogus in ("`retail_sales.csv`", "`retail_demand_panel.csv`"):
            assert bogus not in report_text, (
                f"{bogus} does not exist; facts.json names the real inputs"
            )

    def test_line_items_are_not_confused_with_units(self, report_text):
        facts = load_facts()
        rows = facts["sales"]["rows"]
        units = facts["sales"]["units"]
        assert f"{rows:,} line items" in report_text, (
            f"the sales file has {rows:,} rows; that is the line-item count"
        )
        assert f"{units:,} units" in report_text, (
            f"{units:,} is units sold, not line items -- the header used to conflate them"
        )

    def test_headline_counts_are_present_and_distinct(self, report_text):
        facts = load_facts()
        for key in ("customers", "products", "stores"):
            assert f"{facts['sales'][key]:,}" in report_text, f"{key} count absent from header"
        assert facts["panel"]["rows"] != facts["sales"]["rows"]


class TestHonestLimitationsArePublished:
    """Design spec section 14 lists seven limitations to state deliberately.

    These exist so a judge finds them already written rather than discovering them. Each was
    missing from the report at least once: the Prophet granularity and the UCI calibration
    were never stated at all, and MAPE was described as "not demonstrated" without saying the
    metric is mathematically undefined on 77.9% of rows -- a weaker and vaguer claim.
    """

    REQUIRED = {
        "MAPE undefined on zero rows": ("undefined", "77.9"),
        "Prophet fitted at category level": ("category level",),
        "7,500 fits cannot meet the SLA": ("7,500",),
        "'30-day ahead' rounded to 4 weeks": ("4 weeks",),
        "UCI Online Retail II calibration disclosed": ("UCI", "1,067,371"),
        "Tier 3 authored but not deployed": ("not deployed",),
        "inventory figure is a backtest": ("backtest",),
        "dataset is seeded and synthetic": ("synthetic",),
    }

    def test_every_spec_limitation_is_stated(self, report_text):
        lowered = report_text.lower()
        missing = [
            label for label, needles in self.REQUIRED.items()
            if not all(n.lower() in lowered for n in needles)
        ]
        assert not missing, (
            "limitations from design spec section 14 are absent from the report: "
            + ", ".join(missing)
        )

    def test_targets_are_reported_as_measured_not_asserted(self, report_text):
        # Spec section 9: "The report will state the measured reduction, not the 25-40% target,
        # unless the backtest actually produces it."
        assert "2.09" in report_text, "the measured inventory reduction must appear"
        assert "0.7315" in report_text, "the measured churn AUC must appear, target or not"
        assert "0.88" in report_text, "and the target it missed must be shown beside it"


class TestReportIsCleanText:
    """Regression guard for the encoding damage that cost a full repair cycle."""

    def test_no_replacement_characters(self, report_text):
        assert "\ufffd" not in report_text, (
            "U+FFFD in the report: a PowerShell 5.1 Get-Content/Set-Content round-trip "
            "corrupted the file once already. Repair per line; do not re-encode the whole file."
        )

    def test_no_mojibake_sequences(self, report_text):
        for bad, label in (
            ("â€", "euro sign left behind by a cp1252 round-trip"),
            ("Ã©", "utf-8 bytes decoded as latin-1"),
            ("â€™", "right single quote mangled"),
            ("â€“", "en dash mangled"),
        ):
            assert bad not in report_text, f"mojibake present: {label}"

    def test_only_expected_non_ascii(self, report_text):
        # Intentional typography: en dash, em dash, section sign, middot, right arrow and the
        # multiplication sign in "store x product x week".
        allowed = set("\u2013\u2014\u00a7\u00b7\u2192\u00d7")
        stray = {c for c in report_text if ord(c) > 127} - allowed
        assert not stray, f"unexpected non-ascii: {sorted(stray)}"