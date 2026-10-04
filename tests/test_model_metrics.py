"""`model_metrics.json` must agree with the CSVs it consolidates.

The value of the file is that a judge can check one number in one place. That only holds if
it never disagrees with its sources, so these tests re-derive the headline values from the CSVs
rather than from the JSON.

Also covers the case the consolidation exists to prevent: a missing input must raise, not
produce a plausible-looking partial document.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config, model_metrics


@pytest.fixture
def document() -> dict:
    return model_metrics.build()


class TestHeadlineNumbersMatchTheirSources:
    def test_churn_comes_from_churn_metrics_csv(self, document):
        source = pd.read_csv(config.PROCESSED / "churn_metrics.csv").iloc[0]
        churn = document["churn"]
        assert churn["auc_roc"] == pytest.approx(source["auc"])
        assert churn["pr_auc"] == pytest.approx(source["pr_auc"])
        assert churn["precision_at_top_20pct"] == pytest.approx(source["precision_at_top"])
        assert churn["f1_at_operating_threshold"] == pytest.approx(
            source["f1_at_operating_threshold"])

    def test_confusion_matrix_totals_match_the_row_count(self, document):
        churn = document["churn"]
        matrix = churn["confusion_matrix"]
        total = sum(matrix.values())
        assert total == churn["test_rows"], (
            "the confusion matrix must partition the test set; if these differ, a row was "
            "dropped between scoring and reporting"
        )

    def test_inventory_comes_from_its_summary_csv(self, document):
        source = pd.read_csv(config.PROCESSED / "inventory_backtest_summary.csv").iloc[0]
        inventory = document["inventory"]
        assert inventory["error_reduction_pct"] == pytest.approx(source["error_reduction_pct"])
        assert inventory["meets_target"] == bool(source["meets_25_40_target"])

    def test_forecasting_uses_the_model_the_selection_actually_picked(self, document):
        selection = pd.read_csv(config.PROCESSED / "forecast_model_selection.csv")
        for horizon, chosen in zip(selection["horizon"], selection["model"]):
            block = document["forecasting"][str(int(horizon))]
            assert block["selected_model"] == chosen, (
                "the consolidated file must report the model selection chose, not the "
                "best-looking row"
            )

    def test_every_forecast_horizon_carries_the_full_metric_set(self, document):
        # Spec section 6.2 names six metrics; any horizon missing one is a silent gap.
        required = {"wape", "mape_nonzero", "mae_units", "rmse_units", "mase", "bias_units"}
        for horizon, block in document["forecasting"].items():
            assert required <= set(block), f"{horizon}-week missing {required - set(block)}"

    def test_losing_candidates_are_kept(self, document):
        # The report's most defensible claim about forecasting is that the LSTM lost. That is
        # only checkable if the candidates are in the file.
        for block in document["forecasting"].values():
            assert len(block["candidates"]) >= 2, "only the winner was recorded"

    def test_segment_counts_add_up(self, document):
        segmentation = document["segmentation"]
        assert sum(s["customers"] for s in segmentation["segments"]) == segmentation["n_customers"]


class TestTargetsAreRecordedBesideResults:
    """A metric without its target next to it is the thing that makes a miss look like a pass."""

    def test_churn_records_both_targets_and_both_outcomes(self, document):
        churn = document["churn"]
        assert churn["meets_auc_target"] is False, "AUC 0.7315 is a miss; it must read as one"
        assert churn["meets_precision_target"] is True
        assert churn["target_auc_roc"] == 0.88

    def test_inventory_records_the_range_not_just_the_result(self, document):
        inventory = document["inventory"]
        assert inventory["target_error_reduction_pct"] == [25.0, 40.0]
        assert inventory["meets_target"] is False


class TestFailsLoudly:
    def test_missing_input_raises_instead_of_writing_a_partial_file(self, tmp_path, monkeypatch):
        import src.config as config_module

        real_processed = config_module.PROCESSED
        empty = tmp_path / "processed"
        empty.mkdir()
        monkeypatch.setattr(config_module, "PROCESSED", empty)
        monkeypatch.setattr(model_metrics.config, "PROCESSED", empty)

        with pytest.raises(FileNotFoundError, match="model_metrics.json cannot be built"):
            model_metrics.build()
        assert real_processed.exists()

    def test_write_produces_valid_utf8_json(self, tmp_path):
        target = model_metrics.write(path=tmp_path / "m.json")
        reloaded = json.loads(target.read_text(encoding="utf-8"))
        assert reloaded["project"] == "RetailPulse"
        assert target.read_text(encoding="utf-8").endswith("\n")


class TestCommittedArtefactIsCurrent:
    def test_committed_model_metrics_matches_a_fresh_build(self, document):
        path = config.PROCESSED / "model_metrics.json"
        if not path.exists():
            pytest.skip("model_metrics.json not generated yet")
        committed = json.loads(path.read_text(encoding="utf-8"))
        # Compare the measured blocks only: `written_by`/`note` are prose and are compared
        # by the drift-style tests. A stale file here means the CSVs moved under it.
        for section in ("forecasting", "churn", "inventory", "segmentation"):
            assert committed[section] == document[section], (
                f"committed model_metrics.json disagrees with the CSVs in section {section!r}; "
                f"re-run `python -m src.model_metrics`"
            )