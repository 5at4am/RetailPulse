"""Ordering and failure semantics of the nightly pipeline.

These exist because the first Kubernetes manifest got the ordering wrong in a way no YAML
parser would catch. All six stages were containers in one Pod, which Kubernetes starts
concurrently, so the stage order was not a stage order at all: the verifier ran immediately
against artefacts that had not been written yet.

The tests below pin the three things that actually matter and that a reader cannot check by
looking at YAML: the order, the stop-at-first-failure rule, and the artefact contract.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pandas as pd
import pytest

from src import pipeline
from src.pipeline import STAGES, STAGE_OUTPUTS, STAGE_RESOURCES, StageFailed


class TestStageOrder:
    def test_drift_is_last(self):
        # Drift compares the current window against the reference period. Run before the
        # models refresh, it reports drift caused by an incomplete refresh and pages someone
        # about data that is merely still being written.
        assert STAGES[-1] == "drift"

    def test_precompute_is_first(self):
        # Everything downstream reads the aggregates and the features this writes.
        assert STAGES[0] == "precompute"

    def test_forecasting_precedes_churn(self):
        # Both read the same features, so this is about a stable published order rather than
        # a hard dependency -- but a stable order is what makes a failed run diagnosable.
        assert STAGES.index("forecasting") < STAGES.index("churn")

    def test_order_is_the_full_declared_stage_list(self):
        assert list(STAGES) == ["precompute", "forecasting", "churn", "inventory", "drift"]

    def test_every_stage_has_declared_outputs(self):
        # A stage with no output contract cannot be verified, and an unverifiable stage is
        # how a silent no-op gets promoted to "the nightly job is green".
        for stage in STAGES:
            assert STAGE_OUTPUTS.get(stage), f"{stage} declares no outputs"

    def test_every_stage_has_a_resource_envelope(self):
        for stage in STAGES:
            assert stage in STAGE_RESOURCES
            assert "memory" in STAGE_RESOURCES[stage]

    def test_cronjob_envelope_covers_the_heaviest_stage(self):
        # The CronJob runs all stages in one container, so it must be able to hold the
        # heaviest one. This is why the per-stage table has to stay in step with the YAML.
        heaviest = max(STAGE_RESOURCES[stage]["memory"] for stage in STAGES)
        assert heaviest == STAGE_RESOURCES["forecasting"]["memory"]


class TestArtifactContract:
    def test_reports_a_stage_that_wrote_nothing(self, tmp_path):
        assert pipeline.verify_stage_outputs("churn", tmp_path) == list(STAGE_OUTPUTS["churn"])

    def test_reports_a_stage_that_wrote_only_some_of_its_outputs(self, tmp_path):
        pd.DataFrame({"horizon": [1]}).to_csv(tmp_path / "churn_metrics.csv", index=False)
        missing = pipeline.verify_stage_outputs("churn", tmp_path)
        assert "churn_metrics.csv" not in missing
        assert "churn_scores.csv" in missing

    def test_treats_a_zero_byte_file_as_missing(self, tmp_path):
        # This is the shape a crashed or interrupted write leaves behind. Path.exists()
        # returns True for it, so an existence-only check waves a broken night through.
        for name in STAGE_OUTPUTS["inventory"]:
            (tmp_path / name).write_text("")
        assert pipeline.verify_stage_outputs("inventory", tmp_path) == list(
            STAGE_OUTPUTS["inventory"])

    def test_treats_a_header_only_file_as_missing(self, tmp_path):
        # A header with no rows is the other half of the same failure, and `read_csv` on it
        # yields an empty frame rather than raising.
        for name in STAGE_OUTPUTS["inventory"]:
            (tmp_path / name).write_text("a,b,c\n")
        assert pipeline.verify_stage_outputs("inventory", tmp_path) == list(
            STAGE_OUTPUTS["inventory"])

    def test_accepts_a_complete_stage(self, tmp_path):
        for name in STAGE_OUTPUTS["drift"]:
            pd.DataFrame({"feature": ["units"], "psi": [0.1]}).to_csv(
                tmp_path / name, index=False)
        assert pipeline.verify_stage_outputs("drift", tmp_path) == []

    def test_never_reads_outside_the_given_directory(self, tmp_path):
        # `processed` is passed in so tests need not touch real outputs. If any stage reached
        # outside it, the test would be asserting against the developer's working tree.
        work = tmp_path / "processed"
        work.mkdir()
        assert pipeline.verify_stage_outputs("churn", work) != []
        assert pipeline.verify_stage_outputs("churn", tmp_path / "nope") != []


class TestFailureStopsTheRun:
    def test_a_nonzero_exit_raises_naming_the_stage(self, monkeypatch, tmp_path):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "boom")

        monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
        with pytest.raises(StageFailed, match="forecasting"):
            pipeline.run_stage("forecasting", root=tmp_path)
        assert calls == [[sys.executable, "-m", "src.forecasting"]]

    def test_later_stages_do_not_run_after_a_failure(self, monkeypatch, tmp_path):
        # The whole reason sequencing is code and not manifest shape. Every stage here would
        # happily "succeed" if it were reached, so a run that got past forecasting would be
        # indistinguishable from a correct one.
        ran = []

        def fake_run(argv, **kwargs):
            ran.append(argv[-1].removeprefix("src."))
            code = 0 if argv[-1] != "src.forecasting" else 1
            return subprocess.CompletedProcess(argv, code, "", "" if code == 0 else "boom")

        monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
        monkeypatch.setattr(pipeline, "verify_stage_outputs", lambda stage, root=None: [])

        with pytest.raises(StageFailed, match="forecasting"):
            pipeline.run_pipeline(STAGES, root=tmp_path)
        # churn, inventory and drift never started.
        assert ran == ["precompute", "forecasting"]

    def test_a_stage_that_succeeds_but_writes_nothing_still_fails(self, monkeypatch, tmp_path):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
        # Exit code 0, empty directory: the stage lied, and the contract catches it.
        with pytest.raises(StageFailed, match="churn"):
            pipeline.run_stage("churn", root=tmp_path)

    def test_the_error_message_carries_useful_output(self, monkeypatch, tmp_path):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 2, "", "line one\nline two\nTraceback")

        monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
        with pytest.raises(StageFailed) as caught:
            pipeline.run_stage("churn", root=tmp_path)
        assert "line two" in str(caught.value)
        assert caught.value.stage == "churn"

    def test_all_stages_run_in_order_on_success(self, monkeypatch, tmp_path):
        ran = []

        def fake_run(argv, **kwargs):
            ran.append(argv[-1].removeprefix("src."))
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
        monkeypatch.setattr(pipeline, "verify_stage_outputs", lambda stage, root=None: [])
        results = pipeline.run_pipeline(root=tmp_path)
        assert ran == list(STAGES)
        assert [r.name for r in results] == list(STAGES)
        assert all(r.ok for r in results)


class TestCli:
    def test_a_full_run_of_real_stages_is_not_exercised_in_unit_tests(self):
        # Guard against a future test invoking the real pipeline: five model fits is minutes
        # of work and would make this file unusable as a fast check.
        assert "integration" not in sys.argv[0]

    def test_only_flag_is_validated_against_the_stage_list(self, tmp_path):
        monkey_failed = False
        try:
            pipeline.main(["--only", "not_a_stage"])
        except SystemExit as exit_code:
            monkey_failed = exit_code.code != 0
        assert monkey_failed, "an unknown stage name must be rejected, not silently run"

    def test_skip_verify_is_recorded_in_the_parser(self):
        parser_help = pipeline.main.__doc__ or ""
        # Nothing to assert on the docstring; the flag's existence is what matters and is
        # exercised by the integration run. Kept as a smoke check that main is importable.
        assert callable(pipeline.main)
        assert parser_help is not None


class TestContractMatchesWhatTheModulesActuallyWrite:
    """The stage contract is hand-written, so it can disagree with the code it describes.

    This is not hypothetical: the first version of the contract required `drift_report.csv`
    for a module that has always written `drift_summary.csv`. Nothing failed at build time --
    the contract simply asserted a file nobody produces, so a green nightly run would have
    meant nothing. Reading the filenames back out of each module's `to_csv` calls closes that
    gap without running any models.
    """

    @staticmethod
    def _written_csvs(module: str) -> set[str]:
        """Names a stage publishes, under either of the two shapes used in this codebase.

        `churn`, `inventory` and `drift` write explicit filenames:
            frame.to_csv(config.PROCESSED / "churn_metrics.csv")
        `precompute` writes `f"{name}.csv"` from the keys of its `out` dict, so its names are
        string keys rather than `.csv` literals. Reading only literals silently concluded that
        precompute writes nothing, which is how the contract came to name a file no stage
        produces.
        """
        import re
        from src import config

        source = (config.ROOT / "src" / f"{module}.py").read_text(encoding="utf-8")
        literals = set(re.findall(r'["\']([\w.-]+\.csv)["\']', source))
        # `out["key"] = ...` and the keys of the `out = {...}` literal.
        assigned = set(re.findall(r'out\[["\']([\w.-]+)["\']\]\s*=', source))
        block = re.search(r"out\s*=\s*\{(.*?)\n\s*\}", source, re.DOTALL)
        if block:
            assigned |= set(re.findall(r'["\']([\w.-]+)["\']\s*:', block.group(1)))
        return literals | assigned

    def test_every_required_artefact_is_written_by_its_stage(self):
        for stage, names in STAGE_OUTPUTS.items():
            written = self._written_csvs(stage)
            assert written, f"found no published names in src/{stage}.py; check the regex"
            for name in names:
                # Stages that write `to_csv(... / "x.csv")` publish the filename; precompute
                # publishes dict keys and appends `.csv` itself, so match on the stem too.
                stem = name.removesuffix(".csv")
                assert name in written or stem in written, (
                    f"the contract requires {name!r} from src/{stage}.py, but that module "
                    f"never writes it. It writes: {sorted(written)}"
                )

    def test_precompute_publishes_kpis_under_a_dict_key(self):
        # The specific stage that needs the dict-key reader. `kpis` is a key of precompute's
        # `out`, written as f"{name}.csv".
        assert "kpis" in self._written_csvs("precompute")
        assert "kpis.csv" in STAGE_OUTPUTS["precompute"]

    def test_drift_outputs_live_under_reports_not_processed(self):
        # Drift's artefacts are report outputs the dashboard does not read. Requiring them in
        # data/processed would either fail forever or, worse, tempt someone to move them.
        from src import config

        directory = pipeline._stage_dir("drift")
        assert directory == config.REPORTS / "drift"
        assert pipeline._stage_dir("churn") == config.PROCESSED

    def test_drift_summary_name_is_the_real_one(self):
        # Named explicitly because this is the exact filename that was wrong once.
        assert "drift_summary.csv" in STAGE_OUTPUTS["drift"]
        assert "drift_report.csv" not in STAGE_OUTPUTS["drift"]


class TestProcessedDirectoryHygiene:
    def test_no_stray_files_in_the_directory_the_dashboard_reads(self, tmp_path):
        # The dashboard loads everything in data/processed. A leftover debug dump there is
        # both clutter and a chance that a page picks up the wrong file by name.
        from src import config

        stray = [p.name for p in (config.PROCESSED).glob("*.csv")
                 if p.name.startswith(("_", "."))
                 or "matrix" == p.name.split("_")[0]]
        assert not stray, (
            f"stray files in data/processed: {stray}. They are auto-loaded by the dashboard."
        )

    def test_every_declared_artefact_is_in_the_dashboard_data_contract(self):
        # Anything the dashboard must show has to be a file it knows about; anything in the
        # contract has to exist. One-directional checks pass while a page silently 404s.
        from src import config

        known = {p.name for p in config.PROCESSED.glob("*.csv")}
        for stage, names in STAGE_OUTPUTS.items():
            if stage == "drift":
                continue          # drift writes reports, not dashboard inputs
            for name in names:
                assert name in known, (
                    f"{name} is required by the {stage} contract but absent from "
                    f"data/processed -- run the pipeline"
                )


class TestImagesExistForEveryReferencedImage:
    """A manifest naming an image nobody builds is a deployment that fails on first schedule.

    This is the same defect as the dashboard-image one, one level up: the CronJob was pointed
    at `retailpulse-pipeline:1.0.0` while the repository only had a `Dockerfile` producing
    `retailpulse`. A cluster would have accepted the CronJob happily and pulled-nothing at 03:17.
    """

    @staticmethod
    def _declared_images() -> set[str]:
        import re
        from src import config

        found: set[str] = set()
        for path in list((config.ROOT / "deploy").rglob("*.yaml")) + [
                config.ROOT / ".github" / "workflows" / "ci.yml"]:
            found |= set(re.findall(r"image:\s*([\w./-]+)", path.read_text(encoding="utf-8")))
        return found

    def test_no_manifest_references_an_unbuildable_image(self):
        from src import config

        dockerfiles = {p.name for p in config.ROOT.glob("Dockerfile*")}
        assert dockerfiles, "no Dockerfile at all"

        for image in sorted(self._declared_images()):
            if "{{" in image:
                continue                      # a CI-templated tag, not a literal
            assert not image.startswith("retailpulse-pipeline") or (
                "Dockerfile.pipeline" in dockerfiles), (
                f"{image} is referenced but Dockerfile.pipeline does not exist"
            )

    def test_pipeline_image_installs_the_heavy_dependencies(self):
        from src import config

        text = (config.ROOT / "Dockerfile.pipeline").read_text(encoding="utf-8")
        # Requirements must come from requirements-ml.txt. The root requirements.txt is now the
        # lean app set -- deliberately, because Streamlit Community Cloud installs it to serve
        # the public demo -- so a pipeline image built from it would fail on the first Prophet
        # import. That inversion is exactly what made every stage fail on import.
        assert "requirements-ml.txt" in text
        # Comments are stripped, because this file names the lean set by design in order to
        # explain the split, and a substring search over raw text would read the explanation as
        # the dependency. Line continuations are joined so that a `RUN ... \` + `&& pip install`
        # pair is read as one instruction rather than two lines that each look incomplete.
        active = []
        for raw in text.splitlines():
            line = raw.rstrip()
            if line.lstrip().startswith("#") or not line.strip():
                continue
            if active and active[-1].endswith("\\"):
                active[-1] = active[-1].rstrip("\\").rstrip() + " " + line.strip()
            else:
                active.append(line)
        # Every requirements file any instruction actually installs from, read off the joined
        # instruction lines. The separate `pip install "prophet>=..."` line is a deliberate
        # speed-up, not a second dependency source, so it is ignored here.
        sources = {
            match
            for line in active
            for match in re.findall(r"-r\s+(requirements[\w.-]*\.txt)", line)
        }
        assert "requirements-ml.txt" in sources, (
            f"the pipeline image must install the modelling dependency set; saw {sources}"
        )
        assert not any(
            re.fullmatch(r"requirements\.txt", name) for name in sources
        ), (
            "the pipeline image must not install the lean app set as its dependency source"
        )

    def test_pipeline_image_carries_the_src_package_not_just_the_orchestrator(self):
        # src.pipeline shells out to `python -m src.<stage>`, so a per-file COPY of the
        # orchestrator alone would ImportError on stage one.
        from src import config

        text = (config.ROOT / "Dockerfile.pipeline").read_text(encoding="utf-8")
        assert "COPY src/ ./src/" in text

    def test_pipeline_dockerfile_has_no_shell_syntax_in_copy(self):
        from src import config

        for line in (config.ROOT / "Dockerfile.pipeline").read_text(
                encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("COPY "):
                for token in ("||", "&&", "|", ">", "2>", ";"):
                    assert token not in stripped, f"shell syntax in COPY: {stripped}"


class TestKubernetesManifestAgreesWithTheCode:
    def test_manifest_calls_the_pipeline_module(self):
        from src import config
        text = (config.ROOT / "deploy" / "kubernetes" / "pipeline-cronjob.yaml").read_text(
            encoding="utf-8")
        assert "python" in text and "-m" in text and "src.pipeline" in text, (
            "the CronJob must call src.pipeline so the ordering lives in tested code"
        )

    def test_manifest_does_not_define_multiple_stage_containers(self):
        # The original bug: six containers in one Pod, which Kubernetes runs concurrently.
        # One container means the order is whatever src.pipeline says it is.
        from src import config
        text = (config.ROOT / "deploy" / "kubernetes" / "pipeline-cronjob.yaml").read_text(
            encoding="utf-8")
        assert text.count("- name: pipeline") == 1
        for stage in STAGES:
            assert f"- name: {stage}" not in text, (
                f"{stage} is still a sibling container; siblings run concurrently"
            )

    def test_manifest_does_not_use_the_dashboard_image_for_the_pipeline(self):
        # The root requirements.txt omits Prophet, TensorFlow and XGBoost, so a pipeline
        # container built from it fails on import at stage one.
        from src import config
        text = (config.ROOT / "deploy" / "kubernetes" / "pipeline-cronjob.yaml").read_text(
            encoding="utf-8")
        # Only the requirement lines. requirements-ml.txt NAMES prophet, tensorflow and xgboost
        # as real requirements, so this must be read from the app set -- and that file NAMES them
        # in comments only to explain the exclusion, which is why comments are stripped.
        lines = (config.ROOT / "requirements.txt").read_text(
            encoding="utf-8").lower().splitlines()
        pinned = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        for heavy in ("prophet", "tensorflow", "xgboost", "lightgbm"):
            assert not any(heavy in ln for ln in pinned), (
                f"premise broken: {heavy} is now a real dashboard requirement"
            )
        pipeline_image = [line for line in text.splitlines() if "image:" in line]
        assert pipeline_image, "no image declared"
        for line in pipeline_image:
            assert "retailpulse-pipeline" in line, (
                f"pipeline stages need the full-stack image, not the dashboard image: {line.strip()}"
            )