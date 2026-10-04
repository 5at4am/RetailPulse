"""Static checks on the Tier 3 infrastructure artefacts.

The spec asks for these files to be "authored and statically validated". That claim is only
worth something if something checks it, because a manifest with a typo in an image tag parses
fine and deploys nothing. So these tests read the files and assert the properties that would
otherwise only fail at deploy time.

They deliberately do not require `kubectl`, `docker`, or Airflow. Airflow in particular is not
installed locally, so the DAG is validated by parsing its source and by checking the fallback
path returns None cleanly rather than by building a DAG object. If Airflow ever is installed,
`test_dag_builds_when_airflow_is_available` exercises the real construction too.

What these tests cannot cover, and the report says so: no image was built, no cluster was
contacted, no DAG was executed by a scheduler.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "deploy" / "kubernetes"
DAG = ROOT / "airflow" / "dags" / "retailpulse_nightly.py"
DOCKERFILE = ROOT / "Dockerfile"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def load_documents(path: pathlib.Path) -> list[dict]:
    """Parse a multi-document YAML file, dropping the empty trailing document."""
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def by_kind(documents: list[dict], kind: str) -> list[dict]:
    return [d for d in documents if d.get("kind") == kind]


# --------------------------------------------------------------------------- kubernetes
class TestKubernetesManifests:
    def test_dashboard_manifest_is_valid_multi_document_yaml(self):
        documents = load_documents(K8S / "dashboard.yaml")
        kinds = [d["kind"] for d in documents]
        # A file that only parses as one document is a common failure: the second object's
        # keys get silently swallowed and the cluster sees half a Deployment.
        assert kinds == [
            "Namespace",
            "ServiceAccount",
            "Deployment",
            "Service",
            "PodDisruptionBudget",
            "HorizontalPodAutoscaler",
        ]

    def test_pipeline_manifest_is_a_cronjob(self):
        documents = load_documents(K8S / "pipeline-cronjob.yaml")
        jobs = by_kind(documents, "CronJob")
        assert len(jobs) == 1
        job = jobs[0]
        assert job["spec"]["schedule"], "an empty schedule never fires"
        assert job["spec"]["concurrencyPolicy"] == "Forbid", (
            "two concurrent runs write the same CSVs and race"
        )

    def test_cronjob_has_one_container_because_order_is_not_a_container_property(self):
        # The original manifest listed six stages as six containers in one Pod and these
        # tests asserted their order. That assertion was backwards: Kubernetes starts every
        # container in a Pod concurrently, so the list order was cosmetic and `verify` ran at
        # t=0. The order now lives in src.pipeline.STAGES, tested in tests/test_pipeline.py.
        containers = load_documents(K8S / "pipeline-cronjob.yaml")[0]["spec"][
            "jobTemplate"
        ]["spec"]["template"]["spec"]["containers"]
        assert [c["name"] for c in containers] == ["pipeline"], (
            "multiple containers means concurrent execution, not a sequence"
        )

    def test_cronjob_delegates_sequencing_to_the_pipeline_module(self):
        from src.pipeline import STAGES, STAGE_OUTPUTS

        job = load_documents(K8S / "pipeline-cronjob.yaml")[0]["spec"]["jobTemplate"]["spec"]
        container = job["template"]["spec"]["containers"][0]
        assert container["command"] == ["python", "-m", "src.pipeline"]
        # The order the manifest used to fake, now asserted where it can actually hold.
        assert STAGES.index("drift") > STAGES.index("inventory") > STAGES.index("forecasting")
        for stage in STAGES:
            assert STAGE_OUTPUTS[stage], f"{stage} has no artefact contract to verify"

    def test_verify_fails_loudly_when_an_artefact_is_absent(self, tmp_path, monkeypatch):
        # Same guarantee the old `set -eu` shell loop gave, now enforced by an artefact
        # contract so it is testable without a container. A stage that exits 0 without
        # writing its outputs must fail the run.
        from src import pipeline

        # Empty directory: every stage promises files it has not written.
        assert pipeline.verify_stage_outputs("churn", tmp_path) == list(
            pipeline.STAGE_OUTPUTS["churn"])

        # `python` is an interpreter path, not a callable, so patch the actual subprocess.
        monkeypatch.setattr(
            pipeline.subprocess, "run",
            lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", ""))
        with pytest.raises(pipeline.StageFailed):
            pipeline.run_stage("churn", root=tmp_path)

    def test_containers_run_as_non_root(self):
        documents = load_documents(K8S / "dashboard.yaml")
        deployment = by_kind(documents, "Deployment")[0]
        pod = deployment["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        container = pod["containers"][0]
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
        # readOnlyRootFilesystem must stay False: Streamlit writes its config under $HOME.
        assert container["securityContext"]["readOnlyRootFilesystem"] is False

    def test_liveness_probe_uses_the_real_health_endpoint(self):
        deployment = by_kind(load_documents(K8S / "dashboard.yaml"), "Deployment")[0]
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        for probe in ("livenessProbe", "startupProbe"):
            assert container[probe]["httpGet"]["path"] == "/_stcore/health"
        # The startup probe has to be more forgiving than the liveness one. Cold start reads
        # several aggregates; if liveness runs from second one it kills a container that was
        # about to become healthy.
        assert (
            container["startupProbe"]["failureThreshold"]
            > container["livenessProbe"]["failureThreshold"]
        ), "the startup probe must tolerate a slow cold start that liveness would kill"

    def test_deployment_and_hpa_target_the_same_name(self):
        documents = load_documents(K8S / "dashboard.yaml")
        deployment = by_kind(documents, "Deployment")[0]
        hpa = by_kind(documents, "HorizontalPodAutoscaler")[0]
        assert hpa["spec"]["scaleTargetRef"]["name"] == deployment["metadata"]["name"]
        # minReplicas must not exceed the HPA floor, or the HPA immediately scales down on
        # boot and the demo URL cold-starts for no reason.
        assert hpa["spec"]["minReplicas"] >= deployment["spec"]["replicas"]

    def test_service_selects_the_deployment_pods(self):
        documents = load_documents(K8S / "dashboard.yaml")
        deployment = by_kind(documents, "Deployment")[0]
        service = by_kind(documents, "Service")[0]
        selector = service["spec"]["selector"]
        assert selector["app.kubernetes.io/component"] == "dashboard"
        # Every selector label has to be present in the pod template, or the Service matches
        # nothing and returns connection refused.
        labels = deployment["spec"]["template"]["metadata"]["labels"]
        for key, value in selector.items():
            assert labels.get(key) == value, f"selector {key}={value} matches no pods"

    def test_no_privilege_escalation_anywhere_in_the_pipeline(self):
        cronjob = load_documents(K8S / "pipeline-cronjob.yaml")[0]
        pod = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        assert pod["automountServiceAccountToken"] is False
        assert pod["securityContext"]["runAsNonRoot"] is True


# ------------------------------------------------------------------------------ airflow
class TestAirflowDag:
    def test_dag_file_is_syntactically_valid(self):
        ast.parse(DAG.read_text(encoding="utf-8"))

    def test_dag_imports_without_airflow_installed(self):
        # Airflow is not a local dependency, so the module must degrade rather than explode.
        # An unguarded `from airflow import DAG` at module scope would break every test run
        # and any local `python -c "import ..."`.
        source = DAG.read_text(encoding="utf-8")
        assert "try:" in source and "except ImportError" in source

    def test_fallback_returns_none_and_build_dag_is_callable(self):
        namespace: dict = {}
        exec(  # noqa: S102 - exercising the guard, not importing project code
            compile(DAG.read_text(encoding="utf-8"), str(DAG), "exec"), namespace
        )
        build_dag = namespace["build_dag"]
        if namespace.get("AIRFLOW_AVAILABLE"):
            dag = build_dag()
            assert dag is not None
            assert dag.dag_id == "retailpulse_nightly"
        else:
            assert build_dag() is None, (
                "without Airflow, build_dag must return None rather than raise"
            )

    def test_stage_commands_are_real_modules(self):
        # Checked against the IMPORTED list, not against `"python -m src.x"` string literals
        # in the source. The DAG used to restate the commands itself; it now builds them from
        # src.pipeline.STAGES with an f-string, so a regex over the file finds nothing and
        # would have "passed" on an empty set if the assertion were `not commands`.
        namespace = {"__name__": "retailpulse_dag_probe", "__file__": str(DAG)}
        # Single-namespace exec. With separate globals/locals the module-level assignments land
        # in locals while the functions defined there resolve names in globals, so
        # build_dag() raises NameError on AIRFLOW_AVAILABLE.
        exec(  # noqa: S102 - exercising the import path, not running the pipeline
            compile(DAG.read_text(encoding="utf-8"), str(DAG), "exec"), namespace)
        stages = namespace["STAGES"]
        assert [name for name, _, _ in stages] == [
            "precompute", "forecasting", "churn", "inventory", "drift",
        ]
        for name, command, _ in stages:
            assert command == f"python -m src.{name}"
            path = ROOT / "src" / f"{name}.py"
            assert path.exists(), f"{name} is scheduled but src/{name}.py does not exist"

    def test_dag_does_not_restate_the_stage_list(self):
        # The duplication is what let the DAG require output filenames the CronJob did not.
        source = DAG.read_text(encoding="utf-8")
        assert "from src.pipeline import STAGES" in source
        # No hand-maintained output filename list left behind.
        assert "drift_summary.csv" not in source or "reports/drift/drift_summary.csv" in source

    def test_drift_check_does_not_fail_the_refresh(self):
        source = DAG.read_text(encoding="utf-8")
        drift_block = source.split('task_id="check_drift"')[1]
        # Drift is a finding. Failing the refresh on it would leave yesterday's data in
        # place, which is strictly worse than publishing fresh data with a warning.
        assert "SystemExit(0)" in drift_block


# ------------------------------------------------------------------------------ docker
def active_lines(text: str) -> str:
    """Strip comments, so a file that *talks about* a package is not read as depending on it.

    Both the Dockerfile and requirements-ml.txt explain why Prophet is excluded from the
    runtime image, which means the word appears in them by design. Testing the raw text
    would assert the opposite of the intent.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


class TestDockerfile:
    @pytest.fixture()
    def dockerfile(self) -> str:
        return DOCKERFILE.read_text(encoding="utf-8")

    def test_starts_from_a_pinned_python_base(self, dockerfile):
        assert re.search(r"^FROM python:\d+\.\d+-slim", dockerfile, re.MULTILINE)

    def test_has_a_healthcheck_on_the_streamlit_endpoint(self, dockerfile):
        assert "HEALTHCHECK" in dockerfile
        assert "/_stcore/health" in dockerfile

    def test_runs_as_a_non_root_user(self, dockerfile):
        assert "useradd" in dockerfile
        assert re.search(r"^USER (?!root)\S+", dockerfile, re.MULTILINE), (
            "the image must not stay root"
        )

    def test_does_not_install_the_ml_stack(self, dockerfile):
        # The failure this prevents: a 2 GB image that takes 40s to pull and OOMs on the
        # free tier, for a dashboard that plots four committed CSVs.
        #
        # It installs the root requirements.txt, and that file is lean. Both halves matter:
        # swapping the Dockerfile to requirements-ml.txt, or putting Prophet back in the root
        # file, produces the same bad image by a different route. Streamlit Community Cloud
        # reads the root file too, so "lean at the root" is what keeps the public demo from
        # paying a quarter-gigabyte install on every cold start.
        assert "-r requirements.txt" in active_lines(dockerfile), (
            "the runtime image must install the lean app set"
        )
        assert "requirements-ml.txt" not in active_lines(dockerfile), (
            "the runtime image must not install the modelling stack"
        )
        root = active_lines((ROOT / "requirements.txt").read_text(encoding="utf-8")).lower()
        for heavy in ("prophet", "tensorflow", "xgboost", "shap", "evidently"):
            assert heavy not in root, (
                f"{heavy} in the root requirements.txt is what Streamlit Community Cloud "
                "would have to install to serve a page"
            )

    def test_copies_the_aggregates_the_dashboard_reads(self, dockerfile):
        assert "COPY data/processed/" in dockerfile, (
            "the app reads only data/processed; without it the container starts empty"
        )

    def test_no_shell_syntax_leaks_into_instruction_arguments(self, dockerfile):
        # COPY is not a shell. The first draft carried
        #   COPY src/dashboard_data.py ./src/ 2>/dev/null || true
        # as a "skip if missing" idiom. That is a parse error, not a skip -- and the path it
        # named did not exist either, so the build failed twice over. Redirection, pipes and
        # && belong in RUN, which is the only instruction that goes through a shell.
        for number, line in enumerate(dockerfile.splitlines(), start=1):
            stripped = line.strip()
            if not stripped.upper().startswith("COPY "):
                continue
            for token in ("||", "&&", "|", ">", "2>", ";", "$("):
                assert token not in stripped, (
                    f"Dockerfile:{number}: shell syntax {token!r} in a COPY instruction:\n"
                    f"  {stripped}"
                )

    def test_every_copied_source_path_exists(self, dockerfile):
        # A COPY of a path that is not in the build context fails the build, and this repo
        # has no Docker daemon in CI to catch it. The dashboard's data-access module lives at
        # app/dashboard_data.py, which `COPY app/` already carries -- there is no
        # src/dashboard_data.py, and a copy line for it would have broken the build.
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if not stripped.upper().startswith("COPY "):
                continue
            parts = stripped.split()
            for source in parts[1:-1]:
                assert (ROOT / source).exists(), (
                    f"COPY source does not exist: {source!r}"
                )

    def test_the_dashboard_module_is_carried_by_the_app_copy(self, dockerfile):
        # Spelled out because the previous failure was exactly this file being copied from
        # the wrong directory, and `COPY app/` is the only line that should carry it.
        assert (ROOT / "app" / "dashboard_data.py").exists()
        assert not (ROOT / "src" / "dashboard_data.py").exists()
        assert "COPY app/" in dockerfile
        assert "src/dashboard_data.py" not in active_lines(dockerfile)

    def test_exposes_the_port_streamlit_binds(self, dockerfile):
        assert "EXPOSE 8501" in dockerfile
        assert 'STREAMLIT_SERVER_ADDRESS=0.0.0.0' in dockerfile, (
            "without 0.0.0.0 the container binds loopback and nothing can reach it"
        )

    def test_the_root_requirements_are_what_cloud_installs_and_they_are_lean(self):
        """The invariant that makes the public deploy work, asserted in one place.

        Streamlit Community Cloud reads the root ``requirements.txt`` and offers no way to
        point it elsewhere, so that file *is* the deploy. It must therefore contain what
        serving a page needs and nothing more, and the modelling stack must live in
        requirements-ml.txt. This is the check that fails loudly if someone helpfully moves
        Prophet back into the root file.
        """
        root = active_lines((ROOT / "requirements.txt").read_text(encoding="utf-8")).lower()
        ml = (ROOT / "requirements-ml.txt").read_text(encoding="utf-8").lower()

        # Anything needed to serve a page must be present.
        for package in ("streamlit", "pandas", "numpy"):
            assert package in root, f"{package} is required to serve the dashboard"

        # Nothing needed only to fit a model may be present.
        for heavy in ("prophet", "tensorflow", "xgboost", "shap", "evidently", "jupyter"):
            assert heavy not in root, (
                f"{heavy} does not belong in the file Streamlit Community Cloud installs"
            )
            assert heavy in ml, f"{heavy} still has to be declared, in requirements-ml.txt"

        # The ML file must build on the app file rather than duplicate it, so the two cannot
        # drift apart.
        assert "-r requirements.txt" in ml, (
            "requirements-ml.txt should extend the app set instead of restating it"
        )


# ------------------------------------------------------------------------------------ ci
class TestCiWorkflow:
    @pytest.fixture()
    def workflow(self) -> dict:
        return yaml.safe_load(CI.read_text(encoding="utf-8"))

    def test_is_valid_yaml_with_named_jobs(self, workflow):
        jobs = workflow["jobs"]
        assert {"verify", "models", "dashboard", "image"} <= set(jobs)

    def test_fast_checks_run_before_the_model_suite(self, workflow):
        jobs = workflow["jobs"]
        # The slow suite fits real models; running it first would waste ten minutes on a
        # commit that breaks a CSV header.
        assert "slow" not in jobs["models"].get("if", ""), "models must not be a pull_request gate"
        assert "-m \"not slow\"" in _steps_text(jobs["verify"])

    def test_regenerates_the_gitignored_raw_data(self, workflow):
        text = _steps_text(workflow["jobs"]["models"])
        assert "generate_retail_pulse.py" in text, (
            "the raw CSVs are gitignored, so CI has to regenerate them"
        )

    def test_dashboard_job_asserts_the_load_budget(self, workflow):
        assert "tests/test_dashboard.py" in _steps_text(workflow["jobs"]["dashboard"])

    def test_image_job_is_present_but_not_claimed_as_green(self, workflow):
        # No registry credentials exist in the submission window. The honest thing is a
        # disabled job with a stated reason, not a passing badge that never ran.
        assert workflow["jobs"]["image"].get("if") is False

    def test_requires_only_read_permission(self, workflow):
        assert workflow["permissions"] == {"contents": "read"}, (
            "regenerating artefacts must not push to the repository from CI"
        )

    def test_aggregates_are_uploaded_not_committed(self, workflow):
        text = _steps_text(workflow["jobs"]["models"])
        assert "upload-artifact" in text, (
            "regenerated aggregates have to go somewhere; uploading them is the only "
            "option that does not need a write-scoped token"
        )
        assert "git push" not in text


def _steps_text(job: dict) -> str:
    """Flatten a job's steps to text, including `uses:`.

    Reading only `run:` would miss every action-based step, which is exactly where
    upload-artifact lives. That mistake made this test fail against a workflow that was
    already correct.
    """
    parts = []
    for step in job.get("steps", []):
        parts.append(step.get("run", ""))
        parts.append(step.get("uses", ""))
        parts.append(step.get("if", ""))
    return "\n".join(parts)


# --------------------------------------------------------------------------- monitoring
class TestMonitoring:
    @pytest.fixture()
    def prometheus(self) -> dict:
        return yaml.safe_load((ROOT / "ops" / "prometheus.yml").read_text(encoding="utf-8"))

    @pytest.fixture()
    def grafana(self) -> dict:
        return json.loads((ROOT / "ops" / "grafana-dashboard.json").read_text(encoding="utf-8"))

    def test_prometheus_scrapes_the_dashboard(self, prometheus):
        jobs = prometheus["scrape_configs"]
        assert jobs, "no scrape jobs means no metrics"
        targets = [t for job in jobs for t in job.get("static_configs", [])
                   for t in t.get("targets", [])]
        assert any("8501" in t for t in targets), (
            "the Streamlit deployment is annotated for scraping; nothing reads that"
        )

    def test_grafana_dashboard_is_importable_json_with_panels(self, grafana):
        assert grafana["panels"], "an empty dashboard is not a dashboard"
        for panel in grafana["panels"]:
            assert panel["type"], "every panel needs a type or Grafana rejects the import"
            assert panel["title"], "an untitled panel is unidentifiable during an incident"

    def test_grafana_covers_availability_latency_and_saturation(self, grafana):
        # The three things you cannot diagnose without. A dashboard of only request rate is
        # a dashboard that tells you the site is slow and nothing about why.
        titles = " ".join(p["title"].lower() for p in grafana["panels"])
        assert "up" in titles or "availability" in titles
        assert "latency" in titles
        assert "memory" in titles or "saturation" in titles or "cpu" in titles

    def test_dashboard_probes_are_declared_for_prometheus(self):
        # The k8s manifest annotates the pod; these are the paths it points at.
        deployment = by_kind(load_documents(K8S / "dashboard.yaml"), "Deployment")[0]
        annotations = deployment["spec"]["template"]["metadata"]["annotations"]
        assert annotations["prometheus.io/scrape"] == "true"
        assert annotations["prometheus.io/port"] == "8501"
        assert "metrics" in annotations["prometheus.io/path"]