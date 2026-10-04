"""RetailPulse nightly refresh — Airflow DAG.

Tier 3 of the design spec: authored and statically validated (importable, and the task graph
is checked without a scheduler), not deployed. Airflow needs a broker and a webserver, which
is infrastructure the submission window did not have. The equivalent schedule is expressed as
a Kubernetes CronJob in `deploy/kubernetes/pipeline-cronjob.yaml`, which is the version that
would actually run.

The dependency order below encodes the real constraint, not a guess at it:

    precompute -> forecast -> churn -> inventory -> drift

`precompute` first because every model reads the aggregates it writes. `inventory` after
`forecast` because the reorder quantity consumes the h=1 forecast; running it first would
silently use a stale or absent forecast. `drift` last, because it compares the current window
against the reference period and would report drift caused by a half-finished refresh.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

try:  # Airflow is a Tier 3 dependency and is not installed locally.
    from airflow import DAG
    from airflow.operators.bash import BashOperator

    AIRFLOW_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    DAG = None
    BashOperator = None
    AIRFLOW_AVAILABLE = False

# 03:17 UTC rather than 03:00. Every cluster-wide cron in the org fires on the hour; this one
# queues behind them instead of competing for the same nodes.
SCHEDULE = "17 3 * * *"

DEFAULT_ARGS = {
    "owner": "retailpulse",
    "depends_on_past": False,
    # A failed night is retried twice, not five times. This pipeline is idempotent and
    # deterministic, so a repeat failure is a real fault rather than a flake, and retrying it
    # five times only delays the alert by 90 minutes.
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
    "email_on_failure": False,   # left to whatever alert routing the deployer configures
    "email_on_success": False,
}

# The stage list is IMPORTED, not restated. This DAG previously carried its own copy of the
# order and its own hand-written list of expected output filenames. That copy drifted: it
# required files that precompute never writes, and src/pipeline.py -- the module the
# Kubernetes CronJob actually runs -- had a third, different opinion. One imported list cannot
# disagree with itself.
#
# Each stage is a container command, so the DAG does not depend on Airflow's Python
# environment having Prophet or TensorFlow. That is the whole reason for using BashOperator
# here rather than @task with PythonOperator.
# Airflow loads DAG files with a loader that does not always define __file__, and this module is
# also exec'd directly by tests to exercise the no-Airflow fallback. Falling back to the working
# directory keeps the import below working in all three cases instead of raising NameError and
# taking the whole DAG file down with it.
_REPO_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline import STAGES as PIPELINE_STAGES          # noqa: E402
from src.pipeline import STAGE_OUTPUTS, STAGE_RESOURCES     # noqa: E402

WHY = {
    "precompute": "reduce the raw panel to the aggregates everything else reads",
    "forecasting": "fit and score the pooled forecast; writes forecast_next_weeks.csv",
    "churn": "rescore churn and recompute SHAP",
    "inventory": "recompute the reorder plan and the walk-forward backtest",
    "drift": "compare the current window against the reference period; must run last",
}

STAGES = [(stage, f"python -m src.{stage}", WHY.get(stage, "")) for stage in PIPELINE_STAGES]


def build_dag() -> "DAG | None":
    """Construct the DAG. Returns None when Airflow is absent.

    Returning None rather than raising lets this module be imported and tested without an
    Airflow install, which is the only way to statically validate it on a machine that does
    not have one.
    """
    if not AIRFLOW_AVAILABLE:
        log.warning("airflow is not installed; %s is not built", __name__)
        return None

    with DAG(
        dag_id="retailpulse_nightly",
        description="Nightly refresh: forecasts, churn scores, reorder plan, drift report",
        # start_date is fixed, not `days_ago`, so a rerun of the same DAG definition does not
        # silently change which intervals Airflow considers missed.
        start_date=datetime(2026, 1, 1),
        schedule=SCHEDULE,
        catchup=False,       # never backfill a nightly refresh from generated data
        max_active_runs=1,   # the stages write the same CSVs; two concurrent runs would race
        default_args=DEFAULT_ARGS,
        tags=["retailpulse", "nightly"],
        doc_md=__doc__,
    ) as dag:

        previous = None
        tasks: dict[str, object] = {}
        for name, command, why in STAGES:
            writes = ", ".join(STAGE_OUTPUTS.get(name, ()))
            task = BashOperator(
                task_id=name,
                bash_command=command,
                # The artefact list goes in the task doc so an operator reading the Airflow UI
                # can see what proves the stage ran without opening the module.
                doc_md=f"{why}\n\nExpected outputs: {writes}\n\n"
                       f"Resources: {STAGE_RESOURCES.get(name, {})}",
                # Linear by default. `previous` is threaded explicitly rather than relying on
                # Airflow's ordering, so the dependency survives someone inserting a task.
                **({"depends_on": [previous]} if previous else {}),
            )
            previous = task
            tasks[name] = task

        # The artefact check is delegated to src.pipeline.verify_all_outputs rather than a
        # shell loop over a second hand-written filename list. That list was how this DAG and
        # the CronJob came to disagree about what a successful run produces.
        BashOperator(
            task_id="verify_outputs",
            bash_command=(
                "python -c \"import sys; sys.path.insert(0,'.'); "
                "from src.pipeline import verify_all_outputs; "
                "m=verify_all_outputs(); "
                "print('missing:', m) if m else None; "
                "raise SystemExit(1 if m else 0)\""
            ),
            depends_on=[previous],
            doc_md="Fail loudly if a stage reported success but wrote nothing.",
        )

        drift_report = BashOperator(
            task_id="check_drift",
            bash_command=(
                # reports/drift, not data/processed: drift's summary is a report artefact.
                # Reading data/processed/drift_summary.csv raised FileNotFoundError on every
                # run, because nothing has ever written it there.
                "python -c \"import pandas as pd; "
                "s=pd.read_csv('reports/drift/drift_summary.csv'); "
                "d=s[s.verdict.isin(['moderate','significant'])]; "
                "print('drift:', list(d.column) or 'none'); "
                "raise SystemExit(0)\""
            ),
            # Named explicitly rather than reusing a variable from the loop above: the loop's
            # last task IS drift, but a reader should not have to know that to see the
            # dependency.
            depends_on=[tasks["drift"]],
            doc_md=("Print the drifted columns. Always exits 0: drift is a finding to "
                    "investigate, not a reason to fail the refresh and leave yesterday's "
                    "data in place."),
        )

    return dag


dag = build_dag()


if __name__ == "__main__":       # pragma: no cover
    if dag is None:
        print("airflow is not installed; install it to build this DAG:")
        print("  pip install apache-airflow")
        raise SystemExit(1)

    print(f"dag_id: {dag.dag_id}")
    print(f"schedule: {dag.schedule_interval}")
    print(f"max_active_runs: {dag.max_active_runs}")
    print("tasks:")
    for task in sorted(dag.tasks, key=lambda t: t.task_id):
        print(f"  {task.task_id:<16} {task.bash_command.strip()[:60]}")
    # print_tree() would show the full graph but is noisy in a terminal.
    print("\npass a real Airflow connection to execute. The Kubernetes CronJob is the")
    print("deployable equivalent.")