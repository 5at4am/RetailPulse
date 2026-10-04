"""The nightly refresh, as an ordered list of stages.

Why this module exists
----------------------
The first draft of `deploy/kubernetes/pipeline-cronjob.yaml` put all six steps in one Pod as
six containers. That reads as a pipeline in YAML and is not one: Kubernetes starts every
container in a Pod concurrently, so `verify` ran at t=0 against outputs that did not exist
yet, and a container exiting non-zero did not stop the steps beside it. The stages were
simultaneous, not sequential.

Sequencing belongs in code, not in the shape of a manifest, because code can be tested. The
YAML is now a thin wrapper that calls `python -m src.pipeline`; `tests/test_pipeline.py`
asserts the order, the stop-on-failure behaviour and the output contract.

Contract
--------
Each stage runs as a subprocess (`python -m src.<stage>`) so a crash in a model cannot take
the orchestrator down with it, and each stage is the same `__main__` entry point a developer
runs by hand. The first failing stage ends the run: later stages depend on earlier ones, and
continuing would overwrite good artefacts with garbage derived from a partial refresh.

`verify` is a stage rather than a Kubernetes probe because the check is "every expected
artefact exists AND is non-empty", which is a statement about the whole run, not about one
container's liveness.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config

# Where each stage writes. Drift is the exception: its summary and its Evidently HTML are
# report artefacts, not dashboard inputs, so they live under reports/drift rather than in
# data/processed. Hard-coding one output directory for every stage is what produced a contract
# naming `drift_report.csv` for a module that has always written `drift_summary.csv`.
def _stage_dir(stage: str, root: Path | None = None) -> Path:
    """The directory a stage's artefacts land in.

    `root` overrides everything, so tests can point a check at a temp directory without
    touching either real output tree.
    """
    if root is not None:
        return Path(root)
    return config.REPORTS / "drift" if stage == "drift" else config.PROCESSED


# Artefacts each stage must leave behind. A stage that "succeeds" without writing these is a
# failure, and the run has to say so rather than passing a half-finished night to the
# dashboard. This is the single list; `verify_all_outputs` is derived from it rather than
# maintained beside it, because two hand-written lists of the same filenames drift apart.
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "precompute": ("kpis.csv",),
    "forecasting": ("forecast_next_weeks.csv", "forecast_backtest.csv"),
    "churn": ("churn_metrics.csv", "churn_scores.csv", "churn_shap_local.csv"),
    "inventory": ("inventory_reorder_plan.csv", "inventory_backtest_summary.csv"),
    "drift": ("drift_summary.csv", "category_mix_drift.csv"),
}

# Order is the whole point of this module.
STAGES: tuple[str, ...] = ("precompute", "forecasting", "churn", "inventory", "drift")

# Per-stage resource envelopes for the Airflow path, where each stage is its own pod and so
# can be sized honestly. The CronJob path runs everything in one container and uses the
# largest envelope here (forecasting) for the lot.
STAGE_RESOURCES: dict[str, dict[str, str]] = {
    "precompute": {"cpu": "500m", "memory": "2Gi"},
    "forecasting": {"cpu": "1000m", "memory": "4Gi"},
    "churn": {"cpu": "1000m", "memory": "2Gi"},
    "inventory": {"cpu": "500m", "memory": "2Gi"},
    "drift": {"cpu": "300m", "memory": "1Gi"},
}


@dataclass
class StageResult:
    """What one stage did, kept for the run summary and for the test assertions."""

    name: str
    ok: bool
    seconds: float
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    missing: list[str] = field(default_factory=list)


class StageFailed(RuntimeError):
    """A stage exited non-zero. Carries the stage name so the log says which one."""

    def __init__(self, stage: str, result: StageResult):
        self.stage = stage
        self.result = result
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        super().__init__(
            f"stage {stage!r} failed with exit code {result.returncode}"
            + (f":\n  " + "\n  ".join(tail) if tail else "")
        )


def verify_stage_outputs(stage: str, root: Path | None = None) -> list[str]:
    """Return the artefacts `stage` promised but did not leave. Empty list means it kept its word.

    Checks that a file exists AND has rows. A zero-byte CSV is the shape a crashed write
    leaves behind, and `Path.exists()` alone would wave it through.
    """
    directory = _stage_dir(stage, root)
    missing: list[str] = []
    for name in STAGE_OUTPUTS.get(stage, ()):
        path = directory / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            if path.stat().st_size == 0 or len(pd.read_csv(path)) == 0:
                missing.append(name)
        except Exception:                                   # noqa: BLE001
            # Unreadable is indistinguishable from absent for our purpose: either way the
            # dashboard cannot use it.
            missing.append(name)
    return missing


def verify_all_outputs() -> dict[str, list[str]]:
    """Missing artefacts keyed by stage, across every stage. Empty dict means the run is good."""
    return {stage: verify_stage_outputs(stage) for stage in STAGES
            if verify_stage_outputs(stage)}


def run_stage(stage: str, python: str | None = None, root: Path | None = None,
              echo=print) -> StageResult:
    """Run one stage as a subprocess and check it left what it promised.

    Raises `StageFailed` on a non-zero exit, so a caller that forgets to inspect `ok` cannot
    accidentally continue past a broken stage.
    """
    started = time.monotonic()
    completed = subprocess.run(
        [python or sys.executable, "-m", f"src.{stage}"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    seconds = time.monotonic() - started
    echo(f"[{seconds:7.1f}s] {stage}")

    result = StageResult(name=stage, ok=completed.returncode == 0, seconds=seconds,
                         returncode=completed.returncode, stdout=completed.stdout,
                         stderr=completed.stderr)
    if not result.ok:
        raise StageFailed(stage, result)

    result.missing = verify_stage_outputs(stage, root)
    if result.missing:
        result.ok = False
        raise StageFailed(stage, result)
    return result


def run_pipeline(stages: tuple[str, ...] = STAGES, python: str | None = None,
                 root: Path | None = None, echo=print) -> list[StageResult]:
    """Run the stages in order, stopping at the first failure.

    Returns only on success. The raised `StageFailed` names the stage, which is the single
    fact an operator needs at 03:17.
    """
    results: list[StageResult] = []
    for stage in stages:
        results.append(run_stage(stage, python=python, root=root, echo=echo))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RetailPulse nightly refresh.")
    parser.add_argument("--only", nargs="+", choices=STAGES,
                        help="run just these stages, in the order given")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the artefact contract check (debugging only)")
    args = parser.parse_args(argv)

    stages = tuple(args.only) if args.only else STAGES
    started = time.monotonic()
    try:
        results = run_pipeline(stages)
    except StageFailed as failure:
        print(f"PIPELINE FAILED at {failure}", file=sys.stderr)
        return 1

    for result in results:
        print(f"  ok  {result.name:<12} {result.seconds:7.1f}s")

    if not args.skip_verify:
        missing = verify_all_outputs()
        if missing:
            for stage, names in missing.items():
                print(f"VERIFY FAILED, {stage} missing or empty: {', '.join(names)}",
                      file=sys.stderr)
            return 1
        total = sum(len(names) for names in STAGE_OUTPUTS.values())
        print(f"verify ok: {total} artefacts present")

    print(f"pipeline complete in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())