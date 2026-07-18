"""Round-19 VS6 real-subprocess acceptance dry-run composition."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.cli import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_PAUSED,
    _persisted_environment_generations,
)
from menagerie.crawler.driver import DriverConfig
from menagerie.crawler.reducer import materialize_current
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.status import funnel_counts
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.dry_run_support import (
    DRY_RUN_CASES,
    TinyModelAuthor,
    create_dry_run_snapshot,
    dry_run_paths,
)


def _dry_run_command(
    repo_root: Path,
    campaign_root: Path,
    environment_prefix: Path,
    command: str,
    *,
    inject_source_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one documented dry-run command under the selected environment Python.

    Parameters
    ----------
    repo_root, campaign_root, environment_prefix:
        Checked-out source, disposable campaign, and strictly bound real environment roots.
    command:
        Either the initial ``run`` or subsequent ``resume`` command.
    inject_source_failure:
        Whether the deterministic author lane should fail every source resolution.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured real CLI process result.
    """

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    environment["MENAGERIE_RELEASE_GATE"] = "1"
    if inject_source_failure:
        environment["MENAGERIE_DRY_RUN_INJECT_ALL_SOURCE_FAILURE"] = "1"
    argv = [
        str(environment_prefix / "bin" / "python"),
        "-B",
        "-m",
        "menagerie.crawler",
        "--repo-root",
        str(repo_root),
        command,
        "--dry-run",
        "--dry-run-root",
        str(campaign_root),
        "--dry-run-environment-prefix",
        str(environment_prefix),
        "--review-checkpoint-at",
        "0" if inject_source_failure else "2",
        "--progress-milestones",
        "3" if not inject_source_failure else "",
    ]
    if command == "resume":
        argv.append("--after-review")
    return subprocess.run(
        argv,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def _json_output(completed: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
    """Decode the final structured stdout line from one CLI subprocess.

    Parameters
    ----------
    completed:
        Captured CLI subprocess result.

    Returns
    -------
    Mapping[str, Any]
        Final JSON object printed by the crawler CLI.
    """

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stderr
    value = json.loads(lines[-1])
    assert isinstance(value, dict)
    return value


def _current_projection(campaign_root: Path) -> tuple[Mapping[str, Mapping[str, Any]], Path]:
    """Materialize the authenticated current projection for one dry-run campaign.

    Parameters
    ----------
    campaign_root:
        Disposable campaign root shared by the CLI subprocesses.

    Returns
    -------
    tuple[Mapping[str, Mapping[str, Any]], pathlib.Path]
        Current model mapping and authenticated attempt-ledger path.
    """

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    config = DriverConfig(
        target="osx-arm64",
        run_id="crawler-run",
        author_model="fake-claude",
        author_version="dry-run",
        checker_model="fake-codex",
        checker_version="dry-run",
    )
    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=config.author_model,
        author_version=config.author_version,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
    )
    attempts = scan_jsonl(paths.ledgers.attempts)
    context = replace(
        context,
        environment_generations=_persisted_environment_generations(attempts),
    )
    return materialize_current(paths.ledgers, context=context), paths.ledgers.attempts


def test_tiny_model_author_requires_authority_context() -> None:
    """The dry-run author must implement the mandatory four-argument authority signature."""

    parameters = tuple(inspect.signature(TinyModelAuthor.author).parameters)
    assert parameters == ("self", "item", "work_root", "config", "context")


def test_documented_dry_run_and_resume_use_real_environment(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Documented run/resume commands must pause then award every tiny model for real."""

    repo_root = Path(__file__).resolve().parents[3]
    campaign_root = tmp_path / "dry-run-campaign"
    run = _dry_run_command(
        repo_root,
        campaign_root,
        real_environment_fixture.prefix,
        "run",
    )
    assert run.returncode == EXIT_PAUSED, (run.stdout, run.stderr)
    run_output = _json_output(run)
    assert run_output["status"] == "paused:review-checkpoint"
    assert run_output["acceptance"]["status"] == "pending"
    assert "failed:source" not in f"{run.stdout}\n{run.stderr}"
    assert "identity-unresolved" not in f"{run.stdout}\n{run.stderr}"
    paused_current, _attempt_path = _current_projection(campaign_root)
    assert run_output["funnel"] == funnel_counts(paused_current)
    assert funnel_counts(paused_current) == {
        "framework:pytorch": 2,
        "metadata:accepted": 2,
        "mode:eval": 2,
        "mode:train": 2,
        "models:total": 2,
        "rung:R1_LIBRARY": 2,
        "status:runs": 2,
    }

    resume = _dry_run_command(
        repo_root,
        campaign_root,
        real_environment_fixture.prefix,
        "resume",
    )
    assert resume.returncode == EXIT_OK, (resume.stdout, resume.stderr)
    resume_output = _json_output(resume)
    assert resume_output["status"] == "complete"
    assert resume_output["acceptance"]["status"] == "passed"
    assert "failed:source" not in f"{resume.stdout}\n{resume.stderr}"
    assert "identity-unresolved" not in f"{resume.stdout}\n{resume.stderr}"

    current, attempt_path = _current_projection(campaign_root)
    snapshot = create_dry_run_snapshot(campaign_root)
    expected_ids = {item.stable_id for item in snapshot.items}
    assert set(current) == expected_ids
    assert resume_output["funnel"] == funnel_counts(current)
    assert funnel_counts(current) == {
        "framework:pytorch": 10,
        "metadata:accepted": 10,
        "mode:eval": 10,
        "mode:train": 10,
        "models:total": 10,
        "rung:R1_LIBRARY": 9,
        "rung:R3_PORT": 1,
        "status:runs": 10,
    }
    attempts = scan_jsonl(attempt_path)
    succeeded_by_id = {
        stable_id: [
            attempt
            for attempt in attempts
            if attempt["stable_id"] == stable_id and attempt["result"] == "succeeded"
        ]
        for stable_id in expected_ids
    }
    selected_python = str(real_environment_fixture.prefix / "bin" / "python")
    for stable_id, model in current.items():
        assert model["status"]["code"] == "runs"
        assert str(model["record_revision"]).startswith("sha256:")
        authenticated = succeeded_by_id[stable_id]
        assert authenticated
        assert set(model["execution"]["accepted_attempt_ids"]) == {
            attempt["attempt_id"] for attempt in authenticated
        }
        for attempt in authenticated:
            assert attempt["raw_award_receipt"] is not None
            assert attempt["raw_award_receipt_sha256"] is not None
            assert (
                attempt["parent_attestation"]["named_raw_award_receipt_sha256"]
                == attempt["raw_award_receipt_sha256"]
            )
            argv = list(attempt["invocation"]["argv"])
            interpreter_index = argv.index(selected_python)
            assert argv[interpreter_index : interpreter_index + 4] == [
                selected_python,
                "-B",
                "-m",
                "menagerie.crawler.worker",
            ]
    assert all("import menagerie_round19_sentinel" in case.adapter_source for case in DRY_RUN_CASES)
    outside = subprocess.run(
        [sys.executable, "-B", "-c", "import menagerie_round19_sentinel"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert outside.returncode != 0


def test_dry_run_all_source_failure_is_acceptance_error(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """An all-source-failure terminal partition must not false-complete the CLI gate."""

    repo_root = Path(__file__).resolve().parents[3]
    failed = _dry_run_command(
        repo_root,
        tmp_path / "all-source-failure",
        real_environment_fixture.prefix,
        "run",
        inject_source_failure=True,
    )
    assert failed.returncode == EXIT_ERROR, (failed.stdout, failed.stderr)
    output = _json_output(failed)
    assert output["status"] == "terminal-partition-complete"
    assert output["acceptance"] == {
        "status": "acceptance-failed",
        "reason_code": "all-source-failure",
        "expected_runnable_models": 10,
        "runs_revisions": 0,
        "authenticated_attempt_models": 0,
        "unexpected_statuses": {"failed:source": 10},
    }
    current, attempt_path = _current_projection(tmp_path / "all-source-failure")
    assert output["funnel"] == funnel_counts(current)
    assert output["funnel"]["status:failed:source"] == 10
    attempts = scan_jsonl(attempt_path)
    assert len(attempts) == 10
    assert {attempt["stage"] for attempt in attempts} == {"source"}
    assert all(attempt["raw_award_receipt"] is None for attempt in attempts)
