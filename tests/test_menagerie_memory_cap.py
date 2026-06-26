"""Tests for menagerie validator worker memory caps."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from menagerie.catalog import CatalogRow, load_rows
from menagerie.validate_menagerie import (
    WORKER_MEMORY_CAP_STATUS,
    WORKER_MEMORY_TEST_ALLOC_ENV,
    ValidationResult,
    _ledger_status,
    _resolve_scheduler_memory_budget_gb,
    append_validation_ledger,
    append_manifest,
    build_parser,
    manifest_records,
    validate_with_timeout,
)
from menagerie.ledger import connect


def _row(**overrides: object) -> CatalogRow:
    """Build a compact catalog row fixture for memory-cap tests.

    Parameters
    ----------
    overrides:
        Field overrides for the default catalog row.

    Returns
    -------
    CatalogRow
        Test catalog row.
    """

    data = {
        "model_id": 999001,
        "display_index": 999001,
        "stable_id": "memcap-test",
        "name": "MemoryCapToy",
        "variant": "",
        "family": "toy",
        "family_normalized": "toy",
        "domain": "unit",
        "zoo": "unit-zoo",
        "constructor_call": "torch.nn.Linear(1, 1)",
        "input_shape": "(1, 1)",
        "input_dtype": "float32",
        "era": "2026",
        "verified": False,
        "notes": "",
        "source": "catalog",
        "recipe_revision_sha256": "recipe",
    }
    data.update(overrides)
    return CatalogRow(**data)


def _catalog_row(model_id: int) -> CatalogRow:
    """Return one real catalog row by model ID.

    Parameters
    ----------
    model_id:
        Catalog model identifier.

    Returns
    -------
    CatalogRow
        Matching real catalog row.
    """

    rows_by_id = {row.model_id: row for row in load_rows()}
    return rows_by_id[model_id]


def test_worker_memory_cap_records_honest_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker that exceeds its RSS cap returns a memory-cap validation failure."""

    pytest.importorskip("psutil")
    monkeypatch.setenv(WORKER_MEMORY_TEST_ALLOC_ENV, "192")
    manifest_path = tmp_path / "validation_manifest.tsv"

    result = validate_with_timeout(
        _row(),
        dry_run=True,
        scope="forward",
        device="cpu",
        timeout_sec=20.0,
        worker_memory_cap_gb=0.12,
    )

    assert result.status == WORKER_MEMORY_CAP_STATUS
    assert result.peak_rss_mb is not None
    assert result.peak_rss_mb > 120
    assert "--worker-memory-cap-gb=0.120" in result.error
    append_manifest(manifest_path, result)
    [record] = manifest_records(manifest_path).values()
    assert record["status"] == WORKER_MEMORY_CAP_STATUS
    assert int(record["peak_rss_mb"]) == result.peak_rss_mb


def test_parent_accepts_normal_worker_result_with_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A capped worker result under the cap is parsed normally by the parent."""

    row = _row()
    expected = ValidationResult(
        row.name,
        row.model_id,
        "validated",
        3,
        True,
        "forward",
        0.25,
        "unit-zoo",
        "forward=True",
        "shape",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        peak_rss_mb=64,
    )
    worker_event = json.dumps({"event": "worker_result", "result": expected.__dict__})
    captured_command: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Capture the worker command and return a successful worker event.

        Parameters
        ----------
        command:
            Subprocess command built by the parent.
        kwargs:
            Additional subprocess options.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Completed process with one worker result event.
        """

        captured_command.extend(command)
        assert kwargs["timeout"] == 5.0
        return subprocess.CompletedProcess(command, 0, stdout=f"{worker_event}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = validate_with_timeout(
        row,
        dry_run=False,
        scope="forward",
        device="cpu",
        timeout_sec=5.0,
        tmp_dir=tmp_path,
        worker_memory_cap_gb=2.0,
    )

    assert result == expected
    assert "--worker-memory-cap-gb" in captured_command
    assert "2.000000" in captured_command


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected_status"),
    [
        (-signal.SIGKILL, "kernel: oom-kill: killed process", "failed:oom"),
        (-signal.SIGKILL, "terminated by administrator", "failed:killed"),
        (-signal.SIGSEGV, "segmentation fault", "failed:native_crash"),
        (-signal.SIGABRT, "abort trap", "failed:native_crash"),
        (-signal.SIGBUS, "bus error", "failed:native_crash"),
        (-signal.SIGTERM, "terminated by scheduler", "failed:killed"),
        (-signal.SIGHUP, "hangup", "failed:killed"),
        (-signal.SIGINT, "interrupted", "failed:killed"),
        (17, "Traceback: boom", "failed:exception"),
    ],
)
def test_worker_returncode_classification_is_signal_based(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stderr: str,
    expected_status: str,
) -> None:
    """Parent classifies worker failures by signal before generic exceptions."""

    row = _row()

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return a scripted worker process failure."""

        del kwargs
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = validate_with_timeout(
        row,
        dry_run=False,
        scope="forward",
        device="cpu",
        timeout_sec=5.0,
    )

    assert result.status == expected_status


def test_bare_oom_substring_does_not_route_to_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Words containing oom are not OOM evidence for SIGKILL classification."""

    row = _row()

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return a SIGKILL with non-OOM words in stderr."""

        del kwargs
        return subprocess.CompletedProcess(
            command,
            -signal.SIGKILL,
            stdout="",
            stderr="zoom bloom room doom",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = validate_with_timeout(
        row,
        dry_run=False,
        scope="forward",
        device="cpu",
        timeout_sec=5.0,
    )

    assert result.status == "failed:killed"


def test_signal_failure_statuses_survive_ledger_mapping() -> None:
    """Specific signal-derived failure statuses are not collapsed to failed."""

    assert _ledger_status("failed:oom") == "oom"
    assert _ledger_status("failed:native_crash") == "native_crash"
    assert _ledger_status("failed:killed") == "killed"


def test_forward_backward_validation_result_persists_scope_and_backward_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward-plus-backward validation writes a distinct ledger identity scope."""

    ledger_db = tmp_path / "verification.db"

    def connect_test_ledger() -> Any:
        """Connect append_validation_ledger to a temporary test ledger."""

        return connect(ledger_db)

    row = _row(stable_id="backward-scope", recipe_revision_sha256="recipe-bw")
    result = ValidationResult(
        row.name,
        row.model_id,
        "validated",
        3,
        True,
        "forward+backward",
        0.25,
        "unit-zoo",
        "forward=True; backward=True",
        "shape",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        input_scale=1.0,
    )
    monkeypatch.setattr("menagerie.validate_menagerie.connect_ledger", connect_test_ledger)

    append_validation_ledger(row, result, "cpu", result.input_scale)
    conn = connect(ledger_db)
    stored = conn.execute(
        "SELECT scope, forward_pass, backward_pass FROM verification_runs"
    ).fetchone()

    assert dict(stored) == {"scope": "forward+backward", "forward_pass": 1, "backward_pass": 1}


def test_worker_under_memory_cap_validates() -> None:
    """A real worker below its RSS cap returns the normal validated status."""

    result = validate_with_timeout(
        _catalog_row(65),
        dry_run=False,
        scope="forward",
        device="cpu",
        timeout_sec=20.0,
        worker_memory_cap_gb=4.0,
    )

    assert result.status == "validated", result.error
    assert result.peak_rss_mb is not None
    assert result.peak_rss_mb < 4096


def test_worker_memory_cap_sets_scheduler_budget_from_effective_jobs() -> None:
    """A per-worker cap bounds the scheduler's effective in-flight budget."""

    assert _resolve_scheduler_memory_budget_gb(None, 24.0, 3) == 72.0
    assert _resolve_scheduler_memory_budget_gb(64.0, 24.0, 3) == 64.0


def test_worker_memory_cap_default_is_off() -> None:
    """The new worker cap is disabled unless explicitly requested."""

    args = build_parser().parse_args([])
    assert args.worker_memory_cap_gb is None
